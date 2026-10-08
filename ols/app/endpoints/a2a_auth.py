"""Authentication and token-exchange helpers for the A2A endpoint.

The stock REST API (``ols/app/endpoints/ols.py`` and friends) authenticates
callers with Kubernetes ``TokenReview``/``SubjectAccessReview`` via
``ols.src.auth.k8s``. That mechanism is unchanged by this module and remains
the only way to reach ``/v1/*``.

The A2A endpoint (``ols/app/endpoints/a2a.py``) is reached from the ACME
A2A agent through the Praxis gateway and is not a Kubernetes client, so it
cannot use ``k8s`` auth. Instead it:

1. Independently validates the inbound Keycloak access token ("Token A")
   against the issuing realm's JWKS -- signature, issuer, the
   ``lightspeed-a2a`` audience, expiry, and ``azp=acme-agent`` -- even though
   the Praxis gateway already checked it. Defense in depth: this service must
   not trust an unauthenticated header alone.
2. For every query that will call the OpenShift MCP server, fetches this
   pod's own SPIFFE JWT-SVID and performs an RFC 8693 token exchange with the
   confidential ``lightspeed-mcp`` Keycloak client, trading Token A for a
   fresh "Token B" scoped to the ``openshift-mcp`` audience. Token B -- never
   Token A -- is what ``ols/app/endpoints/a2a.py`` feeds into OLS's existing
   MCP token-resolution path (see ``ols/utils/mcp_utils.py``).

This mirrors the pattern already proven by ``agents/rca_agent/rca_agent/identity.py``
in this same GitOps pattern repository, adapted to this service's async,
FastAPI-native style (the RCA agent used a synchronous ``httpx.Client`` inside
an ASGI middleware; this module uses ``httpx.AsyncClient`` directly so it does
not block the event loop).

Security notes:
    * Raw bearer tokens are never logged. Only non-sensitive claims (``sub``,
      ``azp``) are logged, and only at debug level.
    * Token B is minted fresh per request and is never cached or persisted;
      it is held only for the duration of a single ``SendMessage`` call.
    * No ``client_id`` form parameter is sent during token exchange -- Keycloak
      resolves the federated SPIFFE client from the assertion's ``sub`` claim,
      and including a mismatched ``client_id`` causes Keycloak to reject the
      request outright.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import jwt
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Frozen interface decisions (see LIGHTSPEED_MIGRATION_PLAN.md / IMPLEMENTATION
# plan). These are not meant to vary between deployments; they are kept as
# module constants, used as defaults, and may still be overridden by
# environment variable for testing.
CLUSTER_HEADER_NAME = "X-OLS-Cluster"
# Optional inbound correlation id. If the caller (ACME, via Praxis) already
# stamped one, it is preserved so every hop of a request shares an id;
# otherwise one is generated per request. Never required configuration.
REQUEST_ID_HEADER_NAME = "X-Request-Id"
DEFAULT_INBOUND_AUDIENCE = "lightspeed-a2a"
DEFAULT_INBOUND_AZP = "acme-agent"
DEFAULT_EXCHANGE_AUDIENCE = "openshift-mcp"
DEFAULT_EXCHANGE_CLIENT_ASSERTION_TYPE = (
    "urn:ietf:params:oauth:client-assertion-type:jwt-spiffe"
)

# The Keycloak client this service authenticates *as* when it performs the
# RFC 8693 exchange and drives the MCP call -- i.e. the workload that actually
# does the action, recorded as the audit ``actor``. The exchange resolves this
# client from the SPIFFE assertion's ``sub``, so it is not configuration; it is
# a fixed property of this deployment's Keycloak realm (see the ``rca`` realm in
# charts/all/keycloak-oidc).
EXCHANGE_CLIENT_AZP = "lightspeed-mcp"

# Prefix for the single structured audit line emitted per A2A request (see
# ``audit_record``). Stable and greppable: ``grep a2a_audit`` surfaces every
# "OLS did X on behalf of <caller>" record.
AUDIT_LOG_PREFIX = "a2a_audit"


class A2AAuthenticationError(Exception):
    """Raised when an inbound A2A caller cannot be authenticated."""


class A2AWorkloadIdentityError(Exception):
    """Raised when ZTWIM/SPIRE cannot issue this pod a JWT-SVID."""


class A2AConfigurationError(Exception):
    """Raised when required A2A configuration is missing or invalid."""


class A2AExchangeError(Exception):
    """Raised when the RFC 8693 exchange for an MCP-scoped token fails."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _parse_azp_set(raw: str) -> frozenset[str]:
    """Parse a comma-separated allow-list of permitted inbound ``azp`` values.

    The chart renders ``A2A_INBOUND_AZP`` as a comma-separated string (e.g.
    ``acme-agent`` or ``acme-agent,orchestrator``, no spaces). Entries are
    split on comma, stripped, and empties dropped. An empty/unset value falls
    back to the single default caller, preserving the original single-value
    behavior while allowing a second caller to be added by config alone.
    """
    values = frozenset(item.strip() for item in raw.split(",") if item.strip())
    return values or frozenset({DEFAULT_INBOUND_AZP})


@dataclass(frozen=True)
class A2ASettings:
    """Environment-derived configuration for the A2A endpoint.

    All values are deployment-specific (issuer, cluster id, public RPC
    origin) except where noted; the frozen audiences/azp default to the
    values pinned in the migration plan and normally do not need overriding.
    """

    keycloak_issuer_url: str
    keycloak_ca_bundle: str
    inbound_audience: str
    inbound_azp: frozenset[str]
    exchange_audience: str
    exchange_client_assertion_type: str
    spiffe_jwt_audience: str
    spiffe_endpoint_socket: str
    spiffe_timeout_seconds: float
    cluster_id: str
    rpc_url: str
    agent_name: str
    agent_description: str
    keycloak_timeout_seconds: float
    jwks_cache_seconds: int
    exchange_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "A2ASettings":
        """Build settings from the process environment.

        Raises:
            A2AConfigurationError: if a deployment-specific, required value
                is missing.
        """
        issuer_url = _env("A2A_KEYCLOAK_ISSUER_URL")
        if not issuer_url:
            raise A2AConfigurationError("A2A_KEYCLOAK_ISSUER_URL must be configured")
        cluster_id = _env("A2A_CLUSTER_ID")
        if not cluster_id:
            raise A2AConfigurationError("A2A_CLUSTER_ID must be configured")
        rpc_url = _env("A2A_RPC_URL")
        if not rpc_url:
            raise A2AConfigurationError("A2A_RPC_URL must be configured")

        return cls(
            keycloak_issuer_url=issuer_url.rstrip("/"),
            keycloak_ca_bundle=_env("A2A_KEYCLOAK_CA_BUNDLE"),
            inbound_audience=_env("A2A_INBOUND_AUDIENCE", DEFAULT_INBOUND_AUDIENCE),
            inbound_azp=_parse_azp_set(_env("A2A_INBOUND_AZP")),
            exchange_audience=_env("A2A_EXCHANGE_AUDIENCE", DEFAULT_EXCHANGE_AUDIENCE),
            exchange_client_assertion_type=_env(
                "A2A_EXCHANGE_CLIENT_ASSERTION_TYPE",
                DEFAULT_EXCHANGE_CLIENT_ASSERTION_TYPE,
            ),
            # Per the migration plan, the JWT-SVID assertion audience is the
            # Keycloak realm issuer URL unless explicitly overridden.
            spiffe_jwt_audience=_env("A2A_SPIFFE_JWT_AUDIENCE", issuer_url.rstrip("/")),
            spiffe_endpoint_socket=_env("A2A_SPIFFE_ENDPOINT_SOCKET"),
            spiffe_timeout_seconds=float(_env("A2A_SPIFFE_TIMEOUT_SECONDS", "5")),
            cluster_id=cluster_id,
            rpc_url=rpc_url,
            agent_name=_env("A2A_AGENT_NAME", "openshift-lightspeed"),
            agent_description=_env(
                "A2A_AGENT_DESCRIPTION",
                "OpenShift Lightspeed: investigates and answers questions about "
                "this OpenShift cluster using the cluster's own MCP tools.",
            ),
            keycloak_timeout_seconds=float(_env("A2A_KEYCLOAK_TIMEOUT_SECONDS", "5")),
            jwks_cache_seconds=int(_env("A2A_JWKS_CACHE_SECONDS", "300")),
            exchange_timeout_seconds=float(
                _env("A2A_TOKEN_EXCHANGE_TIMEOUT_SECONDS", "10")
            ),
        )

    @property
    def tls_verify(self) -> bool | ssl.SSLContext:
        """httpx-compatible verify argument for Keycloak HTTPS calls.

        When ``A2A_KEYCLOAK_CA_BUNDLE`` is set, return an SSLContext that
        trusts the system default CAs *plus* that bundle -- not the bundle
        alone. httpx's ``verify=<path>`` replaces the default trust store
        entirely, which breaks Routes whose leaf chains to a public CA
        (ZeroSSL/Let's Encrypt) when the synced managed-ingress bundle only
        has intermediates. Same pattern as
        ``agents/acme_agent/src/acme_agent/auth.py`` ``_tls_context_trusting``.
        """
        if not self.keycloak_ca_bundle:
            return True
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=self.keycloak_ca_bundle)
        return context


@dataclass(frozen=True)
class CallerIdentity:
    """A validated inbound caller, bound to the cluster it targeted.

    ``token`` (Token A, the original caller bearer) is kept only long enough
    to serve as the ``subject_token`` of the RFC 8693 exchange in
    ``A2AAuthenticator.exchange_for_mcp``. It is never logged and never sent
    to MCP -- only the *exchanged* token (Token B) is.
    """

    sub: str
    claims: dict[str, Any]
    cluster_id: str
    token: str
    request_id: str

    @property
    def task_scope(self) -> str:
        """Scope key isolating conversation/task state per (caller, cluster)."""
        return f"{self.cluster_id}:{self.sub}"


def _audit_value(value: Any) -> str:
    """Render one audit field value, quoting it only if it contains spaces."""
    text = str(value)
    if not text:
        return '""'
    return f'"{text}"' if any(char.isspace() for char in text) else text


def audit_record(
    caller: CallerIdentity,
    *,
    task_id: str,
    context_id: str,
    action: str,
    outcome: str,
) -> str:
    """Build the single greppable audit line for one A2A request.

    Records that the OLS workload (``actor`` -- the ``lightspeed-mcp`` client
    that performs the RFC 8693 exchange and drives MCP, i.e. who *did* it)
    acted ``on_behalf_of`` the validated inbound caller (its ``azp`` and
    ``subject``, i.e. who it was done *for*), against ``cluster``, for one a2a
    ``task``. ``request_id`` correlates this with the caller's own outbound log
    line and every other hop of the same request. No token is ever included --
    only non-sensitive identity claims.
    """
    fields = {
        "request_id": caller.request_id,
        "actor": EXCHANGE_CLIENT_AZP,
        "on_behalf_of": caller.claims.get("azp", ""),
        "subject": caller.sub,
        "cluster": caller.cluster_id,
        "task": task_id,
        "context": context_id,
        "action": action,
        "outcome": outcome,
    }
    rendered = " ".join(f"{key}={_audit_value(value)}" for key, value in fields.items())
    return f"{AUDIT_LOG_PREFIX} {rendered}"


class _JWKSCache:
    """Caches Keycloak OIDC discovery and JWKS documents."""

    def __init__(self, settings: A2ASettings) -> None:
        self._settings = settings
        self._discovery: Optional[dict[str, Any]] = None
        self._jwks: Optional[dict[str, Any]] = None
        self._jwks_expiry = 0.0
        self._lock = threading.Lock()

    async def discovery(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Return (and cache) the OIDC discovery document."""
        if self._discovery is not None:
            return self._discovery
        url = f"{self._settings.keycloak_issuer_url}/.well-known/openid-configuration"
        try:
            response = await client.get(url)
            response.raise_for_status()
            discovery = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise A2AAuthenticationError(
                "Keycloak OIDC discovery is unavailable"
            ) from error
        if discovery.get("issuer", "").rstrip("/") != self._settings.keycloak_issuer_url:
            raise A2AAuthenticationError("Keycloak issuer does not match configuration")
        if not discovery.get("jwks_uri"):
            raise A2AAuthenticationError("Keycloak discovery does not contain jwks_uri")
        self._discovery = discovery
        return discovery

    async def jwks(self, client: httpx.AsyncClient, force_refresh: bool = False) -> dict[str, Any]:
        """Return (and cache) the JWKS document."""
        if (
            self._jwks is not None
            and not force_refresh
            and time.monotonic() < self._jwks_expiry
        ):
            return self._jwks
        discovery = await self.discovery(client)
        try:
            response = await client.get(discovery["jwks_uri"])
            response.raise_for_status()
            jwks = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise A2AAuthenticationError("Keycloak JWKS is unavailable") from error
        with self._lock:
            self._jwks = jwks
            self._jwks_expiry = time.monotonic() + self._settings.jwks_cache_seconds
        return jwks


class KeycloakTokenValidator:
    """Validates inbound A2A bearer tokens against Keycloak's JWKS."""

    def __init__(self, settings: A2ASettings) -> None:
        self._settings = settings
        self._cache = _JWKSCache(settings)

    async def _signing_key(self, client: httpx.AsyncClient, token: str) -> Any:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as error:
            raise A2AAuthenticationError("Bearer token has an invalid JWT header") from error
        key_id = header.get("kid")
        if header.get("alg") != "RS256" or not key_id:
            raise A2AAuthenticationError("Bearer token uses an unsupported signing key")

        for refresh in (False, True):
            jwks = await self._cache.jwks(client, force_refresh=refresh)
            for jwk in jwks.get("keys", []):
                if jwk.get("kid") == key_id:
                    try:
                        return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
                    except (TypeError, ValueError) as error:
                        raise A2AAuthenticationError(
                            "Keycloak signing key is invalid"
                        ) from error
        raise A2AAuthenticationError("Bearer token signing key is not trusted by Keycloak")

    async def validate(self, token: str) -> dict[str, Any]:
        """Validate ``token``, returning its claims.

        Checks signature (RS256), issuer, the configured inbound audience,
        expiry, and that ``azp`` is one of the configured allowed callers.

        Raises:
            A2AAuthenticationError: on any validation failure.
        """
        if not token:
            raise A2AAuthenticationError("Bearer token is required")
        async with httpx.AsyncClient(
            verify=self._settings.tls_verify, timeout=self._settings.keycloak_timeout_seconds
        ) as client:
            key = await self._signing_key(client, token)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self._settings.inbound_audience,
                issuer=self._settings.keycloak_issuer_url,
                options={"require": ["exp", "iat", "iss", "sub"]},
            )
        except jwt.PyJWTError as error:
            raise A2AAuthenticationError(
                "Bearer token was rejected by Keycloak validation"
            ) from error
        if claims.get("azp") not in self._settings.inbound_azp:
            raise A2AAuthenticationError("Bearer token was not issued to an allowed caller")
        return claims


class SpiffeWorkloadIdentity:
    """Fetches this pod's own short-lived JWT-SVID from the Workload API."""

    def __init__(self, settings: A2ASettings) -> None:
        if not settings.spiffe_jwt_audience:
            raise A2AConfigurationError("A2A_SPIFFE_JWT_AUDIENCE must be configured")
        self._settings = settings

    def _fetch_sync(self) -> str:
        from spiffe import WorkloadApiClient

        try:
            with WorkloadApiClient(
                socket_path=self._settings.spiffe_endpoint_socket or None,
                default_timeout=self._settings.spiffe_timeout_seconds,
            ) as client:
                svid = client.fetch_jwt_svid(audience={self._settings.spiffe_jwt_audience})
        except Exception as error:  # noqa: BLE001 - SDK raises its own broad errors
            raise A2AWorkloadIdentityError("ZTWIM/SPIRE did not issue a JWT-SVID") from error
        token = str(getattr(svid, "token", "") or getattr(svid, "jwt_svid", "") or "")
        if not token:
            raise A2AWorkloadIdentityError("ZTWIM/SPIRE returned an empty JWT-SVID")
        return token

    async def fetch(self) -> str:
        """Fetch a fresh JWT-SVID for this pod, off the event loop thread."""
        import asyncio

        return await asyncio.to_thread(self._fetch_sync)


class KeycloakTokenExchanger:
    """Exchanges the caller's token for one scoped to the OpenShift MCP audience."""

    def __init__(self, settings: A2ASettings) -> None:
        self._settings = settings
        self._cache = _JWKSCache(settings)

    async def exchange(self, subject_token: str, client_assertion: str) -> str:
        """Perform the RFC 8693 token exchange ("Token A" -> "Token B").

        No ``client_id`` form parameter is sent: Keycloak's federated-JWT
        client validators reject the request outright if a ``client_id`` is
        present and differs from the assertion's ``sub`` claim, and for a
        SPIFFE assertion ``sub`` is a SPIFFE ID, never a Keycloak client id.
        The exchanging client is resolved from the assertion alone.

        Raises:
            A2AExchangeError: on any failure to obtain an access token.
        """
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": subject_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "client_assertion_type": self._settings.exchange_client_assertion_type,
            "client_assertion": client_assertion,
            "audience": self._settings.exchange_audience,
        }
        async with httpx.AsyncClient(
            verify=self._settings.tls_verify, timeout=self._settings.exchange_timeout_seconds
        ) as client:
            try:
                discovery = await self._cache.discovery(client)
                token_endpoint = discovery["token_endpoint"]
            except (A2AAuthenticationError, KeyError) as error:
                raise A2AExchangeError("Keycloak token endpoint discovery failed") from error
            try:
                response = await client.post(token_endpoint, data=data)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as error:
                raise A2AExchangeError(
                    "Keycloak token exchange for OpenShift MCP failed"
                ) from error
        token = payload.get("access_token")
        if not token:
            raise A2AExchangeError("Keycloak token exchange did not return access_token")
        return str(token)


class A2AAuthenticator:
    """Composes caller validation and MCP-token exchange for the A2A endpoint."""

    def __init__(self, settings: A2ASettings) -> None:
        self.settings = settings
        self._validator = KeycloakTokenValidator(settings)
        self._workload = SpiffeWorkloadIdentity(settings)
        self._exchanger = KeycloakTokenExchanger(settings)

    @staticmethod
    def _bearer_token(request: Request) -> str:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return ""
        return token.strip()

    async def authenticate_caller(self, request: Request) -> CallerIdentity:
        """Validate the inbound token and cluster header for one A2A request.

        This is the cheap, always-required check for every JSON-RPC call
        (``SendMessage`` and task lookups alike): it does not perform a
        token exchange, since only ``SendMessage`` ever calls MCP.

        Raises:
            HTTPException: 401 if the bearer token is missing/invalid, 403
                if the cluster header is missing, unknown, or the caller is
                not an allowed ``azp``.
        """
        token = self._bearer_token(request)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid Keycloak bearer token is required",
            )
        try:
            claims = await self._validator.validate(token)
        except A2AAuthenticationError as error:
            logger.warning("A2A inbound token rejected: %s", error)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token was rejected",
            ) from error

        cluster_header = request.headers.get(CLUSTER_HEADER_NAME, "").strip()
        if not cluster_header or cluster_header != self.settings.cluster_id:
            logger.warning(
                "A2A request rejected: missing or unknown %s (sub=%s)",
                CLUSTER_HEADER_NAME,
                str(claims.get("sub", ""))[:36],
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing or unknown {CLUSTER_HEADER_NAME} header",
            )

        sub = str(claims.get("sub", ""))
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token has no subject",
            )
        request_id = request.headers.get(REQUEST_ID_HEADER_NAME, "").strip() or uuid.uuid4().hex
        return CallerIdentity(
            sub=sub,
            claims=claims,
            cluster_id=cluster_header,
            token=token,
            request_id=request_id,
        )

    async def exchange_for_mcp(self, caller: CallerIdentity) -> str:
        """Fetch a fresh JWT-SVID and exchange it for an MCP-scoped token.

        Performed once per ``SendMessage`` call (never cached/reused), so
        every downstream MCP call is backed by a token minted for that
        specific request.

        Raises:
            A2AWorkloadIdentityError: if this pod cannot obtain its own
                SPIFFE identity.
            A2AExchangeError: if the Keycloak exchange fails.
        """
        jwt_svid = await self._workload.fetch()
        token = await self._exchanger.exchange(
            subject_token=caller.token,
            client_assertion=jwt_svid,
        )
        return token


_authenticator: Optional[A2AAuthenticator] = None
_authenticator_lock = threading.Lock()


def get_authenticator() -> A2AAuthenticator:
    """Return the process-wide ``A2AAuthenticator``, building it on first use."""
    global _authenticator  # noqa: PLW0603
    if _authenticator is None:
        with _authenticator_lock:
            if _authenticator is None:
                _authenticator = A2AAuthenticator(A2ASettings.from_env())
    return _authenticator


async def authenticate_request(request: Request) -> CallerIdentity:
    """FastAPI dependency: validate the inbound A2A caller and cluster header."""
    return await get_authenticator().authenticate_caller(request)

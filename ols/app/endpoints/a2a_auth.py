"""Authentication bridge and caller identity for the A2A endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import ssl
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from ols import config
from ols.src.auth.auth import get_auth_dependency

logger = logging.getLogger(__name__)

CLUSTER_HEADER_NAME = "X-OLS-Cluster"
REQUEST_ID_HEADER_NAME = "X-Request-Id"
AUDIT_LOG_PREFIX = "a2a_audit"
DEFAULT_EXCHANGE_AUDIENCE = "openshift-mcp"
DEFAULT_EXCHANGE_CLIENT_ASSERTION_TYPE = (
    "urn:ietf:params:oauth:client-assertion-type:jwt-spiffe"
)
EXCHANGE_CLIENT_AZP = "lightspeed-mcp"


class A2AConfigurationError(Exception):
    """Raised when required A2A configuration or authentication mode is invalid."""


class A2AWorkloadIdentityError(Exception):
    """Raised when SPIFFE cannot issue this workload a JWT-SVID."""


class A2AExchangeError(Exception):
    """Raised when the RFC 8693 exchange for an MCP-scoped token fails."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _positive_timeout(name: str, default: str) -> float:
    """Parse a finite timeout in the supported 0-300 second range."""
    try:
        value = float(_env(name, default))
    except ValueError as error:
        raise A2AConfigurationError(f"{name} must be a positive number") from error
    if not math.isfinite(value) or value <= 0 or value > 300:
        raise A2AConfigurationError(f"{name} must be greater than 0 and at most 300")
    return value


def _positive_cache_seconds(name: str, default: str) -> int:
    """Parse a positive discovery-cache duration no greater than one day."""
    try:
        value = int(_env(name, default))
    except ValueError as error:
        raise A2AConfigurationError(f"{name} must be a positive integer") from error
    if value <= 0 or value > 86_400:
        raise A2AConfigurationError(f"{name} must be between 1 and 86400")
    return value


@dataclass(frozen=True)
class A2ASettings:
    """A2A endpoint settings from OLS config and deployment environment."""

    cluster_id: str
    rpc_url: str
    agent_name: str
    agent_description: str
    token_exchange_enabled: bool = False
    keycloak_issuer_url: str = ""
    keycloak_ca_bundle: str = ""
    exchange_audience: str = DEFAULT_EXCHANGE_AUDIENCE
    exchange_client_assertion_type: str = DEFAULT_EXCHANGE_CLIENT_ASSERTION_TYPE
    spiffe_jwt_audience: str = ""
    spiffe_endpoint_socket: str = ""
    spiffe_timeout_seconds: float = 5.0
    keycloak_timeout_seconds: float = 5.0
    exchange_timeout_seconds: float = 10.0
    discovery_cache_seconds: int = 300

    @classmethod
    def from_env(cls) -> "A2ASettings":
        """Build settings from required cluster and public endpoint values."""
        cluster_id = _env("A2A_CLUSTER_ID")
        if not cluster_id:
            raise A2AConfigurationError("A2A_CLUSTER_ID must be configured")
        rpc_url = _env("A2A_RPC_URL")
        if not rpc_url:
            raise A2AConfigurationError("A2A_RPC_URL must be configured")
        token_exchange_enabled = config.a2a.token_exchange_enabled
        exchange_settings: dict[str, Any] = {}
        if token_exchange_enabled:
            issuer_url = _env("A2A_KEYCLOAK_ISSUER_URL")
            if not issuer_url:
                raise A2AConfigurationError(
                    "A2A_KEYCLOAK_ISSUER_URL must be configured when token exchange is enabled"
                )
            try:
                parsed_issuer_url = httpx.URL(issuer_url)
            except httpx.InvalidURL as error:
                raise A2AConfigurationError(
                    "A2A_KEYCLOAK_ISSUER_URL must be a valid HTTPS URL"
                ) from error
            if parsed_issuer_url.scheme != "https" or not parsed_issuer_url.host:
                raise A2AConfigurationError(
                    "A2A_KEYCLOAK_ISSUER_URL must use HTTPS and include a host"
                )
            exchange_settings = {
                "keycloak_issuer_url": issuer_url,
                "keycloak_ca_bundle": _env("A2A_KEYCLOAK_CA_BUNDLE"),
                "exchange_audience": _env(
                    "A2A_EXCHANGE_AUDIENCE", DEFAULT_EXCHANGE_AUDIENCE
                ),
                "exchange_client_assertion_type": _env(
                    "A2A_EXCHANGE_CLIENT_ASSERTION_TYPE",
                    DEFAULT_EXCHANGE_CLIENT_ASSERTION_TYPE,
                ),
                "spiffe_jwt_audience": _env("A2A_SPIFFE_JWT_AUDIENCE", issuer_url),
                "spiffe_endpoint_socket": _env("A2A_SPIFFE_ENDPOINT_SOCKET"),
                "spiffe_timeout_seconds": _positive_timeout(
                    "A2A_SPIFFE_TIMEOUT_SECONDS", "5"
                ),
                "keycloak_timeout_seconds": _positive_timeout(
                    "A2A_KEYCLOAK_TIMEOUT_SECONDS", "5"
                ),
                "exchange_timeout_seconds": _positive_timeout(
                    "A2A_TOKEN_EXCHANGE_TIMEOUT_SECONDS", "10"
                ),
                "discovery_cache_seconds": _positive_cache_seconds(
                    "A2A_KEYCLOAK_DISCOVERY_CACHE_SECONDS", "300"
                ),
            }
        return cls(
            cluster_id=cluster_id,
            rpc_url=rpc_url,
            agent_name=_env("A2A_AGENT_NAME", "openshift-lightspeed"),
            agent_description=_env(
                "A2A_AGENT_DESCRIPTION",
                "OpenShift Lightspeed: investigates and answers questions about "
                "this OpenShift cluster using the cluster's own MCP tools.",
            ),
            token_exchange_enabled=token_exchange_enabled,
            **exchange_settings,
        )

    @property
    def tls_verify(self) -> bool | ssl.SSLContext:
        """Trust system CAs and, when configured, the Keycloak CA bundle."""
        if not self.keycloak_ca_bundle:
            return True
        try:
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=self.keycloak_ca_bundle)
        except (OSError, ssl.SSLError) as error:
            raise A2AConfigurationError(
                "A2A_KEYCLOAK_CA_BUNDLE could not be loaded"
            ) from error
        return context


@dataclass(frozen=True)
class CallerIdentity:
    """A validated caller identity, bound to the cluster it targeted."""

    sub: str
    claims: dict[str, Any]
    cluster_id: str
    token: str
    request_id: str

    @property
    def task_scope(self) -> str:
        """Scope task and conversation state to one caller and cluster."""
        return f"{self.cluster_id}:{self.sub}"


def _audit_value(value: Any) -> str:
    """Render one audit value without allowing control characters to forge log lines."""
    text = str(value)
    if any(
        char.isspace()
        or char in {'"', "\\", "="}
        or ord(char) < 0x20
        or 0x7F <= ord(char) <= 0x9F
        for char in text
    ):
        return json.dumps(text, ensure_ascii=True)
    return text


def audit_record(
    caller: CallerIdentity,
    *,
    task_id: str,
    context_id: str,
    action: str,
    outcome: str,
    actor: str = "ols",
) -> str:
    """Build the greppable audit line for one A2A request."""
    fields = {
        "request_id": caller.request_id,
        "actor": actor,
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


class SpiffeWorkloadIdentity:
    """Fetch this workload's short-lived JWT-SVID from the SPIFFE Workload API."""

    def __init__(self, settings: A2ASettings) -> None:
        """Initialize SPIFFE settings and require an assertion audience."""
        if not settings.spiffe_jwt_audience:
            raise A2AConfigurationError("A2A_SPIFFE_JWT_AUDIENCE must be configured")
        self._settings = settings

    def _fetch_sync(self) -> str:
        """Fetch a JWT-SVID using the synchronous SPIFFE client."""
        from spiffe import WorkloadApiClient

        try:
            with WorkloadApiClient(
                socket_path=self._settings.spiffe_endpoint_socket or None,
                default_timeout=self._settings.spiffe_timeout_seconds,
            ) as client:
                svid = client.fetch_jwt_svid(
                    audience={self._settings.spiffe_jwt_audience}
                )
        except Exception as error:
            raise A2AWorkloadIdentityError("SPIFFE did not issue a JWT-SVID") from error
        token = str(getattr(svid, "token", "") or getattr(svid, "jwt_svid", "") or "")
        if not token:
            raise A2AWorkloadIdentityError("SPIFFE returned an empty JWT-SVID")
        return token

    async def fetch(self) -> str:
        """Fetch a fresh JWT-SVID off the event loop thread."""
        return await asyncio.to_thread(self._fetch_sync)


class KeycloakTokenExchanger:
    """Exchange a caller token for one scoped to the OpenShift MCP audience."""

    def __init__(self, settings: A2ASettings) -> None:
        """Initialize Keycloak exchange settings and discovery cache."""
        if not settings.keycloak_issuer_url:
            raise A2AConfigurationError(
                "A2A_KEYCLOAK_ISSUER_URL must be configured when token exchange is enabled"
            )
        try:
            issuer_url = httpx.URL(settings.keycloak_issuer_url)
        except httpx.InvalidURL as error:
            raise A2AConfigurationError(
                "A2A_KEYCLOAK_ISSUER_URL must be a valid HTTPS URL"
            ) from error
        if issuer_url.scheme != "https" or not issuer_url.host:
            raise A2AConfigurationError(
                "A2A_KEYCLOAK_ISSUER_URL must use HTTPS and include a host"
            )
        self._settings = settings
        self._discovery: dict[str, Any] | None = None
        self._discovery_expiry = 0.0
        self._discovery_lock = threading.Lock()

    async def exchange(self, subject_token: str, client_assertion: str) -> str:
        """Perform an RFC 8693 token exchange without sending a client id."""
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": subject_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "client_assertion_type": self._settings.exchange_client_assertion_type,
            "client_assertion": client_assertion,
            "audience": self._settings.exchange_audience,
        }
        try:
            verify = self._settings.tls_verify
        except (A2AConfigurationError, OSError, ssl.SSLError) as error:
            raise A2AExchangeError("Keycloak TLS configuration is invalid") from error

        try:
            async with httpx.AsyncClient(
                verify=verify,
                timeout=self._settings.exchange_timeout_seconds,
                follow_redirects=False,
            ) as client:
                token_endpoint = await self._token_endpoint(client)
                response = await client.post(token_endpoint, data=data)
                response.raise_for_status()
                payload = response.json()
        except A2AExchangeError:
            raise
        except (httpx.HTTPError, OSError, ssl.SSLError, ValueError) as error:
            raise A2AExchangeError("Keycloak token exchange failed") from error
        token = payload.get("access_token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise A2AExchangeError("Keycloak token exchange returned no access token")
        return token

    async def _token_endpoint(self, client: httpx.AsyncClient) -> str:
        """Return the cached Keycloak token endpoint or fetch discovery metadata."""
        if self._discovery and time.monotonic() < self._discovery_expiry:
            return self._discovery["token_endpoint"]

        issuer_url = self._settings.keycloak_issuer_url.rstrip("/")
        discovery_url = f"{issuer_url}/.well-known/openid-configuration"
        try:
            response = await client.get(
                discovery_url, timeout=self._settings.keycloak_timeout_seconds
            )
            response.raise_for_status()
            discovery = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise A2AExchangeError("Keycloak discovery is unavailable") from error
        if not isinstance(discovery, dict):
            raise A2AExchangeError("Keycloak discovery document is malformed")
        discovered_issuer = discovery.get("issuer")
        if (
            not isinstance(discovered_issuer, str)
            or discovered_issuer != self._settings.keycloak_issuer_url
        ):
            raise A2AExchangeError("Keycloak issuer does not match configuration")
        token_endpoint = discovery.get("token_endpoint")
        if not isinstance(token_endpoint, str) or not token_endpoint:
            raise A2AExchangeError("Keycloak discovery has no token endpoint")
        try:
            token_endpoint_url = httpx.URL(token_endpoint)
        except httpx.InvalidURL as error:
            raise A2AExchangeError("Keycloak token endpoint is invalid") from error
        if token_endpoint_url.scheme != "https" or not token_endpoint_url.host:
            raise A2AExchangeError("Keycloak token endpoint must use HTTPS")
        with self._discovery_lock:
            self._discovery = discovery
            self._discovery_expiry = (
                time.monotonic() + self._settings.discovery_cache_seconds
            )
        return token_endpoint


class A2AAuthenticator:
    """Authenticate A2A requests with OLS's configured authentication module."""

    def __init__(self, settings: A2ASettings) -> None:
        """Build the selected k8s or OIDC dependency once for this process."""
        self.settings = settings
        if config.dev_config.disable_auth:
            raise A2AConfigurationError("A2A does not support disabled authentication")
        auth_module = config.ols_config.authentication_config.module
        if auth_module not in {"k8s", "oidc"}:
            raise A2AConfigurationError(
                "A2A requires the k8s or oidc authentication module"
            )
        self._auth_module = auth_module
        self._auth_dependency = get_auth_dependency(
            config.ols_config, virtual_path="/ols-access"
        )
        self._workload = (
            SpiffeWorkloadIdentity(settings)
            if settings.token_exchange_enabled
            else None
        )
        self._exchanger = (
            KeycloakTokenExchanger(settings)
            if settings.token_exchange_enabled
            else None
        )

    @property
    def audit_actor(self) -> str:
        """Return the workload identity recorded as the actor for A2A queries."""
        return EXCHANGE_CLIENT_AZP if self.settings.token_exchange_enabled else "ols"

    async def authenticate_caller(self, request: Request) -> CallerIdentity:
        """Authenticate the request and enforce its configured cluster scope."""
        if config.dev_config.disable_auth:
            raise A2AConfigurationError("A2A does not support disabled authentication")
        if config.ols_config.authentication_config.module != self._auth_module:
            raise A2AConfigurationError(
                "A2A authentication module changed; restart the service to apply the change"
            )
        if self._auth_module == "k8s":
            auth_result = await run_in_threadpool(
                asyncio.run, self._auth_dependency(request)
            )
        else:
            auth_result = await self._auth_dependency(request)
        if config.dev_config.disable_auth:
            raise A2AConfigurationError("A2A does not support disabled authentication")
        if config.ols_config.authentication_config.module != self._auth_module:
            raise A2AConfigurationError(
                "A2A authentication module changed; restart the service to apply the change"
            )
        user_id, _, _, token = auth_result
        if not isinstance(token, str) or not token.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated caller has no token",
            )
        cluster_header = request.headers.get(CLUSTER_HEADER_NAME, "").strip()
        if not cluster_header or cluster_header != self.settings.cluster_id:
            logger.warning(
                "A2A request rejected: missing or unknown %s (user_id=%s)",
                CLUSTER_HEADER_NAME,
                str(user_id)[:36],
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing or unknown {CLUSTER_HEADER_NAME} header",
            )

        claims = getattr(request.state, "oidc_claims", {})
        subject = getattr(request.state, "oidc_subject", "")
        if not isinstance(subject, str) or not subject:
            subject = str(user_id)
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated caller has no subject",
            )
        request_id = (
            request.headers.get(REQUEST_ID_HEADER_NAME, "").strip() or uuid.uuid4().hex
        )
        return CallerIdentity(
            sub=subject,
            claims=claims if isinstance(claims, dict) else {},
            cluster_id=cluster_header,
            token=token,
            request_id=request_id,
        )

    async def exchange_for_mcp(self, caller: CallerIdentity) -> str:
        """Exchange the caller token when enabled, otherwise return it unchanged."""
        if not self.settings.token_exchange_enabled:
            return caller.token
        if self._workload is None or self._exchanger is None:
            raise A2AConfigurationError("Token exchange is enabled but not initialized")
        client_assertion = await self._workload.fetch()
        return await self._exchanger.exchange(
            subject_token=caller.token,
            client_assertion=client_assertion,
        )


_authenticator: Optional[A2AAuthenticator] = None
_authenticator_lock = threading.Lock()


def get_authenticator() -> A2AAuthenticator:
    """Return the process-wide authenticator, building it on first use."""
    global _authenticator
    if _authenticator is None:
        with _authenticator_lock:
            if _authenticator is None:
                _authenticator = A2AAuthenticator(A2ASettings.from_env())
    return _authenticator

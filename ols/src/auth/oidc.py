"""Authenticate bearer tokens using an OpenID Connect provider."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx
import jwt
from fastapi import HTTPException, Request

from ols import config
from ols.constants import DEFAULT_USER_NAME, DEFAULT_USER_UID, NO_USER_TOKEN

from .auth_dependency_interface import AuthDependencyInterface

if TYPE_CHECKING:
    from ols.app.models.config import AuthenticationConfig

logger = logging.getLogger(__name__)
METRICS_ACCESS_PATH = "/ols-metrics-access"
_PROVIDER_RETRY_BACKOFF_SECONDS = 1.0
_UNKNOWN_KID_REFRESH_BACKOFF_SECONDS = 5.0


class OIDCAuthenticationError(Exception):
    """Raised when an OIDC bearer token is not valid for this service."""


class OIDCProviderError(Exception):
    """Raised when the configured OIDC provider cannot be reached or used."""


class OIDCTokenValidator:
    """Validate JWT bearer tokens using the issuer's OIDC discovery and JWKS."""

    def __init__(self, authentication_config: AuthenticationConfig) -> None:
        """Initialize validation from the selected OIDC authentication settings."""
        self._issuer = str(authentication_config.oidc_issuer_url)
        issuer_url = httpx.URL(self._issuer)
        if issuer_url.scheme != "https" or not issuer_url.host:
            raise OIDCProviderError("OIDC issuer must use HTTPS")
        self._audience = authentication_config.oidc_audience or ""
        self._cache_seconds = authentication_config.oidc_jwks_cache_seconds
        self._discovery_document: dict[str, Any] | None = None
        self._discovery_expiry = 0.0
        self._jwks_document: dict[str, Any] | None = None
        self._jwks_expiry = 0.0
        self._jwks_refresh_lock = asyncio.Lock()
        self._provider_retry_after = 0.0
        self._unknown_kid_refresh_until = 0.0
        self._unknown_kid_refresh_lock = asyncio.Lock()

    async def validate(self, token: str) -> dict[str, Any]:
        """Verify token signature, issuer, audience, lifetime, and required claims."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as error:
            raise OIDCAuthenticationError(
                "Bearer token has an invalid JWT header"
            ) from error
        key_id = header.get("kid")
        if header.get("alg") != "RS256" or not isinstance(key_id, str) or not key_id:
            raise OIDCAuthenticationError(
                "Bearer token uses an unsupported signing key"
            )

        async with httpx.AsyncClient(verify=True, timeout=5.0) as client:
            key = await self._get_signing_key(client, key_id)

        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iat", "iss", "sub"]},
            )
        except jwt.PyJWTError as error:
            raise OIDCAuthenticationError(
                "Bearer token claims were rejected"
            ) from error

        if not isinstance(claims.get("sub"), str) or not claims["sub"].strip():
            raise OIDCAuthenticationError("Bearer token has no subject")
        return claims

    async def _get_signing_key(self, client: httpx.AsyncClient, key_id: str) -> Any:
        """Find a key while throttling refreshes caused by unknown key ids."""
        jwks = await self._get_jwks(client, force_refresh=False)
        key = self._key_from_jwks(jwks, key_id)
        if key is not None:
            return key

        async with self._unknown_kid_refresh_lock:
            jwks = await self._get_jwks(client, force_refresh=False)
            key = self._key_from_jwks(jwks, key_id)
            if key is not None:
                return key
            now = time.monotonic()
            if now < self._unknown_kid_refresh_until:
                raise OIDCAuthenticationError(
                    "Bearer token signing key is not trusted by the issuer"
                )
            jwks = await self._get_jwks(client, force_refresh=True)
            self._unknown_kid_refresh_until = (
                time.monotonic() + _UNKNOWN_KID_REFRESH_BACKOFF_SECONDS
            )
            key = self._key_from_jwks(jwks, key_id)
            if key is not None:
                return key
        raise OIDCAuthenticationError(
            "Bearer token signing key is not trusted by the issuer"
        )

    @staticmethod
    def _key_from_jwks(jwks: dict[str, Any], key_id: str) -> Any | None:
        """Return the matching RSA key, or None when the key id is not present."""
        for jwk in jwks.get("keys", []):
            if (
                jwk.get("kid") != key_id
                or jwk.get("kty") != "RSA"
                or jwk.get("use") not in (None, "sig")
                or jwk.get("alg") not in (None, "RS256")
            ):
                continue
            try:
                return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
            except (jwt.PyJWTError, TypeError, ValueError) as error:
                raise OIDCProviderError("OIDC signing key is invalid") from error
        return None

    @staticmethod
    def _validate_jwk(jwk: Any) -> None:
        """Reject malformed JWKS entries before caching the document."""
        if not isinstance(jwk, dict) or not isinstance(jwk.get("kty"), str):
            raise OIDCProviderError("OIDC signing keys are malformed")
        key_id = jwk.get("kid")
        if key_id is not None and (not isinstance(key_id, str) or not key_id):
            raise OIDCProviderError("OIDC signing keys are malformed")
        if jwk["kty"] == "RSA" and not all(
            isinstance(jwk.get(field), str) and jwk[field]
            for field in ("kid", "n", "e")
        ):
            raise OIDCProviderError("OIDC RSA signing key is malformed")
        if jwk["kty"] == "RSA":
            try:
                jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
            except (jwt.PyJWTError, TypeError, ValueError) as error:
                raise OIDCProviderError("OIDC RSA signing key is invalid") from error

    async def _get_jwks(
        self, client: httpx.AsyncClient, *, force_refresh: bool
    ) -> dict[str, Any]:
        """Return cached keys or perform one single-flight refresh."""
        async with self._jwks_refresh_lock:
            if (
                self._jwks_document is not None
                and not force_refresh
                and time.monotonic() < self._jwks_expiry
            ):
                return self._jwks_document
            if time.monotonic() < self._provider_retry_after:
                raise OIDCProviderError(
                    "OIDC provider refresh is temporarily throttled"
                )

            try:
                discovery = await self._get_discovery(
                    client, force_refresh=force_refresh
                )
                response = await client.get(discovery["jwks_uri"])
                response.raise_for_status()
                jwks = response.json()
                if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
                    raise OIDCProviderError("OIDC signing keys are malformed")
                for jwk in jwks["keys"]:
                    self._validate_jwk(jwk)
            except OIDCProviderError:
                self._provider_retry_after = (
                    time.monotonic() + _PROVIDER_RETRY_BACKOFF_SECONDS
                )
                raise
            except (httpx.HTTPError, KeyError, ValueError) as error:
                self._provider_retry_after = (
                    time.monotonic() + _PROVIDER_RETRY_BACKOFF_SECONDS
                )
                raise OIDCProviderError("OIDC signing keys are unavailable") from error

            self._jwks_document = jwks
            self._jwks_expiry = time.monotonic() + self._cache_seconds
            self._provider_retry_after = 0.0
            return jwks

    async def _get_discovery(
        self, client: httpx.AsyncClient, *, force_refresh: bool
    ) -> dict[str, Any]:
        """Return the cached OIDC discovery document or fetch a fresh one."""
        if (
            self._discovery_document is not None
            and not force_refresh
            and time.monotonic() < self._discovery_expiry
        ):
            return self._discovery_document

        discovery_url = f"{self._issuer.rstrip('/')}/.well-known/openid-configuration"
        try:
            response = await client.get(discovery_url)
            response.raise_for_status()
            discovery = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OIDCProviderError("OIDC discovery is unavailable") from error
        if not isinstance(discovery, dict):
            raise OIDCProviderError("OIDC discovery document is malformed")
        discovered_issuer = discovery.get("issuer")
        if not isinstance(discovered_issuer, str) or discovered_issuer != self._issuer:
            raise OIDCProviderError("OIDC issuer does not match configuration")
        if not isinstance(discovery.get("jwks_uri"), str) or not discovery["jwks_uri"]:
            raise OIDCProviderError("OIDC discovery does not contain jwks_uri")
        try:
            jwks_url = httpx.URL(discovery["jwks_uri"])
        except httpx.InvalidURL as error:
            raise OIDCProviderError("OIDC JWKS endpoint is invalid") from error
        if jwks_url.scheme != "https" or not jwks_url.host:
            raise OIDCProviderError("OIDC JWKS endpoint must use HTTPS")
        self._discovery_document = discovery
        self._discovery_expiry = time.monotonic() + self._cache_seconds
        return discovery


class MultiIssuerOIDCTokenValidator:
    """Validate JWT bearer tokens against multiple trusted OIDC issuers."""

    def __init__(self, issuer_configs: list[Any]) -> None:
        """Initialize validators for each configured issuer.

        Args:
            issuer_configs: List of OIDCIssuerConfig instances. Uses Any type due to
                circular import avoidance; runtime type safety provided by isinstance check.
        """
        from ols.app.models.config import AuthenticationConfig, OIDCIssuerConfig

        self._validators: dict[str, OIDCTokenValidator] = {}
        for issuer_config in issuer_configs:
            if not isinstance(issuer_config, OIDCIssuerConfig):
                raise OIDCProviderError("Invalid issuer configuration")
            auth_config = AuthenticationConfig(
                module="oidc",
                oidc_issuer_url=issuer_config.issuer_url,
                oidc_audience=issuer_config.audience,
                oidc_username_claim=issuer_config.username_claim,
                oidc_jwks_cache_seconds=issuer_config.jwks_cache_seconds,
            )
            self._validators[issuer_config.issuer_url] = OIDCTokenValidator(auth_config)

    async def validate(self, token: str) -> dict[str, Any]:
        """Extract issuer from token and validate with the matching validator."""
        try:
            unverified_claims = jwt.decode(
                token, options={"verify_signature": False, "verify_exp": False}
            )
        except jwt.PyJWTError as error:
            raise OIDCAuthenticationError(
                "Bearer token could not be decoded"
            ) from error

        issuer = unverified_claims.get("iss")
        if not isinstance(issuer, str) or not issuer:
            raise OIDCAuthenticationError("Bearer token has no issuer claim")

        validator = self._validators.get(issuer)
        if validator is None:
            logger.warning("Bearer token from untrusted issuer: %s", issuer)
            raise OIDCAuthenticationError(
                f"Bearer token issuer is not trusted: {issuer}"
            )

        return await validator.validate(token)


class AuthDependency(AuthDependencyInterface):
    """Validate OIDC bearer tokens and return OLS's standard auth tuple."""

    def __init__(
        self,
        authentication_config: AuthenticationConfig,
        virtual_path: str = "/ols-access",
    ) -> None:
        """Initialize the dependency and its path-specific authorization policy."""
        self._authentication_config = authentication_config
        self._virtual_path = virtual_path
        self._multi_issuer = bool(authentication_config.oidc_issuers)
        if self._multi_issuer:
            self._validator = MultiIssuerOIDCTokenValidator(
                authentication_config.oidc_issuers
            )
            self._issuer_config_map = {
                issuer.issuer_url: issuer
                for issuer in authentication_config.oidc_issuers
            }
        else:
            self._validator = OIDCTokenValidator(authentication_config)
            self._issuer_config_map = {}

    def _has_metrics_access(self, claims: dict[str, Any]) -> bool:
        """Return whether the configured OIDC claim grants metrics access."""
        claim_name = self._authentication_config.oidc_metrics_access_claim
        required_value = self._authentication_config.oidc_metrics_access_value
        if not claim_name or not required_value:
            return False
        claim_value = claims.get(claim_name)
        if isinstance(claim_value, str):
            return claim_value == required_value
        if isinstance(claim_value, list):
            return required_value in claim_value
        return False

    async def __call__(self, request: Request) -> tuple[str, str, bool, str]:
        """Authenticate one request while preserving the shared dependency contract."""
        if config.dev_config.disable_auth:
            if self._virtual_path == METRICS_ACCESS_PATH:
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: OIDC metrics access is not configured",
                )
            if (
                config.ols_config.logging_config is None
                or not config.ols_config.logging_config.suppress_auth_checks_warning_in_log
            ):
                logger.warning("Auth checks disabled, skipping")
            return DEFAULT_USER_UID, DEFAULT_USER_NAME, False, NO_USER_TOKEN

        authorization = request.headers.get("Authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not separator or not token.strip():
            raise HTTPException(
                status_code=401, detail="Unauthorized: Bearer token required"
            )

        try:
            claims = await self._validator.validate(token.strip())
        except OIDCAuthenticationError as error:
            logger.warning("OIDC bearer token rejected: %s", error)
            raise HTTPException(
                status_code=401, detail="Unauthorized: Invalid bearer token"
            ) from error
        except OIDCProviderError as error:
            logger.error("OIDC provider validation failed: %s", error)
            raise HTTPException(
                status_code=503, detail="OIDC identity provider is unavailable"
            ) from error

        if self._virtual_path == METRICS_ACCESS_PATH and not self._has_metrics_access(
            claims
        ):
            raise HTTPException(
                status_code=403,
                detail="Forbidden: OIDC metrics access is not authorized",
            )

        subject = claims["sub"]
        issuer = claims.get("iss", "")

        if self._multi_issuer:
            issuer_config = self._issuer_config_map.get(issuer)
            if issuer_config is None:
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized: Token issuer configuration not found",
                )
            claim_name = issuer_config.username_claim
        else:
            claim_name = self._authentication_config.oidc_username_claim
            issuer = str(self._authentication_config.oidc_issuer_url)

        username_claim = claims.get(claim_name)
        username = username_claim.strip() if isinstance(username_claim, str) else ""
        username = username or subject
        user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{issuer}:{subject}"))
        request.state.oidc_claims = claims
        request.state.oidc_subject = subject
        return user_id, username, False, token.strip()

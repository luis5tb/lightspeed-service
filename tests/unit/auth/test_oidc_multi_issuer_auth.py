"""Unit tests for multi-issuer OIDC bearer-token validation."""

from __future__ import annotations

import json
import time
from typing import Any, Callable
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Request

from ols.app.models.config import AuthenticationConfig, OIDCIssuerConfig
from ols.src.auth import oidc
from ols.utils.checks import InvalidConfigurationError

ISSUER_RH_SSO = "https://sso.redhat.com/auth/realms/redhat"
ISSUER_RH_SSO_STAGE = "https://sso.stage.redhat.com/auth/realms/redhat"
ISSUER_KEYCLOAK = "https://keycloak.example.com/realms/ols"
AUDIENCE_LIGHTSPEED = "openshift-lightspeed"
KEY_ID = "test-key"


@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[Any, Any]:
    """Create a test signing key pair."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture
def jwk(rsa_keypair: tuple[Any, Any]) -> dict[str, Any]:
    """Return the public key in the format used by an OIDC JWKS endpoint."""
    _, public_key = rsa_keypair
    key = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key))
    key.update({"kid": KEY_ID, "use": "sig", "alg": "RS256"})
    return key


def _multi_issuer_config(**overrides: Any) -> AuthenticationConfig:
    """Create an AuthenticationConfig with multiple issuers."""
    options = {
        "module": "oidc",
        "oidc_issuers": [
            OIDCIssuerConfig(
                issuer_url=ISSUER_RH_SSO,
                audience=AUDIENCE_LIGHTSPEED,
                username_claim="preferred_username",
            ),
            OIDCIssuerConfig(
                issuer_url=ISSUER_RH_SSO_STAGE,
                audience=AUDIENCE_LIGHTSPEED,
                username_claim="email",
            ),
            OIDCIssuerConfig(
                issuer_url=ISSUER_KEYCLOAK,
                audience=AUDIENCE_LIGHTSPEED,
                username_claim="preferred_username",
            ),
        ],
    }
    options.update(overrides)
    return AuthenticationConfig(**options)


def _token(
    private_key: Any,
    *,
    issuer: str = ISSUER_KEYCLOAK,
    audience: str = AUDIENCE_LIGHTSPEED,
    expires_in: int = 300,
    claims: dict[str, Any] | None = None,
    key_id: str = KEY_ID,
) -> str:
    """Create a signed JWT token."""
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": "user-123",
        "iat": now,
        "exp": now + expires_in,
        "preferred_username": "alice",
        "email": "alice@example.com",
    }
    if claims:
        payload.update(claims)
    return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": key_id})


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, responder: Callable[[str], dict[str, Any]], **_: Any) -> None:
        self._responder = responder

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(self._responder(url))


def _mock_provider(
    monkeypatch: pytest.MonkeyPatch,
    jwk: dict[str, Any],
    issuers: list[str],
) -> None:
    """Mock OIDC discovery and JWKS endpoints for multiple issuers."""

    def responder(url: str) -> dict[str, Any]:
        for issuer in issuers:
            discovery_url = f"{issuer}/.well-known/openid-configuration"
            if url == discovery_url:
                return {
                    "issuer": issuer,
                    "jwks_uri": f"{issuer}/protocol/openid-connect/certs",
                    "token_endpoint": f"{issuer}/protocol/openid-connect/token",
                }
            jwks_url = f"{issuer}/protocol/openid-connect/certs"
            if url == jwks_url:
                return {"keys": [jwk]}
        raise ValueError(f"Unexpected URL: {url}")

    monkeypatch.setattr("httpx.AsyncClient", lambda **_: _FakeAsyncClient(responder))


@pytest.mark.asyncio
async def test_multi_issuer_validator_accepts_tokens_from_all_issuers(
    monkeypatch: pytest.MonkeyPatch,
    jwk: dict[str, Any],
    rsa_keypair: tuple[Any, Any],
) -> None:
    """Verify that multi-issuer validator accepts valid tokens from all configured issuers."""
    private_key, _ = rsa_keypair
    _mock_provider(
        monkeypatch,
        jwk,
        [ISSUER_RH_SSO, ISSUER_RH_SSO_STAGE, ISSUER_KEYCLOAK],
    )

    validator = oidc.MultiIssuerOIDCTokenValidator(
        [
            OIDCIssuerConfig(
                issuer_url=ISSUER_RH_SSO,
                audience=AUDIENCE_LIGHTSPEED,
            ),
            OIDCIssuerConfig(
                issuer_url=ISSUER_RH_SSO_STAGE,
                audience=AUDIENCE_LIGHTSPEED,
            ),
            OIDCIssuerConfig(
                issuer_url=ISSUER_KEYCLOAK,
                audience=AUDIENCE_LIGHTSPEED,
            ),
        ]
    )

    token_rh_sso = _token(private_key, issuer=ISSUER_RH_SSO)
    claims_rh_sso = await validator.validate(token_rh_sso)
    assert claims_rh_sso["iss"] == ISSUER_RH_SSO
    assert claims_rh_sso["sub"] == "user-123"

    token_rh_sso_stage = _token(private_key, issuer=ISSUER_RH_SSO_STAGE)
    claims_rh_sso_stage = await validator.validate(token_rh_sso_stage)
    assert claims_rh_sso_stage["iss"] == ISSUER_RH_SSO_STAGE
    assert claims_rh_sso_stage["sub"] == "user-123"

    token_keycloak = _token(private_key, issuer=ISSUER_KEYCLOAK)
    claims_keycloak = await validator.validate(token_keycloak)
    assert claims_keycloak["iss"] == ISSUER_KEYCLOAK
    assert claims_keycloak["sub"] == "user-123"


@pytest.mark.asyncio
async def test_multi_issuer_validator_rejects_unknown_issuer(
    rsa_keypair: tuple[Any, Any],
) -> None:
    """Verify that multi-issuer validator rejects tokens from unknown issuers."""
    private_key, _ = rsa_keypair

    validator = oidc.MultiIssuerOIDCTokenValidator(
        [
            OIDCIssuerConfig(
                issuer_url=ISSUER_KEYCLOAK,
                audience=AUDIENCE_LIGHTSPEED,
            ),
        ]
    )

    token_unknown = _token(
        private_key, issuer="https://unknown-issuer.example.com/realms/test"
    )

    with pytest.raises(oidc.OIDCAuthenticationError, match="not trusted"):
        await validator.validate(token_unknown)


@pytest.mark.asyncio
async def test_multi_issuer_validator_rejects_token_without_issuer(
    rsa_keypair: tuple[Any, Any],
) -> None:
    """Verify that multi-issuer validator rejects tokens without an issuer claim."""
    private_key, _ = rsa_keypair

    validator = oidc.MultiIssuerOIDCTokenValidator(
        [
            OIDCIssuerConfig(
                issuer_url=ISSUER_KEYCLOAK,
                audience=AUDIENCE_LIGHTSPEED,
            ),
        ]
    )

    token_no_issuer = _token(private_key, claims={"iss": None})

    with pytest.raises(oidc.OIDCAuthenticationError, match="no issuer claim"):
        await validator.validate(token_no_issuer)


@pytest.mark.asyncio
async def test_multi_issuer_auth_dependency_uses_per_issuer_username_claim(
    monkeypatch: pytest.MonkeyPatch,
    jwk: dict[str, Any],
    rsa_keypair: tuple[Any, Any],
) -> None:
    """Verify that multi-issuer AuthDependency extracts username from per-issuer claim."""
    private_key, _ = rsa_keypair
    _mock_provider(
        monkeypatch,
        jwk,
        [ISSUER_RH_SSO, ISSUER_RH_SSO_STAGE, ISSUER_KEYCLOAK],
    )

    with patch("ols.src.auth.oidc.config") as mock_config:
        mock_config.dev_config.disable_auth = False

        auth_config = AuthenticationConfig(
            module="oidc",
            oidc_issuers=[
                OIDCIssuerConfig(
                    issuer_url=ISSUER_RH_SSO,
                    audience=AUDIENCE_LIGHTSPEED,
                    username_claim="preferred_username",
                ),
                OIDCIssuerConfig(
                    issuer_url=ISSUER_RH_SSO_STAGE,
                    audience=AUDIENCE_LIGHTSPEED,
                    username_claim="email",
                ),
            ],
        )
        dependency = oidc.AuthDependency(auth_config)

        token_rh_sso = _token(
            private_key,
            issuer=ISSUER_RH_SSO,
            claims={"preferred_username": "bob", "email": "bob@redhat.com"},
        )
        request_rh_sso = Request(
            {
                "type": "http",
                "headers": [(b"authorization", f"Bearer {token_rh_sso}".encode())],
            }
        )
        _, username, _, _ = await dependency(request_rh_sso)
        assert username == "bob"

        token_rh_sso_stage = _token(
            private_key,
            issuer=ISSUER_RH_SSO_STAGE,
            claims={"preferred_username": "charlie", "email": "charlie@redhat.com"},
        )
        request_rh_sso_stage = Request(
            {
                "type": "http",
                "headers": [
                    (b"authorization", f"Bearer {token_rh_sso_stage}".encode())
                ],
            }
        )
        _, username_stage, _, _ = await dependency(request_rh_sso_stage)
        assert username_stage == "charlie@redhat.com"


@pytest.mark.asyncio
async def test_multi_issuer_auth_dependency_generates_stable_user_id(
    monkeypatch: pytest.MonkeyPatch,
    jwk: dict[str, Any],
    rsa_keypair: tuple[Any, Any],
) -> None:
    """Verify that multi-issuer AuthDependency generates stable user IDs per issuer."""
    private_key, _ = rsa_keypair
    _mock_provider(
        monkeypatch,
        jwk,
        [ISSUER_RH_SSO, ISSUER_KEYCLOAK],
    )

    with patch("ols.src.auth.oidc.config") as mock_config:
        mock_config.dev_config.disable_auth = False

        auth_config = AuthenticationConfig(
            module="oidc",
            oidc_issuers=[
                OIDCIssuerConfig(
                    issuer_url=ISSUER_RH_SSO,
                    audience=AUDIENCE_LIGHTSPEED,
                ),
                OIDCIssuerConfig(
                    issuer_url=ISSUER_KEYCLOAK,
                    audience=AUDIENCE_LIGHTSPEED,
                ),
            ],
        )
        dependency = oidc.AuthDependency(auth_config)

        token_rh_sso = _token(private_key, issuer=ISSUER_RH_SSO)
        request_rh_sso = Request(
            {
                "type": "http",
                "headers": [(b"authorization", f"Bearer {token_rh_sso}".encode())],
            }
        )
        user_id_rh_sso, _, _, _ = await dependency(request_rh_sso)

        token_keycloak = _token(private_key, issuer=ISSUER_KEYCLOAK)
        request_keycloak = Request(
            {
                "type": "http",
                "headers": [(b"authorization", f"Bearer {token_keycloak}".encode())],
            }
        )
        user_id_keycloak, _, _, _ = await dependency(request_keycloak)

        assert user_id_rh_sso != user_id_keycloak


@pytest.mark.asyncio
async def test_backward_compatibility_single_issuer_still_works(
    monkeypatch: pytest.MonkeyPatch,
    jwk: dict[str, Any],
    rsa_keypair: tuple[Any, Any],
) -> None:
    """Verify that single-issuer configuration still works for backward compatibility."""
    private_key, _ = rsa_keypair
    _mock_provider(monkeypatch, jwk, [ISSUER_KEYCLOAK])

    with patch("ols.src.auth.oidc.config") as mock_config:
        mock_config.dev_config.disable_auth = False

        auth_config = AuthenticationConfig(
            module="oidc",
            oidc_issuer_url=ISSUER_KEYCLOAK,
            oidc_audience=AUDIENCE_LIGHTSPEED,
            oidc_username_claim="preferred_username",
        )
        dependency = oidc.AuthDependency(auth_config)

        token = _token(private_key, issuer=ISSUER_KEYCLOAK)
        request = Request(
            {
                "type": "http",
                "headers": [(b"authorization", f"Bearer {token}".encode())],
            }
        )
        _, username, _, _ = await dependency(request)
        assert username == "alice"


def test_config_validation_rejects_both_single_and_multi_issuer() -> None:
    """Verify that config validation rejects setting both oidc_issuer_url and oidc_issuers."""
    with pytest.raises(InvalidConfigurationError, match="Cannot configure both"):
        AuthenticationConfig(
            module="oidc",
            oidc_issuer_url=ISSUER_KEYCLOAK,
            oidc_audience=AUDIENCE_LIGHTSPEED,
            oidc_issuers=[
                OIDCIssuerConfig(
                    issuer_url=ISSUER_RH_SSO,
                    audience=AUDIENCE_LIGHTSPEED,
                )
            ],
        ).validate_yaml()


def test_config_validation_requires_at_least_one_issuer() -> None:
    """Verify that config validation requires either oidc_issuer_url or oidc_issuers."""
    with pytest.raises(InvalidConfigurationError, match="is required"):
        AuthenticationConfig(module="oidc").validate_yaml()


def test_config_validation_rejects_empty_issuer_list() -> None:
    """Verify that config validation rejects an empty oidc_issuers list."""
    with pytest.raises(InvalidConfigurationError, match="at least one issuer"):
        AuthenticationConfig(
            module="oidc",
            oidc_issuers=[],
        ).validate_yaml()


def test_config_validation_rejects_duplicate_issuers() -> None:
    """Verify that config validation rejects duplicate issuer URLs."""
    with pytest.raises(InvalidConfigurationError, match="duplicate issuer"):
        AuthenticationConfig(
            module="oidc",
            oidc_issuers=[
                OIDCIssuerConfig(
                    issuer_url=ISSUER_KEYCLOAK,
                    audience=AUDIENCE_LIGHTSPEED,
                ),
                OIDCIssuerConfig(
                    issuer_url=ISSUER_KEYCLOAK,
                    audience=AUDIENCE_LIGHTSPEED,
                ),
            ],
        ).validate_yaml()

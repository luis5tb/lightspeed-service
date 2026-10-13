"""Unit tests for OIDC bearer-token validation."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable
from unittest.mock import AsyncMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, Request

from ols.app.models.config import AuthenticationConfig
from ols.src.auth import oidc

ISSUER = "https://issuer.example.com/realms/ols"
AUDIENCE = "lightspeed"
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


def _config(**overrides: Any) -> AuthenticationConfig:
    options = {
        "module": "oidc",
        "oidc_issuer_url": ISSUER,
        "oidc_audience": AUDIENCE,
    }
    options.update(overrides)
    return AuthenticationConfig(**options)


def _token(
    private_key: Any,
    *,
    issuer: str = ISSUER,
    audience: str = AUDIENCE,
    expires_in: int = 300,
    claims: dict[str, Any] | None = None,
    key_id: str = KEY_ID,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "sub": "user-123",
        "iat": now,
        "exp": now + expires_in,
        "preferred_username": "alice",
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
    *,
    issuer: str = ISSUER,
    jwks_uri: str | None = None,
    request_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Patch OIDC metadata requests and return HTTP client options used."""
    signing_keys_url = jwks_uri or f"{issuer.rstrip('/')}/jwks"
    client_options: list[dict[str, Any]] = []

    def responder(url: str) -> dict[str, Any]:
        if request_urls is not None:
            request_urls.append(url)
        if url.endswith("/.well-known/openid-configuration"):
            return {"issuer": issuer, "jwks_uri": signing_keys_url}
        if url == signing_keys_url:
            return {"keys": [jwk]}
        raise AssertionError(f"Unexpected provider URL: {url}")

    def make_client(**kwargs: Any) -> _FakeAsyncClient:
        client_options.append(kwargs)
        return _FakeAsyncClient(responder, **kwargs)

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        make_client,
    )
    return client_options


@pytest.mark.asyncio
async def test_oidc_validator_accepts_valid_signed_token(
    monkeypatch: pytest.MonkeyPatch, rsa_keypair: tuple[Any, Any], jwk: dict[str, Any]
) -> None:
    """Accept a valid signature, issuer, audience, expiry, and required claims."""
    _mock_provider(monkeypatch, jwk)
    validator = oidc.OIDCTokenValidator(_config())

    claims = await validator.validate(_token(rsa_keypair[0]))

    assert claims["sub"] == "user-123"
    assert claims["preferred_username"] == "alice"


@pytest.mark.asyncio
async def test_oidc_validator_preserves_bare_host_issuer(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Compare discovery and JWT issuer values without URL normalization."""
    issuer = "https://issuer.example.com"
    request_urls: list[str] = []
    _mock_provider(monkeypatch, jwk, issuer=issuer, request_urls=request_urls)
    settings = _config(oidc_issuer_url=issuer)
    assert settings.oidc_issuer_url == issuer

    claims = await oidc.OIDCTokenValidator(settings).validate(
        _token(rsa_keypair[0], issuer=issuer)
    )

    assert claims["iss"] == issuer
    assert request_urls == [
        f"{issuer}/.well-known/openid-configuration",
        f"{issuer}/jwks",
    ]


def test_oidc_validator_rejects_http_issuer() -> None:
    """Require an HTTPS issuer before any OIDC network request can occur."""
    with pytest.raises(oidc.OIDCProviderError, match="issuer must use HTTPS"):
        oidc.OIDCTokenValidator(
            _config(oidc_issuer_url="http://issuer.example.com/realms/ols")
        )


@pytest.mark.asyncio
async def test_oidc_validation_keeps_tls_verification_enabled(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Ignore the Kubernetes TLS bypass setting for OIDC discovery and JWKS."""
    client_options = _mock_provider(monkeypatch, jwk)
    validator = oidc.OIDCTokenValidator(_config(skip_tls_verification=True))

    await validator.validate(_token(rsa_keypair[0]))

    assert client_options
    assert all(options["verify"] is True for options in client_options)


@pytest.mark.asyncio
async def test_oidc_validator_rejects_http_jwks_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Reject an insecure JWKS URL before sending a request to it."""
    client_options = _mock_provider(
        monkeypatch, jwk, jwks_uri="http://issuer.example/jwks"
    )
    validator = oidc.OIDCTokenValidator(_config())

    with pytest.raises(oidc.OIDCProviderError, match="JWKS endpoint must use HTTPS"):
        await validator.validate(_token(rsa_keypair[0]))

    assert len(client_options) == 1


@pytest.mark.asyncio
async def test_oidc_malformed_jwks_is_not_cached_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Reject malformed key entries, back off briefly, then accept recovered JWKS."""
    clock = [10.0]
    monkeypatch.setattr(oidc.time, "monotonic", lambda: clock[0])
    discovery_requests = 0
    jwks_requests = 0

    def responder(url: str) -> dict[str, Any]:
        nonlocal discovery_requests, jwks_requests
        if url.endswith("/.well-known/openid-configuration"):
            discovery_requests += 1
            return {"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}
        if url == f"{ISSUER}/jwks":
            jwks_requests += 1
            return {"keys": [None]} if jwks_requests == 1 else {"keys": [jwk]}
        raise AssertionError(f"Unexpected provider URL: {url}")

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(responder, **kwargs),
    )
    validator = oidc.OIDCTokenValidator(_config(oidc_jwks_cache_seconds=60))
    token = _token(rsa_keypair[0])

    with pytest.raises(oidc.OIDCProviderError, match="signing keys are malformed"):
        await validator.validate(token)
    assert validator._jwks_document is None

    clock[0] = 10.5
    with pytest.raises(oidc.OIDCProviderError, match="temporarily throttled"):
        await validator.validate(token)
    assert jwks_requests == 1

    clock[0] = 11.1
    claims = await validator.validate(token)

    assert claims["sub"] == "user-123"
    assert discovery_requests == 1
    assert jwks_requests == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_parameter", "invalid_value"),
    [("n", "not-base64!!"), ("e", "AA")],
)
async def test_oidc_invalid_rsa_material_is_not_cached_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
    bad_parameter: str,
    invalid_value: str,
) -> None:
    """Reject invalid RSA parameters before caching keys and recover after backoff."""
    clock = [20.0]
    monkeypatch.setattr(oidc.time, "monotonic", lambda: clock[0])
    invalid_jwk = dict(jwk)
    invalid_jwk[bad_parameter] = invalid_value
    jwks_requests = 0

    def responder(url: str) -> dict[str, Any]:
        nonlocal jwks_requests
        if url.endswith("/.well-known/openid-configuration"):
            return {"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}
        if url == f"{ISSUER}/jwks":
            jwks_requests += 1
            return {"keys": [invalid_jwk if jwks_requests == 1 else jwk]}
        raise AssertionError(f"Unexpected provider URL: {url}")

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(responder, **kwargs),
    )
    validator = oidc.OIDCTokenValidator(_config(oidc_jwks_cache_seconds=60))
    token = _token(rsa_keypair[0])

    with pytest.raises(oidc.OIDCProviderError, match="RSA signing key is invalid"):
        await validator.validate(token)
    assert validator._jwks_document is None

    clock[0] = 20.5
    with pytest.raises(oidc.OIDCProviderError, match="temporarily throttled"):
        await validator.validate(token)
    assert validator._jwks_document is None
    assert jwks_requests == 1

    clock[0] = 21.1
    claims = await validator.validate(token)

    assert claims["sub"] == "user-123"
    assert jwks_requests == 2


@pytest.mark.asyncio
async def test_oidc_unknown_kids_are_throttled_during_fresh_jwks_cache(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Bound JWKS refreshes caused by repeated unauthenticated unknown key ids."""
    monkeypatch.setattr(oidc.time, "monotonic", lambda: 100.0)
    request_urls: list[str] = []
    _mock_provider(monkeypatch, jwk, request_urls=request_urls)
    validator = oidc.OIDCTokenValidator(_config(oidc_jwks_cache_seconds=60))

    await validator.validate(_token(rsa_keypair[0]))
    for key_id in ("unknown-one", "unknown-two", "unknown-three"):
        with pytest.raises(oidc.OIDCAuthenticationError):
            await validator.validate(_token(rsa_keypair[0], key_id=key_id))

    discovery_url = f"{ISSUER}/.well-known/openid-configuration"
    jwks_url = f"{ISSUER}/jwks"
    assert request_urls.count(discovery_url) == 2
    assert request_urls.count(jwks_url) == 2


@pytest.mark.asyncio
async def test_oidc_key_rotation_recovers_after_unknown_kid_probe(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Let a legitimate key rotation refresh shortly after an unknown-kid probe."""
    clock = [100.0]
    monkeypatch.setattr(oidc.time, "monotonic", lambda: clock[0])
    rotated_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rotated_public_key = rotated_private_key.public_key()
    rotated_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(rotated_public_key))
    rotated_jwk.update({"kid": "rotated-key", "use": "sig", "alg": "RS256"})
    published_keys = [jwk]
    jwks_requests = 0

    def responder(url: str) -> dict[str, Any]:
        nonlocal jwks_requests
        if url.endswith("/.well-known/openid-configuration"):
            return {"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}
        if url == f"{ISSUER}/jwks":
            jwks_requests += 1
            return {"keys": list(published_keys)}
        raise AssertionError(f"Unexpected provider URL: {url}")

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(responder, **kwargs),
    )
    validator = oidc.OIDCTokenValidator(_config(oidc_jwks_cache_seconds=60))
    await validator.validate(_token(rsa_keypair[0]))

    with pytest.raises(oidc.OIDCAuthenticationError):
        await validator.validate(_token(rsa_keypair[0], key_id="attacker-controlled"))

    published_keys[:] = [rotated_jwk]
    clock[0] = 105.1
    claims = await validator.validate(_token(rotated_private_key, key_id="rotated-key"))

    assert claims["sub"] == "user-123"
    assert jwks_requests == 3


@pytest.mark.asyncio
async def test_oidc_unknown_kid_refresh_recovers_from_key_rotation(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Accept a newly published signing key after one unknown-kid refresh."""
    new_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    new_public_key = new_private_key.public_key()
    new_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(new_public_key))
    new_jwk.update({"kid": "rotated-key", "use": "sig", "alg": "RS256"})
    jwks_requests = 0
    request_urls: list[str] = []

    def responder(url: str) -> dict[str, Any]:
        nonlocal jwks_requests
        request_urls.append(url)
        if url.endswith("/.well-known/openid-configuration"):
            return {"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}
        if url == f"{ISSUER}/jwks":
            jwks_requests += 1
            return {"keys": [jwk if jwks_requests == 1 else new_jwk]}
        raise AssertionError(f"Unexpected provider URL: {url}")

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(responder, **kwargs),
    )
    validator = oidc.OIDCTokenValidator(_config(oidc_jwks_cache_seconds=60))
    await validator.validate(_token(rsa_keypair[0]))

    claims = await validator.validate(_token(new_private_key, key_id="rotated-key"))

    assert claims["sub"] == "user-123"
    assert jwks_requests == 2


@pytest.mark.asyncio
async def test_oidc_failed_unknown_kid_refresh_retries_after_short_backoff(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Recover rotated signing keys after provider failure without waiting for cache expiry."""
    clock = [100.0]
    monkeypatch.setattr(oidc.time, "monotonic", lambda: clock[0])
    new_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    new_public_key = new_private_key.public_key()
    new_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(new_public_key))
    new_jwk.update({"kid": "rotated-key", "use": "sig", "alg": "RS256"})
    validator = oidc.OIDCTokenValidator(_config(oidc_jwks_cache_seconds=60))
    _mock_provider(monkeypatch, jwk)
    await validator.validate(_token(rsa_keypair[0]))

    release_requests = asyncio.Event()
    provider_requests = {"discovery": 0, "jwks": 0}
    provider_recovered = False

    async def respond(url: str) -> _FakeResponse:
        if url.endswith("/.well-known/openid-configuration"):
            provider_requests["discovery"] += 1
            response = _FakeResponse({"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        elif url == f"{ISSUER}/jwks":
            provider_requests["jwks"] += 1
            response = _FakeResponse({"keys": [new_jwk]})
        else:
            raise AssertionError(f"Unexpected provider URL: {url}")
        if sum(provider_requests.values()) == 1:
            asyncio.get_running_loop().call_soon(release_requests.set)
        await release_requests.wait()
        if url == f"{ISSUER}/jwks" and not provider_recovered:
            raise oidc.httpx.ConnectError("temporary issuer outage")
        return response

    class _ConcurrentRefreshClient:
        async def __aenter__(self) -> _ConcurrentRefreshClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            return await respond(url)

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **_kwargs: _ConcurrentRefreshClient(),
    )
    rotated_token = _token(new_private_key, key_id="rotated-key")
    failed_refreshes = await asyncio.gather(
        *(validator.validate(rotated_token) for _ in range(8)),
        return_exceptions=True,
    )
    assert all(
        isinstance(result, oidc.OIDCProviderError) for result in failed_refreshes
    )
    assert provider_requests == {"discovery": 1, "jwks": 1}

    clock[0] = 100.5
    with pytest.raises(oidc.OIDCProviderError):
        await validator.validate(rotated_token)
    assert provider_requests == {"discovery": 1, "jwks": 1}

    provider_recovered = True
    clock[0] = 101.1
    recovered_claims = await asyncio.gather(
        *(validator.validate(rotated_token) for _ in range(8))
    )

    assert all(claims["sub"] == "user-123" for claims in recovered_claims)
    assert clock[0] < 100 + 60
    assert provider_requests == {"discovery": 2, "jwks": 2}


@pytest.mark.asyncio
async def test_oidc_concurrent_cold_cache_fetch_is_single_flight(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Coalesce concurrent discovery and JWKS cache misses into one fetch."""
    release_requests = asyncio.Event()
    request_counts = {"discovery": 0, "jwks": 0}

    async def respond(url: str) -> _FakeResponse:
        if url.endswith("/.well-known/openid-configuration"):
            request_counts["discovery"] += 1
            response = _FakeResponse({"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        elif url == f"{ISSUER}/jwks":
            request_counts["jwks"] += 1
            response = _FakeResponse({"keys": [jwk]})
        else:
            raise AssertionError(f"Unexpected provider URL: {url}")
        if sum(request_counts.values()) == 1:
            asyncio.get_running_loop().call_soon(release_requests.set)
        await release_requests.wait()
        return response

    class _ConcurrentFakeClient:
        async def __aenter__(self) -> _ConcurrentFakeClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            return await respond(url)

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **_kwargs: _ConcurrentFakeClient(),
    )
    validator = oidc.OIDCTokenValidator(_config())
    token = _token(rsa_keypair[0])

    results = await asyncio.gather(*(validator.validate(token) for _ in range(8)))

    assert all(claims["sub"] == "user-123" for claims in results)
    assert request_counts == {"discovery": 1, "jwks": 1}


@pytest.mark.asyncio
async def test_oidc_concurrent_expired_cache_fetch_is_single_flight(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Coalesce concurrent discovery and JWKS refreshes after cache expiry."""
    clock = [100.0]
    monkeypatch.setattr(oidc.time, "monotonic", lambda: clock[0])
    validator = oidc.OIDCTokenValidator(_config(oidc_jwks_cache_seconds=60))
    token = _token(rsa_keypair[0])
    initial_requests: list[str] = []
    _mock_provider(monkeypatch, jwk, request_urls=initial_requests)
    await validator.validate(token)
    assert len(initial_requests) == 2

    clock[0] = 161.0
    release_requests = asyncio.Event()
    request_counts = {"discovery": 0, "jwks": 0}

    async def respond(url: str) -> _FakeResponse:
        if url.endswith("/.well-known/openid-configuration"):
            request_counts["discovery"] += 1
            response = _FakeResponse({"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        elif url == f"{ISSUER}/jwks":
            request_counts["jwks"] += 1
            response = _FakeResponse({"keys": [jwk]})
        else:
            raise AssertionError(f"Unexpected provider URL: {url}")
        if sum(request_counts.values()) == 1:
            asyncio.get_running_loop().call_soon(release_requests.set)
        await release_requests.wait()
        return response

    class _ConcurrentRefreshClient:
        async def __aenter__(self) -> _ConcurrentRefreshClient:
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            return await respond(url)

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **_kwargs: _ConcurrentRefreshClient(),
    )

    results = await asyncio.gather(*(validator.validate(token) for _ in range(8)))

    assert all(claims["sub"] == "user-123" for claims in results)
    assert request_counts == {"discovery": 1, "jwks": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_options", "expected_error"),
    [
        ({"issuer": "https://other.example.com"}, "claims"),
        ({"audience": "another-service"}, "claims"),
        ({"expires_in": -30}, "claims"),
        ({"claims": {"sub": ""}}, "subject"),
    ],
)
async def test_oidc_validator_rejects_invalid_claims(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
    token_options: dict[str, Any],
    expected_error: str,
) -> None:
    """Reject invalid issuer, audience, expiry, and subject claims."""
    _mock_provider(monkeypatch, jwk)
    validator = oidc.OIDCTokenValidator(_config())

    with pytest.raises(oidc.OIDCAuthenticationError, match=expected_error):
        await validator.validate(_token(rsa_keypair[0], **token_options))


@pytest.mark.asyncio
async def test_oidc_validator_rejects_invalid_signature(
    monkeypatch: pytest.MonkeyPatch, jwk: dict[str, Any]
) -> None:
    """Reject a JWT that is signed by a key outside the issuer's JWKS."""
    _mock_provider(monkeypatch, jwk)
    unrelated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    validator = oidc.OIDCTokenValidator(_config())

    with pytest.raises(oidc.OIDCAuthenticationError, match="claims"):
        await validator.validate(_token(unrelated_key))


@pytest.mark.asyncio
async def test_oidc_validator_rejects_missing_required_claims(
    monkeypatch: pytest.MonkeyPatch, rsa_keypair: tuple[Any, Any], jwk: dict[str, Any]
) -> None:
    """Reject a token missing an OIDC-required expiry claim."""
    _mock_provider(monkeypatch, jwk)
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-123",
        "iat": int(time.time()),
    }
    token = jwt.encode(
        payload, rsa_keypair[0], algorithm="RS256", headers={"kid": KEY_ID}
    )

    with pytest.raises(oidc.OIDCAuthenticationError, match="claims"):
        await oidc.OIDCTokenValidator(_config()).validate(token)


@pytest.mark.asyncio
async def test_oidc_auth_dependency_preserves_auth_tuple_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return user UUID, username, skip flag, and caller token as the shared contract."""
    dependency = oidc.AuthDependency(_config())
    dependency._validator.validate = AsyncMock(
        return_value={"sub": "subject-1", "preferred_username": "alice"}
    )
    monkeypatch.setattr(oidc.config.dev_config, "disable_auth", False)
    request = Request(
        {"type": "http", "headers": [(b"authorization", b"Bearer caller-token")]}
    )

    result = await dependency(request)

    assert len(result) == 4
    assert result[0] == "10770943-e0a2-5aa9-97e1-8c988011ac98"
    assert result[1:] == ("alice", False, "caller-token")
    assert request.state.oidc_subject == "subject-1"


@pytest.mark.asyncio
async def test_oidc_metrics_scope_denies_without_configured_claim(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Deny metrics access for a valid OIDC token when no policy is configured."""
    _mock_provider(monkeypatch, jwk)
    monkeypatch.setattr(oidc.config.dev_config, "disable_auth", False)
    dependency = oidc.AuthDependency(_config(), virtual_path=oidc.METRICS_ACCESS_PATH)
    request = Request(
        {
            "type": "http",
            "headers": [
                (b"authorization", f"Bearer {_token(rsa_keypair[0])}".encode())
            ],
        }
    )

    with pytest.raises(HTTPException) as error:
        await dependency(request)

    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_oidc_metrics_scope_requires_matching_claim_but_rest_does_not(
    monkeypatch: pytest.MonkeyPatch,
    rsa_keypair: tuple[Any, Any],
    jwk: dict[str, Any],
) -> None:
    """Grant metrics only for the configured claim while preserving normal REST auth."""
    _mock_provider(monkeypatch, jwk)
    monkeypatch.setattr(oidc.config.dev_config, "disable_auth", False)
    settings = _config(
        oidc_metrics_access_claim="groups",
        oidc_metrics_access_value="metrics-readers",
    )
    ordinary_token = _token(rsa_keypair[0])
    metrics_dependency = oidc.AuthDependency(
        settings, virtual_path=oidc.METRICS_ACCESS_PATH
    )
    request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {ordinary_token}".encode())],
        }
    )

    with pytest.raises(HTTPException) as error:
        await metrics_dependency(request)
    assert error.value.status_code == 403

    token_with_permission = _token(
        rsa_keypair[0], claims={"groups": ["users", "metrics-readers"]}
    )
    authorized_request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {token_with_permission}".encode())],
        }
    )
    authorized_result = await metrics_dependency(authorized_request)
    assert authorized_result[1] == "alice"

    ordinary_dependency = oidc.AuthDependency(settings)
    ordinary_request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", f"Bearer {ordinary_token}".encode())],
        }
    )
    ordinary_result = await ordinary_dependency(ordinary_request)
    assert ordinary_result[1] == "alice"

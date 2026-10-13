"""Unit tests for routing A2A requests through OLS authentication modules."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from ols import config
from ols.app.endpoints import a2a_auth
from ols.app.models.config import AuthenticationConfig
from ols.constants import DEFAULT_USER_NAME, DEFAULT_USER_UID, NO_USER_TOKEN
from ols.src.auth import k8s

ISSUER = "https://keycloak.example.com/realms/rca"
INBOUND_AUDIENCE = "lightspeed-a2a"
INBOUND_AZP = "acme-agent"
CLUSTER_ID = "api-bm-cluster-e2e-bos-redhat-com-6443"
KID = "test-kid-1"
K8S_TEST_BEARER = "k8s-token"
AUDIT_TEST_BEARER = "secret-caller-token"


@pytest.fixture(scope="module")
def rsa_keypair() -> tuple[Any, Any]:
    """Generate one RSA keypair for signed test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture
def jwk(rsa_keypair: tuple[Any, Any]) -> dict[str, Any]:
    """Return the public key in the form served by an OIDC JWKS endpoint."""
    _, public_key = rsa_keypair
    raw = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key))
    raw.update({"kid": KID, "use": "sig", "alg": "RS256"})
    return raw


def make_token(
    rsa_keypair: tuple[Any, Any],
    *,
    sub: str = "spiffe://trust-domain/ns/acme/sa/acme-agent",
    aud: str = INBOUND_AUDIENCE,
    azp: str = INBOUND_AZP,
    issuer: str = ISSUER,
    expires_in: int = 300,
) -> str:
    """Build a signed test bearer token."""
    private_key, _ = rsa_keypair
    now = int(time.time())
    payload = {
        "iss": issuer,
        "aud": aud,
        "azp": azp,
        "sub": sub,
        "iat": now,
        "exp": now + expires_in,
        "preferred_username": "a2a-caller",
    }
    return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": KID})


def make_settings(**overrides: Any) -> a2a_auth.A2ASettings:
    """Construct deterministic A2A endpoint settings for tests."""
    values = {
        "cluster_id": CLUSTER_ID,
        "rpc_url": "https://praxis.example.com/a2a",
        "agent_name": "openshift-lightspeed",
        "agent_description": "test",
    }
    values.update(overrides)
    return a2a_auth.A2ASettings(**values)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=MagicMock(), response=self)


class _FakeAsyncClient:
    """Stand in for ``httpx.AsyncClient`` without making network requests."""

    def __init__(self, responder: Callable[[str], _FakeResponse], **_: Any) -> None:
        self._responder = responder

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        return self._responder(url)


def patch_oidc_provider(monkeypatch: pytest.MonkeyPatch, jwk: dict[str, Any]) -> None:
    """Patch the OIDC discovery and JWKS requests used by the auth module."""

    def responder(url: str) -> _FakeResponse:
        if url.endswith("/.well-known/openid-configuration"):
            return _FakeResponse({"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        if url == f"{ISSUER}/jwks":
            return _FakeResponse({"keys": [jwk]})
        raise AssertionError(f"Unexpected OIDC URL: {url}")

    from ols.src.auth import oidc

    monkeypatch.setattr(
        oidc.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(responder, **kwargs),
    )


def make_request(
    token: Optional[str],
    cluster_header: Optional[str],
    request_id: Optional[str] = None,
) -> Request:
    """Construct an ASGI request with the desired A2A headers."""
    headers: list[tuple[bytes, bytes]] = []
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    if cluster_header is not None:
        headers.append((b"x-ols-cluster", cluster_header.encode()))
    if request_id is not None:
        headers.append((b"x-request-id", request_id.encode()))
    return Request({"type": "http", "headers": headers})


class TestA2AAuthenticator:
    """Verify selected auth routing and the A2A cluster/token contract."""

    @pytest.mark.asyncio
    async def test_oidc_auth_mode_uses_validated_subject_and_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rsa_keypair: tuple[Any, Any],
        jwk: dict[str, Any],
    ) -> None:
        """Use the existing OIDC module, retaining the original MCP bearer token."""
        patch_oidc_provider(monkeypatch, jwk)
        config.ols_config.authentication_config = AuthenticationConfig(
            module="oidc",
            oidc_issuer_url=ISSUER,
            oidc_audience=INBOUND_AUDIENCE,
        )
        token = make_token(rsa_keypair)
        authenticator = a2a_auth.A2AAuthenticator(make_settings())

        caller = await authenticator.authenticate_caller(
            make_request(token, CLUSTER_ID, request_id="request-123")
        )

        assert caller.sub == "spiffe://trust-domain/ns/acme/sa/acme-agent"
        assert caller.token == token
        assert caller.claims["azp"] == INBOUND_AZP
        assert caller.task_scope == f"{CLUSTER_ID}:{caller.sub}"
        assert caller.request_id == "request-123"

    @pytest.mark.asyncio
    async def test_k8s_auth_mode_uses_existing_auth_dependency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Route k8s mode through the same dependency factory used by REST APIs."""
        config.ols_config.authentication_config = AuthenticationConfig(module="k8s")

        async def fake_k8s_dependency(request: Request) -> tuple[str, str, bool, str]:
            token = request.headers["authorization"].partition(" ")[2]
            return "k8s-user-uid", "alice", False, token

        async def fake_k8s_call(
            _self: k8s.AuthDependency, request: Request
        ) -> tuple[str, str, bool, str]:
            return await fake_k8s_dependency(request)

        monkeypatch.setattr(k8s.AuthDependency, "__call__", fake_k8s_call)
        authenticator = a2a_auth.A2AAuthenticator(make_settings())
        caller = await authenticator.authenticate_caller(
            make_request(K8S_TEST_BEARER, CLUSTER_ID)
        )

        assert isinstance(authenticator._auth_dependency, k8s.AuthDependency)
        assert caller.sub == "k8s-user-uid"
        assert caller.token == K8S_TEST_BEARER
        assert caller.task_scope == f"{CLUSTER_ID}:k8s-user-uid"

    @pytest.mark.asyncio
    async def test_k8s_auth_dependency_does_not_block_event_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run blocking k8s auth in a worker thread while the event loop stays responsive."""
        config.ols_config.authentication_config = AuthenticationConfig(module="k8s")
        release_dependency = threading.Event()
        ticker_signal = asyncio.Event()
        ticker_ran = asyncio.Event()
        loop = asyncio.get_running_loop()
        loop_thread_id = threading.get_ident()
        dependency_thread_id: int | None = None
        watchdog = threading.Timer(5, release_dependency.set)
        watchdog.daemon = True

        async def fake_k8s_call(
            _self: k8s.AuthDependency, _request: Request
        ) -> tuple[str, str, bool, str]:
            nonlocal dependency_thread_id
            dependency_thread_id = threading.get_ident()
            loop.call_soon_threadsafe(ticker_signal.set)
            watchdog.start()
            if not release_dependency.wait(timeout=10):
                raise TimeoutError("Test did not release the blocking k8s dependency")
            return "k8s-user-uid", "alice", False, K8S_TEST_BEARER

        async def event_loop_ticker() -> None:
            await ticker_signal.wait()
            ticker_ran.set()

        monkeypatch.setattr(k8s.AuthDependency, "__call__", fake_k8s_call)
        authenticator = a2a_auth.A2AAuthenticator(make_settings())
        ticker_task = asyncio.create_task(event_loop_ticker())
        auth_task = asyncio.create_task(
            authenticator.authenticate_caller(make_request(K8S_TEST_BEARER, CLUSTER_ID))
        )

        try:
            await asyncio.wait_for(ticker_ran.wait(), timeout=10)
            release_dependency.set()
            caller = await asyncio.wait_for(auth_task, timeout=10)
        finally:
            release_dependency.set()
            watchdog.cancel()
            ticker_task.cancel()
            await asyncio.gather(ticker_task, return_exceptions=True)
            if not auth_task.done():
                auth_task.cancel()
                await asyncio.gather(auth_task, return_exceptions=True)

        assert dependency_thread_id is not None
        assert dependency_thread_id != loop_thread_id
        assert caller.sub == "k8s-user-uid"
        assert caller.token == K8S_TEST_BEARER

    @pytest.mark.asyncio
    async def test_auth_disable_during_k8s_auth_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject a nil identity when auth is disabled while k8s auth is in flight."""
        config.dev_config.disable_auth = False
        config.ols_config.authentication_config = AuthenticationConfig(module="k8s")
        dependency_started = threading.Event()
        release_dependency = threading.Event()
        dependency_results: list[tuple[str, str, bool, str]] = []

        async def fake_k8s_call(
            _self: k8s.AuthDependency, _request: Request
        ) -> tuple[str, str, bool, str]:
            dependency_started.set()
            if not release_dependency.wait(timeout=10):
                raise TimeoutError("Test did not release the k8s dependency")
            if config.dev_config.disable_auth:
                result = (
                    DEFAULT_USER_UID,
                    DEFAULT_USER_NAME,
                    False,
                    NO_USER_TOKEN,
                )
            else:
                result = ("k8s-user-uid", "alice", False, K8S_TEST_BEARER)
            dependency_results.append(result)
            return result

        monkeypatch.setattr(k8s.AuthDependency, "__call__", fake_k8s_call)
        authenticator = a2a_auth.A2AAuthenticator(make_settings())
        auth_task = asyncio.create_task(
            authenticator.authenticate_caller(make_request(None, CLUSTER_ID))
        )

        try:
            assert await asyncio.to_thread(dependency_started.wait, timeout=5)
            config.dev_config.disable_auth = True
            release_dependency.set()
            with pytest.raises(
                a2a_auth.A2AConfigurationError, match="disabled authentication"
            ):
                await asyncio.wait_for(auth_task, timeout=5)
        finally:
            release_dependency.set()
            if not auth_task.done():
                auth_task.cancel()
                await asyncio.gather(auth_task, return_exceptions=True)

        assert dependency_results == [
            (DEFAULT_USER_UID, DEFAULT_USER_NAME, False, NO_USER_TOKEN)
        ]

    @pytest.mark.asyncio
    async def test_cluster_header_must_match_configured_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject a valid caller targeting a cluster other than this service's."""
        config.ols_config.authentication_config = AuthenticationConfig(module="k8s")

        async def fake_dependency(_: Request) -> tuple[str, str, bool, str]:
            return "user-uid", "alice", False, "caller-token"

        monkeypatch.setattr(
            a2a_auth, "get_auth_dependency", lambda *_args, **_kwargs: fake_dependency
        )
        authenticator = a2a_auth.A2AAuthenticator(make_settings())

        with pytest.raises(HTTPException) as error:
            await authenticator.authenticate_caller(
                make_request("token", "other-cluster")
            )

        assert error.value.status_code == 403

    def test_noop_authentication_is_not_supported_for_a2a(self) -> None:
        """Require A2A to use an authenticated production module."""
        config.ols_config.authentication_config = AuthenticationConfig(module="noop")

        with pytest.raises(a2a_auth.A2AConfigurationError, match="k8s or oidc"):
            a2a_auth.A2AAuthenticator(make_settings())

    def test_dev_auth_bypass_is_not_supported_for_a2a(self) -> None:
        """Do not let the global development bypass weaken A2A authentication."""
        config.dev_config.disable_auth = True
        config.ols_config.authentication_config = AuthenticationConfig(module="k8s")

        with pytest.raises(
            a2a_auth.A2AConfigurationError, match="disabled authentication"
        ):
            a2a_auth.A2AAuthenticator(make_settings())

    @pytest.mark.asyncio
    async def test_cached_authenticator_rejects_auth_module_change(self) -> None:
        """Fail closed if a reload changes the authentication module."""
        config.ols_config.authentication_config = AuthenticationConfig(module="k8s")
        authenticator = a2a_auth.A2AAuthenticator(make_settings())
        config.ols_config.authentication_config = AuthenticationConfig(module="oidc")

        with pytest.raises(a2a_auth.A2AConfigurationError, match="restart the service"):
            await authenticator.authenticate_caller(make_request(None, CLUSTER_ID))

    @pytest.mark.asyncio
    async def test_auth_module_change_during_authentication_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recheck the selected module after awaiting its authentication dependency."""
        config.dev_config.disable_auth = False
        config.ols_config.authentication_config = AuthenticationConfig(module="oidc")

        async def change_auth_module(_request: Request) -> tuple[str, str, bool, str]:
            config.ols_config.authentication_config = AuthenticationConfig(module="k8s")
            return "user-id", "alice", False, "caller-token"

        monkeypatch.setattr(
            a2a_auth,
            "get_auth_dependency",
            lambda *_args, **_kwargs: change_auth_module,
        )
        authenticator = a2a_auth.A2AAuthenticator(make_settings())

        with pytest.raises(a2a_auth.A2AConfigurationError, match="restart the service"):
            await authenticator.authenticate_caller(make_request("token", CLUSTER_ID))

    @pytest.mark.asyncio
    async def test_authentication_rejects_empty_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject dependency results without a caller bearer token."""
        config.dev_config.disable_auth = False
        config.ols_config.authentication_config = AuthenticationConfig(module="oidc")

        async def dependency_without_token(
            _request: Request,
        ) -> tuple[str, str, bool, str]:
            return "user-id", "alice", False, ""

        monkeypatch.setattr(
            a2a_auth,
            "get_auth_dependency",
            lambda *_args, **_kwargs: dependency_without_token,
        )
        authenticator = a2a_auth.A2AAuthenticator(make_settings())

        with pytest.raises(HTTPException) as error:
            await authenticator.authenticate_caller(make_request("token", CLUSTER_ID))

        assert error.value.status_code == 401

    def test_cached_authenticator_rejects_auth_disable_reload_and_recovers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        rsa_keypair: tuple[Any, Any],
        jwk: dict[str, Any],
    ) -> None:
        """Reject disabled auth after config reload and accept callers after recovery."""
        monkeypatch.setenv("A2A_CLUSTER_ID", CLUSTER_ID)
        monkeypatch.setenv("A2A_RPC_URL", "https://praxis.example.com/a2a")
        monkeypatch.setattr(a2a_auth, "_authenticator", None)

        config_text = Path("tests/config/auth_config.yaml").read_text(encoding="utf-8")
        config_text = config_text.replace(
            "  authentication_config:\n    k8s_cluster_api:",
            "  authentication_config:\n"
            "    module: oidc\n"
            f"    oidc_issuer_url: {ISSUER}\n"
            f"    oidc_audience: {INBOUND_AUDIENCE}\n"
            "    k8s_cluster_api:",
        )
        assert "    module: oidc" in config_text
        config_path = tmp_path / "olsconfig.yaml"

        def reload_config(disable_auth: bool) -> None:
            config_path.write_text(
                config_text.replace(
                    "  disable_auth: false",
                    f"  disable_auth: {str(disable_auth).lower()}",
                ),
                encoding="utf-8",
            )
            config.reload_from_yaml_file(str(config_path))

        reload_config(disable_auth=False)
        token = make_token(rsa_keypair)
        authenticator = a2a_auth.get_authenticator()

        from ols.app.endpoints.a2a import A2AAuthMiddleware

        patch_oidc_provider(monkeypatch, jwk)
        app = FastAPI()
        app.add_middleware(A2AAuthMiddleware)

        @app.post("/a2a")
        async def rpc(request: Request) -> dict[str, str]:
            return {"sub": request.state.caller.sub}

        client = TestClient(app)
        authenticated_headers = {
            "Authorization": f"Bearer {token}",
            a2a_auth.CLUSTER_HEADER_NAME: CLUSTER_ID,
        }

        response = client.post("/a2a", headers=authenticated_headers)
        assert response.status_code == 200
        assert response.json() == {"sub": "spiffe://trust-domain/ns/acme/sa/acme-agent"}

        reload_config(disable_auth=True)
        assert a2a_auth.get_authenticator() is authenticator
        response = client.post(
            "/a2a", headers={a2a_auth.CLUSTER_HEADER_NAME: CLUSTER_ID}
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "A2A is not configured"}

        reload_config(disable_auth=False)
        assert a2a_auth.get_authenticator() is authenticator
        response = client.post("/a2a", headers=authenticated_headers)
        assert response.status_code == 200
        assert response.json() == {"sub": "spiffe://trust-domain/ns/acme/sa/acme-agent"}

    def test_settings_require_cluster_and_public_rpc_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Require deployment-specific cluster and agent-card endpoint settings."""
        monkeypatch.delenv("A2A_CLUSTER_ID", raising=False)
        monkeypatch.delenv("A2A_RPC_URL", raising=False)

        with pytest.raises(a2a_auth.A2AConfigurationError, match="A2A_CLUSTER_ID"):
            a2a_auth.A2ASettings.from_env()


def test_audit_record_does_not_include_caller_token() -> None:
    """Record identity and request context without recording bearer credentials."""
    caller = a2a_auth.CallerIdentity(
        sub="user-1",
        claims={"azp": "trusted-client"},
        cluster_id=CLUSTER_ID,
        token=AUDIT_TEST_BEARER,
        request_id="request-1",
    )

    record = a2a_auth.audit_record(
        caller, task_id="task-1", context_id="context-1", action="query", outcome="ok"
    )

    assert "subject=user-1" in record
    assert "on_behalf_of=trusted-client" in record
    assert AUDIT_TEST_BEARER not in record

"""Unit tests for authorized endpoints handlers."""

import asyncio
import inspect
import json
import threading
import time
from typing import Any
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient

from ols import config, constants

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints import authorized  # noqa:E402
from ols.app.endpoints.authorized import is_user_authorized  # noqa:E402
from ols.app.models.config import AuthenticationConfig  # noqa:E402
from ols.app.models.models import AuthorizationResponse  # noqa:E402
from ols.src.auth import oidc  # noqa:E402
from tests.mock_classes.mock_k8s_api import (  # noqa:E402
    MockK8sResponse,
    MockK8sResponseStatus,
)


@pytest.fixture
def _disabled_auth():
    """Fixture for tests that expect disabled auth."""
    assert config.dev_config is not None
    config.dev_config.disable_auth = True


@pytest.fixture
def _enabled_auth():
    """Fixture for tests that expect enabled auth."""
    assert config.dev_config is not None
    config.dev_config.disable_auth = False


@pytest.mark.usefixtures("_disabled_auth")
@pytest.mark.asyncio
async def test_is_user_authorized_auth_disabled():
    """Test the is_user_authorized function when the authentication is disabled."""
    # the tested function returns constant right now
    request = Request(scope={"type": "http"})
    response = await is_user_authorized(request)
    assert response == AuthorizationResponse(
        user_id=constants.DEFAULT_USER_UID,
        username=constants.DEFAULT_USER_NAME,
        skip_user_id_check=False,
    )


@pytest.mark.usefixtures("_enabled_auth")
@pytest.mark.asyncio
async def test_is_user_authorized_false_no_bearer_token():
    """Test the is_user_authorized function when its missing authorization header."""
    # the tested function returns constant right now
    request = Request(scope={"type": "http", "headers": []})

    # Expect an HTTPException for invalid tokens
    with pytest.raises(HTTPException) as exc_info:
        await is_user_authorized(request)

    # Check if the correct status code is returned for unauthenticated access
    assert exc_info.value.status_code == 401


@pytest.mark.usefixtures("_enabled_auth")
@pytest.mark.asyncio
async def test_is_user_authorized_valid_token():
    """Await Kubernetes authentication and return its authorized identity."""
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer valid-token")]}
    )
    with (
        patch(
            "ols.src.auth.k8s.get_user_info",
            return_value=MockK8sResponseStatus(
                True, True, "valid-user", "valid-uid", ["ols-group"]
            ),
        ),
        patch("ols.src.auth.k8s.K8sClientSingleton.get_authz_api") as authz_api,
    ):
        authz_api.return_value.create_subject_access_review.return_value = (
            MockK8sResponse(allowed=True)
        )
        response = await is_user_authorized(request)

    assert response == AuthorizationResponse(
        user_id="valid-uid", username="valid-user", skip_user_id_check=False
    )


@pytest.mark.usefixtures("_enabled_auth")
@pytest.mark.asyncio
async def test_is_user_authorized_invalid_token():
    """Propagate authentication failures from the configured dependency."""
    request = Request(
        scope={
            "type": "http",
            "headers": [(b"authorization", b"Bearer invalid-token")],
        }
    )
    with patch("ols.src.auth.k8s.get_user_info", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            await is_user_authorized(request)

    assert exc_info.value.status_code == 403


@pytest.fixture(scope="module")
def _oidc_signing_keys() -> tuple[Any, dict[str, Any]]:
    """Create signing keys for the concurrent OIDC authorization regression test."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key))
    jwk.update({"kid": "authorized-test-key", "use": "sig", "alg": "RS256"})
    return private_key, jwk


@pytest.mark.asyncio
async def test_authorized_concurrent_oidc_requests_share_application_loop(
    monkeypatch: pytest.MonkeyPatch,
    _enabled_auth: None,
    _oidc_signing_keys: tuple[Any, dict[str, Any]],
) -> None:
    """Await OIDC authentication concurrently through the FastAPI route."""
    assert inspect.iscoroutinefunction(is_user_authorized)
    private_key, jwk = _oidc_signing_keys
    issuer = "https://authorized.example.com/realm"
    audience = "lightspeed-authorized"
    authentication_config = AuthenticationConfig(
        module="oidc", oidc_issuer_url=issuer, oidc_audience=audience
    )
    monkeypatch.setattr(
        config.ols_config, "authentication_config", authentication_config
    )
    monkeypatch.setattr(
        authorized,
        "auth_dependency",
        oidc.AuthDependency(authentication_config, virtual_path="/ols-access"),
    )
    calls: list[str] = []
    release_provider = asyncio.Event()

    class _Response:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class _ProviderClient:
        async def __aenter__(self) -> "_ProviderClient":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get(self, url: str) -> _Response:
            calls.append(url)
            if len(calls) == 1:
                asyncio.get_running_loop().call_soon(release_provider.set)
            await release_provider.wait()
            if url.endswith("/.well-known/openid-configuration"):
                return _Response({"issuer": issuer, "jwks_uri": f"{issuer}/jwks"})
            if url == f"{issuer}/jwks":
                return _Response({"keys": [jwk]})
            raise AssertionError(f"Unexpected OIDC URL: {url}")

    monkeypatch.setattr(oidc.httpx, "AsyncClient", lambda **_kwargs: _ProviderClient())
    token = jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": "authorized-subject",
            "iat": int(time.time()),
            "exp": int(time.time()) + 300,
            "preferred_username": "authorized-user",
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "authorized-test-key"},
    )
    app = FastAPI()
    app.include_router(authorized.router)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *(
                client.post("/authorized", headers={"Authorization": f"Bearer {token}"})
                for _ in range(8)
            )
        )

    assert all(response.status_code == 200 for response in responses)
    assert all(
        response.json()["username"] == "authorized-user" for response in responses
    )
    assert calls.count(f"{issuer}/.well-known/openid-configuration") == 1
    assert calls.count(f"{issuer}/jwks") == 1


@pytest.mark.asyncio
async def test_authorized_k8s_auth_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run blocking Kubernetes auth in a worker while the app loop progresses."""
    config.ols_config.authentication_config.module = "k8s"
    auth_started = threading.Event()
    release_auth = threading.Event()
    loop_progressed = threading.Event()
    observation: dict[str, bool] = {}

    async def blocking_k8s_dependency(
        _request: Request,
    ) -> tuple[str, str, bool, str]:
        auth_started.set()
        release_auth.wait(timeout=2)
        return "user-uid", "user-name", False, "caller-token"

    def observe_loop_progress() -> None:
        auth_started.wait(timeout=2)
        observation["progressed_before_release"] = loop_progressed.wait(timeout=0.5)
        release_auth.set()

    async def report_loop_progress() -> None:
        await asyncio.to_thread(auth_started.wait, 2)
        loop_progressed.set()

    monkeypatch.setattr(authorized, "auth_dependency", blocking_k8s_dependency)
    observer = threading.Thread(target=observe_loop_progress, daemon=True)
    observer.start()
    request = Request(
        {"type": "http", "headers": [(b"authorization", b"Bearer caller-token")]}
    )

    auth_task = asyncio.create_task(is_user_authorized(request))
    progress_task = asyncio.create_task(report_loop_progress())
    result = await auth_task
    await progress_task
    observer.join(timeout=2)

    assert result.user_id == "user-uid"
    assert observation["progressed_before_release"] is True

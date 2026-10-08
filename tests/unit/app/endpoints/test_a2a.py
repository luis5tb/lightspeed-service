"""Integration-style tests for the real ``a2a-sdk``-based A2A endpoint.

Keycloak and the SPIFFE Workload API are mocked (see ``test_a2a_auth.py`` for
the auth primitives tested in isolation); here the full request path through
``ols.app.endpoints.a2a.register_routes`` -- including the real
``A2AAuthMiddleware``/``a2a_auth.authenticate_caller`` -- is exercised with a
minimal FastAPI app built the same way ``ols.app.main`` builds the real one.
OLS's own query pipeline (``ols.app.endpoints.ols.generate_response``, as
called from ``ols.app.endpoints.a2a_executor``) is replaced with a stub async
generator yielding ``StreamedChunk`` objects, so no real LLM/MCP call is made.
The stub's captured arguments are asserted on to prove the *exchanged* token
(Token B), never the original caller's token (Token A), is what would reach
MCP.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Optional
from unittest.mock import AsyncMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ols import config

# ols.app.endpoints.a2a (via a2a_executor) imports ols.app.endpoints.ols,
# whose module-level auth_dependency construction (and ols.app.metrics' own)
# requires an authentication module to already be configured -- same
# requirement as tests/unit/app/endpoints/test_ols.py, same fix.
config.ols_config.authentication_config.module = "k8s"

from a2a.utils.constants import PROTOCOL_VERSION_1_0  # noqa:E402
from a2a.utils.errors import UnsupportedOperationError  # noqa:E402
from ols.app.endpoints import a2a, a2a_auth, a2a_executor, streaming_ols  # noqa:E402
from ols.app.endpoints import ols as ols_endpoint  # noqa:E402
from ols.app.models.config import ConversationCacheConfig, QuotaHandlersConfig  # noqa:E402
from ols.app.models.models import StreamChunkType, StreamedChunk, TokenCounter  # noqa:E402
from ols.constants import QueryMode  # noqa:E402
from tests.unit.app.endpoints.test_a2a_auth import (  # noqa:E402  # reuse the same fakes/fixtures
    CLUSTER_ID,
    ISSUER,
    _FakeAsyncClient,
    _FakeResponse,
    make_settings,
    make_token,
)


@pytest.fixture(scope="module")
def rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture()
def jwk(rsa_keypair):
    _, public_key = rsa_keypair
    raw = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key))
    raw["kid"] = "test-kid-1"
    raw["use"] = "sig"
    raw["alg"] = "RS256"
    return raw


@pytest.fixture()
def client(monkeypatch, jwk):
    """A TestClient for a minimal app exposing only the real A2A routes.

    Installs a real ``A2AAuthenticator`` (built from in-memory settings, not
    the environment) as the process-wide singleton, and fakes the Keycloak
    HTTP calls and SPIFFE fetch so no real network/workload identity is
    needed.
    """
    monkeypatch.setattr(
        config.ols_config,
        "conversation_cache",
        ConversationCacheConfig({"type": "memory", "memory": {}}),
    )
    monkeypatch.setattr(config, "_conversation_cache", None)
    monkeypatch.setattr(config.ols_config, "quota_handlers", QuotaHandlersConfig())
    monkeypatch.setattr(config, "_quota_limiters", None)
    monkeypatch.setattr(config.ols_config.user_data_collection, "transcripts_disabled", True)
    settings = make_settings()
    authenticator = a2a_auth.A2AAuthenticator(settings)
    monkeypatch.setattr(a2a_auth, "_authenticator", authenticator)

    def responder(url: str, data: Optional[dict]) -> _FakeResponse:
        if url.endswith("/.well-known/openid-configuration"):
            return _FakeResponse(
                {
                    "issuer": ISSUER,
                    "jwks_uri": f"{ISSUER}/protocol/openid-connect/certs",
                    "token_endpoint": f"{ISSUER}/protocol/openid-connect/token",
                }
            )
        if url.endswith("/protocol/openid-connect/certs"):
            return _FakeResponse({"keys": [jwk]})
        if url.endswith("/protocol/openid-connect/token"):
            return _FakeResponse({"access_token": "exchanged-token-b"})
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr(
        a2a_auth.httpx, "AsyncClient", lambda **_kwargs: _FakeAsyncClient(responder)
    )
    monkeypatch.setattr(
        a2a_auth.SpiffeWorkloadIdentity, "fetch", AsyncMock(return_value="jwt-svid-value")
    )

    app = FastAPI()
    a2a.register_routes(app)
    return TestClient(app)


def auth_headers(
    token: str, cluster: Optional[str] = CLUSTER_ID, version: Optional[str] = PROTOCOL_VERSION_1_0
) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if cluster is not None:
        headers[a2a_auth.CLUSTER_HEADER_NAME] = cluster
    if version is not None:
        headers["A2A-Version"] = version
    return headers


def send_message_body(text: str = "why is my pod crashlooping?", **overrides: Any) -> dict:
    body = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": "msg-1",
                "role": "ROLE_USER",
                "parts": [{"text": text}],
            }
        },
    }
    body.update(overrides)
    return body


async def _text_stream(*texts: str) -> AsyncGenerator[StreamedChunk, None]:
    """A minimal stand-in for ``generate_response(streaming=True)``'s chunks."""
    for text in texts:
        yield StreamedChunk(type=StreamChunkType.TEXT, text=text)
    yield StreamedChunk(
        type=StreamChunkType.END,
        data={"rag_chunks": [], "truncated": False, "token_counter": TokenCounter()},
    )


def _stub_generate_response(monkeypatch, *texts: str) -> dict[str, Any]:
    """Patch ``a2a_executor.generate_response`` and return a dict of captured args."""
    captured: dict[str, Any] = {}

    def fake_generate_response(
        conversation_id,
        llm_request,
        user_id,
        skip_user_id_check,
        streaming,
        user_token,
        client_headers,
        audit_ctx,
    ):
        captured["user_token"] = user_token
        captured["client_headers"] = client_headers
        captured["query"] = llm_request.query
        captured["mode"] = llm_request.mode
        captured["conversation_id"] = conversation_id
        captured["user_id"] = user_id
        captured["audit_ctx"] = audit_ctx
        return _text_stream(*texts)

    monkeypatch.setattr(a2a_executor, "generate_response", fake_generate_response)
    return captured


class TestAgentCard:
    def test_agent_card_is_public(self, client):
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        card = response.json()
        assert card["supportedInterfaces"][0]["url"] == "https://praxis.example.com/"
        assert card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
        assert card["capabilities"]["streaming"] is True
        assert {skill["id"] for skill in card["skills"]} == {"ask", "troubleshooting"}


class TestAuthenticationEnforcement:
    def test_missing_token_is_rejected(self, client):
        response = client.post(
            "/",
            json=send_message_body(),
            headers={a2a_auth.CLUSTER_HEADER_NAME: CLUSTER_ID, "A2A-Version": PROTOCOL_VERSION_1_0},
        )
        assert response.status_code == 401

    def test_invalid_token_is_rejected(self, client):
        response = client.post(
            "/",
            json=send_message_body(),
            headers=auth_headers("not-a-valid-jwt"),
        )
        assert response.status_code == 401

    def test_missing_cluster_header_is_rejected(self, rsa_keypair, client):
        token = make_token(rsa_keypair)
        response = client.post(
            "/", json=send_message_body(), headers=auth_headers(token, cluster=None)
        )
        assert response.status_code == 403

    def test_unknown_cluster_header_is_rejected(self, rsa_keypair, client):
        token = make_token(rsa_keypair)
        response = client.post(
            "/", json=send_message_body(), headers=auth_headers(token, cluster="some-other-cluster")
        )
        assert response.status_code == 403

    def test_wrong_azp_is_rejected(self, rsa_keypair, client):
        token = make_token(rsa_keypair, azp="some-untrusted-client")
        response = client.post("/", json=send_message_body(), headers=auth_headers(token))
        assert response.status_code == 401

    def test_agent_card_is_not_gated_by_auth(self, client):
        # Only "/" (POST) is gated; the discovery document is not.
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200


class TestSendMessage:
    def test_valid_call_uses_exchanged_token_not_caller_token(
        self, rsa_keypair, client, monkeypatch
    ):
        captured = _stub_generate_response(
            monkeypatch, "the pod is crashlooping because of an OOM kill"
        )

        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["contextId"] = "external-context-id"
        response = client.post("/", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        text = body["result"]["task"]["status"]["message"]["parts"][0]["text"]
        assert text == "the pod is crashlooping because of an OOM kill"
        # The streamed answer is also exposed as an artifact.
        assert body["result"]["task"]["artifacts"][0]["parts"][0]["text"] == (
            "the pod is crashlooping because of an OOM kill"
        )

        # The defining security property: MCP (via generate_response's
        # user_token) must see the *exchanged* token, never Token A.
        assert captured["user_token"] == "exchanged-token-b"
        assert captured["user_token"] != token_a
        assert captured["mode"] == QueryMode.TROUBLESHOOTING
        assert captured["conversation_id"] != "external-context-id"
        uuid.UUID(captured["conversation_id"])
        assert captured["audit_ctx"] is not None
        # And no client-supplied MCP header override is ever forwarded --
        # metadata is consumed only for the OLS mode selector.
        assert captured["client_headers"] is None

    def test_ask_mode_is_selectable_in_message_metadata(self, rsa_keypair, client, monkeypatch):
        captured = _stub_generate_response(monkeypatch, "general answer")
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["metadata"] = {"ols_mode": "ask"}

        response = client.post("/", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        assert captured["mode"] == QueryMode.ASK

    def test_unknown_mode_fails_before_query_or_token_exchange(
        self, rsa_keypair, client, monkeypatch
    ):
        monkeypatch.setattr(
            a2a_executor,
            "generate_response",
            lambda *args, **kwargs: pytest.fail("must not generate a response"),
        )
        exchange = AsyncMock(side_effect=AssertionError("must not exchange a token"))
        monkeypatch.setattr(a2a_auth.A2AAuthenticator, "exchange_for_mcp", exchange)
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["metadata"] = {"ols_mode": "unsupported"}

        response = client.post("/", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert "use 'ask' or 'troubleshooting'" in task["status"]["message"]["parts"][0]["text"]
        exchange.assert_not_awaited()

    def test_shared_stream_pipeline_stores_and_consumes_quota(
        self, rsa_keypair, client, monkeypatch
    ):
        _stub_generate_response(monkeypatch, "stored answer")
        stored: list[tuple[Any, ...]] = []
        consumed: list[tuple[Any, ...]] = []
        quota_checks: list[tuple[Any, ...]] = []

        def redact_query(_conversation_id: str, llm_request: Any) -> Any:
            llm_request.query = "redacted investigation question"
            return llm_request

        def capture_store_data(*args: Any) -> None:
            stored.append(args)
            args[10]["store transcripts"] = time.time()

        monkeypatch.setattr(ols_endpoint, "redact_query", redact_query)
        monkeypatch.setattr(
            ols_endpoint,
            "check_tokens_available",
            lambda *args: quota_checks.append(args),
        )
        monkeypatch.setattr(streaming_ols, "store_data", capture_store_data)
        monkeypatch.setattr(streaming_ols, "consume_tokens", lambda *args: consumed.append(args))
        token_a = make_token(rsa_keypair)

        response = client.post("/", json=send_message_body(), headers=auth_headers(token_a))

        assert response.status_code == 200
        assert response.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert len(stored) == 1
        assert stored[0][2].query == "redacted investigation question"
        assert stored[0][3] == "stored answer"
        assert len(consumed) == 1
        assert len(quota_checks) == 1
        assert quota_checks[0][1] == stored[0][0]

    def test_missing_message_text_fails_the_task(self, rsa_keypair, client, monkeypatch):
        monkeypatch.setattr(
            a2a_executor, "generate_response", lambda *a, **k: pytest.fail("must not be called")
        )
        token_a = make_token(rsa_keypair)
        body = send_message_body(text="")
        response = client.post("/", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert task["status"]["message"]["parts"][0]["text"] == "Message has no text content"

    def test_non_text_parts_fail_before_query(self, rsa_keypair, client, monkeypatch):
        monkeypatch.setattr(
            a2a_executor,
            "generate_response",
            lambda *args, **kwargs: pytest.fail("must not generate a response"),
        )
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["parts"] = [{"raw": "aGk="}]

        response = client.post("/", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert task["status"]["message"]["parts"][0]["text"] == "Only text content is supported"

    def test_tool_approval_stops_without_persisting_partial_turn(
        self, rsa_keypair, client, monkeypatch
    ):
        async def approval_stream():
            yield StreamedChunk(
                type=StreamChunkType.APPROVAL_REQUIRED,
                data={"tool_name": "delete_pod"},
            )
            pytest.fail("query pipeline must stop at the approval gate")

        monkeypatch.setattr(
            a2a_executor,
            "generate_response",
            lambda *args, **kwargs: approval_stream(),
        )
        monkeypatch.setattr(
            streaming_ols,
            "store_data",
            lambda *args: pytest.fail("partial response must not be stored"),
        )
        monkeypatch.setattr(
            streaming_ols,
            "consume_tokens",
            lambda *args: pytest.fail("incomplete response must not consume quota"),
        )
        token_a = make_token(rsa_keypair)

        response = client.post("/", json=send_message_body(), headers=auth_headers(token_a))

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert "interactive tool approval" in task["status"]["message"]["parts"][0]["text"]

    def test_get_task_returns_previously_completed_task(self, rsa_keypair, client, monkeypatch):
        _stub_generate_response(monkeypatch, "answer")
        token_a = make_token(rsa_keypair)
        send_response = client.post("/", json=send_message_body(), headers=auth_headers(token_a))
        task_id = send_response.json()["result"]["task"]["id"]

        get_response = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "req-2", "method": "GetTask", "params": {"id": task_id}},
            headers=auth_headers(token_a),
        )

        assert get_response.status_code == 200
        body = get_response.json()
        assert body["result"]["id"] == task_id
        assert body["result"]["status"]["state"] == "TASK_STATE_COMPLETED"

    def test_get_task_unknown_id_returns_task_not_found_error(self, rsa_keypair, client):
        token_a = make_token(rsa_keypair)
        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "req-3", "method": "GetTask", "params": {"id": "does-not-exist"}},
            headers=auth_headers(token_a),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["error"]["code"] == -32001

    def test_query_failure_becomes_failed_task_not_raw_500(self, rsa_keypair, client, monkeypatch):
        def failing_generate_response(*args, **kwargs):
            raise HTTPException(status_code=500, detail={"response": "LLM is not accessible", "cause": "boom"})

        monkeypatch.setattr(a2a_executor, "generate_response", failing_generate_response)
        token_a = make_token(rsa_keypair)

        response = client.post("/", json=send_message_body(), headers=auth_headers(token_a))

        assert response.status_code == 200  # JSON-RPC: transport succeeds, task fails
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert "LLM is not accessible" in task["status"]["message"]["parts"][0]["text"]


class TestStreaming:
    def test_send_streaming_message_streams_real_task_events(self, rsa_keypair, client, monkeypatch):
        _stub_generate_response(monkeypatch, "hello ", "world")
        token_a = make_token(rsa_keypair)
        body = send_message_body(method="SendStreamingMessage")

        with client.stream("POST", "/", json=body, headers=auth_headers(token_a)) as response:
            assert response.status_code == 200
            events = [json.loads(line[len("data: "):]) for line in response.iter_lines() if line]

        states = [
            event["result"].get("task", {}).get("status", {}).get("state")
            or event["result"].get("statusUpdate", {}).get("status", {}).get("state")
            for event in events
        ]
        assert "TASK_STATE_SUBMITTED" in states
        assert "TASK_STATE_WORKING" in states
        assert states[-1] == "TASK_STATE_COMPLETED"

        artifact_texts = [
            part["text"]
            for event in events
            for part in event["result"].get("artifactUpdate", {}).get("artifact", {}).get("parts", [])
        ]
        assert artifact_texts == ["hello ", "world"]

        final_message_text = events[-1]["result"]["statusUpdate"]["status"]["message"]["parts"][0]["text"]
        assert final_message_text == "hello world"


class TestMcpHeaderOverrideHasNoEffect:
    """The old hand-rolled dispatcher rejected a client-supplied MCP header
    override outright (-32602). The real executor has no code path that
    would ever read one (client_headers is always passed as None), so
    there's nothing to reject -- these assert the call still succeeds
    normally and generate_response still never sees a caller-supplied value.
    """

    def test_mcp_headers_http_header_is_ignored(self, rsa_keypair, client, monkeypatch):
        captured = _stub_generate_response(monkeypatch, "answer")
        token_a = make_token(rsa_keypair)
        headers = auth_headers(token_a)
        headers["MCP-Headers"] = json.dumps({"github-mcp": {"Authorization": "Bearer stolen"}})

        response = client.post("/", json=send_message_body(), headers=headers)

        assert response.status_code == 200
        assert response.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert captured["client_headers"] is None

    def test_mcp_headers_message_metadata_is_ignored(self, rsa_keypair, client, monkeypatch):
        captured = _stub_generate_response(monkeypatch, "answer")
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["metadata"] = {
            "mcpHeaders": {"github-mcp": {"Authorization": "Bearer stolen"}}
        }

        response = client.post("/", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        assert response.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        assert captured["client_headers"] is None


class TestTaskIsolation:
    def test_task_not_visible_to_a_different_caller(self, rsa_keypair, client, monkeypatch):
        _stub_generate_response(monkeypatch, "answer")
        caller_one = make_token(rsa_keypair, sub="spiffe://trust-domain/ns/acme/sa/caller-one")
        caller_two = make_token(rsa_keypair, sub="spiffe://trust-domain/ns/acme/sa/caller-two")

        send_response = client.post("/", json=send_message_body(), headers=auth_headers(caller_one))
        task_id = send_response.json()["result"]["task"]["id"]

        get_response = client.post(
            "/",
            json={"jsonrpc": "2.0", "id": "req-x", "method": "GetTask", "params": {"id": task_id}},
            headers=auth_headers(caller_two),
        )

        assert get_response.json()["error"]["code"] == -32001


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancel_reports_unsupported_rather_than_pretending_to_work(self):
        with pytest.raises(UnsupportedOperationError):
            await a2a_executor.OLSAgentExecutor().cancel(None, None)

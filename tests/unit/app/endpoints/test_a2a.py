"""Integration-style tests for the real ``a2a-sdk``-based A2A endpoint.

OIDC discovery/JWKS and the k8s auth dependency are mocked (see
``test_a2a_auth.py`` for auth routing); here the full request path through
``ols.app.endpoints.a2a.register_routes`` -- including the real
``A2AAuthMiddleware``/``a2a_auth.authenticate_caller`` -- is exercised with a
minimal FastAPI app built the same way ``ols.app.main`` builds the real one.
OLS's own query pipeline (``ols.app.endpoints.ols.generate_response``, as
called from ``ols.app.endpoints.a2a_executor``) is replaced with a stub async
generator yielding ``StreamedChunk`` objects, so no real LLM/MCP call is made.
The stub's captured arguments are asserted on to prove the selected OLS
authentication module's validated caller token reaches MCP unchanged.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Optional

import jwt
import pytest
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import (
    Message,
    Part,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
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
from ols.app.endpoints.a2a_task_store import TaskAdmissionController  # noqa:E402
from ols.app.models.config import (  # noqa:E402
    AuthenticationConfig,
    ConversationCacheConfig,
    QuotaHandlersConfig,
)
from ols.app.models.models import (  # noqa:E402
    StreamChunkType,
    StreamedChunk,
    TokenCounter,
)
from ols.constants import QueryMode  # noqa:E402
from tests.unit.app.endpoints.test_a2a_auth import (  # noqa:E402  # reuse the same fakes/fixtures
    CLUSTER_ID,
    make_settings,
    make_token,
    patch_oidc_provider,
)



@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate an RSA keypair for tests."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture()
def jwk(rsa_keypair):
    """Convert the test RSA public key to a JWKS entry."""
    _, public_key = rsa_keypair
    raw = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(public_key))
    raw["kid"] = "test-kid-1"
    raw["use"] = "sig"
    raw["alg"] = "RS256"
    return raw


@pytest.fixture()
def app_factory(monkeypatch, jwk):
    """Prepare auth fakes and return an app factory for the real A2A routes.

    Installs a real ``A2AAuthenticator`` (built from in-memory settings, not
    the environment) as the process-wide singleton, and fakes OIDC discovery
    and JWKS calls so no real identity provider is needed.
    """
    monkeypatch.setattr(
        config.ols_config,
        "conversation_cache",
        ConversationCacheConfig({"type": "memory", "memory": {}}),
    )
    monkeypatch.setattr(config, "_conversation_cache", None)
    monkeypatch.setattr(config.ols_config, "quota_handlers", QuotaHandlersConfig())
    monkeypatch.setattr(config, "_quota_limiters", None)
    monkeypatch.setattr(
        config.ols_config.user_data_collection, "transcripts_disabled", True
    )
    config.ols_config.authentication_config = AuthenticationConfig(
        module="oidc",
        oidc_issuer_url="https://keycloak.example.com/realms/rca",
        oidc_audience="lightspeed-a2a",
    )
    settings = make_settings()
    authenticator = a2a_auth.A2AAuthenticator(settings)
    monkeypatch.setattr(a2a_auth, "_authenticator", authenticator)

    patch_oidc_provider(monkeypatch, jwk)
    a2a_executor.clear_context_conversations()

    def create_app(
        admission_controller: TaskAdmissionController | None = None,
    ) -> FastAPI:
        app = FastAPI()
        a2a.register_routes(app, admission_controller=admission_controller)
        return app

    return create_app


@pytest.fixture()
def client(app_factory):
    """Build a TestClient for the real A2A routes."""
    return TestClient(app_factory())


def auth_headers(
    token: str,
    cluster: Optional[str] = CLUSTER_ID,
    version: Optional[str] = PROTOCOL_VERSION_1_0,
) -> dict[str, str]:
    """Build HTTP authentication headers for a test request."""
    headers = {"Authorization": f"Bearer {token}"}
    if cluster is not None:
        headers[a2a_auth.CLUSTER_HEADER_NAME] = cluster
    if version is not None:
        headers["A2A-Version"] = version
    return headers


def send_message_body(
    text: str = "why is my pod crashlooping?", **overrides: Any
) -> dict:
    """Build a JSON-RPC SendMessage request body."""
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
    """Yield sample text and end chunks for the response wrapper."""
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
        captured["generate_thread_id"] = threading.get_ident()
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


def _stub_blocking_generate_response(monkeypatch):
    """Block the first query until released and record executor invocations."""
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def fake_generate_response(
        *_args: Any, **_kwargs: Any
    ) -> AsyncGenerator[StreamedChunk, None]:
        calls.append(threading.get_ident())

        async def stream() -> AsyncGenerator[StreamedChunk, None]:
            started.set()
            if not await asyncio.to_thread(release.wait, 5):
                raise TimeoutError("blocked query was not released")
            yield StreamedChunk(type=StreamChunkType.TEXT, text="answer")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={
                    "rag_chunks": [],
                    "truncated": False,
                    "token_counter": TokenCounter(),
                },
            )

        return stream()

    monkeypatch.setattr(a2a_executor, "generate_response", fake_generate_response)
    return started, release, calls


class TestAgentCard:
    """Group tests for agentcard behavior."""

    def test_agent_card_is_public(self, client):
        """Test agent card is public behavior."""
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        card = response.json()
        assert card["supportedInterfaces"][0]["url"] == "https://praxis.example.com/a2a"
        assert card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
        assert card["capabilities"]["streaming"] is True
        assert {skill["id"] for skill in card["skills"]} == {"ask", "troubleshooting"}
        extensions = card["capabilities"]["extensions"]
        assert len(extensions) == 1
        ext = extensions[0]
        assert ext["uri"] == a2a_executor.QUERY_MODE_EXTENSION_URI
        # protobuf JSON omits default false bools
        assert ext.get("required", False) is False
        assert ext["params"]["modes"] == ["ask", "troubleshooting"]
        assert ext["params"]["defaultMode"] == "ask"
        assert ext["params"]["metadataKey"] == "ols_mode"
        assert "single-replica" in card["description"]


class TestAuthenticationEnforcement:
    """Group tests for authenticationenforcement behavior."""

    def test_missing_token_is_rejected(self, client):
        """Test missing token is rejected behavior."""
        response = client.post(
            "/a2a",
            json=send_message_body(),
            headers={
                a2a_auth.CLUSTER_HEADER_NAME: CLUSTER_ID,
                "A2A-Version": PROTOCOL_VERSION_1_0,
            },
        )
        assert response.status_code == 401

    def test_invalid_token_is_rejected(self, client):
        """Test invalid token is rejected behavior."""
        response = client.post(
            "/a2a",
            json=send_message_body(),
            headers=auth_headers("not-a-valid-jwt"),
        )
        assert response.status_code == 401

    def test_missing_cluster_header_is_rejected(self, rsa_keypair, client):
        """Test missing cluster header is rejected behavior."""
        token = make_token(rsa_keypair)
        response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token, cluster=None)
        )
        assert response.status_code == 403

    def test_unknown_cluster_header_is_rejected(self, rsa_keypair, client):
        """Test unknown cluster header is rejected behavior."""
        token = make_token(rsa_keypair)
        response = client.post(
            "/a2a",
            json=send_message_body(),
            headers=auth_headers(token, cluster="some-other-cluster"),
        )
        assert response.status_code == 403

    def test_wrong_audience_is_rejected(self, rsa_keypair, client):
        """Test wrong audience is rejected behavior."""
        token = make_token(rsa_keypair, aud="some-other-service")
        response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token)
        )
        assert response.status_code == 401

    def test_agent_card_is_not_gated_by_auth(self, client):
        # Only "/a2a" (POST) is gated; the discovery document is not.
        """Test agent card is not gated by auth behavior."""
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200


class TestSendMessage:
    """Group tests for sendmessage behavior."""

    def test_valid_call_passes_caller_token_to_mcp(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test valid call passes caller token to mcp behavior."""
        captured = _stub_generate_response(
            monkeypatch, "the pod is crashlooping because of an OOM kill"
        )
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["contextId"] = "external-context-id"
        response = client.post("/a2a", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
        text = body["result"]["task"]["status"]["message"]["parts"][0]["text"]
        assert text == "the pod is crashlooping because of an OOM kill"
        # The streamed answer is also exposed as an artifact.
        assert body["result"]["task"]["artifacts"][0]["parts"][0]["text"] == (
            "the pod is crashlooping because of an OOM kill"
        )

        assert captured["user_token"] == token_a
        assert captured["mode"] == QueryMode.ASK
        assert captured["conversation_id"] != "external-context-id"
        uuid.UUID(captured["conversation_id"])
        assert captured["audit_ctx"] is not None
        # And no client-supplied MCP header override is ever forwarded --
        # metadata is consumed only for the OLS mode selector.
        assert captured["client_headers"] is None

    def test_default_mode_is_ask(self, rsa_keypair, client, monkeypatch):
        """Test default mode is ask behavior."""
        captured = _stub_generate_response(monkeypatch, "general answer")
        token_a = make_token(rsa_keypair)

        response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token_a)
        )

        assert response.status_code == 200
        assert captured["mode"] == QueryMode.ASK

    def test_troubleshooting_mode_is_selectable_in_message_metadata(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test troubleshooting mode is selectable in message metadata behavior."""
        captured = _stub_generate_response(monkeypatch, "diagnostic answer")
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["metadata"] = {"ols_mode": "troubleshooting"}

        response = client.post("/a2a", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        assert captured["mode"] == QueryMode.TROUBLESHOOTING

    def test_context_id_reuses_ols_conversation(self, rsa_keypair, client, monkeypatch):
        """Test context id reuses ols conversation behavior."""
        a2a_executor.clear_context_conversations()
        captured = _stub_generate_response(monkeypatch, "turn one", "turn two")
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["contextId"] = "shared-a2a-context"

        first = client.post("/a2a", json=body, headers=auth_headers(token_a))
        first_conversation = captured["conversation_id"]
        assert first.status_code == 200
        uuid.UUID(first_conversation)

        body["id"] = "req-2"
        body["params"]["message"]["messageId"] = "msg-2"
        second = client.post("/a2a", json=body, headers=auth_headers(token_a))
        assert second.status_code == 200
        assert captured["conversation_id"] == first_conversation

    def test_context_id_isolated_per_caller(self, rsa_keypair, client, monkeypatch):
        """Test context id isolated per caller behavior."""
        a2a_executor.clear_context_conversations()
        captured = _stub_generate_response(monkeypatch, "answer")
        caller_one = make_token(
            rsa_keypair, sub="spiffe://trust-domain/ns/acme/sa/caller-one"
        )
        caller_two = make_token(
            rsa_keypair, sub="spiffe://trust-domain/ns/acme/sa/caller-two"
        )
        body = send_message_body()
        body["params"]["message"]["contextId"] = "shared-looking-context"

        client.post("/a2a", json=body, headers=auth_headers(caller_one))
        conversation_one = captured["conversation_id"]
        client.post("/a2a", json=body, headers=auth_headers(caller_two))
        conversation_two = captured["conversation_id"]
        assert conversation_one != conversation_two

    def test_unknown_mode_fails_before_query(self, rsa_keypair, client, monkeypatch):
        """Test unknown mode fails before query behavior."""
        monkeypatch.setattr(
            a2a_executor,
            "generate_response",
            lambda *args, **kwargs: pytest.fail("must not generate a response"),
        )
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["metadata"] = {"ols_mode": "unsupported"}

        response = client.post("/a2a", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert (
            "use 'ask' or 'troubleshooting'"
            in task["status"]["message"]["parts"][0]["text"]
        )

    def test_shared_stream_pipeline_stores_and_consumes_quota(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test shared stream pipeline stores and consumes quota behavior."""
        captured = _stub_generate_response(monkeypatch, "stored answer")
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
        monkeypatch.setattr(
            streaming_ols, "consume_tokens", lambda *args: consumed.append(args)
        )
        process_request = a2a_executor.process_request
        process_thread_ids: list[int] = []

        def capture_process_request(*args: Any) -> Any:
            process_thread_ids.append(threading.get_ident())
            return process_request(*args)

        monkeypatch.setattr(a2a_executor, "process_request", capture_process_request)
        token_a = make_token(rsa_keypair)

        response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token_a)
        )

        assert response.status_code == 200
        assert (
            response.json()["result"]["task"]["status"]["state"]
            == "TASK_STATE_COMPLETED"
        )
        assert len(stored) == 1
        assert stored[0][2].query == "redacted investigation question"
        assert stored[0][3] == "stored answer"
        assert len(consumed) == 1
        assert len(quota_checks) == 1
        assert quota_checks[0][1] == stored[0][0]
        assert process_thread_ids[0] != captured["generate_thread_id"]

    def test_missing_message_text_fails_the_task(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test missing message text fails the task behavior."""
        monkeypatch.setattr(
            a2a_executor,
            "generate_response",
            lambda *a, **k: pytest.fail("must not be called"),
        )
        token_a = make_token(rsa_keypair)
        body = send_message_body(text="")
        response = client.post("/a2a", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert (
            task["status"]["message"]["parts"][0]["text"]
            == "Message has no text content"
        )

    def test_non_text_parts_fail_before_query(self, rsa_keypair, client, monkeypatch):
        """Test non text parts fail before query behavior."""
        monkeypatch.setattr(
            a2a_executor,
            "generate_response",
            lambda *args, **kwargs: pytest.fail("must not generate a response"),
        )
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["parts"] = [{"raw": "aGk="}]

        response = client.post("/a2a", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert (
            task["status"]["message"]["parts"][0]["text"]
            == "Only text content is supported"
        )

    def test_tool_approval_stops_without_persisting_partial_turn(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test tool approval stops without persisting partial turn behavior."""

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

        response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token_a)
        )

        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert (
            "interactive tool approval" in task["status"]["message"]["parts"][0]["text"]
        )

    def test_get_task_returns_previously_completed_task(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test get task returns previously completed task behavior."""
        _stub_generate_response(monkeypatch, "answer")
        token_a = make_token(rsa_keypair)
        send_response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token_a)
        )
        task_id = send_response.json()["result"]["task"]["id"]

        get_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "req-2",
                "method": "GetTask",
                "params": {"id": task_id},
            },
            headers=auth_headers(token_a),
        )

        assert get_response.status_code == 200
        body = get_response.json()
        assert body["result"]["id"] == task_id
        assert body["result"]["status"]["state"] == "TASK_STATE_COMPLETED"

    def test_get_task_unknown_id_returns_task_not_found_error(
        self, rsa_keypair, client
    ):
        """Test get task unknown id returns task not found error behavior."""
        token_a = make_token(rsa_keypair)
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "req-3",
                "method": "GetTask",
                "params": {"id": "does-not-exist"},
            },
            headers=auth_headers(token_a),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["error"]["code"] == -32001

    def test_query_failure_becomes_failed_task_not_raw_500(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test query failure becomes failed task not raw 500 behavior."""

        def failing_generate_response(*args, **kwargs):
            raise HTTPException(
                status_code=500,
                detail={"response": "LLM is not accessible", "cause": "boom"},
            )

        monkeypatch.setattr(
            a2a_executor, "generate_response", failing_generate_response
        )
        token_a = make_token(rsa_keypair)

        response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token_a)
        )

        assert response.status_code == 200  # JSON-RPC: transport succeeds, task fails
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert "LLM is not accessible" in task["status"]["message"]["parts"][0]["text"]


class TestStreaming:
    """Group tests for streaming behavior."""

    def test_send_streaming_message_streams_real_task_events(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test send streaming message streams real task events behavior."""
        _stub_generate_response(monkeypatch, "hello ", "world")
        token_a = make_token(rsa_keypair)
        body = send_message_body(method="SendStreamingMessage")

        with client.stream(
            "POST", "/a2a", json=body, headers=auth_headers(token_a)
        ) as response:
            assert response.status_code == 200
            events = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line
            ]

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
            for part in event["result"]
            .get("artifactUpdate", {})
            .get("artifact", {})
            .get("parts", [])
        ]
        assert artifact_texts == ["hello ", "world"]

        final_message_text = events[-1]["result"]["statusUpdate"]["status"]["message"][
            "parts"
        ][0]["text"]
        assert final_message_text == "hello world"


class TestMcpHeaderOverrideHasNoEffect:
    """Reject MCP header overrides because the A2A executor ignores them.

    The old hand-rolled dispatcher rejected a client-supplied MCP header
    override outright (-32602). The real executor has no code path that
    would ever read one (client_headers is always passed as None), so
    there's nothing to reject -- these assert the call still succeeds
    normally and generate_response still never sees a caller-supplied value.
    """

    def test_mcp_headers_http_header_is_ignored(self, rsa_keypair, client, monkeypatch):
        """Test mcp headers http header is ignored behavior."""
        captured = _stub_generate_response(monkeypatch, "answer")
        token_a = make_token(rsa_keypair)
        headers = auth_headers(token_a)
        headers["MCP-Headers"] = json.dumps(
            {"github-mcp": {"Authorization": "Bearer stolen"}}
        )

        response = client.post("/a2a", json=send_message_body(), headers=headers)

        assert response.status_code == 200
        assert (
            response.json()["result"]["task"]["status"]["state"]
            == "TASK_STATE_COMPLETED"
        )
        assert captured["client_headers"] is None

    def test_mcp_headers_message_metadata_is_ignored(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test mcp headers message metadata is ignored behavior."""
        captured = _stub_generate_response(monkeypatch, "answer")
        token_a = make_token(rsa_keypair)
        body = send_message_body()
        body["params"]["message"]["metadata"] = {
            "mcpHeaders": {"github-mcp": {"Authorization": "Bearer stolen"}}
        }

        response = client.post("/a2a", json=body, headers=auth_headers(token_a))

        assert response.status_code == 200
        assert (
            response.json()["result"]["task"]["status"]["state"]
            == "TASK_STATE_COMPLETED"
        )
        assert captured["client_headers"] is None


class TestTaskIsolation:
    """Group tests for taskisolation behavior."""

    def test_task_not_visible_to_a_different_caller(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test task not visible to a different caller behavior."""
        _stub_generate_response(monkeypatch, "answer")
        caller_one = make_token(
            rsa_keypair, sub="spiffe://trust-domain/ns/acme/sa/caller-one"
        )
        caller_two = make_token(
            rsa_keypair, sub="spiffe://trust-domain/ns/acme/sa/caller-two"
        )

        send_response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(caller_one)
        )
        task_id = send_response.json()["result"]["task"]["id"]

        get_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "req-x",
                "method": "GetTask",
                "params": {"id": task_id},
            },
            headers=auth_headers(caller_two),
        )

        assert get_response.json()["error"]["code"] == -32001


class TestContextConcurrency:
    """Group tests for context concurrency behavior."""

    @pytest.mark.asyncio
    async def test_same_context_tasks_run_serially(self, monkeypatch):
        """Serialize turns so concurrent messages share an ordered history."""
        a2a_executor.clear_context_conversations()
        executor = a2a_executor.OLSAgentExecutor()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        started_tasks: list[str] = []

        async def fake_execute(context, event_queue):
            started_tasks.append(context.task_id)
            if context.task_id == "task-one":
                first_started.set()
                await release_first.wait()

        monkeypatch.setattr(executor, "_execute", fake_execute)
        caller = SimpleNamespace(task_scope="caller-and-cluster")

        def make_context(task_id):
            return SimpleNamespace(
                call_context=SimpleNamespace(state={"caller": caller}),
                context_id="same-context",
                task_id=task_id,
            )

        first = asyncio.create_task(executor.execute(make_context("task-one"), None))
        await first_started.wait()
        second = asyncio.create_task(executor.execute(make_context("task-two"), None))
        await asyncio.sleep(0)

        assert started_tasks == ["task-one"]

        release_first.set()
        await asyncio.gather(first, second)

        assert started_tasks == ["task-one", "task-two"]
        assert a2a_executor._context_execution_locks == {}

    def test_same_context_tasks_run_serially_across_event_loops(self, monkeypatch):
        """Serialize one context when callers use separate event loops and threads."""
        a2a_executor.clear_context_conversations()
        executor = a2a_executor.OLSAgentExecutor()
        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        started_tasks: list[str] = []
        failures: list[BaseException] = []
        caller = SimpleNamespace(task_scope="caller-and-cluster")

        async def fake_execute(context, _event_queue):
            started_tasks.append(context.task_id)
            if context.task_id == "task-one":
                first_started.set()
                if not await asyncio.to_thread(release_first.wait, 5):
                    raise TimeoutError("first task was not released")
            else:
                second_started.set()

        monkeypatch.setattr(executor, "_execute", fake_execute)

        def make_context(task_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                call_context=SimpleNamespace(state={"caller": caller}),
                context_id="same-context-across-loops",
                task_id=task_id,
            )

        def run_task(task_id: str) -> None:
            try:
                asyncio.run(executor.execute(make_context(task_id), None))
            except BaseException as error:
                failures.append(error)

        first_thread = threading.Thread(target=run_task, args=("task-one",))
        second_thread = threading.Thread(target=run_task, args=("task-two",))
        first_thread.start()
        try:
            assert first_started.wait(timeout=2)
            second_thread.start()
            assert not second_started.wait(timeout=0.05)
        finally:
            release_first.set()
            first_thread.join(timeout=5)
            if second_thread.ident is not None:
                second_thread.join(timeout=5)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert failures == []
        assert started_tasks == ["task-one", "task-two"]
        assert a2a_executor._context_execution_locks == {}

    @pytest.mark.asyncio
    async def test_cancelled_waiter_releases_context_reference(self, monkeypatch):
        """Cancel a queued turn without leaking its shared lock reference."""
        a2a_executor.clear_context_conversations()
        executor = a2a_executor.OLSAgentExecutor()
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        started_tasks: list[str] = []
        caller = SimpleNamespace(task_scope="caller-and-cluster")

        async def fake_execute(context, _event_queue):
            started_tasks.append(context.task_id)
            if context.task_id == "task-one":
                first_started.set()
                await release_first.wait()

        monkeypatch.setattr(executor, "_execute", fake_execute)

        def make_context(task_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                call_context=SimpleNamespace(state={"caller": caller}),
                context_id="cancel-waiter-context",
                task_id=task_id,
            )

        first = asyncio.create_task(executor.execute(make_context("task-one"), None))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        waiter = asyncio.create_task(executor.execute(make_context("task-two"), None))
        await asyncio.sleep(0.03)

        key = (caller.task_scope, "cancel-waiter-context")
        assert a2a_executor._context_execution_locks[key][1] == 2
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert a2a_executor._context_execution_locks[key][1] == 1
        assert started_tasks == ["task-one"]

        release_first.set()
        await first
        await executor.execute(make_context("task-three"), None)

        assert started_tasks == ["task-one", "task-three"]
        assert a2a_executor._context_execution_locks == {}

    @pytest.mark.asyncio
    async def test_cancelled_active_turn_releases_context_lock(self, monkeypatch):
        """Release the per-context lock when an active turn is canceled."""
        a2a_executor.clear_context_conversations()
        executor = a2a_executor.OLSAgentExecutor()
        started = asyncio.Event()
        never_release = asyncio.Event()
        caller = SimpleNamespace(task_scope="caller-and-cluster")

        async def fake_execute(context, _event_queue):
            if context.task_id == "active":
                started.set()
                await never_release.wait()

        monkeypatch.setattr(executor, "_execute", fake_execute)

        def make_context(task_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                call_context=SimpleNamespace(state={"caller": caller}),
                context_id="cancel-active-context",
                task_id=task_id,
            )

        active = asyncio.create_task(executor.execute(make_context("active"), None))
        await asyncio.wait_for(started.wait(), timeout=1)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active

        assert a2a_executor._context_execution_locks == {}
        await executor.execute(make_context("after-cancel"), None)
        assert a2a_executor._context_execution_locks == {}

    def test_context_lru_pins_active_and_waiting_bindings(self, monkeypatch):
        """Preserve referenced contexts and evict the least-recent idle one."""
        a2a_executor.clear_context_conversations()
        monkeypatch.setattr(a2a_executor, "_MAX_CONTEXT_CONVERSATIONS", 2)
        owner = "caller-and-cluster"
        a2a_executor._remember_conversation_id(owner, "context-a", "conversation-a")
        a2a_executor._remember_conversation_id(owner, "context-b", "conversation-b")
        assert (
            a2a_executor._lookup_conversation_id(owner, "context-a") == "conversation-a"
        )

        reference_a = a2a_executor._acquire_context_execution_lock(owner, "context-a")
        reference_b = a2a_executor._acquire_context_execution_lock(owner, "context-b")
        reference_c = a2a_executor._acquire_context_execution_lock(owner, "context-c")
        a2a_executor._remember_conversation_id(owner, "context-c", "conversation-c")

        assert set(a2a_executor._context_conversations) == {
            (owner, "context-a"),
            (owner, "context-b"),
            (owner, "context-c"),
        }

        assert reference_b is not None
        a2a_executor._release_context_execution_lock(*reference_b)
        assert set(a2a_executor._context_conversations) == {
            (owner, "context-a"),
            (owner, "context-c"),
        }
        assert reference_a is not None
        assert reference_c is not None
        a2a_executor._release_context_execution_lock(*reference_a)
        a2a_executor._release_context_execution_lock(*reference_c)

        a2a_executor._lookup_conversation_id(owner, "context-a")
        a2a_executor._remember_conversation_id(owner, "context-d", "conversation-d")
        assert set(a2a_executor._context_conversations) == {
            (owner, "context-a"),
            (owner, "context-d"),
        }

    def test_evicted_context_starts_a_new_conversation(
        self, monkeypatch, client, rsa_keypair
    ):
        """Start a new OLS conversation after an idle context binding is evicted."""
        monkeypatch.setattr(a2a_executor, "_MAX_CONTEXT_CONVERSATIONS", 1)
        captured = _stub_generate_response(monkeypatch, "answer")
        token = make_token(rsa_keypair)

        def send(context_id: str, request_id: str) -> str:
            body = send_message_body()
            body["id"] = request_id
            body["params"]["message"]["contextId"] = context_id
            response = client.post("/a2a", json=body, headers=auth_headers(token))
            assert response.status_code == 200
            return captured["conversation_id"]

        first_conversation = send("context-a", "first")
        send("context-b", "second")
        restarted_conversation = send("context-a", "third")

        assert restarted_conversation != first_conversation
        assert restarted_conversation != "context-a"


class TestAdmissionControl:
    """Verify bounded admission and terminal result retention on real routes."""

    def test_concurrent_submissions_admit_only_one_task(
        self, app_factory, monkeypatch, rsa_keypair
    ):
        """Race new submissions and return a JSON-RPC capacity error for one."""
        controller = TaskAdmissionController(max_active_tasks=1)
        app = app_factory(controller)
        clients = (TestClient(app), TestClient(app))
        for client in clients:
            client.__enter__()
        started, release, calls = _stub_blocking_generate_response(monkeypatch)
        headers = auth_headers(make_token(rsa_keypair))
        barrier = threading.Barrier(3)

        def submit(client: TestClient, request_id: str):
            body = send_message_body()
            body["id"] = request_id
            body["params"]["message"]["messageId"] = request_id
            body["params"]["configuration"] = {"returnImmediately": True}
            barrier.wait(timeout=2)
            return client.post("/a2a", json=body, headers=headers)

        try:
            clients[0].get("/.well-known/agent-card.json")
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = [
                    executor.submit(submit, clients[0], "race-one"),
                    executor.submit(submit, clients[1], "race-two"),
                ]
                barrier.wait(timeout=2)
                responses = [future.result(timeout=5) for future in results]

            assert all(response.status_code == 200 for response in responses)
            successes = [
                response for response in responses if "result" in response.json()
            ]
            errors = [
                response.json()["error"]
                for response in responses
                if "error" in response.json()
            ]
            assert len(successes) == 1
            assert len(errors) == 1
            assert errors[0]["code"] == -32603
            assert started.wait(timeout=2)
            assert len(calls) == 1
            assert controller.active_count == 1

            release.set()
            completed = _wait_for_completed_task(
                clients[0], headers, successes[0].json()["result"]["task"]["id"]
            )
            assert completed["status"]["state"] == "TASK_STATE_COMPLETED"
            assert controller.active_count == 0
        finally:
            release.set()
            for client in clients:
                client.__exit__(None, None, None)

    def test_return_immediately_keeps_admission_until_terminal_save(
        self, app_factory, monkeypatch, rsa_keypair
    ):
        """Reject excess new work but retain the slot after an early response."""
        controller = TaskAdmissionController(max_active_tasks=1)
        client = TestClient(app_factory(controller))
        client.__enter__()
        started, release, calls = _stub_blocking_generate_response(monkeypatch)
        token = make_token(rsa_keypair)
        headers = auth_headers(token)
        first_body = send_message_body()
        first_body["params"]["configuration"] = {"returnImmediately": True}

        try:
            first = client.post("/a2a", json=first_body, headers=headers)
            assert first.status_code == 200
            first_task = first.json()["result"]["task"]
            task_id = first_task["id"]
            assert started.wait(timeout=2)

            rejected = client.post("/a2a", json=send_message_body(), headers=headers)
            assert rejected.status_code == 200
            error = rejected.json()["error"]
            assert error["code"] == -32603
            assert "capacity reached" in error["message"]
            assert error["data"][0]["metadata"] == {
                "reason": "capacity",
                "retryable": True,
            }
            assert len(calls) == 1

            continuation = send_message_body()
            continuation["params"]["message"]["taskId"] = "not-an-existing-task"
            continuation_response = client.post(
                "/a2a", json=continuation, headers=headers
            )
            assert continuation_response.json()["error"]["code"] == -32001
            assert controller.active_count == 1

            release.set()
            completed = _wait_for_completed_task(client, headers, task_id)
            assert completed["status"]["state"] == "TASK_STATE_COMPLETED"
            assert controller.active_count == 0

            next_response = client.post(
                "/a2a", json=send_message_body(), headers=headers
            )
            assert (
                next_response.json()["result"]["task"]["status"]["state"]
                == "TASK_STATE_COMPLETED"
            )
        finally:
            release.set()
            client.__exit__(None, None, None)

    @pytest.mark.asyncio
    async def test_stream_handler_close_keeps_bound_lease_until_terminal_save(self):
        """Keep a background task's lease after its streaming handler closes."""
        owner = "caller-and-cluster"
        controller = TaskAdmissionController(max_active_tasks=1)
        store = a2a.BoundedInMemoryTaskStore(a2a._owner_from_context, controller)

        class _BlockingExecutor:
            """Emit a task, pause, and then mark it complete."""

            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def execute(self, context, event_queue) -> None:
                await event_queue.enqueue_event(
                    Task(
                        id=context.task_id,
                        context_id=context.context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    )
                )
                self.started.set()
                await self.release.wait()
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                    )
                )

            async def cancel(self, _context, _event_queue) -> None:
                raise UnsupportedOperationError("Cancellation is unsupported")

        executor = _BlockingExecutor()
        handler = a2a._AdmissionRequestHandler(
            agent_executor=executor,
            task_store=store,
            agent_card=a2a._static_capabilities_card(),
            admission_controller=controller,
        )
        context = ServerCallContext(state={"caller": SimpleNamespace(task_scope=owner)})
        params = SendMessageRequest(
            message=Message(
                message_id="stream-message",
                role="ROLE_USER",
                parts=[Part(text="question")],
            )
        )
        stream = handler.on_message_send_stream(params, context)

        try:
            event = await asyncio.wait_for(anext(stream), timeout=2)
            assert isinstance(event, Task)
            await asyncio.wait_for(executor.started.wait(), timeout=2)
            assert controller.active_count == 1

            await stream.aclose()
            assert controller.active_count == 1

            executor.release.set()
            deadline = asyncio.get_running_loop().time() + 2
            task = None
            while asyncio.get_running_loop().time() < deadline:
                task = await store.get(event.id, context)
                if task and task.status.state == TaskState.TASK_STATE_COMPLETED:
                    break
                await asyncio.sleep(0.01)

            assert task is not None
            assert task.status.state == TaskState.TASK_STATE_COMPLETED
            assert controller.active_count == 0
        finally:
            executor.release.set()
            await stream.aclose()
            await handler.aclose()


def _wait_for_completed_task(
    client: TestClient, headers: dict[str, str], task_id: str
) -> dict[str, Any]:
    """Wait for a task record to reach a terminal completed state."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "get-after-completion",
                "method": "GetTask",
                "params": {"id": task_id},
            },
            headers=headers,
        )
        body = response.json()
        task = body.get("result")
        if task and task["status"]["state"] == "TASK_STATE_COMPLETED":
            return task
        time.sleep(0.01)
    raise AssertionError(f"Task {task_id} did not complete before timeout")


class TestSubscribeToTask:
    """Group tests for subscribetotask behavior."""

    def test_subscribe_to_unknown_task_is_authenticated_jsonrpc_error(
        self, rsa_keypair, client
    ):
        """Test subscribe to unknown task is authenticated jsonrpc error behavior."""
        token_a = make_token(rsa_keypair)
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "sub-1",
                "method": "SubscribeToTask",
                "params": {"id": "does-not-exist"},
            },
            headers=auth_headers(token_a),
        )
        assert response.status_code == 200
        body = response.json()
        # Method is wired and auth-gated; unknown tasks surface as JSON-RPC errors
        # (not HTTP 401/404). Terminal-task subscribe is rejected by the SDK
        # (-32602); missing tasks use the task-not-found code.
        assert "error" in body
        assert body["error"]["code"] in (-32001, -32602)

    def test_subscribe_requires_auth(self, client):
        """Test subscribe requires auth behavior."""
        response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "sub-1",
                "method": "SubscribeToTask",
                "params": {"id": "any"},
            },
            headers={
                a2a_auth.CLUSTER_HEADER_NAME: CLUSTER_ID,
                "A2A-Version": PROTOCOL_VERSION_1_0,
            },
        )
        assert response.status_code == 401


class TestCancellation:
    """Group tests for cancellation behavior."""

    @pytest.mark.asyncio
    async def test_cancel_reports_unsupported_rather_than_pretending_to_work(self):
        """Test cancel reports unsupported rather than pretending to work behavior."""
        with pytest.raises(UnsupportedOperationError):
            await a2a_executor.OLSAgentExecutor().cancel(None, None)

    def test_cancel_task_jsonrpc_reports_unsupported(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test cancel task jsonrpc reports unsupported behavior."""
        _stub_generate_response(monkeypatch, "answer")
        token_a = make_token(rsa_keypair)
        send_response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token_a)
        )
        task_id = send_response.json()["result"]["task"]["id"]

        cancel_response = client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "id": "cancel-1",
                "method": "CancelTask",
                "params": {"id": task_id},
            },
            headers=auth_headers(token_a),
        )
        assert cancel_response.status_code == 200
        body = cancel_response.json()
        assert "error" in body


class TestProgressChunks:
    """Group tests for progresschunks behavior."""

    def test_tool_call_progress_is_streamed_as_working_update(
        self, rsa_keypair, client, monkeypatch
    ):
        """Test tool call progress is streamed as working update behavior."""

        async def progress_stream():
            yield StreamedChunk(
                type=StreamChunkType.TOOL_CALL,
                data={"tool_name": "get_pod"},
            )
            yield StreamedChunk(type=StreamChunkType.TEXT, text="done")
            yield StreamedChunk(
                type=StreamChunkType.END,
                data={
                    "rag_chunks": [],
                    "truncated": False,
                    "token_counter": TokenCounter(),
                },
            )

        monkeypatch.setattr(
            a2a_executor, "generate_response", lambda *a, **k: progress_stream()
        )
        token_a = make_token(rsa_keypair)
        body = send_message_body(method="SendStreamingMessage")

        with client.stream(
            "POST", "/a2a", json=body, headers=auth_headers(token_a)
        ) as response:
            assert response.status_code == 200
            events = [
                json.loads(line[len("data: ") :])
                for line in response.iter_lines()
                if line
            ]

        working_metas = [
            event["result"].get("statusUpdate", {}).get("metadata", {})
            for event in events
            if event["result"].get("statusUpdate", {}).get("status", {}).get("state")
            == "TASK_STATE_WORKING"
        ]
        assert any(meta.get("a2a_event_type") == "tool_call" for meta in working_metas)


class TestMissingEndChunk:
    """Group tests for missingendchunk behavior."""

    def test_missing_end_chunk_fails_task(self, rsa_keypair, client, monkeypatch):
        """Test missing end chunk fails task behavior."""

        async def truncated_stream():
            yield StreamedChunk(type=StreamChunkType.TEXT, text="partial")

        monkeypatch.setattr(
            a2a_executor, "generate_response", lambda *a, **k: truncated_stream()
        )
        token_a = make_token(rsa_keypair)
        response = client.post(
            "/a2a", json=send_message_body(), headers=auth_headers(token_a)
        )
        assert response.status_code == 200
        task = response.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_FAILED"
        assert (
            "without a final response" in task["status"]["message"]["parts"][0]["text"]
        )


class TestMisconfiguration:
    """Group tests for misconfiguration behavior."""

    def test_missing_a2a_settings_returns_503_on_rpc(self, client, monkeypatch):
        """Test missing a2a settings returns 503 on rpc behavior."""
        monkeypatch.setattr(a2a_auth, "_authenticator", None)

        def boom():
            raise a2a_auth.A2AConfigurationError("A2A_CLUSTER_ID must be configured")

        monkeypatch.setattr(a2a_auth, "get_authenticator", boom)
        response = client.post(
            "/a2a",
            json=send_message_body(),
            headers={
                a2a_auth.CLUSTER_HEADER_NAME: CLUSTER_ID,
                "A2A-Version": PROTOCOL_VERSION_1_0,
                "Authorization": "Bearer unused",
            },
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "A2A is not configured"

"""A2A (Agent2Agent) endpoint for delegated OpenShift investigations.

Built on ``a2a-sdk[http-server,fastapi]`` instead of a hand-rolled JSON-RPC
dispatcher. The SDK owns the wire format, the JSON-RPC method table, task
lifecycle/state machine, and SSE streaming; this module only wires three
things into it:

* ``a2a_executor.OLSAgentExecutor`` -- the business logic (see that module).
* ``A2AAuthMiddleware``/``_A2ACallContextBuilder`` below -- authentication
  through ``a2a_auth.py``.
* ``BoundedInMemoryTaskStore`` with an owner-resolver -- per-(caller, cluster)
  task isolation and a bounded terminal-result window separate from in-flight
  admission.

Exposes:

* ``GET /.well-known/agent-card.json`` -- public, unauthenticated discovery
  document.
* ``POST /a2a`` -- the JSON-RPC 2.0 endpoint, supporting ``SendMessage``,
  ``SendStreamingMessage`` (real SSE streaming of task status/artifact
  updates), ``GetTask``, and ``SubscribeToTask``. ``CancelTask`` is accepted
  but always reports unsupported (see ``a2a_executor.py``).

Every RPC call is authenticated through the configured OLS authentication
module and cluster-scoped (see ``a2a_auth.py``) before the SDK's dispatcher
runs any logic. The validated caller token is passed through to OLS's MCP
path.

Task state and idle context bindings are held in bounded, process-local stores;
older terminal tasks and context continuations may be forgotten. State does not
survive a restart or replica change, so deployments that enable A2A must run a
single replica until durable task and context storage is available.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import (
    DefaultRequestHandler,
    validate_request_params,
)
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.server.routes.common import ServerCallContextBuilder
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_1_0
from a2a.utils.errors import InternalError, UnsupportedOperationError
from fastapi import FastAPI, HTTPException
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ols.app.endpoints import a2a_auth
from ols.app.endpoints.a2a_auth import A2AConfigurationError, CallerIdentity
from ols.app.endpoints.a2a_executor import (
    OLS_MODE_METADATA_KEY,
    QUERY_MODE_EXTENSION_URI,
    OLSAgentExecutor,
)
from ols.app.endpoints.a2a_task_store import (
    ADMISSION_LEASE_STATE_KEY,
    DEFAULT_AUTH_REQUIRED_TIMEOUT_SECONDS,
    MAX_ACTIVE_A2A_TASKS,
    MAX_RETAINED_TERMINAL_A2A_TASKS,
    BoundedInMemoryTaskStore,
    TaskAdmissionController,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.events import Event
    from a2a.server.tasks import TaskStore
    from a2a.types.a2a_pb2 import Message, SendMessageRequest, Task
    from starlette.requests import Request

    from ols.app.endpoints.a2a_task_store import TaskAdmissionLease

# JSON-RPC lives under /a2a so auth middleware can match a stable prefix and
# does not collide with other root-level routes (health, metrics, authorized).
_RPC_PATH = "/a2a"

_NOT_CONFIGURED_DETAIL = "A2A is not configured"

# How long a task paused in TASK_STATE_AUTH_REQUIRED waits for the caller to
# renew its credentials before it fails and frees its admission slot.
AUTH_REQUIRED_TIMEOUT_ENV = "A2A_AUTH_REQUIRED_TIMEOUT_SECONDS"


def is_a2a_enabled() -> bool:
    """Return whether the A2A server surface should be mounted.

    ``A2A_ENABLED`` env (``true``/``false``) overrides olsconfig when set,
    so deployments can toggle the surface without rewriting YAML. When the
    env var is unset, ``a2a.enabled`` from olsconfig is used (default False).
    """
    env = os.environ.get("A2A_ENABLED", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    from ols import config

    return bool(config.a2a.enabled)


class _A2AUser(User):
    """Adapts a validated ``CallerIdentity`` to the a2a-sdk's own ``User`` interface."""

    def __init__(self, identity: CallerIdentity) -> None:
        self._identity = identity

    @property
    def is_authenticated(self) -> bool:
        """Always True: this adapter is only ever constructed for a validated caller."""
        return True

    @property
    def user_name(self) -> str:
        """The caller's validated token subject (informational only)."""
        return self._identity.sub


class A2AAuthMiddleware(BaseHTTPMiddleware):
    """Authenticates every inbound A2A JSON-RPC call before the SDK dispatcher runs.

    This has to be ASGI/Starlette middleware, not a FastAPI ``Depends()``:
    the real a2a-sdk's routes bypass FastAPI's own dependency-injection
    machinery entirely (``a2a.server.routes.fastapi_routes._A2ARoute`` uses
    Starlette's ``request_response`` directly, specifically to skip it), so
    a ``Depends()`` on one of these routes would simply never run.

    Scoped to ``_RPC_PATH`` only: the agent-card route is, and must remain,
    public, and every other route on this app has its own (``k8s``-based)
    authentication untouched by this middleware.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Validate the caller, then forward (or reject) the request."""
        path = request.url.path.rstrip("/") or "/"
        if path != _RPC_PATH or request.method != "POST":
            return await call_next(request)
        try:
            caller = await a2a_auth.get_authenticator().authenticate_caller(request)
        except HTTPException as error:
            return JSONResponse({"detail": error.detail}, status_code=error.status_code)
        except A2AConfigurationError:
            logger.error("A2A requested but required configuration is missing")
            return JSONResponse({"detail": _NOT_CONFIGURED_DETAIL}, status_code=503)
        request.state.caller = caller
        return await call_next(request)


class _A2ACallContextBuilder(ServerCallContextBuilder):
    """Builds the per-call ``ServerCallContext`` from the already-validated caller.

    Synchronous and I/O-free by design: the SDK's JSON-RPC dispatcher calls
    ``build()`` synchronously, so authentication must already have happened
    in ``A2AAuthMiddleware`` above, which runs first and stashes the result on
    ``request.state``.
    """

    def build(self, request: Request) -> ServerCallContext:
        """Wrap the validated caller into a ``ServerCallContext``."""
        caller: CallerIdentity = request.state.caller
        return ServerCallContext(
            user=_A2AUser(caller),
            state={
                "caller": caller,
                # Mirrors the SDK's own DefaultServerCallContextBuilder:
                # a2a.utils.version_validator.validate_version reads the
                # inbound A2A-Version header from here, not from the raw
                # request -- omitting this makes every real client request
                # (which does send this header) look like a legacy,
                # unversioned v0.3 call and get rejected.
                "headers": dict(request.headers),
            },
        )


class _AdmissionRequestHandler(DefaultRequestHandler):
    """Reserve capacity before the SDK creates or starts a new active task."""

    def __init__(
        self,
        agent_executor: AgentExecutor,
        task_store: TaskStore,
        agent_card: AgentCard,
        admission_controller: TaskAdmissionController,
    ) -> None:
        """Initialize the SDK handler with the shared task admission controller."""
        super().__init__(
            agent_executor=agent_executor,
            task_store=task_store,
            agent_card=agent_card,
        )
        self._admission_controller = admission_controller
        self._streaming_supported = agent_card.capabilities.streaming

    async def _expire_paused_tasks(self) -> None:
        """Fail tasks whose credential renewal window closed, freeing their slots."""
        if isinstance(self.task_store, BoundedInMemoryTaskStore):
            await self.task_store.expire_auth_required()

    def _reserve_new_task(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> TaskAdmissionLease | None:
        if params.message.task_id:
            return None
        lease = self._admission_controller.try_acquire()
        if lease is None:
            raise InternalError(
                message="A2A task capacity reached; retry later",
                data={"reason": "capacity", "retryable": True},
            )
        context.state[ADMISSION_LEASE_STATE_KEY] = lease
        return lease

    @validate_request_params
    async def on_message_send(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> Message | Task:
        """Admit a new request before delegating to the SDK handler."""
        await self._expire_paused_tasks()
        lease = self._reserve_new_task(params, context)
        try:
            return await super().on_message_send(params, context)
        finally:
            if lease is not None:
                lease.release_if_unbound()

    @validate_request_params
    async def on_message_send_stream(
        self, params: SendMessageRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event, None]:
        """Admit a streaming request before the SDK starts its active task."""
        if not self._streaming_supported:
            raise UnsupportedOperationError(
                message="Streaming is not supported by the agent"
            )
        await self._expire_paused_tasks()
        lease = self._reserve_new_task(params, context)
        stream = super().on_message_send_stream(params, context)
        try:
            async for event in stream:
                yield event
        finally:
            try:
                await stream.aclose()
            finally:
                if lease is not None:
                    lease.release_if_unbound()


def _query_mode_extension() -> AgentExtension:
    """Build the OLS query-mode AgentExtension for the public agent card."""
    params = ParseDict(
        {
            "modes": ["ask", "troubleshooting"],
            "defaultMode": "ask",
            "metadataKey": OLS_MODE_METADATA_KEY,
        },
        Struct(),
    )
    return AgentExtension(
        uri=QUERY_MODE_EXTENSION_URI,
        description="Selects the OLS query mode for a message.",
        required=False,
        params=params,
    )


def _agent_capabilities() -> AgentCapabilities:
    """Capabilities shared by the static handler card and the public card."""
    return AgentCapabilities(
        streaming=True,
        push_notifications=False,
        extensions=[_query_mode_extension()],
    )


def _static_capabilities_card() -> AgentCard:
    """Build the minimal ``AgentCard`` the SDK's request handler needs at startup.

    The handler only ever consults ``capabilities.streaming``/
    ``capabilities.push_notifications`` from this object (to gate the
    ``SendStreamingMessage``/``SubscribeToTask`` and push-notification
    methods) -- never the deployment-specific fields (name, rpc url, ...).
    Those require ``A2ASettings.from_env()``, which must stay lazy (see
    ``_build_agent_card``), so this intentionally does not call it.
    """
    return AgentCard(capabilities=_agent_capabilities())


def _build_agent_card() -> AgentCard:
    """Build the full, public A2A agent card.

    Called lazily, once per request to the discovery endpoint (see
    ``register_routes``'s ``card_modifier``), not at import/startup time: it
    depends on ``A2ASettings.from_env()``, which raises
    ``A2AConfigurationError`` if the deployment-specific
    ``A2A_CLUSTER_ID``/``A2A_RPC_URL`` env vars are unset.
    """
    settings = a2a_auth.get_authenticator().settings
    description = (
        f"{settings.agent_description} "
        f"At most {MAX_ACTIVE_A2A_TASKS} tasks may be in flight, and the most "
        f"recent {MAX_RETAINED_TERMINAL_A2A_TASKS} terminal task results are "
        "retained. Task state and idle context bindings are process-local; "
        "older terminal results and context continuations may be forgotten. "
        "Use a single-replica deployment until durable A2A storage is available."
    )
    return AgentCard(
        name=settings.agent_name,
        description=description,
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=settings.rpc_url,
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=_agent_capabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="ask",
                name="Ask OpenShift Lightspeed",
                description=(
                    "Answer general questions about OpenShift and related "
                    "Red Hat products. This is the default mode; select it with "
                    f"message metadata {OLS_MODE_METADATA_KEY}=ask."
                ),
                tags=["openshift", "ask"],
            ),
            AgentSkill(
                id="troubleshooting",
                name="Troubleshoot OpenShift issues",
                description=(
                    "Diagnose live issues in the OpenShift cluster this service "
                    "is configured for, using that cluster's own MCP tools. "
                    f"Select it with message metadata {OLS_MODE_METADATA_KEY}=troubleshooting."
                ),
                tags=["openshift", "troubleshooting", "diagnostics"],
            ),
        ],
    )


def _owner_from_context(context: ServerCallContext) -> str:
    """Scope task storage/lookup to (authenticated caller, cluster).

    ``InMemoryTaskStore``'s ``owner_resolver`` hook keys every save/get/list/
    delete by this string, so a task created by one (caller, cluster) is
    reported as not-found -- not merely forbidden -- to any other. Falls
    back to an empty-string owner (shared by nothing real) if, somehow,
    ``A2AAuthMiddleware`` did not run first; it always does for ``_RPC_PATH``.
    """
    caller = context.state.get("caller")
    return caller.task_scope if caller else ""


def _auth_required_timeout_seconds() -> float:
    """Read how long a task may wait for renewed caller credentials."""
    raw = os.environ.get(AUTH_REQUIRED_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_AUTH_REQUIRED_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        logger.error(
            "%s must be a positive number of seconds; using %s",
            AUTH_REQUIRED_TIMEOUT_ENV,
            DEFAULT_AUTH_REQUIRED_TIMEOUT_SECONDS,
        )
        return DEFAULT_AUTH_REQUIRED_TIMEOUT_SECONDS
    return value


def register_routes(
    app: FastAPI,
    admission_controller: TaskAdmissionController | None = None,
    auth_required_timeout_seconds: float | None = None,
) -> None:
    """Mount the A2A agent-card and JSON-RPC routes directly on ``app``.

    Called from ``ols/app/routers.py`` only when ``is_a2a_enabled()`` is true.
    The real a2a-sdk appends raw Starlette routes straight onto ``app.routes``
    (see ``add_a2a_routes_to_fastapi``), so it needs the actual ``FastAPI``
    app object, not an ``APIRouter``.
    """
    logger.warning(
        "A2A enabled with bounded in-memory admission and task/context retention: "
        "use a single replica until durable task and context storage is available"
    )
    app.add_middleware(A2AAuthMiddleware)

    static_card = _static_capabilities_card()
    controller = admission_controller or TaskAdmissionController()
    task_store = BoundedInMemoryTaskStore(
        owner_resolver=_owner_from_context,
        admission_controller=controller,
        auth_required_timeout_seconds=(
            auth_required_timeout_seconds or _auth_required_timeout_seconds()
        ),
    )
    request_handler = _AdmissionRequestHandler(
        agent_executor=OLSAgentExecutor(),
        task_store=task_store,
        agent_card=static_card,
        admission_controller=controller,
    )

    async def _card_modifier(_card: AgentCard) -> AgentCard:
        try:
            return _build_agent_card()
        except A2AConfigurationError as error:
            raise HTTPException(
                status_code=503, detail=_NOT_CONFIGURED_DETAIL
            ) from error

    async def _get_agent_card(_request: Request) -> JSONResponse:
        card = await _card_modifier(static_card)
        card_dict = agent_card_to_dict(card)
        if card.supported_interfaces:
            # ``url`` was the endpoint field in earlier A2A card schemas.
            # Retain it for clients that have not adopted supportedInterfaces.
            card_dict["url"] = card.supported_interfaces[0].url
        return JSONResponse(card_dict)

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=[
            Route(AGENT_CARD_WELL_KNOWN_PATH, _get_agent_card, methods=["GET"])
        ],
        jsonrpc_routes=create_jsonrpc_routes(
            request_handler, rpc_url=_RPC_PATH, context_builder=_A2ACallContextBuilder()
        ),
    )

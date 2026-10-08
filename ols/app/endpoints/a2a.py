"""A2A (Agent2Agent) endpoint for ACME's delegated OpenShift investigations.

Built on the real ``a2a-sdk`` server framework (``a2a-sdk[http-server,fastapi]``,
pinned at the same ``1.1.5`` version ACME's own pinned client resolves to --
see ``agents/acme_agent/requirements.txt``) instead of a hand-rolled JSON-RPC
dispatcher. The SDK owns the wire format, the JSON-RPC method table, task
lifecycle/state machine, and SSE streaming; this module only wires three
things into it:

* ``a2a_executor.OLSAgentExecutor`` -- the business logic (see that module).
* ``A2AAuthMiddleware``/``_A2ACallContextBuilder`` below -- authentication,
  reusing ``a2a_auth.py`` unchanged.
* ``InMemoryTaskStore`` with an owner-resolver -- per-(caller, cluster) task
  isolation, so a task created by one caller is reported as not-found (not
  merely forbidden) to any other, exactly as the previous hand-rolled
  dispatcher's bespoke ``_lookup_task`` check did.

Exposes the same two routes as before:

* ``GET /.well-known/agent-card.json`` -- public, unauthenticated discovery
  document.
* ``POST /`` -- the JSON-RPC 2.0 endpoint, now supporting the full real
  method set ACME's pinned ``a2a-sdk`` client can use: ``SendMessage``,
  ``SendStreamingMessage`` (real SSE streaming of task status/artifact
  updates -- OLS's existing token-by-token LLM streaming, not a buffered
  response relabeled), ``GetTask``, and ``SubscribeToTask``. ``CancelTask``
  is accepted but always reports unsupported (see ``a2a_executor.py``).

Every RPC call is authenticated and cluster-scoped (see ``a2a_auth.py``)
*before* the SDK's dispatcher runs any logic, and ``OLSAgentExecutor``
performs a fresh RFC 8693 token exchange per request so OLS's MCP client only
ever sees the exchanged token -- never the original caller's token.

Task/conversation state is held in this process only (``InMemoryTaskStore``);
it does not survive a restart and is not shared across replicas. A
persistent, multi-replica-safe task store is tracked as later work.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.common import ServerCallContextBuilder
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_1_0

from ols.app.endpoints import a2a_auth
from ols.app.endpoints.a2a_auth import CallerIdentity
from ols.app.endpoints.a2a_executor import OLSAgentExecutor

logger = logging.getLogger(__name__)

# The only path carrying A2A JSON-RPC traffic on this app; everything else
# (the agent card, /v1/*, /health, /metrics, ...) is untouched by
# A2AAuthMiddleware below.
_RPC_PATH = "/"


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
    a ``Depends()`` on one of these routes would simply never run. Plain
    ``app.add_middleware()`` wraps the whole ASGI app ahead of routing
    regardless, which is exactly where this app's own
    ``_RequestBodyLimitMiddleware`` and ``@app.middleware`` functions already
    operate (see ``ols/app/main.py``) -- same layer, same pattern.

    Scoped to ``_RPC_PATH`` only: the agent-card route is, and must remain,
    public, and every other route on this app has its own (``k8s``-based)
    authentication untouched by this middleware.
    """

    async def dispatch(self, request: Request, call_next):
        """Validate the caller, then forward (or reject) the request."""
        if request.url.path != _RPC_PATH or request.method != "POST":
            return await call_next(request)
        try:
            caller = await a2a_auth.get_authenticator().authenticate_caller(request)
        except HTTPException as error:
            return JSONResponse({"detail": error.detail}, status_code=error.status_code)
        request.state.caller = caller
        return await call_next(request)


class _A2ACallContextBuilder(ServerCallContextBuilder):
    """Builds the per-call ``ServerCallContext`` from the already-validated caller.

    Synchronous and I/O-free by design: the SDK's JSON-RPC dispatcher calls
    ``build()`` synchronously, so all the actual (async) Keycloak validation
    must already have happened -- in ``A2AAuthMiddleware`` above, which runs
    first and stashes the result on ``request.state``.
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


def _static_capabilities_card() -> AgentCard:
    """Build the minimal ``AgentCard`` the SDK's request handler needs at startup.

    The handler only ever consults ``capabilities.streaming``/
    ``capabilities.push_notifications`` from this object (to gate the
    ``SendStreamingMessage``/``SubscribeToTask`` and push-notification
    methods) -- never the deployment-specific fields (name, rpc url, ...).
    Those require ``A2ASettings.from_env()``, which must stay lazy (see
    ``_build_agent_card``), so this intentionally does not call it.
    """
    return AgentCard(capabilities=AgentCapabilities(streaming=True, push_notifications=False))


def _build_agent_card() -> AgentCard:
    """Build the full, public A2A agent card.

    Called lazily, once per request to the discovery endpoint (see
    ``register_routes``'s ``card_modifier``), not at import/startup time: it
    depends on ``A2ASettings.from_env()``, which raises
    ``A2AConfigurationError`` if the deployment-specific
    ``A2A_KEYCLOAK_ISSUER_URL``/``A2A_CLUSTER_ID``/``A2A_RPC_URL`` env vars
    are unset. A deployment that never configures A2A at all (``appServerPatch``
    disabled) must still be able to start OLS -- only a request actually
    hitting this endpoint should fail.

    A real ``a2a.types.a2a_pb2.AgentCard`` protobuf message, serialized by
    the SDK's own ``agent_card_to_dict`` -- no more guessing the wire shape
    by hand.
    """
    settings = a2a_auth.get_authenticator().settings
    return AgentCard(
        name=settings.agent_name,
        description=settings.agent_description,
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=settings.rpc_url,
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="ask",
                name="Ask OpenShift Lightspeed",
                description=(
                    "Answer general questions about OpenShift and related "
                    "Red Hat products. Select this mode with message metadata "
                    "ols_mode=ask."
                ),
                tags=["openshift", "ask"],
            ),
            AgentSkill(
                id="troubleshooting",
                name="Troubleshoot OpenShift issues",
                description=(
                    "Diagnose live issues in the OpenShift cluster this service "
                    "is configured for, using that cluster's own MCP tools. "
                    "This is the default mode; select it with message metadata "
                    "ols_mode=troubleshooting."
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


def register_routes(app: FastAPI) -> None:
    """Mount the A2A agent-card and JSON-RPC routes directly on ``app``.

    Called from ``ols/app/routers.py`` instead of ``app.include_router(...)``:
    the real a2a-sdk appends raw Starlette routes straight onto ``app.routes``
    (see ``add_a2a_routes_to_fastapi``), so it needs the actual ``FastAPI``
    app object, not an ``APIRouter``.
    """
    app.add_middleware(A2AAuthMiddleware)

    static_card = _static_capabilities_card()
    request_handler = DefaultRequestHandler(
        agent_executor=OLSAgentExecutor(),
        task_store=InMemoryTaskStore(owner_resolver=_owner_from_context),
        agent_card=static_card,
    )

    async def _card_modifier(_card: AgentCard) -> AgentCard:
        return _build_agent_card()

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(static_card, card_modifier=_card_modifier),
        jsonrpc_routes=create_jsonrpc_routes(
            request_handler, rpc_url=_RPC_PATH, context_builder=_A2ACallContextBuilder()
        ),
    )

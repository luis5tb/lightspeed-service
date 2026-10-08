"""REST API routers."""

from fastapi import FastAPI

from ols.app.endpoints import (
    a2a,
    authorized,
    conversations,
    feedback,
    health,
    mcp_apps,
    mcp_client_headers,
    ols,
    streaming_ols,
    tool_approvals,
)
from ols.app.metrics import metrics


def include_routers(app: FastAPI) -> None:
    """Include FastAPI routers for different endpoints.

    Args:
        app: The `FastAPI` app instance.
    """
    app.include_router(ols.router, prefix="/v1")
    app.include_router(streaming_ols.router, prefix="/v1")
    app.include_router(mcp_client_headers.router, prefix="/v1")
    app.include_router(mcp_apps.router, prefix="/v1")
    app.include_router(tool_approvals.router, prefix="/v1")
    app.include_router(feedback.router, prefix="/v1")
    app.include_router(conversations.router, prefix="/v1")
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(authorized.router)
    # A2A endpoint for the ACME delegation agent, built on the real a2a-sdk
    # server framework (see a2a.py's module docstring). It mounts routes
    # directly onto `app` (not via app.include_router()) since the SDK
    # appends raw Starlette routes to app.routes itself; it is otherwise
    # registered here in the same place/order health/metrics are, since it
    # is a distinct protocol surface with its own authentication (see
    # a2a_auth.py), not part of the versioned REST API.
    a2a.register_routes(app)

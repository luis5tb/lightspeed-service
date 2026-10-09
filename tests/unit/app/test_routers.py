"""Unit tests for routers.py."""

from unittest.mock import MagicMock

from ols import config

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints import (  # noqa:E402
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
from ols.app.metrics import metrics  # noqa:E402
from ols.app.routers import include_routers  # noqa:E402


class MockFastAPI:
    """Mock class for FastAPI."""

    def __init__(self):
        """Initialize mock class."""
        self.routers = []
        self.middlewares = []

    def include_router(self, router, prefix=None):
        """Register new router."""
        self.routers.append(router)

    def add_middleware(self, middleware_cls, *args, **kwargs):
        """Record middleware registration (used by A2A)."""
        self.middlewares.append(middleware_cls)


def test_include_routers_without_a2a(monkeypatch):
    """A2A stays unmounted when disabled (default)."""
    monkeypatch.setattr(a2a, "is_a2a_enabled", lambda: False)
    register = MagicMock()
    monkeypatch.setattr(a2a, "register_routes", register)

    app = MockFastAPI()
    include_routers(app)

    assert len(app.routers) == 10
    register.assert_not_called()
    assert authorized.router in app.routers
    assert conversations.router in app.routers
    assert feedback.router in app.routers
    assert health.router in app.routers
    assert mcp_apps.router in app.routers
    assert mcp_client_headers.router in app.routers
    assert tool_approvals.router in app.routers
    assert metrics.router in app.routers
    assert ols.router in app.routers
    assert streaming_ols.router in app.routers


def test_include_routers_with_a2a_enabled(monkeypatch):
    """A2A register_routes is invoked when enablement is on."""
    monkeypatch.setattr(a2a, "is_a2a_enabled", lambda: True)
    register = MagicMock()
    monkeypatch.setattr(a2a, "register_routes", register)

    app = MockFastAPI()
    include_routers(app)

    assert len(app.routers) == 10
    register.assert_called_once_with(app)


def test_is_a2a_enabled_respects_env_and_config(monkeypatch):
    """Env overrides olsconfig; unset env falls back to a2a.enabled."""
    monkeypatch.delenv("A2A_ENABLED", raising=False)
    monkeypatch.setattr(config.config, "a2a", type("A2A", (), {"enabled": False})())
    assert a2a.is_a2a_enabled() is False

    monkeypatch.setattr(config.config, "a2a", type("A2A", (), {"enabled": True})())
    assert a2a.is_a2a_enabled() is True

    monkeypatch.setenv("A2A_ENABLED", "false")
    assert a2a.is_a2a_enabled() is False

    monkeypatch.setenv("A2A_ENABLED", "true")
    assert a2a.is_a2a_enabled() is True

"""Authentication bridge and caller identity for the A2A endpoint."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from ols import config
from ols.src.auth.auth import get_auth_dependency

logger = logging.getLogger(__name__)

CLUSTER_HEADER_NAME = "X-OLS-Cluster"
REQUEST_ID_HEADER_NAME = "X-Request-Id"
AUDIT_LOG_PREFIX = "a2a_audit"


class A2AConfigurationError(Exception):
    """Raised when required A2A configuration or authentication mode is invalid."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class A2ASettings:
    """Environment-derived configuration for the A2A endpoint."""

    cluster_id: str
    rpc_url: str
    agent_name: str
    agent_description: str

    @classmethod
    def from_env(cls) -> "A2ASettings":
        """Build settings from required cluster and public endpoint values."""
        cluster_id = _env("A2A_CLUSTER_ID")
        if not cluster_id:
            raise A2AConfigurationError("A2A_CLUSTER_ID must be configured")
        rpc_url = _env("A2A_RPC_URL")
        if not rpc_url:
            raise A2AConfigurationError("A2A_RPC_URL must be configured")
        return cls(
            cluster_id=cluster_id,
            rpc_url=rpc_url,
            agent_name=_env("A2A_AGENT_NAME", "openshift-lightspeed"),
            agent_description=_env(
                "A2A_AGENT_DESCRIPTION",
                "OpenShift Lightspeed: investigates and answers questions about "
                "this OpenShift cluster using the cluster's own MCP tools.",
            ),
        )


@dataclass(frozen=True)
class CallerIdentity:
    """A validated caller identity, bound to the cluster it targeted."""

    sub: str
    claims: dict[str, Any]
    cluster_id: str
    token: str
    request_id: str

    @property
    def task_scope(self) -> str:
        """Scope task and conversation state to one caller and cluster."""
        return f"{self.cluster_id}:{self.sub}"


def _audit_value(value: Any) -> str:
    """Render one audit field value, quoting it only if it contains spaces."""
    text = str(value)
    if not text:
        return '""'
    return f'"{text}"' if any(char.isspace() for char in text) else text


def audit_record(
    caller: CallerIdentity,
    *,
    task_id: str,
    context_id: str,
    action: str,
    outcome: str,
) -> str:
    """Build the greppable audit line for one A2A request."""
    fields = {
        "request_id": caller.request_id,
        "actor": "ols",
        "on_behalf_of": caller.claims.get("azp", ""),
        "subject": caller.sub,
        "cluster": caller.cluster_id,
        "task": task_id,
        "context": context_id,
        "action": action,
        "outcome": outcome,
    }
    rendered = " ".join(f"{key}={_audit_value(value)}" for key, value in fields.items())
    return f"{AUDIT_LOG_PREFIX} {rendered}"


class A2AAuthenticator:
    """Authenticate A2A requests with OLS's configured authentication module."""

    def __init__(self, settings: A2ASettings) -> None:
        """Build the selected k8s or OIDC dependency once for this process."""
        self.settings = settings
        if config.dev_config.disable_auth:
            raise A2AConfigurationError("A2A does not support disabled authentication")
        auth_module = config.ols_config.authentication_config.module
        if auth_module not in {"k8s", "oidc"}:
            raise A2AConfigurationError(
                "A2A requires the k8s or oidc authentication module"
            )
        self._auth_module = auth_module
        self._auth_dependency = get_auth_dependency(
            config.ols_config, virtual_path="/ols-access"
        )

    async def authenticate_caller(self, request: Request) -> CallerIdentity:
        """Authenticate the request and enforce its configured cluster scope."""
        if config.dev_config.disable_auth:
            raise A2AConfigurationError("A2A does not support disabled authentication")
        if config.ols_config.authentication_config.module != self._auth_module:
            raise A2AConfigurationError(
                "A2A authentication module changed; restart the service to apply the change"
            )
        if self._auth_module == "k8s":
            auth_result = await run_in_threadpool(
                asyncio.run, self._auth_dependency(request)
            )
        else:
            auth_result = await self._auth_dependency(request)
        if config.dev_config.disable_auth:
            raise A2AConfigurationError("A2A does not support disabled authentication")
        if config.ols_config.authentication_config.module != self._auth_module:
            raise A2AConfigurationError(
                "A2A authentication module changed; restart the service to apply the change"
            )
        user_id, _, _, token = auth_result
        if not isinstance(token, str) or not token.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated caller has no token",
            )
        cluster_header = request.headers.get(CLUSTER_HEADER_NAME, "").strip()
        if not cluster_header or cluster_header != self.settings.cluster_id:
            logger.warning(
                "A2A request rejected: missing or unknown %s (user_id=%s)",
                CLUSTER_HEADER_NAME,
                str(user_id)[:36],
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing or unknown {CLUSTER_HEADER_NAME} header",
            )

        claims = getattr(request.state, "oidc_claims", {})
        subject = getattr(request.state, "oidc_subject", "")
        if not isinstance(subject, str) or not subject:
            subject = str(user_id)
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated caller has no subject",
            )
        request_id = (
            request.headers.get(REQUEST_ID_HEADER_NAME, "").strip() or uuid.uuid4().hex
        )
        return CallerIdentity(
            sub=subject,
            claims=claims if isinstance(claims, dict) else {},
            cluster_id=cluster_header,
            token=token,
            request_id=request_id,
        )


_authenticator: Optional[A2AAuthenticator] = None
_authenticator_lock = threading.Lock()


def get_authenticator() -> A2AAuthenticator:
    """Return the process-wide authenticator, building it on first use."""
    global _authenticator
    if _authenticator is None:
        with _authenticator_lock:
            if _authenticator is None:
                _authenticator = A2AAuthenticator(A2ASettings.from_env())
    return _authenticator

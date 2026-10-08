"""The ``AgentExecutor`` driving OLS's existing pipeline for real A2A tasks.

This is the business-logic half of the A2A endpoint (see ``a2a.py`` for the
transport/routing half, built on the real ``a2a-sdk`` server framework
instead of a hand-rolled JSON-RPC dispatcher). ``OLSAgentExecutor.execute``
is the only place this module calls into OLS's own query pipeline; it never
touches the wire format directly -- that is entirely the SDK's job.

It reuses ``ols/app/endpoints/ols.py``'s ``generate_response(streaming=True)``
async generator, the same one ``streaming_ols.py``'s REST ``/v1/streaming_query``
endpoint already uses: that generator already performs real token-by-token
LLM streaming and real tool-call/tool-result events, so wiring it into
``TaskUpdater`` gives genuine A2A task-status and artifact streaming, not a
single buffered response dressed up as a stream.

As before every exchange is scoped to one ``SendMessage``/``SendStreamingMessage``
call: ``a2a_auth.py``'s RFC 8693 exchange mints a fresh MCP-scoped token per
call, and only that token (never the caller's own) reaches
``generate_response``. ``generate_response`` and
``response_processing_wrapper`` ensure A2A requests share OLS's normal request
validation, redaction, quota, audit, and storage behavior. The ``ols_mode``
message-metadata extension selects ASK or TROUBLESHOOTING (default);
``client_headers`` is always ``None``, so A2A metadata cannot override MCP
credentials.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types.a2a_pb2 import Part, Task, TaskState, TaskStatus
from a2a.utils.errors import UnsupportedOperationError
from fastapi import HTTPException

from ols.app.endpoints import a2a_auth
from ols.app.endpoints.a2a_auth import (
    A2AExchangeError,
    A2AWorkloadIdentityError,
    CallerIdentity,
)
from ols.app.endpoints.ols import generate_response, process_request
from ols.app.endpoints.streaming_ols import response_processing_wrapper
from ols.app.models.models import LLMRequest, ProcessedRequest, StreamChunkType, StreamedChunk
from ols.constants import MEDIA_TYPE_TEXT, QueryMode

logger = logging.getLogger(__name__)

# A single, stable artifact id for the streamed answer text: every TEXT chunk
# in one task appends to this same artifact, so a client accumulates them
# into one coherent answer instead of several unrelated artifacts.
_ANSWER_ARTIFACT_ID = "answer"
_OLS_MODE_METADATA_KEY = "ols_mode"
_DEFAULT_A2A_MODE = QueryMode.TROUBLESHOOTING

# Chunk types that only carry progress telemetry, not user-visible answer
# text -- surfaced as TASK_STATE_WORKING status updates with the chunk's own
# data as metadata, rather than modeled as bespoke per-type A2A events.
_PROGRESS_CHUNK_TYPES = frozenset(
    {
        StreamChunkType.TOOL_CALL,
        StreamChunkType.TOOL_RESULT,
        StreamChunkType.REASONING,
        StreamChunkType.SKILL_SELECTED,
    }
)


@dataclass
class _A2AResponseState:
    """Track answer state while OLS chunks are adapted to A2A events."""

    response_text: str = ""
    first_text_chunk: bool = True
    failed: bool = False
    saw_end_chunk: bool = False


def _derive_user_id(caller: CallerIdentity) -> str:
    """Derive a stable, UUID-shaped user id scoped to (caller, cluster).

    OLS's conversation cache and quota limiters expect a user id; this keeps
    conversation history isolated per (authenticated caller, cluster)
    without reusing a Kubernetes-style UID that does not exist for A2A
    callers.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:ols:a2a:{caller.task_scope}"))


class OLSAgentExecutor(AgentExecutor):
    """Runs one OLS investigation query per A2A task, streaming real updates."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Answer one A2A message by running it through OLS's own pipeline.

        See the ``AgentExecutor.execute`` base docstring for the lifecycle
        contract this follows (submitted -> working -> a terminal state).
        """
        caller: CallerIdentity = context.call_context.state["caller"]
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        def _audit(outcome: str, action: str = "dispatched MCP-grounded query") -> None:
            """Emit the one per-request "OLS did X on behalf of <caller>" record."""
            logger.info(
                "%s",
                a2a_auth.audit_record(
                    caller,
                    task_id=context.task_id,
                    context_id=context.context_id,
                    action=action,
                    outcome=outcome,
                ),
            )

        # The framework requires the first event for a new task to be the
        # Task object itself; TaskUpdater only emits status/artifact *update*
        # events, which it rejects if no Task has been enqueued yet.
        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        )

        try:
            text = _validated_text_input(context)
            mode = _requested_mode(context)
        except A2AInputError as error:
            _audit("error", action="validate A2A request")
            await updater.failed(updater.new_agent_message([Part(text=str(error))]))
            return

        user_id = _derive_user_id(caller)
        try:
            llm_request = LLMRequest(query=text, conversation_id=None, mode=mode)
            # Reuse OLS's ordinary request validation/redaction/quota gate.
            # The exchanged MCP token is installed after that gate succeeds.
            processed_request = process_request(
                (user_id, caller.sub, False, ""),
                llm_request,
            )
        except HTTPException as error:
            _audit("error", action="validate OLS query")
            await updater.failed(updater.new_agent_message([Part(text=_http_error_message(error))]))
            return
        except Exception as error:
            logger.error(
                "A2A OLS request validation failed: %s", type(error).__name__
            )
            _audit("error", action="validate OLS query")
            await updater.failed(
                updater.new_agent_message(
                    [Part(text="Unable to validate the investigation request")]
                )
            )
            return

        try:
            exchanged_token = await a2a_auth.get_authenticator().exchange_for_mcp(caller)
        except (A2AWorkloadIdentityError, A2AExchangeError) as error:
            logger.error("A2A token exchange failed for cluster %s: %s", caller.cluster_id, error)
            if processed_request.audit_ctx:
                processed_request.audit_ctx.logger.request_failed(error="mcp_token_exchange_failed")
            _audit("error", action="exchange MCP credentials")
            await updater.failed(
                updater.new_agent_message(
                    [Part(text="Unable to obtain credentials for the OpenShift MCP server")]
                )
            )
            return

        processed_request.user_token = exchanged_token
        await updater.start_work()

        response_text = await self._run_query(
            caller,
            updater,
            processed_request,
            llm_request,
        )
        if response_text is None:
            _audit("error")
            return

        _audit("ok")
        await updater.complete(updater.new_agent_message([Part(text=response_text)]))

    async def _run_query(
        self,
        caller: CallerIdentity,
        updater: TaskUpdater,
        processed_request: ProcessedRequest,
        llm_request: LLMRequest,
    ) -> str | None:
        """Run the shared OLS response pipeline and return its answer text."""
        state = _A2AResponseState()
        try:
            chunks = generate_response(
                processed_request.conversation_id,
                llm_request,
                processed_request.user_id,
                processed_request.skip_user_id_check,
                True,  # streaming
                processed_request.user_token,
                None,  # client_headers -- never honor a client-supplied MCP override
                processed_request.audit_ctx,
            )
            async for _ in response_processing_wrapper(
                chunks,
                processed_request.user_id,
                processed_request.conversation_id,
                llm_request,
                processed_request.attachments,
                processed_request.query_without_attachments,
                MEDIA_TYPE_TEXT,
                processed_request.timestamps,
                processed_request.skip_user_id_check,
                processed_request.audit_ctx,
                chunk_handler=lambda chunk: self._handle_chunk(chunk, updater, state),
                error_handler=lambda message: self._handle_error(message, updater, state),
            ):
                pass
        except HTTPException as error:
            reason = _http_error_message(error)
            logger.warning(
                "A2A query failed for cluster %s (status=%s): %s",
                caller.cluster_id,
                error.status_code,
                reason,
            )
            if processed_request.audit_ctx:
                processed_request.audit_ctx.logger.request_failed(
                    error=f"http_{error.status_code}"
                )
            await updater.failed(updater.new_agent_message([Part(text=reason)]))
            return None
        except Exception:
            logger.exception("Unexpected error while processing A2A query")
            if processed_request.audit_ctx:
                processed_request.audit_ctx.logger.request_failed(error="internal_error")
            await updater.failed(
                updater.new_agent_message([Part(text="An internal error occurred")])
            )
            return None

        if state.failed:
            return None
        if not state.saw_end_chunk:
            if processed_request.audit_ctx:
                processed_request.audit_ctx.logger.request_failed(
                    error="response_missing_end_chunk"
                )
            await updater.failed(
                updater.new_agent_message(
                    [Part(text="The investigation ended without a final response")]
                )
            )
            return None
        return state.response_text

    async def _handle_chunk(
        self,
        chunk: StreamedChunk,
        updater: TaskUpdater,
        state: _A2AResponseState,
    ) -> bool:
        """Map OLS chunks to A2A events and stop at interactive approval."""
        if chunk.type == StreamChunkType.TEXT:
            if chunk.text:
                state.response_text += chunk.text
                await updater.add_artifact(
                    [Part(text=chunk.text)],
                    artifact_id=_ANSWER_ARTIFACT_ID,
                    append=not state.first_text_chunk,
                )
                state.first_text_chunk = False
        elif chunk.type == StreamChunkType.APPROVAL_REQUIRED:
            state.failed = True
            await updater.failed(
                updater.new_agent_message(
                    [
                        Part(
                            text=(
                                "This investigation requires interactive tool approval, "
                                "which the A2A endpoint does not support. Disable tool "
                                "approval for A2A-driven investigations, or approve the "
                                "tool call via OLS's own REST API."
                            )
                        )
                    ]
                )
            )
            return False
        elif chunk.type in _PROGRESS_CHUNK_TYPES:
            await updater.update_status(
                TaskState.TASK_STATE_WORKING,
                metadata={"a2a_event_type": chunk.type.value, **chunk.data},
            )
        elif chunk.type == StreamChunkType.END:
            state.saw_end_chunk = True
        return True

    @staticmethod
    async def _handle_error(
        message: str,
        updater: TaskUpdater,
        state: _A2AResponseState,
    ) -> None:
        """Turn an OLS pipeline error into a failed A2A task."""
        state.failed = True
        await updater.failed(
            updater.new_agent_message(
                [Part(text=message or "The request could not be processed")]
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Report that mid-run cancellation is not supported.

        OLS's tool-calling loop has no cancellation hook to interrupt once
        started; claiming to support ``CancelTask`` without one would leave a
        client believing a running investigation actually stopped.
        """
        raise UnsupportedOperationError(
            message="OLS's investigation pipeline cannot be canceled once started"
        )


class A2AInputError(ValueError):
    """Raised when an A2A message cannot be mapped to an OLS text query."""


def _validated_text_input(context: RequestContext) -> str:
    """Require a non-empty message containing only text parts."""
    message = context.message
    if not message.parts:
        raise A2AInputError("Message has no text content")
    if any(part.WhichOneof("content") != "text" for part in message.parts):
        raise A2AInputError("Only text content is supported")
    text = context.get_user_input()
    if not text or not text.strip():
        raise A2AInputError("Message has no text content")
    return text


def _requested_mode(context: RequestContext) -> QueryMode:
    """Read the optional OLS mode extension from A2A message metadata."""
    metadata = context.message.metadata
    fields = getattr(metadata, "fields", None)
    if fields is not None:
        mode_value = fields.get(_OLS_MODE_METADATA_KEY) if fields is not None else None
        if mode_value is None:
            raw_mode = None
        elif mode_value.WhichOneof("kind") != "string_value":
            raise A2AInputError("OLS query mode must be a string")
        else:
            raw_mode = mode_value.string_value
    elif isinstance(metadata, Mapping):
        raw_mode = metadata.get(_OLS_MODE_METADATA_KEY)
    else:
        raw_mode = None

    if raw_mode is None:
        return _DEFAULT_A2A_MODE
    try:
        return QueryMode(raw_mode)
    except (TypeError, ValueError) as error:
        raise A2AInputError("Unsupported OLS query mode; use 'ask' or 'troubleshooting'") from error


def _http_error_message(error: HTTPException) -> str:
    """Return the OLS-safe response text without exposing internal causes."""
    detail = error.detail
    if isinstance(detail, dict):
        return str(detail.get("response", "The request could not be processed"))
    return str(detail)

"""Bounded process-local task admission and retention for the A2A endpoint."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types.a2a_pb2 import (
    ListTasksRequest,
    ListTasksResponse,
    Task,
    TaskState,
)
from a2a.utils.errors import InternalError

if TYPE_CHECKING:
    from _thread import LockType

    from a2a.server.owner_resolver import OwnerResolver

ADMISSION_LEASE_STATE_KEY = "_ols_a2a_admission_lease"
MAX_ACTIVE_A2A_TASKS = 1024
MAX_RETAINED_TERMINAL_A2A_TASKS = 1024

_TERMINAL_TASK_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)


@dataclass(frozen=True)
class _TaskOwner:
    task_scope: str


def _owner_context(owner: str) -> ServerCallContext:
    return ServerCallContext(state={"caller": _TaskOwner(task_scope=owner)})


class TaskAdmissionLease:
    """Represent one in-flight slot from reservation through terminal save."""

    def __init__(self, controller: TaskAdmissionController) -> None:
        """Initialize an unbound reservation from its controller."""
        self._controller = controller
        self._task_key: tuple[str, str] | None = None
        self._released = False

    def bind(self, task_key: tuple[str, str]) -> bool:
        """Bind the reservation to its first owner-scoped task record."""
        return self._controller._bind(self, task_key)

    def belongs_to(self, controller: TaskAdmissionController) -> bool:
        """Return whether this lease was issued by the given controller."""
        return self._controller is controller

    def release_if_unbound(self) -> None:
        """Release a request reservation that never became a stored task."""
        self._controller._release(self, only_if_unbound=True)

    def release(self) -> None:
        """Release this reservation at terminal completion or save failure."""
        self._controller._release(self)


class TaskAdmissionController:
    """Atomically limit admitted in-flight A2A tasks across event loops."""

    def __init__(self, max_active_tasks: int = MAX_ACTIVE_A2A_TASKS) -> None:
        """Initialize the global in-flight task limit."""
        if max_active_tasks < 1:
            raise ValueError("max_active_tasks must be positive")
        self._max_active_tasks = max_active_tasks
        self._active_count = 0
        self._lock: LockType = threading.Lock()

    @property
    def active_count(self) -> int:
        """Return the number of reserved or running task slots."""
        with self._lock:
            return self._active_count

    def try_acquire(self) -> TaskAdmissionLease | None:
        """Reserve a slot or return ``None`` when the limit is reached."""
        with self._lock:
            if self._active_count >= self._max_active_tasks:
                return None
            self._active_count += 1
            return TaskAdmissionLease(self)

    def _bind(self, lease: TaskAdmissionLease, task_key: tuple[str, str]) -> bool:
        with self._lock:
            if lease._released:
                return False
            if lease._task_key is None:
                lease._task_key = task_key
            return lease._task_key == task_key

    def _release(
        self, lease: TaskAdmissionLease, only_if_unbound: bool = False
    ) -> None:
        with self._lock:
            if lease._released or (only_if_unbound and lease._task_key is not None):
                return
            lease._released = True
            self._active_count -= 1


class BoundedInMemoryTaskStore(TaskStore):
    """Bound active tasks and separately retain the latest terminal results.

    Admission is reserved by the request handler before SDK task creation.
    Active task records hold their lease until a terminal state is saved. The
    terminal-result window is independent of active load, so a completed task
    is not evicted merely because other tasks are running. Eviction metadata
    contains only owner-scoped IDs, states, and leases, never task copies or
    caller tokens.
    """

    def __init__(
        self,
        owner_resolver: OwnerResolver,
        admission_controller: TaskAdmissionController,
        max_terminal_tasks: int = MAX_RETAINED_TERMINAL_A2A_TASKS,
    ) -> None:
        """Initialize the owner-scoped store and terminal-result window."""
        if max_terminal_tasks < 1:
            raise ValueError("max_terminal_tasks must be positive")
        self._owner_resolver = owner_resolver
        self._admission_controller = admission_controller
        self._max_terminal_tasks = max_terminal_tasks
        self._store = InMemoryTaskStore(owner_resolver=owner_resolver)
        self._active_tasks: dict[tuple[str, str], TaskAdmissionLease] = {}
        self._terminal_tasks: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._metadata_lock: LockType = threading.Lock()

    def _prepare_task_save(
        self,
        key: tuple[str, str],
        context: ServerCallContext,
        is_terminal: bool,
    ) -> TaskAdmissionLease | None:
        """Register a new task lease before saving its first task state."""
        with self._metadata_lock:
            known_active = self._active_tasks.get(key)
            known_terminal = key in self._terminal_tasks
            if known_terminal and not is_terminal:
                raise InternalError(message="A terminal A2A task cannot become active")
            if known_active is not None or known_terminal:
                return None

            candidate = context.state.get(ADMISSION_LEASE_STATE_KEY)
            if (
                not isinstance(candidate, TaskAdmissionLease)
                or not candidate.belongs_to(self._admission_controller)
                or not candidate.bind(key)
            ):
                raise InternalError(
                    message="A2A task has no valid admission reservation"
                )
            self._active_tasks[key] = candidate
            return candidate

    def _rollback_new_task(
        self, key: tuple[str, str], lease: TaskAdmissionLease
    ) -> None:
        """Remove and release a new lease when its initial save fails."""
        with self._metadata_lock:
            if self._active_tasks.get(key) is lease:
                del self._active_tasks[key]
        lease.release()

    def _record_terminal_task(
        self, key: tuple[str, str]
    ) -> tuple[TaskAdmissionLease | None, list[tuple[str, str]]]:
        """Release an active lease and trim the completion-order result window."""
        evictions: list[tuple[str, str]] = []
        with self._metadata_lock:
            completed_lease = self._active_tasks.pop(key, None)
            self._terminal_tasks[key] = None
            self._terminal_tasks.move_to_end(key)
            while len(self._terminal_tasks) > self._max_terminal_tasks:
                evictions.append(self._terminal_tasks.popitem(last=False)[0])
        return completed_lease, evictions

    async def save(self, task: Task, context: ServerCallContext) -> None:
        """Save a task, release terminal slots, and evict old terminal results."""
        owner = self._owner_resolver(context)
        key = (owner, task.id)
        is_terminal = task.status.state in _TERMINAL_TASK_STATES
        new_lease = self._prepare_task_save(key, context, is_terminal)

        try:
            await self._store.save(task, context)
        except BaseException:
            if new_lease is not None:
                self._rollback_new_task(key, new_lease)
            raise

        if not is_terminal:
            return

        completed_lease, evictions = self._record_terminal_task(key)
        if completed_lease is not None:
            completed_lease.release()
        for task_owner, task_id in evictions:
            await self._store.delete(task_id, _owner_context(task_owner))

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        """Return a task only within its authenticated owner scope."""
        return await self._store.get(task_id, context)

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext,
    ) -> ListTasksResponse:
        """List tasks using the SDK store's owner scoping and pagination."""
        return await self._store.list(params, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        """Delete a task while retaining any active execution reservation."""
        owner = self._owner_resolver(context)
        await self._store.delete(task_id, context)
        with self._metadata_lock:
            self._terminal_tasks.pop((owner, task_id), None)

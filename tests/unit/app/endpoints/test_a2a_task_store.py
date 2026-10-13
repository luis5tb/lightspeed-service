"""Tests for bounded A2A admission and task retention."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import ListTasksRequest, Task, TaskState, TaskStatus

from ols.app.endpoints.a2a_task_store import (
    ADMISSION_LEASE_STATE_KEY,
    BoundedInMemoryTaskStore,
    TaskAdmissionController,
)


def _owner_from_context(context: ServerCallContext) -> str:
    caller = context.state.get("caller")
    return caller.task_scope if caller else ""


def _context(owner: str, lease=None) -> ServerCallContext:
    state = {"caller": SimpleNamespace(task_scope=owner)}
    if lease is not None:
        state[ADMISSION_LEASE_STATE_KEY] = lease
    return ServerCallContext(state=state)


def _task(task_id: str, state: TaskState) -> Task:
    return Task(
        id=task_id,
        context_id=f"context-{task_id}",
        status=TaskStatus(state=state),
    )


async def _save_new_task(
    store: BoundedInMemoryTaskStore,
    controller: TaskAdmissionController,
    owner: str,
    task_id: str,
) -> None:
    lease = controller.try_acquire()
    assert lease is not None
    await store.save(
        _task(task_id, TaskState.TASK_STATE_SUBMITTED), _context(owner, lease)
    )


@pytest.mark.asyncio
async def test_terminal_results_survive_active_task_pressure() -> None:
    """Keep a completed result when later active tasks fill the admission limit."""
    controller = TaskAdmissionController(max_active_tasks=2)
    store = BoundedInMemoryTaskStore(
        _owner_from_context, controller, max_terminal_tasks=2
    )
    owner_one = _context("owner-one")
    finished_lease = controller.try_acquire()
    assert finished_lease is not None

    await store.save(
        _task("finished", TaskState.TASK_STATE_SUBMITTED),
        _context("owner-one", finished_lease),
    )
    await store.save(_task("finished", TaskState.TASK_STATE_COMPLETED), owner_one)
    assert controller.active_count == 0

    await _save_new_task(store, controller, "owner-one", "active-one")
    await _save_new_task(store, controller, "owner-two", "active-two")

    assert controller.active_count == 2
    assert await store.get("finished", owner_one) is not None
    assert await store.get("active-one", owner_one) is not None
    assert await store.get("active-two", _context("owner-two")) is not None


@pytest.mark.asyncio
async def test_terminal_window_evicts_only_oldest_result_and_preserves_owners() -> None:
    """Retain the latest N terminal tasks and delete evictions in their owner scope."""
    controller = TaskAdmissionController(max_active_tasks=1)
    store = BoundedInMemoryTaskStore(
        _owner_from_context, controller, max_terminal_tasks=2
    )
    owner_one = _context("owner-one")
    owner_two = _context("owner-two")

    for owner, task_id, state in (
        (owner_one, "shared-task", TaskState.TASK_STATE_COMPLETED),
        (owner_two, "shared-task", TaskState.TASK_STATE_FAILED),
        (owner_two, "newest-task", TaskState.TASK_STATE_CANCELED),
    ):
        lease = controller.try_acquire()
        assert lease is not None
        await store.save(
            _task(task_id, TaskState.TASK_STATE_SUBMITTED),
            _context(_owner_from_context(owner), lease),
        )
        await store.save(_task(task_id, state), owner)
        assert controller.active_count == 0

    assert await store.get("shared-task", owner_one) is None
    assert await store.get("shared-task", owner_two) is not None
    assert await store.get("newest-task", owner_two) is not None
    listed = await store.list(ListTasksRequest(), owner_two)
    assert {task.id for task in listed.tasks} == {"shared-task", "newest-task"}


@pytest.mark.asyncio
async def test_repeated_status_updates_keep_one_admission_lease() -> None:
    """Keep a single slot through status updates and release it at completion."""
    controller = TaskAdmissionController(max_active_tasks=1)
    store = BoundedInMemoryTaskStore(_owner_from_context, controller)
    owner = _context("owner-one")
    lease = controller.try_acquire()
    assert lease is not None

    await store.save(
        _task("task-one", TaskState.TASK_STATE_SUBMITTED), _context("owner-one", lease)
    )
    lease.release_if_unbound()
    assert controller.active_count == 1
    for _ in range(4):
        await store.save(_task("task-one", TaskState.TASK_STATE_WORKING), owner)
        assert controller.active_count == 1

    await store.save(_task("task-one", TaskState.TASK_STATE_COMPLETED), owner)

    assert controller.active_count == 0
    assert await store.get("task-one", owner) is not None


@pytest.mark.asyncio
async def test_store_preserves_sdk_owner_scoping_and_delete() -> None:
    """Delegate list/get/delete while isolating tasks by owner."""
    controller = TaskAdmissionController(max_active_tasks=1)
    store = BoundedInMemoryTaskStore(_owner_from_context, controller)
    owner_one = _context("owner-one")
    owner_two = _context("owner-two")
    lease = controller.try_acquire()
    assert lease is not None

    await store.save(
        _task("owned-task", TaskState.TASK_STATE_SUBMITTED),
        _context("owner-one", lease),
    )
    await store.save(_task("owned-task", TaskState.TASK_STATE_COMPLETED), owner_one)

    assert await store.get("owned-task", owner_two) is None
    listed = await store.list(ListTasksRequest(), owner_one)
    assert [task.id for task in listed.tasks] == ["owned-task"]
    await store.delete("owned-task", owner_one)
    assert await store.get("owned-task", owner_one) is None


def test_admission_limit_is_atomic_across_threads() -> None:
    """Admit at most the configured number of simultaneous contenders."""
    max_active = 4
    contender_count = 20
    controller = TaskAdmissionController(max_active_tasks=max_active)
    barrier = threading.Barrier(contender_count)

    def compete():
        barrier.wait(timeout=2)
        return controller.try_acquire()

    with ThreadPoolExecutor(max_workers=contender_count) as executor:
        leases = list(executor.map(lambda _: compete(), range(contender_count)))

    admitted = [lease for lease in leases if lease is not None]
    assert len(admitted) == max_active
    assert controller.active_count == max_active
    for lease in admitted:
        lease.release()
        lease.release()
    assert controller.active_count == 0


def test_unbound_admission_lease_releases_after_setup_failure() -> None:
    """Release a reserved slot when request setup never stores a task."""
    controller = TaskAdmissionController(max_active_tasks=1)
    lease = controller.try_acquire()

    assert lease is not None
    assert controller.active_count == 1
    lease.release_if_unbound()
    lease.release_if_unbound()
    assert controller.active_count == 0

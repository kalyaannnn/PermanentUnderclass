import asyncio
from dataclasses import replace

import pytest

from lagrl.sandbox.lifecycle import SessionManager
from lagrl.sandbox.records import (
    CreateSessionRequest,
    ExecuteRequest,
    ExecutionConflict,
    ExecutionReference,
    ExecutionResult,
    ExecutionState,
    FailureCategory,
    InvalidRequest,
    ResourceExhausted,
    ResourceLimits,
    Session,
    UnknownExecution,
    UnknownSession,
)
from lagrl.testing.fixtures import ManualClock


class ControlledExecutor:
    """Scripted execution result and controlled completion; never runs a command.

    The actual SessionManager must supply deduplication, state, deadlines and cleanup.
    """

    def __init__(self, clock: ManualClock, mode: str = "success") -> None:
        self.clock = clock
        self.mode = mode
        self.calls: list[tuple[Session, ExecuteRequest]] = []
        self.released: list[str] = []
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(self, session: Session, request: ExecuteRequest) -> ExecutionResult:
        self.calls.append((session, request))
        self.started.set()
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.mode == "infrastructure":
            raise RuntimeError("scripted isolation transport failure")
        return ExecutionResult(
            ExecutionReference(session.session_id, request.execution_id),
            ExecutionState.SUCCEEDED if self.mode == "success" else ExecutionState.FAILED,
            b"scripted output longer than four bytes",
            b"scripted stderr",
            0 if self.mode == "success" else 7,
            self.clock(),
            self.clock(),
            self.clock(),
            FailureCategory.NONE if self.mode == "success" else FailureCategory.COMMAND_EXIT,
        )

    async def release(self, session: Session) -> None:
        self.released.append(session.session_id)


def make_manager(mode: str = "success") -> tuple[SessionManager, ControlledExecutor, ManualClock]:
    clock = ManualClock()
    executor = ControlledExecutor(clock, mode)
    return SessionManager(executor, clock=clock), executor, clock


def request_for(
    session: Session, execution_id: str = "exec", deadline: float = 110.0
) -> ExecuteRequest:
    return ExecuteRequest(session.session_id, execution_id, ("recorded-command",), deadline, 4)


async def terminal_result(
    manager: SessionManager, reference: ExecutionReference
) -> ExecutionResult:
    """Bounded observation only, not a lifecycle implementation or test oracle."""
    async with asyncio.timeout(1):
        while True:
            result = await manager.get_execution(reference)
            if result.state not in (ExecutionState.QUEUED, ExecutionState.RUNNING):
                return result
            await asyncio.sleep(0)


async def test_prompt_acceptance_and_concurrent_duplicate_only_start_once() -> None:
    manager, executor, _ = make_manager()
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 30)
    )
    try:
        request = request_for(session)
        refs = await asyncio.wait_for(
            asyncio.gather(manager.execute(request), manager.execute(request)), 1
        )
        assert refs[0] == refs[1]
        await asyncio.wait_for(executor.started.wait(), 1)
        assert len(executor.calls) == 1
        assert not executor.finish.is_set()
        with pytest.raises(ExecutionConflict):
            await manager.execute(replace(request, argv=("conflicting-command",)))
    finally:
        await manager.close()


async def test_absolute_deadline_boundary_does_not_launch() -> None:
    manager, executor, clock = make_manager()
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 30)
    )
    try:
        reference = await manager.execute(request_for(session, deadline=clock()))
        result = await manager.get_execution(reference)
        assert result.state is ExecutionState.TIMED_OUT
        assert result.failure_category is FailureCategory.COMMAND_DEADLINE
        assert result.exit_code is None
        assert executor.calls == []
    finally:
        await manager.close()


async def test_running_deadline_preserves_live_session() -> None:
    manager, executor, clock = make_manager()
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 30)
    )
    try:
        reference = await manager.execute(request_for(session))
        await asyncio.wait_for(executor.started.wait(), 1)
        clock.advance(10)
        await manager.expire_executions()
        result = await manager.get_execution(reference)
        assert result.state is ExecutionState.TIMED_OUT
        assert result.failure_category is FailureCategory.COMMAND_DEADLINE
        assert executor.cancelled.is_set()
        assert executor.released == []
        next_ref = await manager.execute(request_for(session, "next", 120))
        assert next_ref.execution_id == "next"
    finally:
        await manager.close()


async def test_lease_boundary_cancels_and_invalidates_records() -> None:
    manager, executor, clock = make_manager()
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 5)
    )
    assert session.expires_at == clock() + 5
    reference = await manager.execute(request_for(session))
    await asyncio.wait_for(executor.started.wait(), 1)
    clock.advance(5)
    await manager.expire_sessions()
    assert executor.cancelled.is_set()
    assert executor.released == [session.session_id]
    with pytest.raises(UnknownSession):
        await manager.execute(request_for(session, "next"))
    with pytest.raises(UnknownExecution):
        await manager.get_execution(reference)
    await manager.expire_sessions()
    await manager.close()
    assert executor.released == [session.session_id]


@pytest.mark.parametrize(
    "mode,category,exit_code",
    [
        ("success", FailureCategory.NONE, 0),
        ("command_exit", FailureCategory.COMMAND_EXIT, 7),
        ("infrastructure", FailureCategory.INFRASTRUCTURE, None),
    ],
)
async def test_bounded_output_and_failure_classification(
    mode: str, category: FailureCategory, exit_code: int | None
) -> None:
    manager, executor, _ = make_manager(mode)
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 30)
    )
    try:
        reference = await manager.execute(request_for(session))
        executor.finish.set()
        result = await terminal_result(manager, reference)
        assert result.failure_category is category
        assert result.exit_code == exit_code
        assert len(result.stdout) + len(result.stderr) <= 4
        if mode != "infrastructure":
            assert result.output_truncated is True
        assert result.finished_at is not None
    finally:
        await manager.close()


async def test_destroy_and_close_idempotent_and_no_work_survives() -> None:
    manager, executor, _ = make_manager()
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 30)
    )
    reference = await manager.execute(request_for(session))
    await asyncio.wait_for(executor.started.wait(), 1)
    assert await manager.destroy_session(session.session_id)
    assert await manager.destroy_session(session.session_id)
    assert await manager.destroy_session("unknown")
    assert executor.released == [session.session_id]
    assert executor.cancelled.is_set()
    with pytest.raises(UnknownExecution):
        await manager.get_execution(reference)
    await manager.close()
    await manager.close()
    with pytest.raises(InvalidRequest):
        await manager.create_session(CreateSessionRequest("python-minimal", ResourceLimits(), 30))


@pytest.mark.parametrize(
    "session_request",
    [
        CreateSessionRequest("unapproved", ResourceLimits(), 1),
        CreateSessionRequest("python-minimal", ResourceLimits(), 0),
        CreateSessionRequest("python-minimal", ResourceLimits(), 61),
        CreateSessionRequest("python-minimal", ResourceLimits(memory_bytes=1_000_000_000), 1),
    ],
)
async def test_unapproved_environment_or_budget_rejected(
    session_request: CreateSessionRequest,
) -> None:
    manager, _, _ = make_manager()
    with pytest.raises(InvalidRequest):
        await manager.create_session(session_request)


async def test_execution_concurrency_is_bounded_and_workspace_stable() -> None:
    manager, executor, _ = make_manager()
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 30)
    )
    try:
        first = await manager.execute(request_for(session))
        await asyncio.wait_for(executor.started.wait(), 1)
        with pytest.raises(ResourceExhausted):
            await manager.execute(request_for(session, "second"))
        executor.finish.set()
        await terminal_result(manager, first)
        second = await manager.execute(request_for(session, "second"))
        await terminal_result(manager, second)
        assert len(executor.calls) == 2
        assert executor.calls[0][0].workspace_ref == executor.calls[1][0].workspace_ref
    finally:
        await manager.close()


async def test_restart_invalidates_old_session_identity() -> None:
    manager, _, _ = make_manager()
    session = await manager.create_session(
        CreateSessionRequest("python-minimal", ResourceLimits(), 30)
    )
    await manager.close()
    restarted, _, _ = make_manager()
    try:
        with pytest.raises(UnknownSession):
            await restarted.execute(request_for(session))
    finally:
        await restarted.close()

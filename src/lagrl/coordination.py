"""Async coordination exercises. Transport and fixtures do not implement scheduling."""

import asyncio
from collections.abc import Callable

from lagrl.contracts import CompleteRolloutGroup, GroupRequest
from lagrl.rollout import RolloutBackend
from lagrl.training import TrainerBackend


class BufferClosed(RuntimeError):
    """Buffer closed to new puts; gets fail after admitted contents are drained."""


class GroupBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity

    async def put(self, group: CompleteRolloutGroup) -> None:
        """TODO[T06]: Enqueue one complete group with bounded backpressure.

        Input: immutable group; output: None after admission. Capacity is groups, not
        tokens. Preconditions: open buffer. Full buffers wait without polling.
        Edge cases: closed buffers raise BufferClosed; cancelled puts never insert a
        group later and must release any waiter bookkeeping.
        Question: When does ownership of a waiting producer's group transfer?
        Tests: tests/exercises/test_t06.py. Docs: docs/exercises.md#t06.
        """
        raise NotImplementedError("TODO[T06] GroupBuffer.put")

    async def get(self) -> CompleteRolloutGroup:
        """TODO[T06]: Transfer one FIFO group from the buffer to a consumer.

        Input: buffer state; output: one group. Preconditions: no concurrent mutation
        outside these methods. Empty open buffer waits; closed buffer drains existing
        groups, then raises BufferClosed. Cancellation must not lose a queued group.
        Question: What observable event wakes a blocked producer?
        Tests: tests/exercises/test_t06.py. Docs: docs/exercises.md#t06.
        """
        raise NotImplementedError("TODO[T06] GroupBuffer.get")

    async def close(self) -> None:
        """TODO[T06]: Close admission and wake all waiting producers and consumers.

        Input: any buffer state; output: None. Shape: no tensors.
        Preconditions: initialized buffer; it may already be closed.
        Idempotent close rejects pending/new puts, preserves FIFO groups for draining, and
        wakes empty consumers with BufferClosed. No background tasks remain owned.
        Question: How does graceful drain differ from cancelling in-flight production?
        Tests: tests/exercises/test_t06.py. Docs: docs/exercises.md#t06.
        """
        raise NotImplementedError("TODO[T06] GroupBuffer.close")


async def run_coordination(
    requests: tuple[GroupRequest, ...],
    rollout: RolloutBackend,
    trainer: TrainerBackend,
    buffer: GroupBuffer,
    stop: asyncio.Event,
    current_version: Callable[[], int],
    max_policy_lag: int,
) -> None:
    """TODO[T06]: Own trainer/rollout tasks, bounded handoff, cancellation and shutdown.

    Inputs: requests, backends, buffer, stop event, current version callback and lag bound;
    output: None after all owned tasks and backends close. Preconditions: valid requests.
    Assemble/admit whole groups using T04, then train; never emit partial groups.
    Exhausting all requests drains admitted work and returns without requiring stop.
    Edge cases: stop cancels active generation, drains admitted work, closes each backend
    exactly once; worker errors propagate after cleanup, caller cancellation propagates.
    No dropped exceptions or orphan tasks. Shape: groups [G] at the handoff boundary.
    Question: Which component owns cancellation when generation outlives training?
    Tests: tests/exercises/test_t06.py. Docs: docs/architecture.md.
    """
    raise NotImplementedError("TODO[T06] run_coordination")

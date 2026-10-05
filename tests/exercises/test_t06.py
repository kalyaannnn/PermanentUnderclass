import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from lagrl.contracts import CompleteRolloutGroup, GroupRequest, TrainingResult, Trajectory
from lagrl.coordination import BufferClosed, GroupBuffer, run_coordination
from lagrl.testing.fixtures import group_fixture


async def test_backpressure_and_fifo() -> None:
    buffer = GroupBuffer(1)
    first = group_fixture()
    second = replace(
        first, trajectories=(replace(first.trajectories[0], reward=1.0), first.trajectories[1])
    )
    await buffer.put(first)
    waiting = asyncio.create_task(buffer.put(second))
    try:
        await asyncio.sleep(0)
        assert not waiting.done()
        assert await buffer.get() == first
        await asyncio.wait_for(waiting, 1)
        assert await buffer.get() == second
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await buffer.close()


async def test_cancelled_put_does_not_insert_later() -> None:
    buffer = GroupBuffer(1)
    group = group_fixture()
    await buffer.put(group)
    waiting = asyncio.create_task(buffer.put(group))
    try:
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert await buffer.get() == group
        await buffer.close()
        with pytest.raises(BufferClosed):
            await buffer.get()
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await buffer.close()


async def test_close_wakes_waiters_and_drains_admitted_groups() -> None:
    buffer = GroupBuffer(1)
    group = group_fixture()
    await buffer.put(group)
    waiting = asyncio.create_task(buffer.put(group))
    try:
        await asyncio.sleep(0)
        await buffer.close()
        with pytest.raises(BufferClosed):
            await asyncio.wait_for(waiting, 1)
        assert await buffer.get() == group
        with pytest.raises(BufferClosed):
            await buffer.get()
        with pytest.raises(BufferClosed):
            await buffer.put(group)
        await buffer.close()
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_close_wakes_empty_consumer() -> None:
    buffer = GroupBuffer(1)
    waiting = asyncio.create_task(buffer.get())
    try:
        await asyncio.sleep(0)
        assert not waiting.done()
        await buffer.close()
        with pytest.raises(BufferClosed):
            await asyncio.wait_for(waiting, 1)
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


class BlockingRollout:
    """Event-controlled source only; no assembly, admission or scheduler."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = 0

    async def generate(self, request: GroupRequest) -> AsyncIterator[Trajectory]:
        self.started.set()
        try:
            await asyncio.Event().wait()
            yield group_fixture().trajectories[0]
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def close(self) -> None:
        self.closed += 1


class RecordingTrainer:
    def __init__(self) -> None:
        self.groups: list[CompleteRolloutGroup] = []
        self.closed = 0

    async def train(self, group: CompleteRolloutGroup) -> TrainingResult:
        self.groups.append(group)
        return TrainingResult(1, 3, 0.0, 4)

    async def close(self) -> None:
        self.closed += 1


async def test_coordinator_stop_cancels_rollout_and_closes_backends() -> None:
    source, trainer, stop = BlockingRollout(), RecordingTrainer(), asyncio.Event()
    task = asyncio.create_task(
        run_coordination(
            (group_fixture().request,), source, trainer, GroupBuffer(1), stop, lambda: 2, 1
        )
    )
    try:
        # Race startup against early failure so an unfinished TODO fails directly.
        started = asyncio.create_task(source.started.wait())
        try:
            await asyncio.wait({task, started}, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                await task
            assert source.started.is_set()
            stop.set()
            await asyncio.wait_for(task, 1)
            assert source.cancelled.is_set()
            assert source.closed == trainer.closed == 1
            assert trainer.groups == []
        finally:
            started.cancel()
            await asyncio.gather(started, return_exceptions=True)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class RecordedRollout:
    """Fixed two-record source; assembly and admission remain production responsibilities."""

    def __init__(self) -> None:
        self.closed = 0

    async def generate(self, request: GroupRequest) -> AsyncIterator[Trajectory]:
        for trajectory in group_fixture().trajectories[::-1]:
            yield trajectory

    async def close(self) -> None:
        self.closed += 1


async def test_coordinator_trains_only_complete_admitted_group_then_finishes() -> None:
    source, trainer = RecordedRollout(), RecordingTrainer()
    await asyncio.wait_for(
        run_coordination(
            (group_fixture().request,),
            source,
            trainer,
            GroupBuffer(1),
            asyncio.Event(),
            lambda: 2,
            1,
        ),
        1,
    )
    assert trainer.groups == [group_fixture()]
    assert source.closed == trainer.closed == 1


async def test_coordinator_caller_cancellation_propagates_after_cleanup() -> None:
    source, trainer = BlockingRollout(), RecordingTrainer()
    task = asyncio.create_task(
        run_coordination(
            (group_fixture().request,),
            source,
            trainer,
            GroupBuffer(1),
            asyncio.Event(),
            lambda: 2,
            1,
        )
    )
    started = asyncio.create_task(source.started.wait())
    try:
        await asyncio.wait({task, started}, timeout=1, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task
        assert source.started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert source.cancelled.is_set()
        assert source.closed == trainer.closed == 1
    finally:
        task.cancel()
        started.cancel()
        await asyncio.gather(task, started, return_exceptions=True)

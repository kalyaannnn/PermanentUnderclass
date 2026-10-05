import asyncio
from dataclasses import replace

import pytest

from lagrl.contracts import WeightAcknowledgement, WeightSnapshot
from lagrl.publication import PolicyPublisher, validate_acknowledgement

SNAPSHOT = WeightSnapshot(1, "memory://weights/1", "a" * 64)


def test_matching_ack_metadata() -> None:
    assert validate_acknowledgement(SNAPSHOT, "a", WeightAcknowledgement("a", 1, "a" * 64)) is None


@pytest.mark.parametrize(
    "ack",
    [
        WeightAcknowledgement("other", 1, "a" * 64),
        WeightAcknowledgement("a", 2, "a" * 64),
        WeightAcknowledgement("a", 1, "b" * 64),
    ],
)
def test_mismatched_ack_metadata_rejected(ack: WeightAcknowledgement) -> None:
    with pytest.raises(ValueError):
        validate_acknowledgement(SNAPSHOT, "a", ack)


class RecordingReceiver:
    """Records transport requests; no publication or acknowledgement state machine."""

    def __init__(self, worker_id: str, wrong_version: bool = False) -> None:
        self.worker_id = worker_id
        self.wrong_version = wrong_version
        self.calls: list[WeightSnapshot] = []

    async def apply_weights(self, snapshot: WeightSnapshot) -> WeightAcknowledgement:
        self.calls.append(snapshot)
        return WeightAcknowledgement(
            self.worker_id, 999 if self.wrong_version else snapshot.version, snapshot.sha256
        )


async def test_matching_acknowledgements_and_committed_version() -> None:
    workers = {name: RecordingReceiver(name) for name in ("b", "a")}
    publisher = PolicyPublisher(workers)
    assert publisher.committed_version is None
    acks = await publisher.publish(SNAPSHOT)
    assert acks == (
        WeightAcknowledgement("a", 1, "a" * 64),
        WeightAcknowledgement("b", 1, "a" * 64),
    )
    assert publisher.committed_version == 1
    assert all(worker.calls == [SNAPSHOT] for worker in workers.values())


async def test_identical_publication_idempotent_and_conflict_rejected() -> None:
    worker = RecordingReceiver("a")
    publisher = PolicyPublisher({"a": worker})
    first = await publisher.publish(SNAPSHOT)
    assert await publisher.publish(SNAPSHOT) == first
    assert worker.calls == [SNAPSHOT]
    with pytest.raises(ValueError):
        await publisher.publish(replace(SNAPSHOT, sha256="b" * 64))
    assert publisher.committed_version == 1


async def test_wrong_ack_does_not_commit() -> None:
    publisher = PolicyPublisher({"a": RecordingReceiver("a", wrong_version=True)})
    with pytest.raises(ValueError):
        await publisher.publish(SNAPSHOT)
    assert publisher.committed_version is None


async def test_empty_workers_rejected() -> None:
    with pytest.raises(ValueError):
        await PolicyPublisher({}).publish(SNAPSHOT)


class GatedReceiver(RecordingReceiver):
    def __init__(self, worker_id: str) -> None:
        super().__init__(worker_id)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def apply_weights(self, snapshot: WeightSnapshot) -> WeightAcknowledgement:
        self.started.set()
        await self.release.wait()
        return await super().apply_weights(snapshot)


async def test_publication_waits_for_all_acks_and_cancellation_does_not_commit() -> None:
    worker = GatedReceiver("b")
    publisher = PolicyPublisher({"a": RecordingReceiver("a"), "b": worker})
    task = asyncio.create_task(publisher.publish(SNAPSHOT))
    started = asyncio.create_task(worker.started.wait())
    try:
        await asyncio.wait({task, started}, timeout=1, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task
        assert worker.started.is_set()
        assert publisher.committed_version is None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert publisher.committed_version is None
    finally:
        task.cancel()
        started.cancel()
        await asyncio.gather(task, started, return_exceptions=True)


async def test_stale_publication_rejected() -> None:
    publisher = PolicyPublisher({"a": RecordingReceiver("a")})
    await publisher.publish(SNAPSHOT)
    with pytest.raises(ValueError):
        await publisher.publish(replace(SNAPSHOT, version=0))
    assert publisher.committed_version == 1

"""Versioned weight-publication exercise; SGLang/NCCL adapters come later."""

from typing import Protocol

from lagrl.contracts import WeightAcknowledgement, WeightSnapshot


class WeightReceiver(Protocol):
    async def apply_weights(self, snapshot: WeightSnapshot) -> WeightAcknowledgement: ...


def validate_acknowledgement(
    snapshot: WeightSnapshot, worker_id: str, acknowledgement: WeightAcknowledgement
) -> None:
    """TODO[T07]: Verify one applied-weight acknowledgement against its intended recipient.

    Inputs: immutable snapshot, expected worker ID and one acknowledgement; output:
    None for a matching worker/version/digest. Shape: scalar metadata, no weight tensors.
    Preconditions: a validated snapshot has been sent to this nonempty worker ID.
    Edge cases: wrong worker, version or digest raises ValueError; no state is committed
    here and acknowledging receipt alone must not claim sampler application.
    Question: What evidence distinguishes received weights from applied weights?
    Tests: tests/exercises/test_t07.py. Docs: docs/contracts.md#weights-and-updates.
    """
    raise NotImplementedError("TODO[T07] validate_acknowledgement")


class PolicyPublisher:
    def __init__(self, receivers: dict[str, WeightReceiver]) -> None:
        self.receivers = dict(receivers)
        self.committed_version: int | None = None

    async def publish(self, snapshot: WeightSnapshot) -> tuple[WeightAcknowledgement, ...]:
        """TODO[T07]: Publish an immutable snapshot and collect matching worker acks.

        Input: snapshot version/URI/digest, configured receivers [W>0]. Output: [W]
        verified acks ordered by worker ID; visible version advances only after all ack.
        Preconditions: monotonically increasing versions and valid digest.
        Edge cases: repeated identical publication is idempotent; same version with
        different digest, stale version, wrong worker/version/digest or empty workers
        raises ValueError. Failure/cancellation leaves committed version unchanged.
        Question: Does receipt of bytes prove that a worker's sampler uses those weights?
        Tests: tests/exercises/test_t07.py. Docs: docs/exercises.md#t07.
        """
        raise NotImplementedError("TODO[T07] PolicyPublisher.publish")

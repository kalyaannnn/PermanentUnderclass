"""Production rollout assembly exercises and the future generator boundary."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from lagrl.contracts import (
    CompleteRolloutGroup,
    GenerationEvent,
    GroupRequest,
    SamplingSettings,
    Termination,
    ToolEvent,
    Trajectory,
)


class RolloutBackend(Protocol):
    def generate(self, request: GroupRequest) -> AsyncIterator[Trajectory]: ...
    async def close(self) -> None: ...


def assemble_trajectory(
    task_id: str,
    group_id: str,
    trajectory_id: str,
    prompt_token_ids: tuple[int, ...],
    events: Sequence[GenerationEvent | ToolEvent],
    sampling: SamplingSettings,
    reward: float | None,
    termination: Termination,
) -> Trajectory:
    """TODO[T03]: Assemble chronological generation and tool events into one record.

    Inputs: nonempty prompt [P], event token blocks [Li], IDs and sampling settings.
    Output: immutable trajectory [N=P+sum(Li)] with every per-token field aligned.
    Preconditions: generated token/log-prob lengths match, versions are nonnegative.
    Edge cases: empty action sequence or malformed event raises ValueError; infrastructure
    failure requires reward=None, valid zero reward remains 0.0. Tool/prompt tokens have
    no behavior log-prob; mixed-version actions retain all half-open policy spans.
    Question: Which policy actually sampled each action after a tool call?
    Tests: tests/exercises/test_t03.py. Contract: docs/contracts.md#token-indexing.
    """
    raise NotImplementedError("TODO[T03] assemble_trajectory")


def assemble_complete_group(
    request: GroupRequest, trajectories: Sequence[Trajectory]
) -> CompleteRolloutGroup:
    """TODO[T04]: Verify all requested responses before constructing a complete group.

    Inputs: request with G>=2 distinct expected IDs, unordered response records [G].
    Output: group in request-ID order, each with finite reward and completed status.
    Preconditions: expected task/group identity is authoritative.
    Edge cases: missing, extra, duplicate, mismatched or failed records raise ValueError;
    no fast subset is selected. Zero rewards are valid; infrastructure failures are not.
    Question: What changes if only fast responses enter the reward normalization?
    Tests: tests/exercises/test_t04.py. Docs: docs/contracts.md#groups-and-policy-lag.
    """
    raise NotImplementedError("TODO[T04] assemble_complete_group")


def admit_group(group: CompleteRolloutGroup, current_version: int, max_policy_lag: int) -> bool:
    """TODO[T04]: Decide admission for the whole complete group using every action span.

    Inputs: complete group [G], nonnegative current version and lag bound.
    Output: True exactly when every action policy lag is within the inclusive bound.
    Preconditions: T03/T04 completeness and span invariants hold.
    Edge cases: future versions or negative bounds raise ValueError; any stale span
    rejects the entire group. Admission never rewrites provenance or drops responses.
    Question: Why is the oldest action span relevant within a single trajectory?
    Tests: tests/exercises/test_t04.py. Docs: docs/contracts.md#groups-and-policy-lag.
    """
    raise NotImplementedError("TODO[T04] admit_group")

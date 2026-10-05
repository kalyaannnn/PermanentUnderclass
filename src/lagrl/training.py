"""Production numerical exercises; importing this module does not import PyTorch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from lagrl.contracts import CompleteRolloutGroup, TrainingResult

if TYPE_CHECKING:
    import torch


class TrainerBackend(Protocol):
    async def train(self, group: CompleteRolloutGroup) -> TrainingResult: ...
    async def close(self) -> None: ...


@dataclass(frozen=True)
class TrainingBatch:
    """Trainer inputs: full token coordinates; old-policy values are separate data."""

    token_ids: torch.Tensor  # [B, L], integer
    action_mask: torch.Tensor  # [B, L], bool
    old_policy_log_probs: torch.Tensor  # [B, L], optimizer reference, not sampler values
    rewards: torch.Tensor  # [B], one complete group


def group_relative_advantages(rewards: Sequence[float], epsilon: float = 1e-6) -> tuple[float, ...]:
    """TODO[T01]: Compute group-relative advantages for one complete reward group.

    Inputs: finite rewards [G], G>=2; finite epsilon>0. Output: plain floats [G].
    Definition: (reward - group mean)/(population standard deviation + epsilon).
    Preconditions: completeness was established before normalization; no failed rewards.
    Edge cases: equal rewards return exact zeros; empty/singleton/nonfinite rewards or
    invalid epsilon raise ValueError. Returned floats have no autograd history.
    Question: Why use population standard deviation for this complete sampled group?
    Tests: tests/exercises/test_t01.py. Docs: docs/contracts.md#grpo-baseline.
    """
    raise NotImplementedError("TODO[T01] group_relative_advantages")


def grpo_loss(
    current_log_probs: torch.Tensor,
    old_policy_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """TODO[T02]: Compute the clipped baseline policy loss with response-wise reduction.

    Inputs: current/old log-probs [B,L], advantages [B], bool action mask [B,L].
    Output: scalar loss retaining gradient only through current action log-probs.
    Preconditions: finite valid entries, matching shapes, B>0, 0<clip_epsilon<1;
    old-policy values and advantages are detached, KL coefficient is zero.
    Edge cases: each response needs an action; invalid shapes/masks raise ValueError.
    Prompt/tool/padding values are excluded even if nonfinite. Reduce within response,
    then across responses. Sampler values are not inputs and are never overwritten.
    Question: Why does averaging all tokens weight longer responses differently?
    Tests: tests/exercises/test_t02.py. Docs: docs/contracts.md#grpo-baseline.
    """
    raise NotImplementedError("TODO[T02] grpo_loss")


def evaluate_action_log_probs(
    model: torch.nn.Module, token_ids: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """TODO[T05]: Evaluate causal model log-probabilities in full token coordinates.

    Inputs: model returning .logits [B,L,V], integer tokens [B,L], bool mask [B,L].
    Output: log-probs [B,L], differentiable at actions; non-action entries are zero.
    Preconditions: L>=2; token t is predicted by logits t-1; token 0 cannot be action.
    Edge cases: malformed shapes, empty responses or first-token actions raise
    ValueError. Tool observations remain context but have no policy-loss contribution.
    Question: Which logit position predicts the first action after a prompt?
    Tests: tests/exercises/test_t05.py. Docs: docs/contracts.md#token-indexing.
    """
    raise NotImplementedError("TODO[T05] evaluate_action_log_probs")


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    step: int,
    policy_version: int,
    epsilon: float = 1e-6,
    clip_epsilon: float = 0.2,
) -> TrainingResult:
    """TODO[T05]: Evaluate, normalize, optimize once, and return finite training metrics.

    Inputs: model/optimizer, one complete group batch [B,L], nonnegative step/version.
    Output: result for step+1/version+1, scalar loss and valid action-token count.
    Preconditions: T01/T02 and causal evaluation contracts; no snapshot is published here.
    Edge cases: clear accumulated gradients before update; reject invalid batches before
    updating parameters. Preserve old-policy inputs; never mutate sampler provenance.
    Question: At what point may an updated policy version become externally visible?
    Tests: tests/exercises/test_t05.py. Docs: docs/exercises.md#t05.
    """
    raise NotImplementedError("TODO[T05] training_step")

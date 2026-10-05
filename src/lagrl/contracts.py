"""Immutable records. Token and numerical semantics are in docs/contracts.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TokenRole(StrEnum):
    PROMPT = "prompt"
    ACTION = "action"
    TOOL = "tool"
    PADDING = "padding"


class Termination(StrEnum):
    COMPLETED = "completed"
    INFRA_FAILURE = "infra_failure"
    COMMAND_TIMEOUT = "command_timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 32
    seed: int = 0


@dataclass(frozen=True)
class PolicySpan:
    """Half-open [start, stop) action-only span, in full sequence token coordinates."""

    start: int
    stop: int
    version: int


@dataclass(frozen=True)
class GenerationEvent:
    token_ids: tuple[int, ...]
    behavior_log_probs: tuple[float, ...]
    policy_version: int


@dataclass(frozen=True)
class ToolEvent:
    execution_id: str
    argv: tuple[str, ...]
    observation_token_ids: tuple[int, ...]
    stdout: str
    stderr: str
    failure_category: str | None = None


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    group_id: str
    trajectory_id: str
    token_ids: tuple[int, ...]
    token_roles: tuple[TokenRole, ...]
    action_mask: tuple[bool, ...]
    behavior_log_probs: tuple[float | None, ...]
    sampling: SamplingSettings
    policy_spans: tuple[PolicySpan, ...]
    tool_events: tuple[ToolEvent, ...]
    reward: float | None
    termination: Termination


@dataclass(frozen=True)
class GroupRequest:
    task_id: str
    group_id: str
    trajectory_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompleteRolloutGroup:
    """T04 guarantees completeness against request; construction alone does not."""

    request: GroupRequest
    trajectories: tuple[Trajectory, ...]


@dataclass(frozen=True)
class WeightSnapshot:
    version: int
    artifact_uri: str
    sha256: str


@dataclass(frozen=True)
class WeightAcknowledgement:
    worker_id: str
    version: int
    sha256: str


@dataclass(frozen=True)
class TrainingResult:
    step: int
    policy_version: int
    loss: float
    valid_action_tokens: int
    metrics: tuple[tuple[str, float], ...] = ()

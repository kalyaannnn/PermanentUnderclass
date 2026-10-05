"""Sandbox wire-domain records. Times are Unix seconds from an injectable clock."""

from dataclasses import dataclass
from enum import StrEnum


class SessionState(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    DESTROYED = "destroyed"


class ExecutionState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class FailureCategory(StrEnum):
    NONE = "none"
    COMMAND_EXIT = "command_exit"
    COMMAND_DEADLINE = "command_deadline"
    INFRASTRUCTURE = "infrastructure"
    SESSION_EXPIRED = "session_expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ResourceLimits:
    cpu_millis: int = 1000
    memory_bytes: int = 67_108_864
    max_processes: int = 8
    max_concurrent_executions: int = 1


@dataclass(frozen=True)
class CreateSessionRequest:
    environment: str
    limits: ResourceLimits
    lease_seconds: float


@dataclass(frozen=True)
class Session:
    session_id: str
    environment: str
    limits: ResourceLimits
    created_at: float
    expires_at: float
    state: SessionState
    workspace_ref: str


@dataclass(frozen=True)
class ExecuteRequest:
    session_id: str
    execution_id: str
    argv: tuple[str, ...]
    command_deadline: float
    output_limit_bytes: int


@dataclass(frozen=True)
class ExecutionReference:
    session_id: str
    execution_id: str


@dataclass(frozen=True)
class ExecutionResult:
    reference: ExecutionReference
    state: ExecutionState
    stdout: bytes
    stderr: bytes
    exit_code: int | None
    accepted_at: float
    started_at: float | None
    finished_at: float | None
    failure_category: FailureCategory
    output_truncated: bool = False


class SandboxError(RuntimeError):
    """Domain errors map to gRPC status without deciding lifecycle in the transport."""


class InvalidRequest(SandboxError):
    pass


class UnknownSession(SandboxError):
    pass


class UnknownExecution(SandboxError):
    pass


class ExecutionConflict(SandboxError):
    pass


class ResourceExhausted(SandboxError):
    pass


class IsolationUnavailable(SandboxError):
    pass

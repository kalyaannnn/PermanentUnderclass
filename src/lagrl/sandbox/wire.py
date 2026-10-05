"""Explicit lossless conversion between domain records and generated messages."""

from dataclasses import asdict

from lagrl.sandbox.proto import sandbox_pb2 as pb
from lagrl.sandbox.records import (
    CreateSessionRequest,
    ExecuteRequest,
    ExecutionReference,
    ExecutionResult,
    ExecutionState,
    FailureCategory,
    ResourceLimits,
    Session,
    SessionState,
)


def encode_limits(value: ResourceLimits) -> pb.ResourceLimits:
    return pb.ResourceLimits(**asdict(value))


def decode_limits(value: pb.ResourceLimits) -> ResourceLimits:
    return ResourceLimits(
        value.cpu_millis, value.memory_bytes, value.max_processes, value.max_concurrent_executions
    )


def encode_create(value: CreateSessionRequest) -> pb.CreateSessionRequest:
    return pb.CreateSessionRequest(
        environment=value.environment,
        limits=encode_limits(value.limits),
        lease_seconds=value.lease_seconds,
    )


def decode_create(value: pb.CreateSessionRequest) -> CreateSessionRequest:
    return CreateSessionRequest(value.environment, decode_limits(value.limits), value.lease_seconds)


def encode_session(value: Session) -> pb.Session:
    return pb.Session(
        session_id=value.session_id,
        environment=value.environment,
        limits=encode_limits(value.limits),
        created_at=value.created_at,
        expires_at=value.expires_at,
        state=value.state.value,
        workspace_ref=value.workspace_ref,
    )


def decode_session(value: pb.Session) -> Session:
    return Session(
        value.session_id,
        value.environment,
        decode_limits(value.limits),
        value.created_at,
        value.expires_at,
        SessionState(value.state),
        value.workspace_ref,
    )


def encode_execute(value: ExecuteRequest) -> pb.ExecuteRequest:
    return pb.ExecuteRequest(**asdict(value))


def decode_execute(value: pb.ExecuteRequest) -> ExecuteRequest:
    return ExecuteRequest(
        value.session_id,
        value.execution_id,
        tuple(value.argv),
        value.command_deadline,
        value.output_limit_bytes,
    )


def encode_reference(value: ExecutionReference) -> pb.ExecutionReference:
    return pb.ExecutionReference(**asdict(value))


def decode_reference(value: pb.ExecutionReference) -> ExecutionReference:
    return ExecutionReference(value.session_id, value.execution_id)


def encode_result(value: ExecutionResult) -> pb.ExecutionResult:
    return pb.ExecutionResult(
        reference=encode_reference(value.reference),
        state=value.state.value,
        stdout=value.stdout,
        stderr=value.stderr,
        exit_code=value.exit_code,
        accepted_at=value.accepted_at,
        started_at=value.started_at,
        finished_at=value.finished_at,
        failure_category=value.failure_category.value,
        output_truncated=value.output_truncated,
    )


def decode_result(value: pb.ExecutionResult) -> ExecutionResult:
    return ExecutionResult(
        decode_reference(value.reference),
        ExecutionState(value.state),
        value.stdout,
        value.stderr,
        value.exit_code if value.HasField("exit_code") else None,
        value.accepted_at,
        value.started_at if value.HasField("started_at") else None,
        value.finished_at if value.HasField("finished_at") else None,
        FailureCategory(value.failure_category),
        value.output_truncated,
    )

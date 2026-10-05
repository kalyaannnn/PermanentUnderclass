"""Narrow asynchronous lifecycle and isolated-executor interfaces."""

from typing import Protocol

from lagrl.sandbox.records import (
    CreateSessionRequest,
    ExecuteRequest,
    ExecutionReference,
    ExecutionResult,
    Session,
)


class SandboxBackend(Protocol):
    async def create_session(self, request: CreateSessionRequest) -> Session: ...
    async def execute(self, request: ExecuteRequest) -> ExecutionReference: ...
    async def get_execution(self, reference: ExecutionReference) -> ExecutionResult: ...
    async def destroy_session(self, session_id: str) -> bool: ...
    async def close(self) -> None: ...


class IsolatedExecutor(Protocol):
    async def run(self, session: Session, request: ExecuteRequest) -> ExecutionResult: ...
    async def release(self, session: Session) -> None: ...

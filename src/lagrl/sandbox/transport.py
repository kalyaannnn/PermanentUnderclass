"""grpc.aio transport. Backend decisions remain wholly behind SandboxBackend."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import grpc

from lagrl.config import SandboxConfig
from lagrl.sandbox import wire
from lagrl.sandbox.backend import SandboxBackend
from lagrl.sandbox.proto import sandbox_pb2 as pb
from lagrl.sandbox.proto import sandbox_pb2_grpc as rpc
from lagrl.sandbox.records import (
    CreateSessionRequest,
    ExecuteRequest,
    ExecutionConflict,
    ExecutionReference,
    ExecutionResult,
    InvalidRequest,
    IsolationUnavailable,
    ResourceExhausted,
    SandboxError,
    Session,
    UnknownExecution,
    UnknownSession,
)

T = TypeVar("T")
ERROR_CODES = {
    InvalidRequest: grpc.StatusCode.INVALID_ARGUMENT,
    ExecutionConflict: grpc.StatusCode.ALREADY_EXISTS,
    UnknownSession: grpc.StatusCode.NOT_FOUND,
    UnknownExecution: grpc.StatusCode.NOT_FOUND,
    ResourceExhausted: grpc.StatusCode.RESOURCE_EXHAUSTED,
    IsolationUnavailable: grpc.StatusCode.FAILED_PRECONDITION,
}


async def _delegate(context: grpc.aio.ServicerContext, call: Callable[[], Awaitable[T]]) -> T:
    try:
        return await call()
    except NotImplementedError as error:
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
        raise AssertionError("abort returned") from error
    except SandboxError as error:
        await context.abort(ERROR_CODES.get(type(error), grpc.StatusCode.INTERNAL), str(error))
        raise AssertionError("abort returned") from error
    except ValueError as error:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        raise AssertionError("abort returned") from error


class SandboxService(rpc.SandboxServicer):
    def __init__(self, backend: SandboxBackend) -> None:
        self.backend = backend

    async def CreateSession(
        self, request: pb.CreateSessionRequest, context: grpc.aio.ServicerContext
    ) -> pb.Session:
        return wire.encode_session(
            await _delegate(
                context, lambda: self.backend.create_session(wire.decode_create(request))
            )
        )

    async def Execute(
        self, request: pb.ExecuteRequest, context: grpc.aio.ServicerContext
    ) -> pb.ExecutionReference:
        return wire.encode_reference(
            await _delegate(context, lambda: self.backend.execute(wire.decode_execute(request)))
        )

    async def GetExecution(
        self, request: pb.ExecutionReference, context: grpc.aio.ServicerContext
    ) -> pb.ExecutionResult:
        return wire.encode_result(
            await _delegate(
                context, lambda: self.backend.get_execution(wire.decode_reference(request))
            )
        )

    async def DestroySession(
        self, request: pb.DestroySessionRequest, context: grpc.aio.ServicerContext
    ) -> pb.DestroySessionResult:
        return pb.DestroySessionResult(
            released=await _delegate(
                context, lambda: self.backend.destroy_session(request.session_id)
            )
        )


@asynccontextmanager
async def running_server(backend: SandboxBackend, config: SandboxConfig) -> AsyncIterator[str]:
    """Own the loopback server; caller owns backend startup and close."""
    if config.host != "127.0.0.1":
        raise ValueError("development server must bind 127.0.0.1")
    options = [
        ("grpc.max_receive_message_length", config.max_message_bytes),
        ("grpc.max_send_message_length", config.max_message_bytes),
    ]
    server = grpc.aio.server(options=options, maximum_concurrent_rpcs=config.max_concurrent_rpcs)
    try:
        rpc.add_SandboxServicer_to_server(SandboxService(backend), server)
        port = server.add_insecure_port(f"{config.host}:{config.port}")
        if port == 0:
            raise RuntimeError("sandbox server failed to bind")
        await server.start()
        yield f"{config.host}:{port}"
    finally:
        await server.stop(grace=0)
        await server.wait_for_termination()


class SandboxClient:
    def __init__(self, target: str, config: SandboxConfig) -> None:
        self.target = target
        self.timeout = config.rpc_timeout_seconds
        self.channel = grpc.aio.insecure_channel(
            target,
            options=[
                ("grpc.max_receive_message_length", config.max_message_bytes),
                ("grpc.max_send_message_length", config.max_message_bytes),
            ],
        )
        self.stub = rpc.SandboxStub(self.channel)

    async def __aenter__(self) -> SandboxClient:
        import asyncio

        try:
            await asyncio.wait_for(self.channel.channel_ready(), timeout=self.timeout)
        except BaseException:
            await self.channel.close()
            raise
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.channel.close()

    async def create_session(self, request: CreateSessionRequest) -> Session:
        return wire.decode_session(
            await self.stub.CreateSession(wire.encode_create(request), timeout=self.timeout)
        )

    async def execute(self, request: ExecuteRequest) -> ExecutionReference:
        return wire.decode_reference(
            await self.stub.Execute(wire.encode_execute(request), timeout=self.timeout)
        )

    async def get_execution(self, reference: ExecutionReference) -> ExecutionResult:
        return wire.decode_result(
            await self.stub.GetExecution(wire.encode_reference(reference), timeout=self.timeout)
        )

    async def destroy_session(self, session_id: str) -> bool:
        response = await self.stub.DestroySession(
            pb.DestroySessionRequest(session_id=session_id), timeout=self.timeout
        )
        return response.released

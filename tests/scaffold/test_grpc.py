from dataclasses import replace

import grpc
import pytest

from lagrl.config import SandboxConfig
from lagrl.sandbox import wire
from lagrl.sandbox.gvisor import GVisorExecutor
from lagrl.sandbox.lifecycle import SessionManager
from lagrl.sandbox.proto import sandbox_pb2 as pb
from lagrl.sandbox.records import CreateSessionRequest, ExecutionState, FailureCategory, Session
from lagrl.sandbox.transport import SandboxClient, running_server
from lagrl.testing.fixtures import RecordedSandbox


def test_proto_rpc_contract() -> None:
    service = pb.DESCRIPTOR.services_by_name["Sandbox"]
    assert [method.name for method in service.methods] == [
        "CreateSession",
        "Execute",
        "GetExecution",
        "DestroySession",
    ]


def test_wire_round_trips() -> None:
    fixture = RecordedSandbox()
    pairs = [
        (fixture.create_request, wire.encode_create, wire.decode_create),
        (fixture.session, wire.encode_session, wire.decode_session),
        (fixture.execute_request, wire.encode_execute, wire.decode_execute),
        (fixture.result.reference, wire.encode_reference, wire.decode_reference),
        (fixture.result, wire.encode_result, wire.decode_result),
    ]
    for record, encode, decode in pairs:
        message = encode(record)
        restored = type(message).FromString(message.SerializeToString())
        assert decode(restored) == record


@pytest.mark.parametrize("exit_code", [None, 0, 1])
def test_optional_wire_fields_preserve_absence_and_zero(exit_code: int | None) -> None:
    result = replace(
        RecordedSandbox().result,
        exit_code=exit_code,
        started_at=None,
        finished_at=None,
        state=ExecutionState.QUEUED,
        failure_category=FailureCategory.NONE,
        stdout=b"\x00\xff",
    )
    message = wire.encode_result(result)
    assert message.HasField("exit_code") is (exit_code is not None)
    assert not message.HasField("started_at")
    assert wire.decode_result(message) == result


async def test_fixture_grpc_round_trip() -> None:
    fixture = RecordedSandbox()
    config = SandboxConfig()
    try:
        async with running_server(fixture, config) as target:
            assert target.startswith("127.0.0.1:")
            async with SandboxClient(target, config) as client:
                assert await client.create_session(fixture.create_request) == fixture.session
                reference = await client.execute(fixture.execute_request)
                assert reference == fixture.result.reference
                assert await client.get_execution(reference) == fixture.result
                assert await client.destroy_session(fixture.session.session_id)
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await client.execute(replace(fixture.execute_request, argv=("outside",)))
                assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await fixture.close()
    assert fixture.closed


async def test_handler_preserves_unimplemented_error_without_fixture_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SessionManager(GVisorExecutor())

    async def unfinished(request: CreateSessionRequest) -> Session:
        raise NotImplementedError("injected unfinished method for transport testing")

    monkeypatch.setattr(backend, "create_session", unfinished)
    config = SandboxConfig()
    async with running_server(backend, config) as target:
        async with SandboxClient(target, config) as client:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await client.create_session(RecordedSandbox().create_request)
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
            assert "injected unfinished method" in error.value.details()


async def test_nonloopback_bind_rejected() -> None:
    with pytest.raises(ValueError):
        async with running_server(RecordedSandbox(), SandboxConfig(host="0.0.0.0")):
            pass

"""CPU fixture round trip; no learning, scheduling or command execution."""

from lagrl.config import Config
from lagrl.observability import JsonlLogger
from lagrl.sandbox.transport import SandboxClient, running_server
from lagrl.serialization import trajectory_from_json, trajectory_to_json
from lagrl.testing.fixtures import RecordedSandbox, trajectory_fixture


async def run_smoke(config: Config) -> None:
    print("FAKE SMOKE: no learning and no command execution; validates wiring only.")
    backend = RecordedSandbox()
    with JsonlLogger(config.logging.path) as logger:
        logger.emit("smoke_start", backend="fake", fixture=True, learning=False)
        try:
            trajectory = trajectory_fixture()
            if trajectory_from_json(trajectory_to_json(trajectory)) != trajectory:
                raise RuntimeError("trajectory fixture serialization mismatch")
            async with running_server(backend, config.sandbox) as target:
                logger.emit("grpc_started", target=target, fixture=True)
                async with SandboxClient(target, config.sandbox) as client:
                    session = await client.create_session(backend.create_request)
                    try:
                        reference = await client.execute(backend.execute_request)
                        result = await client.get_execution(reference)
                        if session != backend.session or result != backend.result:
                            raise RuntimeError("gRPC fixture mismatch")
                        logger.emit(
                            "fixture_round_trip",
                            trajectory_id=trajectory.trajectory_id,
                            execution_id=reference.execution_id,
                            fixture=True,
                        )
                    finally:
                        if not await client.destroy_session(session.session_id):
                            raise RuntimeError("fixture destroy response mismatch")
            logger.emit("smoke_success", fixture=True, learning=False)
        finally:
            await backend.close()
            logger.emit("smoke_cleanup", backend_closed=backend.closed, fixture=True)
    print(f"Fixture round trips passed; server/channel/logger closed. JSONL: {config.logging.path}")

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lagrl.config import Config, LoggingConfig
from lagrl.testing import smoke
from lagrl.testing.fixtures import RecordedSandbox


def test_cli_smoke(tmp_path: Path) -> None:
    config = tmp_path / "cpu.yaml"
    log = tmp_path / "smoke.jsonl"
    config.write_text(f"logging:\n  path: {log}\n")
    result = subprocess.run(
        [sys.executable, "-m", "lagrl", "smoke", "--backend", "fake", "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert "FAKE SMOKE: no learning and no command execution" in result.stdout
    assert "validates wiring only" in result.stdout
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [x["event"] for x in events] == [
        "smoke_start",
        "grpc_started",
        "fixture_round_trip",
        "smoke_success",
        "smoke_cleanup",
    ]
    assert all(event["fixture"] for event in events)
    assert events[-1]["backend_closed"] is True


async def test_smoke_cleans_up_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = RecordedSandbox()
    monkeypatch.setattr(smoke, "RecordedSandbox", lambda: fixture)

    async def fail(*args: object) -> None:
        raise RuntimeError("injected transport failure")

    monkeypatch.setattr(smoke.SandboxClient, "get_execution", fail)
    log = tmp_path / "failed.jsonl"
    with pytest.raises(RuntimeError, match="injected transport failure"):
        await smoke.run_smoke(Config(logging=LoggingConfig(str(log))))
    assert fixture.closed
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[-1]["event"] == "smoke_cleanup"
    assert "smoke_success" not in [x["event"] for x in events]

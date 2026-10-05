import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from lagrl.config import load_config
from lagrl.observability import JsonlLogger
from lagrl.serialization import trajectory_from_json, trajectory_to_json
from lagrl.testing.fixtures import trajectory_fixture

ROOT = Path(__file__).resolve().parents[2]


def test_core_imports_without_optional_backends() -> None:
    code = """
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ray', 'sglang'}:
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
import lagrl
import lagrl.contracts, lagrl.training, lagrl.rollout, lagrl.coordination
import lagrl.publication, lagrl.variance, lagrl.sandbox.lifecycle, lagrl.sandbox.gvisor
assert lagrl.__version__ == '0.1.0'
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_cpu_config() -> None:
    config = load_config(ROOT / "configs/cpu.yaml")
    assert config.sandbox.host == "127.0.0.1"
    assert config.training.kl_coefficient == 0
    assert config.coordination.buffer_capacity == 2


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "unknown: {}",
        "sandbox: {unknown: 1}",
        "sandbox: {host: '0.0.0.0'}",
        "training: {normalization_epsilon: 0}",
        "training: {kl_coefficient: 0.1}",
        "coordination: {buffer_capacity: 0}",
        "sandbox: {port: true}",
        "sandbox: {rpc_timeout_seconds: .nan}",
        "training: {group_size: '2'}",
        "sandbox: {max_message_bytes: -1}",
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, text: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_trajectory_round_trip_preserves_provenance() -> None:
    original = trajectory_fixture()
    restored = trajectory_from_json(trajectory_to_json(original))
    assert restored == original
    assert restored.behavior_log_probs == (None, None, -0.25, None, -0.5, None)
    assert tuple(span.version for span in restored.policy_spans) == (1, 2)
    assert isinstance(restored.behavior_log_probs, tuple)
    with pytest.raises(FrozenInstanceError):
        restored.reward = 1.0


def test_serialization_rejects_nonfinite_json() -> None:
    with pytest.raises(ValueError):
        trajectory_to_json(replace(trajectory_fixture(), reward=float("nan")))


def test_jsonl_lifetime_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlLogger(path)
    with pytest.raises(RuntimeError):
        logger.emit("before_open")
    with logger:
        logger.emit("one", fixture=True, policy_version=2)
        with pytest.raises(ValueError):
            logger.emit("bad", timestamp="override")
        with pytest.raises(ValueError):
            logger.emit("nonfinite", loss=float("nan"))
    with pytest.raises(RuntimeError):
        logger.emit("after_close")
    record = json.loads(path.read_text())
    assert record["event"] == "one"
    assert record["fixture"] is True
    assert record["policy_version"] == 2
    assert record["timestamp"].endswith("+00:00")


def test_protobuf_generation_is_reproducible() -> None:
    subprocess.run([sys.executable, "scripts/generate_proto.py", "--check"], cwd=ROOT, check=True)

"""Small strict YAML configuration; no backend initialization or credentials."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SandboxConfig:
    host: str = "127.0.0.1"
    port: int = 0
    rpc_timeout_seconds: float = 3.0
    max_message_bytes: int = 1_048_576
    max_concurrent_rpcs: int = 4


@dataclass(frozen=True)
class LoggingConfig:
    path: str = "artifacts/smoke.jsonl"


@dataclass(frozen=True)
class TrainingConfig:
    group_size: int = 2
    normalization_epsilon: float = 1e-6
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.0


@dataclass(frozen=True)
class CoordinationConfig:
    buffer_capacity: int = 2
    max_policy_lag: int = 1


@dataclass(frozen=True)
class Config:
    sandbox: SandboxConfig = SandboxConfig()
    logging: LoggingConfig = LoggingConfig()
    training: TrainingConfig = TrainingConfig()
    coordination: CoordinationConfig = CoordinationConfig()


def load_config(path: str | Path) -> Config:
    """Read a strict mapping, rejecting unknown keys, invalid types and unsafe binds."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    sections = {
        "sandbox": SandboxConfig,
        "logging": LoggingConfig,
        "training": TrainingConfig,
        "coordination": CoordinationConfig,
    }
    if raw.keys() - sections.keys():
        raise ValueError("unknown configuration section")
    values: dict[str, Any] = {}
    for name, cls in sections.items():
        section = raw.get(name, {})
        if not isinstance(section, dict) or section.keys() - {f.name for f in fields(cls)}:
            raise ValueError(f"invalid keys in {name}")
        defaults = cls()
        for key, value in section.items():
            default = getattr(defaults, key)
            valid = type(value) is type(default)
            if isinstance(default, float):
                valid = type(value) in (int, float) and math.isfinite(value)
            if not valid:
                raise ValueError(f"invalid type/value for {name}.{key}")
        values[name] = cls(**section)
    config = Config(**values)
    s, t, c = config.sandbox, config.training, config.coordination
    if s.host != "127.0.0.1" or not 0 <= s.port <= 65535:
        raise ValueError("development server must bind 127.0.0.1 at a valid port")
    if min(s.rpc_timeout_seconds, s.max_message_bytes, s.max_concurrent_rpcs) <= 0:
        raise ValueError("sandbox limits must be positive")
    if t.group_size < 2 or t.normalization_epsilon <= 0 or not 0 < t.clip_epsilon < 1:
        raise ValueError("invalid training contract")
    if t.kl_coefficient != 0:
        raise ValueError("baseline contract requires KL coefficient zero")
    if c.buffer_capacity < 1 or c.max_policy_lag < 0 or not config.logging.path:
        raise ValueError("invalid coordination/logging configuration")
    return config

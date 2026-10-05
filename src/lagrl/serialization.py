"""Explicit trajectory JSON codec; never recomputes behavior log-probabilities."""

import json
from dataclasses import asdict
from typing import Any

from lagrl.contracts import (
    PolicySpan,
    SamplingSettings,
    Termination,
    TokenRole,
    ToolEvent,
    Trajectory,
)


def trajectory_to_json(trajectory: Trajectory) -> str:
    return json.dumps(asdict(trajectory), sort_keys=True, allow_nan=False)


def trajectory_from_json(payload: str) -> Trajectory:
    data: dict[str, Any] = json.loads(payload)
    return Trajectory(
        task_id=data["task_id"],
        group_id=data["group_id"],
        trajectory_id=data["trajectory_id"],
        token_ids=tuple(data["token_ids"]),
        token_roles=tuple(TokenRole(x) for x in data["token_roles"]),
        action_mask=tuple(data["action_mask"]),
        behavior_log_probs=tuple(data["behavior_log_probs"]),
        sampling=SamplingSettings(**data["sampling"]),
        policy_spans=tuple(PolicySpan(**x) for x in data["policy_spans"]),
        tool_events=tuple(
            ToolEvent(
                **{
                    **x,
                    "argv": tuple(x["argv"]),
                    "observation_token_ids": tuple(x["observation_token_ids"]),
                }
            )
            for x in data["tool_events"]
        ),
        reward=data["reward"],
        termination=Termination(data["termination"]),
    )

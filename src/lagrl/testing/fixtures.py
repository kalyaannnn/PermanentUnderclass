"""Tiny deterministic records, prerecorded replay, and manually controlled clock."""

import json
from dataclasses import dataclass
from importlib.resources import files

from lagrl.contracts import (
    CompleteRolloutGroup,
    GroupRequest,
    PolicySpan,
    SamplingSettings,
    Termination,
    TokenRole,
    ToolEvent,
    Trajectory,
)
from lagrl.sandbox.records import (
    CreateSessionRequest,
    ExecuteRequest,
    ExecutionReference,
    ExecutionResult,
    ExecutionState,
    FailureCategory,
    InvalidRequest,
    ResourceLimits,
    Session,
    SessionState,
)


def trajectory_fixture(trajectory_id: str = "response-a", reward: float = 0.0) -> Trajectory:
    """Fixed full sequence: two prompt, action, tool observation, action, padding."""
    return Trajectory(
        "task-1",
        "group-1",
        trajectory_id,
        (10, 11, 20, 30, 21, 0),
        (
            TokenRole.PROMPT,
            TokenRole.PROMPT,
            TokenRole.ACTION,
            TokenRole.TOOL,
            TokenRole.ACTION,
            TokenRole.PADDING,
        ),
        (False, False, True, False, True, False),
        (None, None, -0.25, None, -0.5, None),
        SamplingSettings(),
        (PolicySpan(2, 3, 1), PolicySpan(4, 5, 2)),
        (ToolEvent("fixture-execution", ("python", "-c", "print(2 + 2)"), (30,), "4\n", ""),),
        reward,
        Termination.COMPLETED,
    )


def group_fixture() -> CompleteRolloutGroup:
    """Two fully authored fixture responses; no assembly or admission algorithm."""
    return CompleteRolloutGroup(
        GroupRequest("task-1", "group-1", ("response-a", "response-b")),
        (trajectory_fixture(), trajectory_fixture("response-b", 2.0)),
    )


@dataclass
class ManualClock:
    now: float = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordedSandbox:
    """Static fixture lookup ONLY: no lifecycle, deduplication, clock or subprocess.

    Repeated RPC calls return recorded values. This does not validate session or
    execution state semantics. Requests outside the transcript fail explicitly.
    """

    def __init__(self) -> None:
        data = json.loads(files("lagrl.testing").joinpath("tool_transcript.json").read_text())
        create = data["create_request"]
        self.create_request = CreateSessionRequest(
            create["environment"], ResourceLimits(**create["limits"]), create["lease_seconds"]
        )
        session = data["session"]
        self.session = Session(
            **{
                **session,
                "limits": ResourceLimits(**session["limits"]),
                "state": SessionState(session["state"]),
            }
        )
        request = data["execute_request"]
        self.execute_request = ExecuteRequest(**{**request, "argv": tuple(request["argv"])})
        result = data["result"]
        self.result = ExecutionResult(
            **{
                **result,
                "reference": ExecutionReference(**result["reference"]),
                "state": ExecutionState(result["state"]),
                "stdout": result["stdout"].encode(),
                "stderr": result["stderr"].encode(),
                "failure_category": FailureCategory(result["failure_category"]),
            }
        )
        self.closed = False

    async def create_session(self, request: CreateSessionRequest) -> Session:
        if request != self.create_request:
            raise InvalidRequest("request is outside prerecorded fixture")
        return self.session

    async def execute(self, request: ExecuteRequest) -> ExecutionReference:
        if request != self.execute_request:
            raise InvalidRequest("request is outside prerecorded fixture")
        return self.result.reference

    async def get_execution(self, reference: ExecutionReference) -> ExecutionResult:
        if reference != self.result.reference:
            raise InvalidRequest("reference is outside prerecorded fixture")
        return self.result

    async def destroy_session(self, session_id: str) -> bool:
        if session_id != self.session.session_id:
            raise InvalidRequest("session is outside prerecorded fixture")
        return True

    async def close(self) -> None:
        self.closed = True

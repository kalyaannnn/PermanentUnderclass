"""Real isolation boundary. Host subprocess execution is never a fallback."""

from lagrl.sandbox.records import ExecuteRequest, ExecutionResult, Session


class GVisorExecutor:
    def __init__(self, runtime_path: str = "runsc") -> None:
        self.runtime_path = runtime_path

    async def run(self, session: Session, request: ExecuteRequest) -> ExecutionResult:
        """TODO[T09]: Execute argv in the intended gVisor-isolated session workspace.

        Inputs: live session, validated request; output: terminal result with bounded
        byte outputs, exit status, timings and failure category. Shape: scalar records.
        Preconditions: Linux, reviewed isolation policy and configured runsc available.
        Missing isolation raises IsolationUnavailable; never invoke argv on the host.
        Deadlines, cancellation and resource bounds must terminate isolated work.
        Question: Which filesystem and network capabilities can this command observe?
        Tests: tests/exercises/test_t09.py (opt-in). Docs: docs/sandbox.md.
        """
        raise NotImplementedError("TODO[T09] GVisorExecutor.run")

    async def release(self, session: Session) -> None:
        """TODO[T09]: Release only the isolation resources belonging to this session.

        Input: session workspace reference; output: None after cleanup. Scalar records.
        Preconditions: ownership verified. Repeated cleanup is idempotent; unavailable
        isolation raises IsolationUnavailable and foreign workspace references fail.
        No generic host command runner is permitted.
        Question: How is resource ownership proven before removal?
        Tests: tests/exercises/test_t09.py (opt-in). Docs: docs/sandbox.md.
        """
        raise NotImplementedError("TODO[T09] GVisorExecutor.release")

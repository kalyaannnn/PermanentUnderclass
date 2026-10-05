# Sandbox transport and required lifecycle

The four RPCs use `proto/lagrl/sandbox/proto/sandbox.proto`. The development server binds
only `127.0.0.1`, uses an ephemeral port by default, and bounds message bytes and concurrent
RPCs. Insecure transport is a local development choice; no remote security or isolation
claim is made. Client timeout applies per RPC and does not specify command lifetime.

| RPC | Request | Response |
|---|---|---|
| CreateSession | Allowlisted environment name, resource limits, lease duration | Session ID, lease times and workspace reference |
| Execute | Session ID, caller execution ID, argv, absolute Unix command deadline, combined output-byte limit | Execution reference promptly, before completion |
| GetExecution | Session/execution reference | State, bounded byte stdout/stderr, optional exit code, accepted/start/end times, failure category and truncation flag |
| DestroySession | Session ID | Idempotent release acknowledgement |

Thin handlers decode, delegate and encode. Domain errors map to INVALID_ARGUMENT,
NOT_FOUND, ALREADY_EXISTS, RESOURCE_EXHAUSTED or FAILED_PRECONDITION. Unfinished production
methods map to UNIMPLEMENTED with the TODO ID; this is an explicit error, not a fallback.

T08 owns these transitions and their concurrency control:

- Workspace state persists across executions within one session, and is released on
  destroy/lease expiry. Environment names must be allowlisted; requested budgets must
  be positive and no larger than server budgets.
- Duplicate identity is `(session_id, execution_id)`. A byte-for-byte identical request
  returns the same reference without another start. Changing argv, deadline or output
  limit while reusing the ID raises ExecutionConflict, including during concurrent calls.
- Accepted execution moves through queued/running into succeeded, failed, timed_out or
  cancelled. Nonzero exit is COMMAND_EXIT, runtime failure is INFRASTRUCTURE, elapsed
  command deadline is COMMAND_DEADLINE. A deadline already elapsed creates a terminal
  result without starting. stdout+stderr is bounded in bytes, with truncation observable.
- Bound sessions, lease duration, execution concurrency, argv size, resources and outputs.
  Command deadlines use an injected Unix clock. Clock jumps and platform runtime timers
  need explicit handling during T08/T09 implementation; monotonic elapsed timers should
  enforce actual run-time budgets after admission.
- At lease expiry, cancel and await owned work, release workspace and invalidate records.
  Command-deadline enforcement and lease sweeps must be invoked by the owning service
  loop when T08 is integrated; the current transport starts no lifecycle polling tasks.
- Destroy is idempotent for repeated or unknown IDs, but must not report release while
  cleanup is incomplete. Close stops admission and awaits cleanup. In-memory restart
  invalidates all sessions and execution records; no durable recovery is promised.

An RPC deadline means the caller lacks a response, not that the accepted command stopped.
After an ambiguous Execute failure, query GetExecution before deciding to retry. If the
reference is unknown, use the same execution identity and original arguments to resolve
ambiguity while the same service instance is known to be alive. After restart, the old
record cannot prove whether an external side effect occurred. There is no exactly-once
claim. Generated commands must never run on the host because isolation is unavailable.

T09 must use the intended gVisor backend on Linux, with reviewed filesystem/network
capabilities, resource budgets and bounded output collection. If runsc or required
isolation is unavailable, raise IsolationUnavailable. No host subprocess helper exists.
`RecordedSandbox` merely matches the packaged transcript; its repeated responses do not
test any of the transitions above and it never invokes commands.

Primary references: [gRPC AsyncIO](https://grpc.github.io/grpc/python/grpc_asyncio.html),
[gRPC deadlines](https://grpc.io/docs/guides/deadlines/),
[gVisor architecture](https://gvisor.dev/docs/architecture/),
[gVisor runsc quick start](https://gvisor.dev/docs/user_guide/quick_start/).

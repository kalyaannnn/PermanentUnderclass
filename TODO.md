# Learning roadmap

All newly created tasks are **unimplemented**. Support code and fixture smoke are
implemented; they do not satisfy these tasks. Stable IDs must stay attached to the
production functions. Update status here only after tests and review. Each command below
explicitly discovers exercise tests, which initially fail with the intended TODO ID.

| ID / status | Production functions | Learning objective and prerequisites | Acceptance command | Requirements |
|---|---|---|---|---|
| T01 — unimplemented | `group_relative_advantages`, `src/lagrl/training.py` | Population normalization, equal rewards, finite input; independent numeric task, read complete-group contract first | `uv run --locked pytest tests/exercises/test_t01.py` | Default CPU; no torch |
| T02 — unimplemented | `grpo_loss`, `src/lagrl/training.py` | Detached references, masked clipping, response means and gradients; T01 concepts (tests supply advantages directly) | `uv run --locked --extra training pytest tests/exercises/test_t02.py` | Optional CPU PyTorch |
| T03 — unimplemented | `assemble_trajectory`, `src/lagrl/rollout.py` | Event alignment and immutable behavior provenance; independent of numerics, read token contract | `uv run --locked pytest tests/exercises/test_t03.py` | Default CPU |
| T04 — unimplemented | `assemble_complete_group`, `admit_group`, `src/lagrl/rollout.py` | Complete identity set first, then inclusive all-span lag; T03 contract | `uv run --locked pytest tests/exercises/test_t04.py` | Default CPU |
| T05 — unimplemented | `evaluate_action_log_probs`, `training_step`, `src/lagrl/training.py` | Causal indexing independently, then one optimizer step/metrics; full step needs T01/T02 and T03/T04-valid data | `uv run --locked --extra training pytest tests/exercises/test_t05.py` | Optional CPU PyTorch; tiny local model |
| T06 — unimplemented | `GroupBuffer.put/get/close`, `run_coordination`, `src/lagrl/coordination.py` | FIFO ownership, capacity, backpressure, cancellation, graceful drain; buffer independent, coordinator needs T04/T05 interfaces | `uv run --locked pytest tests/exercises/test_t06.py` | Default CPU, controlled events |
| T07 — unimplemented | `validate_acknowledgement`, `PolicyPublisher.publish`, `src/lagrl/publication.py` | Ack identity first, then publication, partial failure and version visibility; metadata independent, runtime integration after T05/T06 | `uv run --locked pytest tests/exercises/test_t07.py` | Default CPU metadata tests; SGLang/NCCL/GPU integration later |
| T08 — unimplemented | `SessionManager.create_session/execute/get_execution/destroy_session/expire_sessions/expire_executions/close`, `src/lagrl/sandbox/lifecycle.py` | Bounded lifecycle, prompt acceptance, deduplication, command/lease deadlines, cleanup; independent with scripted isolated executor, read RPC contract | `uv run --locked pytest tests/exercises/test_t08.py` | Default CPU, injected clock/events; no isolation |
| T09 — unimplemented | `GVisorExecutor.run/release`, `src/lagrl/sandbox/gvisor.py` | Real capability-limited execution and owned cleanup; T08 semantics and prepared Linux gVisor runtime | `uv run --locked pytest tests/exercises/test_t09.py --run-isolation` | Explicit opt-in boundary tests; later integration needs Linux + runsc; no GPU |
| T10 — unimplemented | `run_variance_experiment`, `src/lagrl/variance.py` | Choose and evaluate a variance-control intervention only after baseline/diagnostics; T01–T08 working and agreed protocol | `uv run --locked pytest tests/exercises/test_t10.py` | Default CPU boundary test now; scientific tests/protocol to be specified before implementation |

Suggested order: T01 → T02 alongside T03 → T04 → T05; T06 buffer and T07 metadata can
start independently. T08 can proceed independently of all RL mathematics. Integrate
coordination/publication after their prerequisites, then T09. T10 comes last, with no
new estimator invented by this scaffold.

Each split function has local inputs, outputs, preconditions and edge cases in its
docstring. [Exercise notes](docs/exercises.md) provide conceptual questions and primary
references; [contracts](docs/contracts.md) define numerical and token acceptance criteria.
T09's initial tests cover unavailable isolation and foreign-workspace rejection only;
expand with controlled real workspace/deadline/resource tests once the environment is
prepared. T10's current test is only a boundary contract and must be expanded after
the learner chooses the experiment. Neither limited harness proves integration complete.

Promote reviewed passing test files to normal discovery and regression CI as described
in [README.md](README.md). Keep torch checks in an optional CPU training job and GPU /
real-isolation checks explicitly opt-in. Do not make the unfinished suite green with
skip/xfail, weakened assertions, or fake substitution.

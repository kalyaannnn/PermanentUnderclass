# Ownership and workflow

- The learner owns core algorithms and important state transitions. Codex owns packaging, interfaces, configuration, transport plumbing, fixtures, documentation, and tests.
- Do not implement learner-owned TODOs without an explicit request to implement or fix them. Do not hide equivalent solutions in helpers, fakes, test oracles, comments, or examples.
- When reviewing learner code, explain the violated invariant and give the smallest useful hint. Let the learner revise before providing a complete solution.
- Unfinished production functions must fail explicitly with their stable TODO ID.
- Keep default installation, smoke tests, and scaffold CI CPU-only.
- Clearly distinguish implemented behavior, fixture behavior, and unimplemented features.
- Preserve immutable sampler log-probabilities and policy-version provenance; trainer recomputations are separate data.
- Never execute generated commands on the host as a fallback for unavailable sandbox isolation.
- Report tests actually run, expected TODO failures, missing dependencies, and limitations honestly.

Changing exercise status belongs in `TODO.md`; setup commands belong in `README.md`.

# Data and numerical contracts

Frozen dataclasses and tuple fields provide immutable authored records. Type annotations
define field types; constructors intentionally do not implement T03/T04 validation.
Those production functions establish the alignment and completeness invariants. JSON
serialization is for trusted local records; it is not a schema-validating ingestion API.

## Token indexing

Use full-sequence, zero-based token coordinates, including prompt, generated actions,
tool observations and padding. For length N, `token_ids`, `token_roles`, `action_mask`
and `behavior_log_probs` all have length N. Index t describes the token **at t**.
In a causal model, logits at t-1 predict token t. Index 0 cannot be an action. Every
action has a finite sampled log-probability and mask=True; prompt/tool/padding positions
have mask=False and behavior value=None. A sampled EOS is an action if generated.

`PolicySpan(start, stop, version)` covers half-open [start, stop) action-only positions,
in full-sequence coordinates. Spans are ordered and disjoint; together they cover every
action exactly once. A tool observation splits spans even if the policy is unchanged.
Every version is nonnegative and refers to the weights actually acknowledged by the
sampler before those tokens were generated. `SamplingSettings` is held constant within
one trajectory; if settings change later, extend the record with settings spans first.
Tool events preserve argv, execution IDs, observations, outputs and failure evidence.

`behavior_log_probs` is immutable evidence of the **sampling distribution**, including
temperature/top-p behavior. `TrainingBatch.old_policy_log_probs` is the detached optimizer
reference evaluated under a declared old model. They may differ and are never aliases
for trainer writes. SGLang integration must verify what distribution its returned
log-probabilities describe. Changes to sampling settings or model weights cannot be
silently assigned a single provenance version.

Trajectory termination distinguishes completed responses, command timeout, cancellation
and infrastructure failure. Completed trajectories have finite rewards; failed or
cancelled trajectories have reward=None and do not enter a baseline complete group.
A valid task-level zero reward is 0.0. Later task specifications may define how a tool
failure becomes a valid completed response; infrastructure failure never becomes zero.

## Groups and policy lag

`GroupRequest` declares G>=2 distinct trajectory IDs for a task/group. T04 accepts
exactly those completed trajectories, with identities and finite rewards checked, and
orders them by request IDs. A constructor alone does not prove completeness. Missing,
extra or duplicate members and failed trajectories raise ValueError. Upstream T06 owns
waiting/cancellation and failure reporting; no fast-response selection is permitted.

Lag for each action span is current_version - sampled_version. T04 accepts a group
only when every span's lag lies in [0, max_policy_lag], inclusive. Future versions are
invalid; any overly stale span rejects the entire group. Log rejection rather than
silently reducing group size. This admission policy is an experiment setting, not an
importance-sampling correction.

## GRPO baseline

For a complete group of rewards r_i, population mean mu and population standard deviation
sigma, A_i = (r_i - mu)/(sigma + epsilon), epsilon>0. Equal rewards yield exact zeros.
For [0, 2] and epsilon=1, expected advantages are [-0.5, 0.5]. T01 returns plain floats;
T02 must detach both tensor advantages and old-policy log-probabilities.

For action-token probability ratio rho = exp(current_log_prob - old_policy_log_prob),
the loss contribution is -min(rho*A, clip(rho, 1-c, 1+c)*A), with default c=0.2.
KL coefficient is zero in this baseline. First average valid action-token contributions
within each response, then average those response losses. Prompt, padding and tool
observation positions never contribute; masked nonfinite values must not poison loss.
Every response must contain at least one valid action; reject empty action sequences.
Reject invalid shapes, non-bool masks, nonfinite valid inputs or clipping parameters.

With rho=1 and positive advantage, descent should increase the action log-probability;
with negative advantage, descent should decrease it. The test suite uses scalar expected
values and gradient signs, not a second vectorized loss implementation.

This contract specifies a clipped baseline; it is not claimed to be an unbiased
asynchronous objective. Correction for stale behavior-policy distributions is a separate
future objective extension requiring sampling-support and estimator assumptions. T10
selects no new estimator. The [DeepSeekMath paper](https://arxiv.org/abs/2402.03300)
motivates GRPO; the precise exercise conventions here are the project's specification.

## Weights and updates

`WeightSnapshot` identifies immutable bytes with a monotone integer version, artifact URI
and SHA-256 digest. `WeightAcknowledgement` identifies the worker, applied version and
digest. T07 commits a visible publication only after all required workers acknowledge
the same snapshot. A failed update may leave some workers on newer weights: provenance
must still report their actual applied versions. A retry must reconcile that state,
not invent an atomic global rollback. Same-version different-content requests are errors.

`TrainingResult` contains the updated step/version, finite scalar loss, valid-token count
and named metrics. T05 performs exactly one optimizer update; T07 owns publication.

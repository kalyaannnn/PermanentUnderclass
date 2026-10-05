# Exercise notes

Each TODO lives in its eventual production module. Implement the documented contract,
run its targeted tests, request review and revise before moving on. Tests contain tiny
hand-computable outcomes or event-driven observations; there is no reference algorithm.

## T01

Read [the numerical contract](contracts.md#grpo-baseline) first. Why does treating this
complete group as a population change the normalization? What happens when every reward
is equal? Epsilon is added after standard deviation, not inside the variance or square
root. [PyTorch std](https://docs.pytorch.org/docs/stable/generated/torch.std.html) documents
the correction setting; T01 itself requires only Python floats.

## T02

Which inputs should gradients reach? Why do response lengths affect a token-global mean?
Check masked observations, both signs of advantage and both clipping directions.
[PyTorch autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) and
[Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
describe gradient ownership. The project's clipping/reduction convention is in the
[contract](contracts.md#grpo-baseline); asynchronous correction is a different objective.

## T03

Which token does each log-probability describe? Does the policy change during a tool call?
Keep sampler evidence even when optimizer old-policy evaluation differs. Refer to
[token indexing](contracts.md#token-indexing) and eventually verify
[SGLang sampling parameters](https://docs.sglang.ai/basic_usage/sampling_params.html)
against the exact installed backend version before integration.

## T04

Can the fastest responses be representative when duration and reward correlate?
Which span determines admission of a mixed-version trajectory? Use the declared IDs
as the group contract. [Group and lag contract](contracts.md#groups-and-policy-lag).

## T05

Which causal logits predict each action, and which parameters should change in one step?
When should gradients be cleared? Result version is not publication acknowledgement.
Use [PyTorch optimizer documentation](https://docs.pytorch.org/docs/stable/optim.html)
and [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html).
Tests use a tiny local model, never a downloaded model.

## T06

Who owns each task and admitted group during cancellation? How are blocked producers
woken on close? Controlled events test observable progress; no scheduling oracle is
provided. [asyncio tasks and cancellation](https://docs.python.org/3/library/asyncio-task.html)
and [queues](https://docs.python.org/3/library/asyncio-queue.html) are useful primary references.

## T07

Does a completed transfer imply the sampler has switched weights? How can a partially
acknowledged update be retried without corrupting provenance? Start with CPU metadata
semantics, then verify [PyTorch distributed/NCCL](https://docs.pytorch.org/docs/stable/distributed.html),
[Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html), and the installed
[SGLang documentation](https://docs.sglang.ai/) before writing adapters.

## T08

What does a client know after a timeout? Which acceptance boundary deduplicates concurrent
requests? How do expiry and shutdown release resources once? Read [sandbox semantics](sandbox.md)
and [gRPC deadlines](https://grpc.io/docs/guides/deadlines/) before implementing transitions.
Inject controlled executor events and a manual clock into the production manager.

## T09

Which capabilities does the isolated command receive, and how is workspace ownership
verified at cleanup? Review [gVisor's security model](https://gvisor.dev/docs/architecture_guide/security/)
and [quick start](https://gvisor.dev/docs/user_guide/quick_start/). Host execution is never
an isolation fallback. Integration tests need explicit opt-in and a prepared Linux runtime.

## T10

Which diagnosed variance source should be controlled, and what is the comparison protocol?
Only select an experiment after baseline numerics, duration/lag diagnostics and reproducible
runs are established. Specify the hypothesis, metrics, sampling assumptions and statistical
acceptance tests first. The current boundary test checks rejection of empty data, not the
scientific result. No estimator or complete experiment design is supplied. Revisit the
[GRPO primary paper](https://arxiv.org/abs/2402.03300) and extend the objective contract only
after selecting relevant primary research for the specific experiment.

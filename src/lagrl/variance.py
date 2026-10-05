"""Reserved experiment boundary; the variance-control design is deliberately open."""

from dataclasses import dataclass

from lagrl.contracts import CompleteRolloutGroup


@dataclass(frozen=True)
class VarianceReport:
    experiment_name: str
    sample_count: int
    metrics: tuple[tuple[str, float], ...]


def run_variance_experiment(
    groups: tuple[CompleteRolloutGroup, ...], experiment_name: str, seed: int
) -> VarianceReport:
    """TODO[T10]: Run a later, explicitly chosen variance-control comparison.

    Inputs: fixed complete groups [M,G], nonempty experiment name and deterministic
    seed. Output: named report with sample_count=M and finite documented metrics.
    Preconditions: baseline, lag diagnostics and an agreed experiment protocol exist.
    Edge cases: empty data, incomplete groups or unknown experiment raise ValueError;
    preserve inputs and provenance. No estimator is selected by this scaffold.
    Question: Which measured source of variance would the chosen intervention change?
    Tests: tests/exercises/test_t10.py (boundary only until protocol is selected).
    Docs: docs/exercises.md#t10. Statistical acceptance criteria must precede coding.
    """
    raise NotImplementedError("TODO[T10] run_variance_experiment")

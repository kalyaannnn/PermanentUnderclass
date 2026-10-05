import pytest

from lagrl.variance import run_variance_experiment


def test_empty_experiment_data_rejected() -> None:
    # Statistical tests must be specified after the learner selects the experiment.
    with pytest.raises(ValueError):
        run_variance_experiment((), "not-selected", seed=0)

from dataclasses import replace

import pytest

from lagrl.contracts import PolicySpan, Termination
from lagrl.rollout import admit_group, assemble_complete_group
from lagrl.testing.fixtures import group_fixture, trajectory_fixture


def test_complete_group_reordered_by_declared_ids() -> None:
    fixture = group_fixture()
    assert assemble_complete_group(fixture.request, fixture.trajectories[::-1]) == fixture


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra", "wrong_task", "failure"])
def test_incomplete_or_invalid_group_rejected(case: str) -> None:
    fixture = group_fixture()
    a, b = fixture.trajectories
    cases = {
        "missing": (a,),
        "duplicate": (a, a),
        "extra": (a, b, trajectory_fixture("unexpected")),
        "wrong_task": (a, replace(b, task_id="other")),
        "failure": (a, replace(b, reward=None, termination=Termination.INFRA_FAILURE)),
    }
    with pytest.raises(ValueError):
        assemble_complete_group(fixture.request, cases[case])


def test_policy_lag_inclusive_boundary_and_oldest_span() -> None:
    group = group_fixture()
    assert admit_group(group, current_version=2, max_policy_lag=1) is True
    assert admit_group(group, current_version=2, max_policy_lag=0) is False
    assert admit_group(group, current_version=3, max_policy_lag=1) is False


def test_future_policy_version_rejected() -> None:
    group = group_fixture()
    future = replace(group.trajectories[1], policy_spans=(PolicySpan(2, 3, 1), PolicySpan(4, 5, 3)))
    with pytest.raises(ValueError):
        admit_group(replace(group, trajectories=(group.trajectories[0], future)), 2, 1)


def test_negative_lag_bound_rejected() -> None:
    with pytest.raises(ValueError):
        admit_group(group_fixture(), 2, -1)

import pytest

from lagrl.training import group_relative_advantages


def test_equal_rewards_are_exact_zero() -> None:
    assert group_relative_advantages([3.0, 3.0, 3.0]) == (0.0, 0.0, 0.0)


def test_population_std_and_epsilon_outside_std() -> None:
    assert group_relative_advantages([0.0, 2.0], epsilon=1.0) == pytest.approx((-0.5, 0.5))


def test_three_member_hand_computable_case() -> None:
    assert group_relative_advantages([1.0, 2.0, 3.0], epsilon=1.0) == pytest.approx(
        (-0.5505102572168219, 0.0, 0.5505102572168219)
    )


def test_negative_rewards() -> None:
    assert group_relative_advantages([-3.0, -1.0], epsilon=1.0) == pytest.approx((-0.5, 0.5))


@pytest.mark.parametrize("rewards", [[], [1.0], [0, float("nan")], [0, float("inf")]])
def test_invalid_group_rejected(rewards: list[float]) -> None:
    with pytest.raises(ValueError):
        group_relative_advantages(rewards)


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf")])
def test_invalid_epsilon_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError):
        group_relative_advantages([0, 2], epsilon)

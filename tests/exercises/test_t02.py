import math

import pytest

try:
    import torch
except ModuleNotFoundError as error:
    raise RuntimeError("T02 requires CPU PyTorch: uv sync --locked --extra training") from error

from lagrl.training import grpo_loss

pytestmark = pytest.mark.training


def test_mask_and_response_then_batch_reduction() -> None:
    current = torch.tensor(
        [[float("nan"), 0.0, float("nan"), float("nan")], [float("nan"), 0.0, 0.0, float("inf")]],
        requires_grad=True,
    )
    old = torch.tensor(
        [[float("nan"), 0.0, float("nan"), float("nan")], [float("nan"), 0.0, 0.0, float("inf")]],
        requires_grad=True,
    )
    advantages = torch.tensor([1.0, 3.0], requires_grad=True)
    mask = torch.tensor([[False, True, False, False], [False, True, True, False]])
    loss = grpo_loss(current, old, advantages, mask)
    assert loss.item() == pytest.approx(-2.0)
    loss.backward()
    assert current.grad[0, 1].item() == pytest.approx(-0.5)
    assert current.grad[1, 1].item() == pytest.approx(-0.75)
    assert current.grad[1, 2].item() == pytest.approx(-0.75)
    assert torch.equal(current.grad[~mask], torch.zeros(5))
    assert old.grad is None
    assert advantages.grad is None


@pytest.mark.parametrize(
    "ratio,advantage,expected,gradient_sign",
    [
        (1.0, 1.0, -1.0, -1),
        (1.0, -1.0, 1.0, 1),
        (1.5, 1.0, -1.2, 0),
        (0.5, -1.0, 0.8, 0),
        (0.5, 1.0, -0.5, -1),
        (1.5, -1.0, 1.5, 1),
    ],
)
def test_clipping_and_gradient_direction(
    ratio: float, advantage: float, expected: float, gradient_sign: int
) -> None:
    current = torch.tensor([[0.0, math.log(ratio)]], requires_grad=True)
    loss = grpo_loss(
        current,
        torch.zeros((1, 2)),
        torch.tensor([advantage]),
        torch.tensor([[False, True]]),
        0.2,
    )
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.sign(current.grad[0, 1]).item() == gradient_sign
    assert current.grad[0, 0].item() == 0.0


def test_empty_action_response_rejected() -> None:
    with pytest.raises(ValueError):
        grpo_loss(
            torch.zeros((2, 2)),
            torch.zeros((2, 2)),
            torch.ones(2),
            torch.tensor([[False, True], [False, False]]),
        )


@pytest.mark.parametrize("clip", [0.0, 1.0, -0.1, float("nan")])
def test_invalid_clip_rejected(clip: float) -> None:
    with pytest.raises(ValueError):
        grpo_loss(
            torch.zeros((1, 2)),
            torch.zeros((1, 2)),
            torch.ones(1),
            torch.tensor([[False, True]]),
            clip,
        )


def test_nonfinite_valid_action_rejected() -> None:
    with pytest.raises(ValueError):
        grpo_loss(
            torch.tensor([[0.0, float("nan")]]),
            torch.zeros((1, 2)),
            torch.ones(1),
            torch.tensor([[False, True]]),
        )


def test_shape_mismatch_and_nonbool_mask_rejected() -> None:
    with pytest.raises(ValueError):
        grpo_loss(
            torch.zeros((1, 2)),
            torch.zeros((1, 1)),
            torch.ones(1),
            torch.ones((1, 2), dtype=torch.bool),
        )
    with pytest.raises(ValueError):
        grpo_loss(torch.zeros((1, 1)), torch.zeros((1, 1)), torch.ones(1), torch.ones((1, 1)))

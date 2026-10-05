import math
from types import SimpleNamespace

import pytest

try:
    import torch
except ModuleNotFoundError as error:
    raise RuntimeError("T05 requires CPU PyTorch: uv sync --locked --extra training") from error

from lagrl.training import TrainingBatch, evaluate_action_log_probs, training_step

pytestmark = pytest.mark.training


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scores = torch.nn.Parameter(
            torch.tensor([[0.0, 0.0, 0.0], [math.log(3), 0.0, 0.0], [0.0, math.log(3), 0.0]])
        )

    def forward(self, token_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.scores[token_ids])


def test_causal_alignment_and_masking() -> None:
    model = TinyModel()
    values = evaluate_action_log_probs(
        model, torch.tensor([[1, 0, 2]]), torch.tensor([[False, True, True]])
    )
    assert tuple(values.shape) == (1, 3)
    assert values.detach().tolist()[0] == pytest.approx([0.0, math.log(0.6), math.log(1 / 3)])
    assert values.requires_grad


def test_one_training_update_and_input_preservation() -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    old = torch.tensor(
        [[0.0, 0.0, math.log(1 / 3)], [0.0, 0.0, math.log(1 / 3)]], requires_grad=True
    )
    before = old.detach().clone()
    batch = TrainingBatch(
        torch.tensor([[1, 0, 2], [1, 0, 1]]),
        torch.tensor([[False, False, True], [False, False, True]]),
        old,
        torch.tensor([0.0, 2.0]),
    )
    model.scores.grad = torch.full_like(model.scores, 999.0)
    result = training_step(model, optimizer, batch, step=3, policy_version=7, epsilon=1.0)
    assert result.step == 4
    assert result.policy_version == 8
    assert result.loss == pytest.approx(0.0, abs=1e-6)
    assert result.valid_action_tokens == 2
    assert model.scores[0, 1] > model.scores[0, 2]
    assert torch.equal(model.scores.detach()[1], torch.tensor([math.log(3), 0.0, 0.0]))
    assert old.grad is None
    assert torch.equal(old.detach(), before)
    assert all(math.isfinite(value) for _, value in result.metrics)


def test_first_token_cannot_be_action() -> None:
    with pytest.raises(ValueError):
        evaluate_action_log_probs(
            TinyModel(), torch.tensor([[1, 0]]), torch.tensor([[True, False]])
        )

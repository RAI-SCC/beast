import pytest
import torch
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss

from beast.evaluation import calculate_mae, calculate_mse


@pytest.mark.parametrize(
    "prediction,reference,target",
    [
        [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0)],
        [
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0.5, 0.5, 1.5]),
            torch.tensor(0.5),
        ],
        [a := torch.randn(3, 4, 5), b := torch.randn(3, 4, 5), l1_loss(a, b)],
    ],
)
def test_mae(prediction: Tensor, reference: Tensor, target: Tensor) -> None:
    mae = calculate_mae(prediction, reference)
    assert mae == pytest.approx(target)


@pytest.mark.parametrize(
    "prediction,reference,target",
    [
        [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0)],
        [
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0.5, 0.5, 1.5]),
            torch.tensor(0.25),
        ],
        [a := torch.randn(3, 4, 5), b := torch.randn(3, 4, 5), mse_loss(a, b)],
    ],
)
def test_mse(prediction: Tensor, reference: Tensor, target: Tensor) -> None:
    mae = calculate_mse(prediction, reference)
    assert mae == pytest.approx(target)

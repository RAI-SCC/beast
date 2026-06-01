import pytest
import torch
from torch.nn.functional import l1_loss, mse_loss

from beast.evaluation import mae, mse


@pytest.mark.parametrize(
    "prediction,reference,target,batch_dim",
    [
        [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), 0],
        [
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0.5, 0.6, 1.4]),
            torch.tensor(0.5),
            0,
        ],
        [
            a := torch.randn(3, 4, 5),
            b := torch.randn(3, 4, 5),
            l1_loss(a, b, reduction="none").mean(dim=0),
            0,
        ],
        [a, b, l1_loss(a, b, reduction="none").mean(dim=2), 2],
    ],
)
def test_mae(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    batch_dim: int,
) -> None:
    """Test the MAE calculation."""
    mae_val = mae(prediction, reference, batch_dimension=batch_dim)
    assert mae_val == pytest.approx(target)


@pytest.mark.parametrize(
    "prediction,reference,target,batch_dim",
    [
        [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), 0],
        [
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0.5, 0.9, 1.8]),
            torch.tensor(0.1),
            0,
        ],
        [
            a := torch.randn(3, 4, 5),
            b := torch.randn(3, 4, 5),
            mse_loss(a, b, reduction="none").mean(dim=0),
            0,
        ],
        [a, b, mse_loss(a, b, reduction="none").mean(dim=2), 2],
    ],
)
def test_mse(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    batch_dim: int,
) -> None:
    """Test the MSE calculation."""
    mse_val = mse(prediction, reference, batch_dimension=batch_dim)
    assert mse_val == pytest.approx(target)

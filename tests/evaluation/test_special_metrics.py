"""Tests from beast.evaluation.special_metrics."""

import pytest
import torch
from torch import Tensor

from beast.evaluation import calculate_crps
from beast.evaluation.special_metrics import _calculate_crps_locally


@pytest.mark.parametrize(
    "prediction, target",
    [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([1.5, 2.5])),
        (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor([0.0, 0.0])),
    ],
)
def test_calculate_crps_local_redirection(prediction: Tensor, target: Tensor) -> None:
    assert torch.allclose(
        calculate_crps(prediction, target), _calculate_crps_locally(prediction, target)
    ), "CRPS local calculation does not match expected local behavior."


@pytest.mark.parametrize(
    "prediction, target, result",
    [
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]]),
            torch.tensor([1.5, 2.5]),
            torch.tensor([1.0 / 6, 1.0 / 6]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        ),
        (
            torch.tensor([[1.0, 2.0] for _ in range(6)]),
            torch.tensor([1.5, 2.5]),
            torch.tensor([0.5, 0.5]),
        ),
        (
            torch.ones(10, 2, 3, 4, 5),
            torch.ones(2, 3, 4, 5) * 1.5,
            torch.ones(2, 3, 4, 5) * 0.5,
        ),
    ],
)
def test_calculate_crps_local_calculation(
    prediction: Tensor, target: Tensor, result: Tensor
) -> None:
    function_result = _calculate_crps_locally(prediction, target)
    assert torch.allclose(function_result, result), (
        "CRPS local calculation is incorrect."
    )


def test_calculate_crps_distributed() -> None:
    """
    This test should be run in a distributed environment with multiple processes.
    TODO: Implementation currently missing.
    """
    pass

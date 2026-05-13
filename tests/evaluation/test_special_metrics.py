"""Tests from beast.evaluation.special_metrics."""

import pytest
import torch
import torch.distributed as dist
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


@pytest.mark.dist()
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
def test_calculate_crps_distributed(
    prediction: Tensor, target: Tensor, result: Tensor
) -> None:
    """
    This test should be run in a distributed environment with multiple processes.
    TODO: Implementation currently missing.
    """
    assert dist.is_initialized(), "Distributed environment is not initialized."
    assert dist.get_world_size() > 1, (
        "This test should only run if there is more than 1 rank."
    )

    process_group = dist.group.WORLD
    assert process_group is not None, "Failed to get the default process group."
    local_rank = process_group.rank()
    world_size = process_group.size()

    if prediction.shape[0] % world_size != 0:
        pytest.skip(
            f"This parametrization of the test requires the world size to be a divisor of {prediction.shape[0]}."
        )

    local_slice = slice(
        local_rank * (prediction.shape[0] // world_size),
        (local_rank + 1) * (prediction.shape[0] // world_size),
    )
    if local_rank == world_size - 1:  # last rank takes the remainder
        local_slice = slice(
            local_rank * (prediction.shape[0] // world_size), prediction.shape[0]
        )
    local_prediction = prediction[local_slice, ...]

    local_result = calculate_crps(local_prediction, target, up_group=process_group)
    dist.barrier(process_group)  # Ensure all processes have finished their calculations

    assert torch.allclose(local_result, result), (
        f"CRPS distributed calculation is incorrect on rank {local_rank} / {world_size}."
    )

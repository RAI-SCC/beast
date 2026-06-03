"""Tests from beast.evaluation.special_metrics."""

import random

import pytest
import torch
import torch.distributed as dist
from torch import Tensor

from beast.evaluation import crps
from beast.evaluation.special_metrics import _calculate_crps_locally


@pytest.mark.parametrize(
    "prediction, target",
    [
        (torch.tensor([[[1.0, 2.0]]]), torch.tensor([[1.5, 2.5]])),
        (torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]), torch.tensor([[1.5, 2.5]])),
        (torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]]), torch.tensor([[0.0, 0.0]])),
    ],
)
def test_calculate_crps_local_redirection(
    prediction: Tensor, target: Tensor, device: torch.device
) -> None:
    prediction = prediction.to(device)
    target = target.to(device)
    assert torch.allclose(
        crps(prediction, target), _calculate_crps_locally(prediction, target)
    ), "CRPS local calculation does not match expected local behavior."


@pytest.mark.parametrize(
    "prediction, target, result",
    [
        (
            torch.tensor([[[1.0, 2.0]]]),
            torch.tensor([[1.5, 2.5]]),
            torch.tensor([0.5, 0.5]),
        ),
        (
            torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[2.0, 3.0]], [[1.5, 2.5]]]),
            torch.tensor([[1.5, 2.5]]),
            torch.tensor([1.0 / 12, 1.0 / 12]),
        ),
        (
            torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]]),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([0.0, 0.0]),
        ),
        (
            torch.tensor([[[1.0, 2.0]] for _ in range(6)]),
            torch.tensor([[1.5, 2.5]]),
            torch.tensor([0.5, 0.5]),
        ),
        (
            torch.ones(10, 2, 3, 4, 5),
            torch.ones(2, 3, 4, 5) * 1.5,
            torch.ones(3, 4, 5) * 0.5,
        ),
    ],
)
def test_calculate_crps_local_calculation(
    prediction: Tensor, target: Tensor, result: Tensor, device: torch.device
) -> None:
    prediction = prediction.to(device)
    target = target.to(device)
    result = result.to(device)

    function_result = _calculate_crps_locally(prediction, target)
    assert torch.allclose(function_result, result), (
        "CRPS local calculation is incorrect."
    )


@pytest.mark.dist()
@pytest.mark.parametrize(
    "prediction, target, result",
    [
        (
            torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[2.0, 3.0]], [[1.5, 2.5]]]),
            torch.tensor([[1.5, 2.5]]),
            torch.tensor([1.0 / 12, 1.0 / 12]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([0.0, 0.0]),
        ),
        (
            torch.tensor([[1.0, 2.0] for _ in range(6)]),
            torch.tensor([[1.5, 2.5]]),
            torch.tensor([0.5, 0.5]),
        ),
        (
            torch.ones(10, 2, 3, 4, 5),
            torch.ones(2, 3, 4, 5) * 1.5,
            torch.ones(3, 4, 5) * 0.5,
        ),
    ],
)
def test_calculate_crps_distributed(
    prediction: Tensor, target: Tensor, result: Tensor, device: torch.device
) -> None:
    prediction = prediction.to(device)
    target = target.to(device)
    result = result.to(device)

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

    local_result = crps(local_prediction, target, up_group=process_group)
    dist.barrier(process_group)  # Ensure all processes have finished their calculations

    assert torch.allclose(local_result, result), (
        f"CRPS distributed calculation is incorrect on rank {local_rank} / {world_size}."
    )


@pytest.mark.parametrize(
    "prediction, target, result",
    [
        (
            torch.ones(10, 2, 3, 4, 5),
            torch.ones(2, 3, 4, 5) * 1.5,
            torch.ones(3, 4, 5) * 0.5,
        ),
    ],
)
def test_calculate_crps_dimensions(
    prediction: Tensor, target: Tensor, result: Tensor, device: torch.device
) -> None:
    prediction = prediction.to(device)
    target = target.to(device)
    result = result.to(device)

    result_permutation = torch.randperm(target.ndim - 1, device=device)
    batch_dim = random.randint(0, target.ndim - 1)
    target_permutation = torch.tensor(
        (result_permutation + 1)[:batch_dim].tolist()
        + [0]
        + (result_permutation + 1)[batch_dim:].tolist(),
        device=device,
    )
    sample_dim = random.randint(0, target.ndim)
    prediction_permutation = torch.tensor(
        (target_permutation + 1)[:sample_dim].tolist()
        + [0]
        + (target_permutation + 1)[sample_dim:].tolist(),
        device=device,
    )
    batch_dim = (
        batch_dim if batch_dim < sample_dim else batch_dim + 1
    )  # Adjust batch_dim index for prediction permutation

    target_permuted = target.permute(*target_permutation)
    prediction_permuted = prediction.permute(*prediction_permutation)
    permuted_result = result.permute(*result_permutation)

    permuted_output = _calculate_crps_locally(
        prediction_permuted, target_permuted, sample_dim=sample_dim, batch_dim=batch_dim
    )

    print(
        f"shapes: prediction {prediction_permuted.shape}, target {target_permuted.shape}, result {permuted_result.shape}, output {permuted_output.shape}"
    )

    assert torch.allclose(permuted_output, permuted_result), (
        "Local CRPS calculation does not produce the expected result when dimensions are permuted."
    )


@pytest.mark.dist()
@pytest.mark.parametrize(
    "prediction, target, result",
    [
        (
            torch.ones(10, 2, 3, 4, 5),
            torch.ones(2, 3, 4, 5) * 1.5,
            torch.ones(3, 4, 5) * 0.5,
        ),
    ],
)
def test_calculate_crps_dimensions_distributed(
    prediction: Tensor, target: Tensor, result: Tensor, device: torch.device
) -> None:
    prediction = prediction.to(device)
    target = target.to(device)
    result = result.to(device)

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

    result_permutation = torch.randperm(target.ndim - 1, device=device)
    batch_dim = random.randint(0, target.ndim - 1)
    target_permutation = torch.tensor(
        (result_permutation + 1)[:batch_dim].tolist()
        + [0]
        + (result_permutation + 1)[batch_dim:].tolist(),
        device=device,
    )
    sample_dim = random.randint(0, target.ndim)
    prediction_permutation = torch.tensor(
        (target_permutation + 1)[:sample_dim].tolist()
        + [0]
        + (target_permutation + 1)[sample_dim:].tolist(),
        device=device,
    )
    batch_dim = (
        batch_dim if batch_dim < sample_dim else batch_dim + 1
    )  # Adjust batch_dim index for prediction permutation

    target_permuted = target.permute(*target_permutation).contiguous()
    prediction_permuted = local_prediction.permute(*prediction_permutation).contiguous()
    permuted_result = result.permute(*result_permutation).contiguous()

    permuted_output = crps(
        prediction_permuted,
        target_permuted,
        sample_dim=sample_dim,
        batch_dim=batch_dim,
        up_group=process_group,
    )
    dist.barrier(process_group)  # Ensure all processes have finished their calculations

    print(
        f"Rank {local_rank} / {world_size}: shapes: prediction {prediction_permuted.shape}, target {target_permuted.shape}, result {permuted_result.shape}, output {permuted_output.shape}"
    )

    assert torch.allclose(permuted_output, permuted_result), (
        "Distributed CRPS calculation does not produce the expected result when dimensions are permuted."
    )

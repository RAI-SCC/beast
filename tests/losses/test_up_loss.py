"""Tests for the UP loss and related functionality."""

import math
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch_blue.vi as vi
from torch import Tensor, nn

from beast.losses.up_loss import MeanStdPriorMatchingAggregation, UPAllGaussLoss


class DummyModule(nn.Module):
    """A dummy module to test the UP loss."""

    def __init__(self) -> None:
        super().__init__()

        self.parameter = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.parameter


def _make_dummy_vimodule(device: torch.device) -> nn.Module:
    """Helper function to create a dummy VIModule for testing."""
    model = DummyModule().to(device)
    vi.convert_to_vimodule(
        model,
        keep_weights=True,
        variational_distribution=vi.distributions.Normal(std=0.5),
    )
    return model


@pytest.mark.dist(2)
def test_split_forward(device: torch.device) -> None:
    """Test that the forward method of the UP loss correctly computes the data fitting term and prior matching term."""

    model = _make_dummy_vimodule(device)
    dataset_size = 10
    up_group = dist.group.WORLD
    up_loss = UPAllGaussLoss(model=model, dataset_size=dataset_size, up_group=up_group)

    assert up_loss.loss_weights is None, (
        "Loss weights should be None unless specified otherwise."
    )

    target = torch.tensor([1.0], device=device)
    samples = torch.tensor([[0.5], [1.5], [0.75], [1.25]], device=device)
    global_mean = samples.mean(dim=0)
    global_std = samples.std(dim=0)
    expected_data_fit = dataset_size * (
        (global_mean - target) ** 2 / (2 * global_std**2) + torch.log(global_std)
    )

    world_size = up_group.size() if up_group is not None else 1
    rank = up_group.rank() if up_group is not None else 0
    local_samples = samples[rank::world_size, ...]

    log_std_value = cast(nn.Parameter, model._parameter_log_std).item()
    mean_value = cast(nn.Parameter, model._parameter_mean).item()
    variance_value = math.exp(2 * log_std_value)
    expected_prior_matching = torch.tensor(
        0.5 * (mean_value**2 + variance_value - 2 * log_std_value - 1), device=device
    )

    data_fitting, prior_matching = up_loss.split_forward(local_samples, target)

    assert torch.allclose(data_fitting, expected_data_fit), (
        f"Rank {rank} / {world_size}: Data fitting term mismatch: {data_fitting} vs {expected_data_fit}"
    )
    assert torch.allclose(prior_matching, expected_prior_matching), (
        f"Rank {rank} / {world_size}: Prior matching term mismatch: {prior_matching} vs {expected_prior_matching}"
    )


@pytest.mark.parametrize("heat", [0.0, 0.5, 1.0])
def test_heat(device: torch.device, heat: float) -> None:
    """Test that the heat property of the UP loss correctly returns the value from the KL loss."""
    model = _make_dummy_vimodule(device)
    up_loss = UPAllGaussLoss(model=model, dataset_size=10, up_group=None, heat=heat)

    target = torch.tensor([1.0], device=device)
    samples = torch.tensor([[0.5], [1.5], [0.75], [1.25]], device=device)

    data_fit, prior_matching = up_loss.split_forward(samples, target)
    expected_loss = data_fit + heat * prior_matching
    calculated_loss = up_loss(samples, target)

    assert torch.allclose(calculated_loss, expected_loss), (
        f"Loss mismatch with heat={heat}: {calculated_loss} vs {expected_loss}"
    )


@pytest.mark.dist(2)
def test_aggregation(device: torch.device) -> None:
    """Test that the MeanStdPriorMatchingAggregation correctly aggregates statistics (and the corresponding gradients) and computes the prior matching term."""

    model = _make_dummy_vimodule(device)
    up_group = dist.group.WORLD
    world_size = up_group.size() if up_group is not None else 1
    rank = up_group.rank() if up_group is not None else 0

    samples = torch.tensor([[0.5], [1.5], [0.75], [1.25]], device=device)
    local_samples = samples[rank::world_size, ...].requires_grad_(True)

    parameter_mean = cast(nn.Parameter, model._parameter_mean)
    parameter_log_std = cast(nn.Parameter, model._parameter_log_std)

    mean, std, prior_matching, _, _ = MeanStdPriorMatchingAggregation.apply(
        local_samples,
        up_group,
        torch.zeros_like(parameter_mean),
        torch.zeros_like(parameter_log_std),
        parameter_mean,
        parameter_log_std,
        None,
    )

    global_mean = samples.mean(dim=0)
    global_std = samples.std(dim=0)

    log_std_value = parameter_log_std.item()
    mean_value = parameter_mean.item()
    variance_value = math.exp(2 * log_std_value)
    expected_prior_matching = torch.tensor(
        0.5 * (mean_value**2 + variance_value - 2 * log_std_value - 1), device=device
    )

    assert torch.allclose(mean, global_mean), f"Mean mismatch: {mean} vs {global_mean}"
    assert torch.allclose(std, global_std), f"Std mismatch: {std} vs {global_std}"
    assert torch.allclose(prior_matching, expected_prior_matching), (
        f"Prior matching term mismatch: {prior_matching} vs {expected_prior_matching}"
    )

    mean.backward()
    expected_grads_mean = torch.ones_like(local_samples) / samples.numel()
    assert torch.allclose(cast(Tensor, local_samples.grad), expected_grads_mean), (
        f"Gradient mismatch for mean: {local_samples.grad} vs {expected_grads_mean}"
    )

    _, std2, _, _, _ = MeanStdPriorMatchingAggregation.apply(
        local_samples,
        up_group,
        torch.zeros_like(parameter_mean),
        torch.zeros_like(parameter_log_std),
        parameter_mean,
        parameter_log_std,
        None,
    )  # this is necessary because you cannot backward through the same output multiple times, so we need to recompute the std for the second backward pass

    local_samples.grad = None  # reset gradients before backward pass for std
    std2.backward()
    expected_grads_std = (local_samples - global_mean) / (
        global_std * (samples.numel() - 1)
    )
    assert torch.allclose(cast(Tensor, local_samples.grad), expected_grads_std), (
        f"Gradient mismatch for std: {local_samples.grad} vs {expected_grads_std}"
    )

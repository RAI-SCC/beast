"""Tests for the UP loss and related functionality."""

import math
from typing import Optional, cast

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


@pytest.fixture(scope="module")
def dummy_module(device: torch.device) -> nn.Module:
    """Helper function to create a dummy VIModule for testing."""
    model = DummyModule().to(device)
    vi.convert_to_vimodule(
        model,
        keep_weights=True,
        variational_distribution=vi.distributions.Normal(std=0.5),
    )
    return model


@pytest.mark.dist(2)
@pytest.mark.parametrize("use_fmod", [True, False])
def test_data_fitting(
    dummy_module: vi.VIModule, use_fmod: bool, device: torch.device
) -> None:
    """Test data fitting term calculation."""

    dataset_size = 10
    up_group = dist.group.WORLD
    up_loss = UPAllGaussLoss(
        model=dummy_module,
        dataset_size=dataset_size,
        up_group=up_group,
        use_fmod=use_fmod,
    )

    target = torch.randn(3, 5, device=device)
    samples = torch.randn(6, 3, 5, device=device)

    sample_mean = samples.mean(dim=0)
    sample_std = samples.std(dim=0)

    ref_log_prob = (
        vi.distributions.Normal().log_prob_from_samples(target, samples).mean(dim=0)
    )

    if use_fmod:
        ref_log_prob = torch.fmod(ref_log_prob, 1.0)

    ref_loss = -1 * dataset_size * ref_log_prob.sum()
    loss = up_loss.data_fitting(sample_mean, sample_std, target)

    assert torch.allclose(ref_loss, loss)


@pytest.mark.dist(2)
def test_split_forward(dummy_module: vi.VIModule, device: torch.device) -> None:
    """Test that the forward method of the UP loss correctly computes the data fitting term and prior matching term."""

    dataset_size = 10
    up_group = dist.group.WORLD
    up_loss = UPAllGaussLoss(
        model=dummy_module, dataset_size=dataset_size, up_group=up_group
    )

    assert up_loss.feature_weights is None, (
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

    log_std_value = cast(nn.Parameter, dummy_module._parameter_log_std).item()
    mean_value = cast(nn.Parameter, dummy_module._parameter_mean).item()
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


@pytest.mark.dist(2)
@pytest.mark.parametrize(
    "feature_weights", [None, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])]
)
def test_feature_weights(
    feature_weights: Optional[Tensor], dummy_module: vi.VIModule, device: torch.device
) -> None:
    """Test feature weighting."""

    dataset_size = 10
    up_group = dist.group.WORLD
    loss = UPAllGaussLoss(
        model=dummy_module,
        dataset_size=dataset_size,
        up_group=up_group,
        feature_weights=feature_weights,
    )
    ref_loss = UPAllGaussLoss(
        model=dummy_module,
        dataset_size=dataset_size,
        up_group=up_group,
        feature_weights=None,
    )

    target = torch.randn(4, 5, 2, 2, device=device, generator=torch.manual_seed(13))
    samples = torch.randn(6, 4, 5, 2, 2, device=device, generator=torch.manual_seed(1))

    world_size = up_group.size() if up_group is not None else 1
    rank = up_group.rank() if up_group is not None else 0
    local_samples = samples[rank::world_size, ...]

    data_fit, prior_matching = loss.split_forward(local_samples, target)

    ref_data_fit_list, ref_prior_matching_list = [], []
    if feature_weights is None:
        feature_weights = torch.ones(5)
    for i in range(5):
        rdf, rpm = ref_loss.split_forward(
            local_samples[..., i, :, :], target[..., i, :, :]
        )
        ref_data_fit_list.append(rdf * feature_weights[i])
        ref_prior_matching_list.append(rpm)

    ref_data_fit = torch.stack(ref_data_fit_list, dim=-1)
    ref_prior_matching = torch.stack(ref_prior_matching_list, dim=-1)

    ref_data_fit = ref_data_fit.sum()

    assert torch.allclose(prior_matching, ref_prior_matching), (
        f"Rank {rank} / {world_size}: Prior matching term mismatch: {prior_matching} vs {ref_prior_matching}"
    )
    assert torch.allclose(data_fit, ref_data_fit), (
        f"Rank {rank} / {world_size}: Data fitting term mismatch: {data_fit} vs {ref_data_fit}"
    )


@pytest.mark.parametrize("heat", [0.0, 0.5, 1.0])
def test_heat(dummy_module: vi.VIModule, device: torch.device, heat: float) -> None:
    """Test that the heat property of the UP loss correctly returns the value from the KL loss."""
    up_loss = UPAllGaussLoss(
        model=dummy_module, dataset_size=10, up_group=None, heat=heat
    )

    target = torch.tensor([1.0], device=device)
    samples = torch.tensor([[0.5], [1.5], [0.75], [1.25]], device=device)

    data_fit, prior_matching = up_loss.split_forward(samples, target)
    expected_loss = data_fit + heat * prior_matching
    calculated_loss = up_loss(samples, target)

    assert torch.allclose(calculated_loss, expected_loss), (
        f"Loss mismatch with heat={heat}: {calculated_loss} vs {expected_loss}"
    )


@pytest.mark.dist(2)
def test_aggregation(dummy_module: vi.VIModule, device: torch.device) -> None:
    """Test that the MeanStdPriorMatchingAggregation correctly aggregates statistics (and the corresponding gradients) and computes the prior matching term."""

    up_group = dist.group.WORLD
    world_size = up_group.size() if up_group is not None else 1
    rank = up_group.rank() if up_group is not None else 0

    samples = torch.randn(8, 3, 4, 2, 2, device=device, generator=torch.manual_seed(7))
    local_samples = samples[rank::world_size, ...].requires_grad_(True)

    parameter_mean = cast(nn.Parameter, dummy_module._parameter_mean)
    parameter_log_std = cast(nn.Parameter, dummy_module._parameter_log_std)

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


@pytest.mark.dist(2)
def test_backward(dummy_module: vi.VIModule, device: torch.device) -> None:
    up_group = dist.group.WORLD
    world_size = up_group.size() if up_group is not None else 1
    rank = up_group.rank() if up_group is not None else 0

    samples = torch.tensor([[0.5], [1.5], [0.75], [1.25]], device=device)
    global_mean = samples.mean(dim=0)
    global_std = samples.std(dim=0)
    local_samples = samples[rank::world_size, ...].requires_grad_(True)

    parameter_mean = cast(nn.Parameter, dummy_module._parameter_mean)
    parameter_log_std = cast(nn.Parameter, dummy_module._parameter_log_std)

    mean, std, prior_matching, _, _ = MeanStdPriorMatchingAggregation.apply(
        local_samples,
        up_group,
        torch.zeros_like(parameter_mean),
        torch.zeros_like(parameter_log_std),
        parameter_mean,
        parameter_log_std,
        None,
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

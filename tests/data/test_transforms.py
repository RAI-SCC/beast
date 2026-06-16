"""Tests for beast.data.transforms"""

import torch

from beast.data.transforms import Downsample, DownsampleMeanStd, Identity


def test_identity(device: torch.device) -> None:
    x = torch.randn((3, 4), device=device)
    idtrf = Identity()

    assert idtrf(x) is x, "Identity operation did not return the same object"


def test_downsampling(device: torch.device) -> None:
    x = torch.randn(4, 4, device=device)
    dims = (-2,)

    trivial_downsampling1 = Downsample(factor=1, dims=dims)
    trivial_downsampling2 = DownsampleMeanStd(factor=1, dims=dims)
    x_target = x

    assert torch.allclose(x_target, trivial_downsampling1(x)), (
        "Trivial downsampling failed for class Downsample"
    )
    assert torch.allclose(x_target, trivial_downsampling2(x)[..., 0]), (
        "Trivial downsampling failed for class DownsampleMeanStd"
    )

    downsampling_by_two1 = Downsample(factor=2, dims=dims)
    downsampling_by_two2 = DownsampleMeanStd(factor=2, dims=dims)
    x_target = x.reshape(2, 2, 4).mean(dim=1)
    x_target_std = x.reshape(2, 2, 4).std(dim=1)

    assert torch.allclose(x_target, downsampling_by_two1(x)), (
        "Factor 2 downsampling failed for class Downsample"
    )
    assert torch.allclose(
        torch.stack((x_target, x_target_std), dim=-1), downsampling_by_two2(x)
    ), "Factor 2 downsampling failed for class DownsampleMeanStd"

    other_dims = (1,)

    downsampling_by_two1 = Downsample(factor=2, dims=other_dims)
    downsampling_by_two2 = DownsampleMeanStd(factor=2, dims=other_dims)
    x_target = x.reshape(4, 2, 2).mean(dim=2)
    x_target_std = x.reshape(4, 2, 2).std(dim=2)

    assert torch.allclose(x_target, downsampling_by_two1(x)), (
        "Factor 2 downsampling failed for class Downsample"
    )
    assert torch.allclose(
        torch.stack((x_target, x_target_std), dim=-1), downsampling_by_two2(x)
    ), "Factor 2 downsampling failed for class DownsampleMeanStd"

"""Tests for beast.data.transforms"""

import re

import pytest
import torch
import torchvision.transforms as trf

from beast.data.transforms import Downsample, DownsampleMeanStd


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


def test_exception(device: torch.device) -> None:
    x = torch.randn(4, 3, device=device)
    dims = (-1,)

    downsampler = Downsample(2, dims)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"not all dimensions at positions {dims} of tensor with shape {list(x.shape)} are divisible by {2}"
        ),
    ):
        downsampler(x)


def test_torchvision_integration(device: torch.device) -> None:
    x = torch.randn(4, 4, device=device)
    dims1 = (-1,)
    dims2 = (0,)

    downsampler1 = Downsample(2, dims1)
    random_flip = trf.RandomHorizontalFlip(p=0.5)
    downsampler2 = Downsample(2, dims2)

    full_transformation = trf.Compose([downsampler1, random_flip, downsampler2])

    output = full_transformation(x)

    assert list(output.shape) == [2, 2], (
        f"Composed transformations did not yield the correct dimensions ({list(output.shape)} instead of [2, 2])"
    )

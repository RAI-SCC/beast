from typing import Any

import pytest
import torch
import torch.distributed as dist


class TestDeviceFixture:
    """Test the device fixture provided by conftest.py."""

    def test_device_fixture(self, device: torch.device, request: Any) -> None:
        """Test that the device fixture returns a valid torch.device."""
        assert isinstance(device, torch.device)
        if torch.cuda.is_available() and request.config.getoption("--device") == "cuda":
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"

    def test_device_fixture_consistency(self, device: torch.device) -> None:
        """Test that the device fixture returns the same device across multiple calls."""
        device2 = device  # Get the device again
        assert device == device2  # Should be the same device instance

    def test_default_device(self, device: torch.device) -> None:
        """Test that the default device is set to CPU."""
        x = torch.tensor([1.0])
        assert x.device.type == "cpu"

    @pytest.mark.gpu
    def test_gpu_marker(self, device: torch.device) -> None:
        """Test that the gpu marker correctly selects a CUDA device."""
        assert device.type == "cuda"


class TestDistMarker:
    """Test the custom 'dist' marker for distributed tests."""

    @pytest.mark.dist()
    def test_dist_marker_no_ranks(self) -> None:
        assert (
            dist.get_world_size() > 1
        )  # This test should only run if there are more than 1 ranks

    @pytest.mark.dist(4)
    def test_dist_marker_with_ranks(self) -> None:
        """Test that the dist marker with rank specification works."""
        assert dist.is_initialized()
        assert dist.get_world_size() == 4

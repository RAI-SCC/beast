"""Tests for beast.layers.nice_window_transformer module"""

import torch

from beast.layers import nice_norm


class TestNiceNorm:
    """Test the Nice Norm."""

    def test_nice_init(self) -> None:
        """Test Nice Norm initialization.

        Assert that the output of the nice norm retains the shape of the input data.
        """
        batch_dim = 10
        seq_dim, channel_dim = 15, 20
        noise_dim = 32
        x = torch.randn(batch_dim, seq_dim, channel_dim)
        noise = torch.randn(batch_dim, noise_dim)
        norm = nice_norm.NiceNorm(normalized_dim=channel_dim, noise_dim=noise_dim)
        y = norm(x, noise)
        assert isinstance(norm, nice_norm.NiceNorm)

        assert y.shape == x.shape

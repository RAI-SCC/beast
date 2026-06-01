"""Tests for beast.layers.nice_window_transformer module"""

import pytest
import torch

from beast.comm import create_process_mesh_dict
from beast.layers import nice_window_transformer as nice
from jigsaw import utils_aggregate


class TestTransformerBlock:

    def test_transformer_init(self, device):
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        noise_dim = 16

        model = nice.TransformerBlock(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
        )

        assert isinstance(model, nice.TransformerBlock)

    def test_nice_transformer_forward(self, device):
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        noise_dim = 16
        seq_length = 32

        model = nice.TransformerBlock(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
        )
        # x dummy data
        X = torch.randn(1, seq_length, in_dim, device=device)
        y = model(X)
        assert isinstance(y, torch.Tensor)


class TestNiceWindowedTransformerBlock:

    def test_nice_transformer_init(self, device):
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        noise_dim = 16

        model = nice.NiceWindowedTransformerBlock(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
        )

        assert isinstance(model, nice.NiceWindowedTransformerBlock)

    def test_nice_transformer_forward(self, device):
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        noise_dim = 16
        seq_length = 32

        model = nice.NiceWindowedTransformerBlock(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
        )
        # x dummy data
        X = torch.randn(1, seq_length, in_dim, device=device)
        noise = torch.randn(1, noise_dim, device=device)
        y = model(X, noise)
        assert isinstance(y, torch.Tensor)

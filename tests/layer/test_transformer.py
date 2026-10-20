"""Tests for beast.layers.nice_window_transformer module"""

import pytest
import torch
import torch.distributed as dist
from jigsaw import parallel  # type: ignore

from beast.comm import create_process_mesh_dict
from beast.layers import transformer


class TestTransformer:
    def test_transformer_init(self, device: torch.device) -> None:
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4

        model = transformer.TransformerBlock(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            dropout=0.0,
            device=device,
        )

        assert isinstance(model, transformer.TransformerBlock)

    def test_transformer_forward(self, device: torch.device) -> None:
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        seq_length = 32

        model = transformer.TransformerBlock(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            dropout=0.0,
            device=device,
        )
        # x dummy data
        x = torch.randn(1, seq_length, in_dim, device=device)
        pred = model(x)
        assert isinstance(pred, torch.Tensor)


class TestNiceTransformer:
    def test_nice_transformer_init(self, device: torch.device) -> None:
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        noise_dim = 16

        model = transformer.NiceTransformer(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=0.0,
            device=device,
        )

        assert isinstance(model, transformer.NiceTransformer)

    def test_nice_transformer_forward(self, device: torch.device) -> None:
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        noise_dim = 16
        seq_length = 32

        model = transformer.NiceTransformer(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=0.0,
            device=device,
        )
        # x dummy data
        x = torch.randn(1, seq_length, in_dim, device=device)
        noise = torch.randn(1, noise_dim, device=device)
        y = model(x, noise)
        assert isinstance(y, torch.Tensor)


class TestDistributedTransformer:
    @pytest.mark.dist(4)
    def test_distributed_transformer_init(self, device: torch.device) -> None:
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        dropout = 0.0
        model = transformer.DistributedTransformer(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            dropout=dropout,
            local_rank=process_mesh_dict["DTP"].rank(),
            device=device,
            model_group=process_mesh_dict["DTP"],
            spatial_group=process_mesh_dict["JSpatial"],
            channel_group=process_mesh_dict["JChannel"],
            qk_norm=False,
            dtype=torch.float32,
        )

        assert isinstance(model, transformer.DistributedTransformer)

        batch_size, n_windows = 10, 10
        seq_local = 64

        x = torch.randn(
            batch_size * n_windows,
            seq_local,
            in_dim // process_mesh_dict["JChannel"].size(),
            device=device,
        )
        pred = model(x)
        assert isinstance(pred, torch.Tensor)


class TestDistributedNiceTransformer:
    @pytest.mark.dist(4)
    def test_distributed_nice_transformer_init(self, device: torch.device) -> None:
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        dropout = 0.0
        noise_dim = 32

        model = transformer.DistributedNICETransformer(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=dropout,
            local_rank=process_mesh_dict["DTP"].rank(),
            device=device,
            model_group=process_mesh_dict["DTP"],
            spatial_group=process_mesh_dict["JSpatial"],
            channel_group=process_mesh_dict["JChannel"],
            qk_norm=False,
            dtype=torch.float32,
        )

        assert isinstance(model, transformer.DistributedNICETransformer)

    @pytest.mark.dist(4)
    def test_distributed_transformer_forward(self, device: torch.device) -> None:
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        in_dim, embed_dim, out_dim = 256, 256, 256
        heads = 4
        dropout = 0.0
        noise_dim = 32  # global noise dim
        model = transformer.DistributedNICETransformer(
            in_dim=in_dim,
            embed_dim=embed_dim,
            out_dim=out_dim,
            heads=heads,
            noise_dim=noise_dim,
            dropout=dropout,
            local_rank=process_mesh_dict["DTP"].rank(),
            device=device,
            model_group=process_mesh_dict["DTP"],
            spatial_group=process_mesh_dict["JSpatial"],
            channel_group=process_mesh_dict["JChannel"],
            qk_norm=False,
            dtype=torch.float32,
        )

        seq_local = 10
        batch_size = 2
        n_windows = 10

        x = torch.randn(
            batch_size * n_windows,
            seq_local,
            in_dim // process_mesh_dict["JChannel"].size(),
            device=device,
        )
        noise = torch.randn(batch_size * n_windows, noise_dim, device=device)
        pred = model(x, noise)
        assert isinstance(pred, torch.Tensor)


class TestShardQkNorm:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("shard", [False, True])
    def test_replica_groups_and_gradient(self, device: torch.device, shard: bool) -> None:
        """A shared q/k norm is replicated over JSpatial and JChannel, a sharded
        one over JSpatial only: each JChannel shard keeps its own values, and its
        gradient is summed over JSpatial alone."""
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        spatial, channel = process_mesh_dict["JSpatial"], process_mesh_dict["JChannel"]
        in_dim = 256
        model = transformer.DistributedTransformer(
            in_dim=in_dim,
            embed_dim=in_dim,
            out_dim=in_dim,
            heads=4,
            dropout=0.0,
            local_rank=process_mesh_dict["DTP"].rank(),
            device=device,
            model_group=process_mesh_dict["DTP"],
            spatial_group=spatial,
            channel_group=channel,
            qk_norm=True,
            shard_qk_norm=shard,
            dtype=torch.float32,
        )
        attention = model.attention_block
        expected = [spatial] if shard else [spatial, channel]
        expected_ranks = [dist.get_process_group_ranks(g) for g in expected]
        for norm in (attention.q_norm, attention.k_norm):
            for p in norm.parameters():
                ranks = [
                    dist.get_process_group_ranks(g) for g in parallel.replica_groups(p)
                ]
                assert ranks == expected_ranks
        if not shard:
            return

        with torch.no_grad():
            attention.q_norm.weight.fill_(1.0 + channel.rank())
        x = torch.randn(4, 16, in_dim // channel.size(), device=device)
        model(x).sum().backward()
        assert torch.all(attention.q_norm.weight == 1.0 + channel.rank())
        grad = attention.q_norm.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        grads = [torch.empty_like(grad) for _ in range(spatial.size())]
        dist.all_gather(grads, grad.contiguous(), group=spatial)
        for g in grads:
            torch.testing.assert_close(g, grad, rtol=0, atol=0)

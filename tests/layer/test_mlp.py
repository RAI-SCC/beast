"""Tests for beast.layers.mlp module"""

import pytest

from beast.comm import create_process_mesh_dict
from beast.layers import mlp


class TestDistributedMLPNice:
    # @pytest.fixture(scope="session")
    # @pytest.mark.dist(4)
    # def process_mesh_dict(self):
    #     return create_process_mesh_dict(mesh_dims="1,1,1,2,2")

    @pytest.mark.dist(4)
    def test_distributed_mlp_nice_init(self, device, process_mesh_dict):
        process_mesh_dict = create_process_mesh_dict(mesh_dims="1,1,1,2,2")
        dtp_group, spatial_group, channel_group = (
            process_mesh_dict["DTP"],
            process_mesh_dict["JSpatial"],
            process_mesh_dict["JChannel"],
        )

        in_dim, hidden_dim, out_dim = 4, 6, 8
        model = mlp.DistributedMLPNICE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            bias=True,
            local_rank=channel_group.rank(),
            device=device,
            parallelism=channel_group.rank(),
            spatial_group=spatial_group,
            channel_group=channel_group,
        )

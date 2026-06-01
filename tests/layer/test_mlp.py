"""Tests for beast.layers.mlp module"""

import pytest
import torch

from beast.comm import create_process_mesh_dict
from beast.layers import mlp
from jigsaw import utils_aggregate


class TestDistributedMLPNice:
    @pytest.fixture(scope="session")
    def process_mesh_dict(self):
        return create_process_mesh_dict(mesh_dims=[1,1,1,2,2])

    @pytest.mark.dist(4)
    def test_distributed_mlp_nice_init(self, device, process_mesh_dict):
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
            spatial_group=spatial_group,
            channel_group=channel_group,
        )

        assert isinstance(model, mlp.DistributedMLPNICE)
        # collect weights
        weights_gathered = utils_aggregate.gather_results_as_list(
            model.linear1.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(weights_gathered[0], weights_gathered[2])
        assert torch.allclose(weights_gathered[1], weights_gathered[3])
        assert not torch.allclose(weights_gathered[0], weights_gathered[1])
        assert not torch.allclose(weights_gathered[2], weights_gathered[3])

        # Second linear layer: weights should be same across pairs of processes
        weights_gathered = utils_aggregate.gather_results_as_list(
            model.linear2.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(weights_gathered[0], weights_gathered[2])
        assert torch.allclose(weights_gathered[1], weights_gathered[3])
        assert not torch.allclose(weights_gathered[0], weights_gathered[1])
        assert not torch.allclose(weights_gathered[2], weights_gathered[3])

    @pytest.mark.dist(4)
    def test_distributed_mlp_nice_sync(self, device):
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 1, 2, 2])
        dtp_group, spatial_group, channel_group = (
            process_mesh_dict["DTP"],
            process_mesh_dict["JSpatial"],
            process_mesh_dict["JChannel"],
        )

        in_dim, hidden_dim, out_dim = 4, 8, 8
        model = mlp.DistributedMLPNICE(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            bias=True,
            local_rank=channel_group.rank(),
            device=device,
            spatial_group=spatial_group,
            channel_group=channel_group,
        )

        seq, local_dim = 12, in_dim

        torch.manual_seed(channel_group.rank())

        for _ in range(20):
            X = torch.randn(1, seq, local_dim, device=device).requires_grad_(True)
            pred = model(X)
            pred.backward(torch.randn_like(pred))

        # assert
        X_grad = utils_aggregate.gather_results_as_list(
            X.grad,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        
        # dx should be the same across channel groups
        assert torch.allclose(X_grad[0], X_grad[1])
        assert torch.allclose(X_grad[2], X_grad[3])
        
        W = utils_aggregate.gather_results_as_list(
            model.linear1.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(W[0], W[2])
        assert torch.allclose(W[1], W[3])
        
        W = utils_aggregate.gather_results_as_list(
            model.linear2.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(W[0], W[2])
        assert torch.allclose(W[1], W[3])
        

class TestDistributedMLPSync:
    @pytest.fixture(scope="session")
    def process_mesh_dict(self):
        return create_process_mesh_dict(mesh_dims=[1,1,1,2,2])

    @pytest.mark.dist(4)
    def test_distributed_mlp_sync_init(self, device, process_mesh_dict):
        dtp_group, spatial_group, channel_group = (
            process_mesh_dict["DTP"],
            process_mesh_dict["JSpatial"],
            process_mesh_dict["JChannel"],
        )

        in_dim, hidden_dim, out_dim = 4, 6, 8
        model = mlp.DistributedSynchronizedMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            bias=True,
            dropout=0.0,
            local_rank=channel_group.rank(),
            device=device,
            model_group=channel_group,
            sync_group=spatial_group,
        )

        assert isinstance(model, mlp.DistributedSynchronizedMLP)
        # collect weights
        weights_gathered = utils_aggregate.gather_results_as_list(
            model.linear1.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(weights_gathered[0], weights_gathered[2])
        assert torch.allclose(weights_gathered[1], weights_gathered[3])
        assert not torch.allclose(weights_gathered[0], weights_gathered[1])
        assert not torch.allclose(weights_gathered[2], weights_gathered[3])

        # Second linear layer: weights should be same across pairs of processes
        weights_gathered = utils_aggregate.gather_results_as_list(
            model.linear2.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(weights_gathered[0], weights_gathered[2])
        assert torch.allclose(weights_gathered[1], weights_gathered[3])
        assert not torch.allclose(weights_gathered[0], weights_gathered[1])
        assert not torch.allclose(weights_gathered[2], weights_gathered[3])

    @pytest.mark.dist(4)
    def test_distributed_mlp_sync(self, device):
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 1, 2, 2])
        dtp_group, spatial_group, channel_group = (
            process_mesh_dict["DTP"],
            process_mesh_dict["JSpatial"],
            process_mesh_dict["JChannel"],
        )

        in_dim, hidden_dim, out_dim = 4, 8, 8
        model = mlp.DistributedSynchronizedMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            bias=True,
            dropout=0.0,
            local_rank=channel_group.rank(),
            device=device,
            sync_group=spatial_group,
            model_group=channel_group,
        )

        seq, local_dim = 12//2, in_dim//2

        torch.manual_seed(channel_group.rank())

        for _ in range(20):
            X = torch.randn(1, seq, local_dim, device=device).requires_grad_(True)
            pred = model(X)
            pred.backward(torch.randn_like(pred))

        # assert
        X_grad = utils_aggregate.gather_results_as_list(
            X.grad,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        
        # dx should not be the same across channel groups
        assert not torch.allclose(X_grad[0], X_grad[1])
        assert not torch.allclose(X_grad[2], X_grad[3])
        
        W = utils_aggregate.gather_results_as_list(
            model.linear1.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(W[0], W[2])
        assert torch.allclose(W[1], W[3])
        
        W = utils_aggregate.gather_results_as_list(
            model.linear2.weight.data,
            process_mesh=process_mesh_dict,
            pm_key_row="JSpatial",
            pm_key_col="JChannel",
            pm_key_model="DTP",
        )
        assert torch.allclose(W[0], W[2])
        assert torch.allclose(W[1], W[3])


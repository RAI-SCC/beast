import pathlib

import pytest
import torch
import torch.distributed as dist
from small_models import SMALL_BEAST_OVERRIDES, small_beast_config

from beast.checkpoint import (
    is_checkpoint_writer,
    load_checkpoint,
    load_state_dict,
    save_checkpoint,
)
from beast.comm import create_process_mesh_dict, get_process_group, wrap_ddp
from beast.model import convert_jigsaw_vi, get_model
from beast.utils import set_all_seeds


class TestCheckpoint:
    def test_save_checkpoint(
        self,
        shared_tmp_path: pathlib.Path,
        device: torch.device,
    ) -> None:
        """Test to save and load a dummy model. If running distributed, assumes only ddp group, so a single checkpoint should be written."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)

        full_path = save_checkpoint(
            str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
        )
        dist.barrier()
        checkpoint_list = list(shared_tmp_path.iterdir())
        assert len(checkpoint_list) == 1

        checkpoint = load_checkpoint(full_path)

        assert isinstance(checkpoint["model_state_dict"], dict)
        assert isinstance(checkpoint["optimizer_state_dict"], dict)
        assert isinstance(checkpoint["epoch"], int)
        assert isinstance(checkpoint["step"], int)
        assert checkpoint["loss"] == loss_value

    @pytest.mark.dist(4)
    def test_dist_save_checkpoint(
        self,
        shared_tmp_path: pathlib.Path,
        device: torch.device,
    ) -> None:
        """Test to save and load a dummy model. DTP parallelism, should produce two checkpoint files."""
        create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)

        full_path = save_checkpoint(
            str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
        )
        dist.barrier()
        checkpoint_list = list(shared_tmp_path.iterdir())
        assert len(checkpoint_list) == 2
        checkpoint = load_checkpoint(full_path)

        assert isinstance(checkpoint["model_state_dict"], dict)
        assert isinstance(checkpoint["optimizer_state_dict"], dict)
        assert isinstance(checkpoint["epoch"], int)
        assert isinstance(checkpoint["step"], int)
        assert checkpoint["loss"] == loss_value

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "mesh_dims", [[1, 1, 2, 2], [1, 1, 4, 1], [2, 1, 2, 1], [1, 2, 2, 1]]
    )
    def test_one_writer_per_channel_shard(
        self, device: torch.device, mesh_dims: list[int]
    ) -> None:
        """Exactly one rank per JChannel shard writes, also with JSpatial > 1."""
        create_process_mesh_dict(mesh_dims=mesh_dims)
        writers = torch.zeros(mesh_dims[-1], device=device)
        writers[get_process_group("JChannel").rank()] = float(is_checkpoint_writer())
        dist.all_reduce(writers)
        assert writers.tolist() == [1.0] * mesh_dims[-1]

    @pytest.mark.gpu
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bayesian", [False, True])
    @pytest.mark.parametrize("mesh_dims", [[4, 1, 1, 1], [1, 1, 2, 2]])
    def test_save_beast_checkpoint(
        self,
        device: torch.device,
        shared_tmp_path: pathlib.Path,
        bayesian: bool,
        mesh_dims: list[int],
    ) -> None:
        """Test that initializing, saving, and loading beast works."""
        create_process_mesh_dict(mesh_dims=mesh_dims)
        channel_group = get_process_group("JChannel")
        spatial_group = get_process_group("JSpatial")
        dtp_group = get_process_group("DTP")
        up_group = get_process_group("SP")

        config = small_beast_config()
        model = get_model(
            "BEASTLING",
            **SMALL_BEAST_OVERRIDES,
            dropout=0 if bayesian else 1e-1,
            bayesian=bayesian,
            device=device,
            dtp_group=dtp_group,
            channel_group=channel_group,
            spatial_group=spatial_group,
        ).to(device)
        if bayesian:
            convert_jigsaw_vi(model, channel_group, spatial_group, up_group)
            model.return_log_probs = False

        dist.barrier()
        model = wrap_ddp(
            model,
            device_id=device.index,
            ddp_group=get_process_group("DP" if bayesian else "DDP"),
        )

        input_beastling = torch.rand(
            2,
            config["in_channels"] // channel_group.size(),
            config["height"],
            config["width"] // spatial_group.size(),
            device=device,
        )
        with torch.no_grad():
            set_all_seeds(42)
            if bayesian:
                output_native = model(input_beastling, samples=2)
            else:
                output_native = model(input_beastling)

        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)

        full_path = save_checkpoint(
            str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
        )
        dist.barrier()

        checkpoint_list = list(shared_tmp_path.iterdir())
        assert len(checkpoint_list) == channel_group.size()

        checkpoint = load_checkpoint(full_path)

        assert isinstance(checkpoint["model_state_dict"], dict)
        assert isinstance(checkpoint["optimizer_state_dict"], dict)
        assert isinstance(checkpoint["epoch"], int)
        assert isinstance(checkpoint["step"], int)
        assert checkpoint["loss"] == loss_value

        model_loaded, optimizer_loaded, start_epoch = load_state_dict(
            model,
            optimizer,
            checkpoint["model_state_dict"],
            checkpoint["optimizer_state_dict"],
            checkpoint["epoch"],
        )

        with torch.no_grad():
            set_all_seeds(42)
            if bayesian:
                output_loaded = model_loaded(input_beastling, samples=2)
            else:
                output_loaded = model_loaded(input_beastling)
        assert torch.allclose(output_loaded, output_native)

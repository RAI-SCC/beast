import pathlib

import pytest
import torch
import torch.distributed as dist

from beast.checkpoint import (
    load_checkpoint,
    load_state_dict,
    parse_checkpoint_path,
    parse_latest_model_state,
    save_checkpoint,
)
from beast.comm import create_process_mesh_dict, shutdown_comm
from beast.model import convert_jigsaw_vi, get_model
from beast.model.gb.beast_config import BEAST_CONFIGURATIONS
from beast.utils import set_all_seeds


class TestCheckpoint:
    @pytest.mark.dist(4)
    def test_save_checkpoint(
        self,
        shared_tmp_path: pathlib.Path,
        device: torch.device,
        process_mesh_dict: dict,
    ) -> None:
        """Test to save and load a dummy model."""
        print(shared_tmp_path)
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)
        shared_tmp_path = str(shared_tmp_path)

        full_path = save_checkpoint(
            shared_tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value
        )
        checkpoint = load_checkpoint(full_path)

        assert isinstance(checkpoint["model_state_dict"], dict)
        assert isinstance(checkpoint["optimizer_state_dict"], dict)
        assert isinstance(checkpoint["epoch"], int)
        assert isinstance(checkpoint["step"], int)
        assert checkpoint["loss"] == loss_value

    @pytest.mark.gpu
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bayesian", [False, True])
    def test_save_beast_checkpoint_ddp(
        self,
        device: torch.device,
        shared_tmp_path: pathlib.Path,
        bayesian: bool,
    ) -> None:
        """Test that initializing, saving, and loading beast works."""
        create_process_mesh_dict(mesh_dims=[1, 4, 1, 1, 1])

        config = BEAST_CONFIGURATIONS["BEASTLING"]["config"].copy()
        model = get_model("BEASTLING", bayesian=True, device=device)
        model = convert_jigsaw_vi(model)

        height = config["height"]
        width = config["width"]
        in_channels = config["in_channels"]
        input_beasting = torch.ones((1, in_channels, height, width), device=device)
        set_all_seeds(42)
        output_native = model(input_beasting)

        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)

        full_path = save_checkpoint(
            str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
        )
        dist.barrier()
        files = list(shared_tmp_path.iterdir())
        assert len(files) == 1

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

        set_all_seeds(42)
        output_loaded = model_loaded(input_beasting)
        assert torch.equal(output_loaded, output_native)

        shutdown_comm()

    @pytest.mark.gpu
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bayesian", [False, True])
    def test_save_beast_checkpoint_dtp(
        self,
        device: torch.device,
        shared_tmp_path: pathlib.Path,
        bayesian: bool,
        process_mesh_dict: dict,
    ) -> None:

        create_process_mesh_dict(mesh_dims=[1, 1, 1, 2, 2])
        """Test that initializing, saving, and loading distributed beast works."""
        jchannel_pg = process_mesh_dict["JChannel"]
        jspatial_pg = process_mesh_dict["JSpatial"]
        dtp_pg = process_mesh_dict["DTP"]

        config = BEAST_CONFIGURATIONS["BEASTLING"]["config"].copy()
        model = get_model("BEASTLING", bayesian=True, device=device)
        convert_jigsaw_vi(model)

        height = config["height"]
        width = config["width"]
        in_channels = config["in_channels"]
        input_beast = torch.ones((1, in_channels, height, width), device=device)
        set_all_seeds(42)
        output_native = model(input_beast)

        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)

        full_path = save_checkpoint(
            str(shared_tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
        )
        dist.barrier()
        assert len(list(shared_tmp_path.iterdir()) == 2)
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
        set_all_seeds(42)
        output_loaded = model_loaded(input_beast)
        assert torch.equal(output_loaded, output_native)

    def test_parse_checkpoint(self) -> None:
        """Test that parsing a checkpoint file name works."""
        fname = "cp-epoch7-step0-jc1.pt"
        m = parse_checkpoint_path(fname)
        assert set(m.keys()) == {
            "epoch",
            "step",
            "jchannel_rank",
            "fname",
        }
        assert m["epoch"] == 7
        assert m["step"] == 0
        assert m["jchannel_rank"] == 1
        assert m["fname"] == fname

        fname = "gibberish"
        m = parse_checkpoint_path(fname)
        assert m is None

    def test_parse_latest_model_state(self, shared_tmp_path: pathlib.Path) -> None:
        """Test that the latest model state is returned when multiple states are in a dir."""
        model = torch.nn.Linear(10, 10)
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2])
        shared_tmp_path = str(shared_tmp_path)
        save_checkpoint(
            shared_tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value
        )
        save_checkpoint(
            shared_tmp_path, model, optimizer, epoch=1, step=2, loss=loss_value
        )
        save_checkpoint(
            shared_tmp_path, model, optimizer, epoch=1, step=0, loss=loss_value
        )
        dist.barrier()
        print(f"path {shared_tmp_path}")
        m = parse_latest_model_state(shared_tmp_path)
        assert m["epoch"] == 1
        assert m["step"] == 2

    def test_load_state_dict(self, shared_tmp_path: pathlib.Path) -> None:
        """Test load state dict function."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = 1.0
        shared_tmp_path = str(shared_tmp_path)
        checkpoint_path = save_checkpoint(
            shared_tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value
        )
        m = load_checkpoint(checkpoint_path)
        model_1, optimizer_1, start_epoch = load_state_dict(
            model,
            optimizer,
            m["model_state_dict"],
            m["optimizer_state_dict"],
            m["epoch"],
        )
        assert torch.allclose(model_1.weight, model.weight)
        assert start_epoch == 1

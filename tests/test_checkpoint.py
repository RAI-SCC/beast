import pytest
import torch
import pathlib

from beast.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    parse_checkpoint,
    parse_latest_model_state,
    load_state_dict,
)
from beast.model.gb.beast_config import BEAST_CONFIGURATIONS
from beast.model.gb.beast_model import Beast
from beast.comm import create_process_mesh_dict, shutdown_comm


@pytest.fixture(scope="module")
def process_mesh_dict():
    process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 1, 2, 2])
    yield process_mesh_dict
    shutdown_comm()


class TestCheckpoint:
    @pytest.mark.dist(4)
    def test_save_checkpoint(
        self, tmp_path: pathlib.Path, device: torch.device, process_mesh_dict: dict
    ) -> None:
        """Test to save and load a dummy model."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)
        tmp_path = str(tmp_path)
        full_path = save_checkpoint(
            tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value
        )
        checkpoint = load_checkpoint(full_path)

        assert isinstance(checkpoint["model_state_dict"], dict)
        assert isinstance(checkpoint["optimizer_state_dict"], dict)
        assert isinstance(checkpoint["epoch"], int)
        assert isinstance(checkpoint["step"], int)
        assert checkpoint["loss"] == loss_value

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bayesian", [False, True])
    def test_save_beast_checkpoint(
        self,
        device: torch.device,
        tmp_path: pathlib.Path,
        bayesian: bool,
        process_mesh_dict: dict,
    ) -> None:
        """Test that initializing, saving, and loading beast works."""
        small_config = BEAST_CONFIGURATIONS["BEASTLING"]["config"].copy()
        small_config["bayesian"] = bayesian
        small_config["in_channels"] = small_config["in_channels"]
        model = Beast(**small_config, dropout=0 if bayesian else 1e-1, device=device)

        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)

        full_path = save_checkpoint(
            str(tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
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
    def test_save_dist_beast_checkpoint(
        self,
        device: torch.device,
        tmp_path: pathlib.Path,
        bayesian: bool,
        process_mesh_dict: dict,
    ) -> None:
        """Test that initializing, saving, and loading distributed beast works."""
        jchannel_pg = process_mesh_dict["JChannel"]
        jspatial_pg = process_mesh_dict["JSpatial"]
        dtp_pg = process_mesh_dict["DTP"]

        large_config = BEAST_CONFIGURATIONS["BEAST-JUPITER-0.7B-1.8K"]["config"].copy()
        large_config["bayesian"] = bayesian
        large_config["in_channels"] = large_config["in_channels"]
        model = Beast(
            **large_config,
            dropout=0 if bayesian else 1e-1,
            device=device,
            dtp_group=dtp_pg,
            channel_group=jchannel_pg,
            spatial_group=jspatial_pg,
        )

        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2], device=device)

        full_path = save_checkpoint(
            str(tmp_path), model, optimizer, epoch=0, step=1, loss=loss_value
        )
        checkpoint = load_checkpoint(full_path)

        assert isinstance(checkpoint["model_state_dict"], dict)
        assert isinstance(checkpoint["optimizer_state_dict"], dict)
        assert isinstance(checkpoint["epoch"], int)
        assert isinstance(checkpoint["step"], int)
        assert checkpoint["loss"] == loss_value

    def test_parse_checkpoint(self) -> None:
        """Test that parsing a checkpoint file name works."""
        fname = "cp-epoch7-step0-jc1.pt"
        m = parse_checkpoint(fname)
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
        m = parse_checkpoint(fname)
        assert m is None

    def test_parse_latest_model_state(self, tmp_path: pathlib.Path) -> None:
        """Test that the latest model state is returned when multiple states are in a dir."""
        model = torch.nn.Linear(10, 10)
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = torch.tensor([1.2])
        tmp_path = str(tmp_path)
        save_checkpoint(tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value)
        save_checkpoint(tmp_path, model, optimizer, epoch=1, step=2, loss=loss_value)
        save_checkpoint(tmp_path, model, optimizer, epoch=1, step=0, loss=loss_value)
        print(f"path {tmp_path}")
        m = parse_latest_model_state(tmp_path)
        assert m["epoch"] == 1
        assert m["step"] == 2

    def test_load_state_dict(self, tmp_path: pathlib.Path) -> None:
        """Test load state dict function."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = 1.0
        tmp_path = str(tmp_path)
        checkpoint_path = save_checkpoint(
            tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value
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

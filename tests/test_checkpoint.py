import pytest
import torch
import re

from beast.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    parse_checkpoint,
    parse_latest_model_state,
    load_state_dict,
    transplant_weights,
)
from beast.model import model_gb
from beast.comm import create_process_mesh_dict


def test_save_checkpoint(tmp_path):
    """Test to save and load a dummy model."""
    process_mesh = create_process_mesh_dict([1, 1, 1, 2, 2])
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_value = 1.2
    save_checkpoint(tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value)
    checkpoint = load_checkpoint(tmp_path)

    assert isinstance(checkpoint["model_state_dict"], dict)
    assert isinstance(checkpoint["optimizer_state_dict"], dict)
    assert isinstance(checkpoint["epoch"], int)
    assert isinstance(checkpoint["step"], int)
    assert checkpoint["ep_group_rank"] == process_mesh["Expert"].rank()
    assert checkpoint["channel_group_rank"] == process_mesh["JChannel"].rank()
    assert checkpoint["loss"] == loss_value


@pytest.mark.dist(4)
# TODO: parameterize
def test_save_beast_checkpoint(device, tmp_path, bayesian):
    """Test that initializing, saving, and loading beast works."""
    process_mesh = create_process_mesh_dict([1, 1, 1, 2, 2])
    model = model_gb.Bellbeast(
        xlat=720,
        xlon=1440,
        in_channels=80,
        embed_dim=60,
        out_channels=80,
        heads=6,
        kernel_size=2,
        window_size_outer=6,
        window_size_inner=6,
        n_full_blocks=1,
        noise_dim=32,
        dtp_rank=process_mesh["DTP"].rank(),
        device=device,
        dtp_group=process_mesh["DTP"],
        spatial_group=process_mesh["JSpatial"],
        channel_group=process_mesh["JChannel"],
        ep_group=process_mesh["Expert"],
        up_group=process_mesh["SP"],
        bayesian=bayesian,
        parallelism=process_mesh["DTP"].size(),
        dtype=torch.float32,
        flash_attn=True,
        nice_norm=True,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    loss_value = 1.2
    save_checkpoint(tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value)
    checkpoint = load_checkpoint(tmp_path)

    assert isinstance(checkpoint["model_state_dict"], dict)
    assert isinstance(checkpoint["optimizer_state_dict"], dict)
    assert isinstance(checkpoint["epoch"], int)
    assert isinstance(checkpoint["step"], int)
    assert checkpoint["ep_group_rank"] == process_mesh["Expert"].rank()
    assert checkpoint["channel_group_rank"] == process_mesh["JChannel"].rank()
    assert checkpoint["loss"] == loss_value


def test_parse_checkpoint():
    fname = "cp-epoch7-step0-e0-jc1.pt"
    m = parse_checkpoint(fname)
    assert set(m.keys()) == {"epoch", "step", "expert_rank", "jchannel_rank", "fname"}
    assert m["epoch"] == 7
    assert m["step"] == 0
    assert m["expert_rank"] == 0
    assert m["jchannel_rank"] == 1
    assert m["fname"] == fname

    fname = "gibberish"
    m = parse_checkpoint(fname)
    assert m is None


def test_parse_latest_model_state(tmp_path):
    """Test that the latest model state is returned when multiple states are in a dir."""
    model = torch.nn.Linear(10, 10)
    process_mesh = create_process_mesh_dict([1, 1, 1, 2, 2])
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_value = 1.2
    save_checkpoint(tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value)
    save_checkpoint(tmp_path, model, optimizer, epoch=1, step=2, loss=loss_value)
    save_checkpoint(tmp_path, model, optimizer, epoch=1, step=0, loss=loss_value)

    m = parse_latest_model_state(tmp_path)
    assert m["epoch"] == 1
    assert m["step"] == 2


def test_load_state_dict(tmp_path):
    """Test load state dict function"""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_value = 1.0
    save_checkpoint(tmp_path, model, optimizer, epoch=0, step=1, loss=loss_value)
    m = load_checkpoint(tmp_path)
    model_1, optimizer_1, start_epoch = load_state_dict(
        model, optimizer, m["model_state_dict"], m["optimizer_state_dict"], m["epoch"]
    )
    assert torch.allclose(model_1.weight, model.weight)
    assert start_epoch == 1
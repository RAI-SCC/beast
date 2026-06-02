import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import re
import os

from beast.comm import get_process_group


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int | None,
    loss: float,
):
    """
    Save checkpoint to file.

    If there is no channel_group defined, this assumes that the model is
    not distributed. Model will save on global rank = 0.

    Parameters
    ----------
    checkpoint_path: str
        Name of checkpoint
    model: torch.nn.Module
        Model to save
    optimizer: torch.optim
        Optimizer to save
    epoch: int
        Current epoch
    step: int
        Current step within batches
    loss: torch.Tensor
        Average loss value
    """

    try:
        channel_group = model.channel_group
    except Exception:
        channel_group = None

    if channel_group is None:
        channel_group_rank = -1
        ddp_rank = dist.get_rank(group=None)
    else:
        channel_group_rank = channel_group.rank()
        ddp_rank = get_process_group("DDP").rank()

    cp = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "channel_group_rank": channel_group_rank,
        "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
        "format": 1,
    }

    model_name = (
        f"cp-epoch{epoch}-step{step}"
        if channel_group is None
        else f"cp-epoch{epoch}-step{step}-jc{channel_group.rank()}"
    )

    if ddp_rank == 0:
        torch.save(cp, f"{checkpoint_path}/{model_name}.pt")

    return f"{checkpoint_path}/{model_name}.pt"


def load_checkpoint(
    checkpoint_path: str, map_location="cpu"
) -> tuple[dict, dict, int, int | None, int, int, float]:
    """Loads a checkpoint and returns the model state dict, optimizer state dict, scheduler state dict, epoch, step, and channel group rank."""
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    return checkpoint


def parse_checkpoint(fname: str) -> dict:
    """Parse epochs, step, and channel rank from the checkpoint name.

    Parameters
    ----------
    fname: str
        Name of checkpoint file. Should be formatted as f"cp-epoch{num}-step{num}-jc{num}.pt"
        If the model is not distributed, the file name is just f"cp-epoch{num}-step{num}.pt"

    Returns
    -------
    dictionary of epoch, step, channel rank, checkpoint file name.
    """
    if "jc" in fname:
        m = re.match(r"cp-epoch(\d+)(?:-step(\d+))?-jc(\d+)\.pt$", fname)
        if not m:
            return m
        epoch, step, jchannel_rank = (
            int(m.group(1)),
            int(m.group(2)) if m.group(2) else -1,
            int(m.group(3)),
        )
    else:
        m = re.match(r"cp-epoch(\d+)(?:-step(\d+))?.pt$", fname)
        if not m:
            return m
        epoch, step, jchannel_rank = (
            int(m.group(1)),
            int(m.group(2)) if m.group(2) else -1,
            -1,
        )

    return {
        "epoch": epoch,
        "step": step,
        "jchannel_rank": jchannel_rank,
        "fname": fname,
    }


def parse_latest_model_state(directory: str) -> dict:
    """Parse the latest model state and return checkpoint."""
    end_of_epoch_directory = os.path.join(directory)
    path_base = end_of_epoch_directory
    checkpoints = [
        fname for fname in os.listdir(end_of_epoch_directory) if fname.endswith(".pt")
    ]
    print(f"list of checpoints {checkpoints}")
    local_checkpoint = ""
    try:
        jc_group_rank = get_process_group("JChannel").rank()
    except Exception:
        jc_group_rank = -1

    parsed = []
    for fname in checkpoints:
        info = parse_checkpoint(fname)
        if info is None:
            continue
        if info["jchannel_rank"] != jc_group_rank:
            continue
        parsed.append(info)

    if parsed:
        latest = max(parsed, key=lambda x: (x["epoch"], x["step"]))
        local_checkpoint = latest["fname"]

    full_path = os.path.join(path_base, local_checkpoint)

    checkpoint = load_checkpoint(full_path)

    return checkpoint


def load_state_dict(
    model,
    optimizer,
    model_state_dict,
    optimizer_state_dict,
    last_epoch,
):
    # DDP checks
    is_state_dict_ddp = any(k.startswith("module.") for k in model_state_dict.keys())
    if isinstance(model, DDP) and not is_state_dict_ddp:
        model_state_dict = {f"module.{k}": v for k, v in model_state_dict.items()}
    elif not isinstance(model, DDP) and is_state_dict_ddp:
        model_state_dict = {
            k.replace("module.", "", 1): v for k, v in model_state_dict.items()
        }

    try:
        model.load_state_dict(model_state_dict)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        if last_epoch is not None:
            start_epoch = last_epoch + 1
    except Exception as err:
        if dist.get_rank() == 0:
            print(err)
            print("Model cannot be loaded from checkpoint. Reinitializing model.")

        start_epoch = 0

    if optimizer is None:
        return model
    else:
        return model, optimizer, start_epoch

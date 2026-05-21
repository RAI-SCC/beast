import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import re
import os

from gb.comm import get_process_group


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int | None,
    loss: float,
):

    channel_group = get_process_group("JChannel")
    ep_group = get_process_group("Expert")

    # rank = dist.get_rank()
    ddp_rank = get_process_group("DDP").rank()

    if ddp_rank == 0:
        cp = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "ep_group_rank": ep_group.rank(),
            "channel_group_rank": channel_group.rank(),
            "loss": loss.item(),
            "format": 1,
        }
        model_name = (
            f"cp-epoch{epoch}-step{step}-e{ep_group.rank()}-jc{channel_group.rank()}"
        )

        torch.save(cp, checkpoint_path + f"/{model_name}.pt")


def load_checkpoint(
    checkpoint_path: str, map_location="cpu"
) -> tuple[dict, dict, int, int | None, int, int, float]:
    """Loads a checkpoint and returns the model state dict, optimizer state dict, scheduler state dict, epoch, step, expert group rank, and channel group rank."""
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    epoch = checkpoint["epoch"]
    step = checkpoint.get("step", None)
    ep_group_rank = checkpoint["ep_group_rank"]
    channel_group_rank = checkpoint["channel_group_rank"]
    loss = checkpoint["loss"]

    return (
        model_state_dict,
        optimizer_state_dict,
        epoch,
        step,
        ep_group_rank,
        channel_group_rank,
        loss,
    )


def parse_checkpoint(fname):
    m = re.match(r"cp-epoch(\d+)(?:-step(\d+))?-e(\d+)-jc(\d+)\.pt$", fname)
    if not m:
        return None
    return {
        "epoch": int(m.group(1)),
        "step": int(m.group(2)) if m.group(2) else -1,
        "expert_rank": int(m.group(3)),
        "jchannel_rank": int(m.group(4)),
        "fname": fname,
    }


def parse_latest_model_states(directory):
    end_of_epoch_directory = os.path.join(directory, "checkpoints")
    path_base = end_of_epoch_directory
    checkpoints = [
        fname for fname in os.listdir(end_of_epoch_directory) if fname.endswith(".pt")
    ]
    local_checkpoint = ""
    latest_step = -1
    ep_group_rank = get_process_group("Expert").rank()
    jc_group_rank = get_process_group("JChannel").rank()

    parsed = []
    for fname in checkpoints:
        info = parse_checkpoint(fname)
        if info is None:
            continue
        if info["expert_rank"] != ep_group_rank:
            continue
        if info["jchannel_rank"] != jc_group_rank:
            continue
        parsed.append(info)

    if parsed:
        latest = max(parsed, key=lambda x: (x["epoch"], x["step"]))
        local_checkpoint = latest["fname"]
        latest_step = latest["step"] if latest["step"] != -1 else None

    full_path = os.path.join(path_base, local_checkpoint)
    print(f"Rank {dist.get_rank()}: Loading checkpoint from {full_path}")

    (
        model_state_dict,
        optimizer_state_dict,
        epoch,
        step,
        ep_group_rank,
        channel_group_rank,
        loss,
    ) = load_checkpoint(full_path)

    if not latest_step == -1:
        epoch -= 1

    return (
        model_state_dict,
        optimizer_state_dict,
        epoch,
        step,
        ep_group_rank,
        channel_group_rank,
        loss,
    )


def load_state_dict(
    model,
    optimizer,
    model_state_dict,
    optimizer_state_dict,
    last_epoch,
):
    # DDP checks
    is_state_dict_ddp = any(k.startswith('module.') for k in model_state_dict.keys()) 
    if isinstance(model, DDP) and not is_state_dict_ddp:
        model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
    elif not isinstance(model, DDP) and is_state_dict_ddp:
        model_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict.items()}
        
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

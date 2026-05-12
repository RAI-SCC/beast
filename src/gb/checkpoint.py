import torch
import torch.distributed as dist

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
    # checkpoint_path = os.path.join(results_path, "checkpoints")
    # if dist.get_rank() == 0:
    #     os.makedirs(checkpoint_path, exist_ok=True)
    # dist.barrier()

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
            "loss": loss,
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
    print(f"checkpoint_path = {checkpoint_path}")
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


def load_latest_model_states(directory):
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
    model_ddp,
    optimizer,
    model_state_dict,
    optimizer_state_dict,
    last_epoch,
):
    try:
        model_ddp.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = last_epoch + 1
    except Exception as err:
        if dist.get_rank() == 0:
            print(err)
            print("Model cannot be loaded from checkpoint. Reinitializing model.")
        start_epoch = 0

    return model_ddp, optimizer, start_epoch


def transplant_weights(model, state_dict_path):
    if dist.get_rank() == 0:
        print(f"Transplanting weights from {state_dict_path}")

    # this is not very nice, but how to change?
    model_state_dict, _, _, _, _, _, _ = load_latest_model_states(state_dict_path)

    try:
        state_dict = {}
        for key in model_state_dict.keys():
            state_dict[key[7:]] = model_state_dict[key]

        model.load_state_dict(state_dict)
    except Exception as err:
        if dist.get_rank() == 0:
            print(err)
            print("Model cannot be loaded from checkpoint. Reinitializing model.")

    return model

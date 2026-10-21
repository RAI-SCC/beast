"""Utilities for saving and loading model checkpoints and state_dicts."""

import math
import re
import warnings
from typing import Any, Tuple, cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from beast.comm import get_process_group


def is_checkpoint_writer() -> bool:
    """Return whether this rank writes its JChannel shard's checkpoint file.

    Every rank in ``DDP`` (``DP x SP``) x ``JSpatial`` holds the same shard, so
    exactly one of them, rank 0 in both groups, writes it. Without a process
    mesh, global rank 0 writes.

    Returns
    -------
    bool
        True on the writing rank.
    """
    if not dist.is_initialized():
        return True
    try:
        ddp_group = get_process_group("DDP")
        spatial_group = get_process_group("JSpatial")
    except ValueError:
        return dist.get_rank() == 0
    return ddp_group.rank() == 0 and spatial_group.rank() == 0


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    step: int | None,
    loss: float,
) -> str:
    """Save a checkpoint named ``cp-epoch{epoch}[-step{step}][-jc{rank}].pt``.

    Without a ``JChannel`` process group the model is assumed to be
    non-distributed and is written by global rank 0. Otherwise each channel
    rank writes its own shard, from the rank 0 of the ``DDP`` group.

    Parameters
    ----------
    checkpoint_path : str
        Directory to write the checkpoint to.
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer or None
        Optimizer to save.
    epoch : int
        Current epoch.
    step : int or None
        Current step (batch). Omitted from the file name if None or 0.
    loss : float
        Average loss value.

    Returns
    -------
    str
        Path of the checkpoint file, on every rank.
    """
    try:
        channel_group: dist.ProcessGroup | None = get_process_group("JChannel")

    except ValueError:
        channel_group = None

    channel_group_rank = -1 if channel_group is None else channel_group.rank()

    # Ensure loss is a plain float to avoid tensor attributes on save
    loss_value = float(loss) if isinstance(loss, (int, float, torch.Tensor)) else loss
    cp = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
        "epoch": epoch,
        "step": step,
        "channel_group_rank": channel_group_rank,
        "loss": loss_value,
        "format": 1,
    }
    step_str = f"-step{step}" if step else ""
    channel_str = f"-jc{channel_group.rank()}" if channel_group else ""
    model_name = f"cp-epoch{epoch}" + step_str + channel_str

    if is_checkpoint_writer():
        torch.save(
            cp,
            f"{checkpoint_path}/{model_name}.pt",
            _use_new_zipfile_serialization=False,
        )
    return f"{checkpoint_path}/{model_name}.pt"


def load_checkpoint(
    checkpoint_path: str, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.
    map_location : str or torch.device, default: "cpu"
        Where to remap the stored tensors, as in :func:`torch.load`.

    Returns
    -------
    dict[str, Any]
        The checkpoint, with the keys ``model_state_dict``,
        ``optimizer_state_dict``, ``epoch``, ``step``, ``channel_group_rank``,
        ``loss`` and ``format``.
    """
    return cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location=map_location, weights_only=False),
    )


def parse_checkpoint_path(fname: str) -> dict[str, Any] | None:
    """Parse epoch, step and channel rank from a checkpoint file name.

    Parameters
    ----------
    fname : str
        File name of the form ``cp-epoch{n}[-step{n}][-jc{n}].pt``.

    Returns
    -------
    dict[str, Any] or None
        The keys ``epoch``, ``step``, ``jchannel_rank`` and ``fname``, with
        ``math.inf`` for a missing step or channel rank. None, with a warning,
        if ``fname`` does not match the pattern.
    """
    m = re.match(r"cp-epoch(\d+)(?:-step(\d+))?(?:-jc(\d+))?\.pt$", fname)
    if not m:
        warnings.warn(f"File {fname} does not match the beast cp naming pattern.")
        return None

    epoch, step, jchannel_rank = (
        int(m.group(1)),
        int(m.group(2)) if m.group(2) else math.inf,
        int(m.group(3)) if m.group(3) else math.inf,
    )

    return {
        "epoch": epoch,
        "step": step,
        "jchannel_rank": jchannel_rank,
        "fname": fname,
    }


def load_state_dict(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_state_dict: dict,
    optimizer_state_dict: dict,
    last_epoch: int,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer | None, int]:
    """Load model and optimizer state, adapting to DDP-wrapped models.

    A ``module.`` key prefix is added or stripped as needed to match ``model``.
    If loading fails, the error is printed on rank 0 and the returned epoch is 0.

    Parameters
    ----------
    model : torch.nn.Module
        Model to load the state into.
    optimizer : torch.optim.Optimizer or None
        Optimizer to load the state into.
    model_state_dict : dict
        State dict of the model.
    optimizer_state_dict : dict
        State dict of the optimizer.
    last_epoch : int
        Last completed epoch of the checkpoint.

    Returns
    -------
    tuple[torch.nn.Module, torch.optim.Optimizer or None, int]
        The model, the optimizer and the epoch to resume from.
    """
    # DDP checks
    is_state_dict_ddp = any(k.startswith("module.") for k in model_state_dict.keys())
    if isinstance(model, DDP) and not is_state_dict_ddp:
        model_state_dict = {f"module.{k}": v for k, v in model_state_dict.items()}
    elif not isinstance(model, DDP) and is_state_dict_ddp:
        model_state_dict = {
            k.replace("module.", "", 1): v for k, v in model_state_dict.items()
        }

    model.load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    start_epoch = last_epoch + 1 if last_epoch is not None else 0

    return (
        model,
        optimizer,
        start_epoch,
    )  # return optimizer regardless, even if it is None

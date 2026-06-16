"""Utils module."""

import os
import random
from typing import Any

import numpy as np
import torch


def set_all_seeds(seed_value: int) -> None:
    """Set random seeds."""
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # pytorch random number generator for CPU
    torch.cuda.manual_seed(seed_value)  # pytorch random number generator for all GPUs
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    # torch.backends.cudnn.deterministic = True  # use deterministic algorithms.
    # torch.backends.cudnn.benchmark = False  # disable to be deterministic.
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # python hash seed.


def set_dtype(dtype: str) -> torch.dtype:
    """
    Set datatype from config file.

    Parameters
    ----------
    dtype : str
        Data type as a string, e.g., "float32", "bfloat16"

    Returns
    -------
    torch.dtype
        Corresponding torch data type.
    """
    match dtype:
        case "float32":
            return torch.float32
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case _:
            return torch.float32


def count_flops(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: torch.device,
    dtp_group: dist.ProcessGroup,
    n_samples: int = 1,
    bayesian=True,
):
    """
    Count floating point operations [TFLOP] in forward and backward pass of the model.

    FLOPs are accumulated over the entire group
    """
    x, y, _, _ = next(iter(dataloader))
    x, y = x.to(device), y.to(device)
    with FlopCounterMode(
        display=False
    ) as flop_counter:  # display=True breaks down by op
        if bayesian:
            pred = model(x, samples=n_samples)
        else:
            pred = model(x)

    forward_flops = torch.tensor(flop_counter.get_total_flops(), device=device)
    dist.all_reduce(forward_flops, group=dtp_group)

    with FlopCounterMode(
        display=False
    ) as flop_counter:  # display=True breaks down by op
        loss = loss_fn(pred, y)
        loss.backward()

    backward_flops = torch.tensor(flop_counter.get_total_flops(), device=device)
    dist.all_reduce(backward_flops, group=dtp_group)

    with FlopCounterMode(
        display=False
    ) as flop_counter:  # display=True breaks down by op
        if bayesian:
            pred = model(x, samples=n_samples)
        else:
            pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()

    total_flops = torch.tensor(flop_counter.get_total_flops(), device=device)
    dist.all_reduce(total_flops, group=dtp_group)

    if dist.get_rank(group=None) == 0:
        print(f"Forward flops per iteration is {forward_flops / 1e12} TFLOP")
        print(f"Backward flops per iteration is {backward_flops / 1e12} TFLOP")
        print(
            f"Total forward-backward flops per iteration is {total_flops / 1e12} TFLOP"
        )

    return forward_flops, backward_flops, total_flops


@torch.no_grad()
def get_spatial_weights(
    config_dict: dict[str, Any], device: torch.device
) -> torch.Tensor:
    """
    Calculate spatial weights based on latitude for loss weighting.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary containing data parameters, including "xlat" for the number of latitude points.
    device : torch.device
        Device to place the resulting weights tensor on.

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, 1, n_lat, 1) containing the spatial weights normalized by the cosine of the latitude.
    """
    n_lat = config_dict["data"]["xlat"]
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)
    cossum = np.sum(np.cos(latitude))
    weight = torch.tensor(n_lat * np.cos(latitude) / cossum, device=device).to(
        torch.float32
    )
    return weight.view(1, 1, n_lat, 1)

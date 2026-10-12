"""Utils module."""

import os
import random

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
        Data type as a string: "float32"/"fp32", "bfloat16"/"bf16" or "float16"/"fp16".

    Returns
    -------
    torch.dtype
        Corresponding torch data type.

    Raises
    ------
    ValueError
        If the data type is not one of the accepted values.
    """
    match dtype:
        case "float32" | "fp32":
            return torch.float32
        case "bfloat16" | "bf16":
            return torch.bfloat16
        case "float16" | "fp16":
            return torch.float16
        case _:
            raise ValueError(
                f"Unknown dtype '{dtype}'. Accepted values: 'float32' ('fp32'), "
                "'bfloat16' ('bf16'), 'float16' ('fp16')."
            )


@torch.no_grad()
def get_spatial_weights(n_lat: int, device: torch.device) -> torch.Tensor:
    """
    Calculate spatial weights based on latitude for loss weighting.

    Parameters
    ----------
    n_lat: int
        Number of latitude elements
    device : torch.device
        Device to place the resulting weights tensor on.

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, 1, n_lat, 1) containing the spatial weights normalized by the cosine of the latitude.
    """
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)
    cossum = np.sum(np.cos(latitude))
    weight = torch.tensor(n_lat * np.cos(latitude) / cossum, device=device).to(
        torch.float32
    )
    return weight.view(1, 1, n_lat, 1)

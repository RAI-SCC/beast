"""Seeding, dtype parsing and latitude weighting helpers."""

import os
import random

import numpy as np
import torch


def set_all_seeds(seed_value: int) -> None:
    """Seed Python, PyTorch (CPU and all GPUs) and ``PYTHONHASHSEED``.

    Parameters
    ----------
    seed_value : int
        Seed applied to every generator.
    """
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # pytorch random number generator for CPU
    torch.cuda.manual_seed(seed_value)  # pytorch random number generator for all GPUs
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    # torch.backends.cudnn.deterministic = True  # use deterministic algorithms.
    # torch.backends.cudnn.benchmark = False  # disable to be deterministic.
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # python hash seed.


def set_dtype(dtype: str) -> torch.dtype:
    """Convert a dtype name from a config file to a :class:`torch.dtype`.

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
def get_spatial_weights(
    n_lat: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cropped_lats: int = 1,
) -> torch.Tensor:
    """
    Calculate spatial weights based on latitude for loss weighting.

    This assumes even spacing along the latitude dimension, starting from 90°N, and crops latitude values from starting from the south.

    Parameters
    ----------
    n_lat : int
        Number of latitude elements.
    device : torch.device
        Device to place the resulting weights tensor on.
    dtype : torch.dtype
        Dtype to use for the weights tensor, default torch.float32.
    cropped_lats : int
        Number of latitude points dropped due to cropping, default 1.

    Returns
    -------
    torch.Tensor
        Float32 weights of shape ``(1, 1, n_lat, 1)``, normalized to a mean of 1.
    """
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat + cropped_lats)
    if cropped_lats > 0:
        latitude = latitude[:-cropped_lats]
    cossum = np.mean(np.cos(latitude))
    weight = torch.tensor(np.cos(latitude) / cossum, device=device, dtype=dtype)
    return weight.view(1, 1, n_lat, 1)

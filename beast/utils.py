import math
import os
import random
import re
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.init as init
import torch_blue
import torch_blue.vi.distributions as vidist
import xarray as xr
import yaml
from gb.comm import get_process_group
from gb.model.model_gb_tiny import NiceNorm
from jigsaw import normalization, utils_init
from torch.utils.flop_counter import FlopCounterMode


def execute_in_waves(func, wave_size: int = 128):
    world_size = dist.get_world_size(group=None)
    num_groups = (world_size + wave_size - 1) // wave_size
    global_rank = dist.get_rank(group=None)
    result = None

    for group in range(num_groups):
        wave_start = time.time()
        start_rank = group * wave_size
        end_rank = start_rank + wave_size

        if global_rank < end_rank and global_rank >= start_rank:
            result = func()

        dist.barrier(group=None)  # Synchronize after each wave
        wave_end = time.time()
        if global_rank == 0:
            print(
                f"R{global_rank}/{world_size}: Wave {group} of {func.__name__} executed in {wave_end - wave_start}"
            )

    return result



def convert_jigsaw_vi(
    module,
    variational_distribution=vidist.Normal(std=0.15),
    prior=vidist.Normal(),
    rescale_prior=False,
    kaiming_initialization=True,
    prior_initialization=False,
    return_log_probs=True,
):
    # torch_blue.vi.distributions.Distribution._init_uniform = _init_uniform_override
    torch_blue.vi.utils.convert.ban_convert(normalization.DistributedLayerNorm)
    torch_blue.vi.utils.convert.ban_convert(NiceNorm)
    torch_blue.vi.utils.convert.ban_convert(
        normalization.DistributedNICENorm, ban_mode="submodule"
    )
    torch_blue.vi.utils.convert.ban_convert(NiceNorm, ban_mode="submodule")

    torch_blue.vi.convert_to_vimodule(
        module,
        variational_distribution=variational_distribution,
        prior=prior,
        rescale_prior=rescale_prior,
        kaiming_initialization=kaiming_initialization,
        prior_initialization=prior_initialization,
        return_log_probs=return_log_probs,
        keep_weights=False,
    )

    num_jchannel = get_process_group("JChannel").size()

    def new_cf_channel(self, variable_shapes=None):
        previous = self._old_calculate_fan_in(variable_shapes=variable_shapes)
        return previous * num_jchannel

    fan_in_rescale_list = ["AVIDistributedXWTSync", "AVIConvTranspose2d"]

    weight_sync_list = [
        "VIConv2d",
        "AVIConvTranspose2d",
        "AVIDistributedXWTSync",
        "AVIDistributedXWT_Nice",
    ]

    for mname, m in module.named_modules():
        class_name = m.__class__.__name__

        if class_name in fan_in_rescale_list:
            m._old_calculate_fan_in = m._calculate_fan_in
            m._calculate_fan_in = types.MethodType(new_cf_channel, m)
            m.reset_variational_parameters()

        if class_name in weight_sync_list:
            utils_init.average_weights_module(m, get_process_group("JSpatial"))


def count_model_parameters(
    model: torch.nn.Module, parallelism: int, bayesian: bool
) -> int:
    if parallelism == 1:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        param_sum = 0
        for module in model.modules():
            # Skip the top-level model itself if it has the method
            if module is model:
                continue
            if hasattr(module, "get_parameter_count"):
                param_sum += module.get_parameter_count()

        total = param_sum * parallelism

    if bayesian:
        total = total * 2  # Count every parameter twice, if bayesian

    return total


def set_all_seeds(seed_value: int) -> None:
    """Set random seeds."""
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # pytorch random number generator for CPU
    torch.cuda.manual_seed(seed_value)  # pytorch random number generator for all GPUs
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    # torch.backends.cudnn.deterministic = True  # use deterministic algorithms.
    # torch.backends.cudnn.benchmark = False  # disable to be deterministic.
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # python hash seed.


def set_dtype(dtype: str):
    """Set datatype from config file"""
    match dtype:
        case "float32":
            return torch.float32
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case _:
            return torch.float32



@torch.no_grad()
def get_spatial_weights(config_dict, device):
    n_lat = config_dict["data"]["xlat"]
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)
    cossum = np.sum(np.cos(latitude))
    weight = torch.tensor(n_lat * np.cos(latitude) / cossum, device=device).to(
        torch.float32
    )
    return weight.view(1, 1, n_lat, 1)

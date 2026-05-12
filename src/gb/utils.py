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
from jigsaw import normalization, utils_init
from torch.utils.flop_counter import FlopCounterMode

from gb.comm import get_process_group
from gb.model.model_gb_tiny import NiceNorm


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


def download_1_5_degree_data(data_path):
    # data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")

    file_name = "weather_test_data.zarr"
    full_path = os.path.join(data_path, file_name)

    ds = xr.open_dataset(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    )

    variable_list_surface = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ]
    variable_list_pressure = [
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "specific_humidity",
        "temperature",
    ]
    variable_list = [*variable_list_surface, *variable_list_pressure]

    p_level_list = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    ds = ds[variable_list]

    times = pd.to_datetime(ds["time"].values)
    dataset_in_range = times[
        (times.year == 2020) & (times.month == 1) & (times.day >= 1) & (times.day <= 6)
    ]
    ds = ds.sel(time=dataset_in_range)

    ds_surface = ds[variable_list_surface]
    ds_pressure = ds[variable_list_pressure]
    ds_pressure = ds_pressure.sel(level=p_level_list)

    ds = xr.merge([ds_surface, ds_pressure])

    # print(ds)

    ds.to_zarr(full_path, mode="w", consolidated=True, zarr_format=2)

    return full_path


def _init_uniform_override(self, parameter, fan_in) -> None:
    bound = math.sqrt(6.0 / fan_in) if fan_in > 0 else 0
    init.uniform_(parameter, -bound, bound)


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
    """set datatype from config file"""
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
def get_spatial_weights(config_dict, device):
    n_lat = config_dict["data"]["xlat"]
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)
    cossum = np.sum(np.cos(latitude))
    weight = torch.tensor(n_lat * np.cos(latitude) / cossum, device=device).to(
        torch.float32
    )
    return weight.view(1, 1, n_lat, 1)



def load_configuration_yaml(directory):
    yaml_path = os.path.join(directory, "config.yaml")
    with open(f"{yaml_path}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return config_dict





def dist_clip_grad_norm_(params, max_norm, norm_type=2.0, group=None):
    params = [
        p for p in params if p.grad is not None
    ]  # Convert generator to list for foreach_map
    max_norm = torch.tensor(max_norm, device=params[0].device)

    def _get_local_norms(param):
        if param.grad is not None:
            if norm_type == 2.0:
                norm = torch.sqrt(torch.sum(torch.square(torch.abs(param.grad))))
            elif norm_type == 1.0:
                norm = torch.sum(torch.abs(param.grad))
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")

            return norm
        else:
            return None

    def _reduce_norms(params, norms, norm_type=2.0, group=None):
        reduced_norms = []
        for param, norm in zip(params, norms):
            if (
                hasattr(param, "sharded_dims")
                and (param.sharded_dims is not None)
                and (norm is not None)
            ):
                if norm_type == 2.0:
                    norm = torch.square(norm)
                    norm = dist.dnn.all_reduce(norm, group=group)
                if norm_type == 2.0:
                    norm = torch.sqrt(norm)
            reduced_norms.append(norm)

        return reduced_norms

    with torch.no_grad():
        # compute local norms
        norms = [_get_local_norms(p) for p in params]

        # reduce norms across ranks if required
        norms = _reduce_norms(params, norms, norm_type=norm_type, group=group)

        # now reduce the norms in the vector
        # Filter out None values before stacking
        valid_norms = [n for n in norms if n is not None]

        if len(valid_norms) == 0:
            # we don't want to clip in this case
            total_norm = torch.tensor(0.0, device=params[0].device)
        else:
            normvec = torch.stack(valid_norms)

            if norm_type == 2.0:
                total_norm = torch.sqrt(torch.sum(torch.square(normvec)))
            elif norm_type == 1.0:
                total_norm = torch.sum(torch.abs(normvec))
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")

            # clip grads and return total norm
            clip_coef = max_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for param in params:
                param.grad.mul_(clip_coef_clamped)

    return total_norm

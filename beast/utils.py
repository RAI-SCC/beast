import math
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.init as init
import xarray as xr
import yaml
from torch.utils.flop_counter import FlopCounterMode

from beast.comm import get_process_group


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


@torch.no_grad()
def get_spatial_weights(config_dict, device):
    n_lat = config_dict["data"]["xlat"]
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)
    cossum = np.sum(np.cos(latitude))
    weight = torch.tensor(n_lat * np.cos(latitude) / cossum, device=device).to(
        torch.float32
    )
    return weight.view(1, 1, n_lat, 1)


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


def load_configuration_yaml(directory):
    yaml_path = os.path.join(directory, "config.yaml")
    with open(f"{yaml_path}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return config_dict


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

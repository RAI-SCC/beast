import math
import os
import random
import re
import types
from collections.abc import Sequence

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.init as init
import torch_blue
import torch_blue.vi.distributions as vidist
import xarray as xr
import yaml
from jigsaw import normalization, utils_init
from torch.nn.parallel import DistributedDataParallel
from torch.utils.flop_counter import FlopCounterMode

from gb.model.model_gb_tiny import NiceNorm

_process_group_dict: dict[str, dist.ProcessGroup] = {}
_mesh: dist.device_mesh.DeviceMesh


def init_gb_process_group(
    mesh_dims: Sequence[int], backend: str = "nccl"
) -> dict[str, dist.ProcessGroup]:
    """Initialize the global process group for GrandBellbeast."""
    rank = int(os.getenv("SLURM_PROCID"))  # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS"))  # Total number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID", rank))  # Local GPU ID.

    # print(f"Rank {rank}: world_size {world_size}, slurm_localid {slurm_localid}")
    # # Initialize GPUs
    device = "cuda" if backend == "nccl" else "cpu"
    device_type = device
    if backend == "nccl":
        device_id = slurm_localid if torch.cuda.device_count() != 1 else 0
        device = torch.device("cuda", device_id)
        torch.cuda.set_device(device)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method="env://",
    )
    if not dist.is_initialized():
        raise RuntimeError("Process group initialization failed.")

    # print(f"Rank {rank}: CUDA Available = {torch.cuda.is_available()}")
    # print(f"Rank {rank}: CUDA Device count = {torch.cuda.device_count()}")

    # # Initialize GPUs
    device = "cuda" if backend == "nccl" else "cpu"
    device_type = device
    if backend == "nccl":
        device_id = slurm_localid if torch.cuda.device_count() != 1 else 0
        device = torch.device("cuda", device_id)
        torch.cuda.set_device(device)

    return _create_process_mesh_dict(mesh_dims, device_type=device_type)


def _create_process_mesh_dict(
    mesh_dims: Sequence[int],
    device_type: str = "cuda",
) -> dict[str, dist.ProcessGroup]:
    """Create a process mesh given dimensions and names.

    Input: 5D process mesh for (Expert, DP, SP, JCol, JRow)

    Parallel Process Groups keys:
     - JChannel: Jigsaw Row Tensor Parallel group (part of DTP)
     - JSpatial: Jigsaw Column Tensor Parallel group (part of DTP)
     - DTP: Domain Tensor Parallel group, equal to JSpatial x JChannel
     - SP: Sample Parallel Group
     - DP: Data Parallel Group
     - DDP: Distributed Data Parallel Group (DP x SP), to be pased to the DDP module
     - Expert: Top group for Mixture of Experts
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")

    world_size = dist.get_world_size()

    if torch.prod(torch.tensor(mesh_dims)).item() != world_size:
        raise ValueError("Product of mesh dimensions must equal world size.")

    if len(mesh_dims) == 4:
        mesh_dims = [1] + list(mesh_dims)  # Add Expert dim if missing
    elif len(mesh_dims) != 5:
        raise ValueError("mesh_dims must be of length 4 or 5.")

    mesh_dim_names = ["Expert", "DP", "SP", "JSpatial", "JChannel"]
    _mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_dims),
        mesh_dim_names=tuple(mesh_dim_names),
    )

    # Create process groups
    for dim_name, group in zip(mesh_dim_names, _mesh.get_all_groups()):
        _process_group_dict[dim_name] = group

    # Also create combined groups (Expert, DDP, DPT)
    alternative_mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=(
            mesh_dims[0],
            mesh_dims[1] * mesh_dims[2],
            mesh_dims[3] * mesh_dims[4],
        ),
        mesh_dim_names=("Expert", "DDP", "DTP"),
    )
    alternative_mesh_ddp = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=(
            mesh_dims[0],
            mesh_dims[1] * mesh_dims[2] * mesh_dims[3],
            mesh_dims[4],
        ),
        mesh_dim_names=("Expert", "DDP", "JChannel"),
    )

    _process_group_dict["DTP"] = alternative_mesh.get_group("DTP")
    _process_group_dict["DDP"] = alternative_mesh_ddp.get_group("DDP")

    return _process_group_dict


def get_process_group(name: str) -> dist.ProcessGroup:
    """Get a process group by name."""
    if name not in _process_group_dict:
        raise ValueError(f"Process group '{name}' not found.")
    return _process_group_dict[name]


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
    variational_distribution=vidist.Normal(),
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
        if step:
            model_name = f"cp-epoch{epoch}-step{step}-e{ep_group.rank()}-jc{channel_group.rank()}"
        else:
            model_name = f"cp-epoch{epoch}-e{ep_group.rank()}-jc{channel_group.rank()}"
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


def load_latest_model_states(directory):
    end_of_epoch_directory = os.path.join(directory, "checkpoints")
    path_base = end_of_epoch_directory
    unfiltered_checkpoint_list = os.listdir(end_of_epoch_directory)
    checkpoints = [
        fname for fname in unfiltered_checkpoint_list if fname.endswith(".pt")
    ]
    local_checkpoint = ""
    latest_epoch = -1
    latest_step = -1
    ep_group_rank = get_process_group("Expert").rank()
    jc_group_rank = get_process_group("JChannel").rank()

    intermediate = False

    for fname in checkpoints:
        m = re.match(r"cp-epoch([0-9]+)-e([0-9]+)-jc([0-9]+).pt", fname)
        try:
            epoch = int(m.group(1))
            expert_rank = int(m.group(2))
            jchannel_rank = int(m.group(3))
            step = None
        except IndexError:
            m = re.match(
                r"cp-epoch([0-9]+)-step([0-9]+)-e([0-9]+)-jc([0-9]+).pt", fname
            )
            epoch = int(m.group(1))
            expert_rank = int(m.group(3))
            jchannel_rank = int(m.group(4))
            step = int(m.group(2))
            intermediate = True

        if not expert_rank == ep_group_rank:
            continue

        if not jchannel_rank == jc_group_rank:
            continue

        if not intermediate:
            if epoch <= latest_epoch:
                continue
        else:
            if epoch < latest_epoch:
                continue

            if step is not None:
                if step <= latest_step:
                    continue

        latest_step = step

        local_checkpoint = fname
        latest_epoch = epoch

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
    model, model_state_dict, optimizer_state_dict, last_epoch, device_id, bayesian=True
):
    ddp_group = get_process_group("DDP")

    if bayesian:
        convert_jigsaw_vi(model)
        model.return_log_probs = False

    # Make model data parallel
    model_ddp = DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
        process_group=ddp_group,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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

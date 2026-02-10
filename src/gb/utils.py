import math
import os
import types
from collections.abc import Sequence
from typing import Tuple, cast

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.init as init
import torch_blue
import torch_blue.vi.distributions as vidist
import xarray as xr

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

    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size, init_method="env://"
    )
    if dist.is_initialized():
        print(
            f"Rank {rank}/{world_size}: Process group initialized with torch rank {dist.get_rank()} and torch world size {dist.get_world_size()}."
        )
    else:
        raise RuntimeError("Process group initialization failed.")

    # print(f"Rank {rank}: CUDA Available = {torch.cuda.is_available()}")
    # print(f"Rank {rank}: CUDA Device count = {torch.cuda.device_count()}")

    # # Initialize GPUs
    device = "cuda" if backend == "nccl" else "cpu"
    device_type = device
    if backend == "nccl":
        device = torch.device("cuda", slurm_localid)
        torch.cuda.set_device(device)

    return _create_process_mesh_dict(mesh_dims, device_type=device_type)


def _create_process_mesh_dict(
    mesh_dims: Sequence[int],
    device_type: str = "cuda",
) -> dict[str, dist.ProcessGroup]:
    """Create a process mesh given dimensions and names.

    Input: 5D process mesh for (Expert, DP, SP, JCol, JRow)

    Parallel Process Groups keys:
     - JSpatial: Jigsaw Column Tensor Parallel group (part of DTP)
     - JChannel: Jigsaw Row Tensor Parallel group (part of DTP)
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

    _process_group_dict["DTP"] = alternative_mesh.get_group("DTP")
    _process_group_dict["DDP"] = alternative_mesh.get_group("DDP")

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


def count_parameters(model) -> int:
    """Count num trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_uniform_override(self, parameter, fan_in) -> None:
    bound = math.sqrt(6.0 / fan_in) if fan_in > 0 else 0
    init.uniform_(parameter, -bound, bound)


def _calculate_fan_out(self, variable_shapes=None):
    """
    calculates fan_out for a Bayesian layer assuming that weight shape is (out_dim, in_dim); if weight shape is
    (in_dim, out_dim) instead calculates fan_in, which is what it will be used for in this project
    """

    if variable_shapes is None:
        loop = self.random_variables
        checked_dict = self.variational_distribution
    else:
        loop = checked_dict = variable_shapes  # type:ignore[assignment]

    if len(loop) == 0:
        raise torch_blue.vi.utils.errors.NoVariablesError(
            "All module variables are set to None."
        )

    for var in cast(Tuple[str, ...], loop):
        if checked_dict[var] is None:
            continue

        if variable_shapes is None:
            weight_name = self.variational_parameter_name(
                var,
                self.variational_distribution[var].primary_parameter,
            )
            shape_dummy = getattr(self, weight_name)
        else:
            shape_dummy = torch.zeros(variable_shapes[var])

        if shape_dummy.dim() < 2:
            continue

        _, fan_out = init._calculate_fan_in_and_fan_out(shape_dummy)
        return fan_out
    else:
        # TODO raise warning here
        return 1


def convert_jigsaw_vi(
    module,
    variational_distribution=vidist.Normal(),
    prior=vidist.Normal(),
    rescale_prior=False,
    kaiming_initialization=True,
    prior_initialization=False,
    return_log_probs=True,
):
    torch_blue.vi.distributions.Distribution._init_uniform = _init_uniform_override

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

    num_jcols = get_process_group("JSpatial")  # get_process_group("JCol").size()
    num_jrows = get_process_group("JChannel")  # get_process_group("JRow").size()

    def new_cf_col(self, variable_shapes=None):
        previous = self._old_calculate_fan_in(variable_shapes=variable_shapes)
        return previous * num_jcols

    def new_cf_row(self, variable_shapes=None):
        previous = self._old_calculate_fan_in(variable_shapes=variable_shapes)
        return previous * num_jrows

    for mname, m in module.named_modules():
        class_name = m.__class__.__name__
        if class_name.startswith("AVI"):
            if class_name in ["AVIDistributedXW", "AVIDistributedXTW"]:
                m._calculate_fan_in = types.MethodType(_calculate_fan_out, m)
            # switches fan_in and fan_out dimensions for layers with different weight shape initialization

            match class_name:
                case (
                    "AVIDistributedXWT" | "AVIDistributedXWTSync" | "AVIConvTranspose2d"
                ):
                    m._old_calculate_fan_in = m._calculate_fan_in
                    m._calculate_fan_in = types.MethodType(new_cf_col, m)
                    m.reset_variational_parameters()
                case "AVIDistributedXW" | "AVIDistributedXTW":
                    m._old_calculate_fan_in = m._calculate_fan_in
                    m._calculate_fan_in = types.MethodType(new_cf_row, m)
                    m.reset_variational_parameters()

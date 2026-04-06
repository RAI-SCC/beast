import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
import yaml

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


def load_configuration_yaml(directory):
    yaml_path = os.path.join(directory, "config.yaml")
    with open(f"{yaml_path}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return config_dict

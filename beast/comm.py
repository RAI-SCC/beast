"""Utilities for distributed communication and process mesh management in beast.

This module provides helpers to create and manage process meshes and
process groups for distributed training. It supports a fallback
CustomDeviceMesh implementation when the native torch.device_mesh
functionality is not available or suitable.
"""

from typing import Sequence

import torch
import torch.distributed as dist

_process_group_dict: dict[str, dist.ProcessGroup] = {}
_mesh: dist.device_mesh.DeviceMesh


def create_process_mesh_dict(
    mesh_dims: Sequence[int],
    device_type: str = "cuda",
) -> dict[str, dist.ProcessGroup]:
    """Create a dictionary of process groups for a logical device mesh.

    The function expects a 5-D process mesh corresponding to the logical
    dimensions (Expert, DP, SP, JSpatial, JChannel). If a 4-D mesh is
    supplied, an Expert dimension of size 1 is prepended.

    Parameters
    ----------
    mesh_dims : Sequence[int]
        Sizes for each mesh dimension (length 4 or 5). The product of
        the sizes must equal the distributed world size.
    device_type : str, optional
        Device type string passed to the device mesh initializer
        (default ``"cuda"``).

    Returns
    -------
    dict[str, dist.ProcessGroup]
        Mapping from logical names to :class:`torch.distributed.ProcessGroup`.

    Notes
    -----
    The returned dictionary contains groups for the following keys:
    ``Expert``, ``DP``, ``SP``, ``JSpatial``, ``JChannel``, plus combined
    groups ``DTP`` and ``DDP`` used for different parallelism strategies.
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

    global _mesh
    _mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_dims),
        mesh_dim_names=tuple(mesh_dim_names),
    )

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

    # Create process groups
    for dim_name in mesh_dim_names:
        _process_group_dict[dim_name] = _mesh.get_group(dim_name)

    _process_group_dict["DTP"] = alternative_mesh.get_group("DTP")
    _process_group_dict["DDP"] = alternative_mesh_ddp.get_group("DDP")

    return _process_group_dict


def get_process_group(name: str) -> dist.ProcessGroup:
    """Return a previously created process group by name.

    Parameters
    ----------
    name : str
        Key name of the process group to retrieve.

    Returns
    -------
    torch.distributed.ProcessGroup
        The requested process group.

    Raises
    ------
    ValueError
        If no process group exists for the provided name.
    """
    if name not in _process_group_dict:
        raise ValueError(f"Process group '{name}' not found.")
    return _process_group_dict[name]

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
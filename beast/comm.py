"""Utilities for distributed communication and process mesh management in beast.

This module provides helpers to create and manage process meshes and
process groups for distributed training. It supports a fallback
CustomDeviceMesh implementation when the native torch.device_mesh
functionality is not available or suitable.
"""

import os
import socket
import time
from collections.abc import Callable
from typing import Any, Sequence

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
    dimensions (DP, SP, JSpatial, JChannel).

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
    ``DP``, ``SP``, ``JSpatial``, ``JChannel``, plus combined
    groups ``DTP`` and ``DDP`` used for different parallelism strategies.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")

    world_size = dist.get_world_size()

    print(mesh_dims)
    if torch.prod(torch.tensor(mesh_dims)).item() != world_size:
        raise ValueError("Product of mesh dimensions must equal world size.")

    mesh_dim_names = ["DP", "SP", "JSpatial", "JChannel"]

    global _mesh, _process_group_dict
    _mesh = dist.device_mesh.init_device_mesh(
        device_type=device_type,
        mesh_shape=tuple(mesh_dims),
        mesh_dim_names=tuple(mesh_dim_names),
    )

    # Create process groups
    for dim_name in mesh_dim_names:
        _process_group_dict[dim_name] = _mesh.get_group(dim_name)

    _process_group_dict["DTP"] = (
        _mesh["JSpatial", "JChannel"]._flatten("DTP").get_group()
    )
    _process_group_dict["DDP"] = (
        _mesh["DP", "SP", "JSpatial"]._flatten("DDP").get_group()
    )

    visualize_process_mesh()

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


def execute_in_waves(func: Callable, wave_size: int = 128) -> Any:
    """Execute a function across distributed ranks in waves to reduce contention.

    Parameters
    ----------
    func : Callable
        The function to execute on each rank. Should contain the logic that
        needs to be executed in a distributed manner.
    wave_size : int, optional
        The number of ranks to execute the function in each wave (default is 128).

    Returns
    -------
    Any
        The result of the function execution on the current rank, or None if
        the rank is not part of the current wave.
    """
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


def visualize_process_mesh() -> None:
    """Print a visual representation of the generated process mesh.

    Each rank gathers its identity (hostname, local GPU id and global rank)
    together with its rank within every logical/combined process group. The
    information is collected on rank 0 and printed as a table so the user can
    see how ranks are distributed across the mesh and its groups.

    The output contains, for every global rank:

    * ``host``      -- the hostname the rank runs on
    * ``gpu``       -- the local device (GPU) id of the rank
    * ``global``    -- the global rank in the default process group
    * one column per group (``Expert``, ``DP``, ``SP``, ``JSpatial``,
      ``JChannel``, ``DTP``, ``DDP``) showing the rank *within* that group.

    Notes
    -----
    This is a collective call and must be invoked by all ranks.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")
    if not _process_group_dict:
        raise RuntimeError("Process mesh has not been created yet.")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    hostname = socket.gethostname()
    gpu_id = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else int(os.environ.get("LOCAL_RANK", 0))
    )

    group_names = list(_process_group_dict.keys())

    # Rank within each group (or -1 if this rank is not a member).
    group_ranks = {
        name: dist.get_rank(group=pg) for name, pg in _process_group_dict.items()
    }

    local_info = {
        "host": hostname,
        "gpu": gpu_id,
        "global": global_rank,
        **group_ranks,
    }

    # Collect every rank's info on all ranks.
    gathered: list[dict[str, Any]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, local_info)

    if global_rank != 0:
        return

    # Build a fixed-width table.
    columns = ["host", "gpu", "global", *group_names]
    widths = {
        col: max(len(str(col)), *(len(str(row[col])) for row in gathered))
        for col in columns
    }

    def fmt_row(values: Sequence[Any]) -> str:
        return " | ".join(f"{str(v):>{widths[col]}}" for col, v in zip(columns, values))

    header = fmt_row(columns)
    separator = "-+-".join("-" * widths[col] for col in columns)

    print("\nProcess mesh visualization (rank within each group):")
    print(header)
    print(separator)
    for row in sorted(gathered, key=lambda r: r["global"]):
        print(fmt_row([row[col] for col in columns]))
    print()


def shutdown_comm() -> None:
    """Shutdown existing process groups and mesh."""
    global _mesh, _process_group_dict
    for key, pg in _process_group_dict.items():
        pg.shutdown()

    _process_group_dict = {}

"""Process mesh and process group management for distributed execution.

The 4-D mesh is built on :func:`torch.distributed.device_mesh.init_device_mesh`
and its groups are kept in module-level state, retrievable by name through
:func:`get_process_group`.
"""

import os
import socket
import time
from collections.abc import Callable
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

_process_group_dict: dict[str, dist.ProcessGroup] = {}
_mesh: dist.device_mesh.DeviceMesh | None = None


def create_process_mesh_dict(
    mesh_dims: Sequence[int],
    device_type: str = "cuda",
) -> dict[str, dist.ProcessGroup]:
    """Create the 4-D process mesh and its named process groups.

    The four mesh dimensions and the two combined groups are:

    * ``DP``: data parallelism, each rank group sees different data items.
    * ``SP``: sample parallelism, the ensemble members of a Bayesian model.
    * ``JSpatial``: spatial parallelism, each rank holds a tile of the grid.
    * ``JChannel``: channel parallelism, each rank holds a slice of the channels.
    * ``DTP``: ``JSpatial`` x ``JChannel``, the ranks that share one data item.
    * ``DDP``: ``DP`` x ``SP`` x ``JSpatial``, the ranks gradients are averaged over.

    Parameters
    ----------
    mesh_dims : Sequence[int]
        Sizes of the mesh dimensions in the order ``(DP, SP, JSpatial,
        JChannel)``. Their product must equal the world size.
    device_type : str, default: "cuda"
        Device type passed to ``init_device_mesh``.

    Returns
    -------
    dict[str, dist.ProcessGroup]
        The process groups, keyed ``DP``, ``SP``, ``JSpatial``, ``JChannel``,
        ``DTP`` and ``DDP``.

    Raises
    ------
    RuntimeError
        If the default process group is not initialized.
    ValueError
        If ``mesh_dims`` does not have four entries or its product differs from
        the world size.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")

    world_size = dist.get_world_size()

    mesh_dim_names = ["DP", "SP", "JSpatial", "JChannel"]

    # Check the number of dimensions first, for a clearer error than the one
    # init_device_mesh raises.
    if len(mesh_dims) != len(mesh_dim_names):
        raise ValueError(
            f"mesh_dims must be of length {len(mesh_dim_names)} "
            f"{tuple(mesh_dim_names)}, got {len(mesh_dims)}: {list(mesh_dims)}."
        )

    if torch.prod(torch.tensor(mesh_dims)).item() != world_size:
        raise ValueError("Product of mesh dimensions must equal world size.")

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
    """Return a process group created by :func:`create_process_mesh_dict`.

    Parameters
    ----------
    name : str
        Group name, e.g. ``"DP"`` or ``"DDP"``.

    Returns
    -------
    dist.ProcessGroup
        The requested process group.

    Raises
    ------
    ValueError
        If no group with that name exists.
    """
    if name not in _process_group_dict:
        raise ValueError(f"Process group '{name}' not found.")
    return _process_group_dict[name]


def execute_in_waves(func: Callable, wave_size: int = 128) -> Any:
    """Run ``func`` on all ranks, ``wave_size`` ranks at a time.

    Staggering the calls reduces contention on shared resources such as the
    filesystem. This is a collective call: every rank must reach it.

    Parameters
    ----------
    func : Callable
        Zero-argument callable to run on each rank.
    wave_size : int, default: 128
        Number of ranks that run ``func`` concurrently.

    Returns
    -------
    Any
        The return value of ``func`` on this rank.
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
    """Print a table of every rank's host, GPU and rank within each group.

    The table has one row per global rank with the columns ``host``, ``gpu``,
    ``global`` and one column per process group. Rank 0 prints it.

    Raises
    ------
    RuntimeError
        If the default process group is not initialized or no mesh exists yet.

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
        """Format one table row.

        Parameters
        ----------
        values : Sequence[Any]
            One value per column.

        Returns
        -------
        str
            The values, right-aligned to the column widths and separated by ``|``.
        """
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
    """Destroy the process groups created by :func:`create_process_mesh_dict`.

    Each group is destroyed once, even if it is stored under several names. The
    default (``WORLD``) group is kept, so collectives keep working and a new mesh
    can be created afterwards.
    """
    global _mesh, _process_group_dict

    destroyed: set[int] = set()
    for pg in _process_group_dict.values():
        if pg is dist.group.WORLD or id(pg) in destroyed:
            continue
        destroyed.add(id(pg))
        dist.destroy_process_group(pg)

    _process_group_dict = {}
    # The mesh references the groups just destroyed; dropping it stops a stale
    # DeviceMesh from outliving them.
    _mesh = None


def wrap_ddp(
    model: torch.nn.Module,
    device_id: int | None,
    ddp_group: dist.ProcessGroup,
    reduction_size_mb: int = 100,
) -> DistributedDataParallel:
    """Wrap a model in DistributedDataParallel over the given process group.

    Gradients are averaged over ``ddp_group``, usually the ``DDP`` group of
    :func:`create_process_mesh_dict`.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap.
    device_id : int or None
        CUDA device index the model lives on, or None for CPU.
    ddp_group : dist.ProcessGroup
        Process group to synchronize gradients over.
    reduction_size_mb : int, default: 100
        Size of every gradient bucket, including the first, in MiB.

    Returns
    -------
    DistributedDataParallel
        The wrapped model.

    Notes
    -----
    The model is wrapped with ``static_graph=True``, so its autograd graph must
    be the same in every iteration and every parameter must be used in every
    step.
    """
    # DDP only exposes a setting for the size of the buckets after the first, so
    # the first bucket size is set through the private module default for the
    # duration of the constructor and then restored.
    previous_first_bucket_bytes = dist._DEFAULT_FIRST_BUCKET_BYTES  # type: ignore[attr-defined]
    dist._DEFAULT_FIRST_BUCKET_BYTES = reduction_size_mb * 1048576  # type: ignore[attr-defined]
    try:
        return DistributedDataParallel(
            model,
            device_ids=[device_id] if device_id is not None else None,
            output_device=device_id,
            process_group=ddp_group,
            gradient_as_bucket_view=True,
            static_graph=True,
            bucket_cap_mb=reduction_size_mb,
        )
    finally:
        dist._DEFAULT_FIRST_BUCKET_BYTES = previous_first_bucket_bytes  # type: ignore[attr-defined]

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
from jigsaw import parallel  # type: ignore[import-not-found]
from torch.nn.parallel import DistributedDataParallel

_process_group_dict: dict[str, dist.ProcessGroup] = {}
_mesh: dist.device_mesh.DeviceMesh | None = None


def create_process_mesh_dict(
    mesh_dims: Sequence[int],
    device_type: str = "cuda",
) -> dict[str, dist.ProcessGroup]:
    """Create a dictionary of process groups for a logical device mesh.

    The function expects a 4-D process mesh corresponding to the logical
    dimensions (DP, SP, JSpatial, JChannel).

    Parameters
    ----------
    mesh_dims : Sequence[int]
        Sizes for each mesh dimension, in the order (DP, SP, JSpatial,
        JChannel). The product of the sizes must equal the distributed world
        size.
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
    ``DP``, ``SP``, ``JSpatial``, ``JChannel``, plus combined groups:

    * ``DTP`` (``JSpatial x JChannel``): one model-parallel instance.
    * ``DDP`` (``DP x SP x JSpatial``): every rank holding the same model shard;
      used to pick one checkpoint writer per shard. Despite its name, it is
      *not* the group to wrap DDP over: ``JSpatial`` (and, for a Bayesian model,
      ``SP``) are replica axes that jigsaw reduces itself, which
      ``jigsaw.parallel.check_ddp`` rejects in a DDP group.
    * ``DPSP`` (``DP x SP``): the DDP group of a deterministic model, whose
      ``SP`` ranks are plain copies. A Bayesian model uses ``DP``.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")

    world_size = dist.get_world_size()

    mesh_dim_names = ["DP", "SP", "JSpatial", "JChannel"]

    # Checked before the product so a stale 5-D layout (which used to carry a
    # leading Expert dimension) reports the arity rather than failing later
    # inside init_device_mesh with a mesh_shape/mesh_dim_names mismatch.
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
    _process_group_dict["DPSP"] = _mesh["DP", "SP"]._flatten("DPSP").get_group()

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
    * one column per group (``DP``, ``SP``, ``JSpatial``, ``JChannel``,
      ``DTP``, ``DDP``, ``DPSP``) showing the rank *within* that group.

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
    """Destroy the process groups created by :func:`create_process_mesh_dict`.

    The default (``WORLD``) group is left intact, so the caller can keep using
    plain collectives and build a fresh mesh afterwards.

    Two cases are guarded defensively, because whether they occur depends on the
    torch version and the backend rather than on anything here:

    * A mesh dimension may be *backed by* the default process group when it
      spans the whole world. Destroying that would take down every later
      collective, including the ``all_gather_object`` in
      :func:`visualize_process_mesh`. On torch 2.14 ``init_device_mesh`` hands
      out a distinct group per dimension, so the skip is a no-op there.
    * ``DTP``, ``DDP`` and ``DPSP`` are flattened views and may alias a
      dimension group, so the same object can appear under more than one key.
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
    joint_hook: bool = True,
) -> DistributedDataParallel:
    """Wrap a jigsaw model in DistributedDataParallel over the given process group.

    Built with ``jigsaw.parallel.wrap_ddp``, which first makes every marked
    parameter identical across its replica groups and then checks the wrap
    with ``jigsaw.parallel.check_ddp``. DDP *averages* gradients over
    ``ddp_group``, so the group must span only the data-parallel axes:
    ``DP`` for a Bayesian model (whose ``SP`` ranks hold different ensemble
    members, and are a replica axis of every parameter) and ``DPSP`` for a
    deterministic one. The model-parallel replica axes are reduced by jigsaw,
    in the backward pass or, with ``joint_hook``, in DDP's buckets.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap.
    device_id : int | None
        CUDA device index the model lives on, or None for CPU.
    ddp_group : dist.ProcessGroup
        Process group to average gradients over.
    reduction_size_mb : int, default: 100
        Gradient bucket size in MiB, applied both to the first bucket and to
        all subsequent ones.
    joint_hook : bool, default: True
        Register ``jigsaw.parallel.joint_comm_hook``: each replicated gradient
        is averaged over ``ddp_group`` x its replica groups in one all-reduce
        per bucket, instead of summed over the replica groups in the backward
        pass and then averaged over ``ddp_group``. Same result, fewer bytes over
        the inter-node links. Collective over the default group.

    Returns
    -------
    DistributedDataParallel
        The wrapped model.

    Notes
    -----
    The wrap is built with ``gradient_as_bucket_view=True`` and
    ``static_graph=True``. The latter is a promise about the model: the autograd
    graph must not change between iterations, and no parameter may be unused in
    some steps but used in others. A model taking data-dependent branches will
    mis-reduce under it.

    ``bucket_cap_mb`` only sizes the buckets after the first. torch exposes no
    public setter for the first one, so the private
    ``dist._DEFAULT_FIRST_BUCKET_BYTES`` is patched for the duration of the
    constructor and restored afterwards. Leaving it patched would resize the
    first bucket of every other DDP instance built later in the same process.
    """
    previous_first_bucket_bytes = dist._DEFAULT_FIRST_BUCKET_BYTES  # type: ignore[attr-defined]
    dist._DEFAULT_FIRST_BUCKET_BYTES = reduction_size_mb * 1048576  # type: ignore[attr-defined]
    try:
        ddp: DistributedDataParallel = parallel.wrap_ddp(
            model,
            {"DP": ddp_group},
            dp_key="DP",
            joint_hook=joint_hook,
            device_ids=[device_id] if device_id is not None else None,
            output_device=device_id,
            gradient_as_bucket_view=True,
            static_graph=True,
            bucket_cap_mb=reduction_size_mb,
        )
        return ddp
    finally:
        dist._DEFAULT_FIRST_BUCKET_BYTES = previous_first_bucket_bytes  # type: ignore[attr-defined]

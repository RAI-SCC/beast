import math
import os
from typing import Sequence

import torch
import torch.distributed as dist

USE_CUSTOM_MESH = False

_process_group_dict: dict[str, dist.ProcessGroup] = {}
_mesh: dist.device_mesh.DeviceMesh


class CustomDeviceMesh:
    def __init__(
        self,
        mesh_dims: list[int],
        mesh_dim_names: list[str],
        backend: str = "nccl",
        device: torch.device = None,
    ):
        self._mesh_dims = mesh_dims
        self._mesh_dim_names = mesh_dim_names
        self._backend = backend
        self._device = device
        self._groups = {}
        self._rank = torch.distributed.get_rank()
        self.__initialize_groups()

    @property
    def ndims(self) -> int:
        return len(self._mesh_dims)

    @property
    def mesh_dim_names(self):
        """Returns the names of mesh dimensions."""
        return self._mesh_dim_names

    def get_group(self, dim_name: str):
        return self._groups[dim_name]

    def __initialize_groups(self):
        rank_tensor = torch.arange(
            math.prod(self._mesh_dims), dtype=torch.int64, device="cpu"
        )
        rank_tensor = rank_tensor.reshape(*self._mesh_dims)

        # now iterate through the mesh dimensions and create subgroups:
        for dim in range(self.ndims - 1, -1, -1):
            rank_tensor_mv = torch.movedim(rank_tensor, dim, -1)
            rank_tensor_rs = rank_tensor_mv.flatten(0, -2)
            group_ranks = [x.flatten() for x in rank_tensor_rs.unbind(0)]

            for granks in group_ranks:
                granks_list = granks.numpy().tolist()
                group = dist.new_group(
                    ranks=granks_list, backend=self._backend, device_id=self._device
                )
                if self._rank in granks_list:
                    self._groups[self._mesh_dim_names[dim]] = group

        # # flatten last two dimensions:
        # rank_tensor_fl = rank_tensor.flatten(-2)
        # rank_tensor_rs = rank_tensor_fl.flatten(0, -2)
        # group_ranks = [x.flatten() for x in rank_tensor_rs.unbind(0)]
        # for granks in group_ranks:
        #     granks_list = granks.numpy().tolist()
        #     group = dist.new_group(
        #         ranks=granks_list, backend=self._backend, device_id=self._device
        #     )
        #     if self._rank in granks_list:
        #         self._groups["yx"] = group

        # self.yx_group = self._groups["yx"]


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

    if not USE_CUSTOM_MESH:
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
    else:
        _mesh = CustomDeviceMesh(
            mesh_dims=tuple(mesh_dims),
            mesh_dim_names=tuple(mesh_dim_names),
            backend="nccl",
            device=device_type,
        )

        alternative_mesh = CustomDeviceMesh(
            mesh_dims=(
                mesh_dims[0],
                mesh_dims[1] * mesh_dims[2],
                mesh_dims[3] * mesh_dims[4],
            ),
            mesh_dim_names=("Expert", "DDP", "DTP"),
            backend="nccl",
            device=device_type,
        )
        alternative_mesh_ddp = CustomDeviceMesh(
            mesh_dims=(
                mesh_dims[0],
                mesh_dims[1] * mesh_dims[2] * mesh_dims[3],
                mesh_dims[4],
            ),
            mesh_dim_names=("Expert", "DDP", "JChannel"),
            backend="nccl",
            device=device_type,
        )

    # Create process groups
    for dim_name in mesh_dim_names:
        _process_group_dict[dim_name] = _mesh.get_group(dim_name)

    _process_group_dict["DTP"] = alternative_mesh.get_group("DTP")
    _process_group_dict["DDP"] = alternative_mesh_ddp.get_group("DDP")

    return _process_group_dict


def get_process_group(name: str) -> dist.ProcessGroup:
    """Get a process group by name."""
    if name not in _process_group_dict:
        raise ValueError(f"Process group '{name}' not found.")
    return _process_group_dict[name]

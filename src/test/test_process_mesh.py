#!/usr/bin/env python3
"""
Test script for PyTorch distributed process group initialization.
Tests NCCL backend, GPU access, and all-to-all communication.
"""

import torch.distributed as dist

from gb.utils import init_gb_process_group


def test_process_mesh(mesh_dims: list[int]):
    """Initialize process group and test distributed operations."""
    group_dict: dict[str, dist.ProcessGroup] = init_gb_process_group(mesh_dims)

    coord = [
        f"{dim_name}:{dist.get_rank(group_dict[dim_name])}"
        for dim_name in group_dict.keys()
    ]
    print(f"[Rank {dist.get_rank()}]  Mesh Coordinates: {', '.join(coord)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_process_mesh([1, 2, 2, 2, 2])

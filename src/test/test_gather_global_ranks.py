import sys

sys.path.append("./src/")  # for utils import

import torch
import torch.distributed as dist

from evaluation import gather_global_ranks
from gb.utils import init_gb_process_group


def my_gather_global_ranks(group_dict):
    dtp_group = group_dict["DTP"]
    spatial_group = group_dict["JSpatial"]
    channel_group = group_dict["JChannel"]
    up_group = group_dict["SP"]
    dp_group = group_dict["DP"]
    ep_group = group_dict["Expert"]

    mesh_dims = [
        ep_group.size(),
        dp_group.size(),
        up_group.size(),
        spatial_group.size(),
        channel_group.size(),
    ]
    up_stride = mesh_dims[-1] * mesh_dims[-2]
    dp_stride = torch.prod(torch.tensor(mesh_dims[-3:]))
    ep_stride = torch.prod(torch.tensor(mesh_dims[-4:]))
    gathered = torch.arange(0, up_group.size() * up_stride, up_stride)
    gathered += (
        dtp_group.rank() + dp_group.rank() * dp_stride + ep_group.rank() * ep_stride
    )

    return gathered.to(torch.int32)


def test_gather_global_ranks(mesh_dims: list[int]):
    """Initialize process group and test distributed operations."""
    group_dict: dict[str, dist.ProcessGroup] = init_gb_process_group(mesh_dims)
    up_group = group_dict["SP"]

    device_id = torch.cuda.current_device()

    device = f"cuda:{device_id}"

    my_gathered = my_gather_global_ranks(group_dict)

    old_gathered = gather_global_ranks(up_group, device)
    print(
        f"Rank {dist.get_rank(group=None)}. Expensive communication stuff {old_gathered}, Charlie and Markus awesome hacking stuff {my_gathered}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    test_gather_global_ranks([1, 2, 8, 2, 2])

"""Communication functions for evaluation metrics."""

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


@torch.no_grad()
def gather_along_dimension(
    x: Tensor,
    group: Optional[ProcessGroup] = None,
    dim: int = -1,
    variable_size: bool = False,
) -> Tensor:
    """
    Gather and stack tensors along a specified dimension across a process group.

    Parameters
    ----------
    x : Tensor
        The input tensor to be gathered.
    group : ProcessGroup | None, default=None
        The process group to perform the gather operation within.
        If None or contains only one process, the input tensor is returned as-is.
    dim : int, default=-1
        The dimension along which to gather and concatenate the tensors.
    variable_size : bool, default=False
        If True, enables handling of tensors that may have different sizes
        along the specified dimension by padding them to the global
        maximum size before gathering.

    Returns
    -------
    Tensor
        A tensor containing the concatenated results from all processes in
        the group along the specified dimension.
    """
    if group is None or group.size() == 1:
        return x

    if not variable_size:
        # Efficient gather for tensors of identical shape
        gathered = [torch.zeros_like(x) for _ in range(group.size())]
        dist.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=dim)

    # Handle tensors with potentially different sizes along 'dim'
    local_size = x.shape[dim]
    all_sizes = [
        torch.tensor(0, device=x.device, dtype=torch.int32) for _ in range(group.size())
    ]
    dist.all_gather(
        all_sizes,
        torch.tensor(local_size, device=x.device, dtype=torch.int32),
        group=group,
    )
    global_max_size = max([size.item() for size in all_sizes])

    # Pad local tensor to the global maximum size
    if local_size < global_max_size:
        pad_size = list(x.shape)
        pad_size[dim] = int(global_max_size - local_size)
        padding = torch.zeros(pad_size, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, padding], dim=dim)
    else:
        x_padded = x.clone()

    # Gather tensors as objects to handle the padding/structure
    gathered_t_obj = [torch.zeros_like(x_padded) for _ in range(group.size())]
    dist.all_gather_object(gathered_t_obj, x_padded, group=group)

    # Remove padding and concatenate
    multislice = [[slice(None)] * x.dim() for _ in range(group.size())]
    for i in range(group.size()):
        multislice[i][dim] = slice(0, all_sizes[i].item())

    # Slice the gathered tensors to remove padding and move to the correct device
    gathered_t_obj = [
        tensor[*(multislice[i])].to(x.device) for i, tensor in enumerate(gathered_t_obj)
    ]

    return torch.cat(gathered_t_obj, dim=dim)

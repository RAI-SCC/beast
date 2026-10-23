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
    """Concatenate the tensors of all ranks in a process group along a dimension.

    Parameters
    ----------
    x : Tensor
        Local tensor to gather.
    group : ProcessGroup, optional
        Process group to gather over. If None or of size 1, `x` is returned
        unchanged.
    dim : int, default: -1
        Dimension to concatenate along.
    variable_size : bool, default: False
        Whether the size of `x` along `dim` may differ between ranks. Sizes
        are then exchanged and the tensors padded to the maximum before
        gathering, at extra communication cost.

    Returns
    -------
    Tensor
        The concatenation of the tensors of all ranks, in rank order.
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
    global_max_size = torch.stack(all_sizes).max().item()

    # Pad local tensor to the global maximum size
    if local_size < global_max_size:
        pad_size = list(x.shape)
        pad_size[dim] = int(global_max_size - local_size)
        padding = torch.zeros(pad_size, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([x, padding], dim=dim)
    else:
        x_padded = x

    # Gather tensors as objects to handle the padding/structure
    gathered_t_obj = [torch.zeros_like(x_padded) for _ in range(group.size())]
    dist.all_gather(gathered_t_obj, x_padded, group=group)

    # Remove padding and concatenate
    multislice = [[slice(None)] * x.dim() for _ in range(group.size())]
    for i in range(group.size()):
        multislice[i][dim] = slice(0, all_sizes[i].item())

    # Slice the gathered tensors to remove padding and move to the correct device
    gathered_t_obj = [
        tensor[*(multislice[i])].to(x.device) for i, tensor in enumerate(gathered_t_obj)
    ]

    return torch.cat(gathered_t_obj, dim=dim)

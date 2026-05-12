"""Contains metrics that are evaluated inside the validation loop, but require special handling, such as in-loop communication."""

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


def calculate_crps(
    prediction: Tensor, target: Tensor, up_group: Optional[ProcessGroup] = None
) -> Tensor:
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for the given predictions and targets.

    Parameters
    ----------
    prediction : Tensor
        The predicted ensemble on the local device.
    target : Tensor
        The true values.
    up_group : Optional[ProcessGroup], default=None
        The process group for uncertainty parallelization. If None, the calculation is done locally without.

    Returns
    -------
    Tensor
        The calculated CRPS synchronized across UP processes.
    """
    if up_group is None or up_group.size() == 1:
        return _calculate_crps_locally(prediction, target)

    mae_part = torch.abs((prediction - target.unsqueeze(dim=0))).mean(dim=0)
    mae_handle = dist.all_reduce(
        mae_part, op=dist.ReduceOp.AVG, group=up_group, async_op=True
    )

    local_rank = up_group.rank()
    world_size = up_group.size()

    samples_per_worker = prediction.shape[0]
    total_samples = samples_per_worker * world_size
    denominator = 2 * total_samples * (total_samples - 1)

    # local spread
    local_diffs = torch.abs((prediction.unsqueeze(0) - prediction.unsqueeze(1))).sum(
        dim=(0, 1)
    )
    spread_part = local_diffs / denominator

    # MGR buffers
    send_buffer = prediction.clone()
    recv_buffer = torch.zeros_like(prediction)

    next_rank = (local_rank + 1) % world_size
    prev_rank = (local_rank - 1 + world_size) % world_size

    for _ in range(1, world_size):
        # swap buffers
        ops = [
            dist.P2POp(dist.isend, send_buffer, group_peer=next_rank, group=up_group),
            dist.P2POp(dist.irecv, recv_buffer, group_peer=prev_rank, group=up_group),
        ]
        if local_rank % 2 == 1:
            ops.reverse()  # avoid deadlock by reversing order for odd ranks

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        # calculate cross diffs
        cross_diffs = torch.abs(
            (prediction.unsqueeze(dim=0) - recv_buffer.unsqueeze(dim=1))
        ).sum(dim=(0, 1))
        spread_part += cross_diffs / denominator

        # adjust buffers for next loop
        send_buffer.copy_(recv_buffer)

    # final sync
    dist.all_reduce(spread_part, op=dist.ReduceOp.SUM, group=up_group)

    mae_handle.wait()
    crps = mae_part - spread_part

    return crps


def _calculate_crps_locally(prediction: Tensor, target: Tensor) -> Tensor:
    """
    Calculate the CRPS without parallelization of the samples.

    Parameters
    ----------
    prediction : Tensor
        The predicted ensemble.
    target : Tensor
        The true values.

    Returns
    -------
    Tensor
        The calculated CRPS.
    """
    crps = torch.mean(torch.abs(prediction - target.unsqueeze(dim=0)), dim=0)

    sample_size = prediction.shape[0]
    if sample_size > 1:
        crps -= torch.sum(
            torch.abs(prediction.unsqueeze(dim=0) - prediction.unsqueeze(dim=1)),
            dim=(0, 1),
        ) / (2 * sample_size * (sample_size - 1))

    return crps

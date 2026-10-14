"""Metrics evaluated in the validation loop that need special handling, such as communication."""

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


def crps(
    prediction: Tensor,
    target: Tensor,
    up_group: Optional[ProcessGroup] = None,
    sample_dim: int = 0,
    batch_dim: int = 1,
) -> Tensor:
    """Compute the continuous ranked probability score (CRPS) of an ensemble.

    If `up_group` has more than one rank, the ensemble is spread over its
    ranks, and the pairwise sample differences are computed by passing the
    samples around the ranks in a ring.

    Parameters
    ----------
    prediction : Tensor
        Local part of the predicted ensemble.
    target : Tensor
        True values, without a sample dimension.
    up_group : ProcessGroup, optional
        The ``SP`` process group the ensemble is spread over. If None or of
        size 1, only the local
        ensemble is used.
    sample_dim : int, default: 0
        Dimension of `prediction` holding the ensemble samples.
    batch_dim : int, default: 1
        Dimension of `prediction` holding the batch, averaged over.

    Returns
    -------
    Tensor
        The CRPS, averaged over the batch and identical on all UP ranks.
    """
    if up_group is None or up_group.size() == 1:
        return _calculate_crps_locally(prediction, target, sample_dim, batch_dim)

    mae_part = torch.abs((prediction - target.unsqueeze(dim=sample_dim))).mean(
        dim=sample_dim
    )
    mae_handle = dist.all_reduce(
        mae_part, op=dist.ReduceOp.AVG, group=up_group, async_op=True
    )

    local_rank = up_group.rank()
    world_size = up_group.size()

    samples_per_worker = prediction.shape[sample_dim]
    total_samples = samples_per_worker * world_size
    denominator = 2 * total_samples * (total_samples - 1)

    # local spread
    local_diffs = torch.abs(
        (
            prediction.unsqueeze(dim=sample_dim)
            - prediction.unsqueeze(dim=(sample_dim + 1))
        )
    ).sum(dim=(sample_dim, sample_dim + 1))
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
            (
                prediction.unsqueeze(dim=sample_dim)
                - recv_buffer.unsqueeze(dim=sample_dim + 1)
            )
        ).sum(dim=(sample_dim, sample_dim + 1))
        spread_part += cross_diffs / denominator

        # adjust buffers for next loop
        send_buffer.copy_(recv_buffer)

    # final sync
    dist.all_reduce(spread_part, op=dist.ReduceOp.SUM, group=up_group)

    mae_handle.wait()
    crps = mae_part - spread_part
    crps = crps.mean(
        dim=(batch_dim if batch_dim < sample_dim else batch_dim - 1)
    )  # average over batch

    return crps


def _calculate_crps_locally(
    prediction: Tensor, target: Tensor, sample_dim: int = 0, batch_dim: int = 1
) -> Tensor:
    """Compute the CRPS of an ensemble held entirely on this rank.

    Parameters
    ----------
    prediction : Tensor
        The predicted ensemble.
    target : Tensor
        True values, without a sample dimension.
    sample_dim : int, default: 0
        Dimension of `prediction` holding the ensemble samples.
    batch_dim : int, default: 1
        Dimension of `prediction` holding the batch, averaged over.

    Returns
    -------
    Tensor
        The CRPS, averaged over the batch.
    """
    crps = torch.mean(
        torch.abs(prediction - target.unsqueeze(dim=sample_dim)), dim=sample_dim
    )

    sample_size = prediction.shape[sample_dim]
    if sample_size > 1:
        crps -= torch.sum(
            torch.abs(
                prediction.unsqueeze(dim=sample_dim)
                - prediction.unsqueeze(dim=sample_dim + 1)
            ),
            dim=(sample_dim, sample_dim + 1),
        ) / (2 * sample_size * (sample_size - 1))

    crps = crps.mean(
        dim=(batch_dim if batch_dim < sample_dim else batch_dim - 1)
    )  # average over batch

    return crps

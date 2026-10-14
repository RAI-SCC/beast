"""Tests for beast.data.sampler"""

import pytest
import torch
from torch.utils.data import TensorDataset

from beast.data.sampler import ResumableDistributedSampler


def make_sampler(
    n_items: int, rank: int, drop_last: bool, resume_idx: int = 0
) -> ResumableDistributedSampler:
    """Create a shuffling sampler that splits ``n_items`` over two ranks."""
    return ResumableDistributedSampler(
        TensorDataset(torch.arange(n_items)),
        num_replicas=2,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=drop_last,
        resume_idx=resume_idx,
    )


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("n_items", [10, 11])
def test_resumable_sampler(n_items: int, rank: int, drop_last: bool) -> None:
    full_epoch = list(make_sampler(n_items, rank, drop_last))
    for resume_idx in range(len(full_epoch) + 1):
        sampler = make_sampler(n_items, rank, drop_last, resume_idx)
        indices = list(sampler)
        assert len(indices) == len(sampler), (
            f"Sampler yielded {len(indices)} indices but has length {len(sampler)}"
        )
        assert indices == full_epoch[resume_idx:], (
            f"Resuming at {resume_idx} does not continue the epoch"
        )

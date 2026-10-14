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
        length = len(sampler)
        indices = list(sampler)
        assert len(indices) == length, (
            f"Sampler yielded {len(indices)} indices but has length {length}"
        )
        assert indices == full_epoch[resume_idx:], (
            f"Resuming at {resume_idx} does not continue the epoch"
        )
        assert len(sampler) == len(full_epoch) and list(sampler) == full_epoch, (
            "The pass after a resumed pass is not a full epoch"
        )


def test_resume_idx_reset_with_last_index() -> None:
    sampler = make_sampler(10, rank=0, drop_last=False, resume_idx=3)
    iterator = iter(sampler)
    for _ in range(len(sampler) - 1):
        next(iterator)
    assert sampler.resume_idx == 3, "resume_idx was reset before the last index"
    next(iterator)
    assert sampler.resume_idx == 0, "resume_idx was not reset with the last index"


def test_resume_at_end_of_epoch() -> None:
    sampler = make_sampler(10, rank=0, drop_last=False, resume_idx=5)
    assert len(sampler) == 0
    assert list(sampler) == []
    assert sampler.resume_idx == 0, "resume_idx was not reset after an empty pass"

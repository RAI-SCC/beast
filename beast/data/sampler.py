"""Custom resumable distributed sampler."""

from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """Distributed sampler that can resume part-way through an epoch.

    The index order of an epoch depends only on `seed` and the epoch, so a
    resumed run skips the first `resume_idx` indices of this rank's shard and
    continues where it stopped.

    Parameters
    ----------
    dataset : Dataset
        Dataset to sample from.
    num_replicas : int, optional
        Number of ranks to split the data over, usually the size of the ``DP``
        process group. Defaults to the world size.
    rank : int, optional
        Rank of this process among the `num_replicas` ranks. Defaults to the
        global rank.
    shuffle : bool, default: True
        Whether to shuffle the indices every epoch.
    seed : int, default: 0
        Shuffling seed. Must be identical on all ranks.
    drop_last : bool, default: False
        Whether to drop the tail of the data so that every rank gets the same
        number of indices. If False, indices are repeated instead.
    resume_idx : int, default: 0
        Number of leading indices of this rank's shard to skip in the first pass.

    Attributes
    ----------
    resume_idx : int
        Number of leading indices of this rank's shard to skip in the next pass.
        Reset to 0 when a pass returns its last index.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        resume_idx: int = 0,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.resume_idx = resume_idx

    def __iter__(self) -> Iterator[int]:
        """Iterate over this rank's indices for the current epoch.

        Returns
        -------
        Iterator[int]
            The indices of this rank's shard after skipping the first
            `resume_idx`. Unless `drop_last` is set, they are repeated
            until there are as many as in a full epoch.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        # Apply rank sharding (DistributedSampler's logic)
        indices = indices[self.rank :: self.num_replicas]

        if not self.drop_last:
            while len(indices) < self.num_samples:
                indices += indices
        indices = indices[: self.num_samples]

        # Skip to resume point.
        indices = indices[self.resume_idx :]
        yield from indices[:-1]
        self.resume_idx = 0
        yield from indices[-1:]

    def __len__(self) -> int:
        """Return the number of indices left after skipping `resume_idx`.

        Returns
        -------
        int
            The number of indices of this rank for a full epoch, without the
            first ``resume_idx``.
        """
        return max(0, self.num_samples - self.resume_idx)

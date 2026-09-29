"""Custom resumable distributed sampler."""

from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """Distributed sampler that can resume part-way through an epoch.

    The full index sequence of an epoch is generated up front, so resuming only
    requires skipping the first ``resume_idx`` indices of this rank's shard.
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
        """Initialize the sampler.

        Parameters
        ----------
        dataset : Dataset
            Dataset to sample from.
        num_replicas : int, optional
            Number of data-parallel replicas, i.e. the size of the DP group.
        rank : int, optional
            Data-parallel rank of this process.
        shuffle : bool, default: True
            Whether to shuffle the indices each epoch.
        seed : int, default: 0
            Shuffling seed. Must be identical on all ranks.
        drop_last : bool, default: False
            Drop the tail of the data to make it evenly divisible across
            replicas. If False, indices are repeated to pad it instead.
        resume_idx : int, default: 0
            Number of leading indices of this rank's shard to skip.
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.resume_idx = resume_idx

    def __iter__(self) -> Iterator[int]:
        """Iterate over this rank's remaining indices for the current epoch."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        # Apply rank sharding (DistributedSampler's logic)
        indices = indices[self.rank :: self.num_replicas]

        # Skip to resume point
        indices = indices[self.resume_idx :]

        if not self.drop_last:
            while len(indices) < self.num_samples:
                indices += indices
            indices = indices[: self.num_samples]

        return iter(indices)

    def __len__(self) -> int:
        """Return the number of indices left after skipping ``resume_idx``."""
        return max(0, self.num_samples - self.resume_idx)

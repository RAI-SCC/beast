"""Custom resumable distributed sampler."""

from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """
    Custom sampler that allows for restart part-way through epochs.

    The distributed sampler pre-generates the sequences of indices associated with an epoch.
    It can restart halfway through an epoch by slicing the indices when necessary.
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
        """
        Initialize ResumableDistributedSampler.

        Inputs
        ------
        dataset: Dataset
            Dataset that we are sampling from
        num_replicas: Optional[int]
            Number of processes participating in distributed training per shard.
            i.e., size of dp group.
        rank: Optional[int]
            data-parallel rank.
        shuffle: bool
            Boolean flag for shuffling.
        seed: int
            Random seed. Should be fixed and same for all ranks in process mesh.
        drop_last: bool
            If True, the sampler will drop the tail of the data to make it evenly divisible
            across the number of replicas.
            If False, the sampler will add extra indices to make the data evenly divisible
            across the replicas.
        resume_idx: int
            Starting index for the indices. i.e., the global indices wlil be
            sliced starting from resume_idx:
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.resume_idx = resume_idx

    def __iter__(self) -> Iterator[int]:
        """Generate permutation of dataset."""
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
        """Length of the dataset, adjusted for the resume_idx index."""
        return max(0, self.num_samples - self.resume_idx)

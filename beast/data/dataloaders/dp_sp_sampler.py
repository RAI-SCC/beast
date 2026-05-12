import math

import torch
from torch.utils.data import Sampler


class DP_SP_Sampler(Sampler):
    def __init__(self, dataset, dp_group, shuffle=True, seed=0):
        self.dataset = dataset
        self.dp_group = dp_group
        self.shuffle = shuffle
        self.seed = seed

        self.dp_rank = dp_group.rank()
        self.dp_world_size = dp_group.size()

        self.num_samples = int(math.ceil(len(dataset) / self.dp_world_size))
        self.total_size = self.num_samples * self.dp_world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = list(range(len(self.dataset)))

        if self.shuffle:
            indices = torch.randperm(len(indices), generator=g).tolist()

        # Pad to evenly divide across DP
        if len(indices) < self.total_size:
            indices += indices[: self.total_size - len(indices)]

        # Shard by DP rank only
        indices = indices[self.dp_rank : self.total_size : self.dp_world_size]

        return iter(indices)

    def __len__(self):
        return self.num_samples

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

from typing import Optional

class ResumableDistributedSampler(DistributedSampler):
    """
    Custom sampler that allows for restart part-way through epochs.
    
    The distributed sampler pre-generates the sequences of indices associated with an epoch.
    """
    def __init__(
        self, 
        dataset: Dataset,
        num_replicas: Optional[int]=None,
        rank: Optional[int]=None, 
        shuffle: bool=True, 
        seed: int=0, 
        drop_last: bool=False, 
        resume_idx: int=0
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.resume_idx = resume_idx
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Apply rank sharding (DistributedSampler's logic)
        indices = indices[self.rank::self.num_replicas]
        
        # Skip to resume point
        indices = indices[self.resume_idx:]
        
        if not self.drop_last:
            while len(indices) < self.num_samples:
                indices += indices
            indices = indices[:self.num_samples]
        
        return iter(indices)
    
    def __len__(self):
        return max(0, self.num_samples - self.resume_idx)
    
    

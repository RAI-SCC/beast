from beast import get_model
import torch
from typing import Any
import pandas as pd
from beast.data.dataloader import get_dataloader
import torch.distributed as dist
from beast.comm import create_process_mesh_dict, get_process_group, shutdown_comm
import os
from beast.losses.up_loss import UPAllGaussLoss
from beast.utils import set_all_seeds
from torch.nn.parallel import DistributedDataParallel

def setup(rank, world_size):
    # Initialize distributed backend
    dist.init_process_group(
        backend="nccl",  # Use NCCL for CUDA
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """
    Clean up the process group.
    """
    dist.destroy_process_group()

def create_mock_config() -> dict[str, Any]:
    """Create a mock dataset and corresponding mock config dictionary."""
    config = {
        "data": {
            "dt": 6,
            "train_data_path": "test_data.zarr",
            "valid_data_path": "test_data.zarr",
            "constant_masks_path": "test_masks",
            "constant_masks": ["mask_0001.npy", "mask_0002.npy", "mask_0003.npy"],
            "normalization_dir": "normalization",
            "train_subset": -1,
            "valid_subset": -1,
            "train_time_range": ["2000-01-01", "2000-12-31"],
            "valid_time_range": ["2001-01-01", "2001-12-31"],
            "xlat": 720,
            "xlon": 1440,
            "n_variables": 84,
            "n_in_timesteps": 1,
            "n_out_timesteps": 1,
            "prefetch_factor": 2,
            "num_workers": 2,
        },
        "training": {
            "batch_size": 1,
            "dtype": torch.float32,
        },
    }

    return config


base_seed = 42

epochs = 1

mesh_dims = [1,2,2,1,1] # ["Expert", "DP", "SP", "JSpatial", "JChannel"]

rank = int(os.getenv("SLURM_PROCID"))  # Get individual process ID.
world_size = int(os.getenv("SLURM_NTASKS"))  # Total number of processes.
slurm_localid = int(os.getenv("SLURM_LOCALID", rank))


device_id = slurm_localid if torch.cuda.device_count() != 1 else 0
device = torch.device("cuda", device_id)
torch.cuda.set_device(device)

dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method="env://",
    )

create_process_mesh_dict(mesh_dims)

set_all_seeds(base_seed)

# Model Part
model = get_model(
            "BEASTLING",
            bayesian=True,
            device=device,
        )

model.return_log_probs = False

model = DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
        process_group=get_process_group("DDP"),
    )

optimizer = torch.optim.AdamW(model.parameters())


# Dataloader Part
config_dict = create_mock_config()
dtp_rank = dist.get_rank(group=get_process_group("DTP"))
dtp_size = dist.get_world_size(group=get_process_group("DTP"))
dp_rank = dist.get_rank(group=get_process_group("DP"))
dp_size = dist.get_world_size(group=get_process_group("DP"))

train_loader = get_dataloader(
    config_dict,
    dtp_rank,
    dtp_size,
    dp_rank,
    dp_size,
    mode="train",
    dtype=torch.float32,
)

valid_loader = get_dataloader(
    config_dict,
    dtp_rank,
    dtp_size,
    dp_rank,
    dp_size,
    mode="validation",
    dtype=torch.float32,
)

loss_fn = UPAllGaussLoss(
    model= model,
    dataset_size= len(train_loader.dataset),
    up_group= get_process_group("SP"),
)

# Train one epoch
set_all_seeds(base_seed + dist.get_rank(group=get_process_group("SP")))


model.train()
loss_sum = torch.tensor([0.0], device=device, requires_grad=False)
n_samples = 1

print(type(model.module))

for epoch in range(epochs):
    if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
        
    for x,y,_,_ in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, samples=n_samples)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.detach()
    
    average_loss = loss_sum.item() / max(1, len(dataloader))
    print(average_loss)
        

shutdown_comm()


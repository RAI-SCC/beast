import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from gb.dataloaders.dataset_0_25_degree import ERA5DatasetDistributed


def get_dataloader(config_dict, dtp_process_group, dp_process_group, mode="train"):
    dt = config_dict["data"]["dt"]
    batch_size = config_dict["training"]["batch_size"]
    if mode == "train":
        data_path = config_dict["data"]["train_data_path"]
    else:
        data_path = config_dict["data"]["valid_data_path"]

    dataset = ERA5DatasetDistributed(
        config_dict,
        data_path,
        mode=mode,
        patch_size=(2, 4, 4),
        rank=dtp_process_group.rank(),
        forecast_length=1,
        normalize=True,
        dt=dt,
    )
    if config_dict["data"]["train_subset"] > 0:
        subset_indices = torch.arange(config_dict["data"]["train_subset"])
        dataset = Subset(dataset, subset_indices)

    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        rank=dp_process_group.rank(),
        num_replicas=dp_process_group.size(),
        seed=42,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=config_dict["data"]["prefetch_factor"],
        num_workers=config_dict["data"]["num_workers"],
    )
    return dataloader

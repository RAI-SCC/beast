import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
)
from torch.utils.data.distributed import DistributedSampler

from gb.dataloaders.dataset_0_25_degree import (
    ERA5DatasetDistributed,
    ERA5TwoTimestepsDatasetDistributed,
)
from gb.dataloaders.dataset_1_5_degree import (
    DummyDataset,
    ERA5PreprocessedDataset15degree,
    ERA5TwoTimestepsPreprocessedDataset15degree,
)


class GBDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __next__(self):
        start = time.time_ns()

        data = super().__next__()
        end = time.time_ns()

        return *data, start, end


class GBDataLoader(DataLoader):
    def _get_iterator(self) -> _BaseDataLoaderIter:
        return GBDataLoaderIter(self)


def get_dataloader(
    config_dict,
    dtp_process_group,
    dp_process_group,
    device,
    mode="train",
    dtype=torch.float32,
):
    dt = config_dict["data"]["dt"]
    batch_size = config_dict["training"]["batch_size"]
    if config_dict["data"]["xlat"] == 120 and config_dict["data"]["xlon"] == 240:
        if os.path.exists(config_dict["data"]["train_data_path"]):
            if config_dict["data"]["n_in_timesteps"] == 2:
                dataset = ERA5TwoTimestepsPreprocessedDataset15degree(
                    config_dict, mode=mode, normalize=True, dt=dt, dtype=dtype
                )
            else:
                dataset = ERA5PreprocessedDataset15degree(
                    config_dict, mode=mode, normalize=True, dt=dt, dtype=dtype
                )
        else:
            if dist.get_rank(group=None) == 0:
                print("train_data_path not found. Running Dummy Dataset")
            dataset = DummyDataset(
                config_dict, mode=mode, normalize=True, dt=dt, dtype=dtype
            )
    else:
        if config_dict["data"]["n_in_timesteps"] == 2:
            dataset = ERA5TwoTimestepsDatasetDistributed(
                config_dict,
                mode=mode,
                process_group=dtp_process_group,
                device=device,
                normalize=True,
                dt=dt,
                dtype=dtype,
            )
        else:
            dataset = ERA5DatasetDistributed(
                config_dict,
                mode=mode,
                process_group=dtp_process_group,
                device=device,
                normalize=True,
                dt=dt,
                dtype=dtype,
            )

    if mode == "train" and config_dict["data"]["train_subset"] > 0:
        subset_indices = torch.arange(config_dict["data"]["train_subset"])
        dataset = Subset(dataset, subset_indices)
    elif mode == "validation" and config_dict["data"]["valid_subset"] > 0:
        subset_indices = torch.arange(config_dict["data"]["valid_subset"])
        dataset = Subset(dataset, subset_indices)

    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        rank=dp_process_group.rank(),
        num_replicas=dp_process_group.size(),
        seed=42,
    )

    dataloader = GBDataLoader(
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

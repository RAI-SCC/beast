import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from gb.dataloaders.dataloader_0_25_degree import GBDataLoader
from gb.dataloaders.dataset_1_5_degree import (
    DummyDataset,
    ERA5Dataset15degree,
    ERA5TwoTimestepsDataset15degree,
    WeatherTestDataset,
)


def get_dataloader_1_5_deg(
    config_dict, dtp_process_group, dp_process_group, mode="train", dtype=torch.float32
):
    dt = config_dict["data"]["dt"]
    batch_size = config_dict["training"]["batch_size"]

    if os.path.exists(config_dict["data"]["train_data_path"]):
        if config_dict["data"]["n_in_timesteps"] == 2:
            dataset = ERA5TwoTimestepsDataset15degree(
                config_dict, mode=mode, normalize=True, dt=dt, dtype=dtype
            )
        else:
            dataset = ERA5Dataset15degree(
                config_dict, mode=mode, normalize=True, dt=dt, dtype=dtype
            )
    else:
        dataset = DummyDataset(
            config_dict, mode=mode, normalize=True, dt=dt, dtype=dtype
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


def get_test_dataloader(config, pg):
    dataset = WeatherTestDataset(config)

    dataset_info = {
        "out_surface_variables": dataset.out_surface_variables,
        "out_pressure_variables": dataset.out_pressure_variables,
        "out_means": dataset.out_means,
        "out_stds": dataset.out_stds,
        "resolution": dataset.resolution,
        "padded_resolution": dataset.padded_resolution,
        "padding": dataset.padding,
        "size": len(dataset),
    }

    if dist.is_initialized():
        # pg = get_process_group("DP")

        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            rank=pg.rank(),
            num_replicas=pg.size(),
            seed=42,  # Put in config?
        )
        # sampler = DP_SP_Sampler(dataset, dp_group=pg, sp_group=spg, validate=validate_sampler, shuffle=True, seed=0)

        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            sampler=sampler,
            num_workers=config["data"]["num_workers"],
            prefetch_factor=config["data"]["prefetch_factor"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            sampler=None,
            num_workers=config["data"]["num_workers"],
            prefetch_factor=config["data"]["prefetch_factor"],
            drop_last=True,
            persistent_workers=True,
        )

    return dataloader, dataset_info

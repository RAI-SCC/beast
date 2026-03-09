# from .dataset import KanguDataset, UnpaddedDataset, GlobePaddedKanguDataset, AugmentedKanguDataset
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gb.dataloaders.dataset_1_5_degree import WeatherTestDataset


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

"""Custom GB Dataloaders and Dataloader factory."""

import os
import time
import warnings
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
)

from beast.data.era5 import (
    ERA5Distributed,
    ERA5Dummy,
)
from beast.data.sampler import ResumableDistributedSampler
from beast.utils import set_dtype

DSYNC_INIT = os.getenv("BEAST_DL_DSYNC_INIT", "False") in ["True", "1"]


class GBDataLoaderIter(_MultiProcessingDataLoaderIter):
    """
    Custom Data Loader Iter, that tracks the start and stop time of the ``__next__`` functionality.

    Inherits from PyTorchs ``_MultiProcessingDataLoaderIter``
    """

    def __next__(self) -> tuple[Any, float, float]:
        """
        Get the next item of the Iterator.

        Returns
        -------
        *data : torch.Tensor
            The next data item
        start : float
            The time stamp before fetching the next data item in ns
        stop : float
            The time stamp after fetching the next data item in ns
        """
        start = time.perf_counter_ns()

        data = super().__next__()
        end = time.perf_counter_ns()

        return *data, start, end


class GBDataLoader(DataLoader):
    """
    Custom Data Loader, that tracks the start and stop time of the ``__next__`` functionality.

    Calls GBDataLoaderIter.
    """

    def _get_iterator(self) -> _BaseDataLoaderIter:
        return GBDataLoaderIter(self)


def get_dataloader(
    config_dict: dict[str, Any],
    dtp_rank: int,
    dtp_size: int,
    dp_rank: int,
    dp_size: int,
    mode: str = "train",
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    **kwargs: dict,
) -> GBDataLoader:
    """
    Create a GBDataLoader instance from config specifications.

    If the config_dict contains a path to a valid zarr directory of ERA5, this function will instantiate a suitable dataset, perform potential subsetting and return an iterable Dataloader.
    If no valid path is given (i.e. if the path doesn't exist), it will create a dummy dataset for the time range specified in the config file, filled with random numbers.

    Parameters
    ----------
    config_dict : Dict
        The configuration dictionary, that contains information on the dataset, i.e. the path to all relevant files, specifications of the time frame to be used, etc.
    dtp_rank : int
        The rank in the domain-tensor-parallel process group, that shares a model and a data item
    dtp_size : int
        The size of the domain-tensor-parallel process group, that shares a model and a data item
    dp_rank : int
        The rank in the data-parallel process group, that shares a model and a data item
    dp_size : int
        The size of the data-parallel process group (torch.DistributedDataParallel)
    mode : str
        The type of dataset and corresponding loader. Can be either "train" or "validation". "test" is not supported yet
    transform: Callable
        The transformation to be applied to the input weather data
    target_transform: Callable
        The transformation to be applied to the output weather data. If None, defaults to the transformation given in transform argument

    Returns
    -------
    dataloader : GBDataLoader
        The dataloader instance
    """
    data_path = config_dict["data"]["data_path"]

    if mode == "train":
        years = config_dict["data"]["train_time_range"]
    elif mode == "validation":
        years = config_dict["data"]["valid_time_range"]
    else:
        years = None

    batch_size = config_dict["training"]["batch_size"]
    dtype = set_dtype(config_dict["training"]["dtype"])

    dataset: torch.utils.data.Dataset
    if os.path.exists(data_path):
        dataset = ERA5Distributed(
            pg_rank=dtp_rank,
            pg_size=dtp_size,
            data_path=data_path,
            constant_masks_path=config_dict["data"]["constant_masks_path"],
            constant_masks=config_dict["data"]["constant_masks"],
            normalization_dir=config_dict["data"]["normalization_dir"],
            time_range=years,
            dt=config_dict["data"]["dt"],
            n_in_timesteps=config_dict["data"]["n_in_timesteps"],
            n_out_timesteps=config_dict["data"]["n_out_timesteps"],
            dtype=dtype,
            transform=transform,
            target_transform=target_transform,
            dsync_access=DSYNC_INIT,
            **kwargs,
        )
    else:
        if dist.get_rank(group=None) == 0:
            warnings.warn(
                f"`{data_path}` not found. Running dummy dataset for time range {years}",
                UserWarning,
            )

        shape = (
            config_dict["data"]["n_variables"],
            config_dict["data"]["xlat"],
            config_dict["data"]["xlon"],
        )
        dataset = ERA5Dummy(
            pg_rank=dtp_rank,
            pg_size=dtp_size,
            global_shape=shape,
            time_range=years,
            dt=config_dict["data"]["dt"],
            n_in_timesteps=config_dict["data"]["n_in_timesteps"],
            n_out_timesteps=config_dict["data"]["n_out_timesteps"],
            dtype=dtype,
            transform=transform,
            target_transform=target_transform,
            **kwargs,
        )

    # if train_subset <=0, the entire dataset will be used for training
    if (
        mode == "train"
        and config_dict["data"]["train_subset"] > 0
        and (config_dict["data"]["train_subset"] < len(dataset))
    ):
        subset_indices = torch.arange(config_dict["data"]["train_subset"])
        dataset = Subset(dataset, subset_indices.tolist())
    # if valid_subset <=0, the entire dataset will be used for validation
    elif (
        mode == "validation"
        and config_dict["data"]["valid_subset"] > 0
        and (config_dict["data"]["valid_subset"] < len(dataset))
    ):
        subset_indices = torch.arange(config_dict["data"]["valid_subset"])
        dataset = Subset(dataset, subset_indices.tolist())

    sampler = ResumableDistributedSampler(
        dataset,
        shuffle=True,
        rank=dp_rank,
        num_replicas=dp_size,
        seed=42,
        resume_idx=0,  # set from checkpoint on resume
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
        persistent_workers=True,
    )

    return dataloader

"""Timing data loader and a config-driven dataloader factory."""

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


class BeastDataLoaderIter(_MultiProcessingDataLoaderIter):  # numpydoc ignore=PR02
    """Multiprocessing data loader iterator that also times each batch fetch.

    Parameters
    ----------
    loader : DataLoader
        The data loader to iterate over.
    """

    def __next__(self) -> tuple[Any, float, float]:
        """Fetch the next batch and time the fetch.

        Returns
        -------
        tuple
            The items of the batch, followed by the ``time.perf_counter_ns``
            timestamps taken before and after the fetch.
        """
        start = time.perf_counter_ns()

        data = super().__next__()
        end = time.perf_counter_ns()

        return *data, start, end


class BeastDataLoader(DataLoader):
    """Data loader that also reports how long each batch took to fetch.

    It takes the same arguments as :class:`torch.utils.data.DataLoader`. Each
    batch is followed by the ``time.perf_counter_ns`` timestamps taken before
    and after fetching it.

    See Also
    --------
    torch.utils.data.DataLoader : The data loader this class extends.
    """

    def _get_iterator(self) -> _BaseDataLoaderIter:
        """Create the iterator that times each batch fetch.

        Returns
        -------
        _BaseDataLoaderIter
            A :class:`BeastDataLoaderIter` over this data loader.
        """
        return BeastDataLoaderIter(self)


def get_dataloader(
    config_dict: dict[str, Any],
    dtp_rank: int,
    dtp_size: int,
    dp_rank: int,
    dp_size: int,
    mode: str = "train",
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    **kwargs: Any,
) -> BeastDataLoader:
    """Create a :class:`BeastDataLoader` from a configuration dictionary.

    If ``data.data_path`` exists, an :class:`ERA5Distributed` dataset is read
    from it. Otherwise a warning is issued on rank 0 and an :class:`ERA5Dummy`
    dataset of random numbers is used for the configured time range. In
    ``"train"`` and ``"validation"`` mode the dataset is cut to the first
    ``train_subset`` or ``valid_subset`` samples if that value is positive.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration with the sections ``data`` and ``training``, described
        in the configuration reference of the documentation.
    dtp_rank : int
        Rank in the ``DTP`` process group, the ranks that share one data item.
    dtp_size : int
        Size of the ``DTP`` process group.
    dp_rank : int
        Rank in the ``DP`` process group.
    dp_size : int
        Size of the ``DP`` process group.
    mode : {"train", "validation"}, default: "train"
        Selects the time range and subset size.
    transform : Callable, optional
        Transformation applied to the input data.
    target_transform : Callable, optional
        Transformation applied to the target data. Defaults to ``transform``.
    **kwargs
        Passed on to the dataset.

    Returns
    -------
    BeastDataLoader
        The dataloader, sampling with a :class:`ResumableDistributedSampler`.
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

    dataloader = BeastDataLoader(
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

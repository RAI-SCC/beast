"""Dataset definitions."""

import os
from datetime import datetime
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import Dataset

from beast.comm import execute_in_waves


class ERA5Distributed(Dataset):
    """
    Genaralized ERA5 dataset for native and domain-parallel data.

    Loads data from a zarr directory, to return an (input, target) pair for model training.
    Assumes that zarr data is chunked along the time domain, as each time-step marks a fundamental data sampel unit.
    In a distributed (domain-parallel) setting, each process loads only it's shard of the channels and spatial latitude.
    Works only on 2-way channel splitting.
    Both input and target consist of one or more timesteps, which are concatenated in the channel dimension.

    Attributes
    ----------
    dtype : torch.dtype
        dtype of the input and target tensors to be generated
    world_size : int
        The size of the domain-parallel process group.
    local_rank : int
        The rank withing the domain-parallel process group.
    datapath : str
        The path to the zarr directory.
    norm_values: dict
        A dictionary holding the "mean" and "std" numpy arrays with normalization values
    zarr_data :
        The handle to the zarr data directory from which data will be loaded
    years : Tuple(str,st)
        The temporal range to be included into the dataset, in the format`%Y-%m-%d`
    in_timesteps : int
        Number of timesteps to be concatenated for the input; concatenation will be done along the channel dimension
    out_timesteps : int
        Number of timesteps to be concatenated for the target; concatenation will be done along the channel dimension
    dt : int
        The temporal subsampling timestep in hours
    n_data_items:
        Number of data samples (time steps) in self.zarr_data
    xlat : int
        Global size of data samples in self.zarr_data in the latitude dimension; if this is an odd number, the last value will be clipped.
    xlon : int
        Global size of data samples in self.zarr_data in the longitude dimension; if this is an odd number, the last value will be clipped.
    num_vars: int
        Global number of variables (channels) in self.zarr_data
    constant_masks: torch.Tensor
        Tensor with the local shard of the constant masks (chunked in channels and longitude)
    n_masks : int
        Number of constant masks
    in_var : Tuple(int, int)
        The start and stop value of the range of channels for this particluar rank
    in_masks: Tuple(int, int)
        The start and stop value of the range of masks for this particluar rank
    in_out_xlon : Tuple(int, int)
        The start and stop value of the spatial range (in the longitude) for this particluar rank
    input_pad_oddrank: int
        The number of channels that need to be padded on odd ranks for
    transform: Callable
        The transformation to be applied to the input weather data
    target_transform: Callable
        The transformation to be applied to the output weather data. If None, defaults to the transformation given in transform argument


    Parameters
    ----------
    pg_rank : int
        The domain-parallel process group rank.
    pg_size : int
        The size of the domain-parallel process group.
    data_path : str
        The path to the zarr directory.
    constant_masks_path : str
        The path to the constant mask files
    constant_masks : List(str)
        The file names of the constant masks to be used
    normalization_dir : str
        The path to the .npy files containing normalization values `norm_mean.npy` and `norm_std.npy`
    dt : int
        The temporal subsampling timestep in hours
    n_in_timesteps : int
        The number of input time setps
    n_out_timesteps : int
        The number of output time setps
    dtype : torch.dtype
        The dtype for input and target tensors
    dsync_access: bool
        Switch for waved (grouped) accessing of the zarr directory, for desynchronized file system access at scale. Recommended to use when going beyond 512 nodes.
    time_range : Tuple(str, str)
        The temporal range to be included into the dataset, in the format`%Y-%m-%d`
    transform: Callable
        The transformation to be applied to the input weather data
    target_transform: Callable
        The transformation to be applied to the output weather data. If None, defaults to the transformation given in transform argument

    """

    def __init__(
        self,
        pg_rank: int,
        pg_size: int,
        data_path: str,
        constant_masks_path: str,
        constant_masks: List[str],
        normalization_dir: str,
        dt: int = 6,
        n_in_timesteps: int = 1,
        n_out_timesteps: int = 1,
        dtype: torch.dtype = torch.float32,
        dsync_access: bool = False,
        time_range: Tuple[str, str] | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.dt = dt
        self.in_timesteps = n_in_timesteps
        self.out_timesteps = n_out_timesteps

        self.dtype = dtype
        self.world_size = pg_size
        self.local_rank = pg_rank

        self.datapath = data_path
        self.years = time_range
        self.mask_path = constant_masks_path
        mask_list = constant_masks

        self.norm_values: dict[str, np.array] = {}

        def load_data() -> None:
            """Inline utility function for initial data access."""
            try:
                self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")
            except Exception:
                raise OSError(
                    "Given path does not point to a zarr directory. Data must be in zarr format."
                )

            try:
                self.xlat = self.zarr_data.sizes["latitude"]
                self.xlon = self.zarr_data.sizes["longitude"]
            except Exception:
                raise Exception(
                    "Zarr directory has no dimensions `latitude` and/or `longitude`. Cannot access spatial dimension"
                )

            if self.xlat % 2 == 1:
                self.xlat -= 1
            if self.xlon % 2 == 1:
                self.xlon -= 1

            try:
                times = pd.to_datetime(self.zarr_data.time)
            except Exception:
                raise Exception(
                    "Time values of data cannot be accessed. Zarr data must be chunked in the time domain. "
                )

            if self.years is not None:
                train_years = times[(times >= self.years[0]) & (times < self.years[1])]
                self.zarr_data = self.zarr_data.sel(time=train_years)

            try:
                self.n_data_items = self.zarr_data.sizes["time"]
            except Exception:
                raise Exception(
                    "Zarr directory has no dimension `time`. Cannot access temporal dimension"
                )

            try:
                self.n_variables = self.zarr_data.sizes["feature"]
            except Exception:
                raise Exception(
                    "Zarr directory has no dimension `feature`. Cannot access feature dimension"
                )

            self.constant_masks = self._load_constant_mask_2d(self.mask_path, mask_list)
            self.n_masks = self.constant_masks.shape[0]

            self.norm_values["mean"] = np.load(f"{normalization_dir}/norm_mean.npy")
            self.norm_values["std"] = np.load(f"{normalization_dir}/norm_std.npy")

        if dsync_access:
            execute_in_waves(load_data)
        else:
            load_data()

        if self.world_size > 1:
            row, col = self.local_rank % 2, self.local_rank // 2
            # rank 0: [0,0], rank 1 = [1,0], rank 2 = [0,1], rank 3 = [1,1]
            if row == 0:
                self.in_var = (
                    0,
                    (
                        self.n_variables // 2
                        if self.n_variables % 2 == 0
                        else self.n_variables // 2 + 1
                    ),
                )
            else:
                self.in_var = (
                    (
                        self.n_variables // 2
                        if self.n_variables % 2 == 0
                        else self.n_variables // 2 + 1
                    ),
                    self.n_variables,
                )

            # We split the channels into two ranks; however, if we have an odd number of channels this will
            # result in rank 0 having one more variable.
            # since we do this for each timestep we accumulate an inbalance in the rank=1 channel group
            input_inbalance = (self.n_variables % 2) * self.in_timesteps
            output_inbalance = (self.n_variables % 2) * self.out_timesteps
            # We can use the constant masks to counteract this
            # First, we compute how many masks we can add to rank=1 channel group to counteract the imbalance
            self.input_pad_oddrank = 0
            self.output_pad_oddrank = output_inbalance

            # Two cases:
            # If we have more masks then the inbalance, we need to distribute the remaining masks evenly accross channel ranks
            if (self.n_masks - input_inbalance) >= 0:
                remaining_masks = self.n_masks - input_inbalance
                if row == 0:
                    self.in_masks = (0, remaining_masks - remaining_masks // 2)
                else:
                    self.in_masks = (
                        remaining_masks - remaining_masks // 2,
                        self.n_masks,
                    )

                if remaining_masks % 2 == 1:
                    self.input_pad_oddrank = 1

            # If we have fewer masks then the imbalance, we need to pad on rank 1
            else:
                if row == 0:
                    self.in_masks = (0, 0)
                else:
                    self.in_masks = (0, self.n_masks)

                self.input_pad_oddrank = input_inbalance - self.n_masks

            # split into 2 "longitude" pieces for 4-way parallel and 4 for 8-way
            self.in_out_xlon = (
                col * self.xlon // (self.world_size // 2),
                (col + 1) * self.xlon // (self.world_size // 2),
            )

        else:
            # Single-rank (no domain parallelism): all variables.
            self.in_var = (0, self.n_variables)
            self.in_masks = (0, self.n_masks)
            self.in_out_xlon = (0, self.xlon)

        self.transform = transform
        if target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = target_transform

        try:
            _ = self.__getitem__(0)
        except Exception as err:
            raise RuntimeError(
                "Dataset was instantiated, but data items cannot be loaded. Data "
                "seems to be empty, or does not contain `fields`."
            ) from err

    def __len__(self) -> int:
        """Return total number of samples."""
        return int(
            self.n_data_items - (self.in_timesteps + self.out_timesteps - 1) * self.dt
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return single input, target.

        Parameters
        ----------
        idx: int
            Global index of item

        Returns
        -------
        input : torch.Tensor
            The input sample of shape [channel shard * number of input timesteps, xlat, longitudinal shard].

        target : torch.Tensor
            The target sample of shape [channel shard * number of output timesteps, xlat, longitudinal shard].
        """
        target_idx = idx + self.in_timesteps * self.dt

        input_list = []
        for i in range(self.in_timesteps):
            x = self.zarr_data.isel(
                time=(idx + i * self.dt),
                feature=range(self.in_var[0], self.in_var[1]),
                latitude=range(0, self.xlat),
                longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
            )["fields"].values

            x = (
                x
                - self.norm_values["mean"][self.in_var[0] : self.in_var[1]].reshape(
                    -1, 1, 1
                )
            ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]].reshape(
                -1, 1, 1
            )

            input_list.append(torch.tensor(x, dtype=self.dtype))

        input_list.append(self.get_constant_masks().to(self.dtype))
        input = torch.cat(input_list, dim=0)

        # For distributed dataset, we may create inbalances in the variables
        if self.world_size > 1:
            # Odd ranks load second half of variables, so they might have less masks
            if (self.input_pad_oddrank != 0) and (self.local_rank % 2 == 1):
                input = torch.nn.functional.pad(
                    input, (0, 0, 0, 0, 0, self.input_pad_oddrank)
                )

        # Output data
        target_list = []
        for j in range(self.out_timesteps):
            y = self.zarr_data.isel(
                time=(target_idx + j * self.dt),
                feature=range(self.in_var[0], self.in_var[1]),
                latitude=range(0, self.xlat),
                longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
            )["fields"].values

            y = (
                y
                - self.norm_values["mean"][self.in_var[0] : self.in_var[1]].reshape(
                    -1, 1, 1
                )
            ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]].reshape(
                -1, 1, 1
            )

            target_list.append(torch.tensor(y, dtype=self.dtype))

        target = torch.cat(target_list, dim=0)

        if self.world_size > 1:
            # Odd ranks load second half of variables, so they might have less masks
            if (self.output_pad_oddrank != 0) and (self.local_rank % 2 == 1):
                target = torch.nn.functional.pad(
                    target, (0, 0, 0, 0, 0, self.output_pad_oddrank)
                )

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

    def get_constant_masks(self) -> torch.Tensor:
        """
        Return the rank local shard of the constant masks.

        Returns
        -------
        torch.Tensor
            The rank local shard of the constant masks; has shape [number of masks //2, xlat, longitudinal shard].
        """
        return torch.tensor(
            self.constant_masks[
                self.in_masks[0] : self.in_masks[1],
                : self.xlat,
                self.in_out_xlon[0] : self.in_out_xlon[1],
            ]
        )

    def get_feature_weights(self, channel_rank: int, sst_idx: int = 4) -> torch.Tensor:
        """
        Return feature weights with SST being masked over land.

        Parameters
        ----------
        channel_rank : int
            The Jigsaw channel rank of the process.
        sst_idx : int
            The index along the feature dimension at which SST is stored, default: 4.
        """
        feature_weights = torch.ones(
            (self.in_var[1] - self.in_var[0]) * self.out_timesteps,
            self.xlat,
            self.in_out_xlon[1] - self.in_out_xlon[0],
        )
        if channel_rank == 0:
            for i in range(self.out_timesteps):
                feature_weights[i * (self.in_var[1] - self.in_var[0]) + sst_idx] = (
                    1 - self.get_land_sea_mask(folder_path=self.mask_path)
                )

        # modify the correct index
        return feature_weights

    def get_land_sea_mask(
        self, folder_path: str, sst_mask_name: str = "sst_mask.npy"
    ) -> torch.Tensor:
        """Return the local shard of the sst mask."""
        return torch.tensor(
            np.load(os.path.join(folder_path, sst_mask_name))[
                : self.xlat, self.in_out_xlon[0] : self.in_out_xlon[1]
            ]
        ).to(torch.float32)

    def _load_constant_mask_2d(
        self, folder_path: str, mask_list: List[str]
    ) -> np.array:
        """
        Load the constant masks applied in the patch embedding layer.

        folder_path: String
            Path to directory containing constant masks
        mask_list: List[str]
            List of constant masks

        Returns
        -------
        mask: numpy.array
            of shape (n_masks, n_lat, n_lon)
        """
        # Load data from numpy files

        data = []
        for mask in mask_list:
            data.append(np.load(os.path.join(folder_path, mask)))

        masks = np.stack(data, axis=0)
        return masks


class ERA5Dummy(Dataset):
    """
    Generates a dummy dataset, returning tensors (input, target) for a specified shape.

    In a distributed (domain-parallel) setting, each process generates only it's shard of the channels and spatial latitude.
    Both input and target can consist of one or more timesteps, which are concatenated in the channel dimension.

    Attributes
    ----------
    dtype : torch.dtype
        dtype of the input and target tensors to be generated
    world_size : int
        The size of the domain-parallel process group.
    local_rank : int
        The rank withing the domain-parallel process group.
    fill_value : int
        The value with which data samples (input and target) will be filled. If None, tensors will be filled with numbers
    years : Tuple(str,st)
        The temporal range to be included into the dataset, in the format`%Y-%m-%d`
    in_timesteps : int
        Number of timesteps to be concatenated for the input
    out_timesteps : int
        Number of timesteps to be concatenated for the target
    dt : int
        The temporal subsampling timestep in hours
    n_data_items:
        Number of data samples (time steps) in self.zarr_data
    xlat : int
        Global size of data samples in self.zarr_data in the latitude dimension
    xlon : int
        Global size of data samples in self.zarr_data in the longitude dimension
    n_variables: int
        Global number of variables (channels) in self.zarr_data
    constant_masks: torch.Tensor
        Tensor with the local shard of the constant masks (chunked in channels and longitude)
    n_masks : int
        Number of constant masks
    in_var : Tuple(int, int)
        The start and stop value of the range of channels for this particluar rank
    in_masks: Tuple(int, int)
        The start and stop value of the range of masks for this particluar rank
    in_out_xlat : Tuple(int, int)
        The start and stop value of the spatial range (in the latitude) for this particluar rank
    in_out_xlon : Tuple(int, int)
        The start and stop value of the spatial range (in the longitude) for this particluar rank
    transform: Callable
        The transformation to be applied to the input weather data
    target_transform: Callable
        The transformation to be applied to the output weather data. If None, defaults to the transformation given in transform argument

    Parameters
    ----------
    pg_rank : int
        The domain-parallel process group rank.
    pg_size : int
        The size of the domain-parallel process group.
    global shape : Tuple(int, int, int)
        The [channel, latitude, longitude] dimensions of the individual data "time steps" that are generated
    fill_value : int
        A fill value with which input and target are filled. If `None`, random tensors will be generated
    time_range : Tuple(str,st)
        The temporal range to be included into the dataset, in the format`%Y-%m-%d`
    dt : int
        The temporal subsampling timestep in hours
    n_in_timesteps : int
        The number of input time setps
    n_out_timesteps : int
        The number of output time setps
    dtype : torch.dtype
        The dtype for input and target tensors
    transform: Callable
        The transformation to be applied to the input weather data
    target_transform: Callable
        The transformation to be applied to the output weather data. If None, defaults to the transformation given in transform argument

    """

    def __init__(
        self,
        pg_rank: int,
        pg_size: int,
        global_shape: Tuple[int, int, int],
        fill_value: float | None = None,
        time_range: Tuple[str, str] | None = None,
        dt: int = 6,
        n_in_timesteps: int = 1,
        n_out_timesteps: int = 1,
        dtype: torch.dtype = torch.float32,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.dt = dt
        self.in_timesteps = n_in_timesteps
        self.out_timesteps = n_out_timesteps

        self.dtype = dtype
        self.world_size = pg_size
        self.local_rank = pg_rank
        self.years = time_range

        self.fill_value = fill_value
        # Filter desired time points based on parameters and return file statistics.
        if self.years is not None:
            max_date = datetime.strptime(self.years[1], "%Y-%m-%d")
            min_date = datetime.strptime(self.years[0], "%Y-%m-%d")
        else:
            max_date = datetime.strptime("2000-12-31", "%Y-%m-%d")
            min_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
        print(f"Dummy Dataset max_date {max_date}, min_date {min_date} ")

        self.num_vars, self.xlat, self.xlon = global_shape
        self.n_data_items = int((max_date - min_date).days * 24)
        print(f"Dummy Dataset n_data_items {self.n_data_items} ")

        if self.xlat % 2 == 1:
            self.xlat -= 1
        if self.xlon % 2 == 1:
            self.xlon -= 1

        self.in_out_xlat = [0, self.xlat]
        if self.world_size > 1:
            row, col = self.local_rank % 2, self.local_rank // 2
            # rank 0: [0,0], rank 1 = [1,0], rank 2 = [0,1], rank 3 = [1,1]
            self.in_var = [row * self.num_vars // 2, (row + 1) * self.num_vars // 2]
            self.in_out_xlon = [
                col * self.xlon // (self.world_size // 2),
                (col + 1) * self.xlon // (self.world_size // 2),
            ]

        else:
            self.in_var = [0, self.num_vars]
            self.in_out_xlon = [0, self.xlon]

        self.transform = transform
        if target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = target_transform

    def __len__(self) -> int:
        """Return total number of samples."""
        return int(
            self.n_data_items - (self.in_timesteps + self.out_timesteps - 1) * self.dt
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Return single input, target.

        Parameters
        ----------
        idx: int
            Global index of item

        Returns
        -------
        input : Tensor
            The input sample of shape [channel shard * number of input timesteps, xlat, longitudinal shard].

        target : Tensor
            The target sample of shape [channel shard * number of output timesteps, xlat, longitudinal shard].
        """
        shape_in = (
            (self.in_var[1] - self.in_var[0]) * self.in_timesteps,
            self.in_out_xlat[1] - self.in_out_xlat[0],
            self.in_out_xlon[1] - self.in_out_xlon[0],
        )
        shape_out = (
            (self.in_var[1] - self.in_var[0]) * self.out_timesteps,
            self.in_out_xlat[1] - self.in_out_xlat[0],
            self.in_out_xlon[1] - self.in_out_xlon[0],
        )

        if self.fill_value:
            input = torch.full(shape_in, self.fill_value, dtype=self.dtype)
            target = torch.full(shape_out, self.fill_value, dtype=self.dtype)
        else:
            input = torch.rand(shape_in, dtype=self.dtype)
            target = torch.rand(shape_out, dtype=self.dtype)

        if self.transform is not None:
            input = self.transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target

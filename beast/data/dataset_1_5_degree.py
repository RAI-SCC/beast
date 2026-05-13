import os
from datetime import datetime

import netCDF4
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import xarray as xr
import yaml
from torch.utils.data import Dataset


class ERA5PreprocessedDataset15degree(Dataset):
    """Define 3D dataset."""

    def __init__(self, config, mode, normalize=True, dt=None, dtype=torch.float32):
        """
        Initialize.

        params: Dict
            configuration file
        mode: String
            of value 'training', 'testing', 'validation'
        """
        self.dt = config["data"]["dt"]
        self.dtype = dtype
        self.datapath = config["data"]["train_data_path"]
        if mode == "train":  # training case, lite
            self.years = config["data"]["train_time_range"]
        elif mode == "validation":  # validation
            self.years = config["data"]["valid_time_range"]
        else:
            self.years = None

        self.normalize = normalize
        self.batch_size = config["training"]["batch_size"]

        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        mask_path = config["data"]["constant_masks_path"]
        mask_list = config["data"]["constant_masks"]

        # Filter desired time points based on parameters and return file statistics.
        self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")
        times = pd.to_datetime(self.zarr_data.time)
        train_years = times[(times >= self.years[0]) & (times < self.years[1])]
        self.zarr_data = self.zarr_data.sel(time=train_years)

        self.n_data_items = len(self.zarr_data["time"])
        self.num_vars = len(self.zarr_data["feature"])  # self.num_vars

        self.constant_masks = self._load_constant_mask_2d(mask_path, mask_list)
        self.num_masks = self.constant_masks.shape[0]

        # normalization for pressure levels and standard deviation
        self.norm_values = {}
        self.norm_values["mean"] = np.load(
            f"{config['data']['normalization_dir']}/norm_mean.npy"
        )
        self.norm_values["std"] = np.load(
            f"{config['data']['normalization_dir']}/norm_std.npy"
        )

        self.in_var = [0, self.num_vars]
        self.in_masks = [0, self.num_masks]
        self.in_out_xlon = [0, self.xlon]

        if dist.get_rank(group=None) == 0:
            print("Using ERA5PreprocessedDataset15degree")

    def __len__(self):
        """Return total number of samples."""
        return self.n_data_items - self.dt

    def __getitem__(self, idx):
        """
        Return single input, target.

        idx: int
                global index of item
        """
        target_idx = idx + self.dt

        x = self.zarr_data.isel(
            time=idx,
            latitude=range(0, self.xlat),
        )["fields"].values

        x = (x - self.norm_values["mean"]) / self.norm_values["std"]

        input = torch.cat(
            (
                torch.tensor(x, dtype=self.dtype),
                self.get_constant_masks().to(self.dtype),
            ),
            dim=0,
        )

        # Output data
        y = self.zarr_data.isel(
            time=target_idx,
            latitude=range(0, self.xlat),
        )["fields"].values

        y = (y - self.norm_values["mean"]) / self.norm_values["std"]

        target = torch.tensor(y, dtype=self.dtype)

        return input, target

    def get_constant_masks(self):
        return torch.tensor(self.constant_masks[:, : self.xlat, :]).to(torch.float32)

    def _load_constant_mask_2d(self, folder_path, mask_list):
        """
        Load the constant masks applied in the patch embedding layer.

        folder_path: String
            Path to directory containing constant masks
        mask_list: List[String]
            List of constant masks

        Returns
        -------
        mask: Tensor
            of shape (num_masks, n_lat, n_lon)
        """
        # Load data from numpy files
        data_files = [f for f in os.listdir(folder_path) if f in mask_list]
        data = []
        for file in data_files:
            data.append(np.load(os.path.join(folder_path, file)))
        masks = np.stack(data, axis=0)
        return masks


class ERA5TwoTimestepsPreprocessedDataset15degree(ERA5PreprocessedDataset15degree):
    def __init__(self, config, mode, normalize=True, dt=None, dtype=torch.float32):
        super().__init__(config, mode, normalize=normalize, dt=dt, dtype=dtype)
        if dist.get_rank(group=None) == 0:
            print("Loading two timeframes as input...")

    def __len__(self):
        """Return total number of samples."""
        return self.n_data_items - 2 * self.dt

    def __getitem__(self, idx):
        """
        Return single input, target.

        idx: int
                global index of item
        """
        target_idx = idx + 2 * self.dt

        x_t1 = self.zarr_data.isel(
            time=idx,
            latitude=range(0, self.xlat),
        )["fields"].values

        x_t1 = (x_t1 - self.norm_values["mean"]) / self.norm_values["std"]

        x_t2 = self.zarr_data.isel(
            time=(idx + self.dt),
            latitude=range(0, self.xlat),
        )["fields"].values

        x_t2 = (x_t2 - self.norm_values["mean"]) / self.norm_values["std"]

        input = torch.cat(
            (
                torch.tensor(x_t1, dtype=self.dtype),
                torch.tensor(x_t2, dtype=self.dtype),
                self.get_constant_masks().to(self.dtype),
            ),
            dim=0,
        )

        # Output data
        y = self.zarr_data.isel(
            time=target_idx,
            latitude=range(0, self.xlat),
        )["fields"].values

        y = (y - self.norm_values["mean"]) / self.norm_values["std"]

        target = torch.tensor(y, dtype=self.dtype)
        return input, target


class ERA5DatasetDownsampled(Dataset):
    """Define 3D dataset."""

    def __init__(self, config, mode, normalize=True, dt=None, dtype=torch.float32):
        """
        Initialize.

        params: Dict
            configuration file
        mode: String
            of value 'training', 'testing', 'validation'
        """
        self.dt = config["data"]["dt"]
        self.dtype = dtype
        self.datapath = config["data"]["train_data_path"]
        if mode == "train":  # training case, lite
            self.years = config["data"]["train_time_range"]
        elif mode == "validation":  # validation
            self.years = config["data"]["valid_time_range"]
        else:
            self.years = None

        self.normalize = normalize
        self.batch_size = config["training"]["batch_size"]

        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        mask_path = config["data"]["constant_masks_path"]
        mask_list = config["data"]["constant_masks"]

        # Filter desired time points based on parameters and return file statistics.
        self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")
        times = pd.to_datetime(self.zarr_data.time)
        train_years = times[(times >= self.years[0]) & (times < self.years[1])]
        self.zarr_data = self.zarr_data.sel(time=train_years)

        self.n_data_items = len(self.zarr_data["time"])
        self.num_vars = len(self.zarr_data["feature"])  # self.num_vars

        self.constant_masks = self._load_constant_mask_2d(mask_path, mask_list)
        self.num_masks = self.constant_masks.shape[0]

        # normalization for pressure levels and standard deviation
        self.norm = netCDF4.Dataset(f"{config['data']['normalization_dir']}/norms.nc")
        self.norm_values = {
            "mean": np.array(self.norm["mean"]).reshape(self.num_vars, 1, 1),
            "std": np.array(self.norm["std"]).reshape(self.num_vars, 1, 1),
        }

        self.downsampler = torch.nn.AvgPool2d(6, stride=6)
        if dist.get_rank(group=None) == 0:
            print("Using ERA5DatasetDownsampled")

    def __len__(self):
        """Return total number of samples."""
        return self.n_data_items - self.dt

    def __getitem__(self, idx):
        """
        Return single input, target.

        idx: int
                global index of item
        """
        target_idx = idx + self.dt
        x = self.zarr_data.isel(time=idx)["fields"].values

        x = (x - self.norm_values["mean"]) / self.norm_values["std"]

        x = self.downsampler(torch.tensor(x, dtype=self.dtype))
        input = torch.cat((x, self.get_constant_masks().to(self.dtype)), dim=0)

        # Odd ranks load second half of variables, so they might have less masks
        if (((self.num_vars + self.num_masks) % 2) == 1) and (self.local_rank % 2 == 1):
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, 1))

        # Output data
        y = self.zarr_data.isel(time=target_idx)["fields"].values
        y = (y - self.norm_values["mean"]) / self.norm_values["std"]

        target = self.downsampler(torch.tensor(y, dtype=self.dtype))

        return input, target

    def get_constant_masks(self):
        mask = torch.tensor(self.constant_masks[:, : self.xlat, :]).to(torch.float32)
        return mask

    def _load_constant_mask_2d(self, folder_path, mask_list):
        """
        Load the constant masks applied in the patch embedding layer.

        folder_path: String
            Path to directory containing constant masks
        mask_list: List[String]
            List of constant masks

        Returns
        -------
        mask: Tensor
            of shape (num_masks, n_lat, n_lon)
        """
        # Load data from numpy files
        data_files = [f for f in os.listdir(folder_path) if f in mask_list]
        data = []
        for file in data_files:
            data.append(np.load(os.path.join(folder_path, file)))
        masks = np.stack(data, axis=0)
        return masks


class DummyDataset(Dataset):
    """Dataset in the shape of 1.5 degree data, filled with random numbers, for testing on systems without data"""

    def __init__(self, config, mode, normalize=True, dt=None, dtype=torch.float32):
        """
        Initialize.

        params: Dict
            configuration file
        mode: String
            of value 'training', 'testing', 'validation'
        """
        self.dt = config["data"]["dt"]
        self.dtype = dtype
        self.datapath = config["data"]["train_data_path"]
        if mode == "train":  # training case, lite
            self.years = config["data"]["train_time_range"]
        elif mode == "validation":  # validation
            self.years = config["data"]["valid_time_range"]
        else:
            self.years = None

        self.batch_size = config["training"]["batch_size"]

        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        mask_list = config["data"]["constant_masks"]

        # Filter desired time points based on parameters and return file statistics.
        max_date = datetime.strptime(self.years[1], "%Y-%m-%d")
        min_date = datetime.strptime(self.years[0], "%Y-%m-%d")

        self.n_data_items = (max_date - min_date).days * 24
        self.num_masks = len(mask_list)

        if config["data"]["n_variables"] is None:
            self.n_variables = 69
        else:
            self.n_variables = config["data"]["n_variables"]

        if dist.get_rank(group=None) == 0:
            print("Using DummyDataset")

    def __len__(self):
        """Return total number of samples."""
        return self.n_data_items - self.dt

    def __getitem__(self, idx):
        total_vars = self.n_variables + self.num_masks
        shape_in = (total_vars, self.xlat, self.xlon)
        shape_out = (self.n_variables, self.xlat, self.xlon)

        input = torch.rand(shape_in, dtype=self.dtype)
        target = torch.rand(shape_out, dtype=self.dtype)

        return input, target



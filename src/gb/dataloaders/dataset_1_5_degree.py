import csv
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


class ERA5Dataset15degree(Dataset):
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

        self.constant_masks = self._load_constant_mask_2d(mask_path, mask_list)
        self.num_masks = self.constant_masks.shape[0]

        # normalization for pressure levels and standard deviation
        self.p_mean = np.load(
            f"{config['data']['normalization_dir']}/pressure_zarr.npy",
            allow_pickle=True,
        )[0].reshape(5, 13, 1, 1)
        self.p_std = np.load(
            f"{config['data']['normalization_dir']}/pressure_zarr.npy",
            allow_pickle=True,
        )[1].reshape(5, 13, 1, 1)
        self.s_mean = np.load(
            f"{config['data']['normalization_dir']}/surface_zarr.npy",
            allow_pickle=True,
        )[0].reshape(4, 1, 1)
        self.s_std = np.load(
            f"{config['data']['normalization_dir']}/surface_zarr.npy",
            allow_pickle=True,
        )[1].reshape(4, 1, 1)

        self.level_ordering = range(13 - 1, -1, -1)

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

        input_pressure = self.zarr_data.isel(
            time=idx,
            level=self.level_ordering,
            latitude=range(0, self.xlat),
            longitude=range(0, self.xlon),
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        input_surface = self.zarr_data.isel(
            time=idx, latitude=range(0, self.xlat), longitude=range(0, self.xlon)
        )[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]

        # Stack and convert to numpy array
        input_pressure = np.stack(
            [
                input_pressure["geopotential"].values,
                input_pressure["specific_humidity"].values,
                input_pressure["temperature"].values,
                input_pressure["u_component_of_wind"].values,
                input_pressure["v_component_of_wind"].values,
            ],
            axis=0,
        )
        input_surface = np.stack(
            [
                input_surface["mean_sea_level_pressure"].values,
                input_surface["10m_u_component_of_wind"].values,
                input_surface["10m_v_component_of_wind"].values,
                input_surface["2m_temperature"].values,
            ],
            axis=0,
        )
        input_pressure = np.permute_dims(input_pressure, (0, 1, 3, 2))
        input_surface = np.permute_dims(input_surface, (0, 2, 1))

        input_pressure = (input_pressure - self.p_mean) / self.p_std
        input_surface = (input_surface - self.s_mean) / self.s_std
        input_pressure = input_pressure.reshape(-1, self.xlat, self.xlon)

        input = torch.cat(
            (
                torch.tensor(input_surface, dtype=self.dtype),
                torch.tensor(input_pressure, dtype=self.dtype),
                self.get_constant_masks().to(self.dtype),
            ),
            dim=0,
        )

        # Output data
        target_pressure = self.zarr_data.isel(
            time=target_idx,
            level=self.level_ordering,
            latitude=range(0, self.xlat),
            longitude=range(self.xlon),
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        target_surface = self.zarr_data.isel(
            time=target_idx, latitude=range(0, self.xlat), longitude=range(self.xlon)
        )[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]

        # Stack and convert to numpy array
        target_pressure = np.stack(
            [
                target_pressure["geopotential"].values,
                target_pressure["specific_humidity"].values,
                target_pressure["temperature"].values,
                target_pressure["u_component_of_wind"].values,
                target_pressure["v_component_of_wind"].values,
            ],
            axis=0,
        )
        target_surface = np.stack(
            [
                target_surface["mean_sea_level_pressure"].values,
                target_surface["10m_u_component_of_wind"].values,
                target_surface["10m_v_component_of_wind"].values,
                target_surface["2m_temperature"].values,
            ],
            axis=0,
        )

        target_pressure = np.permute_dims(target_pressure, (0, 1, 3, 2))
        target_surface = np.permute_dims(target_surface, (0, 2, 1))

        target_pressure = (target_pressure - self.p_mean) / self.p_std
        target_surface = (target_surface - self.s_mean) / self.s_std
        target_pressure = target_pressure.reshape(-1, self.xlat, self.xlon)

        target = torch.cat(
            (
                torch.tensor(target_pressure, dtype=self.dtype),
                torch.tensor(target_surface, dtype=self.dtype),
            ),
            dim=0,
        )

        return input, target

    def get_constant_masks(self):
        return torch.tensor(self.constant_masks[:, : self.xlat, : self.xlon]).to(
            torch.float32
        )

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


class ERA5TwoTimestepsDataset15degree(ERA5Dataset15degree):
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

        input_pressure1 = self.zarr_data.isel(
            time=idx,
            level=self.level_ordering,
            latitude=range(0, self.xlat),
            longitude=range(self.xlon),
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        input_surface1 = self.zarr_data.isel(
            time=idx, latitude=range(0, self.xlat), longitude=range(self.xlon)
        )[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]

        # Stack and convert to numpy array
        input_pressure1 = np.stack(
            [
                input_pressure1["geopotential"].values,
                input_pressure1["specific_humidity"].values,
                input_pressure1["temperature"].values,
                input_pressure1["u_component_of_wind"].values,
                input_pressure1["v_component_of_wind"].values,
            ],
            axis=0,
        )
        input_surface1 = np.stack(
            [
                input_surface1["mean_sea_level_pressure"].values,
                input_surface1["10m_u_component_of_wind"].values,
                input_surface1["10m_v_component_of_wind"].values,
                input_surface1["2m_temperature"].values,
            ],
            axis=0,
        )

        input_pressure1 = np.permute_dims(input_pressure1, (0, 1, 3, 2))
        input_surface1 = np.permute_dims(input_surface1, (0, 2, 1))
        input_pressure1 = (input_pressure1 - self.p_mean) / self.p_std
        input_surface1 = (input_surface1 - self.s_mean) / self.s_std
        input_pressure1 = input_pressure1.reshape(-1, self.xlat, self.xlon)

        input_pressure2 = self.zarr_data.isel(
            time=idx,
            level=self.level_ordering,
            latitude=range(0, self.xlat),
            longitude=range(self.xlon),
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        input_surface2 = self.zarr_data.isel(
            time=idx, latitude=range(0, self.xlat), longitude=range(self.xlon)
        )[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]

        # Stack and convert to numpy array
        input_pressure2 = np.stack(
            [
                input_pressure2["geopotential"].values,
                input_pressure2["specific_humidity"].values,
                input_pressure2["temperature"].values,
                input_pressure2["u_component_of_wind"].values,
                input_pressure2["v_component_of_wind"].values,
            ],
            axis=0,
        )
        input_surface2 = np.stack(
            [
                input_surface2["mean_sea_level_pressure"].values,
                input_surface2["10m_u_component_of_wind"].values,
                input_surface2["10m_v_component_of_wind"].values,
                input_surface2["2m_temperature"].values,
            ],
            axis=0,
        )

        input_pressure2 = np.permute_dims(input_pressure2, (0, 1, 3, 2))
        input_surface2 = np.permute_dims(input_surface2, (0, 2, 1))

        input_pressure2 = (input_pressure2 - self.p_mean) / self.p_std
        input_surface2 = (input_surface2 - self.s_mean) / self.s_std
        input_pressure2 = input_pressure2.reshape(-1, self.xlat, self.xlon)

        input = torch.cat(
            (
                torch.tensor(input_surface1, dtype=self.dtype),
                torch.tensor(input_pressure1, dtype=self.dtype),
                torch.tensor(input_surface2, dtype=self.dtype),
                torch.tensor(input_pressure2, dtype=self.dtype),
                self.get_constant_masks().to(self.dtype),
            ),
            dim=0,
        )

        # Output data
        target_pressure = self.zarr_data.isel(
            time=target_idx,
            level=self.level_ordering,
            latitude=range(0, self.xlat),
            longitude=range(self.xlon),
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        target_surface = self.zarr_data.isel(
            time=target_idx, latitude=range(0, self.xlat), longitude=range(self.xlon)
        )[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]

        # Stack and convert to numpy array
        target_pressure = np.stack(
            [
                target_pressure["geopotential"].values,
                target_pressure["specific_humidity"].values,
                target_pressure["temperature"].values,
                target_pressure["u_component_of_wind"].values,
                target_pressure["v_component_of_wind"].values,
            ],
            axis=0,
        )
        target_surface = np.stack(
            [
                target_surface["mean_sea_level_pressure"].values,
                target_surface["10m_u_component_of_wind"].values,
                target_surface["10m_v_component_of_wind"].values,
                target_surface["2m_temperature"].values,
            ],
            axis=0,
        )

        target_pressure = np.permute_dims(target_pressure, (0, 1, 3, 2))
        target_surface = np.permute_dims(target_surface, (0, 2, 1))
        target_pressure = (target_pressure - self.p_mean) / self.p_std
        target_surface = (target_surface - self.s_mean) / self.s_std
        target_pressure = target_pressure.reshape(-1, self.xlat, self.xlon)

        target = torch.cat(
            (
                torch.tensor(target_pressure, dtype=self.dtype),
                torch.tensor(target_surface, dtype=self.dtype),
            ),
            dim=0,
        )

        return input, target


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


class WeatherTestDataset(Dataset):
    """Dataloader for 2D model."""

    def __init__(self, config):
        self.config = config["data"]

        self.file_path = self.config["train_data_path"]

        self.dt = 6  # self.config['forecast_step']
        self.deltaTDivisor = 6  # self.config['dataset_subsampling']

        if "target_subsampling" not in self.config.keys():
            self.config["target_subsampling"] = self.deltaTDivisor

        self.patch_size = (config["model"]["patch_size"], config["model"]["patch_size"])
        self.resolution = (self.config["x_lat"], self.config["x_lon"])
        self.padded_resolution = (
            self.resolution[0]
            + (self.patch_size[0] - self.resolution[0] % self.patch_size[0])
            % self.patch_size[0],
            self.resolution[1]
            + (self.patch_size[1] - self.resolution[1] % self.patch_size[1])
            % self.patch_size[1],
        )
        self.padding = (
            (
                (self.padded_resolution[0] - self.resolution[0]) // 2,
                (self.padded_resolution[0] - self.resolution[0])
                - (self.padded_resolution[0] - self.resolution[0]) // 2,
            ),
            (
                (self.padded_resolution[1] - self.resolution[1]) // 2,
                (self.padded_resolution[1] - self.resolution[1])
                - (self.padded_resolution[1] - self.resolution[1]) // 2,
            ),
        )

        if "embedding_kernel_size" in config["model"].keys():
            self.embedding_kernel_size = config["model"]["embedding_kernel_size"]

            if self.embedding_kernel_size > self.patch_size[0]:
                self.padded_resolution = (
                    self.resolution[0]
                    + (self.patch_size[0] - self.resolution[0] % self.patch_size[0])
                    % self.patch_size[0]
                    + self.embedding_kernel_size
                    - self.patch_size[0],
                    self.resolution[1]
                    + (self.patch_size[1] - self.resolution[1] % self.patch_size[1])
                    % self.patch_size[1]
                    + self.embedding_kernel_size
                    - self.patch_size[1],
                )
                self.padding = (
                    (
                        (self.padded_resolution[0] - self.resolution[0]) // 2,
                        (self.padded_resolution[0] - self.resolution[0])
                        - (self.padded_resolution[0] - self.resolution[0]) // 2,
                    ),
                    (
                        (self.padded_resolution[1] - self.resolution[1]) // 2,
                        (self.padded_resolution[1] - self.resolution[1])
                        - (self.padded_resolution[1] - self.resolution[1]) // 2,
                    ),
                )
            elif self.embedding_kernel_size < self.patch_size[0]:
                self.embedding_kernel_size = self.patch_size[0]
        else:
            self.embedding_kernel_size = self.patch_size[0]

        self.translation_dict = None
        self.in_surface_variables = self._translate_variable_names(
            self.config["in_surface_variables"]
        )
        self.in_pressure_variables = self._translate_variable_names(
            self.config["in_pressure_variables"]
        )
        self.out_surface_variables = self._translate_variable_names(
            self.config["out_surface_variables"]
        )
        self.out_pressure_variables = self._translate_variable_names(
            self.config["out_pressure_variables"]
        )

        self._get_data_files_stats()
        self._get_normalization_constants()
        self.uses_constant_masks = len(self.config["constant_masks"]) > 0
        if self.uses_constant_masks:
            self._load_constant_masks()

        self.forecast_length = self.config.get("rollout", 1)

    def _translate_variable_names(self, varname_list):
        if self.translation_dict is None:
            self._load_translation_dict()

        return_list = [
            (
                self.translation_dict[varname]
                if varname in self.translation_dict.keys()
                else varname
            )
            for varname in varname_list
        ]

        return return_list

    def _load_translation_dict(self):
        """Load yaml file that translates shorthands to era5 dataset variable names."""
        self.translation_dict = {
            "t2m": "2m_temperature",
            "u10m": "10m_u_component_of_wind",
            "v10m": "10m_v_component_of_wind",
            "msl": "mean_sea_level_pressure",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "z": "geopotential",
            "q": "specific_humidity",
            "t": "temperature",
        }
        # full_path = os.path.join(self.data_path, "variable_aliases.yaml")
        # with open(full_path) as f:
        #    self.translation_dict = yaml.load(f, Loader=yaml.SafeLoader)

    def _get_data_files_stats(self):
        """Filter desired time points based on parameters and return file statistics."""

        """
        # TODO: generalize to values read from config file
        self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
        times = pd.to_datetime(self.zarr_data['time'].values)
        if self.mode == 'train': # training cases, lite
            train_years = times[(times.year < 2018) & (times.year > 2006)]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif self.mode == 'valid':           # validation
            validation_years = times[(times.year == 2018)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        elif self.mode == 'test':           # validation
            validation_years = times[(times.year == 2020) & (times.month == 1) & (times.day < 5)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        """

        dataset = xr.open_dataset(self.file_path, engine="zarr")
        times = pd.to_datetime(dataset["time"].values)

        self.zarr_in = dataset[self.in_pressure_variables]
        self.zarr_out = dataset[self.out_pressure_variables]
        self.zarr_in_surface = dataset[self.in_surface_variables]
        self.zarr_out_surface = dataset[self.out_surface_variables]

        """
        start_date = datetime.strptime(self.time_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(self.time_range[1], '%Y-%m-%d')

        dataset_in_range = times[
            (times.year >= start_date.year)
            & (times.year <= end_date.year)
            & ((times.month >= start_date.month) | (times.year > start_date.year))
            & ((times.month <= end_date.month) | (times.year < end_date.year))
            & ((times.day >= start_date.day) | (times.year > start_date.year) | (times.month > start_date.month))
            & ((times.day <= end_date.day) | (times.year < end_date.year) | (times.month < end_date.month))
            & (times.hour % self.config['target_subsampling'] == 0)
            ]
        dataset_out_range = times[
            (times.year >= start_date.year)
            & (times.year <= end_date.year)
            & ((times.month >= start_date.month) | (times.year > start_date.year))
            & ((times.month <= end_date.month) | (times.year < end_date.year))
            & ((times.day >= start_date.day) | (times.year > start_date.year) | (times.month > start_date.month))
            & ((times.day <= end_date.day) | (times.year < end_date.year) | (times.month < end_date.month))
            & ((times.hour - self.dt) % self.config['target_subsampling'] == 0)
            ]
        """

        dataset_in_range = times[
            (times.year == 2020)
            & (times.month == 1)
            & (times.day >= 1)
            & (times.day <= 6)
        ]

        dataset_out_range = times[
            (times.year == 2020)
            & (times.month == 1)
            & (times.day >= 1)
            & (times.day <= 6)
        ]

        self.zarr_in = self.zarr_in.sel(time=dataset_in_range)
        self.zarr_in = self.zarr_in.sel(level=self.config["in_pressure_levels"])
        self.zarr_in = self.zarr_in.transpose("time", "level", "latitude", "longitude")

        self.zarr_out = self.zarr_out.sel(time=dataset_out_range)
        self.zarr_out = self.zarr_out.sel(level=self.config["out_pressure_levels"])
        self.zarr_out = self.zarr_out.transpose(
            "time", "level", "latitude", "longitude"
        )

        self.zarr_in_surface = self.zarr_in_surface.sel(time=dataset_in_range)
        self.zarr_in_surface = self.zarr_in_surface.transpose(
            "time", "latitude", "longitude"
        )

        self.zarr_out_surface = self.zarr_out_surface.sel(time=dataset_out_range)
        self.zarr_out_surface = self.zarr_out_surface.transpose(
            "time", "latitude", "longitude"
        )

        self.n_samples_total = len(self.zarr_in["time"])
        self.img_shape_x = len(self.zarr_in["latitude"])
        self.img_shape_y = len(self.zarr_in["longitude"])
        # N channels

        self.num_masks = len(self.config["constant_masks"])
        self.n_in_channels = len(self.zarr_in.data_vars) + self.num_masks

    def _get_normalization_constants(self):
        surface_normalizations = []
        # with open(f"{self.data_path}/normalizations/surface_norms.csv", "r") as csvfile:
        with open(
            self.config["normalization_dir"] + "/surface_norms.csv", "r"
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                surface_normalizations.append([row[0], float(row[1]), float(row[2])])

        in_s_mean_data = []
        in_s_std_data = []
        for variable_name in self.in_surface_variables:
            for row in surface_normalizations:
                if variable_name == row[0]:
                    in_s_mean_data.append(row[1])
                    in_s_std_data.append(row[2])
                    break
            else:
                raise ValueError(
                    f"Surface variable {variable_name} has no stored normalization constants"
                )

        out_s_mean_data = []
        out_s_std_data = []
        for variable_name in self.out_surface_variables:
            for row in surface_normalizations:
                if variable_name == row[0]:
                    out_s_mean_data.append(row[1])
                    out_s_std_data.append(row[2])
                    break
            else:
                raise ValueError(
                    f"Surface variable {variable_name} has no stored normalization constants"
                )

        pressure_normalizations = []
        with open(
            self.config["normalization_dir"] + "/pressure_norms.csv", "r"
        ) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                pressure_normalizations.append([row[0], float(row[1]), float(row[2])])

        in_p_mean_data = []
        in_p_std_data = []
        for variable_name in self.in_pressure_variables:
            for level in self.config["in_pressure_levels"]:
                combined_variable_name = f"{variable_name}_{level}"
                for row in pressure_normalizations:
                    if combined_variable_name == row[0]:
                        in_p_mean_data.append(row[1])
                        in_p_std_data.append(row[2])
                        break
                else:
                    raise ValueError(
                        f"Variable {variable_name} has no stored normalization constants at pressure level {level}"
                    )

        out_p_mean_data = []
        out_p_std_data = []
        for variable_name in self.out_pressure_variables:
            for level in self.config["out_pressure_levels"]:
                combined_variable_name = f"{variable_name}_{level}"
                for row in pressure_normalizations:
                    if combined_variable_name == row[0]:
                        out_p_mean_data.append(row[1])
                        out_p_std_data.append(row[2])
                        break
                else:
                    raise ValueError(
                        f"Variable {variable_name} has no stored normalization constants at pressure level {level}"
                    )

        self.in_means = np.array(in_s_mean_data + in_p_mean_data)
        self.in_means = self.in_means.reshape((self.in_means.shape[0], 1, 1))
        self.in_stds = np.array(in_s_std_data + in_p_std_data)
        self.in_stds = self.in_stds.reshape((self.in_stds.shape[0], 1, 1))
        self.out_means = np.array(out_s_mean_data + out_p_mean_data)
        self.out_means = self.out_means.reshape((self.out_means.shape[0], 1, 1))
        self.out_stds = np.array(out_s_std_data + out_p_std_data)
        self.out_stds = self.out_stds.reshape((self.out_stds.shape[0], 1, 1))

        # order: [svar_1, ..., svar_N, pvar_1@p_1, pvar_1@p_2, ..., pvar_M@p_K], len(...) = N + M * K
        # for pangu variables: 4 + 5 * 13 = 69

    def _load_constant_masks(self):
        mask_file_paths = {}
        with open(self.config["constant_masks_path"] + "/file_paths.yaml") as f:
            mask_file_paths = yaml.load(f, Loader=yaml.SafeLoader)

        ordered_path_list = []

        for mask_name in self.config["constant_masks"]:
            full_key = f"{mask_name}_{self.resolution[0]}_{self.resolution[1]}"
            if full_key in mask_file_paths.keys():
                ordered_path_list.append(
                    (
                        mask_file_paths[full_key]["path"],
                        mask_file_paths[full_key]["normalize"],
                    )
                )
            else:
                raise ValueError(
                    f"Mask {mask_name} not supported at resolution {self.resolution[0]}x{self.resolution[1]}"
                )

        array_list = []
        for path, normalize in ordered_path_list:
            data = np.load(f"{self.config['constant_masks_path']}/{path}")

            if (
                data.shape[1] == self.resolution[0]
                and data.shape[0] == self.resolution[1]
            ):
                data = data.transpose((1, 0))

            if not (
                data.shape[0] == self.resolution[0]
                and data.shape[1] == self.resolution[1]
            ):
                raise ValueError(
                    f"Data stored at {path} does not have the required shape ({self.resolution[0]}, {self.resolution[1]})"
                )

            if normalize:
                mean = data.mean()
                std = data.std()
                data = (data - mean) / std

            array_list.append(data)

        stacked_data = np.stack(array_list, axis=0)

        self.constant_mask_data = torch.tensor(
            stacked_data, requires_grad=False
        )  # shape: (num_masks, x_lat, x_lon)

    def __len__(self):
        """Return total number of samples."""
        return (
            self.n_samples_total
            - self.forecast_length * self.dt // (self.config["target_subsampling"])
        )  # -1 to avoid last data point

    def __getitem__(self, global_idx, normalize=True):
        """
        Return an input output pair.

        global_idx: int
                global index of item
        normalize: bool
                flag for whether the data should be normalized
        """
        # TODO: not yet safety checked for edge cases or other errors
        output = []
        step = self.dt
        # if forecast_length = 3, output_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        output_file_idxs = (
            global_idx
            + torch.arange(1, self.forecast_length + 1)
            * step
            // self.config["target_subsampling"]
        )

        for output_idx in output_file_idxs:
            if output_idx >= self.__len__():
                output_idx = self.__len__() - 1

        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low

        input_pressure_ds = self.zarr_in.isel(time=global_idx).sel(
            level=self.config["in_pressure_levels"]
        )[self.in_pressure_variables]
        input_surface_ds = self.zarr_in_surface.isel(time=global_idx)[
            self.in_surface_variables
        ]
        output_pressure_ds = self.zarr_out.isel(time=output_file_idxs).sel(
            level=self.config["out_pressure_levels"]
        )[self.out_pressure_variables]
        output_surface_ds = self.zarr_out_surface.isel(time=output_file_idxs)[
            self.out_surface_variables
        ]

        # Stack and convert to numpy array
        input_pressure = np.stack(
            [
                input_pressure_ds[var_name].values
                for var_name in self.in_pressure_variables
            ],
            axis=0,
        )
        input_surface = np.stack(
            [
                input_surface_ds[var_name].values
                for var_name in self.in_surface_variables
            ],
            axis=0,
        )
        input_pressure = input_pressure.reshape(
            (-1, input_pressure.shape[-2], input_pressure.shape[-1])
        )
        input_total = np.concatenate((input_surface, input_pressure), axis=0)
        if normalize:
            input_total = (input_total - self.in_means) / self.in_stds

        if self.uses_constant_masks:
            input_joined = np.concatenate(
                (input_total, self.constant_mask_data), axis=0
            )
        else:
            input_joined = input_total

        input_tensor = torch.as_tensor(input_joined, dtype=torch.float32)
        if (
            input_tensor.shape[-2] == self.resolution[1]
            and input_tensor.shape[-1] == self.resolution[0]
        ):
            input_tensor = input_tensor.transpose(-1, -2)

        input_tensor = self._pad_data(input_tensor)

        for i in range(self.forecast_length):
            pressure_ds = output_pressure_ds.isel(time=i)
            surface_ds = output_surface_ds.isel(time=i)
            output_pressure = np.stack(
                [
                    pressure_ds[var_name].values
                    for var_name in self.out_pressure_variables
                ],
                axis=0,
            )
            output_surface = np.stack(
                [
                    surface_ds[var_name].values
                    for var_name in self.out_surface_variables
                ],
                axis=0,
            )
            output_pressure = output_pressure.reshape(
                (-1, output_pressure.shape[-2], output_pressure.shape[-1])
            )
            output_total = np.concatenate((output_surface, output_pressure), axis=0)

            if normalize:
                output_total = torch.as_tensor(
                    (output_total - self.out_means) / self.out_stds
                )
            else:
                output_total = torch.as_tensor(output_total)

            if (
                output_total.shape[-2] == self.resolution[1]
                and output_total.shape[-1] == self.resolution[0]
            ):
                output_total = output_total.transpose(-1, -2)

            output_total = self._pad_data(output_total)

            output.append(output_total)

        """
        # p_ and s_means is a stack of the mean and standard deviation values

        # shape of input_pressure
        # (5, 13, 721, 1440)
        #output_pressure[0], output_surface[0] = self._pad_data(output_pressure[0], output_surface[0])
        # Reshape into 2D
        shape_pressure  = input_pressure.shape
        input_pressure  = input_pressure.reshape(-1, shape_pressure[2], shape_pressure[3])
        for i in range(self.forecast_length):
            output_pressure[i] = output_pressure[i].reshape( -1, shape_pressure[2], shape_pressure[3])
        """

        return input_tensor, output

    def _pad_data(self, t):
        if self.padding[0][0] > 0 or self.padding[0][1] > 0:
            shift = t.shape[-1] // 2

            t = torch.nn.functional.pad(
                t, pad=(0, 0, self.padding[0][0], self.padding[0][1]), mode="reflect"
            )
            t[:, : self.padding[0][0], :] = torch.roll(
                t[:, : self.padding[0][0], :], shifts=shift, dims=2
            )
            t[:, -self.padding[0][1] :, :] = torch.roll(
                t[:, -self.padding[0][1] :, :], shifts=-shift, dims=2
            )

        if self.padding[1][0] > 0 or self.padding[1][1] > 0:
            t = torch.nn.functional.pad(
                t, pad=(self.padding[1][0], self.padding[1][1], 0, 0), mode="circular"
            )

        return t

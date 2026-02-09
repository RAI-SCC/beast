import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class ERA5DatasetDistributed(Dataset):
    """Define 3D dataset."""

    def __init__(
        self,
        config,
        file_path,
        mode,
        patch_size,
        rank,
        forecast_length=1,
        normalize=True,
        dt=None,
    ):
        """
        Initialize.

        params: Dict
            configuration file
        file_path: String
            path to data directory
        distributed: bool
            flag for DDP
        mode: String
            of value 'training', 'testing', 'validation'
        device: String
            device that the code is running/offloaded on
        patch_size: Tuple(int, int, Optional[int])
            Number of pixels in ([vert], lat, lon) dimensions per patch
        forecast_length: int
            For training, always 1. For validation, defines the number of autoregressive steps to roll out to.
        """
        self.file_path = file_path
        self.mode = mode
        self.dt = dt if dt is not None else config["data"]["dt"]
        self.rank = rank
        self.datapath = config["data"]["train_data_path"]

        self.deltaTDivisor = 1

        self.forecast_length = forecast_length
        self.normalize = normalize

        self._get_files_stats(mode=self.mode)

        self.level_ordering = range(0, 13)
        # normalization for pressure levels and standard deviation
        self.p_mean = np.load(
            f"{config['data']['normalization_dir']}/pressure_zarr.npy",
            allow_pickle=True,
        )[0].reshape(65, 1, 1)
        self.p_std = np.load(
            f"{config['data']['normalization_dir']}/pressure_zarr.npy",
            allow_pickle=True,
        )[1].reshape(65, 1, 1)
        self.s_mean = np.load(
            f"{config['data']['normalization_dir']}/surface_zarr.npy",
            allow_pickle=True,
        )[0].reshape(4, 1, 1)
        self.s_std = np.load(
            f"{config['data']['normalization_dir']}/surface_zarr.npy",
            allow_pickle=True,
        )[1].reshape(4, 1, 1)

        self.patch_size = patch_size
        self.xlat, self.xlon = config["data"]["xlat"], config["data"]["xlon"]
        self.batch_size = config["training"]["batch_size"]
        if self.rank % 4 in [0, 1]:  # Todo: Move to process group
            self.start, self.end = 0, self.xlon // 2
        else:
            self.start, self.end = self.xlon // 2, self.xlon

        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(
            config["data"]["constant_masks_path"], xlat=self.xlat
        )
        self.land_mask = torch.unsqueeze(self.land_mask[:, self.start : self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:, self.start : self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:, self.start : self.end], 0)

    def _get_files_stats(self, mode="train"):
        """Filter desired time points based on parameters and return file statistics."""
        self.zarr_data = xr.open_dataset(self.file_path, engine="zarr")
        times = pd.to_datetime(self.zarr_data["time"].values)
        if mode == "train":  # training case, lite
            train_years = times[(times.year < 2018) & (times.year > 1979)]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == "validation":  # validation
            validation_years = times[(times.year == 2018)]
            self.zarr_data = self.zarr_data.sel(time=validation_years)

        self.n_samples_total = len(self.zarr_data["time"])

        # N channels
        self.n_in_channels = len(self.zarr_data.data_vars)

    def __len__(self):
        """Return total number of samples."""
        return (
            self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor
        )

    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_surface = []
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(
            torch.arange(1, self.forecast_length + 1) * step,
            self.deltaTDivisor,
            rounding_mode="trunc",
        )
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__():
                target_idx = self.__len__() - 1

        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input_pressure = self.zarr_data.isel(
            time=global_idx, level=self.level_ordering
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        input_surface = self.zarr_data.isel(time=global_idx)[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]
        target_pressure_ds = self.zarr_data.isel(
            time=target_file_idxs, level=self.level_ordering
        )[
            [
                "geopotential",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ]
        ]
        target_surface_ds = self.zarr_data.isel(time=target_file_idxs)[
            [
                "mean_sea_level_pressure",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]

        target = []

        if self.rank % 2 == 0:  # first half of data
            # Input data
            # Stack and convert to numpy array
            input_pressure = (
                input_pressure[["geopotential", "specific_humidity", "temperature"]]
                .isel(
                    latitude=range(0, self.xlat), longitude=range(self.start, self.end)
                )
                .to_array("pressure_variables")
                .values
            ).reshape(-1, len(range(0, self.xlat)), len(range(self.end - self.start)))[
                :32
            ]

            input_surface = (
                input_surface[
                    [
                        "mean_sea_level_pressure",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_temperature",
                    ]
                ]
                .isel(
                    latitude=range(0, self.xlat), longitude=range(self.start, self.end)
                )
                .to_array("surface_variables")
                .values
            )

            input_pressure = (input_pressure - self.p_mean[:32]) / self.p_std[:32]
            input_surface = (input_surface - self.s_mean) / self.s_std

            input = np.concatenate([input_surface, input_pressure], axis=0)

            # Output data
            for i in range(self.forecast_length):
                pressure_ds = target_pressure_ds.isel(time=i)
                surface_ds = target_surface_ds.isel(time=i)

                temp = (
                    pressure_ds[["geopotential", "specific_humidity", "temperature"]]
                    .isel(
                        latitude=range(0, self.xlat),
                        longitude=range(self.start, self.end),
                    )
                    .to_array("pressure_variables")
                    .values
                ).reshape(
                    -1, len(range(0, self.xlat)), len(range(self.end - self.start))
                )[:32]

                target_surface.append(
                    surface_ds[
                        [
                            "mean_sea_level_pressure",
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                            "2m_temperature",
                        ]
                    ]
                    .isel(
                        latitude=range(0, self.xlat),
                        longitude=range(self.start, self.end),
                    )
                    .to_array("surface_variables")
                    .values
                )

                temp = (temp - self.p_mean[:32]) / self.p_std[:32]
                target_surface[i] = (target_surface[i] - self.s_mean) / self.s_std

                temp = np.concatenate([target_surface[i], temp], axis=0)
                target.append(torch.tensor(temp))
        else:
            input = (
                input_pressure[
                    ["temperature", "u_component_of_wind", "v_component_of_wind"]
                ]
                .isel(
                    latitude=range(0, self.xlat), longitude=range(self.start, self.end)
                )
                .to_array("pressure_variables")
                .values
            ).reshape(-1, len(range(0, self.xlat)), len(range(self.end - self.start)))[
                6:
            ]

            input = (input - self.p_mean[32:]) / self.p_std[32:]

            input = np.concatenate(
                [input, self.land_mask, self.soil_type, self.topography], axis=0
            )

            for i in range(self.forecast_length):
                pressure_ds = target_pressure_ds.isel(time=i)

                temp = (
                    pressure_ds[
                        ["temperature", "u_component_of_wind", "v_component_of_wind"]
                    ]
                    .isel(
                        latitude=range(0, self.xlat),
                        longitude=range(self.start, self.end),
                    )
                    .to_array("pressure_variables")
                    .values
                ).reshape(
                    -1, len(range(0, self.xlat)), len(range(self.end - self.start))
                )[6:]

                temp = (temp - self.p_mean[32:]) / self.p_std[32:]

                target.append(
                    torch.nn.functional.pad(torch.tensor(temp), (0, 0, 0, 0, 0, 3))
                )

        input = torch.tensor(input)

        return input, target


def load_constant_mask_2d(folder_path, xlat=721):
    """
    Load the constant masks applied in the patch embedding layer.

    patch_size: Tuple(int, int)
        Number of pixels in (lat, lon) dimensions per patch
    folder_path: String
        Path to directory containing constant masks

    Returns
    -------
    land_mask: Tensor
        of shape (n_lat, n_lon) after padding
    soil_type: Tensor
        of shape (n_lat, n_lon) after padding
    topography: Tensor
        of shape (n_lat, n_lon) after padding
    """
    # Load data from numpy files
    data_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    data = {}
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        data[file] = np.load(file_path)

    soil_type = data["soil_type.npy"]
    topography = data["topography.npy"]

    soil_type = (soil_type - np.mean(soil_type)) / np.std(soil_type)
    topography = (topography - np.mean(topography)) / np.std(topography)
    # Torch tensors
    land_mask = torch.tensor(data["land_mask.npy"]).to(torch.float32)
    soil_type = torch.tensor(soil_type).to(torch.float32)
    topography = torch.tensor(topography).to(torch.float32)

    # Check that the shapes of all the data are the same
    assert land_mask.shape == soil_type.shape == topography.shape, (
        "Shapes of the three constant masks are not equal."
    )

    land_mask = land_mask[:xlat]
    soil_type = soil_type[:xlat]
    topography = topography[:xlat]

    return land_mask, soil_type, topography

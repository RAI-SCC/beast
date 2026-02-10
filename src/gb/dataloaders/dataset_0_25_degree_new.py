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
        mode,
        process_group,
        normalize=True,
        dt=None,
    ):
        """
        Initialize.

        params: Dict
            configuration file
        mode: String
            of value 'training', 'testing', 'validation'
        process_group:
            DP Process group
        """
        self.dt = config["data"]["dt"]
        self.process_group = process_group
        self.local_rank = process_group.rank()
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
        self.num_masks = config["data"]["n_masks"]

        # Filter desired time points based on parameters and return file statistics.
        self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")
        times = pd.to_datetime(self.zarr_data.time)
        train_years = times[(times >= self.years[0]) & (times < self.years[1])]
        self.zarr_data = self.zarr_data.sel(time=train_years)

        self.n_data_items = len(self.zarr_data["time"])
        self.num_vars = len(self.zarr_data["feature"])  # self.num_vars
        self.total_vars = self.num_vars + self.num_masks
        if self.total_vars % 2 == 1:
            self.total_vars += 1
        # normalization for pressure levels and standard deviation
        # self.norm_values = np.load(
        #    f"{config['data']['normalization_dir']}/normalization_zarr.npy",
        #    allow_pickle=True,
        # )
        # [0].reshape(65, 1, 1)
        self.norm_values = {
            "mean": np.ones((self.num_vars, 1, 1), dtype=np.float32),
            "std": np.ones((self.num_vars, 1, 1), dtype=np.float32),
        }

        if self.local_rank == 0:
            self.var_start, self.var_end = 0, self.total_vars // 2
            self.xlon_start, self.xlon_end = 0, self.xlon // 2
        elif self.local_rank == 1:
            self.var_start, self.var_end = 0, self.total_vars // 2
            self.xlon_start, self.xlon_end = self.xlon // 2, self.xlon
        elif self.local_rank == 2:
            self.var_start, self.var_end = self.total_vars // 2, self.num_vars
            self.xlon_start, self.xlon_end = 0, self.xlon // 2
        elif self.local_rank == 3:
            self.var_start, self.var_end = self.total_vars // 2, self.num_vars
            self.xlon_start, self.xlon_end = self.xlon // 2, self.xlon
        else:
            self.var_start, self.var_end = 0, self.num_vars
            self.xlon_start, self.xlon_end = 0, self.xlon

        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(
            config["data"]["constant_masks_path"], xlat=self.xlat
        )
        self.land_mask = torch.unsqueeze(
            self.land_mask[:, self.xlon_start : self.xlon_end], 0
        )
        self.soil_type = torch.unsqueeze(
            self.soil_type[:, self.xlon_start : self.xlon_end], 0
        )
        self.topography = torch.unsqueeze(
            self.topography[:, self.xlon_start : self.xlon_end], 0
        )

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
            feature=range(self.var_start, self.var_end),
            latitude=range(0, self.xlat),
            longitude=range(self.xlon_start, self.xlon_end),
        )["fields"].values
        x = (
            x - self.norm_values["mean"][self.var_start : self.var_end]
        ) / self.norm_values["std"][self.var_start : self.var_end]
        if self.local_rank // 2 == 1:
            x = np.concatenate(
                [x, self.land_mask, self.soil_type, self.topography], axis=0
            )

        if (((self.num_vars + self.num_masks) % 2) == 1) and (
            self.local_rank // 2 == 1
        ):
            input = torch.nn.functional.pad(torch.tensor(x), (0, 0, 0, 0, 0, 1))
        else:
            input = torch.tensor(x)

        # Output data
        y = self.zarr_data.isel(
            time=target_idx,
            feature=range(self.var_start, self.var_end),
            latitude=range(0, self.xlat),
            longitude=range(self.xlon_start, self.xlon_end),
        )["fields"].values
        y = (
            y - self.norm_values["mean"][self.var_start : self.var_end]
        ) / self.norm_values["std"][self.var_start : self.var_end]

        if (((self.num_vars + self.num_masks) % 2) == 1) and (
            self.local_rank // 2 == 1
        ):
            target = torch.nn.functional.pad(
                torch.tensor(y), (0, 0, 0, 0, 0, self.num_masks + 1)
            )
        else:
            target = torch.tensor(y)

        return input, target


def load_constant_mask_2d(folder_path, xlat=721):
    """
    Load the constant masks applied in the patch embedding layer.

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

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
        mask_path = config["data"]["constant_masks_path"]
        mask_list = config["data"]["constant_masks"]

        # Filter desired time points based on parameters and return file statistics.
        self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")
        times = pd.to_datetime(self.zarr_data.time)
        train_years = times[(times >= self.years[0]) & (times < self.years[1])]
        self.zarr_data = self.zarr_data.sel(time=train_years)

        self.n_data_items = len(self.zarr_data["time"])
        self.num_vars = len(self.zarr_data["feature"])  # self.num_vars

        self.constant_masks = self.__load_constant_mask_2d(mask_path, mask_list)
        self.num_masks = self.constant_masks.shape[0]

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
            self.in_var = [0, self.num_vars // 2]
            self.in_masks = [
                0,
                self.num_masks // 2
                if self.num_masks % 2 == 0
                else self.num_masks // 2 + 1,
            ]
            self.in_out_xlon = [0, self.xlon // 2]
        elif self.local_rank == 1:
            self.in_var = [0, self.num_vars // 2]
            self.in_masks = [
                0,
                self.num_masks // 2
                if self.num_masks % 2 == 0
                else self.num_masks // 2 + 1,
            ]
            self.in_out_xlon = [self.xlon // 2, self.xlon]
        elif self.local_rank == 2:
            self.in_var = [self.num_vars // 2, self.num_vars]
            self.in_masks = [
                self.num_masks // 2
                if self.num_masks % 2 == 0
                else self.num_masks // 2 + 1,
                self.num_masks,
            ]
            self.in_out_xlon = [0, self.xlon // 2]
        elif self.local_rank == 3:
            self.in_var = [self.num_vars // 2, self.num_vars]
            self.in_masks = [
                self.num_masks // 2
                if self.num_masks % 2 == 0
                else self.num_masks // 2 + 1,
                self.num_masks,
            ]
            self.in_out_xlon = [self.xlon // 2, self.xlon]
        else:
            self.in_var = [0, self.num_vars]
            self.in_masks = [0, self.num_masks]
            self.in_out_xlon = [0, self.xlon]

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
            feature=range(self.in_var[0], self.in_var[1]),
            latitude=range(0, self.xlat),
            longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
        )["fields"].values
        x = (
            x - self.norm_values["mean"][self.in_var[0] : self.in_var[1]]
        ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]]

        input = torch.cat((torch.tensor(x), self.get_constant_masks()), dim=0)

        if (((self.num_vars + self.num_masks) % 2) == 1) and (
            self.local_rank // 2 == 1
        ):
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, 1))

        # Output data
        y = self.zarr_data.isel(
            time=target_idx,
            feature=range(self.in_var[0], self.in_var[1]),
            latitude=range(0, self.xlat),
            longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
        )["fields"].values
        y = (
            y - self.norm_values["mean"][self.in_var[0] : self.in_var[1]]
        ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]]

        target = torch.tensor(y)
        print(f"input.shape {input.shape}, target.shape {target.shape}")
        return input, target

    def __load_constant_mask_2d(self, folder_path, mask_list):
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

    def get_constant_masks(self):
        return torch.tensor(
            self.constant_masks[
                self.in_masks[0] : self.in_masks[1],
                : self.xlat,
                self.in_out_xlon[0] : self.in_out_xlon[1],
            ]
        ).to(torch.float32)

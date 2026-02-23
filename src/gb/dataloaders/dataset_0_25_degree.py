import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
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
            DTP Process group
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

        if self.process_group.size() > 1:
            row, col = self.local_rank % 2, self.local_rank // 2
            # rank 0: [0,0], rank 1 = [1,0], rank 2 = [0,1], rank 3 = [1,1]
            n_masks = (
                self.num_masks // 2
                if self.num_masks % 2 == 0
                else self.num_masks // 2 + 1
            )

            self.in_var = [row * self.num_vars // 2, (row + 1) * self.num_vars // 2]
            self.in_masks = [row * n_masks, (row + 1) * n_masks]
            self.in_out_xlon = [col * self.xlon // 2, (col + 1) * self.xlon // 2]

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

        # Odd ranks load second half of variables, so they might have less masks
        if (((self.num_vars + self.num_masks) % 2) == 1) and (self.local_rank % 2 == 1):
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
        return input, target

    def get_constant_masks(self):
        return torch.tensor(
            self.constant_masks[
                self.in_masks[0] : self.in_masks[1],
                : self.xlat,
                self.in_out_xlon[0] : self.in_out_xlon[1],
            ]
        ).to(torch.float32)

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


class ERA5DatasetEightwayDistributed(ERA5DatasetDistributed):
    def __init__(
        self,
        config,
        mode,
        process_group,
        normalize=True,
        dt=None,
    ):
        super().__init__(config, mode, process_group, normalize=normalize, dt=dt)
        if self.process_group.size() > 1:
            row, col = self.local_rank % 2, self.local_rank // 2

            # rank 0: [0,0], rank 1 = [1,0],
            # rank 2 = [0,1], rank 3 = [1,1],
            # rank 4 = [0,2], rank 5 = [1,2],
            # rank 6 = [0,3], rank 7 = [1,3]
            n_masks = (
                self.num_masks // 2
                if self.num_masks % 2 == 0
                else self.num_masks // 2 + 1
            )

            self.in_var = [row * self.num_vars // 2, (row + 1) * self.num_vars // 2]
            self.in_masks = [row * n_masks, (row + 1) * n_masks]
            self.in_out_xlon = [col * self.xlon // 4, (col + 1) * self.xlon // 4]

        else:
            self.in_var = [0, self.num_vars]
            self.in_masks = [0, self.num_masks]
            self.in_out_xlon = [0, self.xlon]


## Note: this doesn't work so far due to backend trouble between cpu and gpu; it will act as boilerplate if needed for data io optimization
class ERA5DatasetScattered(Dataset):
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

        self.constant_masks = self._load_constant_mask_2d(mask_path, mask_list)
        self.num_masks = self.constant_masks.shape[0]

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

        self.in_var = [0, self.num_vars // 2, self.num_vars]
        self.in_out_xlon = [0, self.xlon // 2, self.xlon]

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
        loaderID = 0

        # create empty buffer
        x = torch.empty(
            (
                self.in_var[1] - self.in_var[0],
                self.xlat,
                self.in_out_xlon[1] - self.in_out_xlon[0],
            ),
            dtype=torch.float32,
        )
        y = torch.empty(
            (
                self.in_var[1] - self.in_var[0],
                self.xlat,
                self.in_out_xlon[1] - self.in_out_xlon[0],
            ),
            dtype=torch.float32,
        )

        if self.local_rank == loaderID:
            data_x = self.zarr_data.isel(time=idx, latitude=range(0, self.xlat))[
                "fields"
            ].values
            data_x = (data_x - self.norm_values["mean"]) / self.norm_values["std"]

            data_x = torch.tensor(data_x)

            scatter_list = [
                data_x[
                    self.in_var[0] : self.in_var[1],
                    :,
                    self.in_out_xlon[0] : self.in_out_xlon[1],
                ].clone(),
                data_x[
                    self.in_var[0] : self.in_var[1],
                    :,
                    self.in_out_xlon[1] : self.in_out_xlon[2],
                ].clone(),
                data_x[
                    self.in_var[1] : self.in_var[2],
                    :,
                    self.in_out_xlon[0] : self.in_out_xlon[1],
                ].clone(),
                data_x[
                    self.in_var[1] : self.in_var[2],
                    :,
                    self.in_out_xlon[1] : self.in_out_xlon[2],
                ].clone(),
            ]

        dist.scatter(
            tensor=x,
            scatter_list=scatter_list,
            group=self.process_group,
            group_src=loaderID,
        )

        x = torch.cat((x, self.get_constant_masks()), dim=0)

        if (((self.num_vars + self.num_masks) % 2) == 1) and (
            self.local_rank // 2 == 1
        ):
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 1))

        # Output data
        if self.local_rank == loaderID:
            data_y = self.zarr_data.isel(time=target_idx, latitude=range(0, self.xlat))[
                "fields"
            ].values
            data_y = (data_y - self.norm_values["mean"]) / self.norm_values["std"]

            data_y = torch.tensor(data_y)

            scatter_list = [
                data_y[
                    self.in_var[0] : self.in_var[1],
                    :,
                    self.in_out_xlon[0] : self.in_out_xlon[1],
                ].clone(),
                data_y[
                    self.in_var[0] : self.in_var[1],
                    :,
                    self.in_out_xlon[1] : self.in_out_xlon[2],
                ].clone(),
                data_y[
                    self.in_var[1] : self.in_var[2],
                    :,
                    self.in_out_xlon[0] : self.in_out_xlon[1],
                ].clone(),
                data_y[
                    self.in_var[1] : self.in_var[2],
                    :,
                    self.in_out_xlon[1] : self.in_out_xlon[2],
                ].clone(),
            ]

        dist.scatter(
            tensor=y,
            scatter_list=scatter_list,
            group=self.process_group,
            group_src=loaderID,
        )

        return x, y

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

    def get_constant_masks(self):
        row, col = self.local_rank // 2, self.local_rank % 2
        n_masks = (
            self.num_masks // 2 if self.num_masks % 2 == 0 else self.num_masks // 2 + 1
        )

        in_masks = [row * n_masks, (row + 1) * n_masks]
        in_lon = [col * self.xlon // 2, (col + 1) * self.xlon // 2]

        return torch.tensor(
            self.constant_masks[
                in_masks[0] : in_masks[1],
                : self.xlat,
                in_lon[0] : in_lon[1],
            ]
        ).to(torch.float32)

import os
import time

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
        device,
        normalize=True,
        dt=None,
        dtype=torch.float32,
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
        self.dtype = dtype
        self.device = device
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
        start = time.perf_counter()
        self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")
        times = pd.to_datetime(self.zarr_data.time)
        train_years = times[(times >= self.years[0]) & (times < self.years[1])]
        self.zarr_data = self.zarr_data.sel(time=train_years)

        end = time.perf_counter()
        delta_t = end - start
        sleep_time = 1200 - delta_t
        if sleep_time <= 0:
            sleep_time = 1200 + abs(sleep_time)
        time.sleep(sleep_time)

        self.n_data_items = len(self.zarr_data["time"])
        self.num_vars = len(self.zarr_data["feature"])  # self.num_vars

        self.constant_masks = self._load_constant_mask_2d(mask_path, mask_list)
        self.num_masks = self.constant_masks.shape[0]
        # self.num_masks = 3
        # normalization for pressure levels and standard deviation
        self.norm_values = {}
        self.norms = self._load_normalizations(config)

        # self.norms_handle.wait()
        self.norm_values["mean"] = (
            self.norms[0, :].unsqueeze(dim=1).unsqueeze(dim=2).numpy()
        )
        self.norm_values["std"] = (
            self.norms[1, :].unsqueeze(dim=1).unsqueeze(dim=2).numpy()
        )

        self.received_masks_and_norms = False

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
            # split into 2 "longitude" pieces for 4-way parallel and 4 for 8-way
            self.in_out_xlon = [
                col * self.xlon // (process_group.size() // 2),
                (col + 1) * self.xlon // (process_group.size() // 2),
            ]

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
        # if not self.received_masks_and_norms:
        #     #self.mask_handle.wait()
        #     #self.norms_handle.wait()
        #     self.constant_masks = self.constant_masks.to("cpu")

        #     self.norm_values["mean"] = self.norms[0, :].to("cpu").numpy()
        #     self.norm_values["std"] = self.norms[1, :].to("cpu").numpy()

        #     self.received_masks_and_norms = True

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

        input = torch.cat(
            (
                torch.tensor(x, dtype=self.dtype),
                self.get_constant_masks().to(self.dtype),
            ),
            dim=0,
        )

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

        target = torch.tensor(y, dtype=self.dtype)
        return input, target

    def get_constant_masks(self):
        return self.constant_masks[
            self.in_masks[0] : self.in_masks[1],
            : self.xlat,
            self.in_out_xlon[0] : self.in_out_xlon[1],
        ]

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

        masks = torch.zeros(
            (3, self.xlat, self.xlon), dtype=self.dtype, requires_grad=False
        ).to(self.device)

        if self.local_rank == 0:
            data_files = [f for f in os.listdir(folder_path) if f in mask_list]
            for i, file in enumerate(data_files):
                data = np.load(os.path.join(folder_path, file))
                masks[i, :, :] = torch.from_numpy(data[: self.xlat, :]).detach().clone()

        # handle = dist.broadcast(masks, src=0, async_op=True)
        dist.broadcast(masks, src=0, group=None)
        masks = masks.to(torch.device("cpu"))
        return masks

    def _load_normalizations(self, config):

        norms = torch.zeros((2, self.num_vars), requires_grad=False).to(self.device)
        if self.local_rank == 0:
            mean = np.load(f"{config['data']['normalization_dir']}/norm_mean.npy")
            std = np.load(f"{config['data']['normalization_dir']}/norm_std.npy")
            norms[0, :] = torch.from_numpy(mean).detach().clone().squeeze()
            norms[1, :] = torch.from_numpy(std).detach().clone().squeeze()

        # handle = dist.broadcast(norms, src=0, async_op=True)
        dist.broadcast(norms, src=0, group=None)
        norms = norms.to(torch.device("cpu"))

        return norms


class ERA5TwoTimestepsDatasetDistributed(ERA5DatasetDistributed):
    def __init__(
        self,
        config,
        mode,
        process_group,
        device,
        normalize=True,
        dt=None,
        dtype=torch.float32,
    ):
        super().__init__(
            config, mode, process_group, device, normalize=normalize, dt=dt, dtype=dtype
        )
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
            feature=range(self.in_var[0], self.in_var[1]),
            latitude=range(0, self.xlat),
            longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
        )["fields"].values
        x_t1 = (
            x_t1 - self.norm_values["mean"][self.in_var[0] : self.in_var[1]]
        ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]]

        x_t2 = self.zarr_data.isel(
            time=(idx + self.dt),
            feature=range(self.in_var[0], self.in_var[1]),
            latitude=range(0, self.xlat),
            longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
        )["fields"].values
        x_t2 = (
            x_t2 - self.norm_values["mean"][self.in_var[0] : self.in_var[1]]
        ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]]

        input = torch.cat(
            (
                torch.tensor(x_t2, dtype=self.dtype),
                torch.tensor(x_t1, dtype=self.dtype),
                self.get_constant_masks().to(self.dtype),
            ),
            dim=0,
        )

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

        target = torch.tensor(y, dtype=self.dtype)
        return input, target

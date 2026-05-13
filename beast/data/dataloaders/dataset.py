import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import xarray as xr
from gb.utils import execute_in_waves
from torch.utils.data import Dataset


class ERA5DatasetDistributed(Dataset):
    """Define 3D dataset."""

    def __init__(
        self,
        process_group,
        data_path,
        constant_masks_path,
        constant_masks,
        normalization_dir,
        time_range=None,
        dt=6,
        n_in_timesteps=1,
        n_out_timesteps=1,
        normalize=True,
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
        self.dt = dt
        self.in_timesteps = n_in_timesteps
        self.out_timesteps = n_out_timesteps

        self.dtype = dtype
        self.process_group = process_group
        self.local_rank = process_group.rank()

        self.datapath = data_path
        self.years = time_range

        self.normalize = normalize

        mask_path = constant_masks_path
        mask_list = constant_masks

        # Filter desired time points based on parameters and return file statistics.
        # start = time.perf_counter()
        self.norm_values = {}

        def load_data():
            self.zarr_data = xr.open_dataset(self.datapath, engine="zarr")

            self.xlat = self.zarr_data.dims["latitude"]
            self.xlon = self.zarr_data.dims["longitude"]
            if self.xlat%2 == 1:
                self.xlat -= 1
            if self.xlon%2 == 1:
                self.xlon -= 1

            times = pd.to_datetime(self.zarr_data.time)
            train_years = times[(times >= self.years[0]) & (times < self.years[1])]
            self.zarr_data = self.zarr_data.sel(time=train_years)

            self.n_data_items = self.zarr_data.dims["time"]
            self.num_vars = self.zarr_data.dims["feature"]  # self.num_vars

            self.constant_masks = self._load_constant_mask_2d(mask_path, mask_list)
            self.num_masks = self.constant_masks.shape[0]

            self.norm_values["mean"] = np.load(
                f"{normalization_dir}/norm_mean.npy"
            )
            self.norm_values["std"] = np.load(
                f"{normalization_dir}/norm_std.npy"
            )

        execute_in_waves(load_data)

        if self.process_group.size() > 1:
            row, col = self.local_rank % 2, self.local_rank // 2
            # rank 0: [0,0], rank 1 = [1,0], rank 2 = [0,1], rank 3 = [1,1]
            self.in_var = [row * self.num_vars // 2, (row + 1) * self.num_vars // 2]
          
            # We split the channels into two ranks; however, if we have an odd number of channels this will
            # result in rank 0 having one more variable.
            # since we do this for each timestep we accumulate an inbalance in the rank=1 channel group
            input_inbalance = (self.num_vars % 2) * self.in_timesteps
            output_inbalance = (self.num_vars % 2) * self.out_timesteps
            # We can use the constant masks to counteract this 
            # First, we compute how many masks we can add to rank=1 channel group to counteract the imbalance
            self.input_pad_oddrank = 0
            self.output_pad_oddrank = output_inbalance

            n_masks = (input_inbalance if row == 1 else 0)
            # Two cases:
            # If we have more masks then the inbalance, we need to distribute the remaining masks evenly accross channel ranks
            if (input_inbalance - self.num_masks) <= 0: 
                n_masks += (
                    (self.num_masks - input_inbalance)// 2
                    if self.num_masks % 2 == 0
                    else (self.num_masks - input_inbalance) // 2 + 1
                )
            # If we have fewer masks then the imbalance, we need to pad on rank 1
            else:
                self.input_pad_oddrank = input_inbalance - self.num_masks

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
        return self.n_data_items - (self.in_timesteps + self.out_timesteps) * self.dt

    def __getitem__(self, idx):
        """
        Return single input, target.

        idx: int
                global index of item
        """
        target_idx = idx + self.in_timesteps * self.dt

        input_list = []
        for i in range(self.in_timesteps): 
            x = self.zarr_data.isel(
                time=(idx + i*self.dt),
                feature=range(self.in_var[0], self.in_var[1]),
                latitude=range(0, self.xlat),
                longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
            )["fields"].values

            x = (
                x - self.norm_values["mean"][self.in_var[0] : self.in_var[1]]
            ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]]

            input_list.append(torch.tensor(x, dtype=self.dtype))
        
        input_list.append(self.get_constant_masks().to(self.dtype))
        input = torch.cat(input_list, dim=0)

        # For distributed dataset, we may create inbalances in the variables
        if self.process_group.size() > 1:
            # Odd ranks load second half of variables, so they might have less masks
            if (self.input_pad_oddrank != 0) and (self.local_rank % 2 == 1):
                input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, self.input_pad_oddrank))

        
        # Output data
        target_list = []
        for j in range(self.out_timesteps): 
            y = self.zarr_data.isel(
                time=(target_idx + j*self.dt),
                feature=range(self.in_var[0], self.in_var[1]),
                latitude=range(0, self.xlat),
                longitude=range(self.in_out_xlon[0], self.in_out_xlon[1]),
            )["fields"].values

            y = (
                y - self.norm_values["mean"][self.in_var[0] : self.in_var[1]]
            ) / self.norm_values["std"][self.in_var[0] : self.in_var[1]]

            target_list.append(torch.tensor(y, dtype=self.dtype))
        
        target = torch.cat(target_list, dim=0)

        if self.process_group.size() > 1:
            # Odd ranks load second half of variables, so they might have less masks
            if (self.output_pad_oddrank != 0) and (self.local_rank % 2 == 1):
                target = torch.nn.functional.pad(target, (0, 0, 0, 0, 0, self.output_pad_oddrank))

        return input, target

    def get_constant_masks(self):
        return torch.tensor(
            self.constant_masks[
                self.in_masks[0] : self.in_masks[1],
                : self.xlat,
                self.in_out_xlon[0] : self.in_out_xlon[1],
            ]
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

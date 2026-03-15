#!/usr/bin/env python3
"""
Test script for loading data from storage to GPUs.
Purpose is to test different loading strategies, depending on the system architecture
"""

import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

from torch.utils.data import Dataset
import torch.distributed as dist


def test_data_io():
    group_dict: dict[str, dist.ProcessGroup] = init_gb_process_group([1,1,1,2,2])
    device = torch.device(f'cuda:{rank}')
    if torch.distributed.get_rank(group_dict["ddp"]) == 0:
        data_item = torch.zeros((84, 360, 720))
        data_item[]
   zarr_data = xr.open_dataset(data_path, engine="zarr")
    '/lsdf/kit/scc/projects/bellbeast/era5_stackedvars_test.zarr/'

if __name__ == "__main__":
    group_dict: dict[str, dist.ProcessGroup] = init_gb_process_group([1,1,1,2,2])

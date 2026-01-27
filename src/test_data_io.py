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
from gb.utils import init_gb_process_group, get_process_group


def load_data_chunked(data_path, idx, process_group, device, dim = [84, 721, 1440]):

    zarr_data = xr.open_dataset(data_path, engine="zarr")
    
    var,xlat,xlon = dim
                
    if torch.distributed.get_rank(group_dict["DTP"]) in [0, 1]:
        var_start, var_end = 0, var // 2
    else:
        var_start, var_end = var // 2 , var
        
    if torch.distributed.get_rank(group_dict["DTP"]) % 2 == 0:
        xlon_start, xlon_end = 0, xlon // 2
    else:
        xlon_start, xlon_end = xlon // 2, xlon
    
    #print(f'Rank {rank} reads data in [{var_start}:{var_end}, 0:{xlat}, {xlon_start}:{xlon_end}]')
    data_chunk = torch.tensor(zarr_data.isel(time=idx, feature=range(var_start, var_end), latitude=range(0, xlat), longitude=range(xlon_start, xlon_end))["fields"].values, device=device)
    
   # print(f'Rank {rank} has tensor of size {data_chunk.shape} on device {device}')
    #print(f'Rank {rank} filled data chunk: element[0,1,0]={data_chunk[0,1,0]}')

    
def load_data_scatter(data_path, idx, process_group, device, dim = [84, 721, 1440]):

        
    var,xlon,xlat = dim

    scatter_list = None
    data_chunk = torch.empty((var//2, xlon, xlat//2), dtype=torch.float32, device=device)
    #print(f'Rank {rank} empty data chunk: element[0,0,0]={data_chunk[0,0,0]}')

    #Only Node-local Rank 0 gets to load the data
    if torch.distributed.get_rank(group_dict["DTP"]) == 0:
        #data_item = torch.zeros((var, x_lon, x_lat), dtype=torch.float32, device=torch.device('cpu'))
        #data_item[:var//2,:,:x_lat//2]= 0
        #data_item[:var//2,:,x_lat//2:]= 1
        #data_item[var//2:,:,:x_lat//2]= 2
        #data_item[var//2:,:,x_lat//2:]= 3
        
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(zarr_data.isel(time=idx)["fields"].values, device=device)

        scatter_list = [data_item[:var//2,:,:xlat//2].clone(),
                        data_item[:var//2,:,xlat//2:].clone(), 
                        data_item[var//2:,:,:xlat//2].clone(), 
                        data_item[var//2:,:,xlat//2:].clone() ]

    
    dist.scatter(tensor=data_chunk, scatter_list=scatter_list, group=process_group, group_src=0)


    #print(f'Rank {rank} has tensor of size {data_chunk.shape} on device {device}')
    #print(f'Rank {rank} filled data chunk: element[0,1,0]={data_chunk[0,1,0]}')


if __name__ == "__main__":
    
    mesh_dims = [1, 1, 1, 2, 2]
    group_dict: dict[str, dist.ProcessGroup] = init_gb_process_group(mesh_dims)

    process_group = get_process_group("DTP")
    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    rank = process_group.rank()
    device = f"cuda:{slurm_localid}"
    world_size = process_group.size()

    device = torch.device(f'cuda:{slurm_localid}')

    dim = [84, 721, 1440]
    data_path = '/lsdf/kit/scc/projects/bellbeast/era5_stackedvars_test.zarr/' 
    
    for idx in range(10):
        load_data_chunked(data_path, idx, process_group, device, dim)
        #load_data_scatter(data_path, idx, process_group, device, dim)

    dist.destroy_process_group()


#!/usr/bin/env python3
"""
Test script for loading data from storage to GPUs.
Purpose is to test different loading strategies, depending on the system architecture
"""

import os
import argparse
import yaml

import numpy as np
import pandas as pd
import torch
import xarray as xr

from torch.utils.data import Dataset
import torch.distributed as dist
from gb.utils import init_gb_process_group, get_process_group


def load_data_chunked_4way(data_path, idx, process_group, device, load_order=[0,1,2,3], dim = [84, 721, 1440]):
    """
    Load data in a chunked way, i.e. each rank loads its shard of the data directly from the zarr file.
    Mapping of the data chunks to the local_ranks will be determined by load_order
        load_order[0] --> [0:var/2, 0:xlon/2]
        load_order[1] --> [0:var/2, xlon/2:xlon]
        load_order[2] --> [var/2:var, 0:xlon/2]
        load_order[3] --> [var/2:var, xlon/2:var]
    
    Parameters
    ----------
    data_path: str
        Path to the dataset (zarr format)
    idx: int
        Data item index, as in time point
    process_group: torch.distributed.process_group
        Process group from the process_dict on which to operate. Should be "DTP" for all things and purposes here 
    device: torch.device
        Device on which to load the data; should be either 'cpu' or 'cuda:local_rank' 
    load_order: List(int)
        A list of 4 integers, indicating which chunk of the data to load onto which local_rank
    dim: List(int)
        Dimensions of the datacube. Can probably be inferred from the zarr file itself...
    """
    
    local_rank=torch.distributed.get_rank(group_dict["DTP"])
    zarr_data = xr.open_dataset(data_path, engine="zarr")
    var, xlat, xlon = dim
                
    if local_rank == load_order[0]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = 0, xlon // 2
    elif local_rank == load_order[1]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = xlon // 2, xlon
    elif local_rank == load_order[2]:
        var_start, var_end = var // 2 , var
        xlon_start, xlon_end = 0, xlon // 2
    elif local_rank == load_order[3]:
        var_start, var_end = var // 2 , var
        xlon_start, xlon_end = xlon // 2, xlon
    else:
        print(f'Unknown local_rank {local_rank}. I`m not loading anything')
        return
    
    #print(f'Rank {rank} reads data in [{var_start}:{var_end}, 0:{xlat}, {xlon_start}:{xlon_end}]')
    #print(f'#######CHUNKED:\n Rank {rank} loading data {data_path} item {idx}')

    data_chunk = torch.tensor(zarr_data.isel(time=idx, feature=range(var_start, var_end), latitude=range(0, xlat), longitude=range(xlon_start, xlon_end))["fields"].values, device=device)
    
   # print(f'Rank {rank} has tensor of size {data_chunk.shape} on device {device}')
    #print(f'Rank {rank} filled data chunk: element[0,1,0]={data_chunk[0,1,0]}')

    
def load_data_scatter_4way(data_path, idx, process_group, loaderID, gpu_direct=True , dim = [84, 721, 1440]):
    """
    Load data item `idx` from `data_path` zarr file to rank `loaderID`, and scatter to all other processes in the `process_group`.
    Parameters
    ----------
    data_path: str
        Path to the dataset (zarr format)
    idx: int
        Data item index, as in time point
    process_group: torch.distributed.process_group
        Process group from the process_dict on which to operate. Should be "DTP" for all things and purposes here 


    """
    var, xlon, xlat = dim

    scatter_list = None
    data_chunk = torch.empty((var//2, xlon, xlat//2), dtype=torch.float32, device=device)
    #print(f'Rank {rank} empty data chunk: element[0,0,0]={data_chunk[0,0,0]}')

    #Only Node-local Rank 0 gets to load the data
    if torch.distributed.get_rank(group_dict["DTP"]) == 0:
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(zarr_data.isel(time=idx)["fields"].values, device=device)

        scatter_list = [data_item[:var//2,:,:xlat//2].clone(),
                        data_item[:var//2,:,xlat//2:].clone(), 
                        data_item[var//2:,:,:xlat//2].clone(), 
                        data_item[var//2:,:,xlat//2:].clone() ]

    dist.scatter(tensor=data_chunk, scatter_list=scatter_list, group=process_group, group_src=0)



def load_data_scatter_8way(data_path, idx, process_group, device, dim = [84, 721, 1440]):
    
    var,xlon,xlat = dim

    scatter_list = None
    data_chunk = torch.empty((var//2, xlon, xlat//4), dtype=torch.float32, device=device)
    #print(f'Rank {rank} empty data chunk: element[0,0,0]={data_chunk[0,0,0]}')

    #Only Node-local Rank 0 gets to load the data
    if torch.distributed.get_rank(group_dict["DTP"]) == 0:        
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(zarr_data.isel(time=idx)["fields"].values, device=device)

        scatter_list = [data_item[:var//2,:,:xlat//4].clone(),
                        data_item[:var//2,:,xlat//4:2*xlat//4].clone(),
                        data_item[:var//2,:,2*xlat//4:3*xlat//4].clone(), 
                        data_item[:var//2,:,3*xlat//4:].clone(),
                        data_item[var//2:,:,:xlat//4].clone(),
                        data_item[var//2:,:,xlat//4:2*xlat//4].clone(),
                        data_item[var//2:,:,2*xlat//4:3*xlat//4].clone(), 
                        data_item[var//2:,:,3*xlat//4:].clone()]

    
    dist.scatter(tensor=data_chunk, scatter_list=scatter_list, group=process_group, group_src=0)


    #print(f'Rank {rank} has tensor of size {data_chunk.shape} on device {device}')
    #print(f'Rank {rank} filled data chunk: element[0,1,0]={data_chunk[0,1,0]}')


def load_data_chunked_8way(data_path, idx, process_group, device, dim = [84, 721, 1440]):

    local_rank = torch.distributed.get_rank(group_dict["DTP"])
    zarr_data = xr.open_dataset(data_path, engine="zarr")
    
    var,xlat,xlon = dim
                
    if local_rank%2 == 0:
        var_start, var_end = 0, var // 2
    else:
        var_start, var_end = var // 2 , var
        
    if  local_rank // 2 == 0:
        xlon_start, xlon_end = 0, xlon // 4
    elif local_rank // 2 == 1:
        xlon_start, xlon_end = xlon // 4, 2* xlon // 4
    elif local_rank // 2 == 2:
        xlon_start, xlon_end = 2*xlon // 4, 3* xlon // 4
    else:
        xlon_start, xlon_end = 3* xlon // 4, xlon
        
    data_chunk = torch.tensor(zarr_data.isel(time=idx, feature=range(var_start, var_end), latitude=range(0, xlat), longitude=range(xlon_start, xlon_end))["fields"].values, device=device)


def load_data_scatterchunked_8way(data_path, idx, process_group, device, dim = [84, 721, 1440]):
    
    var,xlon,xlat = dim
    
    local_rank = torch.distributed.get_rank(group_dict["DTP"])
    if local_rank%2 == 0:
        var_start, var_end = 0, var // 2
    else:
        var_start, var_end = var // 2 , var

    if  local_rank // 2 == 0:
        xlon_start, xlon_end = 0, xlon // 4
    elif local_rank // 2 == 1:
        xlon_start, xlon_end = xlon // 4, 2* xlon // 4
    elif local_rank // 2 == 2:
        xlon_start, xlon_end = 2*xlon // 4, 3* xlon // 4
    else:
        xlon_start, xlon_end = 3* xlon // 4, xlon

    data_chunk = torch.empty((var//2, xlon, xlat//4), dtype=torch.float32, device=device)

    #Even Node-local rank load the data and send half of it to odd node-local ranks
    if local_rank%2 == 0:        
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(zarr_data.isel(time=idx, longitude=range(xlon_start, xlon_end))["fields"].values, device=device)
        data_chunk = data_item[0:var // 2,:,:]
        dist.isend(data_item[var // 2:,:,:].clone(), group=process_group, tag=0, group_dst=(local_rank+1))
    
    else:
        dist.irecv(data_chunk, src=None, group=process_group, group_src=(local_rank-1))

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(f"{args.config_file}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    mesh_dims = config_dict["mesh_dims"]
    group_dict: dict[str, dist.ProcessGroup] = init_gb_process_group(mesh_dims)

    process_group = get_process_group("DTP")
    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    rank = process_group.rank()
    device = f"cuda:{slurm_localid}"
    world_size = process_group.size()

    device = torch.device(f'cuda:{slurm_localid}')

    dim = config_dict["data"]["dims"]
    if os.getenv("DATA_DIR") is not None:
        data_path = (os.getenv("DATA_DIR") + config_dict["data"]["path"])
    else:
        data_path = '/lsdf/kit/scc/projects/bellbeast/era5_stackedvars_test.zarr/' 
    
    for idx in range(config_dict["data"]["num_items"]):
        load_data_chunked_8way(data_path, idx, process_group, device, dim)
        load_data_scatter_8way(data_path, idx, process_group, device, dim)

    dist.destroy_process_group()


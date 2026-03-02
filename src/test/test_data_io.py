#!/usr/bin/env python3
"""
Test script for loading data from storage to GPUs.
Purpose is to test different loading strategies, depending on the system architecture
"""

import sys

sys.path.append("./src/")  # for utils import

import argparse
import os
import time

import torch
import torch.distributed as dist
import xarray as xr
import yaml

from gb.utils import get_process_group, init_gb_process_group


def load_data_chunked_4way(
    data_path,
    idx,
    process_group,
    gpu_direct=True,
    load_order=[0, 1, 2, 3],
    dim=[84, 721, 1440],
):
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
    gpu_direct: bool
        Whether to load the data directly to GPU. If true, device on which to load data will be 'cuda:local_rank'
    load_order: List(int)
        A list of 4 integers, indicating which chunk of the data to load onto which local_rank
    dim: List(int)
        Dimensions of the datacube. Can probably be inferred from the zarr file itself...
    """

    local_rank = torch.distributed.get_rank(group_dict["DTP"])
    device_id = torch.cuda.current_device()
    zarr_data = xr.open_dataset(data_path, engine="zarr")
    var, xlat, xlon = dim

    if gpu_direct:
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    if local_rank == load_order[0]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = 0, xlon // 2
    elif local_rank == load_order[1]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = xlon // 2, xlon
    elif local_rank == load_order[2]:
        var_start, var_end = var // 2, var
        xlon_start, xlon_end = 0, xlon // 2
    elif local_rank == load_order[3]:
        var_start, var_end = var // 2, var
        xlon_start, xlon_end = xlon // 2, xlon
    else:
        print(f"Unknown local_rank {local_rank}. I`m not loading anything")
        return

    # print(f'Rank {rank} reads data in [{var_start}:{var_end}, 0:{xlat}, {xlon_start}:{xlon_end}]')
    # print(f'#######CHUNKED:\n Rank {rank} loading data {data_path} item {idx}')

    data_chunk = torch.tensor(
        zarr_data.isel(
            time=idx,
            feature=range(var_start, var_end),
            latitude=range(0, xlat),
            longitude=range(xlon_start, xlon_end),
        )["fields"].values,
        device=device,
    )
    torch.cuda.synchronize()
    return data_chunk


# print(f'Rank {rank} has tensor of size {data_chunk.shape} on device {device}')
# print(f'Rank {rank} filled data chunk: element[0,1,0]={data_chunk[0,1,0]}')


def load_data_scatter_4way(
    data_path, idx, process_group, loaderID, gpu_direct=True, dim=[84, 721, 1440]
):
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
    loader_ID: int
        Rank on process group that loads the data from storage and scatters to the other ranks in the pg; change to determine differences in storage connection
    gpu_direct: bool
        Whether to load the data directly to GPU. If true, device on which to load data will be 'cuda:local_rank'
    dim: List(int)
        Dimensions of the datacube. Can probably be inferred from the zarr file itself...

    """
    var, xlon, xlat = dim
    scatter_list = None
    local_rank = torch.distributed.get_rank(group_dict["DTP"])
    device_id = torch.cuda.current_device()

    if loaderID not in range(torch.distributed.get_world_size(group_dict["DTP"])):
        print(
            f"Loader ID {loaderID} not in Process Group Size {torch.distributed.get_world_size(group_dict['DTP'])}. I`m not loading anything"
        )
        return

    if gpu_direct:
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    data_chunk = torch.empty(
        (var // 2, xlon, xlat // 2), dtype=torch.float32, device=device
    )
    # print(f'Rank {rank} empty data chunk: element[0,0,0]={data_chunk[0,0,0]}')

    # Only Node-local Rank <loaderID> gets to load the data
    if local_rank == loaderID:
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(
            zarr_data.isel(time=idx)["fields"].values, device=device
        )

        scatter_list = [
            data_item[: var // 2, :, : xlat // 2].clone(),
            data_item[: var // 2, :, xlat // 2 :].clone(),
            data_item[var // 2 :, :, : xlat // 2].clone(),
            data_item[var // 2 :, :, xlat // 2 :].clone(),
        ]

    dist.scatter(
        tensor=data_chunk,
        scatter_list=scatter_list,
        group=process_group,
        group_src=loaderID,
    )
    torch.cuda.synchronize()


def load_data_scatter_8way(
    data_path, idx, process_group, loaderID, gpu_direct=True, dim=[84, 721, 1440]
):
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
    loader_ID: int
        Rank on process group that loads the data from storage and scatters to the other ranks in the pg; change to determine differences in storage connection
    gpu_direct: bool
        Whether to load the data directly to GPU. If true, device on which to load data will be 'cuda:local_rank'
    dim: List(int)
        Dimensions of the datacube. Can probably be inferred from the zarr file itself...
    """
    var, xlon, xlat = dim
    scatter_list = None

    device_id = torch.cuda.current_device()

    if loaderID not in range(torch.distributed.get_world_size(group_dict["DTP"])):
        print(
            f"Loader ID {loaderID} not in Process Group Size {torch.distributed.get_world_size(group_dict['DTP'])}. I`m not loading anything"
        )
        return

    if gpu_direct:
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    data_chunk = torch.empty(
        (var // 2, xlon, xlat // 4), dtype=torch.float32, device=device
    )

    # Only Node-local Rank <loaderID> gets to load the data
    if torch.distributed.get_rank(group_dict["DTP"]) == loaderID:
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(
            zarr_data.isel(time=idx)["fields"].values, device=device
        )

        scatter_list = [
            data_item[: var // 2, :, : xlat // 4].clone(),
            data_item[: var // 2, :, xlat // 4 : 2 * xlat // 4].clone(),
            data_item[: var // 2, :, 2 * xlat // 4 : 3 * xlat // 4].clone(),
            data_item[: var // 2, :, 3 * xlat // 4 :].clone(),
            data_item[var // 2 :, :, : xlat // 4].clone(),
            data_item[var // 2 :, :, xlat // 4 : 2 * xlat // 4].clone(),
            data_item[var // 2 :, :, 2 * xlat // 4 : 3 * xlat // 4].clone(),
            data_item[var // 2 :, :, 3 * xlat // 4 :].clone(),
        ]

    dist.scatter(
        tensor=data_chunk, scatter_list=scatter_list, group=process_group, group_src=0
    )
    torch.cuda.synchronize()


from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_world_size = comm.Get_size()

dt_extent = (1440 // 4) * MPI.FLOAT.Get_extent()[1]
subarray_type = (
    MPI.FLOAT.Create_subarray(
        sizes=[84, 721, 1440], subsizes=[84 // 2, 721, 1440 // 4], starts=[0] * 3
    )
    .Create_resized(0, dt_extent)
    .Commit()
)


def load_data_scatter_8way_mpi_scatter_dt(
    data_path, idx, process_group, loaderID, gpu_direct=False, dim=[84, 721, 1440]
):
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
    loader_ID: int
        Rank on process group that loads the data from storage and scatters to the other ranks in the pg; change to determine differences in storage connection
    gpu_direct: bool
        Whether to load the data directly to GPU. If true, device on which to load data will be 'cuda:local_rank'
    dim: List(int)
        Dimensions of the datacube. Can probably be inferred from the zarr file itself...
    """
    var, xlon, xlat = dim

    device_id = torch.cuda.current_device()

    if loaderID not in range(torch.distributed.get_world_size(group_dict["DTP"])):
        print(
            f"Loader ID {loaderID} not in Process Group Size {torch.distributed.get_world_size(group_dict['DTP'])}. I`m not loading anything"
        )
        return

    device = f"cuda:{device_id}"

    if gpu_direct:
        start_device = device
    else:
        start_device = "cpu"

    data_chunk = torch.zeros(
        (var // 2, xlon, xlat // 4), dtype=torch.float32, device=start_device
    )
    send_buffer = None

    # Only Node-local Rank <loaderID> gets to load the data
    if torch.distributed.get_rank(group_dict["DTP"]) == loaderID:
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(
            zarr_data.isel(time=idx)["fields"].values, device=start_device
        )
        var_jump = 4 * xlon * (var // 2)
        send_buffer = (
            MPI.buffer.fromaddress(data_item.untyped_storage().data_ptr(), 0),
            (
                (1,) * 8,
                (0, 1, 2, 3, var_jump, var_jump + 1, var_jump + 2, var_jump + 3),
            ),
            subarray_type,
        )

    recv_buffer = (
        MPI.buffer.fromaddress(data_chunk.untyped_storage().data_ptr(), 0),
        data_chunk.numel(),
        MPI.FLOAT,
    )

    comm.Scatterv(send_buffer, recv_buffer, root=0)
    if not gpu_direct:
        data_chunk = data_chunk.to(device)

    torch.cuda.synchronize()
    return data_chunk


def load_data_scatter_8way_mpi_cont(
    data_path, idx, process_group, loaderID, gpu_direct=False, dim=[84, 721, 1440]
):
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
    loader_ID: int
        Rank on process group that loads the data from storage and scatters to the other ranks in the pg; change to determine differences in storage connection
    gpu_direct: bool
        Whether to load the data directly to GPU. If true, device on which to load data will be 'cuda:local_rank'
    dim: List(int)
        Dimensions of the datacube. Can probably be inferred from the zarr file itself...
    """
    var, xlon, xlat = dim
    scatter_list = None

    device_id = torch.cuda.current_device()

    device = f"cuda:{device_id}"

    if gpu_direct:
        start_device = device
    else:
        start_device = "cpu"

    if loaderID not in range(torch.distributed.get_world_size(group_dict["DTP"])):
        print(
            f"Loader ID {loaderID} not in Process Group Size {torch.distributed.get_world_size(group_dict['DTP'])}. I`m not loading anything"
        )
        return

    # Only Node-local Rank <loaderID> gets to load the data
    if torch.distributed.get_rank(group_dict["DTP"]) == loaderID:
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(
            zarr_data.isel(time=idx)["fields"].values, device=start_device
        )

        scatter_list = [
            data_item[: var // 2, :, : xlat // 4].clone().contiguous(),
            data_item[: var // 2, :, xlat // 4 : 2 * xlat // 4].clone().contiguous(),
            data_item[: var // 2, :, 2 * xlat // 4 : 3 * xlat // 4]
            .clone()
            .contiguous(),
            data_item[: var // 2, :, 3 * xlat // 4 :].clone().contiguous(),
            data_item[var // 2 :, :, : xlat // 4].clone().contiguous(),
            data_item[var // 2 :, :, xlat // 4 : 2 * xlat // 4].clone().contiguous(),
            data_item[var // 2 :, :, 2 * xlat // 4 : 3 * xlat // 4]
            .clone()
            .contiguous(),
            data_item[var // 2 :, :, 3 * xlat // 4 :].clone().contiguous(),
        ]

    data_chunk = comm.scatter(scatter_list, root=0)
    if not gpu_direct:
        data_chunk = data_chunk.to(device)
    torch.cuda.synchronize()
    return data_chunk


def load_data_chunked_8way(
    data_path,
    idx,
    process_group,
    gpu_direct=True,
    load_order=[0, 1, 2, 3, 4, 5, 6, 7],
    dim=[84, 721, 1440],
):
    """
    Load data in a chunked way, i.e. each rank loads its shard of the data directly from the zarr file.
    Mapping of the data chunks to the local_ranks will be determined by load_order
        rank <load_order[0]> --> [0:var/2, 0:xlon/4]
        rank <load_order[1]> --> [var/2:var, 0:xlon/4]
        rank <load_order[2]> --> [0:var/2, xlon/4:2*xlon/4]
        rank <load_order[3]> --> [var/2:var, xlon/4:2*xlon/4]
        rank <load_order[4]> --> [0:var/2, 2*xlon/4:3*xlon/4]
        rank <load_order[5]> --> [var/2:var, 2*xlon/4:3*xlon/4]
        rank <load_order[6]> --> [0:var/2, 3*xlon/4:xlon]
        rank <load_order[7]> --> [var/2:var, 3*xlon/4:xlon]

    Parameters
    ----------
    data_path: str
        Path to the dataset (zarr format)
    idx: int
        Data item index, as in time point
    process_group: torch.distributed.process_group
        Process group from the process_dict on which to operate. Should be "DTP" for all things and purposes here
    gpu_direct: bool
        Whether to load the data directly to GPU. If true, device on which to load data will be 'cuda:local_rank'
    load_order: List(int)
        A list of 4 integers, indicating which chunk of the data to load onto which local_rank
    dim: List(int)
        Dimensions of the datacube. Can probably be inferred from the zarr file itself...
    """

    local_rank = torch.distributed.get_rank(group_dict["DTP"])
    device_id = torch.cuda.current_device()
    zarr_data = xr.open_dataset(data_path, engine="zarr", mode="r")

    var, xlat, xlon = dim

    if local_rank == load_order[0]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = 0, xlon // 4
    elif local_rank == load_order[1]:
        var_start, var_end = var // 2, var
        xlon_start, xlon_end = 0, xlon // 4

    elif local_rank == load_order[2]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = xlon // 4, 2 * xlon // 4
    elif local_rank == load_order[3]:
        var_start, var_end = var // 2, var
        xlon_start, xlon_end = xlon // 4, 2 * xlon // 4

    elif local_rank == load_order[4]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = 2 * xlon // 4, 3 * xlon // 4
    elif local_rank == load_order[5]:
        var_start, var_end = var // 2, var
        xlon_start, xlon_end = 2 * xlon // 4, 3 * xlon // 4

    elif local_rank == load_order[6]:
        var_start, var_end = 0, var // 2
        xlon_start, xlon_end = 3 * xlon // 4, xlon
    elif local_rank == load_order[7]:
        var_start, var_end = var // 2, var
        xlon_start, xlon_end = 3 * xlon // 4, xlon
    else:
        print(f"Unknown local_rank {local_rank}. I`m not loading anything")
        return

    if gpu_direct:
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    data_chunk = torch.tensor(
        zarr_data.isel(
            time=idx,
            feature=range(var_start, var_end),
            latitude=range(0, xlat),
            longitude=range(xlon_start, xlon_end),
        )["fields"].values,
        device=device,
    )
    return data_chunk


def load_data_scatterchunked_8way(
    data_path, idx, process_group, gpu_direct=True, dim=[84, 721, 1440]
):
    var, xlat, xlon = dim
    scatter_list = None

    channel_group = group_dict["JChannel"]
    local_rank = torch.distributed.get_rank(group_dict["DTP"])
    device_id = torch.cuda.current_device()

    if gpu_direct:
        device = f"cuda:{device_id}"
    else:
        device = "cpu"

    if local_rank // 2 == 0:
        xlon_start, xlon_end = 0, xlon // 4
    elif local_rank // 2 == 1:
        xlon_start, xlon_end = xlon // 4, 2 * xlon // 4
    elif local_rank // 2 == 2:
        xlon_start, xlon_end = 2 * xlon // 4, 3 * xlon // 4
    else:
        xlon_start, xlon_end = 3 * xlon // 4, xlon

    data_chunk = torch.empty(
        (var // 2, xlat, xlon // 4), dtype=torch.float32, device=device
    )

    # Even Node-local rank load the data and send half of it to odd node-local ranks
    if local_rank % 2 == 0:
        zarr_data = xr.open_dataset(data_path, engine="zarr")
        data_item = torch.tensor(
            zarr_data.isel(time=idx, longitude=range(xlon_start, xlon_end))[
                "fields"
            ].values,
            device=device,
        )
        scatter_list = [
            data_item[0 : var // 2, :, :].clone(),
            data_item[var // 2 :, :, :].clone(),
        ]
    dist.scatter(
        tensor=data_chunk, scatter_list=scatter_list, group=channel_group, group_src=0
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(f"{args.config_file}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    mesh_dims = config_dict["mesh_dims"]
    group_dict: dict[str, dist.ProcessGroup] = init_gb_process_group(mesh_dims)

    # num_workers = int(os.getenv("SLURM_CPUS_PER_TASK") or 1)
    # zarr.config.set({"threading.max_workers": (num_workers - 2)})

    process_group = get_process_group("DTP")
    rank = process_group.rank()

    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)
    group_size = process_group.size()
    world_size = dist.get_world_size(group=None)
    print(
        f"Local Rank {rank}, global rank {dist.get_rank(group=None)} \n Local size {group_size}, global size {dist.get_world_size(group=None)}"
    )

    dim = config_dict["data"]["dims"]
    if os.getenv("DATA_DIR") is not None:
        data_path = os.getenv("DATA_DIR") + config_dict["data"]["path"]
    else:
        data_path = "/lsdf/kit/scc/projects/bellbeast/era5_stackedvars_test.zarr/"

    ############# ACTUAL EXPERIMENTS#############
    ####### JUPITER Configuration #######
    if group_size == 4:
        # Dry run for warm up
        t = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            _ = load_data_chunked_4way(
                data_path,
                idx,
                process_group,
                gpu_direct=True,
                load_order=[0, 1, 2, 3],
                dim=[84, 721, 1440],
            )
        dist.barrier()

        # Cunked loading (load order [0,1,2,3])
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            _ = load_data_chunked_4way(
                data_path,
                idx,
                process_group,
                gpu_direct=True,
                load_order=[0, 1, 2, 3],
                dim=[84, 721, 1440],
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_chunked_4way = torch.zeros(4, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(time_chunked_4way, tensor_in, group=process_group)
        dist.barrier()

        # Cunked loading (load order [0,2,1,3])
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            _ = load_data_chunked_4way(
                data_path,
                idx,
                process_group,
                gpu_direct=True,
                load_order=[0, 2, 1, 3],
                dim=[84, 721, 1440],
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_chunked2_4way = torch.zeros(4, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(time_chunked2_4way, tensor_in, group=process_group)
        dist.barrier()

        ### Rank 0 is loading
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_4way(
                data_path, idx, process_group, 0, gpu_direct=True, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_4way_0 = torch.zeros(4, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(time_scatter_4way_0, tensor_in, group=process_group)
        dist.barrier()

        ### Rank 1 is loading
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_4way(
                data_path, idx, process_group, 1, gpu_direct=True, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_4way_1 = torch.zeros(4, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(time_scatter_4way_1, tensor_in, group=process_group)
        dist.barrier()

        ### Rank 2 is loading
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_4way(
                data_path, idx, process_group, 2, gpu_direct=True, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_4way_2 = torch.zeros(4, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(time_scatter_4way_2, tensor_in, group=process_group)
        dist.barrier()

        ### Rank 3 is loading
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_4way(
                data_path, idx, process_group, 3, gpu_direct=True, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_4way_3 = torch.zeros(4, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(time_scatter_4way_3, tensor_in, group=process_group)
        dist.barrier()

        if rank == 0:
            print(
                f"RESULTS on rank {dist.get_rank(group=None)} (out of {time_chunked_4way.shape} elements):\n Chunked with load order [0,1,2,3] : Mean {time_chunked_4way.mean().item()}, Max {time_chunked_4way.max().item()}\n Chunked with load order [0,2,1,3] : Mean {time_chunked2_4way.mean().item()}, Max {time_chunked2_4way.max().item()}\n Scatter with loaderID = 0: Mean {time_scatter_4way_0.mean().item()}, Max {time_scatter_4way_0.max().item()}\n Scatter with loaderID = 1: Mean {time_scatter_4way_1.mean().item()}, Max {time_scatter_4way_1.max().item()}\n Scatter with loaderID = 2: Mean {time_scatter_4way_2.mean().item()}, Max {time_scatter_4way_2.max().item()}\n Scatter with loaderID = 3: Mean {time_scatter_4way_3.mean().item()}, Max {time_scatter_4way_3.max().item()}\n"
            )

    ####### LUMI Configuration #######
    elif group_size == 8:
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            _ = load_data_chunked_8way(
                data_path,
                idx,
                process_group,
                gpu_direct=True,
                load_order=[0, 1, 2, 3, 4, 5, 6, 7],
                dim=[84, 721, 1440],
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_chunked_8way_loadorder1 = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_chunked_8way_loadorder1, tensor_in)
        dist.barrier()
        if rank == 0:
            print("First")

        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            _ = load_data_chunked_8way(
                data_path,
                idx,
                process_group,
                gpu_direct=True,
                load_order=[0, 2, 4, 6, 1, 3, 5, 7],
                dim=[84, 721, 1440],
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_chunked_8way_loadorder2 = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_chunked_8way_loadorder2, tensor_in)
        dist.barrier()
        if rank == 0:
            print("Second")

        ### Rank 0 is loading
        # Scatter
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_8way(
                data_path, idx, process_group, 0, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_8way_0 = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_scatter_8way_0, tensor_in)
        dist.barrier()
        if rank == 0:
            print("Third")

        # Scatter mpi dt
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_8way_mpi_scatter_dt(
                data_path, idx, process_group, 0, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_8way_mpi_dt_0 = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_scatter_8way_mpi_dt_0, tensor_in)
        dist.barrier()

        # Scatter mpi dt direct
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_8way_mpi_scatter_dt(
                data_path, idx, process_group, 0, True, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_8way_mpi_dt_direct_0 = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_scatter_8way_mpi_dt_direct_0, tensor_in)
        dist.barrier()
        if rank == 0:
            print("Fourth")

        # Scatter mpi cont
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_8way_mpi_cont(
                data_path, idx, process_group, 0, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_8way_mpi_cont_0 = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_scatter_8way_mpi_cont_0, tensor_in)
        dist.barrier()

        # Scatter mpi cont direct
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatter_8way_mpi_cont(
                data_path, idx, process_group, 0, True, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatter_8way_mpi_cont_direct_0 = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_scatter_8way_mpi_cont_direct_0, tensor_in)
        dist.barrier()
        if rank == 0:
            print("Fifth")

        # Scatter chunked
        start_time = time.perf_counter()
        for idx in range(config_dict["data"]["num_items"]):
            load_data_scatterchunked_8way(
                data_path, idx, process_group, gpu_direct=True, dim=[84, 721, 1440]
            )
        end_time = time.perf_counter()
        tensor_in = torch.tensor(
            [end_time - start_time], dtype=torch.float64, device=device
        )
        time_scatterchunked_8way = torch.zeros(
            world_size, dtype=torch.float64, device=device
        )
        dist.all_gather_into_tensor(time_scatterchunked_8way, tensor_in)
        dist.barrier()
        if rank == 0:
            print("Last")

        if rank == 0:
            print(
                f"""RESULTS:
                Chunked with load order [0,1,2,3,4,5,6,7] : Mean {time_chunked_8way_loadorder1.mean().item()}, Max {time_chunked_8way_loadorder1.max().item()}
                Chunked with load order [0,2,4,6,1,3,5,7] : Mean {time_chunked_8way_loadorder2.mean().item()}, Max {time_chunked_8way_loadorder2.max().item()}
                Scatter with loaderID = 0                 : Mean {time_scatter_8way_0.mean().item()}, Max {time_scatter_8way_0.max().item()}
                Scatter mpi dt with loaderID = 0          : Mean {time_scatter_8way_mpi_dt_0.mean().item()}, Max {time_scatter_8way_mpi_dt_0.max().item()}
                Scatter mpi dt direct with loaderID = 0   : Mean {time_scatter_8way_mpi_dt_direct_0.mean().item()}, Max {time_scatter_8way_mpi_dt_direct_0.max().item()}
                Scatter mpi cont with loaderID = 0        : Mean {time_scatter_8way_mpi_cont_0.mean().item()}, Max {time_scatter_8way_mpi_cont_0.max().item()}
                Scatter mpi cont direct with loaderID = 0 : Mean {time_scatter_8way_mpi_cont_direct_0.mean().item()}, Max {time_scatter_8way_mpi_cont_direct_0.max().item()}
                Scatter-chunked                           : Mean {time_scatterchunked_8way.mean().item()}, Max {time_scatterchunked_8way.max().item()}
                """
            )
    else:
        print(f"I don`t know what to do with group_size {group_size}")

    dist.barrier()
    dist.destroy_process_group()

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import zarr
import datetime

from beast.data.dataset import ERA5DatasetDistributed, DummyDataset

def generate_mock_zarr_data(zarr_path, n_time, n_features, n_lat, n_lon):
    # Time: 24 datetime64[ns] steps (hourly starting from 2024-01-01)
    time_coords = pd.date_range("2024-01-01", periods=n_time, freq="h")

    surf_vars = [
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "sea_surface_temperature",
        "total_precipitation",
    ]
    plev_vars = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]

    levels = np.arange(1000, n_features//6, -50, dtype=int)
    feature_names = surf_vars + [var + "_" + str(p) for var in plev_vars for p in levels]

    lat_coords = np.linspace(90, -90, n_lat, dtype=np.float32)
    lon_coords = np.linspace(0, 360, n_lon, endpoint=False, dtype=np.float32)

    for t in range(n_time):
        np.random.seed(t)
        data_step = np.random.rand(1, n_features, n_lat, n_lon).astype(np.float32)

        ds_step = xr.Dataset(
            data_vars={
                "fields": xr.DataArray(
                    data=data_step,
                    dims=["time", "feature", "latitude", "longitude"],
                    coords={
                        "time": [time_coords[t]],
                        "feature": feature_names[:n_features],
                        "latitude": lat_coords,
                        "longitude": lon_coords,
                    },
                )
            }
        )
 
        if t == 0:
            ds_step.to_zarr(zarr_path, mode="w")
        else:
            ds_step.to_zarr(zarr_path, append_dim="time")

    return zarr_path

def generate_mock_masks(path, xlat, xlon, n_masks):
    """Generate n_masks numpy files, each containing an [xlat, xlon] array of ones.

    Args:
        path: Directory path where the .npy files will be saved.
        xlat: Number of latitude points (rows).
        xlon: Number of longitude points (columns).
        n_masks: Number of mask files to generate.
    """
    mask_list = []
    for i in range(n_masks):
        mask_name = f"mask_{i:04d}.npy"
        mask = np.ones((xlat, xlon), dtype=np.float32) * i
        filepath = path / mask_name
        mask_list.append(mask_name)
        np.save(filepath, mask)
    
    return mask_list

def generate_mock_normalization(path, n_features):
    mean = np.zeros((n_features, 1,1), dtype=np.float32) 
    std = np.ones((n_features,1,1 ), dtype=np.float32) 

    np.save(path/f"norm_mean.npy", mean)
    np.save(path/f"norm_std.npy", std)


test_array = [
    (3, 8, 121, 240, 1, 1), 
    # (3, 8, 60,  120, 1, 1), 
    # (3, 8, 121, 240, 2, 2), 
    # (3, 8, 60,  120, 2, 2),
    # (3, 8, 121, 240, 2, 1), 
    # (3, 8, 60,  120, 2, 1),
    # (4, 8, 121, 240, 2, 1), 
    # (4, 8, 60,  120, 2, 1),
    # (3, 7, 121, 240, 2, 1), 
    # (3, 7, 60,  120, 2, 1),
    # (4, 7, 121, 240, 2, 1), 
    # (4, 7, 60,  120, 2, 1)
    ]


test_array_xlat_xlon = [(120, 240),
    (121, 240),
    (120, 241),
    (121, 241)]

test_array_mask_features = [(3, 8),
    (4, 8),
    (3, 7), 
    (4, 7)]


class TestDataset:
    @pytest.mark.parametrize("n_time, n_features, n_lat, n_lon", [(24,84,721,1440), (24,84,120,240)])
    def test_mock_dataset(self, tmp_path_factory, n_time, n_features, n_lat, n_lon):
        zarr_path = tmp_path_factory.mktemp("data") / "test_data.zarr"
        generate_mock_zarr_data(zarr_path, n_time,n_features,n_lat,n_lon)
        zarr_data = xr.open_dataset(zarr_path, engine="zarr")
        assert zarr_data.dims["latitude"] == n_lat

    @pytest.mark.parametrize("n_masks, n_lat, n_lon", [(3,721,1440), (3,120,240)])
    def test_generate_masks(self, tmp_path_factory, n_masks, n_lat, n_lon):
        mask_path = tmp_path_factory.mktemp("data") 

        mask_list = generate_mock_masks(mask_path, n_lat, n_lon, n_masks)

        files = sorted(mask_path.glob("mask_*.npy"))
        assert len(files) == n_masks
        assert len(mask_list) == n_masks

        for f in files:
            mask = np.load(f)
            assert mask.shape == (n_lat, n_lon)



    @pytest.mark.parametrize("n_masks, n_features", test_array_mask_features)
    def test_ERA5DatasetDistributed_mask_features(self,tmp_path_factory, n_masks, n_features): 
        data_path = tmp_path_factory.mktemp("data") 
        n_time = 48
        world_size=4

        xlat = 120
        xlon = 240
        in_timesteps = 1
        out_timesteps = 1

        n_vars = n_masks + n_features
        if n_vars %2 ==1:
            n_vars +=1

        zarr_path = generate_mock_zarr_data((data_path/ "test_data.zarr"), n_time, n_features, xlat, xlon)
        mask_list = generate_mock_masks(data_path, xlat, xlon, n_masks)
        generate_mock_normalization(data_path, n_features)

        dataset_r0 = ERA5DatasetDistributed(
            0,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
            )

        dataset_r1 = ERA5DatasetDistributed(
            1,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
        )

        dataset_r2 = ERA5DatasetDistributed(
            2,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
            )

        dataset_r3 = ERA5DatasetDistributed(
            3,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
            )

        x_0, target_0 = dataset_r0[0]       
        x_1, _ = dataset_r1[0]
        x_2, _ = dataset_r2[0]
        x_3, _ = dataset_r3[0]

        assert len(dataset_r0) == n_time - 6
        assert list(x_0.shape) == [n_vars//2, 120, 120]
        if n_features % 2 == 1:
            n_features+=1
        assert list(target_0.shape) == [n_features//2, 120, 120]
        assert list(x_0.shape) == list(x_1.shape) #== list(x_2.shape) == list(x_3.shape) 


    @pytest.mark.parametrize("lat, lon", test_array_xlat_xlon)
    def test_ERA5DatasetDistributed_spatial(self,tmp_path_factory, lat, lon): 
        data_path = tmp_path_factory.mktemp("data") 
        n_time = 48
        world_size=4
        in_timesteps = 1
        out_timesteps = 1
        n_masks = 3
        n_features = 8
        n_vars = n_masks + 1 + n_features

        zarr_path = generate_mock_zarr_data((data_path/ "test_data.zarr"), n_time, n_features, lat, lon)
        mask_list = generate_mock_masks(data_path, lat, lon, n_masks)
        generate_mock_normalization(data_path, n_features)

        dataset_r0 = ERA5DatasetDistributed(
            0,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
            )

        assert len(dataset_r0) == n_time - (in_timesteps + out_timesteps) * 6

        dataset_r1 = ERA5DatasetDistributed(
            1,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
        )

        dataset_r2 = ERA5DatasetDistributed(
            2,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
            )

        dataset_r3 = ERA5DatasetDistributed(
            3,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
            dsync_access=False,
            )


        xlat = lat
        xlon = lon
        if xlat%2 ==1:
            xlat -= 1
        
        if xlon%2 ==1:
            xlon -= 1        
        
        x_0, target_0 = dataset_r0[0]
        assert list(x_0.shape) == [n_vars//2 , xlat, xlon//2]
        assert list(target_0.shape) == [n_features//2, xlat, xlon//2]
       
        x_1, _ = dataset_r1[0]
        x_2, _ = dataset_r2[0]
        x_3, _ = dataset_r3[0]

        assert list(x_0.shape) == list(x_1.shape) #== list(x_2.shape) == list(x_3.shape) 


    @pytest.mark.parametrize("n_masks, n_features, n_lat, n_lon, in_timesteps, out_timesteps", test_array)
    def test_DummyDataset(self,n_masks, n_features, n_lat, n_lon, in_timesteps, out_timesteps): 
        n_time = 48
        time_coords = pd.date_range("2024-01-01", periods=n_time, freq="h").strftime("%Y-%m-%d")
        rank = 1
        world_size = 4

        shape = (n_features, n_lat, n_lon)

        dataset = DummyDataset(
            rank,
            world_size,
            shape, 
            time_range=[str(time_coords[0]), str(time_coords[-1])],
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,)
        

        #assert len(dataset) == n_time - (in_timesteps+out_timesteps) * 6
        
        if n_lat%2 ==1:
            xlat = n_lat-1
        
        if n_lon%2 ==1:
            xlon = n_lon-1

        x, target = dataset[0]

        #assert list(x.shape) == [in_timesteps* n_features//2 + n_masks//2, xlat, xlon//2]
        #assert list(target.shape) == [out_timesteps* n_features//2 + n_masks//2, xlat, xlon//2]


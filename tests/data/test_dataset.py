import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import zarr


from beast.data.dataset import ERA5DatasetDistributed

def mock_zarr_data(zarr_path, n_time, n_features, n_lat, n_lon):
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

    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    feature_names = surf_vars + [var + "_" + str(p) for var in plev_vars for p in levels]

    # Latitude and Longitude as float32
    lat_coords = np.linspace(90, -90, n_lat, dtype=np.float32)
    lon_coords = np.linspace(0, 360, n_lon, endpoint=False, dtype=np.float32)

    # --- Write to Zarr one time step at a time (memory efficient) ---

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
                        "feature": feature_names,
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

def generate_masks(path, xlat, xlon, n_masks):
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

def generate_normalization(path, n_features):
    mean = np.zeros((n_features, ), dtype=np.float32) 
    std = np.ones((n_features, ), dtype=np.float32) 

    np.save(path/f"norm_mean.npy", mean)
    np.save(path/f"norm_std.npy", std)



class TestDataset:
    contents of test_image.py
    @pytest.mark.parametrize("n_time, n_features, n_lat, n_lon", [(24,84,721,1440), (24,84,120,240)])
    def test_mock_dataset(self, tmp_path_factory, n_time, n_features, n_lat, n_lon):
        zarr_path = tmp_path_factory.mktemp("data") / "test_data.zarr"
        mock_zarr_data(zarr_path, n_time,n_features,n_lat,n_lon)
        zarr_data = xr.open_dataset(zarr_path, engine="zarr")
        assert zarr_data.dims["latitude"] == n_lat

    @pytest.mark.parametrize("n_masks, n_lat, n_lon", [(3,721,1440), (3,120,240)])
    def test_generate_masks(self, tmp_path_factory, n_masks, n_lat, n_lon):
        mask_path = tmp_path_factory.mktemp("data") 

        mask_list = generate_masks(mask_path, n_lat, n_lon, n_masks)

        files = sorted(mask_path.glob("mask_*.npy"))
        assert len(files) == n_masks
        assert len(mask_list) == n_masks

        for f in files:
            mask = np.load(f)
            assert mask.shape == (n_lat, n_lon)


    @pytest.mark.parametrize("n_masks, n_time, n_features, n_lat, n_lon, in_timesteps, out_timesteps", [(3, 48,84,721,1440, 2,1), (3, 48,84,120,240,2,1)])
    def test_ERA5DatasetDistributed(self,tmp_path_factory, n_masks, n_time, n_features, n_lat, n_lon,in_timesteps, out_timesteps): 
        data_path = tmp_path_factory.mktemp("data") 
        
        zarr_path = mock_zarr_data((data_path/ "test_data.zarr"), n_time,n_features,n_lat,n_lon)
        mask_list = generate_masks(data_path, n_lat, n_lon, n_masks)
        generate_normalization(data_path, n_features)
        rank = 1
        world_size = 4

        dataset = ERA5DatasetDistributed(
            rank,
            world_size,
            zarr_path,
            data_path,
            mask_list,
            data_path,
            time_range=None,
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            normalize=True,
            dtype=torch.float32,)

        assert len(dataset) == n_time - (in_timesteps+out_timesteps)*6



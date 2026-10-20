import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Callable, List, cast

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from pytest import TempPathFactory

import beast.data.transforms as trf
from beast.data.era5 import ERA5Distributed, ERA5Dummy, EWBEra5


def generate_mock_zarr_data(
    zarr_path: str, n_time: int, n_features: int, n_lat: int, n_lon: int
) -> str:
    """Utility function that creates a mock dataset in zarr format, in the format of ERA5 data."""
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

    levels = np.arange(1000, n_features // 6, -50, dtype=int)
    feature_names = surf_vars + [
        var + "_" + str(p) for var in plev_vars for p in levels
    ]

    lat_coords = np.linspace(90, -90, n_lat, dtype=np.float32)
    lon_coords = np.linspace(0, 360, n_lon, endpoint=False, dtype=np.float32)
    encoding = {"time": {"units": f"hours since {time_coords[0]}", "dtype": "float64"}}
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
            ds_step.to_zarr(zarr_path, mode="w", encoding=encoding)
        else:
            ds_step.to_zarr(zarr_path, append_dim="time")

    return zarr_path


@pytest.fixture(scope="session")
def cached_mock_zarr(tmp_path_factory: TempPathFactory) -> Callable[..., str]:
    """Build mock zarr stores once per shape and reuse them across tests.

    ``generate_mock_zarr_data`` reseeds numpy per timestep, so a store is fully
    determined by ``(n_time, n_features, n_lat, n_lon)`` -- regenerating it for
    every parametrisation produced byte-identical copies. Twenty-two of those
    copies is roughly a gigabyte of scratch space, which is what made the suite
    fail on a tmpfs ``/tmp``.

    The stores are read-only as far as the tests are concerned: ``ERA5Distributed``
    opens them for reading, and masks and normalization files still go to each
    test's own directory, so nothing is shared that a test could mutate.

    Returns
    -------
    Callable[..., str]
        ``get(n_time, n_features, n_lat, n_lon)`` returning the store's path.
    """
    root = tmp_path_factory.mktemp("zarr_cache")
    cache: dict[tuple[int, int, int, int], str] = {}

    def get(n_time: int, n_features: int, n_lat: int, n_lon: int) -> str:
        key = (n_time, n_features, n_lat, n_lon)
        if key not in cache:
            path = root / f"mock_{n_time}_{n_features}_{n_lat}_{n_lon}.zarr"
            cache[key] = generate_mock_zarr_data(path, *key)
        return cache[key]

    return get


def generate_mock_masks(path: str, xlat: int, xlon: int, n_masks: int) -> List[str]:
    """Utility function that creates mock masks in .npy format, filled with ones."""

    mask_list = []
    for i in range(n_masks):
        mask_name = f"mask_{i:04d}.npy"
        mask = np.ones((xlat, xlon), dtype=np.float32) * i
        filepath = os.path.join(path, mask_name)
        mask_list.append(mask_name)
        np.save(filepath, mask)

    return mask_list


def generate_mock_normalization(path: str, n_features: int) -> None:
    """Utility function that creates mock normalization files in .npy format, filled with ones."""
    mean = np.zeros((n_features, 1, 1), dtype=np.float32)
    std = np.ones((n_features, 1, 1), dtype=np.float32)

    np.save(os.path.join(path, "norm_mean.npy"), mean)
    np.save(os.path.join(path, "norm_std.npy"), std)


test_array_dummy = [(8, 120, 240), (8, 121, 240), (7, 120, 241), (7, 121, 241)]
test_array_xlat_xlon = [(120, 240), (121, 240), (120, 241), (121, 241)]
test_array_mask_features = [(3, 8), (4, 8), (3, 7), (4, 7)]
test_array_timesteps = [(1, 1), (2, 1), (1, 2), (2, 2)]
test_array = [
    (3, 8, 121, 240, 1, 1),
    (3, 8, 60, 120, 1, 1),
    (3, 8, 121, 240, 2, 2),
    (3, 8, 60, 120, 2, 2),
    (3, 8, 121, 240, 2, 1),
    (3, 8, 60, 120, 2, 1),
    (4, 8, 121, 240, 2, 1),
    (4, 8, 60, 120, 2, 1),
    (3, 7, 121, 240, 2, 1),
    (3, 7, 60, 120, 2, 1),
    (4, 7, 121, 240, 2, 1),
    (4, 7, 60, 120, 2, 1),
]


class TestDataset:
    # The full ERA5 grid (721x1440, odd latitude count) is the point of the first
    # case -- it checks that the generator gets the coordinate bookkeeping right at
    # production resolution. The feature count is not: the assertion below only
    # looks at the latitude size, and 84 features at full resolution wrote 597 MB
    # of random floats per run. Eight keeps the grid and drops the store to ~57 MB.
    @pytest.mark.parametrize(
        "n_time, n_features, n_lat, n_lon", [(2, 8, 721, 1440), (2, 84, 120, 240)]
    )
    def test_mock_dataset(
        self,
        tmp_path_factory: TempPathFactory,
        n_time: int,
        n_features: int,
        n_lat: int,
        n_lon: int,
    ) -> None:
        """Test for utility function generate_mock_zarr_data"""
        zarr_path = tmp_path_factory.mktemp("data") / "test_data.zarr"
        generate_mock_zarr_data(zarr_path, n_time, n_features, n_lat, n_lon)
        zarr_data = xr.open_dataset(zarr_path, engine="zarr")
        assert zarr_data.sizes["latitude"] == n_lat

    @pytest.mark.parametrize("n_masks, n_lat, n_lon", [(3, 721, 1440), (3, 120, 240)])
    def test_generate_masks(
        self, tmp_path_factory: TempPathFactory, n_masks: int, n_lat: int, n_lon: int
    ) -> None:
        """Test for utility function generate_mock_masks"""
        mask_path = tmp_path_factory.mktemp("data")

        mask_list = generate_mock_masks(mask_path, n_lat, n_lon, n_masks)

        files = sorted(mask_path.glob("mask_*.npy"))
        assert len(files) == n_masks
        assert len(mask_list) == n_masks

        for f in files:
            mask = np.load(f)
            assert mask.shape == (n_lat, n_lon)

    @pytest.mark.parametrize("n_masks, n_features", test_array_mask_features)
    def test_era5_dataset_distributed_mask_features(
        self,
        tmp_path_factory: TempPathFactory,
        cached_mock_zarr: Callable[..., str],
        n_masks: int,
        n_features: int,
    ) -> None:
        """Test for class ERA5Distributed, for different numbers of masks and features to evaluate channel splitting functionality."""
        data_path = tmp_path_factory.mktemp("data")
        n_time = 48
        world_size = 4

        xlat = 120
        xlon = 240
        in_timesteps = 1
        out_timesteps = 1

        n_vars = n_masks + n_features
        if n_vars % 2 == 1:
            n_vars += 1

        zarr_path = cached_mock_zarr(n_time, n_features, xlat, xlon)
        mask_list = generate_mock_masks(data_path, xlat, xlon, n_masks)
        generate_mock_normalization(data_path, n_features)

        dataset_r0 = ERA5Distributed(
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

        dataset_r1 = ERA5Distributed(
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

        dataset_r2 = ERA5Distributed(
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

        dataset_r3 = ERA5Distributed(
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
        assert list(x_0.shape) == [n_vars // 2, 120, 120]
        if n_features % 2 == 1:
            n_features += 1
        assert list(target_0.shape) == [n_features // 2, 120, 120]
        assert list(x_0.shape) == list(x_1.shape) == list(x_2.shape) == list(x_3.shape)

    @pytest.mark.parametrize("lat, lon", test_array_xlat_xlon)
    def test_era5_dataset_distributed_spatial(
        self,
        tmp_path_factory: TempPathFactory,
        cached_mock_zarr: Callable[..., str],
        lat: int,
        lon: int,
    ) -> None:
        """Test for class ERA5Distributed, for different spatial input sizes, to test automatic size derivation from the dataset."""
        data_path = tmp_path_factory.mktemp("data")
        n_time = 48
        world_size = 4
        in_timesteps = 1
        out_timesteps = 1
        n_masks = 3
        n_features = 8
        n_vars = n_masks + 1 + n_features

        zarr_path = cached_mock_zarr(n_time, n_features, lat, lon)
        mask_list = generate_mock_masks(data_path, lat, lon, n_masks)
        generate_mock_normalization(data_path, n_features)

        dataset_r0 = ERA5Distributed(
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

        dataset_r1 = ERA5Distributed(
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

        dataset_r2 = ERA5Distributed(
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

        dataset_r3 = ERA5Distributed(
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

        assert len(dataset_r0) == n_time - 6

        xlat = lat
        xlon = lon
        if xlat % 2 == 1:
            xlat -= 1

        if xlon % 2 == 1:
            xlon -= 1

        x_0, target_0 = dataset_r0[0]
        assert list(x_0.shape) == [n_vars // 2, xlat, xlon // 2]
        assert list(target_0.shape) == [n_features // 2, xlat, xlon // 2]

        x_1, _ = dataset_r1[0]
        x_2, _ = dataset_r2[0]
        x_3, _ = dataset_r3[0]

        assert list(x_0.shape) == list(x_1.shape) == list(x_2.shape) == list(x_3.shape)

    @pytest.mark.parametrize("in_timesteps, out_timesteps", test_array_timesteps)
    def test_era5_dataset_distributed_timesteps(
        self,
        tmp_path_factory: TempPathFactory,
        cached_mock_zarr: Callable[..., str],
        in_timesteps: int,
        out_timesteps: int,
    ) -> None:
        """Test for class ERA5Distributed, for different number of input and output timesteps, to evaluate temporal concatenation."""
        data_path = tmp_path_factory.mktemp("data")
        n_time = 48
        world_size = 4
        xlat = 120
        xlon = 240
        n_masks = 3
        n_features = 8
        n_in_vars = 2 + 4 * in_timesteps
        n_out_vars = 4 * out_timesteps

        zarr_path = cached_mock_zarr(n_time, n_features, xlat, xlon)
        mask_list = generate_mock_masks(data_path, xlat, xlon, n_masks)
        generate_mock_normalization(data_path, n_features)

        dataset_r0 = ERA5Distributed(
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

        dataset_r1 = ERA5Distributed(
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

        dataset_r2 = ERA5Distributed(
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

        dataset_r3 = ERA5Distributed(
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

        assert len(dataset_r0) == n_time - (in_timesteps + out_timesteps - 1) * 6

        x_0, target_0 = dataset_r0[0]
        assert list(x_0.shape) == [n_in_vars, xlat, xlon // 2]
        assert list(target_0.shape) == [n_out_vars, xlat, xlon // 2]

        x_1, _ = dataset_r1[0]
        x_2, _ = dataset_r2[0]
        x_3, _ = dataset_r3[0]

        assert list(x_0.shape) == list(x_1.shape) == list(x_2.shape) == list(x_3.shape)

    @pytest.mark.parametrize("n_features, xlat, xlon", test_array_dummy)
    def test_dummy_dataset(self, n_features: int, xlat: int, xlon: int) -> None:
        """Test for class ERA5Dummy, for different input timestep sizes (spatial and variables)."""
        n_time = 48
        time_coords = pd.date_range(
            "2024-01-01", periods=n_time + 1, freq="h"
        ).strftime("%Y-%m-%d")
        rank = 1
        world_size = 4

        in_timesteps = 1
        out_timesteps = 1

        shape = (n_features, xlat, xlon)
        print(f"ERA5Dummy time_coords {time_coords}")

        dataset = ERA5Dummy(
            rank,
            world_size,
            shape,
            time_range=(str(time_coords[0]), str(time_coords[-1])),
            dt=6,
            n_in_timesteps=in_timesteps,
            n_out_timesteps=out_timesteps,
            dtype=torch.float32,
        )

        assert len(dataset) == n_time - (in_timesteps + out_timesteps - 1) * 6

        x, target = dataset[0]

        n_vars = n_features
        if n_vars % 2 == 1:
            n_vars += 1
        lat = xlat
        if lat % 2 == 1:
            lat -= 1
        lon = xlon
        if lon % 2 == 1:
            lon -= 1
        assert list(x.shape) == [n_vars // 2, lat, lon // 2]
        assert list(target.shape) == [n_vars // 2, lat, lon // 2]

    @pytest.mark.parametrize("n_masks, n_features", test_array_mask_features)
    def test_era5_dataset_distributed_transform_integration(
        self,
        tmp_path_factory: TempPathFactory,
        cached_mock_zarr: Callable[..., str],
        n_masks: int,
        n_features: int,
    ) -> None:
        """Test for class ERA5Distributed, for different numbers of masks and features to evaluate transformation functionality."""
        data_path = tmp_path_factory.mktemp("data")
        n_time = 48
        world_size = 4

        xlat = 120
        xlon = 240
        in_timesteps = 1
        out_timesteps = 1

        n_vars = n_masks + n_features
        if n_vars % 2 == 1:
            n_vars += 1

        zarr_path = cached_mock_zarr(n_time, n_features, xlat, xlon)
        mask_list = generate_mock_masks(data_path, xlat, xlon, n_masks)
        generate_mock_normalization(data_path, n_features)

        dataset_r0 = ERA5Distributed(
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

        dataset_r1 = ERA5Distributed(
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
            transform=trf.Downsample(factor=6, dims=(-1, -2)),
        )

        dataset_r2 = ERA5Distributed(
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
            transform=trf.DownsampleMeanStd(factor=6, dims=(-1, -2)),
        )

        dataset_r3 = ERA5Distributed(
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
            transform=trf.DownsampleMeanStd(factor=6, dims=(-1, -2)),
            target_transform=trf.Downsample(factor=6, dims=(-1, -2)),
        )

        x_0, target_0 = dataset_r0[0]
        x_1, target_1 = dataset_r1[0]
        x_2, target_2 = dataset_r2[0]
        x_3, target_3 = dataset_r3[0]

        assert len(dataset_r0) == n_time - 6
        assert list(x_0.shape) == [n_vars // 2, 120, 120]
        if n_features % 2 == 1:
            n_features += 1
        assert list(target_0.shape) == [n_features // 2, 120, 120]

        assert list(x_1.shape) == [n_vars // 2, 20, 20]
        assert list(target_1.shape) == [n_features // 2, 20, 20]

        assert list(x_2.shape) == [n_vars // 2, 20, 20, 2]
        assert list(target_2.shape) == [n_features // 2, 20, 20, 2]

        assert list(x_3.shape) == [n_vars // 2, 20, 20, 2]
        assert list(target_3.shape) == [n_features // 2, 20, 20]


class TestEWBEra5:
    """Tests for the ExtremeWeatherBench forecast dataset."""

    @pytest.mark.parametrize("lead_time", [-24, 0, 6, 12, 24, 48])
    @pytest.mark.parametrize("dt", [6, 12])
    @pytest.mark.parametrize("in_timesteps", [1, 2, 3])
    def test_ewb_input_dates(self, in_timesteps: int, dt: int, lead_time: int) -> None:
        """The last input must land exactly on the initialization time.

        Consumers start their rollout clock at ``start_date - lead_time`` and
        step forward by ``dt``. If the last input does not sit on that time, the
        whole forecast is shifted and every field is stored under the wrong
        valid time, while still looking self-consistent.
        """
        dataset = SimpleNamespace(dt=dt, in_timesteps=in_timesteps)
        start_date = datetime(2022, 1, 10, 0)

        dates = EWBEra5._input_dates(cast(EWBEra5, dataset), start_date, lead_time)

        assert len(dates) == in_timesteps
        # The forecast is initialized at start_date - lead_time.
        assert dates[-1] == start_date - timedelta(hours=lead_time)
        # Inputs run forward in time, dt apart.
        assert dates == sorted(dates)
        assert all(
            later - earlier == timedelta(hours=dt)
            for earlier, later in zip(dates, dates[1:])
        )

    def test_ewb_input_dates_single_timestep(self) -> None:
        """With one input there is no history, so it is the initialization time."""
        dataset = SimpleNamespace(dt=12, in_timesteps=1)
        start_date = datetime(2022, 1, 10, 0)

        assert EWBEra5._input_dates(cast(EWBEra5, dataset), start_date, 48) == [
            datetime(2022, 1, 8, 0)
        ]

    def test_ewb_input_dates_spans_history(self) -> None:
        """Multiple inputs reach back before the initialization time, never past it."""
        dataset = SimpleNamespace(dt=12, in_timesteps=2)
        start_date = datetime(2022, 1, 10, 0)

        assert EWBEra5._input_dates(cast(EWBEra5, dataset), start_date, 48) == [
            datetime(2022, 1, 7, 12),
            datetime(2022, 1, 8, 0),
        ]

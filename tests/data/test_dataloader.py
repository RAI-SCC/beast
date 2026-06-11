import math
from typing import Any

import pandas as pd
import pytest
import torch
from pytest import TempPathFactory
from test_dataset import (
    generate_mock_masks,
    generate_mock_normalization,
    generate_mock_zarr_data,
)

from beast.data.dataloader import get_dataloader


def create_mock_config() -> dict[str, Any]:
    """Create a mock dataset and corresponding mock config dictionary."""
    config = {
        "data": {
            "dt": 6,
            "train_data_path": "test_data.zarr",
            "valid_data_path": "test_data.zarr",
            "constant_masks_path": "test_masks",
            "constant_masks": ["mask_0001.npy", "mask_0002.npy", "mask_0003.npy"],
            "normalization_dir": "normalization",
            "train_subset": -1,
            "valid_subset": -1,
            "train_time_range": ["2000-01-01", "2000-12-31"],
            "valid_time_range": ["2001-01-01", "2001-12-31"],
            "xlat": 721,
            "xlon": 1440,
            "n_variables": 84,
            "n_in_timesteps": 1,
            "n_out_timesteps": 1,
            "prefetch_factor": 2,
            "num_workers": 2,
        },
        "training": {
            "batch_size": 1,
            "dtype": torch.float32,
        },
    }

    return config


# n_time, n_features, n_lat, n_lon, n_masks
data_array = [(48, 8, 72, 144, 3), (120, 8, 12, 24, 3)]
# n_time, batch size, dt
train_array = [(48, 1, 6), (48, 4, 6), (48, 4, 1)]
# n_time,train_subset, valid_subset
loader_array = [(48, 5, 1), (48, 5, 5)]

# dtp_rank, dtp_size , dp_rank , dp_size
distributed_array = [
    (0, 4, 0, 1),
    (1, 4, 0, 1),
    (0, 4, 0, 8),
    (1, 4, 0, 8),
    (0, 8, 1, 8),
    (1, 8, 1, 8),
]


class TestDataLoader:
    @pytest.mark.parametrize(
        "n_time, n_features, n_lat, n_lon, n_masks",
        data_array,
    )
    def test_dataloader_dummy(
        self,
        tmp_path_factory: TempPathFactory,
        n_time: int,
        n_features: int,
        n_lat: int,
        n_lon: int,
        n_masks: int,
    ) -> None:
        """Test for function get_dataloader, for different input data sizes when no dataset is availalbe, aka the DummyDataset is loaded."""
        config_dict = create_mock_config()
        dtp_rank = 0
        dtp_size = 4
        dp_rank = 0
        dp_size = 1
        time_coords = pd.date_range(
            "2024-01-01", periods=n_time + 1, freq="h"
        ).strftime("%Y-%m-%d")
        config_dict["data"]["train_time_range"] = [time_coords[0], time_coords[-1]]

        dummy_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="train",
            dtype=torch.float32,
        )

        assert len(dummy_loader) == int(n_time - config_dict["data"]["dt"]) // (
            config_dict["training"]["batch_size"] * dp_size
        )

    @pytest.mark.parametrize(
        "n_time, n_features, n_lat, n_lon, n_masks",
        data_array,
    )
    def test_dataloader_era5(
        self,
        tmp_path_factory: TempPathFactory,
        n_time: int,
        n_features: int,
        n_lat: int,
        n_lon: int,
        n_masks: int,
    ) -> None:
        """Test for function get_dataloader, for different input data sizes when a zarr dataset is availalbe."""
        config_dict = create_mock_config()
        dtp_rank = 0
        dtp_size = 4
        dp_rank = 0
        dp_size = 1
        config_dict["data"]["train_time_range"] = ["2024-01-01", "2024-01-06"]

        data_path = tmp_path_factory.mktemp("data")
        zarr_path = generate_mock_zarr_data(
            (data_path / "test_data.zarr"), n_time, n_features, n_lat, n_lon
        )

        mask_list = generate_mock_masks(data_path, n_lat, n_lon, n_masks)
        generate_mock_normalization(data_path, n_features)

        config_dict["data"]["train_data_path"] = zarr_path
        config_dict["data"]["valid_data_path"] = zarr_path
        config_dict["data"]["constant_masks_path"] = data_path
        config_dict["data"]["constant_masks"] = mask_list
        config_dict["data"]["normalization_dir"] = data_path

        config_dict["data"]["xlat"] = n_lat
        config_dict["data"]["xlon"] = n_lon
        config_dict["data"]["n_variables"] = n_features
        train_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="train",
            dtype=torch.float32,
        )
        assert len(train_loader) == int(n_time - config_dict["data"]["dt"]) // (
            config_dict["training"]["batch_size"] * dp_size
        )

        config_dict["data"]["n_in_timesteps"] = 2
        train_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="train",
            dtype=torch.float32,
        )
        assert len(train_loader) == int(n_time - 2 * config_dict["data"]["dt"]) // (
            config_dict["training"]["batch_size"] * dp_size
        )

        config_dict["data"]["n_out_timesteps"] = 3
        train_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="train",
            dtype=torch.float32,
        )
        assert len(train_loader) == int(n_time - 4 * config_dict["data"]["dt"]) // (
            config_dict["training"]["batch_size"] * dp_size
        )

    @pytest.mark.parametrize(
        "n_time, batch_size, dt",
        train_array,
    )
    def test_dataloader_train(
        self, tmp_path_factory: TempPathFactory, n_time: int, batch_size: int, dt: int
    ) -> None:
        """Test for function get_dataloader (with existing zarr dataset) for different batch size and temporal subsampling."""
        config_dict = create_mock_config()
        dtp_rank = 0
        dtp_size = 4
        dp_rank = 0
        dp_size = 1
        n_features = 8
        n_lat = 72
        n_lon = 144
        n_masks = 3

        config_dict["data"]["train_time_range"] = ["2024-01-01", "2024-01-05"]
        config_dict["data"]["valid_time_range"] = ["2024-01-05", "2024-01-06"]

        config_dict["data"]["xlat"] = n_lat
        config_dict["data"]["xlon"] = n_lon
        config_dict["data"]["n_variables"] = n_features

        config_dict["data"]["dt"] = dt
        config_dict["training"]["batch_size"] = batch_size

        data_path = tmp_path_factory.mktemp("data")
        zarr_path = generate_mock_zarr_data(
            (data_path / "test_data.zarr"), n_time, n_features, n_lat, n_lon
        )

        mask_list = generate_mock_masks(data_path, n_lat, n_lon, n_masks)
        generate_mock_normalization(data_path, n_features)

        config_dict["data"]["train_data_path"] = zarr_path
        config_dict["data"]["valid_data_path"] = zarr_path
        config_dict["data"]["constant_masks_path"] = data_path
        config_dict["data"]["constant_masks"] = mask_list
        config_dict["data"]["normalization_dir"] = data_path

        train_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="train",
            dtype=torch.float32,
        )
        assert len(train_loader) == int(n_time - config_dict["data"]["dt"]) // (
            config_dict["training"]["batch_size"] * dp_size
        )

    @pytest.mark.parametrize(
        "n_time, train_subset, valid_subset",
        loader_array,
    )
    def test_dataloader_loader(
        self,
        tmp_path_factory: TempPathFactory,
        n_time: int,
        train_subset: int,
        valid_subset: int,
    ) -> None:
        """Test for function get_dataloader (with existing zarr dataset) for different training and validation subset sizes."""
        config_dict = create_mock_config()
        dtp_rank = 0
        dtp_size = 4
        dp_rank = 0
        dp_size = 1
        n_features = 8
        n_lat = 72
        n_lon = 144
        n_masks = 3

        config_dict["data"]["train_time_range"] = ["2024-01-01", "2024-01-05"]
        config_dict["data"]["valid_time_range"] = ["2024-01-05", "2024-01-06"]
        config_dict["data"]["train_subset"] = train_subset
        config_dict["data"]["valid_subset"] = valid_subset

        config_dict["data"]["xlat"] = n_lat
        config_dict["data"]["xlon"] = n_lon
        config_dict["data"]["n_variables"] = n_features

        data_path = tmp_path_factory.mktemp("data")
        zarr_path = generate_mock_zarr_data(
            (data_path / "test_data.zarr"), n_time, n_features, n_lat, n_lon
        )

        mask_list = generate_mock_masks(data_path, n_lat, n_lon, n_masks)
        generate_mock_normalization(data_path, n_features)

        config_dict["data"]["train_data_path"] = zarr_path
        config_dict["data"]["valid_data_path"] = zarr_path
        config_dict["data"]["constant_masks_path"] = data_path
        config_dict["data"]["constant_masks"] = mask_list
        config_dict["data"]["normalization_dir"] = data_path

        train_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="train",
            dtype=torch.float32,
        )
        assert len(train_loader) == train_subset

        valid_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="validation",
            dtype=torch.float32,
        )
        assert len(valid_loader) == valid_subset

    @pytest.mark.parametrize(
        "dtp_rank, dtp_size , dp_rank , dp_size",
        distributed_array,
    )
    def test_dataloader_distributed(
        self,
        tmp_path_factory: TempPathFactory,
        dtp_rank: int,
        dtp_size: int,
        dp_rank: int,
        dp_size: int,
    ) -> None:
        """Test for function get_dataloader (with existing zarr dataset) for different dtp and dp ranks and group sizes."""
        config_dict = create_mock_config()

        n_features = 8
        n_lat = 72
        n_lon = 144
        n_masks = 3
        n_time = 48

        config_dict["data"]["train_time_range"] = ["2024-01-01", "2024-01-06"]

        config_dict["data"]["xlat"] = n_lat
        config_dict["data"]["xlon"] = n_lon
        config_dict["data"]["n_variables"] = n_features

        data_path = tmp_path_factory.mktemp("data")
        zarr_path = generate_mock_zarr_data(
            (data_path / "test_data.zarr"), n_time, n_features, n_lat, n_lon
        )

        mask_list = generate_mock_masks(data_path, n_lat, n_lon, n_masks)
        generate_mock_normalization(data_path, n_features)

        config_dict["data"]["train_data_path"] = zarr_path
        config_dict["data"]["valid_data_path"] = zarr_path
        config_dict["data"]["constant_masks_path"] = data_path
        config_dict["data"]["constant_masks"] = mask_list
        config_dict["data"]["normalization_dir"] = data_path

        train_loader = get_dataloader(
            config_dict,
            dtp_rank,
            dtp_size,
            dp_rank,
            dp_size,
            mode="train",
            dtype=torch.float32,
        )
        assert len(train_loader) == math.ceil(
            (n_time - config_dict["data"]["dt"])
            / (config_dict["training"]["batch_size"] * dp_size)
        )

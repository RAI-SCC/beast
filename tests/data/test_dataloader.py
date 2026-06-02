import pandas as pd

from test_dataset import generate_mock_zarr_data, generate_mock_masks, generate_mock_normalization
from beast.data.dataloader import get_dataloader
def create_mock_config():
    """Create a mock dataset and corresponding mock config dictionary."""
    config = {
        "data": {
            "dt": 6
            "train_data_path": "test_data.zarr" 
            "valid_data_path": "test_data.zarr"
            "constant_masks_path": "test_masks"
            "constant_masks" : ["mask_0001.npy", "mask_0002.npy", "mask_0003.npy"]
            "normalization_dir": "normalization"
            "train_subset": -1
            "valid_subset": -1
            "train_time_range": ["2000-01-01", "2000-12-31"]
            "valid_time_range": ["2001-01-01", "2001-12-31"]
            "xlat": 721
            "xlon": 1440
            "n_variables": 84
            "n_in_timesteps": 1
            "n_out_timesteps": 1 
        },
        "training": {
            "batch_size": 1
            "dtype": float32
         },
    }

    return config

class TestDataLoader:
    @pytest.mark.parametrize("n_time, n_features, n_lat, n_lon, m_masks", [(240,84,721,1440,3), (240,84,120,240,3)])
    def test_dataloader(self, tmp_path_factory, n_time, n_features, n_lat, n_lon, m_masks):
        config_dict = create_mock_config()
        dtp_rank = 0
        dtp_size = 4
        dp_rank = 0
        dp_size = 16
        time_coords = pd.date_range("2024-01-01", periods=n_time, freq="h")
        
        dummy_loader = get_dataloader(config_dict, dtp_rank, dtp_size, dp_rank, dp_size, mode="train", dtype=torch.float32)
        assert len(dummy_loader) = n_time - 1

        data_path = tmp_path_factory.mktemp("data") 
        zarr_path = generate_mock_zarr_data((data_path/ "test_data.zarr"), n_time,n_features,n_lat,n_lon)
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
        config_dict["data"]["train_time_range"] = [time_coords[0], time_coords[-1]]

        train_loader_1 = get_dataloader(config_dict, dtp_rank, dtp_size, dp_rank, dp_size, mode="train", dtype=torch.float32)
        assert len(train_loader) == n_time - 1

        config_dict["data"]["n_in_timesteps"] = 2
        train_loader_1 = get_dataloader(config_dict, dtp_rank, dtp_size, dp_rank, dp_size, mode="train", dtype=torch.float32)
        assert len(train_loader) == n_time - 2

        config_dict["data"]["n_out_timesteps"] = 2
        train_loader_1 = get_dataloader(config_dict, dtp_rank, dtp_size, dp_rank, dp_size, mode="train", dtype=torch.float32)
        assert len(train_loader) == n_time - 3
     
    






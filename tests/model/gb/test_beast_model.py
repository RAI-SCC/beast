import pytest
import torch
from torch_blue.vi import convert_to_vimodule

from beast.comm import create_process_mesh_dict
from beast.model.gb.beast_config import BEAST_CONFIGURATIONS
from beast.model.gb.beast_model import Beast


class TestBeastlingModel:
    @pytest.mark.parametrize("n_timesteps", [1, 2])
    @pytest.mark.parametrize("bayesian", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_beastling(
        self, n_timesteps: int, bayesian: bool, batch_size: int, device: torch.device
    ) -> None:
        small_config = BEAST_CONFIGURATIONS["BEASTLING"]["config"].copy()
        small_config["bayesian"] = bayesian
        small_config["in_channels"] = small_config["in_channels"] * n_timesteps
        model = Beast(**small_config, dropout=0 if bayesian else 1e-1, device=device)

        # Test a forward pass with dummy data
        with torch.no_grad():
            x = torch.rand(
                batch_size,
                small_config["in_channels"],
                small_config["height"],
                small_config["width"],
                device=device,
            )
            if bayesian:
                convert_to_vimodule(model, return_log_probs=True)
                y = model(x, samples=2)
                assert y.shape == (
                    2,
                    batch_size,
                    small_config["out_channels"],
                    small_config["height"],
                    small_config["width"],
                )
            else:
                y = model(x)
                assert y.shape == (
                    batch_size,
                    small_config["out_channels"],
                    small_config["height"],
                    small_config["width"],
                )


@pytest.mark.gpu
class TestDistBeastlingModel:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_timesteps", [1, 2])
    @pytest.mark.parametrize("bayesian", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_4way_beastling(
        self, n_timesteps: int, bayesian: bool, batch_size: int, device: torch.device
    ) -> None:

        mesh_dict = create_process_mesh_dict([1, 1, 1, 2, 2], device_type=str(device))
        jchannel_pg = mesh_dict["JChannel"]
        jspatial_pg = mesh_dict["JSpatial"]
        dtp_pg = mesh_dict["DTP"]

        small_config = BEAST_CONFIGURATIONS["BEASTLING"]["config"].copy()
        small_config["bayesian"] = bayesian
        small_config["in_channels"] = small_config["in_channels"] * n_timesteps
        model = Beast(
            **small_config,
            dropout=0 if bayesian else 1e-1,
            dtp_group=dtp_pg,
            channel_group=jchannel_pg,
            spatial_group=jspatial_pg,
            device=device,
        )

        # Test a forward pass with dummy data
        with torch.no_grad():
            x = torch.rand(
                batch_size,
                small_config["in_channels"],
                small_config["height"],
                small_config["width"],
                device=device,
            )
            if bayesian:
                convert_to_vimodule(model, return_log_probs=True)
                y = model(x, samples=2)
                assert y.shape == (
                    2,
                    batch_size,
                    small_config["out_channels"],
                    small_config["height"],
                    small_config["width"],
                )
            else:
                y = model(x)
                assert y.shape == (
                    batch_size,
                    small_config["out_channels"],
                    small_config["height"],
                    small_config["width"],
                )

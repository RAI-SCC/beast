import torch

from beast.model.gb.beast_config import BEAST_CONFIGURATIONS
from beast.model.gb.beast_model import Beast


class TestBeastModel:
    def test_det_beastling_from_config(self, device) -> None:
        small_config = BEAST_CONFIGURATIONS["BEASTLING"]
        model = Beast(**small_config, device=device)

        # Test a forward pass with dummy data
        X = torch.rand(
            1,
            small_config["in_channels"],
            small_config["height"],
            small_config["width"],
            device=device,
        )
        Y = model(X)
        assert Y.shape == (
            1,
            small_config["out_channels"],
            small_config["height"],
            small_config["width"],
        )

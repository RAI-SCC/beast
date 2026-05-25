"""Tests for beast.model.model_factory module."""

import pytest
import torch

from beast.model.gb.beast_model import Beast
from beast.model.model_configs import CONFIGS
from beast.model.model_factory import get_model


class TestImportModelFactory:
    def test_import_model_factory(self) -> None:
        from beast import get_model

        assert get_model is not None
        assert callable(get_model)


class TestModelFactory:
    """Test suite for the get_model factory function."""

    def test_get_model_returns_torch_module(self) -> None:
        """Test that get_model returns a torch.nn.Module instance."""
        model = get_model("BEASTLING", device=torch.device("cpu"))
        assert isinstance(model, torch.nn.Module)

    def test_get_model_invalid_model_id(self) -> None:
        """Test that get_model raises ValueError for invalid model ID."""
        with pytest.raises(ValueError, match="Model ID 'INVALID-MODEL' not found"):
            get_model("INVALID-MODEL", device=torch.device("cpu"))

    def test_get_model_with_kwargs_override(self) -> None:
        """Test that get_model overrides config with kwargs."""
        model = get_model("BEASTLING", height=512, device=torch.device("cpu"))

        # Check that the model was created (though we can't directly inspect config)
        assert isinstance(model, Beast)
        assert model.height == 512

    def test_get_model_with_multiple_kwargs(self) -> None:
        """Test that get_model handles multiple kwargs overrides."""
        model = get_model(
            "BEASTLING",
            height=512,
            width=1024,
            embed_dim=512,
            device=torch.device("cpu"),
        )
        assert isinstance(model, Beast)

    def test_get_model_doesnt_modify_configs(self) -> None:
        """Test that get_model doesn't modify the original CONFIGS."""
        # Get the original config
        original_config = CONFIGS["BEASTLING"]["config"].copy()

        # Create a model with overridden kwargs
        get_model(
            "BEASTLING",
            height=512,
            width=1024,
            device=torch.device("cpu"),
        )

        # Check that CONFIGS wasn't modified
        assert CONFIGS["BEASTLING"]["config"] == original_config, (
            "CONFIGS was modified during get_model call"
        )

"""Tests for beast.model.model_configs module."""

from beast.model.beast_config import BEAST_CONFIGURATIONS
from beast.model.model_configs import CONFIGS


class TestModelConfigs:
    """Test suite for model configuration management."""

    def test_configs_not_empty(self) -> None:
        """Test that CONFIGS dictionary is not empty."""
        assert len(CONFIGS) > 0

    def test_configs_has_beast_configurations(self) -> None:
        """Test that CONFIGS includes all BEAST_CONFIGURATIONS."""
        for model_id in BEAST_CONFIGURATIONS:
            assert model_id in CONFIGS

    def test_configs_contains_all_required_keys(self) -> None:
        """Test that each model config has required keys."""
        required_keys = {"class", "config"}
        for model_id, config in CONFIGS.items():
            assert isinstance(config, dict), f"Config for {model_id} is not a dict"
            assert required_keys.issubset(config.keys()), (
                f"Config for {model_id} missing required keys"
            )

    def test_configs_class_is_callable(self) -> None:
        """Test that the class field in each config is callable."""
        for model_id, config in CONFIGS.items():
            assert callable(config["class"]), f"Class for {model_id} is not callable"

    def test_configs_config_is_dict(self) -> None:
        """Test that the config field in each entry is a dictionary."""
        for model_id, config in CONFIGS.items():
            assert isinstance(config["config"], dict), (
                f"Config for {model_id} is not a dict"
            )

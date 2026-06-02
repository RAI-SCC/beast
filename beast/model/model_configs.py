"""Model configuration management.

This module aggregates model configurations from various model implementations.
"""

from typing import Any

from beast.model.gb import BEAST_CONFIGURATIONS

CONFIGS: dict[str, dict[str, Any]] = {}
CONFIGS.update(BEAST_CONFIGURATIONS)

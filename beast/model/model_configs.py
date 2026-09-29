"""Registry of model configurations available to :func:`~beast.model.get_model`."""

from typing import Any

from beast.model.gb import BEAST_CONFIGURATIONS

CONFIGS: dict[str, dict[str, Any]] = {}
CONFIGS.update(BEAST_CONFIGURATIONS)

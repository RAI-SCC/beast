"""Registry of model configurations available to :func:`~beast.model.get_model`."""

from typing import Any

from beast.model.beast_config import BEAST_CONFIGURATIONS

CONFIGS: dict[str, dict[str, Any]] = {}
"""All model configurations that :func:`~beast.model.get_model` can build, by name.

Holds the entries of
:data:`~beast.model.beast_config.BEAST_CONFIGURATIONS`.
"""
CONFIGS.update(BEAST_CONFIGURATIONS)

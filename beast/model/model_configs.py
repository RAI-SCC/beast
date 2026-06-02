from typing import Any

from .beast.beast_config import BEAST_MODEL_CONFIGS

CONFIGS: dict[str, dict[str, Any]] = {}
CONFIGS.update(BEAST_MODEL_CONFIGS)

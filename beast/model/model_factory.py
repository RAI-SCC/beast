"""Factory for creating model instances based on configuration."""

from typing import Any

import torch

from beast.model.model_configs import CONFIGS


def get_model(model_id: str, **kwargs: Any) -> torch.nn.Module:
    """Instantiate a registered model configuration.

    Parameters
    ----------
    model_id : str
        Key in `~beast.model.model_configs.CONFIGS`, e.g. a named configuration
        from `~beast.model.beast_config`.
    **kwargs
        Overrides for the registered configuration's constructor arguments.

    Returns
    -------
    torch.nn.Module
        The model, built from the registered configuration updated with
        `kwargs`.

    Raises
    ------
    ValueError
        If `model_id` is not in `~beast.model.model_configs.CONFIGS`.
    """
    if model_id not in CONFIGS:
        raise ValueError(f"Model ID '{model_id}' not found in CONFIGS.")

    model_config = CONFIGS[model_id]
    model_class = model_config["class"]
    model_config = model_config[
        "config"
    ].copy()  # Start with default config for the model

    # Update the default kwargs with any provided kwargs
    model_config.update(kwargs)

    return model_class(**model_config)  # type: ignore

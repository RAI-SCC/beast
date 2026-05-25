"""Factory for creating model instances based on configuration."""

import torch

from beast.model.model_configs import CONFIGS


def get_model(model_id: str, **kwargs: dict) -> torch.nn.Module:
    """
    Create a model based on a string id and a configuration dictionary. The string id can be used to specify a particular model architecture, while the configuration dictionary can contain hyperparameters and other settings for the model.

    Parameters
    ----------
    model_id : str
        A string identifier for the model architecture to create. This should correspond to a key in the CONFIGS dictionary.
    kwargs : dict
        A dictionary of keyword arguments to pass to the model constructor. This can include hyperparameters and other settings for the model.

    Returns
    -------
    torch.nn.Module
        An instance of the specified model architecture, initialized with the provided configuration.

    Raises
    ------
    ValueError
        If the provided model_id does not correspond to any entry in the CONFIGS dictionary.
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

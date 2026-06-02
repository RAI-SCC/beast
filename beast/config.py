"""Configuration management for BEAST model.

This module provides utilities to load, validate, and modify model
configuration from YAML files and command-line arguments.
"""

import argparse
import os
from typing import Any

import torch
import yaml


def setup_config_dict(
    config_dict: dict[str, Any], args: argparse.Namespace, resume: bool = False
) -> dict[str, Any]:
    """Clean configuration file and adjust based on environment variables and overwritten options."""
    config_dict["timings"] = args.timings
    config_dict["skip_validation"] = args.skip_validation
    config_dict["profile"] = args.profile
    config_dict["resume_path"] = args.resume
    config_dict["transplant_from"] = args.transplant

    if not resume:
        config_dict = overwrite_config(config_dict, args.configargs)

        ### Data Stuff ###
        workdir = os.getenv("WORKDIR")

        if os.getenv("DATA_DIR") is not None:
            data_dir = os.getenv("DATA_DIR")
            config_dict["data"]["train_data_path"] = (
                data_dir + config_dict["data"]["train_data_path"]
            )
            config_dict["data"]["valid_data_path"] = (
                data_dir + config_dict["data"]["valid_data_path"]
            )
            config_dict["data"]["constant_masks_path"] = (
                data_dir + config_dict["data"]["constant_masks_path"]
            )

        if "normalization_dir" in config_dict["data"] and workdir is not None:
            config_dict["data"]["normalization_dir"] = (
                workdir + "/" + config_dict["data"]["normalization_dir"]
            )

    world_size = int(os.getenv("SLURM_NTASKS", 1))
    mesh_dims = config_dict["mesh_dims"]
    if -1 in config_dict["mesh_dims"]:
        index = mesh_dims.index(-1)
        config_dict["mesh_dims"][index] = (
            world_size // -torch.prod(torch.tensor(mesh_dims)).item()
        )

    # default case: load only one timestep
    if config_dict["data"]["n_in_timesteps"] not in [1, 2]:
        config_dict["data"]["n_in_timesteps"] = 1

    # Only for GrandBellbeast: append n_masks for odd number of masks (the dataset does this, too)
    n_masks = len(config_dict["data"]["constant_masks"])
    if config_dict["jigsaw"]["parallelism"] > 1:
        if n_masks % 2 == 1:
            n_masks += 1

    n_input_vars = (
        config_dict["data"]["n_in_timesteps"] * config_dict["data"]["n_variables"]
    )

    # we'll set this automatically, to avoid confusion when defining the config,
    # as n_input_channels depends on how many timesteps we use as concatenated input
    config_dict["model"]["n_input_channels"] = n_input_vars + n_masks

    return config_dict


def overwrite_config(
    config_dict: dict[str, Any], configargs: list[str]
) -> dict[str, Any]:
    """Change configuration values based on cmd line arguments ('<section>_<key>=<value>')."""
    for i in range(len(configargs)):
        section_key, new_value = configargs[i].split("=")

        # Special case for mesh dim, as it is not part of a section (though it should be parallelism)
        if section_key == "mesh_dims":
            config_dict["mesh_dims"] = [int(x) for x in new_value.split(",")]
            continue

        section, key = section_key.split("_", maxsplit=1)
        if section not in config_dict:
            print(
                f"Section for keyword argument {configargs[i]} does not exist in the configuration file"
            )
            continue
        elif key not in config_dict[section]:
            print(
                f"Configuration option {key} does not exist in the configuration file"
            )
            continue

        original_value = config_dict[section][key]
        if isinstance(original_value, str):
            config_dict[section][key] = new_value
        elif isinstance(original_value, bool):
            config_dict[section][key] = new_value == "True"
        elif isinstance(original_value, int):
            config_dict[section][key] = int(new_value)
        elif isinstance(original_value, list):
            formated_new_value: list[str] | list[int]
            if isinstance(original_value[0], str):
                formated_new_value = [x for x in new_value.split(",")]
            elif isinstance(original_value[0], int):
                formated_new_value = [int(x) for x in new_value.split(",")]
            else:
                formated_new_value = []  # type: ignore
            config_dict[section][key] = formated_new_value

    return config_dict


def load_configuration_yaml(directory: str) -> dict[str, Any]:
    """Load configuration from YAML file in directory.

    Parameters
    ----------
    directory : str
        Directory containing config.yaml file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    yaml_path = os.path.join(directory, "config.yaml")
    with open(f"{yaml_path}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)  # type: ignore

    return config_dict  # type: ignore

import argparse
import os

import torch


def setup_config_dict(
    config_dict: dict[str, dict[str, any]], args: argparse.Namespace, resume=False
) -> dict[str, dict[str, any]]:
    """
    Clean the configuration file, adjust based on environmental variables and overwritten options.
    """
    config_dict["timings"] = args.timings
    config_dict["skip_validation"] = args.skip_validation
    config_dict["profile"] = args.profile
    config_dict["resume_path"] = args.resume

    if not resume:
        config_dict = overwrite_config(config_dict, args.configargs)

        ### Data Stuff ###
        WORKDIR = os.getenv("WORKDIR")

        if os.getenv("DATA_DIR") is not None:
            config_dict["data"]["train_data_path"] = (
                os.getenv("DATA_DIR") + config_dict["data"]["train_data_path"]
            )
            config_dict["data"]["valid_data_path"] = (
                os.getenv("DATA_DIR") + config_dict["data"]["valid_data_path"]
            )
            config_dict["data"]["constant_masks_path"] = (
                os.getenv("DATA_DIR") + config_dict["data"]["constant_masks_path"]
            )

        if "normalization_dir" in config_dict["data"]:
            config_dict["data"]["normalization_dir"] = (
                WORKDIR + "/" + config_dict["data"]["normalization_dir"]
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
    if config_dict["model_parallel"]["parallelism"] > 1:
        if n_masks % 2 == 1:
            n_masks += 1

    n_input_vars = (
        config_dict["data"]["n_in_timesteps"] * config_dict["data"]["n_variables"]
    )

    # we'll set this automatically, to avoid confusion when defining the config,
    # as n_input_channels depends on how many timesteps we use as concatenated input
    config_dict["model"]["n_input_channels"] = n_input_vars + n_masks
    config_dict["model"]["embedding_dimension"] = (
        config_dict["model"]["embedding_dimension"]
        * config_dict["data"]["n_in_timesteps"]
    )

    return config_dict


def overwrite_config(
    config_dict: dict[str, dict[str, any]], configargs: list[str]
) -> dict[str, dict[str, any]]:
    """
    Change configuration file values based on cmd line arguments passed as '<section>_<key>=<value>' after all the optional flags. Arrays are passed with the elements connected by comas.
    """
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
            if isinstance(original_value[0], str):
                formated_new_value = [x for x in new_value.split(",")]
            elif isinstance(original_value[0], int):
                formated_new_value = [int(x) for x in new_value.split(",")]
            config_dict[section][key] = formated_new_value

    return config_dict

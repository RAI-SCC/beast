"""Util functions for the evaluation."""

from typing import Any, Sequence

import torch

## TODO: This is bad - it should be defined centrally and checked at initialization to ensure the config is valid.
## Not sure what this cleanest way to implement this is - I have seen different variants.

DATA_KEY = "data"
"""Key to access the data section in the configuration file."""
EVALUATION_KEY = "evaluation"
"""Key to access the evaluation section in the configuration file."""
SURFACE_VARIABLES_KEY = "surface_variables"
"""Key to access the evaluation surface variables in the configuration file."""
PRESSURE_VARIABLES_KEY = "pressure_variables"
"""Key to access the evaluation pressure variables in the configuration file."""
PRESSURE_LEVELS_KEY = "pressure_levels"
"""Key to access the evaluation pressure levels in the configuration file."""


def select_variables(
    t: torch.Tensor | Sequence[torch.Tensor],
    config: dict[str, Any],
    channel_rank: int,
    channel_group_size: int,
) -> torch.Tensor | Sequence[torch.Tensor]:
    """
    Select variables required for the evalution.

    This function slices either a single tensor or a sequence of tensors to only return a subset of variables required
    for the evaluation.

    Parameters
    ----------
    t: torch.Tensor | Sequence[torch.Tensor]
        A single tensor of sequence of tensors from which certain variables should be selected.
    config: dict[str, Any]
        The configuration dictionary containing information on which variables to be selected.
    channel_rank: int
        The rank of the channel group.
    channel_group_size: int
        The size of the channel group.

    Returns
    -------
    torch.Tensor | Sequence[torch.Tensor]
        A single tensor or a sequence of tensors which only contain the selected variables.
    """
    assert isinstance(t, (torch.Tensor, Sequence)), (
        f"Expected `torch.Tensor` or `Sequence`, got type {type(t)}"
    )

    try:
        data_config = config[DATA_KEY]
        all_surface_variables = data_config[SURFACE_VARIABLES_KEY]
        all_pressure_variables = data_config[PRESSURE_VARIABLES_KEY]
        all_pressure_levels = data_config[PRESSURE_LEVELS_KEY]

    except KeyError as e:
        raise KeyError(
            f"Configuration error: The required key {e} is missing!"
        ) from None

    number_of_variables = len(all_surface_variables) + len(
        all_pressure_variables
    ) * len(all_pressure_levels)

    try:
        eval_config = config[EVALUATION_KEY]

    except KeyError:
        raise KeyError(
            f"Configuration error: The evaluation section {EVALUATION_KEY} is missing!"
        ) from None

    # Define variables used for evaluation.
    eval_surface_variables = eval_config.get(
        SURFACE_VARIABLES_KEY, all_surface_variables
    )
    eval_pressure_variables = eval_config.get(
        PRESSURE_VARIABLES_KEY, all_pressure_variables
    )
    eval_pressure_levels = eval_config.get(PRESSURE_LEVELS_KEY, all_pressure_levels)

    if channel_group_size > 1:
        row = channel_rank
        local_start = row * (number_of_variables // 2)
        local_end = (row + 1) * (number_of_variables // 2)
    else:
        local_start = 0
        local_end = number_of_variables

    indices = []

    # Extract indices for surface variables.
    for s in eval_surface_variables:
        if s in all_surface_variables:
            g_idx = all_surface_variables.index(s)
            if local_start <= g_idx < local_end:
                indices.append(g_idx - local_start)

    # Extract indices for pressure variables.
    number_total_surface_variables = len(all_surface_variables)
    number_pressure_levels = len(all_pressure_levels)
    for pressure_variable in eval_pressure_variables:
        if pressure_variable in all_pressure_variables:
            pressure_variable_idx = all_pressure_variables.index(pressure_variable)
            for level in eval_pressure_levels:
                if level in all_pressure_levels:
                    l_idx = all_pressure_levels.index(level)
                    g_idx = (
                        number_total_surface_variables
                        + (pressure_variable_idx * number_pressure_levels)
                        + l_idx
                    )

                    if local_start <= g_idx < local_end:
                        indices.append(g_idx - local_start)

    if isinstance(t, torch.Tensor):
        return t[:, indices]

    return [item[:, indices] for item in t]

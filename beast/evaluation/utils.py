"""Util functions for the evaluation."""

from typing import Any, Sequence

import torch

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
    *,
    config: dict[str, Any] | None = None,
    channel_rank: int,
    channel_group_size: int,
    all_surface_variables: list[str] | None = None,
    all_pressure_variables: list[str] | None = None,
    all_pressure_levels: list | None = None,
    eval_surface_variables: list[str] | None = None,
    eval_pressure_variables: list[str] | None = None,
    eval_pressure_levels: list | None = None,
    dim: int = 1,
) -> torch.Tensor | Sequence[torch.Tensor]:
    """
    Select variables required for the evaluation.

    This function slices either a single tensor or a sequence of tensors to only return a subset of variables required
    for the evaluation.

    Variables and levels can be supplied directly as keyword arguments. If any of the ``all_*`` or ``eval_*``
    arguments are ``None``, the function falls back to reading the corresponding values from ``config`` using the
    module-level key constants. At least one of the two sources must provide each required value.

    Parameters
    ----------
    t: torch.Tensor | Sequence[torch.Tensor]
        A single tensor or sequence of tensors from which certain variables should be selected.
    config: dict[str, Any] | None, optional
        The configuration dictionary used as a fallback when explicit variable/level arguments are not provided.
    channel_rank: int
        The rank of the channel group.
    channel_group_size: int
        The size of the channel group.
    all_surface_variables: list[str] | None, optional
        All surface variables present in the data. Falls back to ``config`` if ``None``.
    all_pressure_variables: list[str] | None, optional
        All pressure variables present in the data. Falls back to ``None``.
    all_pressure_levels: list | None, optional
        All pressure levels present in the data. Falls back to ``config`` if ``None``.
    eval_surface_variables: list[str] | None, optional
        Surface variables to select for evaluation. Defaults to all surface variables if ``None``.
    eval_pressure_variables: list[str] | None, optional
        Pressure variables to select for evaluation. Defaults to all pressure variables if ``None``.
    eval_pressure_levels: list | None, optional
        Pressure levels to select for evaluation. Defaults to all pressure levels if ``None``.
    dim: int, optional
        The dimension along which to index into the tensor(s). Defaults to ``1``.

    Returns
    -------
    torch.Tensor | Sequence[torch.Tensor]
        A single tensor or a sequence of tensors which only contain the selected variables.

    Raises
    ------
    KeyError
        If a required variable/level list is not provided directly and cannot be found in ``config``.
    AssertionError
        If ``t`` is not a ``torch.Tensor`` or ``Sequence``.
    """
    assert isinstance(t, (torch.Tensor, Sequence)), (
        f"Expected `torch.Tensor` or `Sequence`, got type {type(t)}"
    )

    data_config: dict | None = config.get(DATA_KEY) if config is not None else None
    eval_config: dict | None = (
        config.get(EVALUATION_KEY) if config is not None else None
    )

    # Resolve keyword arguments for all variables and levels.
    if all_surface_variables is None:
        if data_config is None or SURFACE_VARIABLES_KEY not in data_config:
            raise KeyError(
                f"Configuration error: 'all_surface_variables' was not provided as an argument "
                f"and the required key '{SURFACE_VARIABLES_KEY}' is missing from the configuration!"
            )
        all_surface_variables = data_config[SURFACE_VARIABLES_KEY]

    if all_pressure_variables is None:
        if data_config is None or PRESSURE_VARIABLES_KEY not in data_config:
            raise KeyError(
                f"Configuration error: 'all_pressure_variables' was not provided as an argument "
                f"and the required key '{PRESSURE_VARIABLES_KEY}' is missing from the configuration!"
            )
        all_pressure_variables = data_config[PRESSURE_VARIABLES_KEY]

    if all_pressure_levels is None:
        if data_config is None or PRESSURE_LEVELS_KEY not in data_config:
            raise KeyError(
                f"Configuration error: 'all_pressure_levels' was not provided as an argument "
                f"and the required key '{PRESSURE_LEVELS_KEY}' is missing from the configuration!"
            )
        all_pressure_levels = data_config[PRESSURE_LEVELS_KEY]

    number_of_variables = len(all_surface_variables) + len(
        all_pressure_variables
    ) * len(all_pressure_levels)

    # Resolve keyword arguments for evaluation variables and levels.
    if eval_surface_variables is None:
        eval_surface_variables = (
            eval_config.get(SURFACE_VARIABLES_KEY, all_surface_variables)
            if eval_config is not None
            else all_surface_variables
        )

    if eval_pressure_variables is None:
        eval_pressure_variables = (
            eval_config.get(PRESSURE_VARIABLES_KEY, all_pressure_variables)
            if eval_config is not None
            else all_pressure_variables
        )

    if eval_pressure_levels is None:
        eval_pressure_levels = (
            eval_config.get(PRESSURE_LEVELS_KEY, all_pressure_levels)
            if eval_config is not None
            else all_pressure_levels
        )

    if channel_group_size > 1:
        shard_size = number_of_variables // channel_group_size
        local_start = channel_rank * shard_size
        local_end = (channel_rank + 1) * shard_size
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

    idx_tensor = torch.tensor(indices, dtype=torch.long)

    if isinstance(t, torch.Tensor):
        return t.index_select(dim, idx_tensor)

    return [item.index_select(dim, idx_tensor) for item in t]

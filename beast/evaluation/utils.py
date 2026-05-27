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


def _resolve_variables(
    config: dict[str, Any] | None,
    all_surface_variables: list[str] | None,
    all_pressure_variables: list[str] | None,
    all_pressure_levels: list | None,
    eval_surface_variables: list[str] | None,
    eval_pressure_variables: list[str] | None,
    eval_pressure_levels: list | None,
) -> tuple[list[str], list[str], list, list[str], list[str], list]:
    """
    Resolve variable and level lists from explicit arguments or a config dict.

    Parameters
    ----------
    config: dict[str, Any] | None
        The configuration dictionary used as a fallback when explicit arguments are not provided.
    all_surface_variables: list[str] | None
        All surface variables present in the data. Falls back to ``config`` if ``None``.
    all_pressure_variables: list[str] | None
        All pressure variables present in the data. Falls back to ``config`` if ``None``.
    all_pressure_levels: list | None
        All pressure levels present in the data. Falls back to ``config`` if ``None``.
    eval_surface_variables: list[str] | None
        Surface variables to use for evaluation. Defaults to all surface variables if ``None``.
    eval_pressure_variables: list[str] | None
        Pressure variables to use for evaluation. Defaults to all pressure variables if ``None``.
    eval_pressure_levels: list | None
        Pressure levels to use for evaluation. Defaults to all pressure levels if ``None``.

    Returns
    -------
    list[str]list[str], list, list[str], list[str], list]
        All surface variables available.
    list[str]
        All pressure variables available.
    list[str]
        All pressure levels available.
    list[str]
        Evaluation surface variables to be considered.
    list[str]
        Evaluation pressure variables to be considered.
    list[str]
        Evaluation pressure levels to be considered.

    Raises
    ------
    KeyError
        If a required list is not provided directly and cannot be found in the configuration.
    """
    data_config: dict | None = config.get(DATA_KEY) if config is not None else None
    eval_config: dict | None = (
        config.get(EVALUATION_KEY) if config is not None else None
    )

    # Resolve lists of all variables.
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

    # Resolve evaluation lists.
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

    return (
        all_surface_variables,
        all_pressure_variables,
        all_pressure_levels,
        eval_surface_variables,
        eval_pressure_variables,
        eval_pressure_levels,
    )


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
    dim: int = -3,
) -> torch.Tensor | Sequence[torch.Tensor]:
    """
    Select variables required for the evaluation.

    This function slices either a single tensor or a sequence of tensors to only return a subset of variables required
    for the evaluation.

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
        All pressure variables present in the data. Falls back to ``config`` if ``None``.
    all_pressure_levels: list | None, optional
        All pressure levels present in the data. Falls back to ``config`` if ``None``.
    eval_surface_variables: list[str] | None, optional
        Surface variables to select for evaluation. Defaults to all surface variables if ``None``.
    eval_pressure_variables: list[str] | None, optional
        Pressure variables to select for evaluation. Defaults to all pressure variables if ``None``.
    eval_pressure_levels: list | None, optional
        Pressure levels to select for evaluation. Defaults to all pressure levels if ``None``.
    dim: int, optional
        The dimension along which to index into the tensor(s). Defaults to ``-3``, which corresponds to the channel
        dimension for both non-Bayesian tensors of shape ``(batch, channels, ...)`` and Bayesian tensors of shape
        ``(batch, samples, channels, ...)``.

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
    (
        all_surface_variables,
        all_pressure_variables,
        all_pressure_levels,
        eval_surface_variables,
        eval_pressure_variables,
        eval_pressure_levels,
    ) = _resolve_variables(
        config,
        all_surface_variables,
        all_pressure_variables,
        all_pressure_levels,
        eval_surface_variables,
        eval_pressure_variables,
        eval_pressure_levels,
    )

    number_of_variables = len(all_surface_variables) + len(
        all_pressure_variables
    ) * len(all_pressure_levels)

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


def make_variable_names(
    *,
    config: dict[str, Any] | None = None,
    all_surface_variables: list[str] | None = None,
    all_pressure_variables: list[str] | None = None,
    all_pressure_levels: list | None = None,
    eval_surface_variables: list[str] | None = None,
    eval_pressure_variables: list[str] | None = None,
    eval_pressure_levels: list | None = None,
) -> list[str]:
    """
    Build a list of human-readable variable name strings for the evaluation variables. Used for plotting.

    Parameters
    ----------
    config: dict[str, Any] | None, optional
        The configuration dictionary used as a fallback when explicit variable/level arguments are not provided.
    all_surface_variables: list[str] | None, optional
        All surface variables present in the data. Falls back to ``config`` if ``None``.
    all_pressure_variables: list[str] | None, optional
        All pressure variables present in the data. Falls back to ``config`` if ``None``.
    all_pressure_levels: list | None, optional
        All pressure levels present in the data. Falls back to ``config`` if ``None``.
    eval_surface_variables: list[str] | None, optional
        Surface variables to include. Defaults to all surface variables if ``None``.
    eval_pressure_variables: list[str] | None, optional
        Pressure variables to include. Defaults to all pressure variables if ``None``.
    eval_pressure_levels: list | None, optional
        Pressure levels to include. Defaults to all pressure levels if ``None``.

    Returns
    -------
    list[str]
        Ordered list of variable name strings corresponding to the selected evaluation variables.

    Raises
    ------
    KeyError
        If a required variable/level list is not provided directly and cannot be found in ``config``.
    """
    (
        all_surface_variables,
        all_pressure_variables,
        all_pressure_levels,
        eval_surface_variables,
        eval_pressure_variables,
        eval_pressure_levels,
    ) = _resolve_variables(
        config,
        all_surface_variables,
        all_pressure_variables,
        all_pressure_levels,
        eval_surface_variables,
        eval_pressure_variables,
        eval_pressure_levels,
    )

    names: list[str] = []

    # Surface variables: include as-is if they exist in the full list.
    for s in eval_surface_variables:
        if s in all_surface_variables:
            names.append(s)

    # Pressure variables: "{variable}_{level}" for each valid (variable, level) combination.
    for pressure_variable in eval_pressure_variables:
        if pressure_variable in all_pressure_variables:
            for level in eval_pressure_levels:
                if level in all_pressure_levels:
                    names.append(f"{pressure_variable}_{level}")

    return names


def latitude_weighted_average(
    data: torch.Tensor, lat_dim: int = -2, lon_dim: int = -1
) -> torch.Tensor:
    """
    Compute the latitude-weighted average of the input data.

    Parameters
    ----------
    data: torch.Tensor
        The input data tensor with a latitude dimension.
    lat_dim: int, optional
        The dimension corresponding to latitude. Defaults to -2.
    lon_dim: int, optional
        The dimension corresponding to longitude. Defaults to -1.

    Returns
    -------
    torch.Tensor
        The latitude-weighted average of the input data.
    """
    # Compute latitude weights based on the cosine of the latitude values.
    latitudes = torch.linspace(
        -torch.pi / 2.0, torch.pi / 2.0, data.shape[lat_dim] + 1, device=data.device
    )[:-1]

    cosines = torch.cos(latitudes)
    cosine_mean = cosines.mean()
    weights = (cosines / cosine_mean).reshape(
        *[
            -1 if i == ((lat_dim + data.dim()) % data.dim()) else 1
            for i in range(data.dim())
        ]
    )  # Normalize weights to have a mean of 1

    # Apply weights and compute the average.
    weighted_data = data * weights

    return weighted_data.mean(dim=(lat_dim, lon_dim))

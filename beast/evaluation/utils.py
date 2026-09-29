"""Variable selection, naming and latitude weighting for the evaluation."""

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
    """Resolve the variable and level lists from arguments or the configuration.

    Explicit arguments take precedence. Missing ``all_*`` lists are read from the
    ``data`` section of ``config`` and missing ``eval_*`` lists from its
    ``evaluation`` section, defaulting to the corresponding ``all_*`` list.

    Parameters
    ----------
    config : dict[str, Any] or None
        Configuration used as a fallback.
    all_surface_variables : list[str] or None
        All surface variables in the data.
    all_pressure_variables : list[str] or None
        All pressure-level variables in the data.
    all_pressure_levels : list or None
        All pressure levels in the data.
    eval_surface_variables : list[str] or None
        Surface variables to evaluate.
    eval_pressure_variables : list[str] or None
        Pressure-level variables to evaluate.
    eval_pressure_levels : list or None
        Pressure levels to evaluate.

    Returns
    -------
    tuple
        The three ``all_*`` lists followed by the three ``eval_*`` lists, in
        the order of the arguments.

    Raises
    ------
    KeyError
        If an ``all_*`` list is neither given nor found in ``config``.
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
    """Select the evaluation variables from the channels of ``t``.

    Channels are ordered surface variables first, then pressure-level variables
    by level. With a sharded channel dimension, only the variables held by
    ``channel_rank`` are selected.

    Parameters
    ----------
    t : torch.Tensor or Sequence[torch.Tensor]
        Tensor, or sequence of tensors, to select from.
    config : dict[str, Any], optional
        Configuration used as a fallback, see :func:`_resolve_variables`.
    channel_rank : int
        Rank in the channel group.
    channel_group_size : int
        Size of the channel group.
    all_surface_variables : list[str], optional
        All surface variables in the data.
    all_pressure_variables : list[str], optional
        All pressure-level variables in the data.
    all_pressure_levels : list, optional
        All pressure levels in the data.
    eval_surface_variables : list[str], optional
        Surface variables to select.
    eval_pressure_variables : list[str], optional
        Pressure-level variables to select.
    eval_pressure_levels : list, optional
        Pressure levels to select.
    dim : int, default: -3
        Channel dimension. The default fits both ``(batch, channels, ...)`` and
        Bayesian ``(batch, samples, channels, ...)`` tensors.

    Returns
    -------
    torch.Tensor or Sequence[torch.Tensor]
        The selected channels, as a tensor for a tensor input and as a list
        otherwise.

    Raises
    ------
    KeyError
        If an ``all_*`` list is neither given nor found in ``config``.
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

    index_device = t.device if isinstance(t, torch.Tensor) else t[0].device
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=index_device)

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
    """Build the names of the evaluation variables, for plotting.

    Surface variables keep their name and pressure-level variables are named
    ``{variable}_{level}``, in the channel order of :func:`select_variables`.

    Parameters
    ----------
    config : dict[str, Any], optional
        Configuration used as a fallback, see :func:`_resolve_variables`.
    all_surface_variables : list[str], optional
        All surface variables in the data.
    all_pressure_variables : list[str], optional
        All pressure-level variables in the data.
    all_pressure_levels : list, optional
        All pressure levels in the data.
    eval_surface_variables : list[str], optional
        Surface variables to include.
    eval_pressure_variables : list[str], optional
        Pressure-level variables to include.
    eval_pressure_levels : list, optional
        Pressure levels to include.

    Returns
    -------
    list[str]
        The variable names.

    Raises
    ------
    KeyError
        If an ``all_*`` list is neither given nor found in ``config``.
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
    """Average over latitude and longitude with cosine-latitude weights.

    Parameters
    ----------
    data : torch.Tensor
        Data with latitude and longitude dimensions.
    lat_dim : int, default: -2
        Latitude dimension.
    lon_dim : int, default: -1
        Longitude dimension.

    Returns
    -------
    torch.Tensor
        The weighted average, without the latitude and longitude dimensions.
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

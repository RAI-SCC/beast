"""Tests util functions for the evaluation in beast."""

from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beast.evaluation.utils import (
    DATA_KEY,
    EVALUATION_KEY,
    PRESSURE_LEVELS_KEY,
    PRESSURE_VARIABLES_KEY,
    SURFACE_VARIABLES_KEY,
    select_variables,
)

_ALL_SURFACE = [
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "sea_surface_temperature",
    "total_precipitation",
]
_ALL_PRESSURE = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
_ALL_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

_N_TOTAL = len(_ALL_SURFACE) + len(_ALL_PRESSURE) * len(_ALL_LEVELS)  # = 84


def _expected_indices(
    eval_surface: list[str],
    eval_pressure: list[str],
    eval_levels: list[int],
    channel_rank: int,
    channel_group_size: int,
) -> tuple[list[int], int]:
    """
    Compute expected local channel indices independently of the implementation.

    Parameters
    ==========
    eval_surface: list[str]
        List of surface variables to be selected.
    eval_pressure: list[str]
        List of pressure variables to be selected.
    eval_levels: list[int]
        List of pressure levels to be selected.
    channel_rank: int
        Channel rank to be used.
    channel_group_size: int
        Channel group size to be used.

    Returns
    =======
    list[int]
        Expected local channel indices.
    int
        Shard size.
    """
    if channel_group_size > 1:
        shard = _N_TOTAL // 2
        local_start = channel_rank * shard
        local_end = local_start + shard
    else:
        shard = _N_TOTAL
        local_start = 0
        local_end = _N_TOTAL

    indices = []

    for s in eval_surface:
        if s in _ALL_SURFACE:
            g = _ALL_SURFACE.index(s)
            if local_start <= g < local_end:
                indices.append(g - local_start)

    n_levels = len(_ALL_LEVELS)
    n_surf = len(_ALL_SURFACE)
    for pv in eval_pressure:
        if pv in _ALL_PRESSURE:
            pv_idx = _ALL_PRESSURE.index(pv)
            for lv in eval_levels:
                if lv in _ALL_LEVELS:
                    l_idx = _ALL_LEVELS.index(lv)
                    g = n_surf + pv_idx * n_levels + l_idx
                    if local_start <= g < local_end:
                        indices.append(g - local_start)

    return indices, shard


# Define strategies.
_eval_surface_st = st.lists(
    st.sampled_from(_ALL_SURFACE), min_size=0, max_size=len(_ALL_SURFACE), unique=True
)
_eval_pressure_st = st.lists(
    st.sampled_from(_ALL_PRESSURE), min_size=0, max_size=len(_ALL_PRESSURE), unique=True
)
_eval_levels_st = st.lists(
    st.sampled_from(_ALL_LEVELS), min_size=0, max_size=len(_ALL_LEVELS), unique=True
)


@given(
    eval_surface=_eval_surface_st,
    eval_pressure=_eval_pressure_st,
    eval_levels=_eval_levels_st,
    channel_group_size=st.sampled_from([1, 2]),
    channel_rank=st.integers(min_value=0, max_value=1),
    as_list=st.booleans(),
)
@settings(max_examples=100)
def test_select_variables(
    eval_surface: list[str],
    eval_pressure: list[str],
    eval_levels: list[int],
    channel_group_size: int,
    channel_rank: int,
    as_list: bool,
) -> None:
    """
    Test the select variables function.

    The hypothesis test considers a combination of surface variables, pressure variables, and pressure levels as well
    as different channel group sizes and channel ranks. The test ensures that the variables are selected as expected
    by comparing both values and shapes.

    Parameters
    ==========
    eval_surface: list[str]
        List of surface variables to be selected for the evaluation.
    eval_pressure: list[str]
        List of pressure variables to be selected for the evaluation.
    eval_levels: list[str]
        List of levels to be selected for the evaluation.
    channel_group_size: int
        Size of the channel group.
    channel_rank: int
        Rank of the channel.
    as_list: bool
        Whether one or multiple tensors are being considered. Required for the asserts.
    """
    # Channel_rank must be valid for the chosen group size.
    if channel_group_size == 1:
        channel_rank = 0

    # Compute expected indices and local shard size.
    expected, shard = _expected_indices(
        eval_surface, eval_pressure, eval_levels, channel_rank, channel_group_size
    )

    # Expand config based on subsets passed into the test.
    config = {
        DATA_KEY: {
            SURFACE_VARIABLES_KEY: _ALL_SURFACE,
            PRESSURE_VARIABLES_KEY: _ALL_PRESSURE,
            PRESSURE_LEVELS_KEY: _ALL_LEVELS,
        },
        EVALUATION_KEY: {
            SURFACE_VARIABLES_KEY: eval_surface,
            PRESSURE_VARIABLES_KEY: eval_pressure,
            PRESSURE_LEVELS_KEY: eval_levels,
        },
    }

    # Generate dummy input tensor.
    t = torch.arange(2 * shard, dtype=torch.float32).reshape(2, shard)
    inp = [t, t * 2] if as_list else t

    result = select_variables(
        inp, config, channel_rank=channel_rank, channel_group_size=channel_group_size
    )

    # Check results.
    results_to_check: list[tuple[torch.Tensor, torch.Tensor]]
    if as_list:
        assert isinstance(result, list)
        assert len(result) == 2
        results_to_check = list(zip([t, t * 2], result))
    else:
        assert isinstance(result, torch.Tensor)
        results_to_check = [(t, result)]

    for orig, res in results_to_check:
        assert res.shape == (
            2,
            len(expected),
        ), f"Expected {len(expected)} channels, got {res.shape[1]}"
        assert torch.equal(res, orig[:, expected]), (
            f"Channel values mismatch. Expected indices: {expected}"
        )

    # No duplicate indices.
    assert len(expected) == len(set(expected)), "Duplicate channel indices detected"


@pytest.mark.parametrize(
    "bad_config, error_match",
    [
        ({EVALUATION_KEY: {}}, "Configuration error"),
        (
            {
                DATA_KEY: {PRESSURE_VARIABLES_KEY: [], PRESSURE_LEVELS_KEY: []},
                EVALUATION_KEY: {},
            },
            "Configuration error",
        ),
        (
            {
                DATA_KEY: {SURFACE_VARIABLES_KEY: [], PRESSURE_LEVELS_KEY: []},
                EVALUATION_KEY: {},
            },
            "Configuration error",
        ),
        (
            {
                DATA_KEY: {SURFACE_VARIABLES_KEY: [], PRESSURE_VARIABLES_KEY: []},
                EVALUATION_KEY: {},
            },
            "Configuration error",
        ),
        (
            {
                DATA_KEY: {
                    SURFACE_VARIABLES_KEY: [],
                    PRESSURE_VARIABLES_KEY: [],
                    PRESSURE_LEVELS_KEY: [],
                }
            },
            "Configuration error",
        ),
    ],
)
def test_bad_config_for_select_variables(
    bad_config: dict[str, Any], error_match: str
) -> None:
    """
    Ensure error messages are raised with incorrect configs.

    Parameters
    ==========
    bad_config: dict[str, Any]
        Config dict with errors.
    error_match: str
        Matching error string that should appear in error message.
    ."""
    t = torch.zeros(2, 4)
    with pytest.raises(KeyError, match=error_match):
        select_variables(t, bad_config, channel_rank=0, channel_group_size=1)


def test_wrong_input_for_select_variables() -> None:
    """Ensure incorrect input type raises an error."""
    config = {
        DATA_KEY: {
            SURFACE_VARIABLES_KEY: _ALL_SURFACE,
            PRESSURE_VARIABLES_KEY: _ALL_PRESSURE,
            PRESSURE_LEVELS_KEY: _ALL_LEVELS,
        },
        EVALUATION_KEY: {},
    }
    with pytest.raises(AssertionError, match="Expected `torch.Tensor` or `Sequence`"):
        select_variables(42, config, channel_rank=0, channel_group_size=1)  # type: ignore[arg-type]

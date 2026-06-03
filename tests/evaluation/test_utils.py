"""Tests util functions for the evaluation in beast."""

from collections.abc import Callable, Sequence
from typing import Any, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beast.evaluation import latitude_weighted_average
from beast.evaluation.utils import (
    DATA_KEY,
    EVALUATION_KEY,
    PRESSURE_LEVELS_KEY,
    PRESSURE_VARIABLES_KEY,
    SURFACE_VARIABLES_KEY,
    make_variable_names,
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
        shard = _N_TOTAL // channel_group_size
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


def _expected_names(
    eval_surface: list[str],
    eval_pressure: list[str],
    eval_levels: list[int],
) -> list[str]:
    """
    Compute expected variable name strings independently of the implementation.

    Parameters
    ==========
    eval_surface: list[str]
        List of surface variables to be selected.
    eval_pressure: list[str]
        List of pressure variables to be selected.
    eval_levels: list[int]
        List of pressure levels to be selected.

    Returns
    =======
    list[str]
        Expected variable name strings.
    """
    names = []
    for s in eval_surface:
        if s in _ALL_SURFACE:
            names.append(s)
    for pv in eval_pressure:
        if pv in _ALL_PRESSURE:
            for lv in eval_levels:
                if lv in _ALL_LEVELS:
                    names.append(f"{pv}_{lv}")
    return names


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

# Valid group sizes that evenly divide _N_TOTAL (84).
_VALID_GROUP_SIZES = [1, 2, 4]


@st.composite
def _channel_group_st(draw: st.DrawFn) -> tuple[int, int]:
    """Draw a (channel_group_size, channel_rank) pair where rank is always valid."""
    group_size = draw(st.sampled_from(_VALID_GROUP_SIZES))
    rank = draw(st.integers(min_value=0, max_value=group_size - 1))
    return group_size, rank


@given(
    eval_surface=_eval_surface_st,
    eval_pressure=_eval_pressure_st,
    eval_levels=_eval_levels_st,
    channel_group_and_rank=_channel_group_st(),
    as_list=st.booleans(),
)
@settings(max_examples=100)
def test_select_variables_via_config(
    eval_surface: list[str],
    eval_pressure: list[str],
    eval_levels: list[int],
    channel_group_and_rank: tuple[int, int],
    as_list: bool,
) -> None:
    """
    Test the select variables function using a config dict.

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
    channel_group_and_rank: tuple[int, int]
        A (channel_group_size, channel_rank) pair where rank is always valid for the group size.
    as_list: bool
        Whether one or multiple tensors are being considered. Required for the asserts.
    """
    channel_group_size, channel_rank = channel_group_and_rank

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

    # Generate dummy input tensor of shape (2, shard): use dim=1 explicitly.
    t = torch.arange(2 * shard, dtype=torch.float32).reshape(2, shard)
    inp = [t, t * 2] if as_list else t

    result = select_variables(
        inp,
        config=config,
        channel_rank=channel_rank,
        channel_group_size=channel_group_size,
        dim=1,
    )

    _assert_result(result, t, as_list, expected)


@given(
    eval_surface=_eval_surface_st,
    eval_pressure=_eval_pressure_st,
    eval_levels=_eval_levels_st,
    channel_group_and_rank=_channel_group_st(),
    as_list=st.booleans(),
)
@settings(max_examples=100)
def test_select_variables_via_direct_args(
    eval_surface: list[str],
    eval_pressure: list[str],
    eval_levels: list[int],
    channel_group_and_rank: tuple[int, int],
    as_list: bool,
) -> None:
    """
    Test the select variables function using direct keyword arguments (no config dict).

    Parameters
    ==========
    eval_surface: list[str]
        List of surface variables to be selected for the evaluation.
    eval_pressure: list[str]
        List of pressure variables to be selected for the evaluation.
    eval_levels: list[str]
        List of levels to be selected for the evaluation.
    channel_group_and_rank: tuple[int, int]
        A (channel_group_size, channel_rank) pair where rank is always valid for the group size.
    as_list: bool
        Whether one or multiple tensors are being considered. Required for the asserts.
    """
    channel_group_size, channel_rank = channel_group_and_rank

    expected, shard = _expected_indices(
        eval_surface, eval_pressure, eval_levels, channel_rank, channel_group_size
    )

    # Generate dummy input tensor of shape (2, shard): use dim=1 explicitly.
    t = torch.arange(2 * shard, dtype=torch.float32).reshape(2, shard)
    inp = [t, t * 2] if as_list else t

    result = select_variables(
        inp,
        channel_rank=channel_rank,
        channel_group_size=channel_group_size,
        all_surface_variables=_ALL_SURFACE,
        all_pressure_variables=_ALL_PRESSURE,
        all_pressure_levels=_ALL_LEVELS,
        eval_surface_variables=eval_surface,
        eval_pressure_variables=eval_pressure,
        eval_pressure_levels=eval_levels,
        dim=1,
    )

    _assert_result(result, t, as_list, expected)


@pytest.mark.parametrize(
    "dim, shape, expected_shape_fn",
    [
        # dim=1: (batch=2, channels=N) → select along axis 1
        (1, lambda n: (2, n), lambda n_sel: (2, n_sel)),
        # dim=-3: (batch=2, channels=N, h=3, w=3) → select along axis -3 (axis 1)
        (-3, lambda n: (2, n, 3, 3), lambda n_sel: (2, n_sel, 3, 3)),
        # dim=0: (channels=N, batch=2) → select along axis 0
        (0, lambda n: (n, 2), lambda n_sel: (n_sel, 2)),
    ],
)
def test_select_variables_dim(
    dim: int,
    shape: Callable[[int], tuple[int, ...]],
    expected_shape_fn: Callable[[int], tuple[int, ...]],
) -> None:
    """
    Test that the ``dim`` argument selects along the correct tensor dimension.

    Parameters
    ==========
    dim: int
        Dimension to select along.
    shape : Callable[[int], tuple[int, ...]]
        Callable that takes the shard size and returns the tensor shape.
    expected_shape_fn : Callable[[int], tuple[int, ...]]
        Callable that takes the number of selected variables and returns the expected output shape.
    """
    eval_surface = _ALL_SURFACE[:2]
    eval_pressure = _ALL_PRESSURE[:1]
    eval_levels = _ALL_LEVELS[:1]

    expected_indices, shard = _expected_indices(
        eval_surface, eval_pressure, eval_levels, 0, 1
    )
    n_sel = len(expected_indices)

    t = torch.arange(
        int(torch.tensor(shape(shard)).prod()), dtype=torch.float32
    ).reshape(shape(shard))

    result = select_variables(
        t,
        channel_rank=0,
        channel_group_size=1,
        all_surface_variables=_ALL_SURFACE,
        all_pressure_variables=_ALL_PRESSURE,
        all_pressure_levels=_ALL_LEVELS,
        eval_surface_variables=eval_surface,
        eval_pressure_variables=eval_pressure,
        eval_pressure_levels=eval_levels,
        dim=dim,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == expected_shape_fn(n_sel)
    assert torch.equal(result, t.index_select(dim, torch.tensor(expected_indices)))


def _assert_result(
    result: torch.Tensor | Sequence[torch.Tensor],
    t: torch.Tensor,
    as_list: bool,
    expected: list[int],
) -> None:
    """Check results and shapes are as expected."""
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


@given(
    eval_surface=_eval_surface_st,
    eval_pressure=_eval_pressure_st,
    eval_levels=_eval_levels_st,
)
@settings(max_examples=100)
def test_make_variable_names_via_config(
    eval_surface: list[str],
    eval_pressure: list[str],
    eval_levels: list[int],
) -> None:
    """
    Test make_variable_names using a config dict.

    Parameters
    ==========
    eval_surface: list[str]
        List of surface variables to be selected for the evaluation.
    eval_pressure: list[str]
        List of pressure variables to be selected for the evaluation.
    eval_levels: list[int]
        List of levels to be selected for the evaluation.
    """
    expected = _expected_names(eval_surface, eval_pressure, eval_levels)

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

    result = make_variable_names(config=config)

    assert result == expected


@given(
    eval_surface=_eval_surface_st,
    eval_pressure=_eval_pressure_st,
    eval_levels=_eval_levels_st,
)
@settings(max_examples=100)
def test_make_variable_names_via_direct_args(
    eval_surface: list[str],
    eval_pressure: list[str],
    eval_levels: list[int],
) -> None:
    """
    Test make_variable_names using direct keyword arguments (no config dict).

    Parameters
    ==========
    eval_surface: list[str]
        List of surface variables to be selected for the evaluation.
    eval_pressure: list[str]
        List of pressure variables to be selected for the evaluation.
    eval_levels: list[int]
        List of levels to be selected for the evaluation.
    """
    expected = _expected_names(eval_surface, eval_pressure, eval_levels)

    result = make_variable_names(
        all_surface_variables=_ALL_SURFACE,
        all_pressure_variables=_ALL_PRESSURE,
        all_pressure_levels=_ALL_LEVELS,
        eval_surface_variables=eval_surface,
        eval_pressure_variables=eval_pressure,
        eval_pressure_levels=eval_levels,
    )

    assert result == expected


def test_make_variable_names_format() -> None:
    """Verify the exact output format for surface and pressure variable names."""
    result = make_variable_names(
        all_surface_variables=_ALL_SURFACE,
        all_pressure_variables=_ALL_PRESSURE,
        all_pressure_levels=_ALL_LEVELS,
        eval_surface_variables=["mean_sea_level_pressure", "2m_temperature"],
        eval_pressure_variables=["temperature", "geopotential"],
        eval_pressure_levels=[500, 850],
    )

    assert result == [
        "mean_sea_level_pressure",
        "2m_temperature",
        "temperature_500",
        "temperature_850",
        "geopotential_500",
        "geopotential_850",
    ]


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
        select_variables(t, config=bad_config, channel_rank=0, channel_group_size=1)


def test_missing_all_variables_no_config() -> None:
    """Ensure a KeyError is raised when no config and no explicit all_* args are given."""
    t = torch.zeros(2, 4)
    with pytest.raises(KeyError, match="Configuration error"):
        select_variables(t, channel_rank=0, channel_group_size=1)


def test_latitude_weighted_average(device: torch.device) -> None:
    """Test that latitude weighted average computes correct weights and applies them."""
    # Create a dummy input tensor of shape (batch=2, channels=3, lat=4, lon=5)
    x = torch.ones(2, 3, 4, 5, device=device)

    # Compute latitude weighted average
    result = latitude_weighted_average(x)

    assert torch.allclose(result, torch.ones(2, 3, device=device)), (
        f"Expected torch.ones(2, 3), got {result}"
    )

    # Compute latitude weighted average
    result = latitude_weighted_average(x, lat_dim=2, lon_dim=0)

    assert torch.allclose(result, torch.ones(3, 5, device=device)), (
        f"Expected ones(3, 5), got {result}"
    )

    x2 = torch.eye(10, device=device)

    result = latitude_weighted_average(x2, lat_dim=0, lon_dim=1)
    assert torch.allclose(result, torch.tensor(0.1, device=device)), (
        f"Expected torch.tensor(0.1), got {result}"
    )

    max_size = 10
    x2 = torch.zeros(max_size, max_size, device=device, dtype=torch.float32)
    local_random_index1 = torch.randint(0, max_size, (1,)).item()
    local_random_index2 = torch.randint(0, max_size, (1,)).item()
    local_random_index3 = torch.randint(0, max_size, (1,)).item()
    local_random_index4 = torch.randint(0, max_size, (1,)).item()
    while (
        local_random_index3 == local_random_index1
        and local_random_index4 == local_random_index2
    ):
        local_random_index3 = torch.randint(0, max_size, (1,)).item()
        local_random_index4 = torch.randint(0, max_size, (1,)).item()

    x2[cast(int, local_random_index1), cast(int, local_random_index2)] = 1.0
    x2[cast(int, local_random_index3), cast(int, local_random_index4)] = 1.0

    result2 = latitude_weighted_average(x2, lat_dim=0, lon_dim=1)

    local_cosines = torch.cos(
        -torch.pi / 2.0
        + torch.pi
        * torch.tensor(
            [local_random_index1, local_random_index3],
            device=device,
            dtype=torch.float32,
        )
        / max_size
    )
    all_cosine_sum = torch.cos(
        -torch.pi / 2.0
        + torch.pi
        * torch.arange(max_size, device=device, dtype=torch.float32)
        / max_size
    ).sum()

    expected_result2 = local_cosines.sum() / (all_cosine_sum * max_size)

    assert torch.allclose(result2, expected_result2), (
        f"Expected {expected_result2}, got {result2}"
    )

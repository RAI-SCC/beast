"""Tests for beast.evaluation.derived_metrics."""

from typing import Tuple

import pytest
import torch

from beast.evaluation import calculate_rmse, calculate_ssr


@pytest.mark.parametrize("dimensions", [(3, 4), (5,), (2, 2, 2), (2, 3, 4, 5)])
def test_calculate_rmse(dimensions: Tuple[int, ...]) -> None:
    metric_dict = {"mse": torch.randn(*dimensions) ** 2}
    expected_rmse = torch.sqrt(metric_dict["mse"])
    assert torch.allclose(calculate_rmse(metric_dict), expected_rmse), (
        "RMSE calculation is incorrect."
    )

    with pytest.raises(
        KeyError,
        match="Required metric 'mse' for 'rmse' calculation not found in metric_dict",
    ):
        calculate_rmse({})  # Missing 'mse' key


@pytest.mark.parametrize("dimensions", [(3, 4), (5,), (2, 2, 2), (2, 3, 4, 5)])
def test_calculate_ssr(dimensions: Tuple[int, ...]) -> None:
    metric_dict = {
        "mse": torch.randn(*dimensions) ** 2,
        "variance": torch.randn(*dimensions) ** 2,
    }
    expected_ssr = metric_dict["variance"].sqrt() / metric_dict["mse"].sqrt()
    assert torch.allclose(calculate_ssr(metric_dict), expected_ssr), (
        "SSR calculation is incorrect."
    )

    with pytest.raises(
        KeyError,
        match="Required metric 'mse' for 'spread-skill ratio' calculation not found in metric_dict",
    ):
        calculate_ssr({"variance": torch.tensor(2.0)})  # Missing 'mse' key

    with pytest.raises(
        KeyError,
        match="Required metric 'variance' for 'spread-skill ratio' calculation not found in metric_dict",
    ):
        calculate_ssr({"mse": torch.tensor(8.0)})  # Missing 'variance' key

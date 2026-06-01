"""Tests for beast.evaluation.derived_metrics."""

from typing import Tuple

import pytest
import torch

from beast.evaluation import rmse, spread_skill_ratio


@pytest.mark.parametrize("dimensions", [(3, 4), (5,), (2, 2, 2), (2, 3, 4, 5)])
def test_calculate_rmse(dimensions: Tuple[int, ...], device: torch.device) -> None:
    metric_dict = {"mse": torch.randn(*dimensions, device=device) ** 2}
    expected_rmse = torch.sqrt(metric_dict["mse"])
    assert torch.allclose(rmse(metric_dict), expected_rmse), (
        "RMSE calculation is incorrect."
    )

    with pytest.raises(
        KeyError,
        match="Required metric 'mse' for 'rmse' calculation not found in metric_dict",
    ):
        rmse({})  # Missing 'mse' key


@pytest.mark.parametrize("dimensions", [(3, 4), (5,), (2, 2, 2), (2, 3, 4, 5)])
def test_calculate_spread_skill_ratio(
    dimensions: Tuple[int, ...], device: torch.device
) -> None:
    metric_dict = {
        "mse": torch.randn(*dimensions, device=device) ** 2,
        "variance": torch.randn(*dimensions, device=device) ** 2,
    }
    expected_ssr = metric_dict["variance"].sqrt() / metric_dict["mse"].sqrt()
    assert torch.allclose(spread_skill_ratio(metric_dict), expected_ssr), (
        "Spread-Skill Ratio calculation is incorrect."
    )

    with pytest.raises(
        KeyError,
        match="Required metric 'mse' for 'spread-skill ratio' calculation not found in metric_dict",
    ):
        spread_skill_ratio(
            {"variance": torch.tensor(2.0, device=device)}
        )  # Missing 'mse' key

    with pytest.raises(
        KeyError,
        match="Required metric 'variance' for 'spread-skill ratio' calculation not found in metric_dict",
    ):
        spread_skill_ratio(
            {"mse": torch.tensor(8.0, device=device)}
        )  # Missing 'variance' key

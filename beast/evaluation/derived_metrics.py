"""Metrics derived from other metrics, rather than computed in the validation loop."""

from typing import Dict

import torch
from torch import Tensor


def rmse(metric_dict: Dict[str, Tensor]) -> Tensor:
    """Compute the root mean squared error from the mean squared error.

    Parameters
    ----------
    metric_dict : Dict[str, Tensor]
        Metrics, containing ``"mse"``.

    Returns
    -------
    Tensor
        The RMSE.

    Raises
    ------
    KeyError
        If ``"mse"`` is missing from `metric_dict`.
    """
    try:
        rmse = torch.sqrt(metric_dict["mse"])
    except KeyError:
        raise KeyError(
            "Required metric 'mse' for 'rmse' calculation not found in metric_dict"
        )
    return rmse


def spread_skill_ratio(metric_dict: Dict[str, Tensor]) -> Tensor:
    """Compute the spread-skill ratio, ``sqrt(variance) / sqrt(mse)``.

    Parameters
    ----------
    metric_dict : Dict[str, Tensor]
        Metrics, containing ``"mse"`` and ``"variance"``.

    Returns
    -------
    Tensor
        The spread-skill ratio.

    Raises
    ------
    KeyError
        If ``"mse"`` or ``"variance"`` is missing from `metric_dict`.
    """
    try:
        ssr = torch.sqrt(metric_dict["variance"]) / torch.sqrt(metric_dict["mse"])
    except KeyError as e:
        raise KeyError(
            f"Required metric '{e.args[0]}' for 'spread-skill ratio' calculation not found in metric_dict"
        )
    return ssr

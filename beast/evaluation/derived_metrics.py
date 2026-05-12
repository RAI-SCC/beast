"""Contains the calculation of metrics that are not calculated within the validation loop, but instead derived from other metrics."""

from typing import Dict

import torch
from torch import Tensor


def calculate_rmse(metric_dict: Dict[str, Tensor]) -> Tensor:
    """
    Calculate the Root Mean Squared Error (RMSE) from the Mean Squared Error (MSE) provided in the metric_dict.

    Parameters
    ----------
    metric_dict : Dict[str, Tensor]
        A dictionary containing the MSE value under the key 'mse'.

    Returns
    -------
    Tensor
        The calculated RMSE.
    """
    try:
        rmse = torch.sqrt(metric_dict["mse"])
    except KeyError:
        raise KeyError(
            "Required metric 'mse' for 'rmse' calculation not found in metric_dict"
        )
    return rmse


def calculate_ssr(metric_dict: Dict[str, Tensor]) -> Tensor:
    """
    Calculate the spread-skill ratio (SSR) from the Mean Squared Error (MSE) and the total variance provided in the metric_dict.

    Parameters
    ----------
    metric_dict : Dict[str, Tensor]
        A dictionary containing the MSE value under the key 'mse' and the total variance under the key 'variance'.

    Returns
    -------
    Tensor
        The calculated SSR.
    """
    try:
        ssr = torch.sqrt(metric_dict["variance"]) / torch.sqrt(metric_dict["mse"])
    except KeyError as e:
        raise KeyError(
            f"Required metric '{e.args[0]}' for 'spread-skill ratio' calculation not found in metric_dict"
        )
    return ssr

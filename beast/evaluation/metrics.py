"""Elementwise error metrics."""

import torch
from torch.nn.functional import l1_loss, mse_loss


def mse(
    prediction: torch.Tensor, target: torch.Tensor, batch_dimension: int = 0
) -> torch.Tensor:
    """Compute the mean squared error over the batch dimension.

    Parameters
    ----------
    prediction : Tensor
        Model prediction.
    target : Tensor
        Reference values, of the shape of `prediction`.
    batch_dimension : int, default: 0
        Dimension to average over.

    Returns
    -------
    Tensor
        Squared error averaged over `batch_dimension`.
    """
    return mse_loss(prediction, target, reduction="none").mean(dim=batch_dimension)


def mae(
    prediction: torch.Tensor, target: torch.Tensor, batch_dimension: int = 0
) -> torch.Tensor:
    """Compute the mean absolute error over the batch dimension.

    Parameters
    ----------
    prediction : Tensor
        Model prediction.
    target : Tensor
        Reference values, of the shape of `prediction`.
    batch_dimension : int, default: 0
        Dimension to average over.

    Returns
    -------
    Tensor
        Absolute error averaged over `batch_dimension`.
    """
    return l1_loss(prediction, target, reduction="none").mean(dim=batch_dimension)

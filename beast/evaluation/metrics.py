"""Functions to calculate simple metrics."""

import torch
from torch.nn.functional import l1_loss, mse_loss


def mse(
    prediction: torch.Tensor, target: torch.Tensor, batch_dimension: int = 0
) -> torch.Tensor:
    """
    Calculate mean squared error.

    Parameters
    ----------
    prediction: Tensor
        The model prediction.
    target: Tensor
        The reference result.
    batch_dimension: int, default: 0
        Batch dimension to reduce over.

    Returns
    -------
    Tensor
        The squared error averaged over the specified batch dimension.

    """
    return mse_loss(prediction, target, reduction="none").mean(dim=batch_dimension)


def mae(
    prediction: torch.Tensor, target: torch.Tensor, batch_dimension: int = 0
) -> torch.Tensor:
    """
    Calculate mean absolute error.

    Parameters
    ----------
    prediction: Tensor
        The model prediction.
    target: Tensor
        The reference result.
    batch_dimension: int, default: 0
        Batch dimension to reduce over.

    Returns
    -------
    Tensor
        The absolute error averaged over the specified batch dimension.
    """
    return l1_loss(prediction, target, reduction="none").mean(dim=batch_dimension)

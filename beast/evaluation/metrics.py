"""Functions to calculate simple metrics."""

from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss


def calculate_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """
    Calculate mean squared error.

    Parameters
    ----------
    prediction: Tensor
        The model prediction.
    target: Tensor
        The reference result.
    """
    return mse_loss(prediction, target)


def calculate_mae(prediction: Tensor, target: Tensor) -> Tensor:
    """
    Calculate mean absolute error.

    Parameters
    ----------
    prediction: Tensor
        The model prediction.
    target: Tensor
        The reference result.
    """
    return l1_loss(prediction, target)

"""Implements data transformations to be passed to the dataset classes."""

from typing import Tuple

import torch
from torch import Tensor, nn


class Identity(nn.Module):
    """
    Trivial transformation that returns inputs unchanged.

    Notes
    -----
    This module can be used as a placeholder transform when no data
    transformation is required.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the identity transformation.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The same tensor as input.
        """
        return x


def _reshape_for_downsampling(
    x: Tensor, dims: Tuple[int, ...], factor: int, negative_dims: bool = False
) -> Tuple[Tensor, Tuple[int, ...]]:
    """
    Reshape the input tensor into a form suitable for downsampling and return the dimensions that reduction should take place over.

    Parameters
    ----------
    x: Tensor
        The tensor to be reshaped.
    dims: Tuple[int, ...]
        The dimensions the downsampling will take place over
    factor: int
        The factor by which the dimensions are downscaled
    negative_dims: bool, default: False
        Indicates whether the tuple given for dims contains all negative or all positive dimensions (mixed is not implemented)

    Returns
    -------
    x_reshaped: Tensor
        The reshaped tensor with all indicated dimensions split into (dim // factor, factor)
    target_dims: Tuple[int, ...]
        The tuple containing all of the newly created dimensions of size factor
    """
    x_shape = x.shape

    dim_list = list(dims)
    if negative_dims:
        dim_list = [len(x_shape) + dim for dim in dims]

    target_shape = []
    for i, entry in enumerate(x_shape):
        if i in dim_list:
            target_shape.extend([entry // factor, factor])
        else:
            target_shape.append(entry)

    target_dims = []
    for i, dim in enumerate(reversed(dim_list)):
        target_dims.append(i + dim + 1)

    try:
        x = x.reshape(tuple(target_shape))
    except RuntimeError as err:
        if " is invalid for input of size " in str(err):
            raise RuntimeError(
                f"not all dimensions at positions {dims} of tensor with shape {list(x_shape)} are divisible by {factor}"
            )
        else:
            raise err

    return x, tuple(target_dims)


class Downsample(nn.Module):
    """
    Downsampling transformation for a given downsampling factor.

    Parameters
    ----------
    factor : int
        The factor by which the dimensions are downsampled.
    dims : Tuple[int, ...], default: (-1,)
        The dimensions to downsample over. Defaults to only the last dimension.

    Attributes
    ----------
    factor : int
        The factor by which selected dimensions are downsampled.
    dims : Tuple[int, ...]
        The dimensions over which downsampling is applied.
    negative_dims : bool
        Whether ``dims`` contains negative dimension indices.
    """

    def __init__(self, factor: int, dims: Tuple[int, ...] = (-1,)):
        super().__init__()

        assert factor > 0, (
            f"Downsampling factor must be greater than zero (given: {factor})"
        )

        self.factor = factor
        all_positive = True
        all_negative = True
        for dim in dims:
            if dim >= 0:
                all_negative = False
            else:
                all_positive = False

        assert all_positive or all_negative, (
            f"Dimensions must be either all positive or all negative (given: {dims})"
        )

        self.negative_dims = all_negative
        sorted_dims = list(dims)
        sorted_dims.sort(reverse=True)
        self.dims = tuple(sorted_dims)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform downsampling by reshaping and taking the mean over blocks of dimension factor^len(dims).

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The downsampled tensor.
        """
        x, target_dims = _reshape_for_downsampling(
            x, self.dims, self.factor, self.negative_dims
        )

        return x.mean(dim=target_dims)


class DownsampleMeanStd(Downsample):
    """
    Downsampling transformation returning mean and standard deviation.

    Parameters
    ----------
    factor : int
        The factor by which the dimensions are downsampled.
    dims : Tuple[int, ...], default: (-1,)
        The dimensions to downsample over. Defaults to only the last dimension.

    Attributes
    ----------
    factor : int
        The factor by which selected dimensions are downsampled.
    dims : Tuple[int, ...]
        The dimensions over which downsampling is applied.
    negative_dims : bool
        Whether ``dims`` contains negative dimension indices.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform downsampling and calculate both mean and standard deviation over each patch.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            A tensor containing the mean and standard deviation stacked along a new dimension of size 2 appended to end.
        """
        x, target_dims = _reshape_for_downsampling(
            x, self.dims, self.factor, self.negative_dims
        )

        x_stds, x_means = torch.std_mean(x, dim=target_dims)

        return torch.stack((x_means, x_stds), dim=-1)

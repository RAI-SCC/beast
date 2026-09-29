"""Data transformations to pass to the dataset classes."""

from typing import Tuple

import torch
from torch import Tensor, nn


class Identity(nn.Module):
    """Transformation that returns its input unchanged, as a placeholder."""

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` unchanged.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            The input tensor.
        """
        return x


def _reshape_for_downsampling(
    x: Tensor, dims: Tuple[int, ...], factor: int, negative_dims: bool = False
) -> Tuple[Tensor, Tuple[int, ...]]:
    """Split each downsampled dimension into ``(size // factor, factor)``.

    Parameters
    ----------
    x : Tensor
        Tensor to reshape.
    dims : Tuple[int, ...]
        Dimensions to downsample over, all positive or all negative.
    factor : int
        Downsampling factor.
    negative_dims : bool, default: False
        Whether ``dims`` holds negative indices.

    Returns
    -------
    x_reshaped : Tensor
        The reshaped tensor.
    target_dims : Tuple[int, ...]
        The new dimensions of size ``factor``, to be reduced over.

    Raises
    ------
    RuntimeError
        If a dimension in ``dims`` is not divisible by ``factor``.
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
    """Downsample by averaging over blocks of ``factor`` along each dimension.

    Parameters
    ----------
    factor : int
        Downsampling factor.
    dims : Tuple[int, ...], default: (-1,)
        Dimensions to downsample over, all positive or all negative.

    Attributes
    ----------
    factor : int
        Downsampling factor.
    dims : Tuple[int, ...]
        Dimensions to downsample over, sorted in descending order.
    negative_dims : bool
        Whether ``dims`` holds negative indices.
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
        """Average ``x`` over blocks of ``factor ** len(dims)`` elements.

        Parameters
        ----------
        x : Tensor
            Input tensor.

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
    """Downsample to the mean and standard deviation of each block.

    Parameters and attributes are those of :class:`Downsample`.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Compute the mean and standard deviation over blocks of ``x``.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Means and standard deviations stacked along a new last dimension
            of size 2.
        """
        x, target_dims = _reshape_for_downsampling(
            x, self.dims, self.factor, self.negative_dims
        )

        x_stds, x_means = torch.std_mean(x, dim=target_dims)

        return torch.stack((x_means, x_stds), dim=-1)

"""Plotting functionality."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure
from torch import Tensor


def pred_target_bias_plot(
    prediction: Tensor,
    target: Tensor,
    label: str,
) -> Figure:
    """
    Core plotting function.

    Plot prediction, target, and bias.

    Parameters
    ----------
    prediction: Tensor
        Prediction to plot.
    target: Tensor
        Target to plot.
    label: str
        Plot label.

    Returns
    -------
    Figure
        A plot as Figure object.
    """
    # Bind to new names rather than rebinding the Tensor parameters: matplotlib
    # needs arrays, but the signature promises Tensors.
    prediction_np = prediction.cpu().numpy()
    target_np = target.cpu().numpy()

    # bias calculation
    bias_map = target_np - prediction_np

    # shared scale for prediction and target
    state_lim = (
        min(prediction_np.min(), target_np.min()),
        max(prediction_np.max(), target_np.max()),
    )

    # symmetrical scale for bias
    b_max = max(np.abs(bias_map.max()), np.abs(bias_map.min()))
    bias_lim = (-b_max, b_max)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    data_name = ("Prediction", "Target", "Bias")
    data_list = (prediction_np, target_np, bias_map)
    colormaps = ("viridis", "viridis", "RdBu_r")
    limits_list = (state_lim, state_lim, bias_lim)

    for ax, name, data, colormap, limits in zip(
        axes, data_name, data_list, colormaps, limits_list
    ):
        ax.set_title(f"{name}: {label}")
        c = ax.imshow(
            data,
            origin="upper",
            cmap=colormap,
            vmin=limits[0],
            vmax=limits[1],
        )
        fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.08)

    plt.tight_layout()
    return fig

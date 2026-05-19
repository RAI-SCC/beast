"""Plotting functionality."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor


def plot(
    plot_dir: Path,
    prediction: Tensor,
    target: Tensor,
    label: str,
) -> None:
    """Core plotting function."""
    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()

    # bias calculation
    bias_map = target - prediction

    # shared scale for prediction and target
    state_vmin = min(prediction.min(), target.min())
    state_vmax = max(prediction.max(), target.max())

    # symmetrical scale for bias
    bias_vmax = max(np.abs(bias_map.max()), np.abs(bias_map.min()))
    bias_vmin = -bias_vmax

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # prediction
    axes[0].set_title(f"Prediction: {label}")
    c0 = axes[0].imshow(
        prediction,
        origin="upper",
        cmap="viridis",
        vmin=state_vmin,
        vmax=state_vmax,
    )
    fig.colorbar(c0, ax=axes[0], orientation="horizontal", pad=0.08)

    # target
    axes[1].set_title(f"Target: {label}")
    c1 = axes[1].imshow(
        target,
        origin="upper",
        cmap="viridis",
        vmin=state_vmin,
        vmax=state_vmax,
    )
    fig.colorbar(c1, ax=axes[1], orientation="horizontal", pad=0.08)

    # bias
    axes[2].set_title(f"Bias: {label}")
    c2 = axes[2].imshow(
        bias_map, origin="upper", cmap="RdBu_r", vmin=bias_vmin, vmax=bias_vmax
    )
    fig.colorbar(c2, ax=axes[2], orientation="horizontal", pad=0.08)

    plt.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / f"{label}.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")

import os
from pathlib import Path

import pytest
import torch
from matplotlib import pyplot as plt

from beast.evaluation import plot, save_plot


def test_plot() -> None:
    """Test plotting method."""
    prediction = torch.randn(50, 50)
    target = torch.randn(50, 50)
    label = "Something"

    plot(prediction, target, label)


@pytest.mark.parametrize("close_fig", [True, False])
def test_save_plot(close_fig: bool, tmp_path: Path) -> None:
    """Test plot saving."""
    fig, ax = plt.subplots()
    ax.imshow(torch.randn(50, 50))

    fignum = fig.number

    assert plt.fignum_exists(fignum)
    assert not os.path.exists(tmp_path / "test.pdf")

    save_plot(fig, tmp_path, "test.pdf", close_fig)

    assert plt.fignum_exists(fignum) is not close_fig
    assert os.path.exists(tmp_path / "test.pdf")

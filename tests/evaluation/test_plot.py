import torch
from matplotlib import pyplot as plt

from beast.evaluation import pred_target_bias_plot


def test_plot() -> None:
    """Test plotting method."""
    prediction = torch.randn(50, 50)
    target = torch.randn(50, 50)
    label = "Something"

    pred_target_bias_plot(prediction, target, label)


def test_save_plot() -> None:
    """Test plot saving."""
    fig, ax = plt.subplots()
    ax.imshow(torch.randn(50, 50))

    fignum = fig.number

    assert plt.fignum_exists(fignum)

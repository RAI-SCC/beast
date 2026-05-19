"""Submodule containing evaluation functionality, i.e. metrics, plotting, documenting, etc."""

from .metrics import calculate_mae, calculate_mse
from .plot import plot, save_plot

__all__ = [
    "calculate_mae",
    "calculate_mse",
    "plot",
    "save_plot",
]

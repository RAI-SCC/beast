"""Submodule containing evaluation functionality, i.e. metrics, plotting, documenting, etc."""

from .metrics import calculate_mae, calculate_mse

__all__ = [
    "calculate_mae",
    "calculate_mse",
]

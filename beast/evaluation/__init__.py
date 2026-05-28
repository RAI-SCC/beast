"""Submodule containing evaluation functionality, i.e. metrics, plotting, documenting, etc."""

from .comm import gather_along_dimension
from .derived_metrics import calculate_rmse, calculate_spread_skill_ratio
from .keyed_tensor import KeyedTensor
from .metrics import calculate_mae, calculate_mse
from .plot import plot, save_plot
from .special_metrics import calculate_crps
from .utils import latitude_weighted_average, make_variable_names, select_variables

__all__ = [
    "calculate_crps",
    "calculate_rmse",
    "calculate_spread_skill_ratio",
    "calculate_mae",
    "calculate_mse",
    "plot",
    "save_plot",
    "select_variables",
    "make_variable_names",
    "gather_along_dimension",
    "latitude_weighted_average",
    "KeyedTensor",
]

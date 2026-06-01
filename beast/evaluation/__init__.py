"""Submodule containing evaluation functionality, i.e. metrics, plotting, documenting, etc."""

from .comm import gather_along_dimension
from .derived_metrics import rmse, spread_skill_ratio
from .metrics import mae, mse
from .plot import pred_target_bias_plot
from .special_metrics import crps
from .utils import latitude_weighted_average, make_variable_names, select_variables

__all__ = [
    "crps",
    "rmse",
    "spread_skill_ratio",
    "mae",
    "mse",
    "pred_target_bias_plot",
    "select_variables",
    "make_variable_names",
    "gather_along_dimension",
    "latitude_weighted_average",
]

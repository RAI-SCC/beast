"""Model module for BEAST weather prediction model.

This package provides model configurations and factory functions for
the BEAST weather prediction model.
"""

from .gb import BEAST_CONFIGURATIONS, Beast
from .model_factory import get_model

__all__ = ["get_model", "Beast", "BEAST_CONFIGURATIONS"]

"""BEAST-GB model implementation.

This package contains the BEAST-GB (Bayesian Exascale Atmospheric
Swim Transformer - Gordon Bell submission) model implementation.
"""

from .beast_config import BEAST_CONFIGURATIONS
from .beast_model import Beast

__all__ = ["BEAST_CONFIGURATIONS", "Beast"]

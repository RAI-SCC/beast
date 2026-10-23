"""Named BEAST configurations for :func:`~beast.model.get_model`."""

from typing import Any

from beast.model.beast_model import Beast

BEAST_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "BEAST-JUPITER-2.4B-64K": {
        "class": Beast,
        "config": {
            "height": 720,
            "width": 1440,
            "in_channels": 88,
            "out_channels": 84,
            "embed_dim": 3072,
            "heads": [24, 48, 24],
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1,
            "window_size_outer": [360, 180],
            "window_size_inner": [180, 30],
            "n_full_blocks": [1, 1, 1],
            "noise_dim": 32,
            "bayesian": True,
            "nice_norm": True,
            "qk_norm": False,
        },
    },
    "BEAST-JUPITER-0.7B-1.8K": {
        "class": Beast,
        "config": {
            "height": 720,
            "width": 1440,
            "in_channels": 88,
            "out_channels": 84,
            "embed_dim": 1024,
            "heads": [16, 32, 16],
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1,
            "window_size_outer": [30, 60],
            "window_size_inner": [20, 30],
            "n_full_blocks": [2, 3, 2],
            "noise_dim": 32,
            "qk_norm": True,
            "bayesian": True,
            "nice_norm": True,
            "shard_qk_norm": True,
        },
    },
    "BEAST-LUMI-1.1B-64K": {
        "class": Beast,
        "config": {
            "height": 720,
            "width": 1440,
            "in_channels": 88,
            "out_channels": 84,
            "embed_dim": 2048,
            "heads": [16, 32, 16],
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1,
            "window_size_outer": [360, 180],
            "window_size_inner": [180, 30],
            "n_full_blocks": [1, 1, 1],
            "noise_dim": 32,
            "qk_norm": False,
            "bayesian": True,
            "nice_norm": True,
        },
    },
    "BEASTLING": {
        "class": Beast,
        "config": {
            "height": 120,
            "width": 240,
            "in_channels": 88,
            "out_channels": 84,
            "embed_dim": 256,
            "heads": [16, 32, 16],
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1,
            "window_size_outer": [12, 30],
            "window_size_inner": [6, 10],
            "n_full_blocks": [2, 3, 2],
            "noise_dim": 32,
            "qk_norm": False,
            "bayesian": False,
            "nice_norm": False,
        },
    },
}
"""Named BEAST configurations.

Each name maps to the model ``class`` and the constructor arguments in
``config``. Pass the name to :func:`~beast.model.get_model` to build the model.
"""

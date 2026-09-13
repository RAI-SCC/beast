"""Shrunken model configurations for tests.

The production ``BEASTLING`` config is 14 M parameters (53 MiB in fp32) and runs
its forward pass over a 120x240 grid with 88 channels. Multiplied by the
parametrisation of the model tests, and by the four MPI ranks that share one
node's memory, that is far more than a CI runner should need for tests that only
check shapes, checkpoint file counts, and which parameters are shared across the
mesh. None of those assertions depend on the model being large.

``SMALL_BEAST_OVERRIDES`` keeps the architecture -- same block structure, same
attention, same patch embedding -- at 378 K parameters, a 37x reduction. Use
:func:`small_beast_config` to get a full config dict, or splat the overrides into
``get_model``.

The sizes are not arbitrary. ``Beast._build_model`` asserts, with ``S`` the
spatial group size and ``K`` the kernel size::

    width % (S * K * window_size_outer[1]) == 0
    width % (S * 2 * K * window_size_inner[1]) == 0

and all four window extents must be even. The attention windows also have to tile
the height after patch embedding (``height / K``) and after downsampling
(``height / 2K``). The values below satisfy all of that for spatial and channel
group sizes of both 1 and 2, which is what the mesh layouts in the tests use.
"""

from typing import Any

from beast.model.gb.beast_config import BEAST_CONFIGURATIONS

SMALL_BEAST_OVERRIDES: dict[str, Any] = {
    "height": 48,
    "width": 96,
    "in_channels": 16,
    "out_channels": 12,
    "embed_dim": 64,
    "heads": [4, 8, 4],
    "window_size_outer": [12, 12],
    "window_size_inner": [6, 6],
    "n_full_blocks": [1, 1, 1],
    "noise_dim": 8,
}


def small_beast_config(**overrides: Any) -> dict[str, Any]:
    """Return the BEASTLING config shrunk to test size.

    Parameters
    ----------
    **overrides : Any
        Extra config entries applied on top, e.g. ``bayesian=True``.

    Returns
    -------
    dict[str, Any]
        A fresh config dict; callers may mutate it freely.
    """
    config = dict(BEAST_CONFIGURATIONS["BEASTLING"]["config"])
    config.update(SMALL_BEAST_OVERRIDES)
    config.update(overrides)
    return config

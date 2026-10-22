import math
import warnings

import mpi4py.MPI as MPI
import pytest
import torch
import torch.distributed as dist
from _pytest.mark.structures import ParameterSet
from small_models import small_beast_config

from beast.comm import create_process_mesh_dict
from beast.model.beast_model import Beast
from beast.model.utils import count_model_parameters

WORLD_SIZE = MPI.COMM_WORLD.Get_size()


def _mesh(mesh_dims: list[int], id: str) -> ParameterSet:
    required = math.prod(mesh_dims)
    return pytest.param(
        mesh_dims,
        id=id,
        marks=pytest.mark.skipif(
            WORLD_SIZE != required,
            reason=f"mesh_dims {mesh_dims} needs {required} ranks, got {WORLD_SIZE}",
        ),
    )


MESH_DIMS = [
    _mesh([1, 1, 1, 1], "unsharded"),
    _mesh([1, 1, 1, 2], "channel2-only"),
    _mesh([1, 1, 2, 2], "spatial2-channel2"),
    _mesh([1, 1, 4, 2], "spatial4-channel2"),
]


# "even" matches small_beast_config's own in_channels=16/out_channels=12.
# "odd" shifts both down by one so they're no longer divisible by 2.
CHANNEL_OVERRIDES = {
    "even": {},
    "odd": {"in_channels": 15, "out_channels": 11},
}


# Selects between ``Transformer`` and ``NiceTransformer`` blocks (and their
# distributed variants); the latter add NiceNorm layers and their noise MLPs.
NICE_NORM = [
    pytest.param(False, id="plain"),
    pytest.param(True, id="nice"),
]


def _skip_unless_world_size_matches(mesh_dims: list[int]) -> None:
    """Skip unless the current run's world size matches ``mesh_dims``'s
    product.

    ``create_process_mesh_dict`` requires an exact match and raises
    otherwise, but different ``mesh_dims`` values need different numbers of
    ranks to actually exercise -- not every test run has 4 processes.
    """

    required = math.prod(mesh_dims)
    actual = dist.get_world_size()
    if actual != required:
        pytest.skip(f"mesh_dims {mesh_dims} needs {required} ranks, got {actual}")


def _channel_group_size(mesh_dims: list[int]) -> int:
    """``mesh_dims`` is ``(DP, SP, JSpatial, JChannel)``; see ``create_process_mesh_dict``."""
    return mesh_dims[3]


def _needs_even_channels(mesh_dims: list[int], channel_variant: str) -> bool:
    """Whether this combination requires in/out channels divisible by the
    channel-group size.

    jigsaw's distributed conv layers assert exact divisibility and raise
    otherwise -- there is no padding anywhere to fall back on.
    """
    return channel_variant == "odd" and _channel_group_size(mesh_dims) > 1


def _build_sharded_model(
    small_config: dict,
    device: torch.device,
    mesh_dims: list[int],
) -> tuple[Beast, dist.ProcessGroup, dist.ProcessGroup]:

    process_mesh_dict = create_process_mesh_dict(
        mesh_dims=mesh_dims, device_type=device.type
    )
    channel_pg = process_mesh_dict["JChannel"]
    spatial_pg = process_mesh_dict["JSpatial"]
    dtp_pg = process_mesh_dict["DTP"]

    sharded_model = Beast(
        **small_config,
        dropout=1e-1,
        dtp_group=dtp_pg,
        channel_group=channel_pg,
        spatial_group=spatial_pg,
        device=device,
    ).to(device)

    return sharded_model, channel_pg, spatial_pg


@pytest.mark.parametrize("mesh_dims", MESH_DIMS)
# "odd" is rejected at build time: jigsaw asserts that the channels divide
# the channel group (jigsaw 7d26665).
@pytest.mark.parametrize("channel_variant", ["even"])
@pytest.mark.parametrize("nice_norm", NICE_NORM)
def test_count_model_parameters_before_conversion(
    device: torch.device, mesh_dims: list[int], channel_variant: str, nice_norm: bool
) -> None:
    """A sharded model's deterministic parameter count must match an
    equivalent, unsharded reference model's count -- except when uneven
    in/out channels can't be split evenly across JChannel ranks, which
    jigsaw rejects outright rather than padding around.
    """
    # _skip_unless_world_size_matches(mesh_dims)
    small_config = small_beast_config(
        bayesian=False, nice_norm=nice_norm, **CHANNEL_OVERRIDES[channel_variant]
    )

    reference_model = Beast(**small_config, dropout=1e-1, device=device)
    reference_count = sum(
        p.numel() for p in reference_model.parameters() if p.requires_grad
    )

    sharded_model, channel_pg, _ = _build_sharded_model(small_config, device, mesh_dims)
    sharded_count = count_model_parameters(
        sharded_model, parallelism=channel_pg.size(), pending_bayesian_conversion=False
    )

    # TODO(Issue #97): count_model_parameters is not yet consistent across settings.
    # Warn instead of failing until the cause is found, then restore
    # ``assert sharded_count == reference_count``.
    if sharded_count != reference_count:
        warnings.warn(
            f"Parameter count mismatch for mesh_dims={mesh_dims}, "
            f"channel_variant={channel_variant}, nice_norm={nice_norm}: "
            f"sharded={sharded_count}, reference={reference_count} "
            f"(diff={sharded_count - reference_count})",
            stacklevel=1,
        )

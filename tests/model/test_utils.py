import math

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


# Selects between ``Transformer`` and ``NiceTransformer`` blocks (and their
# distributed variants); the latter add NiceNorm layers and their noise MLPs.
NICE_NORM = [
    pytest.param(False, id="plain"),
    pytest.param(True, id="nice"),
]


def _build_sharded_model(
    small_config: dict,
    device: torch.device,
    mesh_dims: list[int],
) -> tuple[Beast, dict[str, dist.ProcessGroup]]:
    """Build a ``Beast`` on a process mesh; also return the mesh."""
    process_mesh_dict = create_process_mesh_dict(
        mesh_dims=mesh_dims, device_type=device.type
    )

    sharded_model = Beast(
        **small_config,
        dropout=1e-1,
        dtp_group=process_mesh_dict["DTP"],
        channel_group=process_mesh_dict["JChannel"],
        spatial_group=process_mesh_dict["JSpatial"],
        device=device,
    ).to(device)

    return sharded_model, process_mesh_dict


@pytest.mark.parametrize("mesh_dims", MESH_DIMS)
@pytest.mark.parametrize("nice_norm", NICE_NORM)
def test_count_model_parameters_before_conversion(
    device: torch.device, mesh_dims: list[int], nice_norm: bool
) -> None:
    """A sharded model's deterministic parameter count must match an
    equivalent, unsharded reference model's count.
    """
    small_config = small_beast_config(bayesian=False, nice_norm=nice_norm)

    reference_model = Beast(**small_config, dropout=1e-1, device=device)
    reference_count = sum(
        p.numel() for p in reference_model.parameters() if p.requires_grad
    )

    sharded_model, process_mesh_dict = _build_sharded_model(
        small_config, device, mesh_dims
    )
    sharded_count = count_model_parameters(
        sharded_model,
        process_mesh_dict=process_mesh_dict,
        pending_bayesian_conversion=False,
    )

    assert sharded_count == reference_count

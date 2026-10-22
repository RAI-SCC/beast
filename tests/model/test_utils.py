import math
import warnings
from typing import Any

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


# The combinations used in practice: Bayesian models use NICE normalization
# (``NiceTransformer`` blocks and their noise MLPs), deterministic ones do not.
MODEL_VARIANTS = [
    pytest.param({"bayesian": False, "nice_norm": False}, id="deterministic-plain"),
    pytest.param({"bayesian": True, "nice_norm": True}, id="bayesian-nice"),
]


QK_NORM = [
    pytest.param(False, id="no-qk-norm"),
    pytest.param(True, id="qk-norm"),
]


# Attributes of ``Beast`` that hold a single layer. The rolls have no
# parameters; they are included to catch one appearing unexpectedly.
SINGLE_LAYERS = (
    "patch_embedding",
    "downsample",
    "upsample",
    "upsample_mix",
    "patch_recovery",
    "roll_out",
    "unroll_out",
    "roll_in",
    "unroll_in",
)

# Attributes of ``Beast`` that hold a ``ModuleList`` of transformer blocks.
TRANSFORMER_LISTS = ("transformer_1", "transformer_down", "transformer_2")


def _build_sharded_model(
    small_config: dict[str, Any],
    device: torch.device,
    mesh_dims: list[int],
) -> tuple[Beast, dict[str, dist.ProcessGroup]]:
    """Build a ``Beast`` on a process mesh; also return the mesh."""
    process_mesh_dict = create_process_mesh_dict(
        mesh_dims=mesh_dims, device_type=device.type
    )
    channel_pg = process_mesh_dict["JChannel"]

    sharded_model = Beast(
        **small_config,
        dropout=1e-1,
        dtp_group=process_mesh_dict["DTP"],
        channel_group=channel_pg,
        spatial_group=process_mesh_dict["JSpatial"],
        device=device,
    ).to(device)

    return sharded_model, process_mesh_dict


def _layer_modules(model: Beast) -> dict[str, torch.nn.Module | None]:
    """Collect the layers of ``model`` by name.

    Transformer blocks are listed individually as ``transformer_1[0]`` etc.
    A layer the model lacks maps to ``None``.
    """
    layers: dict[str, torch.nn.Module | None] = {
        name: getattr(model, name, None) for name in SINGLE_LAYERS
    }
    for list_name in TRANSFORMER_LISTS:
        for i, block in enumerate(getattr(model, list_name, [])):
            layers[f"{list_name}[{i}]"] = block
    return layers


def _reference_layer_count(layer: torch.nn.Module) -> int:
    """Number of trainable parameters of an unsharded layer."""
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)


def _sharded_layer_count(
    layer: torch.nn.Module, process_mesh_dict: dict[str, dist.ProcessGroup]
) -> int:
    """``count_model_parameters`` of a single sharded layer."""
    return count_model_parameters(
        layer,
        process_mesh_dict=process_mesh_dict,
        pending_bayesian_conversion=False,
    )


def _layer_mismatches(
    reference_layers: dict[str, torch.nn.Module | None],
    sharded_layers: dict[str, torch.nn.Module | None],
    process_mesh_dict: dict[str, dist.ProcessGroup],
) -> list[str]:
    """Describe every layer whose sharded count differs from the reference.

    A layer present in only one of the models counts as a mismatch.
    """
    mismatches = []
    for name in sorted(reference_layers.keys() | sharded_layers.keys()):
        reference_layer = reference_layers.get(name)
        sharded_layer = sharded_layers.get(name)
        if reference_layer is None or sharded_layer is None:
            missing_in = "reference" if reference_layer is None else "sharded"
            mismatches.append(f"  {name}: missing in {missing_in} model")
            continue

        reference_count = _reference_layer_count(reference_layer)
        sharded_count = _sharded_layer_count(sharded_layer, process_mesh_dict)
        if sharded_count != reference_count:
            mismatches.append(
                f"  {name} ({type(sharded_layer).__name__}): "
                f"sharded={sharded_count}, reference={reference_count} "
                f"(diff={sharded_count - reference_count})"
            )
    return mismatches


@pytest.mark.parametrize("mesh_dims", MESH_DIMS)
# "odd" is rejected at build time: jigsaw asserts that the channels divide
# the channel group.
@pytest.mark.parametrize("channel_variant", ["even"])
@pytest.mark.parametrize("variant", MODEL_VARIANTS)
@pytest.mark.parametrize("qk_norm", QK_NORM)
def test_count_model_parameters_per_layer(
    device: torch.device,
    mesh_dims: list[int],
    channel_variant: str,
    variant: dict[str, bool],
    qk_norm: bool,
) -> None:
    """Every layer of a sharded model must count the same as its unsharded
    counterpart.

    All layers are compared before reporting, so a single run lists every
    mismatching layer. On a mismatch, rank 0 warns with the per-layer
    mismatches and the overall counts, and the test fails.
    """
    small_config = small_beast_config(
        **variant, qk_norm=qk_norm, **CHANNEL_OVERRIDES[channel_variant]
    )

    reference_model = Beast(**small_config, dropout=1e-1, device=device)
    sharded_model, process_mesh_dict = _build_sharded_model(
        small_config, device, mesh_dims
    )

    reference_layers = _layer_modules(reference_model)
    sharded_layers = _layer_modules(sharded_model)
    mismatches = _layer_mismatches(reference_layers, sharded_layers, process_mesh_dict)

    total_reference = _reference_layer_count(reference_model)
    total_sharded = count_model_parameters(
        sharded_model,
        process_mesh_dict=process_mesh_dict,
        pending_bayesian_conversion=False,
    )

    # Only rank 0 reports, so the output of the ranks does not interleave.
    comm = MPI.COMM_WORLD
    comm.Barrier()
    mismatched = bool(mismatches) or total_sharded != total_reference
    if mismatched and comm.Get_rank() == 0:
        warnings.warn(
            f"Parameter count mismatch for mesh_dims={mesh_dims}, "
            f"channel_variant={channel_variant}, {variant}, qk_norm={qk_norm}; "
            f"{len(mismatches)} of {len(reference_layers | sharded_layers)} "
            "layers differ:\n"
            + "\n".join(mismatches)
            + f"\n  overall: sharded={total_sharded}, "
            f"reference={total_reference} "
            f"(diff={total_sharded - total_reference})"
        )

    if mismatched:
        pytest.fail("Parameter count mismatch", pytrace=False)

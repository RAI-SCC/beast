from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from small_models import small_beast_config
from torch_blue.vi import convert_to_vimodule

from beast.comm import create_process_mesh_dict
from beast.model.beast_config import BEAST_CONFIGURATIONS
from beast.model.beast_model import Beast
from beast.model.utils import convert_jigsaw_vi
from beast.utils import set_all_seeds

pytestmark = pytest.mark.gpu


class TestBeastlingModel:
    @pytest.mark.parametrize("n_timesteps", [1, 2])
    @pytest.mark.parametrize("bayesian", [False])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_beastling(
        self, n_timesteps: int, bayesian: bool, batch_size: int, device: torch.device
    ) -> None:
        small_config = small_beast_config(bayesian=bayesian)
        small_config["in_channels"] = small_config["in_channels"] * n_timesteps
        model = Beast(**small_config, dropout=0 if bayesian else 1e-1, device=device)

        # Test a forward pass with dummy data
        with torch.no_grad():
            x = torch.rand(
                batch_size,
                small_config["in_channels"],
                small_config["height"],
                small_config["width"],
                device=device,
            )
            if bayesian:
                convert_to_vimodule(model, return_log_probs=True)
                y = model(x, samples=2)
                assert y.shape == (
                    2,
                    batch_size,
                    small_config["out_channels"],
                    small_config["height"],
                    small_config["width"],
                )
            else:
                y = model(x)
                assert y.shape == (
                    batch_size,
                    small_config["out_channels"],
                    small_config["height"],
                    small_config["width"],
                )


class TestDistBeastlingModel:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_timesteps", [1, 2])
    @pytest.mark.parametrize("bayesian", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_4way_beastling(
        self,
        n_timesteps: int,
        bayesian: bool,
        batch_size: int,
        device: torch.device,
    ) -> None:
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        jchannel_pg = process_mesh_dict["JChannel"]
        jspatial_pg = process_mesh_dict["JSpatial"]
        dtp_pg = process_mesh_dict["DTP"]

        small_config = small_beast_config(bayesian=bayesian)
        small_config["in_channels"] = small_config["in_channels"] * n_timesteps
        model = Beast(
            **small_config,
            dropout=0 if bayesian else 1e-1,
            dtp_group=dtp_pg,
            channel_group=jchannel_pg,
            spatial_group=jspatial_pg,
            device=device,
        ).to(device)

        # Test a forward pass with dummy data
        with torch.no_grad():
            x = torch.rand(
                batch_size,
                small_config["in_channels"] // jchannel_pg.size(),
                small_config["height"],
                small_config["width"] // jspatial_pg.size(),
                device=device,
            )
            if bayesian:
                convert_jigsaw_vi(
                    model, jchannel_pg, jspatial_pg, process_mesh_dict["SP"]
                )
                y = model(x, samples=2)
                assert y.shape == (
                    2,
                    batch_size,
                    small_config["out_channels"] // jchannel_pg.size(),
                    small_config["height"],
                    small_config["width"] // jspatial_pg.size(),
                )
            else:
                y = model(x)
                assert y.shape == (
                    batch_size,
                    small_config["out_channels"] // jchannel_pg.size(),
                    small_config["height"],
                    small_config["width"] // jspatial_pg.size(),
                )


@pytest.mark.skip("Model to large for Horeka")
@pytest.mark.gpu
class TestDistBeastModel:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("n_timesteps", [1, 2])
    @pytest.mark.parametrize("bayesian", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_4way_beast(
        self,
        n_timesteps: int,
        bayesian: bool,
        batch_size: int,
        device: torch.device,
    ) -> None:
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        jchannel_pg = process_mesh_dict["JChannel"]
        jspatial_pg = process_mesh_dict["JSpatial"]
        dtp_pg = process_mesh_dict["DTP"]

        large_config = BEAST_CONFIGURATIONS["BEAST-JUPITER-0.7B-1.8K"]["config"].copy()
        large_config["bayesian"] = bayesian
        large_config["in_channels"] = large_config["in_channels"] * n_timesteps
        model = Beast(
            **large_config,
            dropout=0 if bayesian else 1e-1,
            dtp_group=dtp_pg,
            channel_group=jchannel_pg,
            spatial_group=jspatial_pg,
            device=device,
        )

        # Test a forward pass with dummy data
        with torch.no_grad():
            x = torch.rand(
                batch_size,
                large_config["in_channels"] // jchannel_pg.size(),
                large_config["height"],
                large_config["width"] // jspatial_pg.size(),
                device=device,
            )
            if bayesian:
                convert_to_vimodule(model, return_log_probs=True)
                y = model(x, samples=2)
                assert y.shape == (
                    2,
                    batch_size,
                    large_config["out_channels"] // jchannel_pg.size(),
                    large_config["height"],
                    large_config["width"] // jspatial_pg.size(),
                )
            else:
                y = model(x)
                assert y.shape == (
                    batch_size,
                    large_config["out_channels"] // jchannel_pg.size(),
                    large_config["height"],
                    large_config["width"] // jspatial_pg.size(),
                )


def _gather_across(
    model: torch.nn.Module, group: dist.ProcessGroup
) -> Iterator[tuple[str, list[torch.Tensor]]]:
    """Yield each parameter of ``model`` gathered across ``group``, with its name."""
    for name, parameter in model.named_parameters():
        local = parameter.detach().contiguous()
        buffers = [torch.empty_like(local) for _ in range(group.size())]
        dist.all_gather(buffers, local, group=group)
        yield name, buffers


@pytest.mark.gpu
class TestBeastlingInit:
    """Which parameters are replicated across the mesh, and which are sharded.

    ``JSpatial`` is domain parallelism: every rank holds the whole model and
    the modules broadcast their parameters over that group, so all parameters
    must be bit-identical along it.

    ``JChannel`` is tensor parallelism: ranks own different shards, so randomly
    initialised parameters must differ along it. Constant initialisations
    (LayerNorm ones/zeros, zeroed biases) are equal on every rank by
    construction and carry no information, so they are excluded rather than
    asserted on.

    Note this only covers *initialisation*. A parameter that is replicated in
    shape but never synchronised will still start out consistent here and only
    diverge once training applies differing per-rank gradients; catching that
    needs an optimizer-step test, not this one.
    """

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bayesian", [False, True])
    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_init_sharing(
        self, qk_norm: bool, bayesian: bool, device: torch.device
    ) -> None:
        process_mesh_dict = create_process_mesh_dict(mesh_dims=[1, 1, 2, 2])
        jchannel_pg = process_mesh_dict["JChannel"]
        jspatial_pg = process_mesh_dict["JSpatial"]
        dtp_pg = process_mesh_dict["DTP"]

        # Seed differently on every rank on purpose. Replicated parameters have
        # to end up identical because the modules broadcast them, not because
        # the ranks happened to draw the same random numbers.
        set_all_seeds(1000 + dist.get_rank())

        config = small_beast_config(qk_norm=qk_norm, bayesian=bayesian)
        model = Beast(
            **config,
            dropout=0 if bayesian else 1e-1,
            dtp_group=dtp_pg,
            channel_group=jchannel_pg,
            spatial_group=jspatial_pg,
            device=device,
        ).to(device)
        if bayesian:
            convert_jigsaw_vi(model, jchannel_pg, jspatial_pg, process_mesh_dict["SP"])

        # --- JSpatial: everything replicated -------------------------------
        not_replicated = [
            name
            for name, shards in _gather_across(model, jspatial_pg)
            if not all(torch.equal(shards[0], shard) for shard in shards[1:])
        ]
        assert not not_replicated, (
            "parameters differ across JSpatial despite the spatial broadcast, "
            f"so ranks start from different weights: {not_replicated}"
        )

        # --- JChannel: randomly initialised parameters are sharded ----------
        not_sharded = []
        for name, shards in _gather_across(model, jchannel_pg):
            # A constant tensor is the same on every rank whether it is sharded
            # or not, so it cannot distinguish the two.
            if shards[0].numel() == 0 or shards[0].unique().numel() == 1:
                continue
            if all(torch.equal(shards[0], shard) for shard in shards[1:]):
                not_sharded.append(name)

        assert not not_sharded, (
            "parameters are identical across JChannel although randomly "
            "initialised, i.e. replicated without any group synchronising "
            f"them: {not_sharded}"
        )

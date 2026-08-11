"""Gradient-reduction semantics across the process mesh.

These tests assert an *invariance*: for a fixed amount of global work, the
gradient of the loss with respect to the model parameters must not depend on
how that work is distributed over the mesh. Splitting an ensemble across the
``SP`` axis, or a domain across the ``JSpatial`` axis, changes which collective
runs where -- it must not change ``dL/dtheta``.

This is deliberately *not* a per-layer test. Every layer is correct in
isolation; the failure mode this file targets lives in the composition of three
things that are individually reasonable:

* ``beast.comm.create_process_mesh_dict`` -- which ranks land in the DDP group
* ``beast.losses.up_loss`` -- normalisation of the per-rank contribution
* the ``DistributedDataParallel`` wrap -- which *averages* over that group

Run with exactly 4 ranks::

    mpirun -n 4 python -m pytest tests/test_grad_reduction.py

Tolerances are tight and the dtype is float64 on purpose: what is under test is
a set of integer reduction factors, not floating point accuracy. A loose atol
would let a factor of two hide behind "close enough" at small rank counts.
"""

import pytest
import torch
import torch.distributed as dist
from jigsaw import hooks  # type: ignore[import-not-found]
from torch import Tensor

from beast.comm import (
    create_process_mesh_dict,
    get_process_group,
    shutdown_comm,
    wrap_ddp,
)
from beast.losses.up_loss import MeanStdPriorMatchingAggregation

# Global problem size, held fixed across every mesh layout below.
N_ENSEMBLE = 4
N_FEATURES = 8
DATASET_SIZE = 1000.0
TOL = 1e-10

_KNOWN_DEFECT = (
    "known gradient-reduction defect: DDP averages over SP and JSpatial, but "
    "the UP loss normalises each rank's contribution by the *global* ensemble "
    "count, so those partials must be summed. Remove this mark once the "
    "reduction is fixed."
)


class ToyUPModel(torch.nn.Module):
    """Minimal stand-in exercising both gradient paths of the UP loss.

    ``theta`` reaches the loss through the ensemble statistics, so its gradient
    must be *summed* across the ensemble axis. ``vmean``/``vlogs`` stand in for
    the variational parameters and reach the loss only through the prior
    matching term, which every rank computes identically, so their gradient
    must be *averaged*. Those two requirements are what a single process group
    with a single reduction op cannot satisfy simultaneously.

    Parameters
    ----------
    n_features : int
        Size of the (global) feature dimension.
    up_group : dist.ProcessGroup
        Process group over which ensemble members are distributed.
    device : torch.device
        Device to place parameters on.
    """

    def __init__(
        self, n_features: int, up_group: dist.ProcessGroup, device: torch.device
    ) -> None:
        super().__init__()
        self.up_group = up_group
        generator = torch.Generator(device="cpu").manual_seed(1)

        def parameter(scale: float, shift: float = 0.0) -> torch.nn.Parameter:
            values = torch.randn(n_features, generator=generator, dtype=torch.float64)
            return torch.nn.Parameter((values * scale + shift).to(device))

        self.theta = parameter(0.5, 2.0)
        self.vmean = parameter(0.1)
        self.vlogs = parameter(0.1)

    def forward(self, noise: Tensor, target: Tensor, features: slice) -> Tensor:
        """Return the total loss for this rank's ensemble members and domain shard.

        Parameters
        ----------
        noise : Tensor
            Shape ``(n_local_members, n_local_features)``. Distinct per ensemble
            member, so the members differ and the ensemble has spread.
        target : Tensor
            Shape ``(n_local_features,)``. Identical across the ensemble axis.
        features : slice
            This rank's shard of the global feature dimension.

        Returns
        -------
        Tensor
            Scalar loss: data fitting plus prior matching.
        """
        samples = self.theta[features].unsqueeze(0) * noise
        mean, std, prior_matching, _, _ = MeanStdPriorMatchingAggregation.apply(
            samples,
            self.up_group,
            torch.zeros_like(self.vmean),
            torch.zeros_like(self.vlogs),
            self.vmean,
            self.vlogs,
            None,
        )
        log_prob = torch.distributions.Normal(mean, std).log_prob(target)
        return -DATASET_SIZE * log_prob.sum() + prior_matching


def _global_problem(device: torch.device) -> tuple[Tensor, Tensor]:
    """Build the noise and target tensors, identical on every rank."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(
        N_ENSEMBLE, N_FEATURES, generator=generator, dtype=torch.float64
    )
    target = torch.randn(N_FEATURES, generator=generator, dtype=torch.float64)
    return noise.to(device), target.to(device)


@pytest.fixture(scope="session")
def serial_reference(device: torch.device) -> dict[str, Tensor]:
    """dL/dtheta with the whole problem on one rank -- the ground truth.

    The aggregation needs a process group, so this uses a single-rank group and
    no collective actually moves data. ``new_group`` is collective, so every
    rank builds the whole list in the same order and then selects its own.
    """
    solo_groups = [dist.new_group([rank]) for rank in range(dist.get_world_size())]
    noise, target = _global_problem(device)
    model = ToyUPModel(N_FEATURES, solo_groups[dist.get_rank()], device)
    model(noise, target, slice(None)).backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters()}


@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "mesh_dims",
    [
        pytest.param([1, 4, 1, 1, 1], id="DP=4"),
        pytest.param([1, 1, 4, 1, 1], id="SP=4", marks=pytest.mark.xfail(
            strict=True, reason=_KNOWN_DEFECT)),
        pytest.param([1, 1, 2, 2, 1], id="SP=2xJSpatial=2", marks=pytest.mark.xfail(
            strict=True, reason=_KNOWN_DEFECT)),
        pytest.param([1, 1, 1, 4, 1], id="JSpatial=4", marks=pytest.mark.xfail(
            strict=True, reason=_KNOWN_DEFECT)),
    ],
)
@pytest.mark.parametrize("spatial_hook", [True, False], ids=["hooked", "unhooked"])
def test_grad_invariant_to_mesh_layout(
    device: torch.device,
    serial_reference: dict[str, Tensor],
    mesh_dims: list[int],
    spatial_hook: bool,
) -> None:
    """dL/dtheta must not depend on how the mesh splits a fixed global problem.

    ``DP=4`` is the control: data-parallel averaging of an identical batch
    reproduces the serial gradient exactly, so a failure there means the
    harness itself is wrong rather than the reduction semantics.

    ``spatial_hook`` mirrors the fact that parameters inside jigsaw modules
    carry ``make_grad_sync_hook`` while others do not. The two classes fail
    differently, so both are covered: the hook's sum over ``JSpatial`` cancels
    against DDP's ``1/J`` for the data-fitting path, but leaves the prior
    matching term over-counted by ``J``.

    The xfail marks are ``strict``, so this test turns red once the reduction is
    fixed -- that is the signal to delete them.
    """
    try:
        create_process_mesh_dict(mesh_dims, device_type=device.type)
        up_group = get_process_group("SP")
        spatial_group = get_process_group("JSpatial")

        noise, target = _global_problem(device)
        n_local = N_ENSEMBLE // mesh_dims[2]
        width = N_FEATURES // mesh_dims[3]
        members = slice(up_group.rank() * n_local, (up_group.rank() + 1) * n_local)
        features = slice(
            spatial_group.rank() * width, (spatial_group.rank() + 1) * width
        )

        model = ToyUPModel(N_FEATURES, up_group, device)
        if spatial_hook:
            for parameter in model.parameters():
                parameter.register_post_accumulate_grad_hook(
                    hooks.make_grad_sync_hook(spatial_group, async_op=False)
                )

        ddp_model = wrap_ddp(
            model,
            device_id=device.index if device.type == "cuda" else None,
            ddp_group=get_process_group("DDP"),
        )
        ddp_model(noise[members, features], target[features], features).backward()

        failures = [
            f"{name}: |g|/|ref| = "
            f"{(p.grad.norm() / serial_reference[name].norm()).item():.6f}"
            for name, p in model.named_parameters()
            if not torch.allclose(p.grad, serial_reference[name], atol=TOL, rtol=TOL)
        ]
        assert not failures, (
            f"gradient depends on mesh layout {mesh_dims}: " + "; ".join(failures)
        )
    finally:
        shutdown_comm()

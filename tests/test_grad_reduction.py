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

from typing import cast

import pytest
import torch
import torch.distributed as dist
from jigsaw import hooks  # type: ignore[import-not-found]
from torch import Tensor

from beast.comm import create_process_mesh_dict, get_process_group, wrap_ddp
from beast.losses.up_loss import MeanStdPriorMatchingAggregation

# beast.comm.create_process_mesh_dict builds a 4-D mesh named
# (DP, SP, JSpatial, JChannel); these index into that layout. Only the two axes
# the reduction actually depends on are named.
_SP, _JSPATIAL = 1, 2

# Global problem size, held fixed across every mesh layout below.
N_ENSEMBLE = 4
N_FEATURES = 8
DATASET_SIZE = 1000.0
TOL = 1e-10

_KNOWN_DEFECT = (
    "known gradient-reduction defect: DDP averages over SP and JSpatial, but "
    "the UP loss normalises each rank's contribution by the *global* ensemble "
    "count, so those partials must be summed. Remove _is_known_defect and the "
    "marks it applies once the reduction is fixed."
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

    def forward(
        self,
        noise: Tensor,
        target: Tensor,
        features: slice,
        weight_synchronization_factor: int = 1,
    ) -> Tensor:
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
        weight_synchronization_factor: int, default: 1
            The size of the JSpatial group if the corresponding gradient hooks are enabled, one otherwise

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
            weight_synchronization_factor,
        )
        log_prob = torch.distributions.Normal(mean, std).log_prob(target)
        return cast(Tensor, -DATASET_SIZE * log_prob.sum() + prior_matching)


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
    return {
        name: cast(Tensor, p.grad).detach().clone()
        for name, p in model.named_parameters()
    }


# Building the mesh and running the distributed backward is collective and
# comparatively expensive, while each parameter's check is trivial. Computing
# both hook settings for a layout in one go and caching them lets every
# parameter be its own test while 24 tests cost 4 mesh builds rather than 24.
# Every rank walks the parametrisation in the same order, so the cache is
# populated identically everywhere.
#
# Teardown is handled by the autouse _reset_process_mesh fixture in
# conftest.py, so nothing here shuts the mesh down explicitly.
_GRAD_CACHE: dict[tuple[int, ...], dict[bool, dict[str, Tensor]]] = {}


def _distributed_grads(
    device: torch.device, mesh_dims: list[int], spatial_hook: bool
) -> dict[str, Tensor]:
    """Return post-DDP parameter gradients for one mesh layout and hook setting."""
    key = tuple(mesh_dims)
    if key in _GRAD_CACHE:
        return _GRAD_CACHE[key][spatial_hook]

    create_process_mesh_dict(mesh_dims, device_type=device.type)
    up_group = get_process_group("SP")
    spatial_group = get_process_group("JSpatial")
    ddp_group = get_process_group("DDP")

    noise, target = _global_problem(device)
    n_local = N_ENSEMBLE // mesh_dims[_SP]
    width = N_FEATURES // mesh_dims[_JSPATIAL]
    members = slice(up_group.rank() * n_local, (up_group.rank() + 1) * n_local)
    features = slice(spatial_group.rank() * width, (spatial_group.rank() + 1) * width)

    # Both hook settings share the mesh; each gets a freshly initialised model.
    per_hook: dict[bool, dict[str, Tensor]] = {}
    for hooked in (True, False):
        model = ToyUPModel(N_FEATURES, up_group, device)
        if hooked:
            for parameter in model.parameters():
                parameter.register_post_accumulate_grad_hook(
                    hooks.make_grad_sync_hook(spatial_group, async_op=False)
                )
        ddp_model = wrap_ddp(
            model,
            device_id=device.index if device.type == "cuda" else None,
            ddp_group=ddp_group,
        )
        ddp_model(
            noise[members, features],
            target[features],
            features,
            weight_synchronization_factor=(spatial_group.size() if hooked else 1),
        ).backward()
        per_hook[hooked] = {
            n: cast(Tensor, p.grad).detach().clone()
            for n, p in model.named_parameters()
        }

    _GRAD_CACHE[key] = per_hook
    return per_hook[spatial_hook]


def _is_known_defect(mesh_dims: list[int], spatial_hook: bool, param: str) -> bool:
    """Whether this (layout, hook, parameter) is broken by the current reduction.

    ``theta`` reaches the loss through the ensemble statistics and so needs its
    per-rank partials *summed* over both SP and JSpatial. DDP averages, and the
    jigsaw spatial hook supplies the sum over JSpatial only -- so theta is
    correct only when SP is trivial and either the hook covers JSpatial or
    JSpatial is trivial too.

    ``vmean``/``vlogs`` reach the loss only through prior matching, which every
    rank computes identically, so they need averaging -- which DDP already
    does. The spatial hook then over-counts them by a factor of JSpatial.
    """
    n_jspatial = mesh_dims[_JSPATIAL]
    if param == "theta":
        return not (spatial_hook or n_jspatial == 1)
    return False


@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "mesh_dims",
    [
        pytest.param([4, 1, 1, 1], id="DP=4"),
        pytest.param([1, 4, 1, 1], id="SP=4"),
        pytest.param([1, 2, 2, 1], id="SP=2xJSpatial=2"),
        pytest.param([1, 1, 4, 1], id="JSpatial=4"),
    ],
)
@pytest.mark.parametrize("spatial_hook", [True, False], ids=["hooked", "unhooked"])
@pytest.mark.parametrize("param", ["theta", "vmean", "vlogs"])
def test_grad_invariant_to_mesh_layout(
    request: pytest.FixtureRequest,
    device: torch.device,
    serial_reference: dict[str, Tensor],
    mesh_dims: list[int],
    spatial_hook: bool,
    param: str,
) -> None:
    """dL/dparam must not depend on how the mesh splits a fixed global problem.

    One test per parameter, so a layout that breaks one gradient but not
    another reports exactly that rather than collapsing to a single failure.
    ``DP=4`` is the control: data-parallel averaging of an identical batch
    reproduces the serial gradient, so a failure there means the harness is
    wrong rather than the reduction semantics.

    Combinations that the current reduction is known to get wrong are marked
    xfail with ``strict=True``, so fixing the reduction turns them into XPASS
    and this test goes red -- that is the signal to delete
    :func:`_is_known_defect` along with the marks.
    """
    if _is_known_defect(mesh_dims, spatial_hook, param):
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=_KNOWN_DEFECT))

    grads = _distributed_grads(device, mesh_dims, spatial_hook)
    actual, expected = grads[param], serial_reference[param]
    ratio = (actual.norm() / expected.norm()).item()

    assert torch.allclose(actual, expected, atol=TOL, rtol=TOL), (
        f"d/d{param} depends on mesh layout {mesh_dims} "
        f"(spatial_hook={spatial_hook}): |g|/|ref| = {ratio:.6f}"
    )

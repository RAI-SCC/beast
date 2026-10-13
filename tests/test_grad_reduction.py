"""Gradient-reduction semantics across the process mesh.

These tests assert an *invariance*: for a fixed amount of global work, the
gradient of the loss with respect to the model parameters must not depend on
how that work is distributed over the mesh. Splitting an ensemble across the
``SP`` axis, a domain across ``JSpatial`` or channels across ``JChannel``
changes which collective runs where -- it must not change ``dL/dtheta``.

The two terms of the UP loss need different reductions over a parameter's
replica axes (every non-DP axis on which ranks hold the same copy):

* data fitting: each rank holds a *partial* gradient (its own ensemble members
  and domain shard), which must be **summed**;
* prior matching: each replica computes the identical, full KL gradient, which
  must be **averaged**.

One process group with one reduction op cannot do both. The model-parallel
axes (``JSpatial``, ``JChannel``) are handled by jigsaw: the parameters are
marked replicated (``jigsaw.parallel.mark_replicated``) and only the
data-fitting path goes through ``jigsaw.parallel.sync_grad``, which sums it
over the replica axes (or, under the joint comm hook, scales it so that the
hook's average amounts to that sum). The prior-matching path reads the raw
parameters. ``SP`` is handled by the UP loss: its backward scales the sample
gradient by the ``SP`` size, so that DDP's average over ``DP`` x ``SP`` sums
the data fit over the ensemble members. DDP spans ``DP`` x ``SP``.

Run with exactly 4 ranks::

    mpirun -n 4 python -m pytest tests/test_grad_reduction.py

Tolerances are tight and the dtype is float64 on purpose: what is under test is
a set of integer reduction factors, not floating point accuracy. At realistic
``heat`` the KL's share of the gradient is small, so a mis-scaled KL shows up
as a relative error around 1e-5, not O(1).
"""

from collections.abc import Callable
from typing import cast

import pytest
import torch
import torch.distributed as dist
from jigsaw import parallel  # type: ignore[import-not-found]
from torch import Tensor

from beast.comm import create_process_mesh_dict, get_process_group, wrap_ddp
from beast.losses.up_loss import MeanStdPriorMatchingAggregation, UPAllGaussLoss
from beast.model.utils import convert_jigsaw_vi

# beast.comm.create_process_mesh_dict builds a 4-D mesh named
# (DP, SP, JSpatial, JChannel).
_SP, _JSPATIAL, _JCHANNEL = 1, 2, 3

# Global problem size, held fixed across every mesh layout below.
N_ENSEMBLE = 4
N_CHANNELS = 4
N_FEATURES = 8
DATASET_SIZE = 1000.0
HEAT = 0.1
TOL = 1e-10

LAYOUTS = [
    [4, 1, 1, 1],
    [1, 4, 1, 1],
    [1, 2, 2, 1],
    [1, 1, 4, 1],
    [2, 2, 1, 1],
    [2, 1, 2, 1],
    [1, 2, 1, 2],
    [1, 1, 2, 2],
]
MODES = ["no_ddp", "ddp", "joint_hook"]


def _layout_id(mesh_dims: list[int]) -> str:
    names = ("DP", "SP", "JSpatial", "JChannel")
    return "x".join(f"{n}={d}" for n, d in zip(names, mesh_dims) if d > 1)


# Replica axes of each toy parameter pair, by name suffix.
_REPLICA_AXES = {
    "": (_JSPATIAL, _JCHANNEL),
    "_ch": (_JSPATIAL,),
    "_sp": (_JCHANNEL,),
}


def _suffix(name: str) -> str:
    return next(s for s in ("_ch", "_sp") if name.endswith(s)) if "_" in name else ""


_SOLO: list[dist.ProcessGroup] = []


def _solo_group() -> dist.ProcessGroup:
    """Return a single-rank group of this rank.

    ``new_group`` is collective, so every rank builds every rank's group, in
    the same order, once.
    """
    if not _SOLO:
        groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
        _SOLO.append(groups[dist.get_rank()])
    return _SOLO[0]


def _shard(n: int, group: dist.ProcessGroup) -> slice:
    width = n // group.size()
    return slice(group.rank() * width, (group.rank() + 1) * width)


class ToyUPModel(torch.nn.Module):
    """Minimal stand-in for a Bayesian beast model under the UP loss.

    The output grid is ``(channel, feature)``: channels are split over
    ``JChannel`` and features (the spatial domain) over ``JSpatial``. Each pair
    of variational parameters both feeds the weight samples (data fitting) and
    enters the KL (prior matching), as in torch_blue, and each has a different
    replica set, so a layout with ``JChannel`` > 1 mixes them:

    * ``mu``/``logs``: one value per feature, replicated over JSpatial x
      JChannel; every spatial rank reads only its features.
    * ``mu_ch``/``logs_ch``: one value per channel, sharded over JChannel and
      replicated over JSpatial (a channel-sharded weight).
    * ``mu_sp``/``logs_sp``: one value per feature, sharded over JSpatial and
      replicated over JChannel.

    The ensemble members are split over SP, which is not a replica axis.

    Parameters
    ----------
    mesh_dims : list[int] | None
        Layout of the mesh, or None for the serial reference.
    device : torch.device
        Device to place parameters on.
    """

    def __init__(self, mesh_dims: list[int] | None, device: torch.device) -> None:
        super().__init__()
        if mesh_dims is None:
            self.up_group = _solo_group()
            self.channels, self.features = slice(None), slice(None)
            groups: dict[str, list[dist.ProcessGroup]] = {s: [] for s in _REPLICA_AXES}
        else:
            axes = {
                _SP: get_process_group("SP"),
                _JSPATIAL: get_process_group("JSpatial"),
                _JCHANNEL: get_process_group("JChannel"),
            }
            self.up_group = axes[_SP]
            self.channels = _shard(N_CHANNELS, axes[_JCHANNEL])
            self.features = _shard(N_FEATURES, axes[_JSPATIAL])
            groups = {s: [axes[a] for a in ax] for s, ax in _REPLICA_AXES.items()}

        generator = torch.Generator().manual_seed(1)

        def parameter(n: int, scale: float, shift: float, keep: slice) -> Tensor:
            values = torch.randn(n, generator=generator, dtype=torch.float64)
            return torch.nn.Parameter((values * scale + shift)[keep].to(device))

        self.mu = parameter(N_FEATURES, 0.5, 2.0, slice(None))
        self.logs = parameter(N_FEATURES, 0.1, -1.0, slice(None))
        self.mu_ch = parameter(N_CHANNELS, 0.3, 1.0, self.channels)
        self.logs_ch = parameter(N_CHANNELS, 0.1, -1.5, self.channels)
        self.mu_sp = parameter(N_FEATURES, 0.3, 0.5, self.features)
        self.logs_sp = parameter(N_FEATURES, 0.1, -1.2, self.features)
        for suffix, replica_groups in groups.items():
            for name in ("mu", "logs"):
                parallel.mark_replicated(getattr(self, name + suffix), replica_groups)

    def forward(
        self, noise: Tensor, target: Tensor, kl_through_sync: bool = False
    ) -> Tensor:
        """Return this rank's total loss.

        Parameters
        ----------
        noise : Tensor
            Shape ``(3, n_local_members, n_local_channels, n_local_features)``.
        target : Tensor
            Shape ``(n_local_channels, n_local_features)``.
        kl_through_sync : bool
            Negative control: route the KL through ``sync_grad`` too, which
            sums its replicated gradient instead of averaging it.

        Returns
        -------
        Tensor
            Scalar loss: data fitting plus ``HEAT`` times prior matching.
        """
        # Data-fit path: what convert_jigsaw_vi's synced sampling does.
        synced = parallel.sync_grad(
            self.mu, self.logs, self.mu_ch, self.logs_ch, self.mu_sp, self.logs_sp
        )
        mu, logs, mu_ch, logs_ch, mu_sp, logs_sp = synced
        weight = mu[self.features] + logs[self.features].exp() * noise[0]
        weight_ch = mu_ch[:, None] + logs_ch.exp()[:, None] * noise[1]
        weight_sp = mu_sp + logs_sp.exp() * noise[2]
        samples = weight * weight_ch + weight_sp

        # Prior-matching path: the raw parameters, as _get_flat_params reads them.
        mu, logs, mu_ch, logs_ch, mu_sp, logs_sp = (
            synced
            if kl_through_sync
            else (
                self.mu,
                self.logs,
                self.mu_ch,
                self.logs_ch,
                self.mu_sp,
                self.logs_sp,
            )
        )
        means = torch.cat([mu, mu_ch, mu_sp])
        log_stds = torch.cat([logs, logs_ch, logs_sp])
        mean, std, prior_matching, _, _ = MeanStdPriorMatchingAggregation.apply(
            samples,
            self.up_group,
            torch.zeros_like(means),
            torch.zeros_like(log_stds),
            means,
            log_stds,
            None,
        )
        log_prob = torch.distributions.Normal(mean, std).log_prob(target)
        return cast(Tensor, -DATASET_SIZE * log_prob.sum() + HEAT * prior_matching)

    def local_slice(self, name: str) -> slice:
        """Return the part of the global parameter ``name`` this rank holds."""
        return {"": slice(None), "_ch": self.channels, "_sp": self.features}[
            _suffix(name)
        ]


def _global_problem(device: torch.device) -> tuple[Tensor, Tensor]:
    """Build the noise and target tensors, identical on every rank."""
    generator = torch.Generator().manual_seed(0)
    shape = (N_ENSEMBLE, N_CHANNELS, N_FEATURES)
    noise = torch.randn(3, *shape, generator=generator, dtype=torch.float64)
    target = torch.randn(*shape[1:], generator=generator, dtype=torch.float64)
    return noise.to(device), target.to(device)


@pytest.fixture(scope="session")
def serial_reference(device: torch.device) -> dict[str, Tensor]:
    """Parameter gradients with the whole problem on one rank: the ground truth."""
    noise, target = _global_problem(device)
    model = ToyUPModel(None, device)
    model(noise, target).backward()
    return {
        name: cast(Tensor, p.grad).detach().clone()
        for name, p in model.named_parameters()
    }


# Building the mesh and running the distributed backward is collective and
# comparatively expensive, while each check is cheap. Every rank walks the
# parametrisation in the same order, so the cache is filled identically
# everywhere. Teardown is handled by the autouse _reset_process_mesh fixture.
_ERROR_CACHE: dict[tuple[tuple[int, ...], str, bool], dict[str, float]] = {}


def _relative_errors(
    device: torch.device,
    mesh_dims: list[int],
    mode: str,
    kl_through_sync: bool,
    reference: dict[str, Tensor],
) -> dict[str, float]:
    """Return ``max|g - g_ref| / max|g_ref|`` per parameter, maximised over ranks."""
    if mode == "no_ddp" and mesh_dims[_SP] > 1:
        pytest.skip("SP is only summed by DDP's average")
    key = (tuple(mesh_dims), mode, kl_through_sync)
    if key in _ERROR_CACHE:
        return _ERROR_CACHE[key]

    create_process_mesh_dict(mesh_dims, device_type=device.type)
    model = ToyUPModel(mesh_dims, device)
    members = _shard(N_ENSEMBLE, model.up_group)
    noise, target = _global_problem(device)
    noise = noise[:, members, model.channels, model.features]
    target = target[model.channels, model.features]

    forward: Callable[..., Tensor] = model
    if mode != "no_ddp":
        forward = wrap_ddp(
            model,
            device_id=device.index if device.type == "cuda" else None,
            ddp_group=get_process_group("DDP"),
            joint_hook=mode == "joint_hook",
        )
    # Two steps: DDP's first iteration uses a single fused bucket, later ones
    # the rebuilt buckets.
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        forward(noise, target, kl_through_sync).backward()

    names, errors = [], []
    for name, p in model.named_parameters():
        expected = reference[name][model.local_slice(name)]
        diff = (cast(Tensor, p.grad) - expected).abs().max() / expected.abs().max()
        names.append(name)
        errors.append(diff.item())
    t = torch.tensor(errors, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)

    _ERROR_CACHE[key] = dict(zip(names, t.tolist()))
    return _ERROR_CACHE[key]


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_dims", LAYOUTS, ids=_layout_id)
@pytest.mark.parametrize("mode", MODES)
def test_grad_invariant_to_mesh_layout(
    device: torch.device,
    serial_reference: dict[str, Tensor],
    mesh_dims: list[int],
    mode: str,
) -> None:
    """dL/dparam must not depend on how the mesh splits a fixed global problem.

    ``DP=4`` is the control: data-parallel averaging of an identical batch
    reproduces the serial gradient, so a failure there means the harness is
    wrong rather than the reduction semantics. Every DP rank sees the same
    batch, so ``no_ddp`` (nothing averages over DP) is exact too, except
    with SP > 1, whose sum needs DDP.
    """
    errors = _relative_errors(device, mesh_dims, mode, False, serial_reference)
    bad = {name: e for name, e in errors.items() if not e < TOL}
    assert not bad, (
        f"gradients depend on mesh layout {mesh_dims} ({mode}): relative errors {bad}"
    )


@pytest.mark.dist(4)
@pytest.mark.parametrize("mesh_dims", LAYOUTS[1:], ids=_layout_id)
@pytest.mark.parametrize("mode", MODES)
def test_negative_control_kl_through_sync_grad(
    device: torch.device,
    serial_reference: dict[str, Tensor],
    mesh_dims: list[int],
    mode: str,
) -> None:
    """Routing the KL through ``sync_grad`` too must be caught.

    That sums the replicated KL gradient over the replica axes instead of
    averaging it, which scales it by the replica count. The harness must
    notice that for exactly the parameters with a replica axis of size > 1;
    if it did not, the test above would prove nothing.
    """
    errors = _relative_errors(device, mesh_dims, mode, True, serial_reference)
    replicated = {
        name
        for name in errors
        if any(mesh_dims[a] > 1 for a in _REPLICA_AXES[_suffix(name)])
    }
    caught = {name for name, e in errors.items() if e > TOL}
    assert caught == replicated, (
        f"on {mesh_dims} ({mode}) the mis-scaled KL should change exactly "
        f"{sorted(replicated)}; it changed {sorted(caught)} ({errors})"
    )


# --- torch_blue: the real sampling path, inside sampled_forward's vmap -------


class _Container(torch.nn.Module):
    """Holds a ``torch.nn.Linear``, so the outermost VI module has no weights."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 3, dtype=torch.float64, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.linear(x))


def _bayesian_container(
    device: torch.device,
    spatial_group: dist.ProcessGroup,
) -> torch.nn.Module:
    """A converted model whose linear layer is replicated over JSpatial.

    The spatial mark is set before the conversion, as beast's layers do, so it
    has to survive ``convert_jigsaw_vi`` replacing the parameters.
    """
    model = _Container(device)
    for p in model.linear.parameters():
        parallel.mark_replicated(p, spatial_group)
    convert_jigsaw_vi(model, _solo_group(), spatial_group)
    return model


@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "mesh_dims", [[2, 2, 1, 1], [1, 2, 2, 1], [2, 1, 2, 1]], ids=_layout_id
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fast_path", [True, False], ids=["fast_path", "sampled"])
def test_sampled_forward_sums_data_fit_only(
    device: torch.device, mesh_dims: list[int], mode: str, fast_path: bool
) -> None:
    """``convert_jigsaw_vi``'s synced sampling gives ``mean_DDP(sum_R g_DF + g_KL)``.

    Runs torch_blue's own ``sampled_forward`` (a ``vmap`` over the samples)
    and the UP loss on a converted ``torch.nn.Linear``. ``VILinear`` reads its
    variational parameters directly on its fast path (log probs off) and
    through ``sample_variable`` otherwise, so both are covered. The expectation
    comes from an unmarked copy run on the same samples: its data-fit gradient
    summed by hand over the replica axes R, plus its KL gradient once, then
    averaged over DP x SP by DDP. Both losses scale the data fit by the SP
    size, so that average sums it over SP.
    """
    create_process_mesh_dict(mesh_dims, device_type=device.type)
    up_group = get_process_group("SP")
    spatial_group = get_process_group("JSpatial")
    ddp_group = get_process_group("DDP")
    replicas = [g for g in (spatial_group,) if g.size() > 1]

    model = _bayesian_container(device, spatial_group)
    assert parallel.replica_groups(model.linear._weight_mean) == tuple(replicas)

    forward: Callable[..., Tensor] = model
    if mode != "no_ddp":
        forward = wrap_ddp(
            model,
            device_id=device.index if device.type == "cuda" else None,
            ddp_group=ddp_group,
            joint_hook=mode == "joint_hook",
        )
    # After the wrap, which broadcasts over DP.
    reference = _bayesian_container(device, _solo_group())
    reference.load_state_dict(model.state_dict())

    # The analytical KL loss switches log probs off on construction, so set
    # them only afterwards.
    loss, reference_loss = (
        UPAllGaussLoss(m, int(DATASET_SIZE), up_group=up_group, heat=HEAT)
        for m in (model, reference)
    )
    for m in (model, reference):
        m.return_log_probs = not fast_path
        assert m.linear._return_log_probs is not fast_path

    # Different data on every rank: each holds a different share of the work.
    generator = torch.Generator().manual_seed(100 + dist.get_rank())
    x = torch.randn(2, 5, generator=generator, dtype=torch.float64).to(device)
    target = torch.randn(2, 3, generator=generator, dtype=torch.float64).to(device)

    torch.manual_seed(7 + dist.get_rank())
    loss(forward(x, samples=3), target).backward()

    torch.manual_seed(7 + dist.get_rank())
    data_fit, prior_matching = reference_loss.split_forward(
        reference(x, samples=3), target
    )
    params = list(reference.parameters())
    g_df = torch.autograd.grad(data_fit, params, retain_graph=True)
    g_kl = torch.autograd.grad(HEAT * prior_matching, params)

    for (name, p), df, kl in zip(model.named_parameters(), g_df, g_kl):
        expected = df.clone()
        for group in replicas:
            dist.all_reduce(expected, group=group)
        expected += kl
        if mode != "no_ddp":
            dist.all_reduce(expected, group=ddp_group)
            expected /= ddp_group.size()
        torch.testing.assert_close(
            p.grad, expected, rtol=TOL, atol=TOL, msg=lambda m: f"{name}: {m}"
        )


@pytest.mark.dist(4)
def test_convert_rejects_parameters_not_replicated_over_space(
    device: torch.device,
) -> None:
    """With DDP off the spatial axis, an unmarked parameter would silently drift."""
    create_process_mesh_dict([1, 1, 4, 1], device_type=device.type)
    with pytest.raises(RuntimeError, match="not marked replicated over the spatial"):
        convert_jigsaw_vi(
            _Container(device),
            get_process_group("JChannel"),
            get_process_group("JSpatial"),
        )

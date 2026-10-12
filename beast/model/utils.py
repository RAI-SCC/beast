"""Utility functions for beast models."""

import types

import torch
import torch.distributed as dist
import torch_blue.vi as vi
import torch_blue.vi.distributions as vidist
from jigsaw import normalization, parallel  # type: ignore[import-not-found]
from jigsaw.modules.attention.attention import JIGSAW_TE_AVAILABLE
from torch._C._functorch import get_unwrapped

from beast.layers.nice_norm import DistributedNICENorm, NiceNorm

if JIGSAW_TE_AVAILABLE:
    import transformer_engine.pytorch as te


def _sample_variable_synced(self: vi.VIModule, variable: str) -> torch.Tensor | None:
    """Draw a sample like ``VIModule.sample_variable``, through ``parallel.sync_grad``.

    Only the sample sees the synced variational parameters, so only the data
    fitting gradient is summed over their replica groups. The prior matching
    term reads the raw parameters (``get_variational_parameters``, from the
    loss) and keeps its full, replicated gradient, which DDP then averages.
    """
    if self.variational_distribution[variable] is None:
        return None

    variational_parameters = parallel.sync_grad(
        *self.get_variational_parameters(variable)
    )
    sample = self.variational_distribution[variable].sample(*variational_parameters)

    if self.return_log_probs:
        log_probs = self.get_log_probs(sample, variable)
        try:
            log_probs = get_unwrapped(log_probs)
        except RuntimeError:
            pass
        self._log_probs[variable].append(log_probs)
    return sample


def _fast_forward_synced(self: vi.VIModule, input_: torch.Tensor) -> torch.Tensor:
    """Run a torch_blue fast path (e.g. ``VILinear``'s) through ``parallel.sync_grad``.

    The fast path reads the variational parameters directly instead of calling
    ``sample_variable``, so it gets the synced parameters swapped in for the
    call. Inside that call they carry no mark, so the re-entered fast path finds
    nothing left to sync. Must not be used on the outermost module, whose
    forward samples.
    """
    synced = parallel.synced_parameters(self)
    if not synced:
        out: torch.Tensor = type(self)._fast_forward(self, input_)  # type: ignore[attr-defined]
        return out
    out = torch.func.functional_call(self, synced, (input_,))
    return out


@torch.no_grad()
def convert_jigsaw_vi(
    module: torch.nn.Module,
    channel_pg: dist.ProcessGroup,
    spatial_pg: dist.ProcessGroup,
    up_pg: dist.ProcessGroup | None,
    variational_distribution: vidist.base.Distribution = vidist.Normal(std=0.15),
    prior: vidist.base.Distribution = vidist.Normal(),
    rescale_prior: bool = False,
    kaiming_initialization: bool = True,
    prior_initialization: bool = False,
    return_log_probs: bool = True,
) -> None:
    """Convert a PyTorch module to a VI module using torch_blue, with custom handling for Jigsaw's specific layers and normalization.

    Every parameter is then marked replicated (``jigsaw.parallel``) over the
    groups it was marked with before the conversion, plus ``up_pg``: ensemble
    members share their parameters, and each Uncertainty Parallelism rank only
    sees the gradient of its own members. Weight samples are drawn through
    ``parallel.sync_grad``, so that partial gradient is summed over the replica
    groups in the backward pass. Wrap the result in DDP over the data-parallel
    axis only (:func:`beast.comm.wrap_ddp`).

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to convert.
    channel_pg : dist.ProcessGroup
        The process group for channel parallelism.
    spatial_pg : dist.ProcessGroup
        The process group for spatial parallelism. Every parameter must be
        marked replicated over it once converted.
    up_pg : dist.ProcessGroup | None
        The process group for Uncertainty Parallelism (``SP``), or None if the
        ensemble is not distributed.
    variational_distribution : torch_blue.vi.distributions.Distribution, optional
        The variational distribution to use for the VI conversion, by default vidist.Normal(std=0.15)
    prior : torch_blue.vi.distributions.Distribution, optional
        The prior distribution to use for the VI conversion, by default vidist.Normal()
    rescale_prior : bool, optional
        Whether to rescale the prior based on the number of processes, by default False
    kaiming_initialization : bool, optional
        Whether to use Kaiming initialization for the variational parameters, by default True
    prior_initialization : bool, optional
        Whether to initialize the variational parameters based on the prior, by default False
    return_log_probs : bool, optional
        Whether the converted module should return log probabilities, by default True

    Raises
    ------
    RuntimeError
        If a parameter is not marked replicated over ``spatial_pg``. DDP no longer
        averages over the spatial axis, so its copies would drift apart.
    """
    # torch_blue.vi.distributions.Distribution._init_uniform = _init_uniform_override
    vi.utils.convert.ban_convert(normalization.DistributedLayerNorm)
    vi.utils.convert.ban_convert(NiceNorm)
    vi.utils.convert.ban_convert(DistributedNICENorm, ban_mode="submodule")
    vi.utils.convert.ban_convert(NiceNorm, ban_mode="submodule")
    if JIGSAW_TE_AVAILABLE:
        vi.utils.convert.ban_convert(te.DotProductAttention)
        vi.utils.convert.ban_convert(te.DotProductAttention, ban_mode="submodule")

    # The conversion replaces each parameter with fresh variational parameters,
    # which carry none of the replica marks of the tensor they replace.
    marks = {
        m: {name: parallel.replica_groups(p) for name, p in m._parameters.items()}
        for m in module.modules()
    }

    vi.convert_to_vimodule(
        module,
        variational_distribution=variational_distribution,
        prior=prior,
        rescale_prior=rescale_prior,
        kaiming_initialization=kaiming_initialization,
        prior_initialization=prior_initialization,
        return_log_probs=return_log_probs,
        keep_weights=False,
    )

    num_jchannel = dist.get_world_size(channel_pg)

    def new_cf_channel(
        self: torch.nn.Module, variable_shapes: dict[str, torch.Size] | None = None
    ) -> int:
        previous: int = self._old_calculate_fan_in(variable_shapes=variable_shapes)  # type: ignore
        return previous * num_jchannel

    fan_in_rescale_list = ["AVIDistributedLinear", "AVIConvTranspose2d"]

    for mname, m in module.named_modules():
        class_name = m.__class__.__name__

        if class_name in fan_in_rescale_list:
            m._old_calculate_fan_in = m._calculate_fan_in
            m._calculate_fan_in = types.MethodType(new_cf_channel, m)
            m.reset_variational_parameters()

        if isinstance(m, vi.VIModule) and m.random_variables is not None:
            for var in m.random_variables:
                if m.variational_distribution[var] is None:
                    continue
                for p in m.get_variational_parameters(var):
                    parallel.mark_replicated(p, marks.get(m, {}).get(var, ()))
            m.sample_variable = types.MethodType(_sample_variable_synced, m)  # type: ignore[method-assign]
            if hasattr(type(m), "_fast_forward"):
                m._fast_forward = types.MethodType(_fast_forward_synced, m)  # type: ignore[assignment]

    for p in module.parameters():
        parallel.mark_replicated(p, [up_pg], extend=True)

    if spatial_pg is not None and spatial_pg.size() > 1:
        spatial_ranks = dist.get_process_group_ranks(spatial_pg)
        unmarked = [
            name
            for name, p in module.named_parameters()
            if all(
                dist.get_process_group_ranks(g) != spatial_ranks
                for g in parallel.replica_groups(p)
            )
        ]
        if unmarked:
            raise RuntimeError(
                f"{len(unmarked)} parameter(s) are not marked replicated over the "
                f"spatial group, e.g. {unmarked[:5]}. Mark them with "
                "jigsaw.parallel.mark_replicated where they are created and route "
                "their module through jigsaw.parallel.synced_parameters."
            )

    # Conversion reinitialized the variational parameters on every rank.
    parallel.broadcast_replicas(module)


# TODO: How are parameters counted in torch_blue? Could this lead to double counting? Additionally, if we just pass a process group (e.g. channel parallel group), the size could be used, and it would lead to a more general method.
def count_model_parameters(
    model: torch.nn.Module, parallelism: int, bayesian: bool
) -> int:
    """
    Count the number of parameters in a model, taking into account parallelism and whether the model is Bayesian.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which to count parameters.
    parallelism : int
        The degree of parallelism.
    bayesian : bool
        Whether the model is Bayesian (which would double the parameter count).

    Returns
    -------
    int
        The total number of parameters in the model, adjusted for parallelism and Bayesian nature.
    """
    if parallelism == 1:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        param_sum = 0
        for module in model.modules():
            # Skip the top-level model itself if it has the method
            if module is model:
                continue
            if hasattr(module, "get_parameter_count"):
                param_sum += module.get_parameter_count()  # type: ignore

        total = param_sum * parallelism

    if bayesian:
        total = total * 2  # Count every parameter twice, if bayesian

    return total

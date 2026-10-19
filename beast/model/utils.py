"""Utility functions for beast models."""

import types

import torch
import torch.distributed as dist
import torch_blue.vi as vi
import torch_blue.vi.distributions as vidist
from jigsaw import parallel  # type: ignore[import-not-found]
from jigsaw.modules import DistributedLayerNorm  # type: ignore[import-not-found]
from torch._C._functorch import get_unwrapped

from beast.layers.nice_norm import DistributedNICENorm, NiceNorm


def _sample_variable_synced(self: vi.VIModule, variable: str) -> torch.Tensor | None:
    """Draw a sample like ``VIModule.sample_variable``, through ``parallel.sync_grad``.

    Only the sample sees the synced variational parameters, so only the data
    fitting gradient is summed over their replica groups. The prior matching
    term reads the raw parameters (``get_variational_parameters``, from the
    loss) and keeps its full, replicated gradient, which DDP then averages.

    Parameters
    ----------
    self : vi.VIModule
        The module the sample is drawn for.
    variable : str
        Name of the variable to sample.

    Returns
    -------
    torch.Tensor or None
        The sample, or ``None`` if the variable has no variational distribution.
    """
    if self.variational_distribution[variable] is None:
        return None

    variational_parameters = parallel.sync_grad(
        *self.get_variational_parameters(variable)
    )
    sample: torch.Tensor = self.variational_distribution[variable].sample(
        tuple(variational_parameters)
    )

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

    Parameters
    ----------
    self : vi.VIModule
        The module to run.
    input_ : torch.Tensor
        Input to the module.

    Returns
    -------
    torch.Tensor
        Output of the module's fast path.
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
    variational_distribution: vidist.base.Distribution = vidist.Normal(std=0.15),
    prior: vidist.base.Distribution = vidist.Normal(),
    rescale_prior: bool = False,
    kaiming_initialization: bool = True,
    prior_initialization: bool = False,
    return_log_probs: bool = True,
) -> None:
    """Make a module with jigsaw layers Bayesian, in place.

    Linear and convolution layers become torch_blue variational layers.
    Normalization layers keep deterministic weights. The variational parameters are replicated over ``spatial_pg``:
    they are broadcast from its first rank and their gradients are synchronized.
    The fan-in of the channel-sharded layers (``AVIDistributedXWT`` and
    ``AVIConvTranspose2d``) is scaled by the size of ``channel_pg``, so that
    they are initialized like the full layer.

    The remaining options are passed to :func:`torch_blue.vi.convert_to_vimodule`.

    Every parameter is then marked replicated (``jigsaw.parallel``) over the
    spatial group. Weight samples are drawn through ``parallel.sync_grad``, so
    that partial gradient is summed over the replica groups in the backward
    pass. Wrap the result in DDP over ``DP`` x ``SP`` (:func:`beast.comm.wrap_ddp`);
    the UP loss scales its data fit so that DDP's average sums it over ``SP``.

    Parameters
    ----------
    module : torch.nn.Module
        Module to convert.
    channel_pg : dist.ProcessGroup
        The ``JChannel`` process group.
    spatial_pg : dist.ProcessGroup
        The ``JSpatial`` process group.
    variational_distribution : torch_blue.vi.distributions.Distribution, default: Normal(std=0.15)
        Variational distribution of the converted weights.
    prior : torch_blue.vi.distributions.Distribution, default: Normal()
        Prior distribution of the converted weights.
    rescale_prior : bool, default: False
        Whether to rescale the prior.
    kaiming_initialization : bool, default: True
        Whether to use Kaiming initialization for the variational parameters.
    prior_initialization : bool, default: False
        Whether to initialize the variational parameters from the prior.
    return_log_probs : bool, default: True
        Whether the converted module returns log probabilities.
    """
    vi.utils.convert.ban_convert(DistributedLayerNorm)
    vi.utils.convert.ban_convert(NiceNorm)
    vi.utils.convert.ban_convert(DistributedNICENorm, ban_mode="submodule")
    vi.utils.convert.ban_convert(NiceNorm, ban_mode="submodule")

    # The conversion replaces each parameter with fresh variational parameters,
    # which do not contain the replica marks.
    marks = {
        m: {
            name: parallel.replica_groups(p)
            for name, p in m.named_parameters(recurse=False)
        }
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
        """Return the fan-in of the full layer, not just of the local shard.

        Parameters
        ----------
        self : torch.nn.Module
            The layer the method is bound to.
        variable_shapes : dict[str, torch.Size], optional
            Shapes of the variational parameters, passed to the original method.

        Returns
        -------
        int
            The fan-in of the local shard times the size of ``channel_pg``.
        """
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


def count_model_parameters(
    model: torch.nn.Module, parallelism: int, bayesian: bool
) -> int:
    """Count model parameters, scaled by parallelism and doubled if Bayesian.

    Parameters
    ----------
    model : torch.nn.Module
        Model to count. With ``parallelism > 1`` only submodules that provide
        ``get_parameter_count()`` contribute.
    parallelism : int
        The degree of parallelism.
    bayesian : bool
        Whether to double the count, for a mean and a spread per weight. A
        single-process model converted with torch_blue already holds both as
        separate parameters, so its count is doubled once more.

    Returns
    -------
    int
        Total parameter count.
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

"""Utility functions for beast models."""

import types

import jigsaw.hooks as hooks  # type: ignore[import-not-found]
import torch
import torch.distributed as dist
import torch_blue.vi as vi
import torch_blue.vi.distributions as vidist
from jigsaw import normalization  # type: ignore[import-not-found]
from jigsaw.modules.attention.attention import JIGSAW_TE_AVAILABLE

from beast.layers.nice_norm import DistributedNICENorm, NiceNorm

if JIGSAW_TE_AVAILABLE:
    import transformer_engine.pytorch as te


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
    """Convert a module to a torch_blue VI module in place, Jigsaw-aware.

    Normalization layers and Transformer Engine attention are excluded from the
    conversion. The fan-in of ``AVIDistributedXWT`` and ``AVIConvTranspose2d``
    layers is scaled by the channel-parallel size, and every variational
    parameter gets a gradient-sync hook over ``spatial_pg`` and is broadcast
    from its rank 0.

    The remaining options are passed through to
    :func:`torch_blue.vi.convert_to_vimodule`.

    Parameters
    ----------
    module : torch.nn.Module
        Module to convert.
    channel_pg : dist.ProcessGroup
        Channel-parallel process group.
    spatial_pg : dist.ProcessGroup
        Spatial-parallel process group.
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
    # torch_blue.vi.distributions.Distribution._init_uniform = _init_uniform_override
    vi.utils.convert.ban_convert(normalization.DistributedLayerNorm)
    vi.utils.convert.ban_convert(NiceNorm)
    vi.utils.convert.ban_convert(DistributedNICENorm, ban_mode="submodule")
    vi.utils.convert.ban_convert(NiceNorm, ban_mode="submodule")
    if JIGSAW_TE_AVAILABLE:
        vi.utils.convert.ban_convert(te.DotProductAttention)
        vi.utils.convert.ban_convert(te.DotProductAttention, ban_mode="submodule")

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
        """Return the fan-in of the layer scaled by the channel-parallel size."""
        previous: int = self._old_calculate_fan_in(variable_shapes=variable_shapes)  # type: ignore
        return previous * num_jchannel

    fan_in_rescale_list = ["AVIDistributedXWT", "AVIConvTranspose2d"]

    for mname, m in module.named_modules():
        class_name = m.__class__.__name__

        if class_name in fan_in_rescale_list:
            m._old_calculate_fan_in = m._calculate_fan_in
            m._calculate_fan_in = types.MethodType(new_cf_channel, m)
            m.reset_variational_parameters()

        if isinstance(m, vi.VIModule):
            for p in m.parameters(recurse=False):
                p.register_post_accumulate_grad_hook(
                    hooks.make_grad_sync_hook(spatial_pg)
                )
                dist.broadcast(p.data, group_src=0, group=spatial_pg)


# TODO: How are parameters counted in torch_blue? Could this lead to double counting? Additionally, if we just pass a process group (e.g. channel parallel group), the size could be used, and it would lead to a more general method.
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
        Whether to count each parameter twice (mean and spread).

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

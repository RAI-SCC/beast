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
    """Convert a PyTorch module to a VI module using torch_blue, with custom handling for Jigsaw's specific layers and normalization.

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch module to convert.
    channel_pg : dist.ProcessGroup
        The process group for channel parallelism.
    spatial_pg : dist.ProcessGroup
        The process group for spatial parallelism.
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

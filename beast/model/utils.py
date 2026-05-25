"""Utility functions for beast models."""

import types

import jigsaw.normalization as normalization  # type: ignore[import-not-found]
import jigsaw.utils_init  # type: ignore[import-not-found]
import torch
import torch.distributed as dist
import torch_blue.vi.distributions as vidist
import torch_blue.vi.utils.convert

from beast.layers.nice_window_transformer import NiceNorm


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
    torch_blue.vi.utils.convert.ban_convert(normalization.DistributedLayerNorm)
    torch_blue.vi.utils.convert.ban_convert(NiceNorm)
    torch_blue.vi.utils.convert.ban_convert(
        normalization.DistributedNICENorm, ban_mode="submodule"
    )
    torch_blue.vi.utils.convert.ban_convert(NiceNorm, ban_mode="submodule")

    torch_blue.vi.convert_to_vimodule(
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

    fan_in_rescale_list = ["AVIDistributedXWTSync", "AVIConvTranspose2d"]

    weight_sync_list = [
        "VIConv2d",
        "AVIConvTranspose2d",
        "AVIDistributedXWTSync",
        "AVIDistributedXWT_Nice",
    ]

    for mname, m in module.named_modules():
        class_name = m.__class__.__name__

        if class_name in fan_in_rescale_list:
            m._old_calculate_fan_in = m._calculate_fan_in
            m._calculate_fan_in = types.MethodType(new_cf_channel, m)
            m.reset_variational_parameters()

        if class_name in weight_sync_list:
            jigsaw.utils_init.average_weights_module(m, spatial_pg)

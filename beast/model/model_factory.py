"""Factory for creating model instances based on configuration."""

import torch

# from beast.model.model_gb import Bellbeast
# from beast.model.model_gb_tiny import TinyBellbeast
# from beast.model.expert import Expert
# from beast.utils import transplant_weights
# from beast.model.model_gb_tiny import NiceNorm, utils_init
# from beast.comm import get_process_group # to be removed but for now


def get_model() -> torch.nn.Module:
    """Create a model based on a string id and a configuration dictionary. The string id can be used to specify a particular model architecture, while the configuration dictionary can contain hyperparameters and other settings for the model."""
    raise NotImplementedError(
        "Model factory is not yet implemented. Please use the Beast class directly for now."
    )


# def count_model_parameters(
#     model: torch.nn.Module, parallelism: int, bayesian: bool
# ) -> int:
#     if parallelism == 1:
#         total = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     else:
#         param_sum = 0
#         for module in model.modules():
#             # Skip the top-level model itself if it has the method
#             if module is model:
#                 continue
#             if hasattr(module, "get_parameter_count"):
#                 param_sum += module.get_parameter_count()

#         total = param_sum * parallelism

#     if bayesian:
#         total = total * 2  # Count every parameter twice, if bayesian

#     return total


# def get_model(
#     *,
#     # comes from model in config
#     n_input_channels, #Will be overwritten internally to fit the data set specifications
#     embedding_dimension,
#     n_output_channels,
#     n_attn_heads,
#     patch_size,
#     noise_dim,
#     window_size_outer,
#     window_size_inner,
#     n_attn_blocks,
#     bayesian,
#     #latent_rollout, # not needed
#     nice_norm,
#     # comes from data in config
#     xlat,
#     xlon,
#     # comes from training in config
#     flash_attn,
#     # comes from jigsaw in config
#     parallelism,
#     # process group mesh
#     channel_group,
#     ep_group,
#     up_group,
#     dtp_group,
#     spatial_group,
#     ddp_group,
#     # this stuff passed down from training script or specified in here?
#     device,
#     dtype,
#     device_id,
#     # from config
#     transplant_from = None,
#     ) -> torch.nn.Module:

#     """
#     Load a certain model from a configuration.

#     Parameters
#     ----------
#     config : DictConfig
#         Configuration that is also passed to each component of the model.

#     Returns
#     -------
#     torch.nn.Module
#         Model instance, based on the provided configuration.
#     """

#     if xlat == 120 and xlon == 240:
#         model = TinyBellbeast(
#             xlat=xlat,
#             xlon=xlon,
#             in_channels=n_input_channels,
#             embed_dim=embedding_dimension,
#             out_channels=n_output_channels,
#             heads=n_attn_heads,  # Extend this to also configure Transformer hidden dimension; Currently embed_dim is kept
#             noise_dim=noise_dim,
#             kernel_size=patch_size,
#             window_size_outer=window_size_outer,
#             window_size_inner=window_size_inner,
#             n_full_blocks=n_attn_blocks,  # Change to List of ints of the numbers of blocks in the outer - inner - outer SWIN
#             device=device,
#             channel_group=channel_group,
#             ep_group=ep_group,
#             up_group=up_group,
#             bayesian=bayesian,
#             dtype=dtype,
#         )
#     ########### Grand Bellbeast
#     else:
#         model = Bellbeast(
#             xlat=xlat,
#             xlon=xlon,
#             in_channels=n_input_channels,
#             embed_dim=embedding_dimension,
#             out_channels=n_output_channels,
#             heads=n_attn_heads,  # Extend this to also configure Transformer hidden dimension; Currently embed_dim is kept
#             noise_dim=noise_dim,
#             kernel_size=patch_size,
#             window_size_outer=window_size_outer,
#             window_size_inner=window_size_inner,
#             n_full_blocks=n_attn_blocks,  # Extend this to also configure Transformer hidden dimension; Currently embed_dim is kept
#             dtp_rank=dtp_group.rank(),
#             device=device,
#             dtp_group=dtp_group,
#             spatial_group=spatial_group,
#             channel_group=channel_group,
#             ep_group=ep_group,
#             up_group=up_group,
#             bayesian=bayesian,
#             parallelism=parallelism,
#             dtype=dtype,
#             flash_attn=flash_attn,
#             nice_norm=nice_norm,
#         )
#     model = model.to(dtype).to(device)

#     # Make model expert
#     model = Expert(model, ep_group)
#     expert_loss_weighting = model.get_loss_weights()

#     # dist.barrier(device_ids=[device.index])

#     if bayesian:
#         # potentially remove
#         if transplant_from is not None:
#             model = transplant_weights(
#                 model,
#                 transplant_from,
#             )

#         convert_jigsaw_vi(model)
#         model.return_log_probs = False

#     model = DistributedDataParallel(
#         model,
#         device_ids=[device_id],
#         output_device=device_id,
#         process_group=ddp_group,
#         gradient_as_bucket_view=True,
#         static_graph=True,
#         bucket_cap_mb=50,
#     )
#     low_lr_params, normal_lr_params = [], []
#     low_key = ["patch_embedding", "downsample", "upsample", "patch_recovery"]
#     for name, param in model.named_parameters():
#         if (
#             low_key[0] in name
#             or low_key[1] in name
#             or low_key[2] in name
#             or low_key[3] in name
#         ):
#             low_lr_params.append(param)
#         else:
#             normal_lr_params.append(param)

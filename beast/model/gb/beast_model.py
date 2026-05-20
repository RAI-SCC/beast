"""Beast (Bayesian Exascale Atmospheric Swim Transformer) model definition."""

from typing import Sequence

import torch
import torch.distributed as dist
from jigsaw import attention, convolutions, linear

from beast.layers import NiceWindowedTransformerBlock


class Beast(torch.nn.Module):
    """Base class for the Beast model.
    Bayesian Exascale Atmospheric Swim Transformer.

    Parameters
    ----------
    height : int
        Height of the input data.
    width : int
        Width of the input data.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    embed_dim : int
        Embedding dimension for the model.
    heads : int
        Number of attention heads.
    kernel_size : int
        Kernel size for the convolutional layers.
    window_size_outer : int
        Window size for the outer attention blocks.
    window_size_inner : int
        Window size for the inner attention blocks.
    n_full_blocks : int
        Number of full attention blocks.
    noise_dim : int
        Dimension of the noise input for the Bayesian model.
    device : int
        Device to run the model on.
    parallelism : int
        Degree of parallelism for the model.
    dtp_group : dist.ProcessGroup
        Process group for data parallelism.
    spatial_group : dist.ProcessGroup
        Process group for spatial parallelism.
    channel_group : dist.ProcessGroup
        Process group for channel parallelism.
    ep_group : dist.ProcessGroup
        Process group for expert parallelism.
    up_group : dist.ProcessGroup
        Process group for unsharded parallelism.
    dtype : torch.dtype
        Data type for the model parameters.
    bayesian : bool
        Whether to use the Bayesian version of the model.
    flash_attn : bool
        Whether to use Flash Attention for the attention blocks.
    nice_norm : bool
        Whether to use the NICE normalization for the attention blocks.

    Notes
    -----
    TODO: Reference to papers and documentation
    """

    def __init__(
        self,
        *,
        # Input
        height: int,
        width: int,
        in_channels: int,
        # Output
        out_channels: int,
        # Model Architecture
        embed_dim: int,
        heads: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        # Windows
        window_size_outer: Sequence[int],
        window_size_inner: Sequence[int],
        n_full_blocks: int,
        # Noise
        noise_dim: int,
        # Dropout
        dropout: float = 1e-1,
        # Boolean flags
        bayesian: bool,
        flash_attn: bool,
        nice_norm: bool,
        # Device,
        device: torch.device | None,
        # Dtype
        dtype: torch.dtype = torch.float32,
        # Process Groups
        channel_group: dist.ProcessGroup | None = None,
        spatial_group: dist.ProcessGroup | None = None,
        dtp_group: dist.ProcessGroup | None = None,
        up_group: dist.ProcessGroup | None = None,
        ep_group: dist.ProcessGroup | None = None,
        **kwargs,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.heads = heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.window_size_outer = window_size_outer
        self.window_size_inner = window_size_inner
        self.n_full_blocks = n_full_blocks
        self.noise_dim = noise_dim
        self.dropout = dropout
        self.device = device
        self.channel_group = channel_group
        self.spatial_group = spatial_group
        self.dtp_group = dtp_group
        self.up_group = up_group
        self.ep_group = ep_group
        self.dtype = dtype
        self.bayesian = bayesian
        self.flash_attn = flash_attn
        self.nice_norm = nice_norm

        self._build_model()

    def _build_model(self) -> None:

        assert self.window_size_outer[0] % 2 == 0, (
            "window_outer[0] size must be divisible by 2"
        )
        assert self.window_size_outer[1] % 2 == 0, (
            "window_outer[1] size must be divisible by 2"
        )
        assert self.window_size_inner[0] % 2 == 0, (
            "window_inner[0] size must be divisible by 2"
        )
        assert self.window_size_inner[1] % 2 == 0, (
            "window_inner[1] size must be divisible by 2"
        )

        # Activation
        self.activation = torch.nn.GELU()
        self.mask_in, self.mask_out = None, None

        self.channel_group_size = (
            dist.get_world_size(group=self.channel_group)
            if self.channel_group is not None
            else 1
        )
        self.spatial_group_size = (
            dist.get_world_size(group=self.spatial_group)
            if self.spatial_group is not None
            else 1
        )
        self.dtp_group_size = (
            dist.get_world_size(group=self.dtp_group)
            if self.dtp_group is not None
            else 1
        )
        self.up_group_size = (
            dist.get_world_size(group=self.up_group) if self.up_group is not None else 1
        )

        if self.dtp_group_size > 1:
            self._parallelism = self.dtp_group_size
        elif self.channel_group_size > 1 and self.spatial_group_size > 1:
            self._parallelism = self.channel_group_size * self.spatial_group_size
        else:
            self._parallelism = 1

        self.local_rank = self.dtp_group.rank() if self.dtp_group is not None else 0

        # Patch embedding
        if self._parallelism > 1:
            self._build_dist_model()
        else:
            self._build_sequential_model()

        self.params = torch.nn.ModuleDict(
            {
                "low_lr": torch.nn.ModuleList(
                    [
                        self.patch_embedding,
                        self.downsample,
                        self.upsample,
                        self.upsample_mix,
                        self.patch_recovery,
                    ]
                ),
                "normal_lr": torch.nn.ModuleList(
                    [self.transformer_1, self.transformer_down, self.transformer_2]
                ),
            }
        )

    def _build_dist_model(self) -> None:

        self.patch_embedding = convolutions.DistributedConv2D(
            self.in_channels,
            self.embed_dim,  # global values
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            groups=1,
            bias=True,
            local_rank=self.local_rank,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            device=self.device,
            parallelism=self._parallelism,  # either 4 or 8 (# GPUs in node)
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        self.downsample = convolutions.DistributedConv2D(
            self.embed_dim,
            self.embed_dim * 2,  # global values
            2,
            2,
            self.padding,
            self.dilation,
            groups=1,
            local_rank=self.local_rank,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            device=self.device,
            parallelism=self._parallelism,
            bias=True,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        self.upsample = convolutions.DistributedConvTranspose2D(
            self.embed_dim * 2,
            self.embed_dim,  # global values
            2,
            2,
            self.padding,
            self.dilation,
            groups=1,
            local_rank=self.local_rank,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            device=self.device,
            parallelism=self._parallelism,
            bias=True,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        # each process would have weights: [embed //2 , embed]
        self.upsample_mix = linear.DistributedXWTSync(
            self.embed_dim // 2,
            self.embed_dim,
            bias=True,
            local_rank=self.local_rank,
            device=self.device,
            parallelism=2,
            model_group=self.channel_group,
            sync_group=self.spatial_group,
            bayesian=self.bayesian,
        )

        self.patch_recovery = convolutions.DistributedConvTranspose2D(
            self.embed_dim * 2,
            self.out_channels,  # global values
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            bias=True,
            groups=1,
            local_rank=self.local_rank,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
            device=self.device,
            parallelism=self._parallelism,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        # Transformer blocks
        if self.nice_norm:
            self.transformer_1 = torch.nn.ModuleList(
                [
                    attention.DistributedNICEWindowedTransformer(
                        x_features=self.embed_dim,
                        embed_dim=self.embed_dim,
                        output_dim=self.embed_dim,
                        heads=self.heads[0],
                        noise_dim=self.noise_dim,
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        parallelism=self.channel_group.size(),
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        dtype=self.dtype,
                        flash_attn=self.flash_attn,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[0])
                ]
            )

            self.transformer_down = torch.nn.ModuleList(
                [
                    attention.DistributedNICEWindowedTransformer(
                        self.embed_dim * 2,
                        self.embed_dim * 2,
                        self.embed_dim * 2,
                        heads=self.heads[1],
                        noise_dim=self.noise_dim,
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        parallelism=self.channel_group.size(),
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        dtype=self.dtype,
                        flash_attn=self.flash_attn,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[1])
                ]
            )

            self.transformer_2 = torch.nn.ModuleList(
                [
                    attention.DistributedNICEWindowedTransformer(
                        x_features=self.embed_dim,
                        embed_dim=self.embed_dim,
                        output_dim=self.embed_dim,
                        heads=self.heads[2],
                        noise_dim=self.noise_dim,
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        parallelism=self.channel_group.size(),
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        dtype=self.dtype,
                        flash_attn=self.flash_attn,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[2])
                ]
            )
        else:
            self.transformer_1 = torch.nn.ModuleList(
                [
                    attention.DistributedWindowedTransformer(
                        x_features=self.embed_dim,
                        embed_dim=self.embed_dim,
                        output_dim=self.embed_dim,
                        heads=self.heads[0],
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        parallelism=self.channel_group.size(),
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        dtype=self.dtype,
                        flash_attn=self.flash_attn,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[0])
                ]
            )

            self.transformer_down = torch.nn.ModuleList(
                [
                    attention.DistributedWindowedTransformer(
                        self.embed_dim * 2,
                        self.embed_dim * 2,
                        self.embed_dim * 2,
                        heads=self.heads[1],
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        parallelism=self.channel_group.size(),
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        dtype=self.dtype,
                        flash_attn=self.flash_attn,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[1])
                ]
            )

            self.transformer_2 = torch.nn.ModuleList(
                [
                    attention.DistributedWindowedTransformer(
                        x_features=self.embed_dim,
                        embed_dim=self.embed_dim,
                        output_dim=self.embed_dim,
                        heads=self.heads[2],
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        parallelism=self.channel_group.size(),
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        dtype=self.dtype,
                        flash_attn=self.flash_attn,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[2])
                ]
            )

        self.roll_out = linear.DistributedRoll(
            shifts=(self.window_size_outer[0] // 2, self.window_size_outer[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self._parallelism,
            spatial_group=self.spatial_group,
            dtype=self.dtype,
        )

        self.unroll_out = linear.DistributedRoll(
            shifts=(
                -(self.window_size_outer[0] // 2),
                -(self.window_size_outer[1] // 2),
            ),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self._parallelism,
            spatial_group=self.spatial_group,
            dtype=self.dtype,
        )
        self.roll_in = linear.DistributedRoll(
            shifts=(self.window_size_inner[0] // 2, self.window_size_inner[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self._parallelism,
            spatial_group=self.spatial_group,
            dtype=self.dtype,
        )

        self.unroll_in = linear.DistributedRoll(
            shifts=(
                -(self.window_size_inner[0] // 2),
                -(self.window_size_inner[1] // 2),
            ),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self._parallelism,
            spatial_group=self.spatial_group,
            dtype=self.dtype,
        )

    def _build_sequential_model(self) -> None:

        self.patch_embedding = torch.nn.Conv2d(
            self.in_channels,
            self.embed_dim,  # global values
            self.kernel_size,
            self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        self.downsample = torch.nn.Conv2d(
            self.embed_dim,
            self.embed_dim * 2,  # global values
            2,
            2,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            device=self.device,
            bias=True,
            dtype=self.dtype,
        )

        self.upsample = torch.nn.ConvTranspose2d(
            self.embed_dim * 2,
            self.embed_dim,  # global values
            2,
            2,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            device=self.device,
            bias=True,
            dtype=self.dtype,
        )

        self.upsample_mix = torch.nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        self.patch_recovery = torch.nn.ConvTranspose2d(
            self.embed_dim * 2,
            self.out_channels,  # global values
            self.kernel_size,
            self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
            groups=1,
            device=self.device,
            dtype=self.dtype,
        )

        self.transformer_1 = torch.nn.ModuleList(
            [
                NiceWindowedTransformerBlock(
                    embed_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    heads=self.heads[0],
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # TODO Make this flag for bayesian vs non-bayesian
                    device=self.device,
                    dtype=self.dtype,
                ).to(self.device)
                for i in torch.arange(2 * self.n_full_blocks[0])
            ]
        )

        self.transformer_down = torch.nn.ModuleList(
            [
                NiceWindowedTransformerBlock(
                    self.embed_dim * 2,
                    self.embed_dim * 2,
                    heads=self.heads[1],
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # TODO Make this flag for bayesian vs non-bayesian
                    device=self.device,
                    dtype=self.dtype,
                ).to(self.device)
                for i in torch.arange(2 * self.n_full_blocks[1])
            ]
        )

        self.transformer_2 = torch.nn.ModuleList(
            [
                NiceWindowedTransformerBlock(
                    embed_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    heads=self.heads[2],
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # TODO Make this flag for bayesian vs non-bayesian
                    device=self.device,
                    dtype=self.dtype,
                ).to(self.device)
                for i in torch.arange(2 * self.n_full_blocks[2])
            ]
        )

        self.roll_out = lambda x: torch.roll(
            x,
            shifts=(self.window_size_outer[0] // 2, self.window_size_outer[1] // 2),
            dims=(-3, -2),
        )

        self.unroll_out = lambda x: torch.roll(
            x,
            shifts=(-self.window_size_outer[0] // 2, -self.window_size_outer[1] // 2),
            dims=(-3, -2),
        )

        self.roll_in = lambda x: torch.roll(
            x,
            shifts=(self.window_size_inner[0] // 2, self.window_size_inner[1] // 2),
            dims=(-3, -2),
        )

        self.unroll_in = lambda x: torch.roll(
            x,
            shifts=(-self.window_size_inner[0] // 2, -self.window_size_inner[1] // 2),
            dims=(-3, -2),
        )

    # @torch.compile
    def _get_windows(self, x, window_size):
        B, xlat, xlon, channels = x.shape
        x = x.reshape(
            B,
            xlat // window_size[0],
            window_size[0],
            xlon // window_size[1],
            window_size[1],
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, window_size[0] * window_size[1], channels)  # windows
        return x

    # @torch.compile
    def _reverse_windows(self, x, B, xlat, xlon, channels, window_size):
        x = x.reshape(
            B,
            xlat // window_size[0],
            xlon // window_size[1],
            window_size[0],
            window_size[1],
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, xlat, xlon, channels)
        return x

    def _gen_mask_(self, batch_dim, n_patch_lat, n_patch_lon, window_size):
        """
        Generate mask for attention mechanism when fields are rolled.

        n_patch_vert: int
            Number of patches in the vertical (first) dimension
        n_patch_lat: int
            Number of patches in the latitude (second) dimension
        n_patch_lon: int
            Number of patches in the longitude (third) dimension

        Returns
        -------
        attention_mask: Tensor
            of shape (n_windows, n_patch_in_window, n_patch_in_window)
            attention mask with value of 0 when attention should not be masked and -10000 when masked.
        """
        # Rolled mask
        # 1 n_patch_lat n_patch_lon 1
        img_mask = torch.zeros((1, n_patch_lat, n_patch_lon, 1), device=self.device)

        img_mask[:, slice(0, window_size[0] // 2)] = 1
        img_mask[:, slice(window_size[0] // 2, None)] = 2

        mask_windows = self._get_windows(
            img_mask, window_size
        )  # n_windows, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
        # (n_windows, window sequence length)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # (n_windows, window sequence length, window sequence length)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100000.0)

        # Generates masks once
        # This assumes constant batch dimension within a run (restarting a job is ok)
        return attention_mask

    def _define_masks(self, x_shape):
        B, C, lat, lon = x_shape
        self.mask_out = (
            self._gen_mask_(
                B,
                lat // self.kernel_size,
                lon // self.kernel_size,
                self.window_size_outer,
            )
            .unsqueeze(1)
            .detach()
            if self.mask_out is None
            else self.mask_out
        )

        self.mask_in = (
            self._gen_mask_(
                B,
                lat // self.kernel_size // 2,
                lon // self.kernel_size // 2,
                self.window_size_inner,
            )
            .unsqueeze(1)
            .detach()
            if self.mask_in is None
            else self.mask_in
        )
        return self.mask_out, self.mask_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Beast model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, height, width).
        """
        # noise has shape [(sample), batch, noise_dim]
        noise = (
            torch.randn(x.shape[0], self.noise_dim, device=self.device).to(self.dtype)
            / self.noise_dim
        )

        # Define masks once upon initialization (dependent on batch dimension)
        # Note: Masks do not currently work with torch flash.
        # self.mask_out, self.mask_in = self._define_masks(x.shape)

        # x has global shape (B, channels, xlat, xlon)
        x = self.patch_embedding(x)
        x_res = x.clone()
        B, channels, xlat, xlon = x.shape
        # TODO: fuse as x.permute(0, 2, 3, 1)
        x = x.reshape(B, channels, xlat * xlon).mT.view(B, xlat, xlon, channels)

        # attention
        for i in range(self.n_full_blocks[0]):
            x = self._get_windows(x, self.window_size_outer)
            # x has shape: [(S), B*n_windows, window_size_outer[0], window_size_outer[1], channels]
            n_windows = x.shape[0] // B
            # noise has shape [B, noise_dim] --> use interleave to make it [B*n_windows]
            noise_transformer = noise.repeat_interleave(n_windows, dim=0)
            if self.nice_norm:
                x = self.transformer_1[i * 2](x, noise_transformer, mask=None)
            else:
                x = self.transformer_1[i * 2](x, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            if self.nice_norm:
                x = self.transformer_1[i * 2 + 1](
                    x, noise_transformer, mask=self.mask_out
                )
            else:
                x = self.transformer_1[i * 2 + 1](x, mask=self.mask_out)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]

        x_res2 = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()
        # (B, channels, lat, lon)
        x = self.downsample(x)
        # (B, 2*channels, lat/2, lon/2)

        B, channels, xlat, xlon = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        for i in range(self.n_full_blocks[1]):
            x = self._get_windows(x, self.window_size_inner)
            n_windows = x.shape[0] // B
            noise_transformer_down = noise.repeat_interleave(n_windows, dim=0)
            if self.nice_norm:
                x = self.transformer_down[i * 2](x, noise_transformer_down, mask=None)
            else:
                x = self.transformer_down[i * 2](x, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_inner
            )
            x = self.roll_in(x)
            x = self._get_windows(x, self.window_size_inner)
            if self.nice_norm:
                x = self.transformer_down[i * 2 + 1](
                    x, noise_transformer_down, mask=self.mask_in
                )
            else:
                x = self.transformer_down[i * 2 + 1](x, mask=self.mask_in)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_inner
            )
            x = self.unroll_in(x)  # [B, lat, lon, channels]

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.upsample(x)

        B, channels, xlat, xlon = x.shape

        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).contiguous()
        x = self.activation(x)
        x = self.upsample_mix(x)
        x = x.reshape(B, xlat, xlon, channels)

        x = x + x_res2

        for i in range(self.n_full_blocks[2]):
            x = self._get_windows(x, self.window_size_outer)
            n_windows = x.shape[0] // B
            noise_transformer = noise.repeat_interleave(n_windows, dim=0)

            if self.nice_norm:
                x = self.transformer_2[i * 2](x, noise_transformer, mask=None)
            else:
                x = self.transformer_2[i * 2](x, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            if self.nice_norm:
                x = self.transformer_2[i * 2 + 1](
                    x, noise_transformer, mask=self.mask_out
                )
            else:
                x = self.transformer_2[i * 2 + 1](x, mask=self.mask_out)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]
        B, xlat, xlon, channels = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.patch_recovery(torch.cat((x, x_res), dim=-3))  # concat along channels

        return x

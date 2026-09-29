"""The BEAST (Bayesian Exascale Atmospheric Swin Transformer) model."""

from typing import Any, Sequence

import torch
import torch.distributed as dist
from jigsaw import hooks, linear, roll  # type: ignore
from jigsaw.modules.convolution import DistributedConv2D, DistributedConv2DTranspose

from beast.layers import transformer


class Beast(torch.nn.Module):
    """Bayesian Exascale Atmospheric Swin Transformer.

    The input is patch-embedded, passed through windowed transformer blocks at
    full resolution, downsampled by 2 for a second set of blocks, upsampled and
    passed through a third set, and finally recovered to the output channels
    from the concatenation with the patch embedding. Each set alternates plain
    and shifted windows.

    The model is sharded across the spatial and channel groups if ``dtp_group``
    spans more than one process. Its inputs and outputs are then local shards.

    Parameters
    ----------
    height : int
        Global height of the input, in grid points.
    width : int
        Global width of the input, in grid points.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    embed_dim : int
        Embedding dimension. Doubled after the downsampling.
    heads : Sequence[int]
        Number of attention heads in each of the three sets of blocks.
    kernel_size : int
        Kernel size of the patch embedding and recovery.
    stride : int
        Stride of the patch embedding and recovery.
    padding : int
        Padding of the convolutions.
    dilation : int
        Dilation of the convolutions.
    window_size_outer : Sequence[int]
        Attention window ``(lat, lon)`` in the first and third set of blocks.
        Entries must be even.
    window_size_inner : Sequence[int]
        Attention window ``(lat, lon)`` in the second set of blocks. Entries
        must be even.
    n_full_blocks : Sequence[int]
        Number of plain-and-shifted block pairs in each of the three sets.
    noise_dim : int
        Size of the noise vector for the NICE normalization.
    device : torch.device
        Device to place the model on.
    dropout : float, default: 0.1
        Dropout probability. Forced to 0 if ``bayesian``, and ignored by the
        non-distributed model, which uses a fixed 1e-3.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.
    bayesian : bool, default: False
        Whether the distributed layers are Bayesian.
    nice_norm : bool, default: False
        Whether to use NICE normalization in the transformer blocks.
    dtype : torch.dtype, default: torch.float32
        Data type of the model.
    channel_group : dist.ProcessGroup, optional
        Channel (tensor-parallel) process group.
    spatial_group : dist.ProcessGroup, optional
        Spatial (domain-parallel) process group.
    dtp_group : dist.ProcessGroup, optional
        Process group the model is sharded over (spatial x channel).
    up_group : dist.ProcessGroup, optional
        Uncertainty-parallel process group. Only stored on the model.
    **kwargs
        Ignored.
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
        heads: Sequence[int],
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        # Windows
        window_size_outer: Sequence[int],
        window_size_inner: Sequence[int],
        n_full_blocks: Sequence[int],
        # Noise
        noise_dim: int,
        # Device,
        device: torch.device,
        # Dropout
        dropout: float | None = None,
        # Norm
        qk_norm: bool = False,
        # Boolean flags
        bayesian: bool = False,
        nice_norm: bool = False,
        # Dtype
        dtype: torch.dtype = torch.float32,
        # Process Groups
        channel_group: dist.ProcessGroup | None = None,
        spatial_group: dist.ProcessGroup | None = None,
        dtp_group: dist.ProcessGroup | None = None,
        up_group: dist.ProcessGroup | None = None,
        **kwargs: Any,
    ) -> None:
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
        if bayesian and dropout:
            warnings.warn(
                f"dropout={dropout} is ignored for Bayesian models, using 0.0.",
                UserWarning,
            )
        self.dropout = 0.0 if bayesian else (dropout if dropout is not None else 1e-1)
        self.device = device
        self.channel_group = channel_group
        self.spatial_group = spatial_group
        self.dtp_group = dtp_group
        self.up_group = up_group
        self.dtype = dtype
        self.bayesian = bayesian
        self.qk_norm = qk_norm
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

        self._parallelism = self.dtp_group_size

        self.local_rank = self.dtp_group.rank() if self.dtp_group is not None else 0

        assert (
            self.width
            % (self.spatial_group_size * self.kernel_size * self.window_size_outer[1])
            == 0
        ), "Width of outer layer is not divisible by outer window size."
        assert (
            self.width
            % (
                self.spatial_group_size
                * 2
                * self.kernel_size
                * self.window_size_inner[1]
            )
            == 0
        ), "Width of inner layer is not divisible by outer window size."

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
        assert self.channel_group is not None, (
            "channel_group must not be None for distributed model"
        )
        assert self.spatial_group is not None, (
            "spatial_group must not be None for distributed model"
        )

        assert self.dtp_group is not None, (
            "dtp_group must not be None for distributed model"
        )

        self.patch_embedding: torch.nn.Module = DistributedConv2D(
            self.in_channels,
            self.embed_dim,  # global values
            self.kernel_size,
            self.device,
            self.spatial_group,
            self.channel_group,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=True,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        self.downsample: torch.nn.Module = DistributedConv2D(
            self.embed_dim,
            self.embed_dim * 2,  # global values
            2,
            self.device,
            self.spatial_group,
            self.channel_group,
            stride=2,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=True,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        self.upsample: torch.nn.Module = DistributedConv2DTranspose(
            self.embed_dim * 2,
            self.embed_dim,  # global values
            2,
            self.device,
            self.spatial_group,
            self.channel_group,
            stride=2,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=True,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        # each process would have weights: [embed //2 , embed]
        self.upsample_mix: torch.nn.Module = linear.DistributedXWT(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            bias=True,
            local_rank=self.local_rank,
            device=self.device,
            model_group=self.channel_group,
            row_group=self.spatial_group,
            column_group=self.channel_group,
            bayesian=self.bayesian,
        )

        self.patch_recovery: torch.nn.Module = DistributedConv2DTranspose(
            self.embed_dim * 2,
            self.out_channels,  # global values
            self.kernel_size,
            self.device,
            self.spatial_group,
            self.channel_group,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=True,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        # Transformer blocks
        if self.nice_norm:
            self.transformer_1 = torch.nn.ModuleList(
                [
                    transformer.DistributedNICETransformer(
                        in_dim=self.embed_dim,
                        embed_dim=self.embed_dim,
                        out_dim=self.embed_dim,
                        heads=self.heads[0],
                        noise_dim=self.noise_dim,
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        qk_norm=self.qk_norm,
                        dtype=self.dtype,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[0])
                ]
            )

            self.transformer_down = torch.nn.ModuleList(
                [
                    transformer.DistributedNICETransformer(
                        in_dim=self.embed_dim * 2,
                        embed_dim=self.embed_dim * 2,
                        out_dim=self.embed_dim * 2,
                        heads=self.heads[1],
                        noise_dim=self.noise_dim,
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        qk_norm=self.qk_norm,
                        dtype=self.dtype,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[1])
                ]
            )

            self.transformer_2 = torch.nn.ModuleList(
                [
                    transformer.DistributedNICETransformer(
                        in_dim=self.embed_dim,
                        embed_dim=self.embed_dim,
                        out_dim=self.embed_dim,
                        heads=self.heads[2],
                        noise_dim=self.noise_dim,
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        qk_norm=self.qk_norm,
                        dtype=self.dtype,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[2])
                ]
            )
        else:
            self.transformer_1 = torch.nn.ModuleList(
                [
                    transformer.DistributedTransformer(
                        in_dim=self.embed_dim,
                        embed_dim=self.embed_dim,
                        out_dim=self.embed_dim,
                        heads=self.heads[0],
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        qk_norm=self.qk_norm,
                        dtype=self.dtype,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[0])
                ]
            )

            self.transformer_down = torch.nn.ModuleList(
                [
                    transformer.DistributedTransformer(
                        in_dim=self.embed_dim * 2,
                        embed_dim=self.embed_dim * 2,
                        out_dim=self.embed_dim * 2,
                        heads=self.heads[1],
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        qk_norm=self.qk_norm,
                        dtype=self.dtype,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[1])
                ]
            )

            self.transformer_2 = torch.nn.ModuleList(
                [
                    transformer.DistributedTransformer(
                        in_dim=self.embed_dim,
                        embed_dim=self.embed_dim,
                        out_dim=self.embed_dim,
                        heads=self.heads[2],
                        dropout=self.dropout,  # dropout
                        local_rank=self.local_rank,
                        device=self.device,
                        model_group=self.dtp_group,
                        spatial_group=self.spatial_group,
                        channel_group=self.channel_group,
                        bayesian=self.bayesian,
                        qk_norm=self.qk_norm,
                        dtype=self.dtype,
                    ).to(self.device)
                    for _ in torch.arange(2 * self.n_full_blocks[2])
                ]
            )

        self.roll_out: torch.nn.Module = roll.DistributedRoll(
            shifts=(self.window_size_outer[0] // 2, self.window_size_outer[1] // 2),
            dims=(-3, -2),
            distributed_dims=[-2],
            spatial_groups=[self.spatial_group],
            device=self.device,
            dtype=self.dtype,
        )

        self.unroll_out: torch.nn.Module = roll.DistributedRoll(
            shifts=(
                -(self.window_size_outer[0] // 2),
                -(self.window_size_outer[1] // 2),
            ),
            dims=(-3, -2),
            distributed_dims=[-2],
            spatial_groups=[self.spatial_group],
            device=self.device,
            dtype=self.dtype,
        )
        self.roll_in: torch.nn.Module = roll.DistributedRoll(
            shifts=(self.window_size_inner[0] // 2, self.window_size_inner[1] // 2),
            dims=(-3, -2),
            distributed_dims=[-2],
            spatial_groups=[self.spatial_group],
            device=self.device,
            dtype=self.dtype,
        )

        self.unroll_in: torch.nn.Module = roll.DistributedRoll(
            shifts=(
                -(self.window_size_inner[0] // 2),
                -(self.window_size_inner[1] // 2),
            ),
            dims=(-3, -2),
            distributed_dims=[-2],
            spatial_groups=[self.spatial_group],
            device=self.device,
            dtype=self.dtype,
        )

        for p in self.upsample_mix.parameters():
            p.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)

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

        # Use normal transformer if `nice_norm` is set to false
        block_cls = (
            transformer.NiceTransformer
            if self.nice_norm
            else transformer.TransformerBlock
        )

        self.transformer_1 = torch.nn.ModuleList(
            [
                block_cls(
                    in_dim=self.embed_dim,
                    embed_dim=self.embed_dim,
                    out_dim=self.embed_dim,
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
                block_cls(
                    in_dim=self.embed_dim * 2,
                    embed_dim=self.embed_dim * 2,
                    out_dim=self.embed_dim * 2,
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
                block_cls(
                    in_dim=self.embed_dim,
                    embed_dim=self.embed_dim,
                    out_dim=self.embed_dim,
                    heads=self.heads[2],
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # TODO Make this flag for bayesian vs non-bayesian
                    device=self.device,
                    dtype=self.dtype,
                ).to(self.device)
                for i in torch.arange(2 * self.n_full_blocks[2])
            ]
        )

        self.roll_out = roll.Roll(
            shifts=(self.window_size_outer[0] // 2, self.window_size_outer[1] // 2),
            dims=(-3, -2),
        )
        self.unroll_out = roll.Roll(
            shifts=(-self.window_size_outer[0] // 2, -self.window_size_outer[1] // 2),
            dims=(-3, -2),
        )
        self.roll_in = roll.Roll(
            shifts=(self.window_size_inner[0] // 2, self.window_size_inner[1] // 2),
            dims=(-3, -2),
        )
        self.unroll_in = roll.Roll(
            shifts=(-self.window_size_inner[0] // 2, -self.window_size_inner[1] // 2),
            dims=(-3, -2),
        )

    def _get_windows(self, x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
        batch_size, xlat, xlon, channels = x.shape
        x = x.reshape(
            batch_size,
            xlat // window_size[0],
            window_size[0],
            xlon // window_size[1],
            window_size[1],
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, window_size[0] * window_size[1], channels)  # windows
        return x

    def _reverse_windows(
        self,
        x: torch.Tensor,
        batch_size: int,
        xlat: int,
        xlon: int,
        channels: int,
        window_size: Sequence[int],
    ) -> torch.Tensor:  # noqa: N803
        x = x.reshape(
            batch_size,
            xlat // window_size[0],
            xlon // window_size[1],
            window_size[0],
            window_size[1],
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(batch_size, xlat, xlon, channels)
        return x

    def _gen_mask_(
        self,
        batch_dim: int,
        n_patch_lat: int,
        n_patch_lon: int,
        window_size: Sequence[int],
    ) -> torch.Tensor:
        """Generate the attention mask for rolled fields.

        Parameters
        ----------
        batch_dim : int
            Unused.
        n_patch_lat : int
            Number of patches along latitude.
        n_patch_lon : int
            Number of patches along longitude.
        window_size : Sequence[int]
            Attention window ``(lat, lon)``.

        Returns
        -------
        torch.Tensor
            Mask of shape ``(n_windows, n_patch_in_window, n_patch_in_window)``,
            0 where attention is allowed and -100000 where it is masked.
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

    def _define_masks(self, x_shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, lat, lon = x_shape
        self.mask_out = (
            self._gen_mask_(
                batch_size,
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
                batch_size,
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
        """Predict the output fields from the input fields.

        Noise for the NICE normalization is sampled on every call.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(batch, out_channels, height, width)``.
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
        batch_size, channels, xlat, xlon = x.shape  # noqa: N806
        # TODO: fuse as x.permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1).contiguous()

        # attention
        for i in range(self.n_full_blocks[0]):
            x = self._get_windows(x, self.window_size_outer)
            # x has shape: [(S), B*n_windows, window_size_outer[0] * window_size_outer[1], channels]
            n_windows = x.shape[0] // batch_size
            # noise has shape [B, noise_dim] --> use interleave to make it [B*n_windows]
            noise_transformer = noise.repeat_interleave(n_windows, dim=0)
            if self.nice_norm:
                x = self.transformer_1[i * 2](x, noise_transformer)
            else:
                x = self.transformer_1[i * 2](x)
            x = self._reverse_windows(
                x, batch_size, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            if self.nice_norm:
                x = self.transformer_1[i * 2 + 1](
                    x,
                    noise_transformer,
                )
            else:
                x = self.transformer_1[i * 2 + 1](
                    x,
                )
            x = self._reverse_windows(
                x, batch_size, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]

        x_res2 = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()
        # (B, channels, lat, lon)
        x = self.downsample(x)
        # (B, 2*channels, lat/2, lon/2)

        batch_size, channels, xlat, xlon = x.shape  # noqa: N806
        x = x.permute(0, 2, 3, 1).contiguous()

        for i in range(self.n_full_blocks[1]):
            x = self._get_windows(x, self.window_size_inner)
            n_windows = x.shape[0] // batch_size
            noise_transformer_down = noise.repeat_interleave(n_windows, dim=0)
            if self.nice_norm:
                x = self.transformer_down[i * 2](
                    x,
                    noise_transformer_down,
                )
            else:
                x = self.transformer_down[i * 2](
                    x,
                )
            x = self._reverse_windows(
                x, batch_size, xlat, xlon, channels, self.window_size_inner
            )
            x = self.roll_in(x)
            x = self._get_windows(x, self.window_size_inner)
            if self.nice_norm:
                x = self.transformer_down[i * 2 + 1](
                    x,
                    noise_transformer_down,
                )
            else:
                x = self.transformer_down[i * 2 + 1](
                    x,
                )
            x = self._reverse_windows(
                x, batch_size, xlat, xlon, channels, self.window_size_inner
            )
            x = self.unroll_in(x)  # [B, lat, lon, channels]

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.upsample(x)

        batch_size, channels, xlat, xlon = x.shape

        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).contiguous()
        x = self.activation(x)
        x = self.upsample_mix(x)
        x = x.reshape(batch_size, xlat, xlon, channels)

        x = x + x_res2

        for i in range(self.n_full_blocks[2]):
            x = self._get_windows(x, self.window_size_outer)
            n_windows = x.shape[0] // batch_size
            noise_transformer = noise.repeat_interleave(n_windows, dim=0)

            if self.nice_norm:
                x = self.transformer_2[i * 2](
                    x,
                    noise_transformer,
                )
            else:
                x = self.transformer_2[i * 2](
                    x,
                )
            x = self._reverse_windows(
                x, batch_size, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            if self.nice_norm:
                x = self.transformer_2[i * 2 + 1](
                    x,
                    noise_transformer,
                )
            else:
                x = self.transformer_2[i * 2 + 1](
                    x,
                )
            x = self._reverse_windows(
                x, batch_size, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]
        batch_size, xlat, xlon, channels = x.shape  # noqa: N806
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.patch_recovery(torch.cat((x, x_res), dim=-3))  # concat along channels

        return x

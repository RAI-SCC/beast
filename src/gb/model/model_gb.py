import torch
from jigsaw import attention, convolutions, linear

from gb.model.expert import ExpertWeightingLayer


class Bellbeast(torch.nn.Module):
    def __init__(
        self,
        xlat,
        xlon,
        in_channels,
        embed_dim,
        out_channels,
        heads,
        kernel_size,
        window_size_outer,
        window_size_inner,
        n_full_blocks,
        noise_dim,
        dtp_rank,
        device,
        dtp_group,
        spatial_group,
        channel_group,
        ep_group,
        up_group,
        bayesian,
        parallelism
    ):
        super().__init__()
        self.xlat, self.xlon = xlat, xlon
        self.window_size_outer = window_size_outer
        self.window_size_inner = window_size_inner
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.padding = 0
        self.dilation = 1
        self.local_rank = dtp_rank
        self.device = device
        self.parallelism = parallelism
        self.ep_group = ep_group
        self.up_group = up_group
        self.dtp_group = dtp_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.heads = heads
        self.out_channels = out_channels

        self.bayesian = bayesian
        self.noise_dim = noise_dim

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
            parallelism=self.parallelism,
            bayesian=self.bayesian,
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
            parallelism=self.parallelism,
            bias=True,
            bayesian=self.bayesian,
        )

        self.expert_layer = ExpertWeightingLayer(
            (self.ep_group.size(), 1, self.embed_dim, 1),
            self.ep_group,
            self.up_group,
            device=self.device,
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
            parallelism=self.parallelism,
            bias=True,
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
            parallelism=self.parallelism,
            bayesian=self.bayesian,
        )

        self.n_full_blocks = n_full_blocks
        self.transformer = torch.nn.ModuleList(
            [
                attention.DistributedNICEWindowedTransformer(
                    x_features=self.embed_dim,
                    embed_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    heads=self.heads,
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # dropout
                    local_rank=self.local_rank,
                    device=self.device,
                    parallelism=2,
                    model_group=self.dtp_group,
                    spatial_group=self.spatial_group,
                    channel_group=self.channel_group,
                    bayesian=self.bayesian,
                ).to(self.device)
                for i in torch.arange(
                    2 * (self.n_full_blocks[0] + self.n_full_blocks[2])
                )
            ]
        )

        self.transformer_down = torch.nn.ModuleList(
            [
                attention.DistributedNICEWindowedTransformer(
                    self.embed_dim * 2,
                    self.embed_dim * 2,
                    self.embed_dim * 2,
                    heads=self.heads,
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # dropout
                    local_rank=self.local_rank,
                    device=self.device,
                    parallelism=2,
                    model_group=self.dtp_group,
                    spatial_group=self.spatial_group,
                    channel_group=self.channel_group,
                    bayesian=self.bayesian,
                ).to(self.device)
                for i in torch.arange(2 * self.n_full_blocks[1])
            ]
        )

        assert window_size_outer[0] % 2 == 0, "window[0] size must be divisible by 2"
        assert window_size_outer[1] % 2 == 0, "window[1] size must be divisible by 2"
        assert window_size_inner[0] % 2 == 0, "window[0] size must be divisible by 2"
        assert window_size_inner[1] % 2 == 0, "window[1] size must be divisible by 2"

        self.roll_out = linear.DistributedRoll(
            shifts=(window_size_outer[0] // 2, window_size_outer[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
        )

        self.unroll_out = linear.DistributedRoll(
            shifts=(-window_size_outer[0] // 2, -window_size_outer[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
        )
        self.roll_in = linear.DistributedRoll(
            shifts=(window_size_inner[0] // 2, window_size_inner[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
        )

        self.unroll_in = linear.DistributedRoll(
            shifts=(-window_size_inner[0] // 2, -window_size_inner[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
        )

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

    def forward(self, x):
        # x has global shape (B, xlat, xlon, channels)

        noise = torch.randn(x.shape[0], self.noise_dim).to(self.device)

        x = self.patch_embedding(x)
        x_res = x.clone()
        B, channels, xlat, xlon = x.shape
        x = x.view(B, channels, xlat * xlon)

        x = x.mT
        x = x.view(B, xlat, xlon, channels)

        # attention
        for i in range(self.n_full_blocks[0]):
            x = self._get_windows(x, self.window_size_outer)
            n_windows = x.shape[0] // B
            # x has shape: [S, B*n_windows, window_, window_]
            # noise has shape [B, noise_dim]
            # we want noise to have [B*n_sw]
            noise_transformer = noise.repeat_interleave(n_windows, dim=0)
            x = self.transformer[i * 2](x, noise_transformer, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            mask = self._gen_mask_(B, xlat, xlon, self.window_size_outer).unsqueeze(
                1
            )  # unsqueeze for head dimension
            mask = mask.detach()
            x = self.transformer[i * 2 + 1](x, noise_transformer, mask)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]

        x_res2 = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.downsample(x)
        # Introduce non-linearity after downsampling layer
        # TODO: GELU

        # downsample

        # here comes the Latent Representation of the experts into play:
        x = self.expert_layer(x)

        B, channels, xlat, xlon = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        for i in range(self.n_full_blocks[1]):
            x = self._get_windows(x, self.window_size_inner)
            n_windows = x.shape[0] // B
            noise_transformer_down = noise.repeat_interleave(n_windows, dim=0)
            x = self.transformer_down[i * 2](x, noise_transformer_down, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_inner
            )
            x = self.roll_in(x)
            x = self._get_windows(x, self.window_size_inner)
            mask = self._gen_mask_(B, xlat, xlon, self.window_size_inner).unsqueeze(
                1
            )  # unsqueeze for head dimension
            mask = mask.detach()
            x = self.transformer_down[i * 2 + 1](x, noise_transformer_down, mask)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_inner
            )
            x = self.unroll_in(x)  # [B, lat, lon, channels]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.upsample(x)  # TODO: second layer and non-linearity

        B, channels, xlat, xlon = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + x_res2

        for i in range(self.n_full_blocks[2]):
            x = self._get_windows(x, self.window_size_outer)
            x = self.transformer[i * 2 + 2 * self.n_full_blocks[0]](
                x, noise=noise_transformer, mask=None
            )
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )

            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            mask = self._gen_mask_(B, xlat, xlon, self.window_size_outer).unsqueeze(
                1
            )  # unsqueeze for head dimension
            mask = mask.detach()
            x = self.transformer[i * 2 + 2 * self.n_full_blocks[0] + 1](
                x, noise_transformer, mask
            )
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]

        B, xlat, xlon, channels = x.shape
        x = x.view(B, xlat * xlon, channels)

        x = x.mT
        x = x.view(B, channels, xlat, xlon).contiguous()
        x = self.patch_recovery(torch.cat((x, x_res), dim=-3))

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
        img_mask = torch.zeros((1, n_patch_lat, n_patch_lon, 1)).to(
            self.device
        )  # 1 n_patch_lat n_patch_lon 1

        img_mask[:, slice(0, window_size[1] // 2)] = 1
        img_mask[:, slice(window_size[1] // 2, None)] = 2

        mask_windows = self._get_windows(
            img_mask, window_size
        )  # n_windows, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])

        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0, float(-10000.0)
        ).masked_fill(attention_mask == 0, float(0.0))
        attention_mask = attention_mask.expand(
            batch_dim, *attention_mask.shape
        ).flatten(start_dim=0, end_dim=1)

        if self.local_rank in [
            0,
            2,
        ]:  # TODO: "left side" ranks. i.e., those in first column
            attention_mask = attention_mask[
                ..., : (window_size[0] * window_size[1] // 2)
            ]
        else:
            attention_mask = attention_mask[
                ..., (window_size[0] * window_size[1] // 2) :
            ]
        return attention_mask

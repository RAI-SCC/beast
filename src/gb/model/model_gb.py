import torch
from jigsaw import attention, convolutions, linear
from torch.cuda import nvtx

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
        parallelism,
        dtype,
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
        self.dtype = dtype
        self.activation = torch.nn.GELU()

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
            parallelism=self.parallelism,  # either 4 or 8 (# GPUs in node)
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
            parallelism=self.parallelism,
            bias=True,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        self.expert_layer = ExpertWeightingLayer(
            (self.ep_group.size(), 1, self.embed_dim, 1),
            self.ep_group,
            self.up_group,
            device=self.device,
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
            parallelism=self.parallelism,
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
            device=device,
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
            parallelism=self.parallelism,
            bayesian=self.bayesian,
            dtype=self.dtype,
        )

        self.n_full_blocks = n_full_blocks
        self.transformer_1 = torch.nn.ModuleList(
            [
                attention.DistributedNICEWindowedTransformer(
                    x_features=self.embed_dim,
                    embed_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    heads=self.heads[0],
                    noise_dim=self.noise_dim,
                    dropout=0 if self.bayesian else 1e-3,  # dropout
                    local_rank=self.local_rank,
                    device=self.device,
                    parallelism=self.channel_group.size(),
                    model_group=self.dtp_group,
                    spatial_group=self.spatial_group,
                    channel_group=self.channel_group,
                    bayesian=self.bayesian,
                    dtype=self.dtype,
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
                    dropout=0 if self.bayesian else 1e-3,  # dropout
                    local_rank=self.local_rank,
                    device=self.device,
                    parallelism=self.channel_group.size(),
                    model_group=self.dtp_group,
                    spatial_group=self.spatial_group,
                    channel_group=self.channel_group,
                    bayesian=self.bayesian,
                    dtype=self.dtype,
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
                    dropout=0 if self.bayesian else 1e-3,  # dropout
                    local_rank=self.local_rank,
                    device=self.device,
                    parallelism=self.channel_group.size(),
                    model_group=self.dtp_group,
                    spatial_group=self.spatial_group,
                    channel_group=self.channel_group,
                    bayesian=self.bayesian,
                    dtype=self.dtype,
                ).to(self.device)
                for _ in torch.arange(2 * self.n_full_blocks[2])
            ]
        )

        assert window_size_outer[0] % 2 == 0, (
            "window_outer[0] size must be divisible by 2"
        )
        assert window_size_outer[1] % 2 == 0, (
            "window_outer[1] size must be divisible by 2"
        )
        assert window_size_inner[0] % 2 == 0, (
            "window_inner[0] size must be divisible by 2"
        )
        assert window_size_inner[1] % 2 == 0, (
            "window_inner[1] size must be divisible by 2"
        )

        self.roll_out = linear.DistributedRoll(
            shifts=(window_size_outer[0] // 2, window_size_outer[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
            dtype=dtype,
        )

        self.unroll_out = linear.DistributedRoll(
            shifts=(-(window_size_outer[0] // 2), -(window_size_outer[1] // 2)),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
            dtype=dtype,
        )
        self.roll_in = linear.DistributedRoll(
            shifts=(window_size_inner[0] // 2, window_size_inner[1] // 2),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
            dtype=dtype,
        )

        self.unroll_in = linear.DistributedRoll(
            shifts=(-(window_size_inner[0] // 2), -(window_size_inner[1] // 2)),
            dims=(-3, -2),
            local_rank=self.local_rank,
            device=self.device,
            parallelism=self.parallelism,
            spatial_group=self.spatial_group,
            dtype=dtype,
        )
        self.mask_out, self.mask_in = None, None

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

        # if self.channel_group.rank() == 0:
        #     attention_mask = attention_mask[
        #         ..., : ((window_size[0] * window_size[1]) // 2)
        #     ]
        # elif self.channel_group.rank() == 1:
        #     attention_mask = attention_mask[
        #         ..., ((window_size[0] * window_size[1]) // 2) :
        #     ]
        # else:
        #     assert self.channel_group.size() == 2, (
        #         "No current implementation for case where channel group has more than 2 ranks."
        #     )

        # save attention mask here
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

    def forward(self, x):
        # noise has shape [(sample), batch, noise_dim]
        noise = torch.randn(x.shape[0], self.noise_dim, device=self.device).to(
            self.dtype
        )

        # Define masks once upon initialization (dependent on batch dimension)
        self.mask_out, self.mask_in = self._define_masks(x.shape)
        self.mask_out, self.mask_in = (
            self.mask_out.to(torch.bfloat16),
            self.mask_in.to(torch.bfloat16),
        )  # hard-coded to bfloat16 if we use torch scaled_dot_product_attention

        # x has global shape (B, xlat, xlon, channels)
        nvtx.range_push("Patch Embedding")
        x = self.patch_embedding(x)
        nvtx.range_pop()
        x_res = x.clone()
        B, channels, xlat, xlon = x.shape
        x = x.reshape(B, channels, xlat * xlon).mT.view(B, xlat, xlon, channels)

        # attention
        for i in range(self.n_full_blocks[0]):
            nvtx.range_push(f"Attn block 0-{i}")
            x = self._get_windows(x, self.window_size_outer)
            # x has shape: [(S), B*n_windows, window_size_outer[0], window_size_outer[1], channels]
            n_windows = x.shape[0] // B
            # noise has shape [B, noise_dim] --> use interleave to make it [B*n_windows]
            noise_transformer = noise.repeat_interleave(n_windows, dim=0)
            x = self.transformer_1[i * 2](x, noise_transformer, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            x = self.transformer_1[i * 2 + 1](x, noise_transformer, mask=self.mask_out)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]
            nvtx.range_pop()

        x_res2 = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()
        # (B, channels, lat, lon)
        nvtx.range_push("Downsample")
        x = self.downsample(x)
        nvtx.range_pop()
        # (B, 2*channels, lat/2, lon/2)

        # here comes the Latent Representation of the experts into play:
        nvtx.range_push("Expert Layer")
        x = self.expert_layer(x)  # does not change the shape of the data
        nvtx.range_pop()

        B, channels, xlat, xlon = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()

        for i in range(self.n_full_blocks[1]):
            nvtx.range_push(f"Attn block 1-{i}")
            x = self._get_windows(x, self.window_size_inner)
            n_windows = x.shape[0] // B
            noise_transformer_down = noise.repeat_interleave(n_windows, dim=0)
            x = self.transformer_down[i * 2](x, noise_transformer_down, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_inner
            )
            x = self.roll_in(x)
            x = self._get_windows(x, self.window_size_inner)
            x = self.transformer_down[i * 2 + 1](
                x, noise_transformer_down, mask=self.mask_in
            )
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_inner
            )
            x = self.unroll_in(x)  # [B, lat, lon, channels]
            nvtx.range_pop()

        x = x.permute(0, 3, 1, 2).contiguous()

        nvtx.range_push("Upsample")
        x = self.upsample(x)
        nvtx.range_pop()

        B, channels, xlat, xlon = x.shape

        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).contiguous()
        nvtx.range_push("Activation + Upsample Mix")
        x = self.activation(x)
        x = self.upsample_mix(x)
        nvtx.range_pop()
        x = x.reshape(B, xlat, xlon, channels)

        x = x + x_res2

        for i in range(self.n_full_blocks[2]):
            nvtx.range_push(f"Attn block 2-{i}")
            x = self._get_windows(x, self.window_size_outer)
            x = self.transformer_2[i * 2](x, noise=noise_transformer, mask=None)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            x = self._get_windows(x, self.window_size_outer)
            x = self.transformer_2[i * 2 + 1](x, noise_transformer, mask=self.mask_out)
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]
            nvtx.range_pop()
        B, xlat, xlon, channels = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()

        nvtx.range_push("Patch Recovery")
        x = self.patch_recovery(torch.cat((x, x_res), dim=-3))  # concat along channels
        nvtx.range_pop()

        return x

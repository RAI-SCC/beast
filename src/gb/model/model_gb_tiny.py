import torch
import torch.distributed as dist
import torch.nn as nn

from gb.model.expert import ExpertWeightingLayer


class TinyBellbeast(torch.nn.Module):
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
        device,
        channel_group,
        ep_group,
        up_group,
    ):
        super().__init__()
        self.xlat, self.xlon = xlat, xlon
        self.window_size_outer = window_size_outer
        self.window_size_inner = window_size_inner
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.padding = 0
        self.dilation = 1
        self.device = device
        self.ep_group = ep_group
        self.up_group = up_group
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.heads = heads
        self.out_channels = out_channels

        self.noise_dim = noise_dim

        self.channel_group = channel_group  # hard coded, to avoid parsing through

        self.patch_embedding = torch.nn.Conv2d(
            self.in_channels,
            self.embed_dim,  # global values
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            groups=1,
            bias=True,
            device=self.device,
        )
        self.downsample = torch.nn.Conv2d(
            self.embed_dim,
            self.embed_dim * 2,  # global values
            2,
            2,
            self.padding,
            self.dilation,
            groups=1,
            device=self.device,
            bias=True,
        )

        self.expert_layer = ExpertWeightingLayer(
            (self.ep_group.size(), 1, self.embed_dim * 2, 1),
            self.ep_group,
            self.up_group,
            device=self.device,
        )

        self.upsample = torch.nn.ConvTranspose2d(
            self.embed_dim * 2,
            self.embed_dim,  # global values
            2,
            2,
            self.padding,
            self.dilation,
            groups=1,
            device=self.device,
            bias=True,
        )

        self.patch_recovery = torch.nn.ConvTranspose2d(
            self.embed_dim * 2,
            self.out_channels,  # global values
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            bias=True,
            groups=1,
            device=self.device,
        )

        self.n_full_blocks = n_full_blocks
        self.transformer = torch.nn.ModuleList(
            [
                NiceWindowedTransformerBlock(
                    embed_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    heads=self.heads,
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # TODO Make this flag for bayesian vs non-bayesian
                    device=self.device,
                ).to(self.device)
                for i in torch.arange(
                    2 * (self.n_full_blocks[0] + self.n_full_blocks[2])
                )
            ]
        )

        self.transformer_down = torch.nn.ModuleList(
            [
                NiceWindowedTransformerBlock(
                    self.embed_dim * 2,
                    self.embed_dim * 2,
                    heads=self.heads,
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # TODO Make this flag for bayesian vs non-bayesian
                    device=self.device,
                ).to(self.device)
                for i in torch.arange(2 * self.n_full_blocks[1])
            ]
        )

        assert window_size_outer[0] % 2 == 0, "window[0] size must be divisible by 2"
        assert window_size_outer[1] % 2 == 0, "window[1] size must be divisible by 2"
        assert window_size_inner[0] % 2 == 0, "window[0] size must be divisible by 2"
        assert window_size_inner[1] % 2 == 0, "window[1] size must be divisible by 2"

        self.roll_out = lambda x: torch.roll(
            x,
            shifts=(window_size_outer[0] // 2, window_size_outer[1] // 2),
            dims=(-3, -2),
        )

        self.unroll_out = lambda x: torch.roll(
            x,
            shifts=(-window_size_outer[0] // 2, -window_size_outer[1] // 2),
            dims=(-3, -2),
        )

        self.roll_in = lambda x: torch.roll(
            x,
            shifts=(window_size_inner[0] // 2, window_size_inner[1] // 2),
            dims=(-3, -2),
        )

        self.unroll_in = lambda x: torch.roll(
            x,
            shifts=(-window_size_inner[0] // 2, -window_size_inner[1] // 2),
            dims=(-3, -2),
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

        return attention_mask

    def forward(self, x):
        # x has  shape (B, channels, xlat, xlon)
        noise = torch.randn(x.shape[0], self.noise_dim).to(self.device)

        x = self.patch_embedding(x)
        # x has shape (B, embed_dim, xlat/patch, xlon/patch)

        x_res = x.clone()
        B, channels, xlat, xlon = x.shape
        x = x.view(B, channels, xlat * xlon)

        x = x.mT
        x = x.view(B, xlat, xlon, channels)
        # x has shape (B, xlat/patch, xlon/patch, embed_dim)

        # attention
        for i in range(self.n_full_blocks[0]):
            x = self._get_windows(x, self.window_size_outer)
            # x has shape [B*num_windows, (xlat/patche/window_x * xlon/patche/window_y), embed_dim]
            n_windows = x.shape[0] // B

            noise_transformer = noise.repeat_interleave(n_windows, dim=0)

            x = self.transformer[i * 2](x, noise_transformer, mask=None)
            print(f"#Attn1 {dist.get_rank(group=None)}, x.shape = {x.shape}")
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.roll_out(x)
            print(f"#RollOut {dist.get_rank(group=None)}, x.shape = {x.shape}")
            x = self._get_windows(x, self.window_size_outer)
            print(f"#Windows2 {dist.get_rank(group=None)}, x.shape = {x.shape}")
            mask = self._gen_mask_(B, xlat, xlon, self.window_size_outer).unsqueeze(
                1
            )  # unsqueeze for head dimension
            mask = mask.detach()
            x = self.transformer[i * 2 + 1](x, noise_transformer, mask)
            print(f"#Attn2 {dist.get_rank(group=None)}, x.shape = {x.shape}")
            x = self._reverse_windows(
                x, B, xlat, xlon, channels, self.window_size_outer
            )
            x = self.unroll_out(x)  # [B, lat, lon, channels]
            print(f"#UnRollOut {dist.get_rank(group=None)}, x.shape = {x.shape}")

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


class NiceWindowedTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        heads: int,
        noise_dim: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.output_dim = output_dim
        self.noise_dim = noise_dim

        self.attention_block = WindowedAttention(
            self.embed_dim,
            self.output_dim,
            self.heads,
        ).to(device)

        self.layer_norm1 = NiceNorm(output_dim, self.noise_dim, device=device).to(
            device
        )
        self.layer_norm2 = NiceNorm(output_dim, self.noise_dim, device=device).to(
            device
        )

        self.MLP = MLP(
            output_dim,
            output_dim,
            output_dim,
        ).to(device)

        # _average_weights(self, spatial_group, device)

    def forward(self, x: torch.tensor, noise: torch.tensor, mask=None):
        x = self.attention_block(self.layer_norm1(x, noise), mask) + x
        x = self.MLP(self.layer_norm2(x, noise)) + x

        return x


class WindowedAttention(nn.Module):
    """
    2D Attention mechanism with absolute position bias term.

    Attention(Q, K, V) = (SoftMax(Q@K^T/sqrt(D)) @ V
    """

    def __init__(
        self,
        embed_dim,
        output_dim,
        num_heads,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Initialize several operations
        # Should make sense to use dim*3 to generate the vectors for a qkv matrix
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(embed_dim, embed_dim)

        # Store several attributes
        self.num_heads = num_heads
        self.scale = (self.embed_dim // self.num_heads) ** -0.5

    def forward(self, x, mask):
        """
        Forward pass of 2D attention mechanism with absolute positional bias term.

        x: Tensor
            of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
        mask: Tensor
            of shape (n_windows, n_patch_in_window, n_patch_in_window)

        Returns
        -------
        x: Tensor
            of shape (n_batch * n_windows, n_patch_in_window, hidden_dim)
        """
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            (
                x.shape[0],
                x.shape[1],
                3,
                self.num_heads,
                self.embed_dim // self.num_heads,
            )
        )
        qkv = torch.permute(qkv, (2, 0, 3, 1, 4))
        q, k, v = torch.tensor_split(qkv, 3, dim=0)

        q = q.squeeze(dim=0) * self.scale

        attention = q @ k.squeeze(dim=0).transpose(-2, -1)

        if mask is not None:
            attention = attention + mask

        attention = self.softmax(attention)

        values = attention @ v.squeeze(dim=0)

        # head merging
        values = torch.transpose(values, 1, 2).reshape(x.shape)
        values = self.linear(values)

        return values


class MLP(nn.Module):
    """MLP."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        device=None,
        dtype=None,
        # dropout_rate
    ):
        """
        Initialize MLP.

        dim: int
        dropout_rate: float
        """
        super().__init__()
        """MLP layers, same as most vision transformer architectures."""

        self.linear1 = nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype)
        self.linear2 = nn.Linear(hidden_dim, output_dim, device=device, dtype=dtype)
        self.activation = nn.GELU()

    def forward(self, x):
        """Forward function of MLP."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class NiceNorm(nn.Module):
    def __init__(self, normalized_shape, noise_dim, eps=1e-5, device=None, dtype=None):
        super().__init__()

        self.norm = nn.LayerNorm(
            normalized_shape,
            eps=eps,
            device=device,
            dtype=dtype,
            elementwise_affine=False,
        )

        self.noise_dim = noise_dim
        self.shape = normalized_shape

        self.scale_mlp = MLP(self.noise_dim, self.shape, self.shape)
        self.shift_mlp = MLP(self.noise_dim, self.shape, self.shape)

    def forward(self, x, noise):
        x = self.norm(x)

        scale = self.scale_mlp(noise)
        shift = self.shift_mlp(noise)

        x = scale.unsqueeze(dim=1) * x + shift.unsqueeze(dim=1)

        return x

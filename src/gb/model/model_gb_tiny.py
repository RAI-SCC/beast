import torch
import torch.nn as nn
import torch.nn.init as init

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
            (self.ep_group.size(), 1, self.embed_dim, 1),
            self.ep_group,
            self.up_group,
            device=self.device,
        )

        self.upsample = torch.nn.Conv2d(
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

        self.patch_recovery = torch.nn.Conv2d(
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
                    x_features=self.embed_dim,
                    embed_dim=self.embed_dim,
                    output_dim=self.embed_dim,
                    heads=self.heads,
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # dropout
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
                    self.embed_dim * 2,
                    heads=self.heads,
                    noise_dim=self.noise_dim,
                    dropout=1e-3,  # dropout
                    device=self.device,
                ).to(self.device)
                for i in torch.arange(2 * self.n_full_blocks[1])
            ]
        )

        assert window_size_outer[0] % 2 == 0, "window[0] size must be divisible by 2"
        assert window_size_outer[1] % 2 == 0, "window[1] size must be divisible by 2"
        assert window_size_inner[0] % 2 == 0, "window[0] size must be divisible by 2"
        assert window_size_inner[1] % 2 == 0, "window[1] size must be divisible by 2"

        self.roll_out = torch.roll(
            shifts=(window_size_outer[0] // 2, window_size_outer[1] // 2),
            dims=(-3, -2),
        )

        self.unroll_out = torch.rolll(
            shifts=(-window_size_outer[0] // 2, -window_size_outer[1] // 2),
            dims=(-3, -2),
        )

        self.roll_in = torch.roll(
            shifts=(window_size_inner[0] // 2, window_size_inner[1] // 2),
            dims=(-3, -2),
        )

        self.unroll_in = torch.roll(
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



class NiceWindowedTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        noise_dim: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.x_features = x_features
        self.embed_dim = embed_dim
        self.heads = heads
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        
        
        self.attention_block = WindowedAttention(
            self.x_features,
            self.embed_dim,
            self.output_dim,
            self.heads,
        ).to(device)

        self.layer_norm1 = NiceNorm(output_dim, self.noise_dim, device).to(device)
        self.layer_norm2 = NiceNorm(output_dim, self.noise_dim, device).to(device)
        
        self.MLP = MLP(
            output_dim,
            output_dim,
            output_dim,
        ).to(device)

        #_average_weights(self, spatial_group, device)

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
        x_features
        embed_dim,
        output_dim,
        heads,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.x_features = x_features

        # Initialize several operations
        # Should make sense to use dim*3 to generate the vectors for a qkv matrix
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(embed_dim, embed_dim)

        # Store several attributes
        self.heads = heads
        self.scale = (embed_dim // heads) ** -0.5
     

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

        # x shape: (B*n_windows, W0*W1*W2, C)
        x_patches_local = x

        # reshape the data to calculate multi-head attention
        qkv = self.qkv(x_patches_local).reshape(
                (
                x.shape[0],
                x.shape[1],
                3,
                self.head_number,
                self.dim // self.head_number,
                )
            )
        q, k, v = ptorch.tensor_split(qkv, 3, dim=-1)

        # Scale the attention
        q = q * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        # Attention shape: nB * n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
        attention = query @ key.transpose(-2, -1)  # @ denotes matrix multiplication

        if mask is not None:
            attention = attention + mask


        attention = self.softmax(attention)

        values = attention @ v
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
        dtype=None
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

        self.activation = nn.GELU()

        self.scale_weight1 = nn.Parameter(
            torch.empty((normalized_shape, noise_dim), dtype=dtype, device=device)
        )
        self.scale_bias1 = nn.Parameter(
            torch.empty(normalized_shape, dtype=dtype, device=device)
        )
        self.scale_weight2 = nn.Parameter(
            torch.empty(
                (normalized_shape, normalized_shape), dtype=dtype, device=device
            )
        )
        self.scale_bias2 = nn.Parameter(
            torch.empty(normalized_shape, dtype=dtype, device=device)
        )

        self.shift_weight1 = nn.Parameter(
            torch.empty((normalized_shape, noise_dim), dtype=dtype, device=device)
        )
        self.shift_bias1 = nn.Parameter(
            torch.empty(normalized_shape, dtype=dtype, device=device)
        )
        self.shift_weight2 = nn.Parameter(
            torch.empty(
                (normalized_shape, normalized_shape), dtype=dtype, device=device
            )
        )
        self.shift_bias2 = nn.Parameter(
            torch.empty(normalized_shape, dtype=dtype, device=device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.scale_weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.scale_weight2, a=math.sqrt(5))
        init.kaiming_uniform_(self.shift_weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.shift_weight2, a=math.sqrt(5))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.scale_weight1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.scale_bias1, -bound, bound)
        init.uniform_(self.shift_bias1, -bound, bound)

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.scale_weight2)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.scale_bias2, -bound, bound)
        init.uniform_(self.shift_bias2, -bound, bound)

    def forward(self, x, noise):
        x = self.norm(x)

        scale = torch.nn.functional.linear(noise, self.scale_weight1, self.scale_bias1)
        scale = self.activation(scale)
        scale = torch.nn.functional.linear(scale, self.scale_weight2, self.scale_bias2)
        scale = scale.reshape(x.shape[0], *[1 for _ in x.shape[1:-1]], self.shape)

        shift = torch.nn.functional.linear(noise, self.shift_weight1, self.shift_bias1)
        shift = self.activation(shift)
        shift = torch.nn.functional.linear(shift, self.shift_weight2, self.shift_bias2)
        shift = shift.reshape(x.shape[0], *[1 for _ in x.shape[1:-1]], self.shape)

        x = scale * x + shift

        return x


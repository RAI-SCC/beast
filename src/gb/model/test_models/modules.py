import math

import torch
import torch.nn as nn
import torch.nn.init as init


class WindowedAttention(nn.Module):
    """
    2D Attention mechanism with absolute position bias term.

    Attention(Q, K, V) = (SoftMax(Q@K^T/sqrt(D)) @ V
    """

    def __init__(
        self,
        x_features,
        embed_dim,
        output_dim,
        heads,
        device,
    ):
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim
        self.x_features = x_features

        # Initialize several operations
        # Should make sense to use dim*3 to generate the vectors for a qkv matrix
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
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
        q, k, v = torch.tensor_split(qkv, 3, dim=-1)

        # Scale the attention
        q = q * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        # Attention shape: nB * n_windows, nHeads, N, N (n_patch_in_window = W0*W1*W2)
        attention = q @ k.transpose(-2, -1)  # @ denotes matrix multiplication

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
        # dropout_rate
    ):
        """
        Initialize MLP.

        dim: int
        dropout_rate: float
        """
        super().__init__()
        """MLP layers, same as most vision transformer architectures."""

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
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


class NICEWindowedTransformerBlock(torch.nn.Module):
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
        self.noise_dim = noise_dim
        self.attention_block = WindowedAttention(
            x_features,
            embed_dim,
            output_dim,
            heads,
            device,
        ).to(device)

        self.layer_norm1 = NiceNorm(output_dim, self.noise_dim, device).to(device)
        self.layer_norm2 = NiceNorm(output_dim, self.noise_dim, device).to(device)

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

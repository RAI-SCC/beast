import torch


class NiceWindowedTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        heads: int,
        noise_dim: int,
        dropout: float,
        device: torch.device,
        dtype,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.output_dim = output_dim
        self.noise_dim = noise_dim

        self.device = device
        self.dtype = dtype

        self.attention_block = WindowedAttention(
            self.embed_dim, self.output_dim, self.heads, self.dtype
        ).to(device)

        self.layer_norm1 = NiceNorm(output_dim, self.noise_dim, device=device).to(
            self.device
        )
        self.layer_norm2 = NiceNorm(output_dim, self.noise_dim, device=device).to(
            self.device
        )

        self.MLP = MLP(
            output_dim,
            output_dim,
            output_dim,
        ).to(self.device)

        # _average_weights(self, spatial_group, device)

    def forward(self, x: torch.tensor, noise: torch.tensor, mask=None):
        x = self.attention_block(self.layer_norm1(x, noise), mask) + x
        x = self.MLP(self.layer_norm2(x, noise)) + x

        return x


class WindowedAttention(torch.nn.Module):
    """
    2D Attention mechanism with absolute position bias term.

    Attention(Q, K, V) = (SoftMax(Q@K^T/sqrt(D)) @ V
    """

    def __init__(
        self,
        embed_dim,
        output_dim,
        num_heads,
        dtype,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Initialize several operations
        # Should make sense to use dim*3 to generate the vectors for a qkv matrix
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        self.dtype = dtype

        # Store several attributes
        self.num_heads = num_heads
        self.scale = (self.embed_dim // self.num_heads) ** -0.5

    @torch.compile(mode="reduce-overhead")
    def _attention_core(self, qkv):
        qkv = qkv.reshape(
            (
                qkv.shape[0],
                qkv.shape[1],
                3,
                self.num_heads,
                qkv.shape[2] // self.num_heads // 3,
            )
        )
        qkv = torch.permute(qkv, (2, 0, 3, 1, 4))

        q, k, v = qkv.unbind(0)

        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=0.0
            )
        return out

    @torch.compile(mode="reduce-overhead")
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
        qkv = self.qkv(x).to(torch.bfloat16)

        out = self._attention_core(qkv)
        # q, k, v = torch.tensor_split(qkv, 3, dim=0)
        # q = q.squeeze(dim=0) * self.scale
        # attention = q @ k.squeeze(dim=0).transpose(-2, -1)
        # if mask is not None:
        #     attention = attention + mask
        # attention = self.softmax(attention)
        # values = attention @ v.squeeze(dim=0)

        # head merging
        out = torch.transpose(out, 1, 2).reshape(x.shape).to(self.dtype)
        values = self.linear(out)

        return values


class MLP(torch.nn.Module):
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

        self.linear1 = torch.nn.Linear(
            input_dim, hidden_dim, device=device, dtype=dtype
        )
        self.linear2 = torch.nn.Linear(
            hidden_dim, output_dim, device=device, dtype=dtype
        )
        self.activation = torch.nn.GELU()

    def forward(self, x):
        """Forward function of MLP."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class NiceNorm(torch.nn.Module):
    def __init__(self, normalized_shape, noise_dim, eps=1e-5, device=None, dtype=None):
        super().__init__()

        self.norm = torch.nn.LayerNorm(
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

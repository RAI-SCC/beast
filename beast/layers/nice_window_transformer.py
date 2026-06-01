"""Transformer blocks and components for windowed attention.

This module provides transformer blocks with windowed attention mechanisms,
including attention, MLP, and NICE normalization layers.
"""

import torch


class NiceWindowedTransformerBlock(torch.nn.Module):
    """Transformer block with windowed attention and NICE normalization.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    output_dim : int
        Output dimension.
    heads : int
        Number of attention heads.
    noise_dim : int
        Noise dimension for NICE normalization.
    dropout : float
        Dropout rate.
    device : torch.device | None, optional
        Device to place tensors on, by default None.
    dtype : torch.dtype, optional
        Data type of parameters, by default torch.float32.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        heads: int,
        noise_dim: int,
        dropout: float,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.output_dim = output_dim
        self.noise_dim = noise_dim

        self.device = device
        self.dtype = dtype

        self.attention_block = WindowedAttention(
            self.embed_dim, self.heads, self.dtype
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

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        noise : torch.Tensor
            Noise tensor for NICE normalization.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.attention_block(self.layer_norm1(x, noise)) + x
        x = self.MLP(self.layer_norm2(x, noise)) + x

        return x


class WindowedAttention(torch.nn.Module):
    """2D attention mechanism.

    Computes scaled dot-product attention using flash attention:
    Attention(Q, K, V) = SoftMax(Q @ K^T / sqrt(d)) @ V

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    dtype : torch.dtype
        Data type for computations.
    dropout : float
        Dropout rate for attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dtype: torch.dtype,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Store several attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dtype = dtype
        self.dropout = dropout

        # Initialize several operations
        # Should make sense to use dim*3 to generate the vectors for a qkv matrix
        self.w_qkv = torch.nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear = torch.nn.Linear(embed_dim, embed_dim)

    @torch.compile(mode="reduce-overhead")
    def _attention_core(self, qkv: torch.Tensor) -> torch.Tensor:
        qkv = qkv.reshape(
            (
                qkv.shape[0],
                qkv.shape[1],
                3,
                self.num_heads,
                self.head_dim // 3,
            )
        )
        qkv = torch.permute(qkv, (2, 0, 3, 1, 4))

        q, k, v = qkv.unbind(0)

        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=self.dropout
            )
        return out

    @torch.compile(mode="reduce-overhead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 2D attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (n_batch * n_windows, n_patch_in_window, hidden_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n_batch * n_windows, n_patch_in_window, hidden_dim).
        """
        qkv = self.w_qkv(x).to(torch.bfloat16)

        out = self._attention_core(qkv)

        # head merging
        out = torch.transpose(out, 1, 2).reshape(x.shape).to(self.dtype)
        values: torch.Tensor = self.linear(out)

        return values

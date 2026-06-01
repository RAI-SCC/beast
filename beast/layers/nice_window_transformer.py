"""Transformer blocks and components for windowed attention.

This module provides transformer blocks with windowed attention mechanisms,
including attention, MLP, and NICE normalization layers.
"""

import torch
from beast.layers import nice_norm, mlp

from jigsaw import attention

class TransformerBlock(torch.nn.Module):
    """Transformer block with windowed attention.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    output_dim : int
        Output dimension.
    heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    device : torch.device | None, optional
        Device to place tensors on, by default None.
    dtype : torch.dtype, optional
        Data type of parameters, by default torch.float32.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        noise_dim: int = 32,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.heads = heads
        self.out_dim = out_dim
        self.noise_dim = noise_dim

        self.device = device
        self.dtype = dtype

        # import attention from jigsaw module
        # Performs QKV mapping, flash attention with sdpa, head merging with a linear layer.
        self.attention_block = attention.Attention(
            self.in_dim, self.embed_dim, self.out_dim, self.heads, self.device, dropout_p=dropout
        )

        self.layer_norm1 = torch.nn.LayerNorm(self.out_dim, device=device)
        self.layer_norm2 = torch.nn.LayerNorm(self.out_dim, device=device)

        self.MLP = mlp.MLP(
            self.out_dim,
            self.out_dim,
            self.out_dim,
            device=self.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.attention_block(self.layer_norm1(x)) + x
        x = self.MLP(self.layer_norm2(x)) + x

        return x
    
class NiceTransformerBlock(torch.nn.Module):
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
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        noise_dim: int = 32,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.heads = heads
        self.out_dim = out_dim
        self.noise_dim = noise_dim

        self.device = device
        self.dtype = dtype

        # import attention from jigsaw module
        # Performs QKV mapping, flash attention with sdpa, head merging with a linear layer.
        self.attention_block = attention.Attention(
            self.in_dim, self.embed_dim, self.out_dim, self.heads, self.device, dropout_p=dropout
        )

        self.layer_norm1 = nice_norm.NiceNorm(self.out_dim, self.noise_dim, device=device).to(
            self.device
        )
        self.layer_norm2 = nice_norm.NiceNorm(self.out_dim, self.noise_dim, device=device).to(
            self.device
        )

        self.MLP = mlp.MLP(
            self.out_dim,
            self.out_dim,
            self.out_dim,
            device=self.device
        )

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
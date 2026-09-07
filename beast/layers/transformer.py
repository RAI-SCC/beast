"""Definition of Transformer layers."""

import itertools
import os

import torch
import torch.distributed as dist
from jigsaw import attention, hooks  # type: ignore

from beast.layers import mlp, nice_norm

USE_TE_DPA = os.getenv("BEAST_USE_TE_DPA", "False") in ["True", "1"]


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
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize Transformer.

        in_dim: int
            Input dimension of the data.
        embed_dim: int
            Embedding dimension of attention.
        out_dim: int
            Output dimension of attention.
        heads: int
            Number of global heads in attention.
        dropout: float
            Dropout fraction.
        device: torch.device
            Device that the computations are occuring on.
        bayesian: bool
            Flag for whether the model is Bayesian.
        qk_norm: bool
            Flag for whether to use qk norms in the attention blocks.
        dtype: torch.dtype
            Data type of the model and input data.
        """
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
            self.in_dim,
            self.embed_dim,
            self.out_dim,
            self.heads,
            self.device,
            dropout_p=dropout,
            qk_norm=qk_norm,
        )

        self.layer_norm1 = torch.nn.LayerNorm(self.out_dim, device=device)
        self.layer_norm2 = torch.nn.LayerNorm(self.out_dim, device=device)

        self.mlp = mlp.MLP(self.out_dim, self.out_dim, self.out_dim, device=self.device)

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
        x = self.mlp(self.layer_norm2(x)) + x

        return x


class NiceTransformer(torch.nn.Module):
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
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize DistributedNICETransformer.

        in_dim: int
            Input dimension of the data.
        embed_dim: int
            Embedding dimension of attention.
        out_dim: int
            Output dimension of attention.
        heads: int
            Number of global heads in attention.
        noise_dim: int
            Global dimension for the noise.
        dropout: float
            Dropout fraction.
        device: torch.device
            Device that the computations are occuring on.
        bayesian: bool
            Flag for whether the model is Bayesian.
        qk_norm: bool
            Flag for whether to use qk norms in the attention blocks.
        dtype: torch.dtype
            Data type of the model and input data.
        """
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.heads = heads
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.qk_norm = qk_norm
        self.device = device
        self.dtype = dtype

        # import attention from jigsaw module
        # Performs QKV mapping, flash attention with sdpa, head merging with a linear layer.
        self.attention_block = attention.Attention(
            self.in_dim,
            self.embed_dim,
            self.out_dim,
            self.heads,
            self.device,
            dropout_p=dropout,
            qk_norm=self.qk_norm,
        )

        self.layer_norm1 = nice_norm.NiceNorm(
            self.out_dim, self.noise_dim, device=device
        ).to(self.device)
        self.layer_norm2 = nice_norm.NiceNorm(
            self.out_dim, self.noise_dim, device=device
        ).to(self.device)

        self.MLP = mlp.MLP(self.out_dim, self.out_dim, self.out_dim, device=self.device)

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


class DistributedTransformer(torch.nn.Module):
    """
    Windowed Transformer across multiple processes.

    The spatial domain is split across domain-parallel instances,
    and the channel domain retains domain and tensor parallel approaches.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize DistributedNICETransformer.

        in_dim: int
            Input dimension of the data.
        embed_dim: int
            Embedding dimension of attention.
        out_dim: int
            Output dimension of attention.
        heads: int
            Number of global heads in attention.
        dropout: float
            Dropout fraction.
        local_rank: int
            Rank of the DTP group.
        device: torch.device
            Device that the computations are occuring on.
        model_group: dist.ProcessGroup
            DTP process group.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        channel_group: dist.ProcessGroup
            Domain- and tensor-parallel process group.
        bayesian: bool
            Flag for whether the model is Bayesian.
        qk_norm: bool
            Flag for whether to use qk norms in the attention blocks.
        dtype: torch.dtype
            Data type of the model and input data.
        """
        super().__init__()

        self.attention_block = attention.DistributedAttention(
            in_dim,
            embed_dim,
            out_dim,
            heads,
            dropout,
            local_rank,
            device,
            spatial_group,
            channel_group,
            bayesian,
            qk_norm=qk_norm,
            ring_attention=False,
            dtype=dtype,
        ).to(device)

        self.layer_norm1 = torch.nn.LayerNorm(
            out_dim // channel_group.size(), device=device
        )
        self.layer_norm2 = torch.nn.LayerNorm(
            out_dim // channel_group.size(), device=device
        )

        self.mlp = mlp.DistributedSynchronizedMLP(
            out_dim,
            out_dim,
            out_dim,
            bias=True,
            dropout=dropout,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
            spatial_group=spatial_group,
            channel_group=channel_group,
            bayesian=bayesian,
        ).to(device)

        for p in itertools.chain(
            self.layer_norm1.parameters(), self.layer_norm2.parameters()
        ):
            p.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(spatial_group)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of distributed windowed transformer."""
        x = self.attention_block(self.layer_norm1(x)) + x
        x = self.mlp(self.layer_norm2(x)) + x
        return x


class DistributedNICETransformer(torch.nn.Module):
    """
    Module for Distributed NICE Transformer.

    The spatial domain is split across domain-parallel instances,
    and the channel domain retains domain and tensor parallel approaches.
    Noise is replicated across all dtp processes.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        noise_dim: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize DistributedNICETransformer.

        in_dim: int
            Input dimension of the data.
        embed_dim: int
            Embedding dimension of attention.
        out_dim: int
            Output dimension of attention.
        heads: int
            Number of global heads in attention.
        noise_dim: int
            Global dimension for the noise.
        dropout: float
            Dropout fraction.
        local_rank: int
            Rank of the DTP group.
        device: torch.device
            Device that the computations are occuring on.
        spatial_group: dist.ProcessGroup
            Domain-parallel process group.
        channel_group: dist.ProcessGroup
            Domain- and tensor-parallel process group.
        bayesian: bool
            Flag for whether the model is Bayesian.
        qk_norm: bool
            Flag for whether to use qk norms in the attention blocks.
        dtype: torch.dtype
            Data type of the model and input data.
        """
        super().__init__()
        self.attention_block = attention.DistributedAttention(
            in_dim,
            embed_dim,
            out_dim,
            heads,
            dropout,
            local_rank,
            device,
            spatial_group,
            channel_group,
            bayesian,
            qk_norm=qk_norm,
            attn_backend='te' if USE_TE_DPA else 'sdpa',
            dtype=dtype,
        ).to(device)
        self.layer_norm1 = nice_norm.DistributedNICENorm(
            out_dim,
            noise_dim,
            local_rank,
            device,
            channel_group,
            spatial_group,
            channel_group,
        ).to(device)
        self.layer_norm2 = nice_norm.DistributedNICENorm(
            out_dim,
            noise_dim,
            local_rank,
            device,
            channel_group,
            spatial_group,
            channel_group,
        ).to(device)

        self.mlp = mlp.DistributedSynchronizedMLP(
            out_dim,
            out_dim,
            out_dim,
            bias=True,
            dropout=dropout,
            local_rank=local_rank,
            device=device,
            model_group=channel_group,
            spatial_group=spatial_group,
            channel_group=channel_group,
            bayesian=bayesian,
        ).to(device)

        for p in itertools.chain(
            self.layer_norm1.parameters(), self.layer_norm2.parameters()
        ):
            p.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(spatial_group)
            )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass for DistribuedNICEWindowedTransformer.

        inputs
        ------
        x: torch.Tensor
            Input data.
        noise: torch.Tensor
            Noise for the layer norm. Noise should be consistent within the model-parallel group.
        """
        x = self.attention_block(self.layer_norm1(x, noise)) + x
        x = self.mlp(self.layer_norm2(x, noise)) + x

        return x

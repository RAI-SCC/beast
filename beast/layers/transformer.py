"""Transformer blocks: plain, NICE-normalized and their distributed variants."""

import itertools
import os
import warnings

import torch
import torch.distributed as dist
from jigsaw import attention, hooks  # type: ignore
from jigsaw.modules.attention.attention import JIGSAW_TE_AVAILABLE

from beast.layers import mlp, nice_norm

USE_TE_DPA = os.getenv("BEAST_USE_TE_DPA", "False") in ["True", "1"]
if USE_TE_DPA and not JIGSAW_TE_AVAILABLE:
    if os.getenv("RANK", "0") == "0":
        warnings.warn(
            "BEAST_USE_TE_DPA is set but Transformer Engine is not available. "
            "Falling back to torch SDPA.",
            UserWarning,
        )
    USE_TE_DPA = False


class TransformerBlock(torch.nn.Module):
    """Pre-norm transformer block with windowed attention.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    embed_dim : int
        Embedding dimension of the attention.
    out_dim : int
        Output dimension.
    heads : int
        Number of attention heads.
    noise_dim : int, default: 32
        Unused.
    dropout : float, default: 0.0
        Attention dropout probability.
    device : torch.device, optional
        Device to place parameters on.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.
    dtype : torch.dtype, default: torch.float32
        Stored on the module but not applied to the parameters.
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
        """Apply attention and the MLP, each with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

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
    in_dim : int
        Input dimension.
    embed_dim : int
        Embedding dimension of the attention.
    out_dim : int
        Output dimension.
    heads : int
        Number of attention heads.
    noise_dim : int, default: 32
        Size of the noise vector for NICE normalization.
    dropout : float, default: 0.0
        Attention dropout probability.
    device : torch.device, optional
        Device to place parameters on.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.
    dtype : torch.dtype, default: torch.float32
        Stored on the module but not applied to the parameters.
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
        """Apply attention and the MLP, each with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        noise : torch.Tensor
            Noise for the NICE normalization.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.attention_block(self.layer_norm1(x, noise)) + x
        x = self.MLP(self.layer_norm2(x, noise)) + x

        return x


class DistributedTransformer(torch.nn.Module):
    """Windowed transformer block sharded across processes.

    The spatial domain is split across the spatial group and the channels are
    split across the channel group.

    Parameters
    ----------
    in_dim : int
        Global input dimension.
    embed_dim : int
        Global embedding dimension of the attention.
    out_dim : int
        Global output dimension.
    heads : int
        Global number of attention heads.
    dropout : float
        Dropout probability.
    local_rank : int
        Rank within the model group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Process group the model is sharded over (spatial x channel).
    spatial_group : dist.ProcessGroup
        Spatial (domain-parallel) process group.
    channel_group : dist.ProcessGroup
        Channel (tensor-parallel) process group.
    bayesian : bool
        Whether the layers are Bayesian.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.
    dtype : torch.dtype, default: torch.float32
        Data type passed to the attention.
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
            ring=False,
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
        """Apply attention and the MLP, each with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Local input shard.

        Returns
        -------
        torch.Tensor
            Local output shard.
        """
        x = self.attention_block(self.layer_norm1(x)) + x
        x = self.mlp(self.layer_norm2(x)) + x
        return x


class DistributedNICETransformer(torch.nn.Module):
    """Windowed transformer block with NICE normalization, sharded across processes.

    The spatial domain is split across the spatial group and the channels are
    split across the channel group. The noise is replicated across the model
    group.

    Parameters
    ----------
    in_dim : int
        Global input dimension.
    embed_dim : int
        Global embedding dimension of the attention.
    out_dim : int
        Global output dimension.
    heads : int
        Global number of attention heads.
    noise_dim : int
        Global size of the noise vector.
    dropout : float
        Dropout probability.
    local_rank : int
        Rank within the model group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Unused; the channel group is used in its place.
    spatial_group : dist.ProcessGroup
        Spatial (domain-parallel) process group.
    channel_group : dist.ProcessGroup
        Channel (tensor-parallel) process group.
    bayesian : bool
        Whether the layers are Bayesian.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.
    dtype : torch.dtype, default: torch.float32
        Data type passed to the attention.
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
            attn_backend="te" if USE_TE_DPA else "sdpa",
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
        """Apply attention and the MLP, each with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Local input shard.
        noise : torch.Tensor
            Noise for the NICE normalization, identical across the model group.

        Returns
        -------
        torch.Tensor
            Local output shard.
        """
        x = self.attention_block(self.layer_norm1(x, noise)) + x
        x = self.mlp(self.layer_norm2(x, noise)) + x

        return x

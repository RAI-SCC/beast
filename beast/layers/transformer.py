"""Transformer blocks: plain, NICE-normalized and their distributed variants."""

import itertools
import os
import warnings

import torch
import torch.distributed as dist
from jigsaw import parallel  # type: ignore
from jigsaw.modules import (  # type: ignore[import-not-found]
    Attention,
    DistributedAttention,
)
from jigsaw.modules.attention import (  # type: ignore[import-not-found]
    JIGSAW_TE_AVAILABLE,
)

from beast.layers import mlp, nice_norm

_TE_DPA_REQUESTED = os.getenv("BEAST_USE_TE_DPA", "False") in ["True", "1"]
if _TE_DPA_REQUESTED and not JIGSAW_TE_AVAILABLE:
    if os.getenv("RANK", "0") == "0":
        warnings.warn(
            "BEAST_USE_TE_DPA is set but Transformer Engine is not available. "
            "Falling back to torch SDPA.",
            UserWarning,
        )

USE_TE_DPA = _TE_DPA_REQUESTED and JIGSAW_TE_AVAILABLE
"""Whether the distributed attention uses Transformer Engine instead of torch SDPA.

Set with the environment variable ``BEAST_USE_TE_DPA`` (``1`` or ``True``); off
by default. Stays off, with a warning, if Transformer Engine is not installed.
"""


def _shard_qk_norm(
    attention: DistributedAttention, spatial_group: dist.ProcessGroup
) -> None:
    """Give each ``JChannel`` shard of ``attention`` its own query and key norm.

    jigsaw shares one norm across all heads, replicated over the spatial and
    the channel group. Models trained before jigsaw ``a1c16e9`` hold one norm
    per channel shard instead, acting on that shard's heads only. Replace the
    norms by unmarked copies and mark those over ``spatial_group`` only, so the
    shards may differ and each gradient is summed over the spatial group alone.
    This ties the model to the ``JChannel`` size it was trained with.

    Parameters
    ----------
    attention : DistributedAttention
        Attention built with ``qk_norm=True``, modified in place.
    spatial_group : dist.ProcessGroup
        The ``JSpatial`` process group.
    """
    for name in ("q_norm", "k_norm"):
        shared = getattr(attention, name)
        own = torch.nn.LayerNorm(
            shared.normalized_shape,
            eps=shared.eps,
            device=shared.weight.device,
            dtype=shared.weight.dtype,
        )
        own.load_state_dict(shared.state_dict())
        for p in own.parameters():
            parallel.mark_replicated(p, spatial_group)
        setattr(attention, name, own)


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
    dropout : float, default: 0.0
        Attention dropout probability.
    device : torch.device, optional
        Device to place parameters on.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.

    Attributes
    ----------
    attention_block : jigsaw.attention.Attention
        Attention, including the query, key, value and output projections.
    layer_norm1 : torch.nn.LayerNorm
        Layer norm applied to the input of `attention_block`.
    layer_norm2 : torch.nn.LayerNorm
        Layer norm applied to the input of `.mlp`.
    mlp : beast.layers.mlp.MLP
        MLP after the attention.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        heads: int,
        dropout: float = 0.0,
        device: torch.device | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.heads = heads
        self.out_dim = out_dim

        self.device = device

        # QKV projection, scaled dot-product attention and output projection
        self.attention_block = Attention(
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

    Attributes
    ----------
    attention_block : jigsaw.attention.Attention
        Attention, including the query, key, value and output projections.
    layer_norm1 : beast.layers.nice_norm.NiceNorm
        NICE normalization applied to the input of `attention_block`.
    layer_norm2 : beast.layers.nice_norm.NiceNorm
        NICE normalization applied to the input of `MLP`.
    MLP : beast.layers.mlp.MLP
        MLP after the attention.
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
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.heads = heads
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.qk_norm = qk_norm
        self.device = device

        # QKV projection, scaled dot-product attention and output projection
        self.attention_block = Attention(
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

    The grid is split over the ``JSpatial`` process group and the channels over
    the ``JChannel`` process group.

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
        Rank of this process in the ``DTP`` process group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Not used by this block. Its MLP layers use
        ``channel_group`` as their model-parallel group.
    spatial_group : dist.ProcessGroup
        The ``JSpatial`` process group.
    channel_group : dist.ProcessGroup
        The ``JChannel`` process group.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.
    shard_qk_norm : bool, default: False
        Give each ``JChannel`` shard its own query and key norm instead of one
        shared by all heads, see :func:`_shard_qk_norm`.
    dtype : torch.dtype, default: torch.float32
        Data type passed to the attention.

    Attributes
    ----------
    attention_block : jigsaw.attention.DistributedAttention
        Attention, including the query, key, value and output projections.
    layer_norm1 : torch.nn.LayerNorm
        Layer norm over this rank's channel shard, applied to the input of
        `attention_block`.
    layer_norm2 : torch.nn.LayerNorm
        Layer norm over this rank's channel shard, applied to the input of
        `.mlp`.
    mlp : beast.layers.mlp.DistributedSynchronizedMLP
        MLP after the attention.
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
        qk_norm: bool = False,
        shard_qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.attention_block = DistributedAttention(
            in_dim,
            embed_dim,
            out_dim,
            heads,
            local_rank=local_rank,
            device=device,
            spatial_group=spatial_group,
            channel_group=channel_group,
            dropout_p=dropout,
            qk_norm=qk_norm,
            ring=False,
            dtype=dtype,
        ).to(device)
        if qk_norm and shard_qk_norm:
            _shard_qk_norm(self.attention_block, spatial_group)

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
            model_group=channel_group,
            spatial_group=spatial_group,
            channel_group=channel_group,
        ).to(device)

        for p in itertools.chain(
            self.layer_norm1.parameters(), self.layer_norm2.parameters()
        ):
            parallel.mark_replicated(p, spatial_group, extend=True)

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
        x = self.attention_block(parallel.synced_call(self.layer_norm1, x)) + x
        x = self.mlp(parallel.synced_call(self.layer_norm2, x)) + x
        return x


class DistributedNICETransformer(torch.nn.Module):
    """Windowed transformer block with NICE normalization, sharded across processes.

    The grid is split over the ``JSpatial`` process group and the channels over
    the ``JChannel`` process group. Every rank receives the same noise.

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
        Rank of this process in the ``DTP`` process group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Not used by this block. Its normalization and MLP layers use
        `channel_group` as their model-parallel group.
    spatial_group : dist.ProcessGroup
        The ``JSpatial`` process group.
    channel_group : dist.ProcessGroup
        The ``JChannel`` process group.
    qk_norm : bool, default: False
        Whether to normalize queries and keys in the attention.
    shard_qk_norm : bool, default: False
        Give each ``JChannel`` shard its own query and key norm instead of one
        shared by all heads, see :func:`_shard_qk_norm`.
    dtype : torch.dtype, default: torch.float32
        Data type passed to the attention.

    Attributes
    ----------
    attention_block : jigsaw.attention.DistributedAttention
        Attention, including the query, key, value and output projections. Uses
        Transformer Engine if `USE_TE_DPA` is set, otherwise torch SDPA.
    layer_norm1 : beast.layers.nice_norm.DistributedNICENorm
        NICE normalization applied to the input of `attention_block`.
    layer_norm2 : beast.layers.nice_norm.DistributedNICENorm
        NICE normalization applied to the input of `.mlp`.
    mlp : beast.layers.mlp.DistributedSynchronizedMLP
        MLP after the attention.
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
        qk_norm: bool = False,
        shard_qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.attention_block = DistributedAttention(
            in_dim,
            embed_dim,
            out_dim,
            heads,
            local_rank=local_rank,
            device=device,
            spatial_group=spatial_group,
            channel_group=channel_group,
            dropout_p=dropout,
            qk_norm=qk_norm,
            attn_backend="te" if USE_TE_DPA else "sdpa",
            dtype=dtype,
        ).to(device)
        if qk_norm and shard_qk_norm:
            _shard_qk_norm(self.attention_block, spatial_group)
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
        ).to(device)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Apply attention and the MLP, each with a residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Local input shard.
        noise : torch.Tensor
            Noise for the NICE normalization, the same on every rank.

        Returns
        -------
        torch.Tensor
            Local output shard.
        """
        x = self.attention_block(self.layer_norm1(x, noise)) + x
        x = self.mlp(self.layer_norm2(x, noise)) + x

        return x

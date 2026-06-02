import torch
import torch.distributed as dist

from jigsaw import attention
from beast.layers import mlp, nice_norm

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
            ring_attention=False,
            dtype=dtype,
        ).to(device)

        self.layer_norm1 = torch.nn.LayerNorm(out_dim // channel_group.size(), device=device)
        self.layer_norm2 = torch.nn.LayerNorm(out_dim // channel_group.size(), device=device)

        self.mlp = mlp.DistributedSynchronizedMLP(
            out_dim,
            out_dim,
            out_dim,
            bias=True,
            dropout=dropout,
            local_rank=local_rank,
            device=device,
            model_group=channel_group,
            sync_group=spatial_group,
            bayesian=bayesian,
        ).to(device)

    def forward(self, x: torch.tensor):
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
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        qk_norm: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize DistributedNICEWindowedTransformer."""
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
            dtype=dtype
        ).to(device)
        self.layer_norm1 = nice_norm.DistributedNICENorm(
            out_dim,
            noise_dim,
            local_rank,
            device,
            spatial_group,
            channel_group,
        ).to(device)
        self.layer_norm2 = nice_norm.DistributedNICENorm(
            out_dim,
            noise_dim,
            local_rank,
            device,
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
            sync_group=spatial_group,
            bayesian=bayesian,
        ).to(device)

    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        """
        Perform forward pass for DistribuedNICEWindowedTransformer.

        inputs
        ------
        x: torch.Tensor
        noise: torch.Tensor
        mask: Union[None, torch.Tensor]
        """
        x = self.attention_block(self.layer_norm1(x, noise)) + x
        x = self.mlp(self.layer_norm2(x, noise)) + x

        return x

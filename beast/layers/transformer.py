import torch
import torch.distributed as dist

from jigsaw import DistributedAttention
from beast.layers import mlp, norm

class DistributedWindowedTransformer(torch.nn.Module):
    """
    Windowed Transformer across multiple processes.

    The spatial domain is split across domain-parallel instances,
    and the channel domain retains domain and tensor parallel approaches.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        dtype: torch.dtype,
        qk_norm: bool = False,
    ):
        super().__init__()

        self.attention_block = DistributedAttention(
            in_dim,
            embed_dim,
            output_dim,
            heads,
            dropout,
            local_rank,
            device,
            parallelism,
            model_group,
            spatial_group,
            channel_group,
            bayesian,
            ring_attention=False,
            qk_norm=qk_norm,
            dtype=dtype,
        ).to(device)

        self.layer_norm1 = torch.nn.LayerNorm(output_dim // channel_group.size())
        self.layer_norm2 = torch.nn.LayerNorm(output_dim // channel_group.size())

        self.mlp = mlp.DistributedSynchronizedMLP(
            output_dim,
            output_dim,
            output_dim,
            bias=True,
            dropout=dropout,
            local_rank=local_rank,
            device=device,
            model_group=channel_group,
            sync_group=spatial_group,
            bayesian=bayesian,
        ).to(device)

    def forward(self, x: torch.tensor, mask=None):
        """Forward pass of distributed windowed transformer."""
        x = self.attention_block(self.layer_norm1(x), mask) + x
        x = self.mlp(self.layer_norm2(x)) + x
        return x


class DistributedNICEWindowedTransformer(torch.nn.Module):
    """
    Module for Distributed NICE Windowed Transformer.

    The spatial domain is split across domain-parallel instances,
    and the channel domain retains domain and tensor parallel approaches.
    Noise is replicated across all dtp processes.
    """

    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        noise_dim: int,
        dropout: float,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool,
        dtype,
    ):
        """Initialize DistributedNICEWindowedTransformer."""
        super().__init__()
        self.noise_dim = noise_dim

        self.attention_block = DistributedAttention(
            x_features,
            embed_dim,
            output_dim,
            heads,
            dropout,
            local_rank,
            device,
            parallelism,
            model_group,
            spatial_group,
            channel_group,
            bayesian,
            dtype,
        ).to(device)
        self.layer_norm1 = norm.DistributedNICENorm(
            output_dim,
            self.noise_dim,
            local_rank,
            device,
            parallelism,
            spatial_group,
            channel_group,
        ).to(device)
        self.layer_norm2 = norm.DistributedNICENorm(
            output_dim,
            self.noise_dim,
            local_rank,
            device,
            parallelism,
            spatial_group,
            channel_group,
        ).to(device)

        self.mlp = mlp.DistributedSynchronizedMLP(
            output_dim,
            output_dim,
            output_dim,
            bias=True,
            dropout=dropout,
            local_rank=local_rank,
            device=device,
            parallelism=parallelism,
            model_group=channel_group,
            sync_group=spatial_group,
            bayesian=bayesian,
        ).to(device)

    @torch.compile
    def forward(self, x: torch.Tensor, noise: torch.Tensor, mask=None):
        """
        Perform forward pass for DistribuedNICEWindowedTransformer.

        inputs
        ------
        x: torch.Tensor
        noise: torch.Tensor
        mask: Union[None, torch.Tensor]
        """
        x = self.attention_block(self.layer_norm1(x, noise), mask) + x
        x = self.mlp(self.layer_norm2(x, noise)) + x

        return x

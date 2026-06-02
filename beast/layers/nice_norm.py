import torch
from beast.layers import mlp

class NiceNorm(torch.nn.Module):
    """NICE normalization layer.

    Parameters
    ----------
    normalized_dim : int
        Shape of the normalized dimension.
    noise_dim : int
        Noise dimension.
    eps : float, optional
        Small value for numerical stability, by default 1e-5.
    device : torch.device | None, optional
        Device to place tensors on, by default None.
    dtype : torch.dtype | None, optional
        Data type of parameters, by default None.
    """

    def __init__(
        self,
        normalized_dim: int,
        noise_dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(
            normalized_dim,
            eps=eps,
            device=device,
            dtype=dtype,
            elementwise_affine=False,
        )

        self.noise_dim = noise_dim
        self.shape = normalized_dim

        self.shift_scale_mlp = mlp.MLP(self.noise_dim, self.shape, self.shape * 2)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass of NICE normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        noise : torch.Tensor
            Noise tensor.

        Returns
        -------
        torch.Tensor
            Normalized and scaled tensor.
        """
        x = self.norm(x)

        scale_shift = self.shift_scale_mlp(noise).unsqueeze(dim=1)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        x = scale * x + shift

        return x

class DistributedNICENorm(torch.nn.Module):
    """NICE Norm inspired by FGN."""

    def __init__(
        self,
        dim,
        noise_dim,
        rank,
        device,
        spatial_group,
        channel_group,
    ):
        """
        Initialize Distributed NICE Norm.

        Inputs
        ------
        dim: int
            Dimension of channels.
        noise_dim: int
            Dimension of noise.
        rank: int
            Local rank.
        device: torch.device
        paralellism: int
        spatial_group: dist.ProcessGroup
        channel_group: dist.ProcessGroup
        """
        super().__init__()

        self.global_dim = dim
        self.noise_dim = noise_dim
        self.rank = rank
        self.device = device
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        # To limit communication, we do grouped layer norm on the local shard
        self.ln = torch.nn.LayerNorm(
            self.global_dim // self.channel_group.size(), elementwise_affine=False
        )

        self.scale_shift = mlp.DistributedMLPNICE(
            in_dim=self.noise_dim,
            hidden_dim=self.global_dim,
            out_dim=self.global_dim * 2,
            bias=True,
            local_rank=self.rank,
            device=self.device,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
        )

    @torch.compile
    def _calc_shift_scale(self, x, scale_shift):
        x = self.ln(x)
        scale_shift = scale_shift.unsqueeze(1)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        x = scale * x + shift
        return x

    def forward(self, x, noise):
        """Forward pass of distributed NICE norm."""
        scale_shift = self.scale_shift(noise)
        x = self._calc_shift_scale(x, scale_shift)

        return x

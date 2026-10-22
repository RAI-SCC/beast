"""NICE normalization layers, single-process and distributed."""

import torch
import torch.distributed as dist

from beast.layers import mlp


class NiceNorm(torch.nn.Module):
    """Layer norm whose scale and shift are computed from a noise vector.

    Parameters
    ----------
    normalized_dim : int
        Size of the normalized dimension.
    noise_dim : int
        Size of the noise vector.
    eps : float, default: 1e-5
        Added to the variance for numerical stability.
    device : torch.device, optional
        Device to place parameters on.
    dtype : torch.dtype, optional
        Data type of the parameters.

    Attributes
    ----------
    norm : torch.nn.LayerNorm
        Layer norm over the last dimension, without learnable scale and shift.
    shape : int
        Size of the normalized dimension, `normalized_dim`.
    shift_scale_mlp : beast.layers.mlp.MLP
        Computes the scale and the shift, concatenated in that order, from the
        noise.
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
        """Normalize `x` and apply the noise-conditioned scale and shift.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, N, normalized_dim)``.
        noise : torch.Tensor
            Noise of shape ``(B, noise_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as `x`.
        """
        x = self.norm(x)

        scale_shift = self.shift_scale_mlp(noise).unsqueeze(dim=1)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        x = scale * x + shift

        return x


class DistributedNICENorm(torch.nn.Module):
    """Noise-conditioned layer norm for a channel-sharded model, inspired by FGN.

    Each rank normalizes its own channel shard, so no communication is needed
    for the statistics. The scale and shift are computed from the noise vector
    by a :class:`~beast.layers.mlp.DistributedMLPNICE`.

    Parameters
    ----------
    dim : int
        Global number of channels.
    noise_dim : int
        Size of the noise vector.
    rank : int
        Rank of this process in the ``DTP`` process group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Passed on to the :class:`~beast.layers.mlp.DistributedMLPNICE`.
    spatial_group : dist.ProcessGroup
        The ``JSpatial`` process group, over which the parameters are replicated.
    channel_group : dist.ProcessGroup
        The ``JChannel`` process group, over which the channels are sharded.

    Attributes
    ----------
    global_dim : int
        Global number of channels, `dim`.
    ln : torch.nn.LayerNorm
        Layer norm over this rank's channel shard, without learnable scale and
        shift.
    scale_shift : beast.layers.mlp.DistributedMLPNICE
        Computes the scale and the shift, concatenated in that order, from the
        noise.
    """

    def __init__(
        self,
        dim: int,
        noise_dim: int,
        rank: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
    ):
        super().__init__()

        self.global_dim = dim
        self.noise_dim = noise_dim
        self.rank = rank
        self.device = device
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        # Normalize each local channel shard separately
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
            model_group=self.model_group,
            spatial_group=self.spatial_group,
            channel_group=self.channel_group,
        )

    @torch.compile
    def _calc_shift_scale(
        self, x: torch.Tensor, scale_shift: torch.Tensor
    ) -> torch.Tensor:
        """Normalize `x` and apply the scale and shift.

        Parameters
        ----------
        x : torch.Tensor
            Local channel shard of shape ``(B, N, channels)``.
        scale_shift : torch.Tensor
            Scale and shift concatenated along the last dimension, of shape
            ``(B, 2 * channels)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as `x`.
        """
        x = self.ln(x)
        scale_shift = scale_shift.unsqueeze(1)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        x = scale * x + shift
        return x

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Normalize the local channel shard and apply the noise-conditioned scale and shift.

        Parameters
        ----------
        x : torch.Tensor
            Local shard of the input, with ``dim // channel_group.size()``
            channels.
        noise : torch.Tensor
            Noise of shape ``(B, noise_dim)``.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as `x`.
        """
        scale_shift = self.scale_shift(noise)
        x = self._calc_shift_scale(x, scale_shift)

        return x

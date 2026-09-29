"""Single-process and distributed MLP blocks."""

from typing import Callable

import torch
import torch.distributed as dist
from jigsaw import hooks, linear  # type: ignore


class MLP(torch.nn.Module):
    """Two-layer perceptron with GELU activation and dropout.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dim : int
        Hidden dimension.
    output_dim : int
        Output dimension.
    bias : bool, default: True
        Whether the linear layers have a bias.
    dropout : float, default: 0.0
        Dropout probability, applied after the activation and after the
        second layer.
    device : torch.device, optional
        Device to place parameters on.
    dtype : torch.dtype, optional
        Data type of the parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.linear1 = torch.nn.Linear(
            input_dim,
            hidden_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.linear2 = torch.nn.Linear(
            hidden_dim, output_dim, bias=bias, device=device, dtype=dtype
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to the last dimension of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(..., input_dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., output_dim)``.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


def make_grad_input_hook(group: dist.ProcessGroup) -> Callable:
    """Create a tensor hook that sums the gradient over ``group``.

    Parameters
    ----------
    group : dist.ProcessGroup
        Process group to all-reduce over, the channel group in practice.

    Returns
    -------
    Callable
        Hook for :meth:`torch.Tensor.register_hook`, returning the reduced
        gradient in place of the original.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)  # sync
        return grad  # replaces grad

    return hook


class DistributedMLPNICE(torch.nn.Module):
    """MLP for the noise input of :class:`~beast.layers.nice_norm.DistributedNICENorm`.

    The first layer is sharded across the channel group and the second is a
    :class:`jigsaw.linear.DistributedXWT`. Parameters are broadcast over the
    spatial group at construction and their gradients are synchronized over it.

    Parameters
    ----------
    in_dim : int
        Global input dimension of the noise.
    hidden_dim : int
        Global hidden dimension. Must be divisible by the channel group size.
    out_dim : int
        Global output dimension.
    bias : bool
        Whether the linear layers have a bias.
    local_rank : int
        Local rank within the model group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Process group the model is sharded over (spatial x channel).
    spatial_group : dist.ProcessGroup
        Spatial process group.
    channel_group : dist.ProcessGroup
        Channel process group.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool,
        local_rank: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        assert hidden_dim % channel_group.size() == 0, (
            "hidden_dim is not divisble by the channel group size."
        )

        self.linear1 = torch.nn.Linear(
            in_dim, hidden_dim // channel_group.size(), bias, device=device
        )

        self.linear2 = linear.DistributedXWT(
            hidden_dim,
            out_dim,
            bias=bias,
            local_rank=local_rank,
            device=device,
            model_group=channel_group,
            row_group=spatial_group,
            column_group=channel_group,
            bayesian=False,
        )

        # The second layer should only be parallel across the channel group
        # thus, all spatial groups need to ensure they remain synchronized
        for p in self.parameters():
            p.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)

        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to the local input.

        Parameters
        ----------
        x : torch.Tensor
            Local input of shape ``(B, x_in, in_dim)``.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        # hook ensures the grad_input is synchronized across the channels in the backward pass
        if x.requires_grad:
            x.register_hook(make_grad_input_hook(self.channel_group))
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class DistributedSynchronizedMLP(torch.nn.Module):
    """MLP sharded across channels and synchronized across space.

    Parameters are broadcast over the spatial group at construction, and
    gradient hooks keep them synchronized during training.

    Parameters
    ----------
    in_dim : int
        Global input dimension.
    hidden_dim : int
        Global hidden dimension.
    out_dim : int
        Global output dimension.
    bias : bool
        Whether the linear layers have a bias.
    dropout : float
        Dropout probability, applied after the activation and after the
        second layer.
    local_rank : int
        Local rank within the model group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Process group the model is sharded over (spatial x channel).
    spatial_group : dist.ProcessGroup
        Spatial process group, over which the model is replicated.
    channel_group : dist.ProcessGroup
        Channel process group, over which the model is tensor parallel.
    bayesian : bool, default: False
        Whether the linear layers are Bayesian.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool,
        dropout: float,
        local_rank: int,
        device: torch.device,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup,
        channel_group: dist.ProcessGroup,
        bayesian: bool = False,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        # pass in global dimensions for linear layers.
        self.linear1 = linear.DistributedXWT(
            in_dim,
            hidden_dim,
            bias=bias,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
            row_group=spatial_group,
            column_group=channel_group,
            bayesian=bayesian,
        )

        self.linear2 = linear.DistributedXWT(
            hidden_dim,
            out_dim,
            bias=bias,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
            row_group=spatial_group,
            column_group=channel_group,
            bayesian=bayesian,
        )

        for p in self.parameters():
            dist.broadcast(p.data, group_src=0, group=self.spatial_group)
            p.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )

        self.activation = torch.nn.GELU()
        self.drop = torch.nn.Dropout(p=dropout)
        self.drop2 = torch.nn.Dropout(p=dropout)
        self.local_rank = local_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to the local input.

        Parameters
        ----------
        x : torch.Tensor
            Local input of shape ``(B, x_in, in_dim)``.

        Returns
        -------
        torch.Tensor
            Local output.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x

"""Definition of MLPs."""

from typing import Callable

import torch
import torch.distributed as dist
from jigsaw import hooks, linear  # type: ignore


class MLP(torch.nn.Module):
    """Multi-layer perceptron.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dim : int
        Hidden dimension.
    output_dim : int
        Output dimension.
    device : torch.device | None, optional
        Device to place tensors on, by default None.
    dtype : torch.dtype | None, optional
        Data type of parameters, by default None.
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
        """Initialize MLP."""
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
        """Forward pass of MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


def make_grad_input_hook(group: dist.ProcessGroup) -> Callable:
    """All-reduce grad_input (activation grad) across the channel group."""

    def hook(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)  # sync
        return grad  # replaces grad

    return hook


class DistributedMLPNICE(torch.nn.Module):
    """Distributed MLP across n processes."""

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
        """
        Initialize the MLP module for NICE Norm.

        Parameters
        ----------
        in_dim : int
            Global input dimension of noise.
        hidden_dim : int
            Global hidden dimension of MLP.
        out_dim : int
            Global output dimension of MLP.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        local_rank : int
            Model local_rank.
        device : torch.device
            Device to perform computation on.
        model_group: torch.ProcessGroup
            Process group of the model. Union of the spatial and channel groups.
        spatial_group: torch.ProcessGroup
            Process group of the spatial ranks.
        channel_group: torch.ProcessGroup
            Process group of the channel ranks.
        """
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
            bias,
            local_rank,
            device,
            model_group=model_group,
            tensor_group=spatial_group,
            domain_group=channel_group,
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
        """
        Forward pass of MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        """
        # hook ensures the grad_input is synchronized across the channels in the backward pass
        if x.requires_grad:
            x.register_hook(make_grad_input_hook(self.channel_group))
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class DistributedSynchronizedMLP(torch.nn.Module):
    """Distributed MLP across channels and synchronized across space.

    Parameters are synchronized upon initialization. Hooks are used to keep
    weights synchronized as training proceeds.
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
        """
        Initialize DistributedSynchronizedMLP Module.

        Parameters
        ----------
        in_dim: int
            input dimension of the data.
        hidden_dim: int
            hidden dimesion of the mlp.
        out_dim: int
            output dimension of the mlp.
        bias: bool
            Flag for whether to use bias.
        dropout: float
            Dropout fraction.
        local_rank: int
            Rank within channel group.
        device: torch.device
            Device to perform computations on.
        model_group: dist.ProcessGroup
            Channel group, where model is sharded.
        spatial_group: dist.ProcessGroup,
            Spatial group, where model is domain-but-not-tensor parallel.
        channel_group: dist.ProcessGroup
            Channel group, where model is tensor parallel.
        bayesian: bool
            Flag for whether the model is Bayesian.
        """
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
            bias,
            local_rank,
            device,
            model_group=model_group,
            tensor_group=spatial_group,
            domain_group=channel_group,
            bayesian=bayesian,
        )

        self.linear2 = linear.DistributedXWT(
            hidden_dim,
            out_dim,
            bias,
            local_rank,
            device,
            model_group=model_group,
            tensor_group=spatial_group,
            domain_group=channel_group,
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
        """
        Forward pass of MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x

"""Single-process and distributed MLP blocks."""

from typing import Callable

import torch
import torch.distributed as dist
from jigsaw import parallel  # type: ignore
from jigsaw.modules import DistributedLinear  # type: ignore[import-not-found]


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

    Attributes
    ----------
    linear1 : torch.nn.Linear
        First layer, from `input_dim` to `hidden_dim`.
    linear2 : torch.nn.Linear
        Second layer, from `hidden_dim` to `output_dim`.
    dropout : torch.nn.Dropout
        Dropout after the activation and after `linear2`.
    activation : torch.nn.GELU
        Activation after `linear1`.
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
        """Apply the MLP to the last dimension of `x`.

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
    """Create a tensor hook that sums the gradient over `group`.

    Parameters
    ----------
    group : dist.ProcessGroup
        Process group to sum the gradient over, usually the ``JChannel`` process
        group.

    Returns
    -------
    Callable
        Hook for :meth:`torch.Tensor.register_hook`, returning the reduced
        gradient in place of the original.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        """Sum a gradient over `group`.

        Parameters
        ----------
        grad : torch.Tensor
            Gradient of the hooked tensor.

        Returns
        -------
        torch.Tensor
            The gradient summed over all ranks of `group`.
        """
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)  # sync
        return grad  # replaces grad

    return hook


class DistributedMLPNICE(torch.nn.Module):
    """MLP for the noise input of :class:`~beast.layers.nice_norm.DistributedNICENorm`.

    The first layer is sharded over the ``JChannel`` process group and the second
    is a :class:`jigsaw.modules.DistributedLinear`. The parameters are replicated
    over the ``JSpatial`` process group: they are broadcast at construction and
    their gradients are synchronized.

    Parameters
    ----------
    in_dim : int
        Global input dimension of the noise.
    hidden_dim : int
        Global hidden dimension. Must be divisible by the size of
        `channel_group`.
    out_dim : int
        Global output dimension.
    bias : bool
        Whether the linear layers have a bias.
    local_rank : int
        Rank of this process in the ``DTP`` process group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Stored on the module. The linear layers use `channel_group` as their
        model-parallel group.
    spatial_group : dist.ProcessGroup
        The ``JSpatial`` process group, over which the parameters are replicated.
    channel_group : dist.ProcessGroup
        The ``JChannel`` process group, over which the layers are sharded.

    Attributes
    ----------
    linear1 : torch.nn.Linear
        First layer, from `in_dim` to this rank's slice of `hidden_dim`.
    linear2 : jigsaw.linear.DistributedXWT
        Second layer, from `hidden_dim` to `out_dim`.
    activation : torch.nn.GELU
        Activation after `linear1`.
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

        self.linear2 = DistributedLinear(
            hidden_dim,
            out_dim,
            bias=bias,
            local_rank=local_rank,
            device=device,
            model_group=channel_group,
            row_group=spatial_group,
            col_group=channel_group,
        )

        # The second layer should only be parallel across the channel group
        # thus, all spatial groups need to ensure they remain synchronized
        for p in self.parameters():
            parallel.mark_replicated(p, self.spatial_group, extend=True)
        parallel.broadcast_replicas(self)

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
        # Sum the input gradient over the channel ranks in the backward pass
        if x.requires_grad:
            x.register_hook(make_grad_input_hook(self.channel_group))
        x = parallel.synced_call(self.linear1, x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class DistributedSynchronizedMLP(torch.nn.Module):
    """MLP sharded across channels and synchronized across space.

    The parameters are replicated over the ``JSpatial`` process group: they are
    broadcast at construction and their gradients are synchronized.

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
        Rank of this process in the ``DTP`` process group.
    device : torch.device
        Device to place parameters on.
    model_group : dist.ProcessGroup
        Model-parallel process group of the linear layers, passed to
        :class:`jigsaw.modules.DistributedLinear`.
    spatial_group : dist.ProcessGroup
        The ``JSpatial`` process group, over which the parameters are replicated.
    channel_group : dist.ProcessGroup
        The ``JChannel`` process group, over which the layers are sharded.

    Attributes
    ----------
    linear1 : jigsaw.linear.DistributedXWT
        First layer, from `in_dim` to `hidden_dim`.
    linear2 : jigsaw.linear.DistributedXWT
        Second layer, from `hidden_dim` to `out_dim`.
    activation : torch.nn.GELU
        Activation after `linear1`.
    drop : torch.nn.Dropout
        Dropout after the activation.
    drop2 : torch.nn.Dropout
        Dropout after `linear2`.
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
    ):
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        # pass in global dimensions for linear layers.
        self.linear1 = DistributedLinear(
            in_dim,
            hidden_dim,
            bias=bias,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
            row_group=spatial_group,
            col_group=channel_group,
        )

        self.linear2 = DistributedLinear(
            hidden_dim,
            out_dim,
            bias=bias,
            local_rank=local_rank,
            device=device,
            model_group=model_group,
            row_group=spatial_group,
            col_group=channel_group,
        )

        # DistributedLinear marks only its bias over row_group (the spatial group);
        # the weight is replicated over it too, and synced in the XWT forward.
        for p in self.parameters():
            parallel.mark_replicated(p, self.spatial_group, extend=True)
        parallel.broadcast_replicas(self)

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

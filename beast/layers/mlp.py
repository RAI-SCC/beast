import torch
import torch.distributed as dist

from jigsaw import linear, hooks


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
            input_dim, hidden_dim, bias=bias, device=device, dtype=dtype
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
        model_group: dist.ProcessGroup = None,
        spatial_group: dist.ProcessGroup = None,
        channel_group: dist.ProcessGroup = None,
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
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes that model is distributed over.
        local_rank : int
            Model local_rank.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.dtp_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        self.linear1 = linear.DistributedXWTNice(
            in_dim,
            hidden_dim // channel_group.size(),
            bias,
            device,
            spatial_group=spatial_group,
            channel_group=channel_group,
        )
        self.linear2 = linear.DistributedXWT(
            hidden_dim // channel_group.size(),
            out_dim,
            bias,
            local_rank,
            device,
            parallelism=channel_group.size(),
            model_group=channel_group,
            spatial_group=spatial_group,
            bayesian=False,
            synchronized_layer=True,
            sync_group=spatial_group,
        )
        
        for p in self.linear1.parameters():
            dist.broadcast(p, group_src=0, group=self.model_group)
            p.register_post_accumulate_grad_hook(hooks.make_grad_sync_hook(self.dtp_group))    
        
        if self.bias is not None:
            self.bias.register_post_accumulate_grad_hook(
                hooks.make_grad_sync_hook(self.spatial_group)
            )
            dist.broadcast(self.bias.data, group_src=0, group=self.spatial_group)

        self.activation = torch.nn.GELU()

    def forward(self, x):
        """
        Forward pass of MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
class DistributedSynchronizedMLP(torch.nn.Module):
    """Distributed MLP across n processes."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool,
        dropout: float,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup = None,
        sync_group: dist.ProcessGroup = None,
        bayesian: bool = False,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group

        
        # in, hidden, out dim are global sizes for two-way model parallelism
        # required to maintain partitioning across final dimension

        # local X has shape [seq_length, in_dim//2]
        self.linear1 = linear.DistributedXWT(
            in_dim // parallelism.size(),
            hidden_dim,
            bias,
            local_rank,
            device,
            parallelism=parallelism,
            model_group=model_group,
            synchronized_layer=True,
            sync_group=sync_group,
            bayesian=bayesian,
        )

        self.linear2 = linear.DistributedXWTSync(
            hidden_dim // 2,
            out_dim,
            bias,
            local_rank,
            device,
            parallelism=parallelism,
            model_group=model_group,
            synchronzied_layer=True,
            sync_group=sync_group,
            bayesian=bayesian,
        )

        self.activation = torch.nn.GELU()
        self.drop = torch.nn.Dropout(p=dropout)
        self.drop2 = torch.nn.Dropout(p=dropout)
        self.local_rank = local_rank

    def forward(self, x):
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


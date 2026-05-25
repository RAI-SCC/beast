"""Expert layers for distributed model parallelism.

This module provides expert weighting and expert layers for distributed
training with expert parallelism.
"""

import math
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import init


class JLR(torch.autograd.Function):
    """Autograd function for batched joint latent representation mixing distributed across n processes."""

    @staticmethod
    def forward(  # noqa: N806
        x: torch.Tensor,
        w: torch.Tensor,
        ep_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        bayesian: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint latent representation mixing across experts.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, xlat, xlon).
        w : torch.Tensor
            Expert weights tensor.
        ep_group : dist.ProcessGroup
            Process group for expert parallelism.
        sp_group : dist.ProcessGroup
            Process group for spatial parallelism.
        bayesian : bool
            Whether using Bayesian variant.
        device : torch.device
            Computation device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Joint latent representation and expert data.
        """
        num_experts = ep_group.size()
        B, channels, xlat, xlon = x.shape  # noqa: N806
        if bayesian:
            num_loc_samples = w.shape[
                0
            ]  # TODO: will break for non-bayesian. Introduce boolean (default False)
        else:
            num_loc_samples = 1

        xc = x.clone()
        dist.all_reduce(xc, op=dist.ReduceOp.AVG, group=sp_group)

        expert_x_list = [
            torch.empty(
                (B, channels, xlat, xlon),
                requires_grad=False,
                device=device,
                dtype=x.dtype,
            )
            for _ in range(num_experts)
        ]
        dist.all_gather(expert_x_list, xc, group=ep_group)

        expert_x = torch.cat(expert_x_list, dim=0)
        del expert_x_list

        # expert_x is of shape (num_experts, batch, channel, xlat, xlon)
        expert_x = expert_x.view(num_experts, B, channels, xlat * xlon)
        if bayesian:
            expert_x = expert_x.unsqueeze(dim=0)

        # expert_x is of shape (1, num_experts, batch, channel, xlat * xlon)
        jlr = expert_x * w
        # jlr is of shape (samples, experts, batch, channel, xlat * xlon)

        if bayesian:
            jlr = torch.sum(jlr, dim=1)
            jlr = jlr.view(num_loc_samples, B, channels, xlat, xlon)
        else:
            jlr = torch.sum(jlr, dim=0)
            jlr = jlr.view(B, channels, xlat, xlon)

        return jlr, expert_x

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], outputs: tuple[Any, ...]
    ) -> None:
        """Set up backward context for JLR autograd function.

        Parameters
        ----------
        ctx : FunctionCtx
            Autograd context.
        inputs : tuple[Any, ...]
            Input tensors (x, w, ep_group, sp_group).
        outputs : tuple[Any, ...]
            Output tensors (jlr, expert_x).
        """
        _, expert_x = outputs
        ctx.mark_non_differentiable(expert_x)
        ctx.save_for_backward(expert_x)

    @staticmethod
    def backward(
        ctx: Any, grad_jlr: torch.Tensor, grad_expert_x: torch.Tensor | None
    ) -> tuple[None, torch.Tensor, None, None, None, None]:
        """Compute gradients for JLR backward pass.

        Parameters
        ----------
        ctx : FunctionCtx
            Autograd context with saved tensors.
        grad_jlr : torch.Tensor
            Gradient w.r.t. jlr output.
        grad_expert_x : torch.Tensor | None
            Gradient w.r.t. expert_x output (unused).

        Returns
        -------
        tuple[None, torch.Tensor, None, None, None, None]
            Gradients for each input (only w gradient is non-None).
        """
        expert_x = ctx.saved_tensors[0]
        # expert_x is of shape (1, num_experts, batch, channel, xlat * xlon)

        # grad_jlr is of shape (samples, batch, channels, xlat, xlon)
        shape = (
            tuple(grad_jlr.shape[0:-4])
            + (1,)
            + tuple(grad_jlr.shape[-4:-2])
            + (grad_jlr.shape[-2] * grad_jlr.shape[-1],)
        )
        grad_jlr = grad_jlr.view(shape)
        # grad_jrl is of shape (samples, 1, batch, channels, xlat * xlon)

        grad_weight = expert_x * grad_jlr
        # grad_weight is of shape (samples, num_experts, batch, channel, xlat * xlon)

        grad_weight = grad_weight.sum(dim=(-1, -3), keepdim=True)
        # grad_weight is of shape (samples, num_experts, 1, channel, 1)

        return None, grad_weight, None, None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: Any,
        x: torch.Tensor,
        w: torch.Tensor,
        ep_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        bayesian: bool,
        device: torch.device,
    ) -> tuple[Any, int]:
        """Vector map over batch dimension.

        Parameters
        ----------
        info : Any
            Vmap info object.
        in_dims : Any
            Input dimensions specification.
        x : torch.Tensor
            Input tensor.
        w : torch.Tensor
            Expert weights.
        ep_group : dist.ProcessGroup
            Expert parallelism group.
        sp_group : dist.ProcessGroup
            Spatial parallelism group.
        bayesian : bool
            Bayesian flag.
        device : torch.device
            Computation device.

        Returns
        -------
        tuple[Any, int]
            Vmapped output and output dimension.
        """
        lr = x.clone().detach()
        local_mean = lr.mean(dim=0)
        out = JLR.apply(local_mean, w, ep_group, sp_group, bayesian, device)

        # out has vmapped dimension at dim 0
        return out, 0


class ExpertWeightingLayer(torch.nn.Module):
    """Layer that applies expert-based weighting to input tensors.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of expert weights tensor.
    ep_group : dist.ProcessGroup
        Process group for expert parallelism.
    sp_group : dist.ProcessGroup
        Process group for spatial parallelism.
    bayesian : bool, optional
        Whether to use Bayesian variant, by default True.
    device : torch.device | None, optional
        Device to place tensors on, by default None.
    dtype : torch.dtype, optional
        Data type of parameters, by default torch.float32.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        ep_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        bayesian: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.ep_group = ep_group
        self.sp_group = sp_group
        self.ep_rank = self.ep_group.rank()
        self.bayesian = bayesian
        self.device = device
        # shape is (experts, batch ,channels, 1 )
        self.weights = torch.nn.parameter.Parameter(
            torch.empty(shape, device=device, dtype=dtype)
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.weights.data[self.ep_rank, :, :, :] = 1.0

        self.get_jlr = JLR.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: N806
        """Apply expert weighting to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channel, xlat, xlon).

        Returns
        -------
        torch.Tensor
            Weighted output tensor.
        """
        # x is of shape (batch, channel, xlat, xlon)
        jlr: torch.Tensor = self.get_jlr(
            x, self.weights, self.ep_group, self.sp_group, self.bayesian, self.device
        )[0]

        return x + jlr


class Expert(torch.nn.Module):
    """Expert module for distributed model parallelism.

    Parameters
    ----------
    model : torch.nn.Module
        Base model to wrap with expert parallelism.
    ep_group : dist.ProcessGroup
        Process group for expert parallelism.
    """

    def __init__(self, model: torch.nn.Module, ep_group: dist.ProcessGroup) -> None:

        super().__init__()
        self.model = model
        self.expert_group = ep_group
        self.channel_group: dist.ProcessGroup = self.model.channel_group  # type: ignore
        self.num_variables: int = self.model.out_channels  # type: ignore

        self.loss_weights = self.make_loss_weights()

    def make_loss_weights(self) -> torch.Tensor:
        """Create loss weights tensor for this expert.

        Returns
        -------
        torch.Tensor
            Loss weights for this expert.
        """
        num_loc_variables = self.num_variables // self.channel_group.size()
        self.weights = torch.zeros(
            (num_loc_variables,),
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )  # this is hard-coded to the 84 variables we have agreed on. Not pretty, but better than string comparisons

        match self.expert_group.rank():
            case 1:
                # surface variable expert: 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'sea_surface_temperature','total_precipitation'
                if dist.get_rank(group=None) == 0:
                    print("Using surface variable expert")
                if self.channel_group.rank() == 0:
                    self.weights[0:6] = 1.0
            case 2:
                if dist.get_rank(group=None) == 0:
                    print("Using wind expert")
                # wind expert: u, v and w at all pressure levels
                if self.channel_group.rank() == 1:
                    self.weights[(45 - num_loc_variables) :] = 1.0
            case 3:
                if dist.get_rank(group=None) == 0:
                    print("Using thermodynamic expert")
                # thermodynamic expert: geopotential, temperature and specific humidity at all pressure levels
                if self.channel_group.rank() == 0:
                    self.weights[6:num_loc_variables] = 1.0
                else:
                    self.weights[: (45 - num_loc_variables)] = 1.0
            case _:
                # universal expert, all variables; default
                if dist.get_rank(group=None) == 0:
                    print("Using default expert")
                self.weights = torch.ones_like(self.weights)

        return self.weights

    def get_loss_weights(self) -> torch.Tensor:
        """Get loss weights for this expert.

        Returns
        -------
        torch.Tensor
            Loss weights tensor.
        """
        return self.loss_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: N806
        """Forward pass through expert model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output from expert model.
        """
        return self.model(x)  # type: ignore

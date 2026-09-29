"""Mixture-of-experts layers for expert-parallel training."""

import math
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import init


class JLR(torch.autograd.Function):
    """Autograd function mixing the experts' latent representations, distributed over the expert group."""

    @staticmethod
    def forward(  # noqa: N806
        x: torch.Tensor,
        w: torch.Tensor,
        ep_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        bayesian: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix the expert inputs with the expert weights.

        Each expert's input is averaged over the spatial group, gathered over
        the expert group and weighted by ``w``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, channels, xlat, xlon)``.
        w : torch.Tensor
            Expert weights.
        ep_group : dist.ProcessGroup
            Expert-parallel process group.
        sp_group : dist.ProcessGroup
            Spatial process group.
        bayesian : bool
            Whether ``w`` carries a leading sample dimension.
        device : torch.device
            Device for the gathered buffers.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The mixed representation and the gathered expert inputs, which are
            not differentiable.
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
        """Save the gathered expert inputs for the backward pass.

        Parameters
        ----------
        ctx : FunctionCtx
            Autograd context.
        inputs : tuple[Any, ...]
            Arguments of :meth:`forward`, unused.
        outputs : tuple[Any, ...]
            Outputs of :meth:`forward`, ``(jlr, expert_x)``.
        """
        _, expert_x = outputs
        ctx.mark_non_differentiable(expert_x)
        ctx.save_for_backward(expert_x)

    @staticmethod
    def backward(
        ctx: Any, grad_jlr: torch.Tensor, grad_expert_x: torch.Tensor | None
    ) -> tuple[None, torch.Tensor, None, None, None, None]:
        """Compute the gradient with respect to the expert weights.

        Parameters
        ----------
        ctx : FunctionCtx
            Autograd context with saved tensors.
        grad_jlr : torch.Tensor
            Gradient with respect to the mixed representation.
        grad_expert_x : torch.Tensor or None
            Gradient with respect to the gathered expert inputs, unused.

        Returns
        -------
        tuple[None, torch.Tensor, None, None, None, None]
            One gradient per input of :meth:`forward`, non-None only for ``w``.
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
        """Vmap rule: average ``x`` over the mapped dimension, then apply :meth:`forward`.

        Parameters
        ----------
        info : Any
            Vmap info object.
        in_dims : Any
            Input dimensions specification.
        x : torch.Tensor
            Input with the mapped dimension first.
        w : torch.Tensor
            Expert weights.
        ep_group : dist.ProcessGroup
            Expert-parallel process group.
        sp_group : dist.ProcessGroup
            Spatial process group.
        bayesian : bool
            Whether ``w`` carries a leading sample dimension.
        device : torch.device
            Device for the gathered buffers.

        Returns
        -------
        tuple[Any, int]
            The outputs of :meth:`forward` and the mapped output dimension, 0.
        """
        lr = x.clone().detach()
        local_mean = lr.mean(dim=0)
        out = JLR.apply(local_mean, w, ep_group, sp_group, bayesian, device)

        # out has vmapped dimension at dim 0
        return out, 0


class ExpertWeightingLayer(torch.nn.Module):
    """Residual layer adding a learned mixture of the experts' inputs.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the expert weights, ``(experts, batch, channels, 1)``.
    ep_group : dist.ProcessGroup
        Expert-parallel process group.
    sp_group : dist.ProcessGroup
        Spatial process group.
    bayesian : bool, default: True
        Whether the weights carry a leading sample dimension.
    device : torch.device, optional
        Device to place parameters on.
    dtype : torch.dtype, default: torch.float32
        Data type of the weights.
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
        """Return ``x`` plus the weighted mixture of the experts' inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, channel, xlat, xlon)``.

        Returns
        -------
        torch.Tensor
            Output with the shape of ``x``, or with a leading sample dimension
            if Bayesian.
        """
        # x is of shape (batch, channel, xlat, xlon)
        jlr: torch.Tensor = self.get_jlr(
            x, self.weights, self.ep_group, self.sp_group, self.bayesian, self.device
        )[0]

        return x + jlr


class Expert(torch.nn.Module):
    """Wrap a model as one expert, with per-variable loss weights by expert rank.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap. Needs ``channel_group`` and ``out_channels`` attributes.
    ep_group : dist.ProcessGroup
        Expert-parallel process group.
    """

    def __init__(self, model: torch.nn.Module, ep_group: dist.ProcessGroup) -> None:
        super().__init__()
        self.model = model
        self.expert_group = ep_group
        self.channel_group: dist.ProcessGroup = self.model.channel_group  # type: ignore
        self.num_variables: int = self.model.out_channels  # type: ignore

        self.loss_weights = self.make_loss_weights()

    def make_loss_weights(self) -> torch.Tensor:
        """Create the loss weights of this expert.

        The weights are hard-coded to the 84-variable layout. Expert rank 1
        covers the surface variables, 2 the winds, 3 the thermodynamic
        variables and any other rank all variables.

        Returns
        -------
        torch.Tensor
            Weights over the local variables.
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
        """Return the loss weights of this expert.

        Returns
        -------
        torch.Tensor
            Weights over the local variables.
        """
        return self.loss_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: N806
        """Apply the wrapped model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the wrapped model.
        """
        return self.model(x)  # type: ignore

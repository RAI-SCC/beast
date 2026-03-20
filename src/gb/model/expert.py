import math

import torch
import torch.distributed as dist
from torch.nn import init


class JLR(torch.autograd.Function):
    """Autograd function for batched joint latent representation mixing distributed across n processes."""

    @staticmethod
    def forward(x, w, ep_group, sp_group, device):
        num_experts = ep_group.size()
        num_samples = sp_group.size()

        B, channels, xlat, xlon = x.shape
        num_loc_samples = w.shape[
            0
        ]  # TODO: will break for non-bayesian. Introduce boolean (default False)

        x /= num_samples
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=sp_group)

        expert_x = [
            torch.empty(
                (B, channels, xlat, xlon),
                requires_grad=False,
                device=device,
                dtype=x.dtype,
            )
            for _ in range(num_experts)
        ]
        dist.all_gather(expert_x, x, group=ep_group)

        expert_x = torch.cat(expert_x, dim=0)

        # expert_x is of shape (num_experts, batch, channel, xlat, xlon)
        expert_x = expert_x.view(1, num_experts, B, channels, xlat * xlon)
        # expert_x is of shape (1, num_experts, batch, channel, xlat * xlon)

        jlr = expert_x * w
        # jlr is of shape (samples, experts, batch, channel, xlat * xlon)

        jlr = torch.sum(jlr, dim=1)
        jlr = jlr.view(num_loc_samples, B, channels, xlat, xlon)

        return jlr, expert_x

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, ep_group, sp_group )
        output: jlr
        """
        _, expert_x = outputs
        ctx.mark_non_differentiable(expert_x)
        ctx.save_for_backward(expert_x)

    @staticmethod
    def backward(ctx, grad_jlr, grad_expert_x: None):
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

        return None, grad_weight, None, None, None

    @staticmethod
    def vmap(info, in_dims, x, w, ep_group, sp_group, device):
        lr = x.clone().detach()
        local_mean = lr.mean(dim=0)

        out = JLR.apply(local_mean, w, ep_group, sp_group, device)

        # out has vmapped dimension at dim 0
        return out, 0


class ExpertWeightingLayer(torch.nn.Module):
    def __init__(self, shape, ep_group, sp_group, device=None, dtype=torch.float32):
        super().__init__()
        self.ep_group = ep_group
        self.sp_group = sp_group
        self.ep_rank = self.ep_group.rank()
        self.device = device
        # shape is (experts, batch ,channels, 1 )
        self.weights = torch.nn.parameter.Parameter(
            torch.empty(shape, device=device, dtype=dtype)
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.weights.data[self.ep_rank, :, :, :] = 1.0

        self.get_jlr = JLR.apply

    def forward(self, x):
        # x is of shape (batch, channel, xlat, xlon)
        jlr = self.get_jlr(x, self.weights, self.ep_group, self.sp_group, self.device)[
            0
        ]

        return x + jlr


class Expert(torch.nn.Module):
    def __init__(self, model, ep_group):

        super().__init__()
        self.model = model
        self.expert_group = ep_group
        self.channel_group = self.model.channel_group
        self.num_variables = self.model.out_channels

        self.loss_weights = self.make_loss_weights()

    def make_loss_weights(self) -> torch.Tensor:
        num_loc_variables = self.num_variables // self.channel_group.size()
        self.weights = torch.zeros(
            (num_loc_variables,),
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )  # this is hard-coded to the 84 variables we have agreed on. Not pretty, but better than string comparisons

        match self.expert_group.rank():
            case 1:
                # surface variable expert: 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'sea_surface_temperature','total_precipitation'
                if self.channel_group.rank() == 0:
                    self.weights[0:6] = 1.0
            case 2:
                # wind expert: u, v and w at all pressure levels
                if self.channel_group.rank() == 1:
                    self.weights[(45 - num_loc_variables) :] = 1.0
            case 3:
                # thermodynamic expert: geopotential, temperature and specific humidity at all pressure levels
                if self.channel_group.rank() == 0:
                    self.weights[6:num_loc_variables] = 1.0
                else:
                    self.weights[: (45 - num_loc_variables)] = 1.0

            case _:
                # universal expert, all variables; default
                self.weights = torch.ones_like(self.weights)

        return self.weights

    def get_loss_weights(self) -> torch.Tensor:
        return self.loss_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

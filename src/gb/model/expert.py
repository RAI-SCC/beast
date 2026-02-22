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

        # lr = x.clone().detach()
        B, channels, xlat, xlon = x.shape

        x /= num_samples
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=sp_group)

        expert_x = [
            torch.empty((B, channels, xlat, xlon), requires_grad=False, device=device)
            for _ in range(num_experts)
        ]
        expert_x = dist.all_gather(expert_x, x, group=sp_group)

        expert_x = torch.cat(expert_x, dim=0)

        # # expert_x is of shape (num_experts, batch, channel, xlat, xlon)
        expert_x = expert_x.view(num_experts, B, channels, xlat * xlon)
        # # expert_x is of shape (num_experts, batch, channel, xlat* xlon)

        jlr = expert_x * w
        # #print(f"#4. Rank {dist.get_rank(group=None)} x.shape={x.shape}")

        jlr = torch.sum(jlr, dim=0)
        # jlr is of shape (batch, channel, xlat* xlon)
        jlr = jlr.view(B, channels, xlat, xlon)

        return jlr

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx: FunctionCtx
        inputs: tuple(x, w, ep_group, sp_group )
        output: jlr
        """
        x, w, ep_group, sp_group = inputs
        ctx.save_for_backward(x, w)

    @staticmethod
    def backward(ctx, grad_jlr):
        x, w = ctx.saved_tensors

        grad_weight = x * grad_jlr

        return None, grad_weight, None, None, None

    @staticmethod
    def vmap(info, in_dims, x, w, ep_group, sp_group):
        lr = x.clone().detach()
        print(f"Under Vmap, x.shape is {lr.shape}")
        local_mean = lr.mean(dim=0)

        out = JLR.apply(local_mean, w, ep_group, sp_group)

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
            torch.empty(shape, device=device, dtype=torch.float32)
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        print(
            f"Rank {dist.get_rank(group=None)} expert weights are initialized , {self.weights.shape}"
        )
        self.weights.data[self.ep_rank, :, :, :] = 1.0

        self.get_jlr = JLR.apply

    def forward(self, x):
        # x is of shape (batch, channel, xlat, xlon)

        jlr = self.get_jlr(x, self.weights, self.ep_group, self.sp_group, self.device)
        x = x + jlr

        print(f"End of expert: Rank {dist.get_rank(group=None)} x.shape={x.shape}")

        return x


class Expert(torch.nn.Module):
    def __init__(self, model, ep_group):
        self.model = model
        self.process_group = ep_group

        self.loss_weights = self.make_loss_weights()

    def make_loss_weights(self) -> torch.Tensor:
        self.weights = torch.zeros(
            (84,), dtype=self.model.dtype, device=self.model.device
        )  # this is hard-coded to the 84 variables we have agreed on. Not pretty, but better than string comparisons

        match self.pg.rank():
            case 0:
                # surface variable expert: 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'sea_surface_temperature','total_precipitation'
                self.weights[0:6] = 1.0
            case 1:
                # wind expert: u, v and w at all pressure levels
                self.weights[45:] = 1.0
            case 2:
                # thermodynamic expert: geopotential, temperature and specific humidity at all pressure levels
                self.weights[6:45] = 1.0
            case _:
                # universal expert, all variables; default
                self.weights = self.ones_like(self.weights)

        return self.weights

    def get_loss_weights(self) -> torch.Tensor:
        return self.loss_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

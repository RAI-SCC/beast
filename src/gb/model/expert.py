import math

import torch
import torch.distributed as dist
from torch.nn import init


class ExpertWeightingLayer(torch.nn.Module):
    def __init__(self, shape, ep_group, sp_group, device=None, dtype=torch.float32):
        super().__init__()
        self.ep_group = ep_group
        self.sp_group = sp_group
        # rank = self.ep_group.rank()
        # shape is (experts, batch ,channels, 1 )
        self.weights = torch.nn.parameter.Parameter(
            torch.empty(shape, device=device, dtype=torch.float32)
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        # self.weights[rank,:,:,:] = 1.0

    def forward(self, x):
        # x is of shape (batch, channel, xlat, xlon)
        B, channels, xlat, xlon = x.shape
        experts = self.ep_group.size()
        up_num = self.sp_group.size()
        # Average latent representations over all samples
        lr = x.clone().detach()
        lr /= self.sp_group.size()
        local_mean = lr.mean(dim=0)
        mean_lr = local_mean / up_num
        dist.all_reduce(mean_lr, op=dist.ReduceOp.SUM, group=self.sp_group)

        print(f"#3. Rank {dist.get_rank(group=None)} lr={mean_lr.shape}")

        # Gather latent representations across all experts
        expert_x = [torch.zeros_like(mean_lr) for _ in range(experts)]
        dist.all_gather(expert_x, lr)

        print(f"#4. Rank {dist.get_rank(group=None)} lr={expert_x.shape}")

        # Allows differentiability (hopefully), so that the latent representation from the other experts is added to the current LR
        expert_x[self.ep_group.rank()] = x
        expert_x = torch.cat(expert_x, dim=0)

        # expert_x is of shape (experts, batch, channel, xlat, xlon)
        expert_x = expert_x.view(experts, B, channels, xlat * xlon)
        # expert_x is of shape (experts, batch, channel, xlat* xlon)

        expert_x = expert_x * self.weights

        print(f"#5. Rank {dist.get_rank(group=None)} lr={expert_x.shape}")

        y = torch.sum(expert_x, dim=0)
        # y is of shape (batch, channel, xlat* xlon)
        y = y.view(B, channels, xlat, xlon)

        return y


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

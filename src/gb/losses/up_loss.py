from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_blue.vi as vi
from torch_blue.vi.distributions import MeanFieldNormal


class UP_AllGaussLoss(nn.Module):
    def __init__(
        self,
        model,
        loss_weights,
        dataset_size,
        dtp_group,
        up_group,
        dp_group,
        ep_group,
        heat=1.0,
        track=False,
    ):
        super().__init__()

        self.dtp_group = dtp_group
        self.up_group = up_group
        self.dp_group = dp_group
        self.ep_group = ep_group

        self.kl_loss = vi.AnalyticalKullbackLeiblerLoss(
            model,
            MeanFieldNormal(),
            divergence_type=vi.NormalNormalDivergence(),
            dataset_size=dataset_size,
            heat=heat,
            track=track,
        )

        self.loss_weights = loss_weights.reshape(
            (1, 1, -1, 1, 1)
        )  # (batch, variables, lat, lon)
        self.dataset_size = dataset_size

        self.aggregate_statistics = GaussianAggregation.apply

    def data_fitting(self, mean, std, target):
        data_fit = (
            -self.dataset_size
            * self.kl_loss.predictive_distribution.log_prob_from_parameters(
                target, (mean, std)
            )
            .mean(dim=0)
            .sum()
        )

        return data_fit

    def forward(self, model_output, target):
        prior_matching = self.kl_loss.heat * self.kl_loss.prior_matching()
        expert_output = model_output * self.loss_weights
        expert_target = target * self.loss_weights

        mean, std = self.aggregate_statistics(
            expert_output, self.up_group, self.loss_weights
        )
        # Here be expert implementation
        # replace_val = self.loss_weights == 0.0
        # replace_val = replace_val.squeeze(dim=0)
        # replace_val = replace_val.expand(std.shape[0], -1, std.shape[2], std.shape[3])
        # std = torch.where(replace_val, 1.0, std)
        # print(
        #     f"Expert Rank {self.ep_group.rank()}, DP Rank {self.dp_group.rank()}, UP Rank {self.up_group.rank()}, DTP Rank {self.dtp_group.rank()}: mean = {mean.sum()}, std = {std.sum()}"
        # )

        data_fit = self.data_fitting(mean, std, expert_target)

        if self.kl_loss._track and self.kl_loss.log is not None:
            self.kl_loss.log["data_fitting"].append(data_fit.item())
            self.kl_loss.log["prior_matching"].append(prior_matching.item())

        return data_fit + prior_matching

    @property
    def log(self):
        return self.kl_loss.log


class GaussianAggregation(torch.autograd.Function):
    @staticmethod
    def forward(samples, up_process_group, loss_weights):
        loc_sample_num = samples.shape[0]
        up_num = up_process_group.size()
        total_samples = loc_sample_num * up_num

        local_sum_scaled = (samples.sum(dim=0) / total_samples).contiguous()
        mu_handle = dist.all_reduce(
            local_sum_scaled,
            op=dist.ReduceOp.SUM,
            group=up_process_group,
            async_op=True,
        )

        local_sum_sq = ((samples**2).sum(dim=0) / (total_samples - 1)).contiguous()
        var_handle = dist.all_reduce(
            local_sum_sq, op=dist.ReduceOp.SUM, group=up_process_group, async_op=True
        )

        mu_handle.wait()
        mean = local_sum_scaled

        var_handle.wait()
        var = local_sum_sq - (mean**2) * (total_samples / (total_samples - 1))
        var = torch.abs(var) + 1e-10
        std = torch.sqrt(var)

        replace_val = loss_weights == 0.0
        replace_val = replace_val.squeeze(dim=0)
        replace_val = replace_val.expand(std.shape[0], -1, std.shape[2], std.shape[3])
        std = torch.where(replace_val, 1.0, std)

        return mean, std

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        samples, up_process_group, loss_weights = inputs
        mean, std = output

        # do we need to save the loss-weights as well???
        ctx.up_process_group = up_process_group
        ctx.save_for_backward(samples, mean, std)

    @staticmethod
    def backward(ctx: Any, grad_mean, grad_std) -> Any:
        up_process_group = ctx.up_process_group
        samples, mean, std = ctx.saved_tensors
        # print(f"Rank {dist.get_rank(group=None)} samples = {samples.sum()}, mean = {mean.sum()}, std = {std.sum()}")

        N_loc = samples.shape[0]
        N_glob = N_loc * up_process_group.size()

        # do we need to account for the fact, that we replaced the std elements with loss-weight = 0 with 1?
        mean = mean.unsqueeze(dim=0)
        std = std.unsqueeze(dim=0)
        grad_mean = grad_mean.unsqueeze(dim=0)
        grad_std = grad_std.unsqueeze(dim=0)

        grad_samples = (grad_mean / N_glob) + (grad_std / (N_glob - 1)) * (
            samples - mean
        ) / std

        return grad_samples, None, None

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_blue.vi as vi
from torch_blue.vi.distributions import MeanFieldNormal


class UP_AllGaussLoss(nn.Module):
    def __init__(self, model, dataset_size, process_group, heat=1.0, track=False):
        super().__init__()

        self.pg = process_group

        self.kl_loss = vi.AnalyticalKullbackLeiblerLoss(
            model,
            MeanFieldNormal(),
            divergence_type=vi.NormalNormalDivergence(),
            dataset_size=dataset_size,
            heat=heat,
            track=track,
        )

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

        mean, std = self.aggregate_statistics(model_output, self.pg)
        # Here be expert implementation
        # std = torch.where(self.loss_weights.is_nonzero(), std, 1.0)

        data_fit = self.data_fitting(mean, std, target)

        if self.kl_loss._track and self.kl_loss.log is not None:
            self.kl_loss.log["data_fitting"].append(data_fit.item())
            self.kl_loss.log["prior_matching"].append(prior_matching.item())

        return data_fit + prior_matching

    @property
    def log(self):
        return self.kl_loss.log


class GaussianAggregation(torch.autograd.Function):
    @staticmethod
    def forward(samples, process_group):
        loc_sample_num = samples.shape[0]
        up_num = process_group.size()
        total_samples = loc_sample_num * up_num

        local_mean = samples.mean(dim=0)
        local_std = samples.std(dim=0)

        mean = local_mean / up_num
        dist.all_reduce(mean, op=dist.ReduceOp.SUM, group=process_group)

        local_contribution = (loc_sample_num / (total_samples - 1)) * (
            (mean - local_mean) ** 2
        )
        local_contribution = local_contribution + (
            (loc_sample_num - 1) / (total_samples - 1)
        ) * (local_std**2)
        dist.all_reduce(local_contribution, op=dist.ReduceOp.SUM, group=process_group)
        std = torch.sqrt(local_contribution)

        return mean, std

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        samples, process_group = inputs
        mean, std = output

        ctx.process_group = process_group
        ctx.save_for_backward(samples, mean, std)

    @staticmethod
    def backward(ctx: Any, grad_mean, grad_std) -> Any:
        process_group = ctx.process_group
        samples, mean, std = ctx.saved_tensors

        N_loc = samples.shape[0]
        N_glob = N_loc * process_group.size()

        mean = mean.unsqueeze(dim=0)
        std = std.unsqueeze(dim=0)
        grad_mean = grad_mean.unsqueeze(dim=0)
        grad_std = grad_std.unsqueeze(dim=0)

        grad_samples = (grad_mean / N_glob) + (grad_std / (N_glob - 1)) * (
            samples - mean
        ) / std

        return grad_samples, None

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_blue.vi as vi
from torch import Tensor
from torch.cuda import nvtx
from torch_blue.vi.distributions import MeanFieldNormal


# @torch.compile(fullgraph=True, dynamic=True)
class UP_AllGaussLoss(nn.Module):
    def __init__(
        self,
        model,
        loss_weights,
        dataset_size,
        up_group,
        dp_group,
        ep_group,
        heat=1.0,
        track=False,
    ):
        super().__init__()

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
        nvtx.range_push("Prior Matching")
        prior_matching = self.kl_loss.heat * self.kl_loss.prior_matching()
        nvtx.range_pop()
        expert_output = model_output * self.loss_weights
        expert_target = target * self.loss_weights

        nvtx.range_push("Aggregate Statistics")
        mean, std = self.aggregate_statistics(
            expert_output, self.up_group, self.loss_weights
        )
        nvtx.range_pop()

        nvtx.range_push("Data Fitting")
        data_fit = self.data_fitting(mean, std, expert_target)
        nvtx.range_pop()

        if self.kl_loss._track and self.kl_loss.log is not None:
            self.kl_loss.log["data_fitting"].append(data_fit.item())
            self.kl_loss.log["prior_matching"].append(prior_matching.item())

        return data_fit + prior_matching

    @property
    def log(self):
        return self.kl_loss.log


class NewUP_AllGaussLoss(nn.Module):
    def __init__(
        self,
        model,
        loss_weights,
        dataset_size,
        up_group,
        dp_group,
        ep_group,
        heat=1.0,
        track=False,
    ):
        super().__init__()

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

        self.aggregate_statistics = FullAggregation.apply

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
        # nvtx.range_push("Prior Matching")
        # prior_matching = self.kl_loss.heat * self.kl_loss.prior_matching()
        # nvtx.range_pop()
        expert_output = model_output * self.loss_weights
        expert_target = target * self.loss_weights

        nvtx.range_push("Aggregate Statistics and Prior Matching (overlapped)")
        mean, std, prior_matching, _, _ = self.aggregate_statistics(
            expert_output,
            self.up_group,
            self.loss_weights,
            *self.kl_loss._get_flat_params(),
        )
        nvtx.range_pop()

        nvtx.range_push("Data Fitting")
        data_fit = self.data_fitting(mean, std, expert_target)
        nvtx.range_pop()

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
        up_num = up_process_group.size()
        total_samples = samples.shape[0] * up_num

        local_sum_result = torch.cat(
            [
                (samples.sum(dim=0) / total_samples),
                ((samples**2).sum(dim=0) / (total_samples - 1)),
            ],
            dim=0,
        )
        ls_shape = local_sum_result.shape

        handle = dist.all_reduce(
            local_sum_result,
            op=dist.ReduceOp.SUM,
            group=up_process_group,
            async_op=True,
        )

        replace_val = loss_weights == 0.0
        replace_val = replace_val.reshape(1, -1, 1, 1)
        # replace_val = replace_val.expand(std.shape[0], -1, std.shape[2], std.shape[3])
        replace_val = replace_val.expand(1, ls_shape[1], ls_shape[2], ls_shape[3])

        handle.wait()
        mean = local_sum_result[0, :, :, :].unsqueeze(dim=0)
        var = local_sum_result[1, :, :, :].unsqueeze(dim=0)

        var = var - (mean**2) * (total_samples / (total_samples - 1))
        var = torch.abs(var) + 1e-10

        std = torch.sqrt(var)

        std = torch.where(replace_val, 1.0, std)

        return mean, std

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        samples, up_process_group, loss_weights = inputs
        mean, std = output

        ctx.up_process_group = up_process_group
        ctx.save_for_backward(samples, mean, std)

    @staticmethod
    def backward(ctx: Any, grad_mean, grad_std) -> Any:
        up_process_group = ctx.up_process_group
        samples, mean, std = ctx.saved_tensors

        N_loc = samples.shape[0]
        N_glob = N_loc * up_process_group.size()

        mean = mean.unsqueeze(dim=0)
        std = std.unsqueeze(dim=0)
        grad_mean = grad_mean.unsqueeze(dim=0)
        grad_std = grad_std.unsqueeze(dim=0)

        grad_samples = (grad_mean / N_glob) + (grad_std / (N_glob - 1)) * (
            samples - mean
        ) / std

        return grad_samples, None, None


class FullAggregation(torch.autograd.Function):
    @staticmethod
    def forward(
        samples,
        up_process_group,
        loss_weights,
        prior_means: Tensor,
        prior_log_stds: Tensor,
        variational_means: Tensor,
        variational_log_stds: Tensor,
    ):
        up_num = up_process_group.size()
        total_samples = samples.shape[0] * up_num

        local_sum_result = torch.cat(
            [
                (samples.sum(dim=0) / total_samples),
                ((samples**2).sum(dim=0) / (total_samples - 1)),
            ],
            dim=0,
        )
        ls_shape = local_sum_result.shape

        handle = dist.all_reduce(
            local_sum_result,
            op=dist.ReduceOp.SUM,
            group=up_process_group,
            async_op=True,
        )

        replace_val = loss_weights == 0.0
        replace_val = replace_val.reshape(1, -1, 1, 1)
        # replace_val = replace_val.expand(std.shape[0], -1, std.shape[2], std.shape[3])
        replace_val = replace_val.expand(1, ls_shape[1], ls_shape[2], ls_shape[3])

        prior_matching, variance_ratio, mse_part_half = gaussian_prior_matching(
            prior_means, prior_log_stds, variational_means, variational_log_stds
        )

        handle.wait()
        mean = local_sum_result[0, :, :, :].unsqueeze(dim=0)
        var = local_sum_result[1, :, :, :].unsqueeze(dim=0)

        var = var - (mean**2) * (total_samples / (total_samples - 1))
        var = torch.abs(var) + 1e-10

        std = torch.sqrt(var)

        std = torch.where(replace_val, 1.0, std)

        return mean, std, prior_matching, variance_ratio, mse_part_half

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        (
            samples,
            up_process_group,
            loss_weights,
            prior_means,
            prior_log_stds,
            variational_means,
            variational_log_stds,
        ) = inputs
        mean, std, prior_matching, variance_ratio, mse_part_half = output

        ctx.mark_non_differentiable(variance_ratio, mse_part_half)
        ctx.up_process_group = up_process_group
        ctx.save_for_backward(
            samples, mean, std, variational_means, variance_ratio, mse_part_half
        )

    @staticmethod
    def backward(
        ctx: Any, grad_mean, grad_std, grad_pm, grad_vr: None, grad_mse_half: None
    ) -> Any:
        up_process_group = ctx.up_process_group
        samples, mean, std, variational_means, variance_ratio, mse_part_half = (
            ctx.saved_tensors
        )

        N_loc = samples.shape[0]
        N_glob = N_loc * up_process_group.size()

        mean = mean.unsqueeze(dim=0)
        std = std.unsqueeze(dim=0)
        grad_mean = grad_mean.unsqueeze(dim=0)
        grad_std = grad_std.unsqueeze(dim=0)

        grad_samples = (grad_mean / N_glob) + (grad_std / (N_glob - 1)) * (
            samples - mean
        ) / std

        grad_pm = grad_pm.expand_as(variance_ratio)
        grad_var_means = grad_pm * variational_means * mse_part_half
        grad_var_log_stds = grad_pm * (variance_ratio - 1)

        return grad_samples, None, None, None, None, grad_var_means, grad_var_log_stds


@torch.compile
def gaussian_prior_matching(
    prior_means: Tensor,
    prior_log_stds: Tensor,
    variational_means: Tensor,
    variational_log_stds: Tensor,
):
    variational_variance = torch.exp(2 * variational_log_stds)
    prior_variance = torch.exp(
        torch.tensor(2, device=variational_means.device) * prior_log_stds
    )
    variance_ratio = variational_variance / prior_variance
    mse_part_half = (prior_means - variational_means) / prior_variance
    mse_part = (prior_means - variational_means) * mse_part_half

    raw_kl = (mse_part + variance_ratio - variance_ratio.log() - 1) / 2

    prior_matching = raw_kl.sum()

    return prior_matching, variance_ratio, mse_part_half

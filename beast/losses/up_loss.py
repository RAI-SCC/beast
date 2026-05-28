"""Implements loss function for Variational Inference BNNs with Uncertainty Parallelism."""

from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_blue.vi as vi
from torch import Tensor
from torch_blue.vi.distributions import MeanFieldNormal


class UPAllGaussLoss(nn.Module):
    """
    Loss function for Unsupervised Prior matching with Gaussian distributions.

    Parameters
    ----------
    model : vi.VIModule
        The variational inference model.
    dataset_size : int
        Size of the dataset.
    up_group : dist.ProcessGroup
        Process group for Uncertainty Parallelism aggregation.
    loss_weights : torch.Tensor | None, optional
        Weights for the loss calculation if there are multiple experts. Defaults to None.
    heat : float, optional
        Heat / cold posterior parameter for the KL loss. Defaults to 1.0.
    track : bool, optional
        Whether to track loss components. Defaults to False.
    fmodded : bool, optional
        Whether to use the fmodded data fitting method. Defaults to False.
    """

    def __init__(
        self,
        model: vi.VIModule,
        dataset_size: int,
        up_group: dist.ProcessGroup,
        loss_weights: torch.Tensor | None = None,
        heat: float = 1.0,
        track: bool = False,
        fmodded: bool = False,
    ) -> None:
        super().__init__()

        self.up_group = up_group

        self.kl_loss = vi.AnalyticalKullbackLeiblerLoss(
            model,
            MeanFieldNormal(),
            divergence_type=vi.NormalNormalDivergence(),
            dataset_size=dataset_size,
            heat=heat,
            track=track,
        )

        if loss_weights is None:
            self.loss_weights = None
        else:
            self.loss_weights = loss_weights.reshape(
                (1, 1, -1, 1, 1)
            )  # (sample, batch, variables, lat, lon)

        self.dataset_size = dataset_size

        self.aggregate_statistics = MeanStdPriorMatchingAggregation.apply

        self.fmodded = fmodded
        self.data_fitting = (
            self._data_fitting_fmodded if fmodded else self._data_fitting_base
        )

    def _data_fitting_base(self, mean: Tensor, std: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the data fitting loss part.

        Parameters
        ----------
        mean : Tensor
            The mean of the predictive distribution.
        std : Tensor
            The standard deviation of the predictive distribution.
        target : Tensor
            The target values.

        Returns
        -------
        Tensor
            The calculated data fitting loss.
        """
        log_prob = self.kl_loss.predictive_distribution.log_prob_from_parameters(
            target, (mean, std)
        ).mean(dim=0)

        data_fit = -self.dataset_size * log_prob.sum()

        return cast(Tensor, data_fit)

    def _data_fitting_fmodded(
        self, mean: Tensor, std: Tensor, target: Tensor
    ) -> Tensor:
        """
        Calculate the fmodded data fitting loss. Reduces the chance of overflow, but makes interpreting the loss value more difficult.

        Parameters
        ----------
        mean : Tensor
            The mean of the predictive distribution.
        std : Tensor
            The standard deviation of the predictive distribution.
        target : Tensor
            The target values.

        Returns
        -------
        Tensor
            The calculated data fitting loss.
        """
        log_prob = self.kl_loss.predictive_distribution.log_prob_from_parameters(
            target, (mean, std)
        ).mean(dim=0)

        log_prob = torch.fmod(log_prob, 1.0)

        data_fit = -self.dataset_size * log_prob.sum()

        return data_fit

    def split_forward(
        self, model_output: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the data fitting and prior matching losses in a communication-overlapped fashion.

        Parameters
        ----------
        model_output : Tensor
            The output from the Bayesian model.
        target : Tensor
            The ground truth targets.

        Returns
        -------
        tuple[Tensor, Tensor]
            A tuple containing (data_fit, prior_matching).
        """
        if self.loss_weights is not None:
            expert_output = model_output * self.loss_weights
            expert_target = target * self.loss_weights
        else:
            expert_output = model_output
            expert_target = target

        mean, std, prior_matching, _, _ = self.aggregate_statistics(
            expert_output,
            self.up_group,
            self.loss_weights,
            *self.kl_loss._get_flat_params(),
        )

        data_fit = self.data_fitting(mean, std, expert_target)

        if self.kl_loss._track and self.kl_loss.log is not None:
            self.kl_loss.log["data_fitting"].append(data_fit.item())
            self.kl_loss.log["prior_matching"].append(prior_matching.item())

        return data_fit, prior_matching

    def forward(self, model_output: Tensor, target: Tensor) -> Tensor:
        """
        Compute the total loss.

        Parameters
        ----------
        model_output : Tensor
            The output from the model.
        target : Tensor
            The ground truth targets.

        Returns
        -------
        Tensor
            The combined loss.
        """
        data_fit, prior_matching = self.split_forward(model_output, target)

        return data_fit + prior_matching

    @property
    def log(self) -> Any:
        """Get the log from the KL loss."""
        return self.kl_loss.log


class MeanStdPriorMatchingAggregation(torch.autograd.Function):
    """Autograd function to aggregate statistics and compute prior matching in parallel to the communication."""

    @staticmethod
    def forward(
        samples: Tensor,
        up_process_group: dist.ProcessGroup,
        prior_means: Tensor,
        prior_log_stds: Tensor,
        variational_means: Tensor,
        variational_log_stds: Tensor,
        loss_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass for mean-std / prior matching aggregation.

        Parameters
        ----------
        samples : Tensor
            The samples to aggregate.
        up_process_group : dist.ProcessGroup
            The process group corresponding to Uncertainty Parallelism.
        prior_means : Tensor
            Means of the prior distributions.
        prior_log_stds : Tensor
            Log standard deviations of the prior distributions.
        variational_means : Tensor
            Means of the variational distributions.
        variational_log_stds : Tensor
            Log standard deviations of the variational distributions.
        loss_weights : Tensor | None, optional
            Expert weights to apply to the aggregation. Defaults to None.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            A tuple containing (mean, std, prior_matching, variance_ratio, mse_part_half).
        """
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

        if loss_weights is not None:
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

        if loss_weights is not None:
            std = torch.where(replace_val, 1.0, std)

        return mean, std, prior_matching, variance_ratio, mse_part_half

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        """
        Set up the context for the backward pass.

        Parameters
        ----------
        ctx : Any
            The autograd context.
        inputs : tuple[Any, ...]
            The inputs to the forward pass.
        output : Any
            The output from the forward pass.
        """
        (
            samples,
            up_process_group,
            prior_means,
            prior_log_stds,
            variational_means,
            variational_log_stds,
            loss_weights,
        ) = inputs
        mean, std, prior_matching, variance_ratio, mse_part_half = output

        ctx.mark_non_differentiable(variance_ratio, mse_part_half)
        ctx.up_process_group = up_process_group
        ctx.save_for_backward(samples, mean, std, variance_ratio, mse_part_half)

    @staticmethod
    def backward(
        ctx: Any,
        grad_mean: Tensor,
        grad_std: Tensor,
        grad_pm: Tensor,
        grad_vr: None,
        grad_mse_half: None,
    ) -> tuple[Tensor, None, None, None, Tensor, Tensor, None]:
        """
        Backward pass for mean-std / prior matching aggregation.

        Parameters
        ----------
        ctx : Any
            The autograd context.
        grad_mean : Tensor
            Gradient with respect to the mean.
        grad_std : Tensor
            Gradient with respect to the standard deviation.
        grad_pm : Tensor
            Gradient with respect to the prior matching loss.
        grad_vr : None
            Gradient with respect to variance ratio (unused but necessary for autograd).
        grad_mse_half : None
            Gradient with respect to mse part half (unused but necessary for autograd).

        Returns
        -------
        tuple[Tensor, None, None, None, Tensor, Tensor, None]
            Gradients corresponding to the forward inputs for the samples and the variational distribution parameters.
        """
        up_process_group = ctx.up_process_group
        samples, mean, std, variance_ratio, mse_part_half = ctx.saved_tensors

        n_loc = samples.shape[0]
        n_glob = n_loc * up_process_group.size()

        mean = mean.unsqueeze(dim=0)
        std = std.unsqueeze(dim=0)
        grad_mean = grad_mean.unsqueeze(dim=0)
        grad_std = grad_std.unsqueeze(dim=0)

        grad_samples = (grad_mean / n_glob) + (grad_std / (n_glob - 1)) * (
            samples - mean
        ) / std

        grad_pm = grad_pm.expand_as(variance_ratio)
        grad_var_means = grad_pm * mse_part_half
        grad_var_log_stds = grad_pm * (variance_ratio - 1)

        return grad_samples, None, None, None, grad_var_means, grad_var_log_stds, None


@torch.compile
def gaussian_prior_matching(
    prior_means: Tensor,
    prior_log_stds: Tensor,
    variational_means: Tensor,
    variational_log_stds: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute of the KL divergence between two Gaussian distributions for prior matching.

    Parameters
    ----------
    prior_means : Tensor
        Means of the prior distribution.
    prior_log_stds : Tensor
        Log standard deviations of the prior distribution.
    variational_means : Tensor
        Means of the variational distribution.
    variational_log_stds : Tensor
        Log standard deviations of the variational distribution.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A tuple containing (prior_matching, variance_ratio, mse_part_half).
    """
    variational_variance = torch.exp(2 * variational_log_stds)
    prior_variance = torch.exp(
        torch.tensor(2, device=variational_means.device) * prior_log_stds
    )
    variance_ratio = variational_variance / prior_variance
    mse_part_half = (variational_means - prior_means) / prior_variance
    mse_part = (variational_means - prior_means) * mse_part_half

    raw_kl = (mse_part + variance_ratio - variance_ratio.log() - 1) / 2

    prior_matching = raw_kl.sum()

    return prior_matching, variance_ratio, mse_part_half

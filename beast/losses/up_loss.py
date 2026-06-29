"""Implements loss function for Variational Inference BNNs with Uncertainty Parallelism."""

from typing import Any, Optional, cast

import torch
import torch.distributed as dist
import torch_blue.vi as vi
from torch import Tensor
from torch_blue.vi.distributions import MeanFieldNormal


class UPAllGaussLoss(vi.AnalyticalKullbackLeiblerLoss):
    """
    Loss function for Unsupervised Prior matching with Gaussian distributions.

    Parameters
    ----------
    model : vi.VIModule
        The variational inference model.
    dataset_size : int
        Size of the dataset.
    up_group : dist.ProcessGroup, default: None
        Process group for Uncertainty Parallelism aggregation. If None, defaults to dist.group.WORLD.
    feature_weights : Tensor, default: None
        Weights for the loss calculation if there are multiple experts.
    heat : float, default: 1.0
        Heat / cold posterior parameter for the KL loss.
    track : bool, default: False
        Whether to track loss components.
    use_fmod : bool, default: False
        If True apply fmod to predictive log probabilities before summing. This may
        avoid accuracy loss due to float precision when summing and does not impact
        gradients.
    """

    def __init__(
        self,
        model: vi.VIModule,
        dataset_size: int,
        up_group: Optional[dist.ProcessGroup] = None,
        feature_weights: Optional[Tensor] = None,
        heat: float = 1.0,
        track: bool = False,
        use_fmod: bool = False,
    ) -> None:
        super().__init__(
            model,
            MeanFieldNormal(),
            divergence_type=vi.NormalNormalDivergence(),
            dataset_size=dataset_size,
            heat=heat,
            track=track,
        )

        self.up_group = up_group if up_group is not None else dist.group.WORLD

        if feature_weights is None:
            self.feature_weights = None
        else:
            self.feature_weights = feature_weights.reshape(
                (1, 1, -1, 1, 1)
            )  # (sample, batch, variables, lat, lon)

        self.aggregate_statistics = MeanStdPriorMatchingAggregation.apply

        self.use_fmod = use_fmod

    def data_fitting(self, mean: Tensor, std: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the data fitting loss part.

        If ``self.use_fmod`` is True, the fmod is applied to the predictive
        log probabilities to avoid accuracy loss due to float precision when summing.

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
        log_prob = self.predictive_distribution.log_prob_from_parameters(
            target, (mean, std)
        ).mean(dim=0)

        if self.use_fmod:
            log_prob = torch.fmod(log_prob, 1.0)

        data_fit = -self.dataset_size * log_prob.sum()

        return cast(Tensor, data_fit)

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
        Tensor
            The data fitting loss term. Captures how well the model fits the data.
        Tensor
            The prior matching loss term. Captures how well the variational distribution matches the prior distribution.
        """
        if self.feature_weights is not None:
            print(f"Devices: model_output - {model_output.device}, feature_weights - {self.feature_weights.device}")
            expert_output = model_output * self.feature_weights
            expert_target = target * self.feature_weights[0]
        else:
            expert_output = model_output
            expert_target = target

        mean, std, prior_matching, _, _ = self.aggregate_statistics(
            expert_output,
            self.up_group,
            *self._get_flat_params(),
            self.feature_weights,
        )

        data_fit = self.data_fitting(mean, std, expert_target)

        if self._track and self.log is not None:
            self.log["data_fitting"].append(data_fit.item())
            self.log["prior_matching"].append(prior_matching.item())

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
            The combined KL loss consisting of the data fitting and prior matching terms.
        """
        data_fit, prior_matching = self.split_forward(model_output, target)

        return data_fit + cast(float, self.heat) * prior_matching


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
        feature_weights: Optional[Tensor] = None,
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
        feature_weights : Tensor, default: None
            Expert weights to apply to the aggregation.

        Returns
        -------
        Tensor
            The aggregated mean across the UP group.
        Tensor
            The aggregated standard deviation across the UP group.
        Tensor
            The computed prior matching loss.
        Tensor
            The variance ratio used for the calculation of the prior matching loss. This is returned for use in the backward pass.
        Tensor
            The rescaled deviation of the samples members from the mean used for the calculation of the prior matching loss. This is returned for use in the backward pass.
        """
        up_num = up_process_group.size()
        total_samples = samples.shape[0] * up_num

        local_sum_result = torch.stack(
            [
                (samples.sum(dim=0) / total_samples),
                ((samples**2).sum(dim=0) / (total_samples - 1)),
            ],
            dim=0,
        )

        handle = dist.all_reduce(
            local_sum_result,
            op=dist.ReduceOp.SUM,
            group=up_process_group,
            async_op=True,
        )

        prior_matching, variance_ratio, mse_part_half = gaussian_prior_matching(
            prior_means, prior_log_stds, variational_means, variational_log_stds
        )

        handle.wait()
        mean = local_sum_result[0, ...]
        var = local_sum_result[1, ...]

        var = var - (mean**2) * (total_samples / (total_samples - 1))
        var = torch.abs(var) + 1e-10

        std = torch.sqrt(var)

        if feature_weights is not None:
            replace_val = feature_weights == 0.0
            replace_val = replace_val.reshape(1, -1, 1, 1)
            replace_val = replace_val.expand(
                std.shape[0], -1, std.shape[2], std.shape[3]
            )
            std = torch.where(replace_val, 1.0, std)

        return mean, std, prior_matching, variance_ratio, mse_part_half

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], outputs: tuple[Any, ...]
    ) -> None:
        """
        Set up the context for the backward pass.

        Parameters
        ----------
        ctx : Any
            The autograd context.
        inputs : tuple[Any, ...]
            The inputs to the forward pass.
        outputs : tuple[Any, ...]
            The outputs from the forward pass.
        """
        (
            samples,
            up_process_group,
            prior_means,
            prior_log_stds,
            variational_means,
            variational_log_stds,
            feature_weights,
        ) = inputs
        mean, std, prior_matching, variance_ratio, mse_part_half = outputs

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
        Tensor
            Gradient with respect to the input samples.
        None
            No gradient for the up_process_group.
        None
            No gradient for the prior_means.
        None
            No gradient for the prior_log_stds.
        Tensor
            Gradient with respect to the variational_means.
        Tensor
            Gradient with respect to the variational_log_stds.
        None
            No gradient for the feature_weights.
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
    Tensor
        The computed prior matching loss.
    Tensor
        The variance ratio used for the calculation of the prior matching loss. This is returned for use in the backward pass.
    Tensor
        The rescaled deviation of the samples members from the mean used for the calculation of the prior matching loss. This is returned for use in the backward pass.
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

"""KL loss for Bayesian neural networks with uncertainty parallelism (UP).

With UP, the ensemble samples of a Bayesian model are spread over the ``SP``
process group.
"""

from typing import Any, Optional, cast

import torch
import torch.distributed as dist
import torch_blue.vi as vi
from torch import Tensor
from torch_blue.vi.distributions import MeanFieldNormal


class UPAllGaussLoss(vi.AnalyticalKullbackLeiblerLoss):
    """KL loss with Gaussian predictive, variational and prior distributions.

    The predictive mean and standard deviation are estimated from the samples of
    all ranks in ``up_group``, and the prior matching term is computed while
    those statistics are being communicated.

    Parameters
    ----------
    model : vi.VIModule
        The variational inference model.
    dataset_size : int
        Size of the dataset.
    up_group : dist.ProcessGroup, optional
        Process group the ensemble samples are spread over, usually the ``SP``
        process group. Defaults to all ranks.
    sync_group : dist.ProcessGroup, optional
        Process group the gradients of the variational parameters are summed
        over, usually the ``JSpatial`` process group. The prior matching
        gradient is divided by its size, so that the sum counts it once.
    feature_weights : Tensor, optional
        Weights applied to the model output and target, per variable or per
        grid point.
    heat : float, default: 1.0
        Heat / cold posterior parameter of the KL loss.
    track : bool, default: False
        Whether to track the loss components.
    use_fmod : bool, default: False
        Whether to apply ``fmod`` to the predictive log probabilities before
        summing, which avoids float precision loss and does not affect
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
        elif len(feature_weights.shape) == 1:
            self.feature_weights = feature_weights.reshape(
                (1, 1, -1, 1, 1)
            )  # (sample, batch, variables, lat, lon)
        else:
            self.feature_weights = feature_weights.reshape(
                (
                    1,
                    1,
                    feature_weights.shape[-3],
                    feature_weights.shape[-2],
                    feature_weights.shape[-1],
                )
            )  # (sample, batch, variables, lat, lon)
        self.aggregate_statistics = MeanStdPriorMatchingAggregation.apply

        self.use_fmod = use_fmod

        # Stored as a plain int: this loss always needs the dataset size
        self._dataset_size: int = dataset_size

    def data_fitting(
        self,
        mean: Tensor,
        std: Tensor,
        target: Tensor,
        dataset_size: Optional[int] = None,
    ) -> Tensor:
        """Compute the data fitting term.

        Parameters
        ----------
        mean : Tensor
            Mean of the predictive distribution.
        std : Tensor
            Standard deviation of the predictive distribution.
        target : Tensor
            Target values.
        dataset_size : int, optional
            Overrides the dataset size given at construction.

        Returns
        -------
        Tensor
            The negative log likelihood, scaled by the dataset size.
        """
        n_data = self._dataset_size if dataset_size is None else dataset_size
        log_prob = self.predictive_distribution.log_prob_from_parameters(
            target, (mean, std)
        ).mean(dim=0)

        if self.use_fmod:
            log_prob = torch.fmod(log_prob, 1.0)

        data_fit = -n_data * log_prob.sum()

        return cast(Tensor, data_fit)

    def split_forward(
        self,
        model_output: Tensor,
        target: Tensor,
        dataset_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute the data fitting and prior matching terms separately.

        The prior matching term is computed while the sample statistics are
        being communicated.

        Parameters
        ----------
        model_output : Tensor
            Output of the Bayesian model, with a leading sample dimension.
        target : Tensor
            Ground truth targets.
        dataset_size : int, optional
            Overrides the dataset size given at construction.

        Returns
        -------
        data_fit : Tensor
            How well the model fits the data.
        prior_matching : Tensor
            How well the variational distribution matches the prior.
        """
        if self.feature_weights is not None:
            weighted_output = model_output * self.feature_weights
            weighted_target = target * self.feature_weights[0]
        else:
            weighted_output = model_output
            weighted_target = target

        mean, std, prior_matching, _, _ = self.aggregate_statistics(
            weighted_output,
            self.up_group,
            *self._get_flat_params(),
            self.feature_weights,
        )

        data_fit = self.data_fitting(mean, std, weighted_target, dataset_size)

        if self._track and self.log is not None:
            # torch_blue annotates self.log as Dict[str, List[Tensor]] but appends
            # the result of .item() to it, i.e. a float. Mirroring the base class's
            # behaviour means mirroring its annotation mismatch.
            self.log["data_fitting"].append(data_fit.item())  # type: ignore[arg-type]
            self.log["prior_matching"].append(prior_matching.item())  # type: ignore[arg-type]

        return data_fit, prior_matching

    def forward(
        self,
        model_output: Tensor,
        target: Tensor,
        dataset_size: Optional[int] = None,
    ) -> Tensor:
        """Compute the total loss, ``data_fit + heat * prior_matching``.

        Parameters
        ----------
        model_output : Tensor
            Output of the Bayesian model, with a leading sample dimension.
        target : Tensor
            Ground truth targets.
        dataset_size : int, optional
            Overrides the dataset size given at construction.

        Returns
        -------
        Tensor
            The combined loss.
        """
        data_fit, prior_matching = self.split_forward(
            model_output, target, dataset_size
        )

        return data_fit + cast(float, self.heat) * prior_matching


class MeanStdPriorMatchingAggregation(torch.autograd.Function):
    """Autograd function that aggregates sample statistics over the ``SP`` process group.

    The prior matching loss is computed while the second all-reduce is in flight.
    """

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
        """Compute the sample mean and standard deviation and the prior matching loss.

        Parameters
        ----------
        samples : Tensor
            Local samples, along dimension 0.
        up_process_group : dist.ProcessGroup
            Process group the samples are spread over, usually the ``SP``
            process group.
        prior_means : Tensor
            Means of the prior distributions.
        prior_log_stds : Tensor
            Log standard deviations of the prior distributions.
        variational_means : Tensor
            Means of the variational distributions.
        variational_log_stds : Tensor
            Log standard deviations of the variational distributions.
        feature_weights : Tensor, optional
            Weights applied to the samples before aggregation.
        weight_synchronization_factor : int, default: 1
            Size of the process group the parameter gradients are summed over,
            usually the ``JSpatial`` process group. The prior matching gradient
            is divided by it.

        Returns
        -------
        mean : Tensor
            Mean over the samples of all ranks in ``up_process_group``.
        std : Tensor
            Standard deviation over the samples of all ranks in
            ``up_process_group``.
        prior_matching : Tensor
            The prior matching loss.
        variance_ratio : Tensor
            Ratio of variational to prior variance, kept for the backward pass.
        mse_part_half : Tensor
            Deviation of the variational from the prior means, scaled by the
            prior variance. Kept for the backward pass.

        Raises
        ------
        ValueError
            If fewer than 2 samples are available in total across the UP group.
        """
        up_num = up_process_group.size()
        total_samples = samples.shape[0] * up_num
        if total_samples < 2:
            raise ValueError(
                f"MeanStdPriorMatchingAggregation needs at least 2 samples in total to estimate the variance, got {total_samples}."
            )

        # Pass 1: exact global mean. One field, blocking -- we need the result
        # before we can form deviations.
        s1 = samples.sum(dim=0)
        dist.all_reduce(s1, op=dist.ReduceOp.SUM, group=up_process_group)
        mean = s1 / total_samples

        # Pass 2: sum of squared deviations about the shared mean. Because the
        # deviations are centred, there is no cancellation between two large
        # nearly-equal quantities the way there is in E[x^2] - E[x]^2.
        dev = samples - mean
        m2 = (dev * dev).sum(dim=0)
        handle = dist.all_reduce(
            m2,
            op=dist.ReduceOp.SUM,
            group=up_process_group,
            async_op=True,
        )

        prior_matching, variance_ratio, mse_part_half = gaussian_prior_matching(
            prior_means, prior_log_stds, variational_means, variational_log_stds
        )

        handle.wait()
        var = m2 / (total_samples - 1)
        std = torch.sqrt(var.clamp_min(1e-10))

        if feature_weights is not None:
            replace_val = feature_weights == 0.0
            if feature_weights.shape[-1] == 1:
                replace_val = replace_val.reshape(1, -1, 1, 1)
            else:
                replace_val = replace_val.reshape(
                    1,
                    feature_weights.shape[-3],
                    feature_weights.shape[-2],
                    feature_weights.shape[-1],
                )
            replace_val = replace_val.expand(
                std.shape[0], -1, std.shape[2], std.shape[3]
            )
            std = torch.where(replace_val, 1.0, std)

        return mean, std, prior_matching, variance_ratio, mse_part_half

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[Any, ...], outputs: tuple[Any, ...]
    ) -> None:
        """Save the tensors needed by the backward pass.

        Parameters
        ----------
        ctx : Any
            Autograd context.
        inputs : tuple[Any, ...]
            Arguments of :meth:`forward`.
        outputs : tuple[Any, ...]
            Outputs of :meth:`forward`.
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
    ) -> tuple[Tensor, None, None, None, Tensor, Tensor, None, None]:
        """Compute the gradients with respect to the samples and variational parameters.

        Parameters
        ----------
        ctx : Any
            Autograd context.
        grad_mean : Tensor
            Gradient with respect to the mean.
        grad_std : Tensor
            Gradient with respect to the standard deviation.
        grad_pm : Tensor
            Gradient with respect to the prior matching loss.
        grad_vr : None
            Gradient with respect to the variance ratio. Ignored, since the
            variance ratio is not differentiable.
        grad_mse_half : None
            Gradient with respect to ``mse_part_half``. Ignored, since
            ``mse_part_half`` is not differentiable.

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
        None
            No gradient for the weight_synchronization_factor.
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

        return (
            grad_samples * up_process_group.size(),
            None,
            None,
            None,
            grad_var_means,
            grad_var_log_stds,
            None,
        )


@torch.compile
def gaussian_prior_matching(
    prior_means: Tensor,
    prior_log_stds: Tensor,
    variational_means: Tensor,
    variational_log_stds: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the KL divergence of the variational from the prior Gaussians.

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
    prior_matching : Tensor
        The KL divergence, summed over all parameters.
    variance_ratio : Tensor
        Ratio of variational to prior variance, clamped to ``[1e-6, 1e6]``.
    mse_part_half : Tensor
        Deviation of the variational from the prior means, scaled by the prior
        variance.
    """
    variational_variance = torch.exp(2 * variational_log_stds)
    prior_variance = torch.exp(
        torch.tensor(2, device=variational_means.device) * prior_log_stds
    )
    variance_ratio = variational_variance / prior_variance
    variance_ratio = torch.clamp(variance_ratio, min=1e-6, max=1e6)

    mse_part_half = (variational_means - prior_means) / prior_variance
    mse_part = (variational_means - prior_means) * mse_part_half

    raw_kl = (mse_part + variance_ratio - variance_ratio.log() - 1) / 2

    prior_matching = raw_kl.sum()

    return prior_matching, variance_ratio, mse_part_half

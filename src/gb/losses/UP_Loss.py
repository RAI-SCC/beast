import torch
import torch.distributed.nn as dnn
import torch.nn as nn
import torch_blue.vi as vi
from torch_blue.vi.distributions import MeanFieldNormal


class UP_AllGaussLoss(nn.Module):
    def __init__(self, model, loss_weights, dataset_size, process_group, heat=1.0, track=False):
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
        self.loss_weights = loss_weights.reshape((1,1,-1,1,1))# (sample, batch, variables, lat, lon)
        self.dataset_size = dataset_size

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

    def aggregate_statistics(self, model_output):
        loc_sample_num = model_output.shape[0]
        up_num = self.pg.size()
        total_samples = loc_sample_num * up_num

        local_mean = model_output.mean(dim=0) #shape =(batch, variables, lat, lon)
        local_std = model_output.std(dim=0) #shape =(batch, variables, lat, lon)

        mean = local_mean / up_num
        mean = dnn.all_reduce(mean, group=self.pg)

        local_contribution = (loc_sample_num / (total_samples - 1)) * (
            (mean - local_mean) ** 2
        )
        local_contribution = local_contribution + (
            (loc_sample_num - 1) / (total_samples - 1)
        ) * (local_std**2)
        local_contribution = dnn.all_reduce(local_contribution, group=self.pg)
        std = torch.sqrt(local_contribution)

        return mean, std

    def forward(self, model_output, target):
        prior_matching = self.kl_loss.heat * self.kl_loss.prior_matching()
        expert_output = model_output * self.loss_weights
        expert_target = target * self.loss_weights

        mean, std = self.aggregate_statistics(expert_output) #shape =(batch, variables, lat, lon)
        std = torch.where(self.loss_weights.is_nonzero(), std, 1.0)

        data_fit = self.data_fitting(mean, std, expert_target)

        if self.kl_loss._track and self.kl_loss.log is not None:
            self.kl_loss.log["data_fitting"].append(data_fit.item())
            self.kl_loss.log["prior_matching"].append(prior_matching.item())

        return data_fit + prior_matching

    @property
    def log(self):
        return self.kl_loss.log

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch_blue
from Jigsaw import plotting_functions as plot_fns
from torch import Tensor

from gb.losses.UP_Loss import GaussianAggregation
from gb.utils import get_process_group


@torch.no_grad()
def aggregate_crps(local_samples: Tensor, target: Tensor, up_group):
    mae_part = (local_samples - target.expand_as(local_samples)).abs().mean(dim=0)
    dist.all_reduce(mae_part, op=dist.ReduceOp.AVG, group=up_group)

    rank = up_group.rank()
    size = up_group.size()
    samples_per_worker = local_samples.shape[0]
    total_samples = samples_per_worker * size

    spread_part = torch.zeros_like(mae_part)
    buffer = torch.zeros_like(local_samples)
    for i in range(size):
        if rank == i:
            buffer = local_samples.clone()

        dist.broadcast(buffer, src=i, group=up_group)

        if not rank == i:
            spread_part += (local_samples - buffer).mean(dim=0) / (
                2 * total_samples * (total_samples - 1)
            )

        dist.barrier(group=up_group)

    return mae_part - spread_part


class AggregationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.up_group = get_process_group("SP")
        self.jchannel_group = get_process_group("JChannel")
        self.jspatial_group = get_process_group("JSpatial")
        self.dtp_group = get_process_group("DTP")

        self.mean_std_agg = GaussianAggregation.apply

    def forward(self, samples, target, model, dataset_size):
        # TODO this is not averaging over the batch dimension currently; also, this might have to change significantly with the introduction of further Jigsaw stuff
        with torch.no_grad():
            mean, std = self.mean_std_agg(samples)
            crps = aggregate_crps(samples, target, up_group=self.up_group)

            rmse = torch.sqrt((mean - target) ** 2)
            spread_skill = std / rmse

            kl_loss = torch_blue.vi.AnalyticalKullbackLeiblerLoss(
                model,
                torch_blue.vi.distributions.MeanFieldNormal(),
                divergence_type=torch_blue.vi.NormalNormalDivergence(),
                dataset_size=dataset_size,
            )

            data_fit = (
                -dataset_size
                * kl_loss.predictive_distribution.log_prob_from_parameters(
                    target, (mean, std)
                )
                .mean(dim=0)
                .sum()
            )
            dist.all_reduce(data_fit, op=dist.ReduceOp.SUM, group=self.dtp_group)

            prior_matching = kl_loss.prior_matching()
            dist.all_reduce(prior_matching, group=self.dtp_group)

        return data_fit, prior_matching, rmse, crps, spread_skill


def calc_weight(n_lat: int) -> torch.Tensor:
    """
    Return latitude-weighted values for loss function.

    n_lat: int
            number of patches in the latitude dimension

    Returns
    -------
    weight: torch.tensor
        latitude_based weighting factor
    """
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)
    cossum = np.sum(np.cos(latitude))
    weight = n_lat * np.cos(latitude) / cossum
    return torch.tensor(weight)


def get_var_names_and_indices(
    config_dict, rank, parallelism, prepend_variable=["epoch"]
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Return string representation of variable names."""
    var_names = prepend_variable.copy()
    var_names.extend(config_dict["data"]["out_surface_variables"])

    num_surface_vars = len(config_dict["data"]["out_surface_variables"])
    num_levels = len(config_dict["data"]["out_pressure_levels"])

    indices = list(range(num_surface_vars))

    eval_pressure_indices = [
        config_dict["data"]["out_pressure_levels"].index(level)
        for level in config_dict["training"]["eval_pressure_levels"]
    ]

    for i, pvar in enumerate(config_dict["data"]["out_pressure_variables"]):
        var_names.extend(
            pvar + "_" + str(pl)
            for pl in config_dict["training"]["eval_pressure_levels"]
        )

        indices.extend(
            [
                num_surface_vars + i * num_levels + p_index
                for p_index in eval_pressure_indices
            ]
        )

    # deal with indices due to parallelism...
    # halfway split = 36
    splitting_index = (
        int(config_dict["data"]["n_global_variables"]) // 2
    )  # + len(prepend_variable)

    global_var_names, global_indices = var_names.copy(), indices.copy()
    # var_names has shape 10
    match parallelism:
        case 1:
            pass
        case _:  # modulus is always % 2 for splitting of variables
            shift = (
                (config_dict["data"]["n_global_variables"] // 2)
                if (rank % 2 == 1)
                else 0
            )
            pairs = [
                (v, i - shift)
                for v, i in zip(var_names[2:], indices)
                if (i < splitting_index) == (rank % 2 == 0)
            ]
            var_names, indices = map(list, zip(*pairs)) if pairs else ([], [])
            if rank % 2 == 0:
                var_names = prepend_variable + var_names
    return var_names, indices, global_var_names, global_indices


def prepare_plot_data(input_data, output_data, target_data, device, model_group):
    # Gather input data
    # input data has shape [nvar, lat, lon]
    input_data, output, target_data = (
        input_data[0].contiguous(),
        output_data[0].contiguous(),
        target_data[0].contiguous(),
    )

    input_tensor_gather = torch.zeros(4, *input_data.shape).to(device)
    torch.distributed.all_gather_into_tensor(
        input_tensor_gather, input_data, group=model_group
    )

    top_row = torch.cat([input_tensor_gather[0], input_tensor_gather[1]], dim=2)
    bottom_row = torch.cat([input_tensor_gather[2], input_tensor_gather[3]], dim=2)

    # left/right side shape: [1, 720, 720, 72]
    input_data_gathered = torch.cat([top_row, bottom_row], dim=1)[..., :69]

    # gather first sample to rank 0
    output_tensor_gather = torch.zeros(4, *output.shape).to(device)

    torch.distributed.all_gather_into_tensor(
        output_tensor_gather, output, group=model_group
    )

    top_row2 = torch.cat([output_tensor_gather[0], output_tensor_gather[1]], dim=2)
    bottom_row2 = torch.cat([output_tensor_gather[2], output_tensor_gather[3]], dim=2)
    # left/right side shape: [1, 720, 720, 72]
    output_data_gathered = torch.cat([top_row2, bottom_row2], dim=1)[:, :, :69]

    # Gather target data. Currently split between 4 devices as well.
    target_tensor_gather = torch.zeros(4, *target_data.shape).to(device)
    target_data = target_data.contiguous()
    torch.distributed.all_gather_into_tensor(
        target_tensor_gather, target_data, group=model_group
    )

    top_row = torch.cat([target_tensor_gather[0], target_tensor_gather[1]], dim=2)
    bottom_row = torch.cat([target_tensor_gather[2], target_tensor_gather[3]], dim=2)
    # left/right side shape: [1, 720, 720, 72]
    target_data_gathered = torch.cat([top_row, bottom_row], dim=1)[:, :, :69]

    return (
        input_data_gathered.unsqueeze(0),
        output_data_gathered.unsqueeze(0),
        target_data_gathered.unsqueeze(0),
    )


def gen_plots(
    config_dict,
    input_data,
    output_data,
    target_data,
    rank,
    file_path,
    epoch,
    var_names,
    indices,
):
    xlat, xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]

    for index, variable in zip(indices, var_names):
        save_path = f"{file_path}/{variable}_{epoch}.png"
        input_data_plot = input_data[0, ..., index].reshape(xlat, xlon)
        output_data_plot = output_data[0, ..., index].reshape(xlat, xlon)
        target_data_plot = target_data[0, ..., index].reshape(xlat, xlon)
        # TODO: incorporate var_names and indices here;
        # modify plot_fields to only require a single variable
        plot_fns.plot_fields(
            input_data_plot, output_data_plot, target_data_plot, save_path=save_path
        )


def init_evaluation_metric_csvs(rollout_time, epoch, var_names, rank, device):
    rmse, acc = {}, {}

    for rollout in rollout_time:
        # each df has [epoch, rollout_time] as first two values
        rmse[rollout] = torch.zeros(len(var_names)).to(device)
        acc[rollout] = torch.zeros(len(var_names)).to(device)
        if rank % 2 == 0:
            rmse[rollout][0] = epoch
            rmse[rollout][1] = rollout

    return rmse, acc


def get_weights(config_dict, parallelism, rank, device):
    surface_weights = torch.tensor(config_dict["training"]["surface_weights"])
    # [z, q, t, u, v]
    pressure_weights = (
        torch.tensor(config_dict["training"]["pressure_weights"])
        .unsqueeze(1)
        .expand(
            len(config_dict["data"]["out_pressure_variables"]),
            len(config_dict["data"]["out_pressure_levels"]),
        )
        .flatten()
    )
    variable_weights = (
        torch.concat([surface_weights, pressure_weights]).view(
            1, 1, 1, -1
        )  # flatten s.t. variables are in the last dimension
    ).to(device)

    area_weights = (
        calc_weight(config_dict["data"]["xlat"])
        .view(1, config_dict["data"]["xlat"], 1, 1)
        .to(device)
    )

    padding_size = (
        config_dict["data"]["n_global_variables"] - variable_weights.shape[-1]
    )
    splitting_index = config_dict["data"]["n_global_variables"] // 2

    match parallelism:
        case 1:
            variable_weights = variable_weights.to(device)
        case _:
            variable_weights = torch.nn.functional.pad(
                variable_weights, (0, padding_size)
            )
            variable_weights = (
                variable_weights[0, 0, 0, :splitting_index]
                if rank % 2 == 0
                else variable_weights[0, 0, 0, splitting_index:]
            )

    return area_weights, variable_weights


def load_climatology(config_dict, rank):
    parallelism = config_dict["model_parallel"]["parallelism"]
    xlat, xlon = config_dict["data"]["xlat"], config_dict["data"]["xlon"]
    if parallelism < 4:
        climatology = get_climatology(config_dict)[:xlat]
    elif parallelism == 4:
        xlon = xlon // 2
        if rank % 4 in [0, 1]:
            climatology = get_climatology(config_dict)[:xlat, :xlon]
        else:
            climatology = get_climatology(config_dict)[:xlat, xlon:]
    return climatology


def load_constant_masks(parallelism, dataloader, rank):
    # Access dataset
    ds = dataloader.dataset
    # Depending on dataset wrapping, may need have doubled dataset attribute
    ds = ds.dataset if hasattr(ds, "dataset") else ds

    xlon = ds.xlon

    if rank % parallelism in [0, 1]:
        start, end = 0, xlon // 2
    else:
        start, end = xlon // 2, xlon
    land_mask = ds.land_mask[:, :, start:end]
    soil_type = ds.soil_type[:, :, start:end]
    topography = ds.topography[:, :, start:end]
    return [land_mask, soil_type, topography]


def make_visualization(
    config_dict,
    parallelism,
    rank,
    input_data,
    output_data,
    target_data,
    file_path,
    epoch,
    var_names,
    indices,
    device,
    model_group,
):
    match parallelism:
        case 1:
            if rank == 0:  # Generate plots before modifying values for loss calculation
                gen_plots(
                    config_dict,
                    input_data,
                    output_data,
                    target_data,
                    rank,
                    file_path,
                    epoch,
                    var_names=var_names[2:],
                    indices=indices,
                )
        case 2:
            if rank < 2:
                gen_plots(
                    config_dict,
                    input_data,
                    output_data,
                    target_data,
                    rank,
                    file_path,
                    epoch,
                    var_names=var_names[2:] if rank == 0 else var_names,
                    indices=indices,
                )
        case 4:
            # Gather all results onto one device
            if rank < 4:
                input_data, output_data, target_data = prepare_plot_data(
                    input_data, output_data, target_data, device, model_group
                )
            # generate plots
            if rank == 0:
                gen_plots(
                    config_dict,
                    input_data,
                    output_data,
                    target_data,
                    rank,
                    file_path,
                    epoch,
                    var_names=var_names[2:] if rank == 0 else var_names,
                    indices=indices,
                )
    return


def calc_rmse_acc(
    rmse,
    acc,
    config_dict,
    rank,
    parallelism,
    indices,
    target_data,
    output_data,
    rollout,
    device,
    area_weights,
    climatology,
    mean,
    std,
):
    indices = np.array(indices)

    match parallelism:
        case 1:
            index_offset = 2
            norm_offset = 0
        case 2 | 4:
            index_offset = 2 if rank % 2 == 0 else 0
            norm_offset = (
                config_dict["data"]["n_global_variables"] // 2 if rank % 2 == 1 else 0
            )

    rmse[rollout][index_offset:] = rmse[rollout][index_offset:] + calc_mse_variable(
        target_data,
        output_data,
        device,
        area_weights,
        std=std,
        indices=indices,
        norm_offset=norm_offset,
    ).to(torch.float32)

    acc[rollout][index_offset:] = acc[rollout][index_offset:] + calc_acc_variable(
        target_data,
        output_data,
        device,
        area_weights,
        mean=mean,
        std=std,
        climatology=climatology,
        indices=indices,
        norm_offset=norm_offset,
    ).to(torch.float32)

    return rmse, acc


def aggregate_rmse_acc(
    rmse,
    acc,
    total_samples,
    rollout_time,
    parallelism,
    rank,
    ddp_group,
    model_subgroup_col,
):
    match parallelism:
        case 1:
            index_offset = 2
        case 2 | 4 | 8:
            index_offset = 2 if rank % 2 == 0 else 0

    for rollout in rollout_time:
        match parallelism:
            case 1 | 2:
                # Each process has squared error of a certain variable. Need to add over the appropriate DDP groups = global groups.
                torch.distributed.all_reduce(
                    rmse[rollout][index_offset:], group=ddp_group
                )
                torch.distributed.all_reduce(
                    acc[rollout][index_offset:], group=ddp_group
                )
            case 4 | 8:
                # First reduce over local groups --> [0,2], [1,3], [4,6]...
                # Adding MSE of two "partner" ranks together
                torch.distributed.all_reduce(
                    rmse[rollout][index_offset:], group=model_subgroup_col
                )
                # Then reduce over global ddp groups
                torch.distributed.all_reduce(
                    rmse[rollout][index_offset:], group=ddp_group
                )
                torch.distributed.all_reduce(
                    acc[rollout][index_offset:], group=ddp_group
                )

    # divide RMSE and acc by the total number of data items
    for rollout in rollout_time:
        rmse[rollout][index_offset:] = torch.sqrt(
            rmse[rollout][index_offset:] / total_samples[0]
        )
        acc[rollout][index_offset:] = acc[rollout][index_offset:] / total_samples[0]

    return rmse, acc


@torch.no_grad()
def validate(
    config_dict: dict,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    rank: int,
    device: torch.device,
    epoch: int,
    file_path: str,
    mse_file_path: str = "rmse",
    acc_file_path: str = "acc",
    world_size: int = 4,
    model_subgroup_col: torch.distributed.ProcessGroup = None,
    ddp_group: torch.distributed.ProcessGroup = None,
    model_group: torch.distributed.ProcessGroup = None,
):
    if config_dict["model"]["model_parallel"]:
        mse_file_path = f"{mse_file_path}_{rank}.csv"
        acc_file_path = f"{acc_file_path}_{rank}.csv"

    loss = torch.zeros(1).to(device)
    parallelism = config_dict["model_parallel"]["parallelism"]
    rollout_time = config_dict["model"]["latent_rollout"]
    dt = config_dict["data"]["dt"]
    num_total_variables = len(config_dict["data"]["out_surface_variable"]) + len(
        config_dict["data"]["out_pressure_levels"]
    ) * len(
        config_dict["data"]["out_pressure_variables"]
    )  # Surface variables + pressure variables
    rollout_time = (np.arange(rollout_time) + 1) * dt
    var_names, indices, global_var_names, global_indices = get_var_names_and_indices(
        config_dict,
        rank=rank,
        parallelism=parallelism,
        prepend_variable=["epoch", "rollout_time"],
    )

    # initialize dictionaries for rmse and acc
    rmse, acc = init_evaluation_metric_csvs(
        rollout_time, epoch, var_names, rank, device
    )

    # begin timing
    start_validation_time = time.perf_counter()

    # read/calculate latitude-weighted area weights and variable weights
    area_weights, variable_weights = get_weights(config_dict, parallelism, rank, device)

    # Load climatology
    climatology = load_climatology(config_dict, rank)

    # Load masks, now from dataloader object
    masks = load_constant_masks(parallelism, dataloader, rank)

    mean, std = load_averages(config_dict)

    xlon = config_dict["data"]["xlon"]
    xlat = config_dict["data"]["xlat"]
    loss_fn = torch.nn.MSELoss()
    if parallelism == 4:
        xlon = xlon // 2

    model.eval()
    loss = torch.tensor([0]).to(device)

    # begin looping through dataloader
    for batch, data in enumerate(dataloader):
        input_data = data[0].to(device, non_blocking=True).detach()

        for step, rollout in enumerate(rollout_time):
            # Write epoch
            rmse[rollout][0] = int(epoch)
            acc[rollout][0] = int(epoch)
            rmse[rollout][1] = int(step)
            acc[rollout][1] = int(step)

            # Input and target data loading
            target_data = data[1][step].to(device, non_blocking=True)
            output_data = target_data  # setting to dummy value

            if step == 0:
                input_data = input_data
            else:  # TODO: this is only valid for four-way rollout
                input_data = output_data.reshape(2, 720, 720, 36)

                if rank % 2 == 1:
                    # change last 3 dims to constant mask
                    input_data[..., -3:] = masks

            output_data = model(
                input_data, samples=config_dict["training"]["n_samples"]
            ).detach()

            if (
                len(output_data.shape) > 3
            ):  # if probabilistic model, need to take the mean
                output_data = output_data.mean(dim=0)

            output_data = output_data.reshape(
                output_data.shape[0], xlat, xlon, output_data.shape[-1]
            )  # TODO: verify that last dimension can be inferred

            if batch == 0:
                make_visualization(
                    config_dict,
                    parallelism,
                    rank,
                    input_data,
                    output_data,
                    target_data,
                    file_path,
                    epoch,
                    global_var_names if parallelism == 4 else var_names,
                    global_indices if parallelism == 4 else indices,
                    device,
                    model_group,
                )

            rmse, acc = calc_rmse_acc(
                rmse,
                acc,
                config_dict,
                rank,
                parallelism,
                indices,
                target_data,
                output_data,
                rollout,
                device,
                area_weights,
                climatology,
                mean,
                std,
            )

        if parallelism > 1 and rank in [
            1,
            3,
        ]:  # TODO: find proper weights given final variable count
            output_loss = (
                output_data[..., :33] * area_weights * variable_weights[..., :33]
            )
            target_loss = (
                target_data[..., :33] * area_weights * variable_weights[..., :33]
            )

        else:
            output_loss = output_data * area_weights * variable_weights
            target_loss = target_data * area_weights * variable_weights

        # @ Lars TODO: implement the loss function --> the variable indices should be passed into the validation script as an argument
        iter_loss = loss_fn(output_loss, target_loss)

        # TODO: when variables are fixed: scale the loss based on the number of variables on each process
        # 36 is the number of local output variables
        # 69 is the total number of output variables
        if parallelism == 4 and rank in [0, 2]:
            iter_loss = iter_loss * (36 / num_total_variables) * 0.5
        elif parallelism == 4 and rank in [1, 3]:
            iter_loss = iter_loss * (33 / num_total_variables) * 0.5

        loss = loss + iter_loss.detach()

        # total local samples
        total_samples = torch.tensor([0]).to(device)
        total_samples = total_samples + input_data.shape[0]

        if parallelism == 1:
            torch.distributed.all_reduce(total_samples)
            torch.distributed.all_reduce(loss)

            loss /= total_samples
        else:
            ds = dataloader.dataset
            ds = ds.dataset if hasattr(ds, "dataset") else ds
            total_samples = [
                len(ds)
            ]  # TODO: verify consistency with sample parallelism etc
            loss = loss / world_size / total_samples[0]

            if parallelism == 4:
                loss = (
                    loss * world_size
                )  # do not divide by world size, since this is handled

            torch.distributed.all_reduce(loss)

    rmse, acc = aggregate_rmse_acc(
        rmse,
        acc,
        total_samples,
        rollout_time,
        parallelism,
        rank,
        ddp_group,
        model_subgroup_col,
    )

    end_validation_time = time.perf_counter()

    if rank == 0:
        print(
            f"Validation time [min]: {(end_validation_time - start_validation_time) / 60}, validation loss {loss.item()}"
        )
    # write to file
    if (parallelism > 1 and rank < 2) or (parallelism == 1 and rank == 0):
        rmse_file, acc_file = Path(mse_file_path), Path(acc_file_path)
        for rollout in rollout_time:
            rmse_values = dict(zip(var_names, rmse[rollout].cpu().tolist()))
            acc_values = dict(zip(var_names, acc[rollout].cpu().tolist()))

            df_rmse, df_acc = pd.DataFrame([rmse_values]), pd.DataFrame([acc_values])

            (
                df_rmse.to_csv(mse_file_path, mode="a", header=False)
                if rmse_file.is_file()
                else df_rmse.to_csv(mse_file_path)
            )
            (
                df_acc.to_csv(acc_file_path, mode="a", header=False)
                if acc_file.is_file()
                else df_acc.to_csv(acc_file_path)
            )

    return loss.item()


def load_averages(config):
    static_slevel = np.load(f"{config['data']['constant_masks_path']}/surface_zarr.npy")
    static_plevel = np.load(
        f"{config['data']['constant_masks_path']}/pressure_zarr.npy"
    )
    num_surface_variables = len(config["data"]["out_surface_variable"])
    num_pressure_variables = len(config["data"]["out_pressure_levels"]) * len(
        config["data"]["out_pressure_variables"]
    )
    mean_slevel = torch.tensor(static_slevel[0].reshape(num_surface_variables))
    std_slevel = torch.tensor(static_slevel[1].reshape(num_surface_variables))
    mean_plevel = torch.tensor(static_plevel[0].reshape(num_pressure_variables))
    std_plevel = torch.tensor(static_plevel[1].reshape(num_pressure_variables))

    mean = torch.concatenate([mean_slevel, mean_plevel], axis=0)
    std = torch.concatenate([std_slevel, std_plevel], axis=0)
    return mean, std


def get_climatology(config):
    num_pressure_variables = len(config["data"]["out_pressure_levels"]) * len(
        config["data"]["out_pressure_variables"]
    )
    climatology_plevel = torch.tensor(
        np.load(f"{config['data']['constant_masks_path']}/pressure_climatology.npy")
    ).reshape(num_pressure_variables, 721, 1440)
    climatology_slevel = torch.tensor(
        np.load(f"{config['data']['constant_masks_path']}/surface_climatology.npy")
    )

    climatology = torch.concatenate(
        [climatology_slevel, climatology_plevel], axis=0
    ).permute(1, 2, 0)
    # climatology [721, 1440, variables]
    return climatology


def calc_mse_variable(
    target, result, device, area_weights, std, indices, norm_offset=0
):
    n_lon = 1440
    n_lat = area_weights.numel()

    # target shape [n_batch, 721, 1440, vars]
    divisor = torch.sqrt(torch.tensor([n_lon * n_lat])).to(device)

    std_multiplier = std[indices + norm_offset].view(1, 1, 1, len(indices)).to(device)
    area_weights = area_weights.view(1, n_lat, 1, 1)

    mean_squared_error = (
        (target[..., indices] - result[..., indices])
        * std_multiplier.flatten()
        / divisor
    ) ** 2 * area_weights
    # MSE will have shape [n_batch, n_vars, ]
    mean_squared_error = mean_squared_error.sum(dim=(0, 1, 2))

    return mean_squared_error


def calc_acc_variable(
    target, result, device, weights, mean, std, climatology, indices=0, norm_offset=0
):
    mean, std = (
        mean[indices + norm_offset].to(device),
        std[indices + norm_offset].flatten().to(device),
    )

    # climatology has shape [var, 721,1440] mean has shape
    climatology = climatology[:, :, indices + norm_offset].to(device)
    climatology = (climatology - mean.view(1, 1, len(indices))).unsqueeze(0)

    std_multiplier = std.view(1, 1, 1, len(indices)).to(device)

    target_anomoly = (target[:, :, :, indices] * std_multiplier) - climatology
    result_anomoly = (result[:, :, :, indices] * std_multiplier) - climatology

    weights = weights.view(1, weights.numel(), 1, 1)
    # numerator has shape [batch_size, vars, n_lat, n_lon]
    numerator = torch.sum(
        weights * torch.mul(target_anomoly, result_anomoly), dim=(1, 2)
    )
    denom1 = torch.sum(weights * target_anomoly**2, dim=(1, 2))
    denom2 = torch.sum(weights * result_anomoly**2, dim=(1, 2))
    acc_vec = numerator / (torch.sqrt(torch.mul(denom1, denom2)))

    acc_sum = torch.sum(acc_vec, dim=(0))

    return acc_sum  # should be a vector of length (indices)


def calc_acc_variable_parallel(
    target,
    result,
    device,
    weights,
    mean,
    std,
    climatology,
    group,
    rank,
    indices=0,
    norm_offset=0,
):
    # per process: target has shape [n_patches, nlat, nlon//2, nvar/2]
    mean, std = (
        mean[indices + norm_offset].to(device),
        std[indices + norm_offset].flatten().to(device),
    )

    climatology = climatology[:, :, indices + norm_offset].to(device)
    climatology = (climatology - mean.view(1, 1, len(indices))).unsqueeze(0)
    std_multiplier = std.view(1, 1, 1, len(indices)).to(device)

    target_anomoly = (target[:, :, :, indices] * std_multiplier) - climatology
    result_anomoly = (result[:, :, :, indices] * std_multiplier) - climatology

    weights = weights.view(1, weights.numel(), 1, 1)

    numerator = torch.sum(
        weights * torch.mul(target_anomoly, result_anomoly), dim=(1, 2)
    )
    denom1 = torch.sum(weights * target_anomoly**2, dim=(1, 2))
    denom2 = torch.sum(weights * result_anomoly**2, dim=(1, 2))

    torch.distributed.all_reduce(numerator, group=group)
    torch.distributed.all_reduce(denom1, group=group)
    torch.distributed.all_reduce(denom2, group=group)
    acc_vec = numerator / (torch.sqrt(torch.mul(denom1, denom2)))
    acc_sum = torch.sum(acc_vec, dim=0)

    return acc_sum

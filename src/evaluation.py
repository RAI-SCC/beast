from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch_blue

from gb.losses.UP_Loss import GaussianAggregation


@torch.no_grad()
def validate(
    config_dict: dict,
    model: torch.nn.Module,
    n_samples: int,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    results_file_path: str,
    groups: dict,
):
    # get group information
    channel_group = groups["JChannel"]

    rank = dist.get_rank()

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    # load spatial weights
    spatial_weights = get_spatial_weights(config_dict, device)

    # TODO: Possibly include these variable weights if we want an aggregated score for all variables.
    # variable_weights = get_variable_weights(config_dict, rank, device)

    # extract means and std for evaluation
    data_means = torch.from_numpy(dataloader.dataset.norm_values["mean"]).to(device)
    data_std = torch.from_numpy(dataloader.dataset.norm_values["std"]).to(device)

    # climatology
    # TODO: This currently doesn't work - seems like the climatology files do not have enough variables?
    # local_climatology = load_climatology(config_dict, rank, device)

    local_indices, local_names = get_eval_names_and_indicies(config_dict, rank)

    # global names for logging
    if dist.get_world_size(group=channel_group) > 1:
        gathered_names_list = [
            None for _ in range(dist.get_world_size(group=channel_group))
        ]
        dist.all_gather_object(gathered_names_list, local_names, group=channel_group)
        global_names = [name for sublist in gathered_names_list for name in sublist]
    else:
        global_names = local_names

    epoch_results = {}
    num_batches = 0

    # loop through dataloader
    for i, (x, y, start, end) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # prediction
        samples = model(x, samples=n_samples)

        val_results = calculate_aggregate_metrics(
            samples,
            y,
            model,
            len(dataloader.dataset),
            data_means,
            data_std,
            groups,
            spatial_weights,
            local_indices,
            climatology=None,
        )

        # accumulate metrics
        if not epoch_results:
            for k, v in val_results.items():
                epoch_results[k] = v.clone() if isinstance(v, torch.Tensor) else v
        else:
            for k, v in val_results.items():
                epoch_results[k] += v

        num_batches += 1

    if num_batches > 0:
        for k in epoch_results.keys():
            epoch_results[k] /= num_batches

    # save results
    if dist.get_rank() == 0:
        print(f"\n{'=' * 50}")
        print(f"EPOCH {epoch} FINAL VALIDATION METRICS")
        print(f"{'=' * 50}")

        for k, v in epoch_results.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                print(f"\n{k.upper()}:")
                for var_name, val in zip(global_names, v):
                    print(f"  {var_name}: {val.item():.6f}")
            else:
                val = v.item() if isinstance(v, torch.Tensor) else v
                print(f"{k.upper()}: {val:.6f}")
        print(f"{'=' * 50}\n")

        # save to hdf5
        h5_path = Path(results_file_path) / "evaluation_metrics.h5"
        try:
            log_epoch_to_hdf5(epoch_results, global_names, epoch, h5_path)
        except Exception as e:
            print(f"WARNING: HDF5 Logging failed! Error: {e}")
            print("Metrics are still preserved in this Slurm log above.")

    data_fit_loss = epoch_results.get("data_fit")
    prior_matching_loss = epoch_results.get("prior_matching")

    if isinstance(data_fit_loss, torch.Tensor):
        data_fit_loss = data_fit_loss.item()
    if isinstance(prior_matching_loss, torch.Tensor):
        prior_matching_loss = prior_matching_loss.item()

    return data_fit_loss, prior_matching_loss


@torch.no_grad()
def calculate_aggregate_metrics(
    samples,
    target,
    model,
    dataset_size,
    data_mean,
    data_std,
    groups,
    spatial_weights,
    indices,
    climatology=None,
):
    up_group = groups["SP"]
    dtp_group = groups["DTP"]
    spatial_group = groups["JSpatial"]
    channel_group = groups["JChannel"]

    target = target[:, indices]
    samples = samples[:, :, indices]

    data_mean = data_mean[indices]
    data_std = data_std[indices]

    if climatology is not None:
        climatology = climatology[indices]

    # calculate mean and std
    mean, std = GaussianAggregation.apply(samples, up_group)

    acc = None
    if climatology is not None:
        acc = calculate_climatology(
            target,
            mean,
            data_mean,
            data_std,
            climatology,
            spatial_weights,
            spatial_group,
        )

    # calculate rmse
    rmse_sq_map = (mean - target) ** 2
    rmse = torch.sqrt(reduce_spatial_mean(rmse_sq_map, spatial_weights, spatial_group))

    # calculate crps
    crps_map = aggregate_crps(samples, target, up_group=up_group)  # sp aggregation
    crps = reduce_spatial_mean(crps_map, spatial_weights, spatial_group)

    # calculate spread-skill ratio
    avg_std = torch.sqrt(reduce_spatial_mean(std**2, spatial_weights, spatial_group))
    spread_skill = avg_std / rmse

    # likeliehood and KL
    kl_loss = torch_blue.vi.AnalyticalKullbackLeiblerLoss(
        model,
        torch_blue.vi.distributions.MeanFieldNormal(),
        divergence_type=torch_blue.vi.NormalNormalDivergence(),
        dataset_size=dataset_size,
    )
    data_fit = (
        -dataset_size
        * kl_loss.predictive_distribution.log_prob_from_parameters(target, (mean, std))
        .mean(dim=0)
        .sum()
    )
    if dist.get_world_size(dtp_group) > 1:
        dist.all_reduce(data_fit, op=dist.ReduceOp.SUM, group=dtp_group)

    prior_matching = kl_loss.prior_matching()
    if dist.get_world_size(dtp_group) > 1:
        dist.all_reduce(prior_matching, op=dist.ReduceOp.SUM, group=dtp_group)

    local_results = {
        "data_fit": data_fit,
        "prior_matching": prior_matching,
        "rmse": rmse,
        "crps": crps,
        "spread_skill": spread_skill,
    }

    if acc is not None:
        local_results["acc"] = acc

    final_results = {}
    for key, val in local_results.items():
        if val.dim() > 0:
            if dist.get_world_size(channel_group) > 1:
                gathered_list = [
                    None for _ in range(dist.get_world_size(channel_group))
                ]
                dist.all_gather_object(gathered_list, val, group=channel_group)
                gathered_list = [t.to(val.device) for t in gathered_list]
                final_results[key] = torch.cat(gathered_list, dim=0)
            else:
                final_results[key] = val
        else:
            final_results[key] = val

    return final_results


def reduce_spatial_mean(tensor_map, weights, spatial_group):
    # local sum over batch, lat, lon
    local_sum = (tensor_map * weights).sum(dim=(0, 2, 3))

    spatial_ones = torch.ones_like(tensor_map[0, 0])
    actual_weight_map = spatial_ones * weights
    local_weight_sum = actual_weight_map.sum() * tensor_map.shape[0]

    if dist.get_world_size(spatial_group) > 1:
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=spatial_group)
        dist.all_reduce(local_weight_sum, op=dist.ReduceOp.SUM, group=spatial_group)

    return local_sum / local_weight_sum


# TODO: Debug properly when cliamtology files are updated.
def calculate_climatology(
    target, mean, data_mean, data_std, climatology, spatial_weights, spatial_group
):
    pred_phys = (mean * data_std) + data_mean
    true_phys = (target * data_std) + data_mean

    pred_anom = pred_phys - climatology.unsqueeze(0)
    true_anom = true_phys - climatology.unsqueeze(0)

    # calculate acc
    local_num = (pred_anom * true_anom * spatial_weights).sum(dim=(2, 3))
    local_den1 = (pred_anom**2 * spatial_weights).sum(dim=(2, 3))
    local_den2 = (true_anom**2 * spatial_weights).sum(dim=(2, 3))

    # reduce accross dtp before calculation
    if dist.get_world_size(spatial_group) > 1:
        dist.all_reduce(local_num, op=dist.ReduceOp.SUM, group=spatial_group)
        dist.all_reduce(local_den1, op=dist.Op.SUM, group=spatial_group)
        dist.all_reduce(local_den2, op=dist.Op.SUM, group=spatial_group)

    # final calculation
    acc = local_num / torch.sqrt(local_den1 * local_den2 + 1e-8)
    return acc


@torch.no_grad()
def aggregate_crps(local_samples: torch.Tensor, target: torch.Tensor, up_group):
    mae_part = (local_samples - target.expand_as(local_samples)).abs().mean(dim=0)

    if up_group is not None and dist.get_world_size(group=up_group) > 1:
        dist.all_reduce(mae_part, op=dist.ReduceOp.AVG, group=up_group)

    size = dist.get_world_size(group=up_group) if up_group is not None else 1
    samples_per_worker = local_samples.shape[0]
    total_samples = samples_per_worker * size

    denominator = 2 * total_samples * (total_samples - 1)

    # local spread
    local_diffs = (
        (local_samples.unsqueeze(0) - local_samples.unsqueeze(1)).abs().sum(dim=(0, 1))
    )
    spread_part = local_diffs / denominator

    if size > 1:
        gather_list = [torch.zeros_like(local_samples) for _ in range(size)]

        dist.all_gather(gather_list, local_samples, group=up_group)

        my_group_rank = dist.get_rank(group=up_group)

        for i, remote_samples in enumerate(gather_list):
            if i != my_group_rank:
                # Compare our local samples to the samples gathered from GPU i
                cross_diffs = (
                    (local_samples.unsqueeze(0) - remote_samples.unsqueeze(1))
                    .abs()
                    .sum(dim=(0, 1))
                )
                spread_part += cross_diffs / denominator

        # Final sync for the spread part
        dist.all_reduce(spread_part, op=dist.ReduceOp.SUM, group=up_group)

    return mae_part - spread_part


def get_eval_names_and_indicies(config_dict, rank):
    # Check config, if no evaluation present use all variables
    eval_s_vars = config_dict.get("training", {}).get(
        "eval_surface_variables", config_dict["data"]["out_surface_variables"]
    )
    eval_p_vars = config_dict.get("training", {}).get(
        "eval_pressure_variables", config_dict["data"]["out_pressure_variables"]
    )
    eval_p_levels = config_dict.get("training", {}).get(
        "eval_pressure_levels", config_dict["data"]["out_pressure_levels"]
    )

    # All global variables
    all_s = config_dict["data"]["out_surface_variables"]
    all_p = config_dict["data"]["out_pressure_variables"]
    all_l = config_dict["data"]["out_pressure_levels"]

    num_vars = int(config_dict["data"]["n_variables"])

    if dist.get_world_size() > 1:
        row = rank % 2
        local_start = row * num_vars // 2
        local_end = (row + 1) * num_vars // 2
    else:
        local_start = 0
        local_end = num_vars

    indices, names = [], []

    # surface variables
    for s in eval_s_vars:
        if s in all_s:
            g_idx = all_s.index(s)
            if local_start <= g_idx < local_end:
                indices.append(g_idx - local_start)
                names.append(s)

    # pressure variables
    num_s_total = len(all_s)
    num_l_total = len(all_l)

    for pressure_var in eval_p_vars:
        if pressure_var in all_p:
            p_v_idx = all_p.index(pressure_var)
            for level in eval_p_levels:
                if level in all_l:
                    l_idx = all_l.index(level)
                    g_idx = num_s_total + (p_v_idx * num_l_total) + l_idx

                    if local_start <= g_idx < local_end:
                        indices.append(g_idx - local_start)
                        names.append(f"{pressure_var}_{level}")

    return torch.tensor(indices, dtype=torch.long), names


def get_spatial_weights(config_dict, device):
    n_lat = config_dict["data"]["xlat"]
    latitude = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)
    cossum = np.sum(np.cos(latitude))
    weight = torch.tensor(n_lat * np.cos(latitude) / cossum, device=device).float()

    return weight.view(1, 1, n_lat, 1)


def get_variable_weights(config_dict, rank, device):
    # check total variables
    n_total_vars = config_dict["data"]["n_variables"]

    # check if weights are in config (if not return unweighted tensor)
    has_surface = "surface_weights" in config_dict.get("training", {})
    has_pressure = "pressure_weights" in config_dict.get("training", {})

    if has_surface and has_pressure:
        surface_weights = torch.tensor(
            config_dict["training"]["surface_weights"], device=device
        )
        p_weights_raw = torch.tensor(
            config_dict["training"]["pressure_weights"], device=device
        )

        n_p_vars = len(config_dict["data"]["out_pressure_variables"])
        n_p_levels = len(config_dict["data"]["out_pressure_levels"])

        pressure_weights = (
            p_weights_raw.unsqueeze(1).expand(n_p_vars, n_p_levels).flatten()
        )
        all_var_weights = torch.cat([surface_weights, pressure_weights])
    else:
        # All weights are 1.0
        if dist.get_rank() == 0:
            print(
                "WARNING: Variable weights missing in config. Using equal weighting (1.0)."
            )
        all_var_weights = torch.ones(n_total_vars, device=device)

    row = rank % 2
    start_idx = row * n_total_vars // 2
    end_idx = (row + 1) * n_total_vars // 2

    local_weights = all_var_weights[start_idx:end_idx]

    return local_weights


def load_climatology(config_dict, rank, device):
    # number of pressure levels
    num_p_vars = len(config_dict["data"]["out_pressure_levels"]) * len(
        config_dict["data"]["out_pressure_variables"]
    )

    # pressure climatology
    clim_p = torch.from_numpy(
        np.load(
            f"{config_dict['data']['constant_masks_path']}/pressure_climatology.npy"
        )
    ).reshape(num_p_vars, 721, 1440)

    # surface climatology
    clim_s = torch.from_numpy(
        np.load(f"{config_dict['data']['constant_masks_path']}/surface_climatology.npy")
    )

    # full cimatology
    climatology = torch.cat([clim_s, clim_p], dim=0).to(device).float()

    # spatial slicing
    xlat = config_dict["data"]["xlat"]
    xlon_half = config_dict["data"]["xlon"] // 2

    col = rank // 2
    if col == 0:
        climatology = climatology[:, :xlat, :xlon_half]
    else:
        climatology = climatology[:, :xlat, xlon_half:]

    # variable slicing
    num_vars = config_dict["data"]["n_variables"]
    row = rank % 2
    start_idx = row * num_vars // 2
    end_idx = (row + 1) * num_vars // 2

    local_climatology = climatology[start_idx:end_idx, :, :]

    return local_climatology


def log_epoch_to_hdf5(epoch_results, global_names, epoch, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "a") as f:
        # results grouped by epoch
        epoch_grp = f.require_group(f"epoch_{epoch}")

        for metric_name, metric_data in epoch_results.items():
            # per-variable metrics
            if isinstance(metric_data, torch.Tensor) and metric_data.dim() > 0:
                metric_grp = epoch_grp.require_group(metric_name)
                for var_name, value in zip(global_names, metric_data):
                    metric_grp.create_dataset(var_name, data=value.item())

            # global scalar losses
            else:
                val = (
                    metric_data.item()
                    if isinstance(metric_data, torch.Tensor)
                    else metric_data
                )
                epoch_grp.create_dataset(metric_name, data=val)

    print(f"Epoch {epoch} validation metrics saved successfully to {filepath}")

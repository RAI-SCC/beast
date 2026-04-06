import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch_blue
from torch import nn

from gb.losses.up_loss import GaussianAggregation
from gb.setup import get_process_group
from gb.utils import get_spatial_weights


@torch.no_grad()
def validate(
    config_dict: dict,
    model: torch.nn.Module,
    expert_weights: torch.Tensor,
    n_samples: int,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    results_path: str,
):
    # get group information
    channel_group = get_process_group("JChannel")
    dp_group = get_process_group("DP")
    ep_group = get_process_group("Expert")
    global_up_ranks = gather_global_ranks().cpu()

    # rank = dist.get_rank()
    dtp_rank = get_process_group("DTP").rank()
    dtp_worldsize = get_process_group("DTP").size()

    # Expert masks
    global_expert_mask, _ = gather_vars_for_plot(
        expert_weights.unsqueeze(dim=0), [""], channel_group
    )
    global_expert_mask = global_expert_mask.squeeze(dim=0)
    local_exp_agg = ExpertAggregation(ep_group, expert_weights)
    exp_agg = ExpertAggregation(ep_group, global_expert_mask)

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(0)  # should always the be same across epochs

    # load spatial weights
    spatial_weights = get_spatial_weights(config_dict, device)

    # # extract means and std for evaluation
    try:
        data_means = torch.from_numpy(dataloader.dataset.norm_values["mean"]).to(device)
        data_std = torch.from_numpy(dataloader.dataset.norm_values["std"]).to(device)
    except AttributeError:
        data_means = torch.from_numpy(
            dataloader.dataset.dataset.norm_values["mean"]
        ).to(device)
        data_std = torch.from_numpy(dataloader.dataset.dataset.norm_values["std"]).to(
            device
        )

    # climatology
    # TODO: Update Climatology Files so this works correctly
    # local_climatology = load_climatology(config_dict, rank, device)

    # Indicies for metrics
    local_indices, local_names = get_names_and_indices(
        config_dict, dtp_rank, dtp_worldsize, for_plot=False
    )
    # Indices for plots
    plot_indices, plot_names = get_names_and_indices(
        config_dict, dtp_rank, dtp_worldsize, for_plot=True
    )

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
    plot_step = config_dict["training"].get("plot_step", 5)

    # loop through dataloader
    for i, (x, y, start, end) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y_unmasked = y.to(device, non_blocking=True)

        # prediction
        samples = model(x, samples=n_samples)

        y = y_unmasked * expert_weights.reshape(1, -1, 1, 1)
        samples = samples * expert_weights.reshape(1, 1, -1, 1, 1)

        val_results = calculate_aggregate_metrics(
            samples,
            y,
            model,
            expert_weights,
            local_exp_agg,
            len(dataloader.dataset),
            data_means,
            data_std,
            spatial_weights,
            local_indices,
            global_up_ranks,
            climatology=None,
        )

        # accumulate metrics
        if not epoch_results:
            for k, v in val_results.items():
                epoch_results[k] = v.clone() if isinstance(v, torch.Tensor) else v
        else:
            for k, v in val_results.items():
                epoch_results[k] += v

        # plot only for first batch
        if epoch % plot_step == 0 and i == 0:
            generate_distributed_plots(
                samples,
                y_unmasked,
                data_means,
                data_std,
                ep_group,
                expert_weights,
                plot_indices,
                plot_names,
                epoch,
                results_path,
            )

        num_batches += 1

    if num_batches > 0:
        for k in epoch_results.keys():
            epoch_results[k] /= num_batches
            if not isinstance(epoch_results[k], torch.Tensor):
                epoch_results[k] = torch.tensor(epoch_results[k], device=device)

            if dp_group.size() > 1:
                dist.all_reduce(epoch_results[k], op=dist.ReduceOp.AVG, group=dp_group)

            if ep_group is not None and dist.get_world_size(ep_group) > 1:
                if epoch_results[k].numel() > 1:
                    # dist.all_reduce(epoch_results[k], op=dist.ReduceOp.SUM, group=ep_group)
                    epoch_results[k] = exp_agg(epoch_results[k])
                # elif k == "data_fit":
                #    pass
                elif k == "prior_matching":
                    dist.all_reduce(
                        epoch_results[k], op=dist.ReduceOp.AVG, group=ep_group
                    )

    if "mse" in epoch_results.keys():
        epoch_results["rmse"] = torch.sqrt(epoch_results["mse"])
        epoch_results.pop("mse")

    if "avg_var" in epoch_results.keys() and "rmse" in epoch_results.keys():
        epoch_results["spread_skill"] = (
            torch.sqrt(epoch_results["avg_var"]) / epoch_results["rmse"]
        )
        epoch_results.pop("avg_var")

    # save results
    if dist.get_rank() == 0:
        print(f"\n{'=' * 50}")
        print(f"EPOCH {epoch} FINAL VALIDATION METRICS")
        print(f"{'=' * 50}")

        for k, v in epoch_results.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.numel() > 1:
                print(f"\n{k.upper()}:")
                for var_name, val in zip(global_names, v):
                    print(f"  {var_name}: {val.item():.6f}")
            else:
                val = v.item() if isinstance(v, torch.Tensor) else v
                print(f"{k.upper()}: {val:.6f}")
        print(f"{'=' * 50}\n")

        # save to hdf5
        h5_path = Path(results_path) / "evaluation_metrics.h5"
        try:
            log_epoch_to_hdf5(
                epoch_results,
                global_names,
                epoch,
                h5_path,
                config_dict=(config_dict if epoch == 0 else None),
            )
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
def validate_deterministic(
    config_dict: dict,
    model: torch.nn.Module,
    expert_weights: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    results_path: str,
):
    # get group information
    channel_group = get_process_group("JChannel")
    dp_group = get_process_group("DP")
    ep_group = get_process_group("Expert")

    # rank = dist.get_rank()
    dtp_rank = get_process_group("DTP").rank()
    dtp_worldsize = get_process_group("DTP").size()

    # Expert masks
    global_expert_mask, _ = gather_vars_for_plot(
        expert_weights.unsqueeze(dim=0), [""], channel_group
    )
    global_expert_mask = global_expert_mask.squeeze(dim=0)
    local_exp_agg = ExpertAggregation(ep_group, expert_weights)
    exp_agg = ExpertAggregation(ep_group, global_expert_mask)

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(
            0
        )  # validation loading order should always be the same across epochs

    # load spatial weights
    spatial_weights = get_spatial_weights(config_dict, device)

    # extract means and std for evaluation
    try:
        data_means = torch.from_numpy(dataloader.dataset.norm_values["mean"]).to(device)
        data_std = torch.from_numpy(dataloader.dataset.norm_values["std"]).to(device)
    except AttributeError:
        data_means = torch.from_numpy(
            dataloader.dataset.dataset.norm_values["mean"]
        ).to(device)
        data_std = torch.from_numpy(dataloader.dataset.dataset.norm_values["std"]).to(
            device
        )

    # climatology
    # TODO: Update Climatology Files so this works correctly
    # local_climatology = load_climatology(config_dict, rank, device)

    # Indicies for metrics
    local_indices, local_names = get_names_and_indices(
        config_dict, dtp_rank, dtp_worldsize, for_plot=False
    )
    # Indices for plots
    plot_indices, plot_names = get_names_and_indices(
        config_dict, dtp_rank, dtp_worldsize, for_plot=True
    )

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
    plot_step = config_dict["training"].get("plot_step", 5)

    # loop through dataloader
    for i, (x, y, start, end) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y_unmasked = y.to(device, non_blocking=True)

        # prediction
        pred = model(x)

        y = y_unmasked * expert_weights.reshape(1, -1, 1, 1)
        pred = pred * expert_weights.reshape(1, -1, 1, 1)

        val_results = calculate_metrics(
            pred,
            y,
            model,
            expert_weights,
            local_exp_agg,
            len(dataloader.dataset),
            data_means,
            data_std,
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

        # plot only for first batch
        if epoch % plot_step == 0 and i == 0:
            generate_distributed_deterministic_plots(
                pred,
                y_unmasked,
                data_means,
                data_std,
                ep_group,
                expert_weights,
                plot_indices,
                plot_names,
                epoch,
                results_path,
            )

        num_batches += 1

    if num_batches > 0:
        for k in epoch_results.keys():
            epoch_results[k] /= num_batches
            if not isinstance(epoch_results[k], torch.Tensor):
                epoch_results[k] = torch.tensor(epoch_results[k], device=device)

            if dp_group.size() > 1:
                dist.all_reduce(epoch_results[k], op=dist.ReduceOp.AVG, group=dp_group)

            if ep_group is not None and dist.get_world_size(ep_group) > 1:
                if epoch_results[k].numel() > 1:
                    # dist.all_reduce(epoch_results[k], op=dist.ReduceOp.SUM, group=ep_group)
                    epoch_results[k] = exp_agg(epoch_results[k])

    if "mse" in epoch_results.keys():
        epoch_results["rmse"] = torch.sqrt(epoch_results["mse"])
        epoch_results.pop("mse")

    # save results
    if dist.get_rank() == 0:
        print(f"\n{'=' * 50}")
        print(f"EPOCH {epoch} FINAL VALIDATION METRICS")
        print(f"{'=' * 50}")

        for k, v in epoch_results.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.numel() > 1:
                print(f"\n{k.upper()}:")
                for var_name, val in zip(global_names, v):
                    print(f"  {var_name}: {val.item():.6f}")
            else:
                val = v.item() if isinstance(v, torch.Tensor) else v
                print(f"{k.upper()}: {val:.6f}")
        print(f"{'=' * 50}\n")

        # save to hdf5
        h5_path = Path(results_path) / "evaluation_metrics.h5"
        try:
            log_epoch_to_hdf5(
                epoch_results,
                global_names,
                epoch,
                h5_path,
                config_dict=(config_dict if epoch == 0 else None),
            )
        except Exception as e:
            print(f"WARNING: HDF5 Logging failed! Error: {e}")
            print("Metrics are still preserved in this Slurm log above.")

    loss = epoch_results.get("loss")

    if isinstance(loss, torch.Tensor):
        loss = loss.item()

    return loss


@torch.no_grad()
def calculate_aggregate_metrics(
    samples,
    target,
    model,
    expert_weights,
    expert_agg,
    dataset_size,
    data_mean,
    data_std,
    spatial_weights,
    indices,
    global_up_ranks,
    climatology=None,
):
    up_group = get_process_group("SP")
    dtp_group = get_process_group("DTP")
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")

    target = target[:, indices]
    samples = samples[:, :, indices]

    if channel_group.size() > 1:
        local_var_slice = slice(0, len(data_mean) // 2)
        if channel_group.rank() > 0:
            local_var_slice = slice(len(data_mean) // 2, None)

        data_mean = data_mean[local_var_slice]
        data_std = data_std[local_var_slice]

    data_mean = data_mean[indices]
    data_std = data_std[indices]

    if climatology is not None:
        climatology = climatology[indices]

    # calculate mean and std
    mean, std = GaussianAggregation.apply(samples, up_group, expert_weights[indices])

    # likelihood and KL
    kl_loss = torch_blue.vi.AnalyticalKullbackLeiblerLoss(
        model,
        torch_blue.vi.distributions.MeanFieldNormal(),
        divergence_type=torch_blue.vi.NormalNormalDivergence(),
        dataset_size=dataset_size,
    )
    data_fit = -dataset_size * (
        kl_loss.predictive_distribution.log_prob_from_parameters(
            target, (mean, std)
        ).mean(dim=0, keepdim=True)
    )
    data_fit = expert_agg(data_fit)
    data_fit = data_fit.sum()
    if dist.get_world_size(dtp_group) > 1:
        dist.all_reduce(data_fit, op=dist.ReduceOp.SUM, group=dtp_group)

    prior_matching = kl_loss.prior_matching()
    if dist.get_world_size(channel_group) > 1:
        dist.all_reduce(prior_matching, op=dist.ReduceOp.SUM, group=channel_group)

    # mask std for other metrics
    std = std * expert_weights.reshape(1, -1, 1, 1)

    # calculate ACC

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

    # calculate rmse (surrogate)
    rmse_sq_map = (mean - target) ** 2
    rmse_sq = reduce_spatial_mean(rmse_sq_map, spatial_weights, spatial_group)

    # calculate crps TODO: find a working implementation later
    crps_map = aggregate_crps(
        samples, target, up_group=up_group, global_up_ranks=global_up_ranks
    )  # sp aggregation
    crps = reduce_spatial_mean(crps_map, spatial_weights, spatial_group)

    # calculate spread-skill ratio (surrogate)
    avg_var = reduce_spatial_mean(std**2, spatial_weights, spatial_group)

    # flatten for calculation
    std_1d = data_std.flatten()

    # return unnormalized metrics
    local_results = {
        "data_fit": data_fit,
        "prior_matching": prior_matching,
        "mse": rmse_sq * (std_1d**2),
        "crps": crps * std_1d,
        "avg_var": avg_var * (std_1d**2),
    }

    if acc is not None:
        local_results["acc"] = acc

    final_results = {}
    for key, val in local_results.items():
        if val.dim() > 0 and key not in ["data_fit", "prior_matching"]:
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


@torch.no_grad()
def calculate_metrics(
    pred,
    target,
    model,
    expert_weights,
    expert_agg,
    dataset_size,
    data_mean,
    data_std,
    spatial_weights,
    indices,
    climatology=None,
):
    dtp_group = get_process_group("DTP")
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")

    target = target[:, indices]
    pred = pred[:, indices]

    if channel_group.size() > 1:
        local_var_slice = slice(0, len(data_mean) // 2)
        if channel_group.rank() > 0:
            local_var_slice = slice(len(data_mean) // 2, None)

        data_mean = data_mean[local_var_slice]
        data_std = data_std[local_var_slice]

    data_mean = data_mean[indices]
    data_std = data_std[indices]

    if climatology is not None:
        climatology = climatology[indices]

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pred, target)

    loss = expert_agg(loss)
    loss = loss.sum()
    if dist.get_world_size(dtp_group) > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=dtp_group)

    # calculate ACC
    acc = None
    if climatology is not None:
        acc = calculate_climatology(
            target,
            pred,
            data_mean,
            data_std,
            climatology,
            spatial_weights,
            spatial_group,
        )

    # calculate rmse (surrogate)
    rmse_sq_map = (pred - target) ** 2
    rmse_sq = reduce_spatial_mean(rmse_sq_map, spatial_weights, spatial_group)

    std_1d = data_std.flatten()

    # return unnormalized metrics
    local_results = {
        "loss": loss,
        "mse": rmse_sq * (std_1d**2),
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
        dist.all_reduce(local_den1, op=dist.ReduceOp.SUM, group=spatial_group)
        dist.all_reduce(local_den2, op=dist.ReduceOp.SUM, group=spatial_group)

    # final calculation
    acc = local_num / torch.sqrt(local_den1 * local_den2 + 1e-8)
    return acc


@torch.no_grad()
def aggregate_crps(
    local_samples: torch.Tensor, target: torch.Tensor, up_group, global_up_ranks
):
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

    # communication via merry-go-round
    if size > 1:
        my_group_rank = dist.get_rank(group=up_group)

        # neighbors (deal with start and end ring)
        next_rank_index = (my_group_rank + 1) % size
        prev_rank_index = (my_group_rank - 1 + size) % size
        next_rank = global_up_ranks[next_rank_index].item()
        prev_rank = global_up_ranks[prev_rank_index].item()

        # MGR buffers
        send_buffer = local_samples.clone()
        recv_buffer = torch.zeros_like(local_samples)

        # loop through all processes
        for step in range(1, size):
            # swap buffers
            ops = [
                dist.P2POp(dist.isend, send_buffer, next_rank, group=up_group),
                dist.P2POp(dist.irecv, recv_buffer, prev_rank, group=up_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            # calculate cross diffs
            cross_diffs = (
                (local_samples.unsqueeze(0) - recv_buffer.unsqueeze(1))
                .abs()
                .sum(dim=(0, 1))
            )
            spread_part += cross_diffs / denominator

            # adjust buffers for next loop
            send_buffer.copy_(recv_buffer)

        # final sync
        dist.all_reduce(spread_part, op=dist.ReduceOp.SUM, group=up_group)

    return mae_part - spread_part


@torch.no_grad()
def get_names_and_indices(config_dict, rank, world_size, for_plot=False):

    training_dict = config_dict.get("training", {})

    # All global variables

    all_s = [
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "sea_surface_temperature",
        "total_precipitation",
    ]

    all_p = [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]

    all_l = [
        "1000",
        "925",
        "850",
        "700",
        "600",
        "500",
        "400",
        "300",
        "250",
        "200",
        "150",
        "100",
        "50",
    ]

    if for_plot:
        target_s_vars = training_dict.get(
            "plot_surface_variables",
            ["mean_sea_level_pressure", "2m_temperature", "total_precipitation"],
        )
        target_p_vars = training_dict.get(
            "plot_pressure_variables", ["geopotential", "u_component_of_wind"]
        )
        target_p_levels = training_dict.get("plot_pressure_levels", ["850", "500"])
    else:
        target_s_vars = all_s.copy()
        target_p_vars = all_p.copy()
        target_p_levels = all_l.copy()

    # num_vars = int(config_dict["data"]["n_variables"])
    num_vars = len(all_s) + len(all_p) * len(all_l)

    if world_size > 1:
        row = rank % 2
        local_start = row * (num_vars // 2)
        local_end = (row + 1) * (num_vars // 2)
    else:
        local_start = 0
        local_end = num_vars

    indices, names = [], []

    # surface variables
    for s in target_s_vars:
        if s in all_s:
            g_idx = all_s.index(s)
            if local_start <= g_idx < local_end:
                indices.append(g_idx - local_start)
                names.append(s)

    # pressure variables
    num_s_total = len(all_s)
    num_l_total = len(all_l)

    for pressure_var in target_p_vars:
        if pressure_var in all_p:
            p_v_idx = all_p.index(pressure_var)
            for level in target_p_levels:
                if level in all_l:
                    l_idx = all_l.index(level)
                    g_idx = num_s_total + (p_v_idx * num_l_total) + l_idx

                    if local_start <= g_idx < local_end:
                        indices.append(g_idx - local_start)
                        names.append(f"{pressure_var}_{level}")

    return torch.tensor(indices, dtype=torch.long), names


@torch.no_grad()
def load_climatology(config_dict, rank, device):
    # TODO: hardcode the variables, "unhardcode" fixed xlat / xlon

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


@torch.no_grad()
def gather_spatial_for_plot(gather_tensor, spatial_group):
    if dist.get_world_size(spatial_group) == 1:
        return gather_tensor

    gathered = [
        torch.zeros_like(gather_tensor)
        for _ in range(dist.get_world_size(spatial_group))
    ]
    dist.all_gather(gathered, gather_tensor, group=spatial_group)
    return torch.cat(gathered, dim=3)


@torch.no_grad()
def gather_vars_for_plot(gather_tensor, local_names, channel_group):
    if dist.get_world_size(channel_group) == 1:
        return gather_tensor, local_names

    # Gather names
    gathered_names = [None for _ in range(dist.get_world_size(channel_group))]
    dist.all_gather_object(gathered_names, local_names, group=channel_group)

    # Gather tensors
    gathered_t_obj = [None for _ in range(dist.get_world_size(channel_group))]
    dist.all_gather_object(gathered_t_obj, gather_tensor, group=channel_group)

    gathered_t_obj = [tensor.to(gather_tensor.device) for tensor in gathered_t_obj]
    global_names_flat = [n for sublist in gathered_names for n in sublist]

    return torch.cat(gathered_t_obj, dim=1), global_names_flat


@torch.no_grad()
def generate_distributed_deterministic_plots(
    pred,
    target,
    data_mean,
    data_std,
    expert_group,
    expert_weights,
    plot_indices,
    plot_names,
    epoch,
    file_path,
):

    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")

    # slice and unnormalize - only indices to be plotted
    if len(plot_indices) > 0:
        local_pred = pred[:, plot_indices]
        local_target = target[:, plot_indices]

        if channel_group.size() > 1:
            local_var_slice = slice(0, len(data_mean) // 2)
            if channel_group.rank() > 0:
                local_var_slice = slice(len(data_mean) // 2, None)
            data_mean = data_mean[local_var_slice]
            data_std = data_std[local_var_slice]

        local_mean = data_mean[plot_indices].view(1, -1, 1, 1)
        local_std = data_std[plot_indices].view(1, -1, 1, 1)

        pred_unnormalized = (local_pred * local_std) + local_mean
        target_unnormalized = (local_target * local_std) + local_mean
    else:
        # Dummy tensors so gather does not break
        target_unnormalized = torch.empty(
            (target.shape[0], 0, target.shape[2], target.shape[3]), device=target.device
        )
        pred_unnormalized = torch.empty_like(target_unnormalized)

    # gather spatially
    full_lon_pred = gather_spatial_for_plot(pred_unnormalized, spatial_group)
    full_lon_target = gather_spatial_for_plot(target_unnormalized, spatial_group)

    # gather channels
    global_pred, global_plot_names = gather_vars_for_plot(
        full_lon_pred, plot_names, channel_group
    )
    global_target, _ = gather_vars_for_plot(full_lon_target, plot_names, channel_group)
    global_mask, _ = gather_vars_for_plot(
        expert_weights[plot_indices].unsqueeze(dim=0), plot_names, channel_group
    )

    expert_agg = ExpertAggregation(expert_group, global_mask)

    # aggregation over all the experts (not needed for targets)
    global_pred = expert_agg(global_pred)

    # plot
    if dist.get_rank() == 0 and len(global_plot_names) > 0:
        plot_dir = Path(file_path) / "plots" / f"epoch_{epoch}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        for idx, var_name in enumerate(global_plot_names):
            out_map = global_pred[0, idx].cpu().numpy()
            tar_map = global_target[0, idx].cpu().numpy()

            # bias calculation
            bias_map = tar_map - out_map

            # shared scale for prediction and target
            state_vmin = min(out_map.min(), tar_map.min())
            state_vmax = max(out_map.max(), tar_map.max())

            # symmetrical scale for bias
            bias_vmax = max(np.abs(bias_map.max()), np.abs(bias_map.min()))
            bias_vmin = -bias_vmax

            # plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # prediction
            axes[0].set_title(f"Prediction: {var_name}")
            c0 = axes[0].imshow(
                out_map,
                origin="upper",
                cmap="viridis",
                vmin=state_vmin,
                vmax=state_vmax,
            )
            fig.colorbar(c0, ax=axes[0], orientation="horizontal", pad=0.08)

            # target
            axes[1].set_title(f"Target: {var_name}")
            c1 = axes[1].imshow(
                tar_map,
                origin="upper",
                cmap="viridis",
                vmin=state_vmin,
                vmax=state_vmax,
            )
            fig.colorbar(c1, ax=axes[1], orientation="horizontal", pad=0.08)

            # bias
            axes[2].set_title(f"Bias: {var_name}")
            c2 = axes[2].imshow(
                bias_map, origin="upper", cmap="RdBu_r", vmin=bias_vmin, vmax=bias_vmax
            )
            fig.colorbar(c2, ax=axes[2], orientation="horizontal", pad=0.08)

            plt.tight_layout()

            save_path = plot_dir / f"{var_name}.pdf"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {save_path}")


@torch.no_grad()
def generate_distributed_plots(
    samples,
    target,
    data_mean,
    data_std,
    expert_group,
    expert_weights,
    plot_indices,
    plot_names,
    epoch,
    file_path,
):

    up_group = get_process_group("SP")
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")

    # slice and unnormalize - only indices to be plotted
    if len(plot_indices) > 0:
        local_samples = samples[:, :, plot_indices]
        local_target = target[:, plot_indices]

        mean, _ = GaussianAggregation.apply(
            local_samples, up_group, expert_weights[plot_indices]
        )

        if channel_group.size() > 1:
            local_var_slice = slice(0, len(data_mean) // 2)
            if channel_group.rank() > 0:
                local_var_slice = slice(len(data_mean) // 2, None)

            data_mean = data_mean[local_var_slice]
            data_std = data_std[local_var_slice]
        local_mean = data_mean[plot_indices].view(1, -1, 1, 1)
        local_std = data_std[plot_indices].view(1, -1, 1, 1)

        pred_unnormalized = (mean * local_std) + local_mean
        target_unnormalized = (local_target * local_std) + local_mean
    else:
        # Dummy tensors so gather does not break
        target_unnormalized = torch.empty(
            (target.shape[0], 0, target.shape[2], target.shape[3]), device=target.device
        )
        pred_unnormalized = torch.empty_like(target_unnormalized)

    # gather spatially
    full_lon_pred = gather_spatial_for_plot(pred_unnormalized, spatial_group)
    full_lon_target = gather_spatial_for_plot(target_unnormalized, spatial_group)

    # gather channels
    global_pred, global_plot_names = gather_vars_for_plot(
        full_lon_pred, plot_names, channel_group
    )
    global_target, _ = gather_vars_for_plot(full_lon_target, plot_names, channel_group)
    global_mask, _ = gather_vars_for_plot(
        expert_weights[plot_indices].unsqueeze(dim=0), plot_names, channel_group
    )

    expert_agg = ExpertAggregation(expert_group, global_mask)

    # aggregation over all the experts (not needed for targets)
    global_pred = expert_agg(global_pred)

    # plot
    if dist.get_rank() == 0 and len(global_plot_names) > 0:
        plot_dir = Path(file_path) / "plots" / f"epoch_{epoch}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        for idx, var_name in enumerate(global_plot_names):
            out_map = global_pred[0, idx].cpu().numpy()
            tar_map = global_target[0, idx].cpu().numpy()

            # bias calculation
            bias_map = tar_map - out_map

            # shared scale for prediction and target
            state_vmin = min(out_map.min(), tar_map.min())
            state_vmax = max(out_map.max(), tar_map.max())

            # symmetrical scale for bias
            bias_vmax = max(np.abs(bias_map.max()), np.abs(bias_map.min()))
            bias_vmin = -bias_vmax

            # plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # prediction
            axes[0].set_title(f"Prediction: {var_name}")
            c0 = axes[0].imshow(
                out_map,
                origin="upper",
                cmap="viridis",
                vmin=state_vmin,
                vmax=state_vmax,
            )
            fig.colorbar(c0, ax=axes[0], orientation="horizontal", pad=0.08)

            # target
            axes[1].set_title(f"Target: {var_name}")
            c1 = axes[1].imshow(
                tar_map,
                origin="upper",
                cmap="viridis",
                vmin=state_vmin,
                vmax=state_vmax,
            )
            fig.colorbar(c1, ax=axes[1], orientation="horizontal", pad=0.08)

            # bias
            axes[2].set_title(f"Bias: {var_name}")
            c2 = axes[2].imshow(
                bias_map, origin="upper", cmap="RdBu_r", vmin=bias_vmin, vmax=bias_vmax
            )
            fig.colorbar(c2, ax=axes[2], orientation="horizontal", pad=0.08)

            plt.tight_layout()

            save_path = plot_dir / f"{var_name}.pdf"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {save_path}")


@torch.no_grad()
def log_epoch_to_hdf5(epoch_results, global_names, epoch, filepath, config_dict=None):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "a") as f:
        if config_dict is not None:
            conf_grp = f.require_group("config")
            conf_grp.create_dataset(
                "json", dtype=h5py.string_dtype(), data=json.dumps(config_dict)
            )

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


class ExpertAggregation(nn.Module):
    def __init__(self, ep_group, expert_weights):
        super().__init__()

        self.ep_group = ep_group

        if ep_group.size() > 1:
            temp_expert_weights = expert_weights.clone().detach().contiguous()
            dist.all_reduce(temp_expert_weights, op=dist.ReduceOp.SUM, group=ep_group)

            self.ep_normalizations = temp_expert_weights.reshape(-1)
            # (batch, channel, lat, lon)

    def forward(self, x):
        if self.ep_group.size() <= 1:
            return x

        if x.dim() > 4:
            normalization = self.ep_normalizations.reshape(1, 1, -1, 1, 1)
        elif x.dim() == 4:
            normalization = self.ep_normalizations.reshape(1, -1, 1, 1)
        else:
            normalization = self.ep_normalizations

        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=self.ep_group)
        y = y / normalization

        return y


# def gather_global_ranks(process_group, device):
#     gather_tensor = torch.tensor([dist.get_rank()], device=device)

#     gathered = [
#         torch.zeros_like(gather_tensor)
#         for _ in range(dist.get_world_size(process_group))
#     ]
#     dist.all_gather(gathered, gather_tensor, group=process_group)
#     return torch.cat(gathered, dim=0).to(torch.int32)


def gather_global_ranks():
    channel_group = get_process_group("JChannel")
    spatial_group = get_process_group("JSpatial")
    dtp_group = get_process_group("DTP")
    up_group = get_process_group("SP")
    dp_group = get_process_group("DP")
    ep_group = get_process_group("Expert")

    mesh_dims = [
        ep_group.size(),
        dp_group.size(),
        up_group.size(),
        spatial_group.size(),
        channel_group.size(),
    ]
    up_stride = torch.prod(torch.tensor(mesh_dims[-2:]))
    dp_stride = torch.prod(torch.tensor(mesh_dims[-3:]))
    ep_stride = torch.prod(torch.tensor(mesh_dims[-4:]))

    gathered = torch.arange(0, up_group.size() * up_stride, up_stride)
    gathered += (
        dtp_group.rank() + dp_group.rank() * dp_stride + ep_group.rank() * ep_stride
    )

    return gathered

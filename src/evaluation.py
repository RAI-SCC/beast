from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch_blue
from Jigsaw import dataset
from Jigsaw.util_fns import plotting_functions as plot_fns

from gb.losses.UP_Loss import GaussianAggregation


@torch.no_grad()
def validate(
    config_dict: dict,
    model: torch.nn.Module,
    n_samples: int,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    results_path: str,
    groups: dict,
):
    # get group information
    channel_group = groups["JChannel"]
    dp_group = groups["DP"]
    ep_group = groups.get("Expert")

    rank = dist.get_rank()

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    # load spatial weights
    spatial_weights = get_spatial_weights(config_dict, device)

    # extract means and std for evaluation
    data_means = torch.from_numpy(dataloader.dataset.norm_values["mean"]).to(device)
    data_std = torch.from_numpy(dataloader.dataset.norm_values["std"]).to(device)

    # climatology
    # TODO: Update Climatology Files so this works correctly
    # local_climatology = load_climatology(config_dict, rank, device)

    # Indicies for metrics
    local_indices, local_names = get_names_and_indices(
        config_dict, rank, for_plot=False
    )
    # Indices for plots
    plot_indices, plot_names = get_names_and_indices(config_dict, rank, for_plot=True)

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
    plot_step = config_dict["training"]["plot_step"]

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

        # plot only for first batch
        if epoch % plot_step == 0 and i == 0:
            generate_distributed_plots(
                samples,
                y,
                data_means,
                data_std,
                groups,
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
                dist.all_reduce(epoch_results[k], op=dist.ReduceOp.SUM, group=ep_group)

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
            if isinstance(v, torch.Tensor) and v.dim() > 0:
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
    rmse_sq = reduce_spatial_mean(rmse_sq_map, spatial_weights, spatial_group)

    # calculate crps
    crps_map = aggregate_crps(samples, target, up_group=up_group)  # sp aggregation
    crps = reduce_spatial_mean(crps_map, spatial_weights, spatial_group)

    # calculate spread-skill ratio
    avg_var = reduce_spatial_mean(std**2, spatial_weights, spatial_group)

    # likelihood and KL
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
    if dist.get_world_size(channel_group) > 1:
        dist.all_reduce(prior_matching, op=dist.ReduceOp.SUM, group=channel_group)

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

    # communication via merry-go-round
    if size > 1:
        my_group_rank = dist.get_rank(group=up_group)

        # neighbors (deal with start and end ring)
        next_rank = (my_group_rank + 1) % size
        prev_rank = (my_group_rank - 1 + size) % size

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


def get_names_and_indices(config_dict, rank, for_plot=False):

    training_dict = config_dict.get("training", {})

    if for_plot:
        target_s_vars = training_dict.get("plot_surface_variables", [])
        target_p_vars = training_dict.get("plot_pressure_variables", [])
        target_p_levels = training_dict.get("plot_pressure_levels", [])
    else:
        target_s_vars = training_dict.get(
            "eval_surface_variables", config_dict["data"]["out_surface_variables"]
        )
        target_p_vars = training_dict.get(
            "eval_pressure_variables", config_dict["data"]["out_pressure_variables"]
        )
        target_p_levels = training_dict.get(
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


def gather_spatial_for_plot(gather_tensor, spatial_group):
    if dist.get_world_size(spatial_group) == 1:
        return gather_tensor

    gathered = [
        torch.zeros_like(gather_tensor)
        for _ in range(dist.get_world_size(spatial_group))
    ]
    dist.all_gather(gathered, gather_tensor, group=spatial_group)
    return torch.cat(gathered, dim=3)


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
def generate_distributed_plots(
    samples,
    target,
    data_mean,
    data_std,
    groups,
    plot_indices,
    plot_names,
    epoch,
    file_path,
):

    up_group = groups["SP"]
    spatial_group = groups["JSpatial"]
    channel_group = groups["JChannel"]

    # slice and unnormalize - only indices to be plotted
    if len(plot_indices) > 0:
        local_samples = samples[:, :, plot_indices]
        local_target = target[:, plot_indices]

        mean, _ = GaussianAggregation.apply(local_samples, up_group)

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

    # TODO maybe there needs to be an expert aggregation here (unsure)

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

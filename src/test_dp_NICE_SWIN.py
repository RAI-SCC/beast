import argparse
import os
import time

import torch
import torch.distributed as dist
import torch_blue.vi as vi
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch_blue.vi.distributions import (
    MeanFieldNormal as MeanFieldNormalPredictiveDistribution,
)

from gb.dataloaders.dataloader_1_5_degree import get_test_dataloader
from gb.model.nice_SWIN import NiceBNNNoEmbeddingModel
from gb.utils import download_1_5_degree_data, get_process_group, init_gb_process_group


def set_all_seeds(seed: int) -> None:
    """Set random seeds."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    """Count num trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def probabilistic_loss(output, output_surface, target, target_surface, kl_loss):
    output_total = torch.cat((output_surface, output), dim=2)
    target_total = torch.cat((target_surface, target), dim=1)
    return kl_loss(output_total, target_total)


def training_loop(
    config: dict,
    device: str,
    slurm_localid: int,
    rank: int,
    world_size: int,
    wall_time: float,
) -> None:
    """Train the model.

    params: Dict
    device: String
    slurm_localid: int
    """

    if wall_time > 0:
        start_up_time = time.perf_counter()

    no_scaler = not (config["training"].get("use_grad_scaler", False))
    if not no_scaler and rank == 0:
        print("Using GradScaler")

    gradient_clipping = config["training"].get("use_gradient_clipping", False)
    clipping_limit = config["training"].get("gradient_clipping_limit", 1.0)
    if gradient_clipping and rank == 0:
        print(f"Using gradient clipping with limit {clipping_limit}")

    no_ddp = config["training"].get("no_ddp", False)
    use_fl64 = config["training"].get("use_float64", False)
    use_fl32 = config["training"].get("use_float32", False)
    use_fl16 = config["training"].get("use_float16", False)

    assert not (use_fl16 and use_fl64), (
        "Config error: you cannot enforce usage of float64 and float16 at the same time"
    )
    assert not (use_fl16 and use_fl32), (
        "Config error: you cannot enforce usage of float32 and float16 at the same time"
    )
    assert not (use_fl32 and use_fl64), (
        "Config error: you cannot enforce usage of float64 and float32 at the same time"
    )

    train_data_loader, train_info = get_test_dataloader(config, get_process_group("DP"))

    print(f"Rank {rank} using device {device}")
    model = NiceBNNNoEmbeddingModel(config=config, device=device)

    if use_fl16:
        model = model.to(torch.float16).to(device)
    elif use_fl64:
        model = model.to(torch.float64).to(device)
    else:
        model = model.to(torch.float32).to(device)

    model.return_log_probs = False

    # DistributedDataParallel wrapper

    # loss_type = config["training"].get("loss_type", "kl")
    predictive_dist = MeanFieldNormalPredictiveDistribution()
    kl_loss = vi.AnalyticalKullbackLeiblerLoss(
        model,
        predictive_dist,
        train_info["size"],
        divergence_type=vi.NormalNormalDivergence(),
    )

    # area_weights_sqrt = torch.sqrt(area_weights)
    def loss_fn(output, output_surface, target, target_surface):
        return probabilistic_loss(
            output, output_surface, target, target_surface, kl_loss
        )

    # probabilistic = True

    if dist.is_initialized() and not no_ddp:
        model_ddp = DistributedDataParallel(
            model,
            device_ids=[slurm_localid],
            output_device=slurm_localid,
            process_group=get_process_group("DDP"),
        )
    else:
        model_ddp = model

    if not config["training"].get("fast_path", False):
        for module in model.modules():
            if hasattr(module, "_fast_path_flag"):
                module._fast_path_flag = False

    if rank == 0:
        print(f"model has {count_parameters(model_ddp)} parameters")

    optimizer_name = config["training"].get("optimizer", "Adam")

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model_ddp.parameters(),
            lr=config["training"].get("lr", 1e-4),
            weight_decay=config["training"].get("weight_decay", 0),
        )
    elif optimizer_name == "RAdam":
        optimizer = torch.optim.RAdam(
            model_ddp.parameters(),
            lr=config["training"].get("lr", 1e-4),
            weight_decay=config["training"].get("weight_decay", 0),
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented.")

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=config["training"]["scheduler"]["t_initial"],
        lr_min=config["training"]["scheduler"]["lr_min"],
    )

    # Load model from checkpoint
    """
    try:
        state = torch.load(config["training"]["load_model"], map_location="cpu")
        model_ddp.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
    except Exception as err:
        if rank == 0:
            print(err)
            print("Model cannot be loaded from checkpoint. Reinitializing model.")
        start_epoch = 0
    """
    start_epoch = 0

    # Training loop
    # train_loss_history = []
    # valid_loss_history = []
    # best_val_loss = 100000  # Arbitrary large number

    if no_scaler:
        scaler = None
    else:
        scaler = GradScaler()
    start_lr = config["training"].get("lr", 1e-4) * 1e-2

    num_epochs = config["training"]["epochs"]

    scheduler.step(start_epoch)

    torch.backends.cuda.matmul.allow_tf32 = True

    sample_num = config["training"].get("sample_num", 10)
    correct_dtype = torch.float32
    if use_fl16:
        correct_dtype = torch.float16
        if rank == 0:
            print("Using float16")
    elif use_fl64:
        correct_dtype = torch.float64
        if rank == 0:
            print("Using float64")
    else:
        if rank == 0:
            print("Using float32")

    if wall_time > 0:
        max_epoch_time = 120.0
        start_up_time = time.perf_counter() - start_up_time
        wall_time -= start_up_time

    for epoch in range(start_epoch, num_epochs):
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]  # write to variable so that we can print

        start_epoch_time = time.perf_counter()
        model_ddp.train()

        if isinstance(train_data_loader.sampler, DistributedSampler):
            train_data_loader.sampler.set_epoch(epoch)
        epoch_average_loss = 0
        loss = 0

        for i, data in enumerate(train_data_loader):
            # Get input data from data loader --> on CPU
            input = data[0]
            # print(f"Input shape: {input.shape}")
            # Push input data to device

            if use_fl16:
                input = input.to(torch.float16)
            elif use_fl64:
                input = input.to(torch.float64)

            input = input.to(device, non_blocking=True)

            # Linear warmup of learning rate over first epoch
            if epoch == 0:
                for g in optimizer.param_groups:
                    g["lr"] = i * (lr - start_lr) / len(train_data_loader) + start_lr

            if "latent_rollout" not in config["training"].keys():
                config["training"]["latent_rollout"] = 1

            # Generate forecast leadtime for latent rollout based on mod
            forecast_leadtime = i % config["training"]["latent_rollout"]

            padding_slice_lat = slice(
                model.padding[0][0], model.padded_resolution[0] - model.padding[0][1]
            )
            padding_slice_lon = slice(
                model.padding[1][0], model.padded_resolution[1] - model.padding[1][1]
            )

            # Asynchronously push target data @ correct lead time to device
            target_all = data[1][forecast_leadtime]
            target_all = target_all[:, :, padding_slice_lat, padding_slice_lon]
            target_all = target_all.to(device, non_blocking=True)
            if use_fl16:
                target_all = target_all.to(torch.float16)
            elif use_fl64:
                target_all = target_all.to(torch.float64)

            target = target_all[:, len(config["data"]["out_surface_variables"]) :]
            target_surface = target_all[
                :, : len(config["data"]["out_surface_variables"])
            ]

            if correct_dtype == torch.float32 and not no_scaler:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    optimizer.zero_grad()

                    # Call the model
                    output = model_ddp(
                        input, rollout=forecast_leadtime + 1, samples=sample_num
                    )

                    output = output[:, :, :, padding_slice_lat, padding_slice_lon]

                    output_pressure = output[
                        :, :, len(config["data"]["out_surface_variables"]) :
                    ]
                    output_surface = output[
                        :, :, : len(config["data"]["out_surface_variables"])
                    ]

                    loss = loss_fn(
                        output_pressure, output_surface, target, target_surface
                    )
            else:
                optimizer.zero_grad()

                # Call the model
                output = model_ddp(
                    input, rollout=forecast_leadtime + 1, samples=sample_num
                )

                output = output[:, :, :, padding_slice_lat, padding_slice_lon]

                output_pressure = output[
                    :, :, len(config["data"]["out_surface_variables"]) :
                ]
                output_surface = output[
                    :, :, : len(config["data"]["out_surface_variables"])
                ]

                loss = loss_fn(output_pressure, output_surface, target, target_surface)

            if no_scaler:
                loss.backward()

                if gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        model_ddp.parameters(), clipping_limit
                    )
            else:
                scaler.scale(loss).backward()

                if gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model_ddp.parameters(), clipping_limit
                    )

            # Call the backward algorithm and calculate the gradient of parameters

            if no_scaler:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            if dist.is_initialized():
                dist.all_reduce(loss)  # Allreduce rank-local mini-batch losses.

            if rank == 0:
                loss /= world_size  # Average allreduced rank-local mini-batch losses over all ranks.
                epoch_average_loss = epoch_average_loss + loss.detach()

                printout = f"Epoch: {epoch + 1:03d}/{int(num_epochs):03d} | Batch {i:04d}/{len(train_data_loader):04d} | Rollout {(forecast_leadtime + 1)} steps | Averaged Loss: {loss.item():.6f}"

                print(printout)

        epoch_average_loss = epoch_average_loss / len(train_data_loader)

        end_epoch_time = time.perf_counter()

        if rank == 0:
            print(
                f"Epoch: {int(epoch + 1):03d}/{int(num_epochs):03d} "
                f"Elapsed training time: {(end_epoch_time - start_epoch_time) / 60:04f} [min] "
                f"Averaged loss {epoch_average_loss.item():05f}",
            )

        # save_path = config["training"]["save_dir"]

        # state = {
        #    "model_state": model_ddp.state_dict(),
        #    "optimizer_state": optimizer.state_dict(),
        #    "scheduler_state": scheduler.state_dict(),
        #    "epoch": epoch,
        # }

        end_epoch_time = time.perf_counter()

        full_epoch_time = end_epoch_time - start_epoch_time

        if wall_time > 0:
            max_epoch_time = max(max_epoch_time, full_epoch_time)
            wall_time = wall_time - full_epoch_time

            if max_epoch_time > 0.98 * wall_time:
                print("The next epoch would not finish within the wall time.")
                break

    return


def main():
    start_time = time.perf_counter()

    # train_subset = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="path to config file",
    )
    parser.add_argument(
        "--wall_time", default="-1.", help="wall time of this job in hours"
    )
    args = parser.parse_args()

    wall_time = float(args.wall_time)

    with open(f"{args.config_file}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Set seeds for reproducibility
    set_all_seeds(2)

    mesh_dims = config_dict["mesh_dims"]  # [1, 4, 1, 1, 1]
    init_gb_process_group(mesh_dims)

    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    rank = get_process_group("DDP").rank()
    device = f"cuda:{slurm_localid}"
    world_size = get_process_group("DDP").size()

    if os.getenv("DATA_DIR") is not None:
        DATA_DIR=os.getenv('DATA_DIR')
        print(f"DATA_DIR:{DATA_DIR}")
        data_path = os.path.join(DATA_DIR, "1_5_degree_test_data/weather_test_data.zarr")
        if rank == 0:
            if not os.path.exists(data_path):
                dirname = os.path.dirname(data_path)
                print(f"Rank {rank} downloading data to {data_path}")
                os.mkdir(dirname)
                path = download_1_5_degree_data(dirname)
            
                print("Download finished")

        dist.barrier()
        config_dict["data"]["train_data_path"] = data_path
    else:
        print("DATA_DIR not set")
        return 

    print(f"Train data path: {config_dict['data']['train_data_path']}")

    config_dict["data"]["constant_masks_path"] = os.path.join(
        os.getenv("WORKDIR"), config_dict["data"]["constant_masks_path"]
    )

    print(f"Constant masks dir: {config_dict['data']['constant_masks_path']}")

    if "normalization_dir" in config_dict["data"]:
        config_dict["data"]["normalization_dir"] = os.path.join(
            os.getenv("WORKDIR"), config_dict["data"]["normalization_dir"]
        )
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    if wall_time > 0:
        wall_time = wall_time * 3600.0 - (time.perf_counter() - start_time)
    else:
        wall_time = -1.0

    training_loop(
        config_dict,
        device,
        slurm_localid,
        rank=rank,
        world_size=world_size,
        wall_time=wall_time,
    )

    if dist.is_initialized():
        dist.destroy_process_group()
        if rank == 0:
            print("Process group destroyed")


if __name__ == "__main__":
    main()

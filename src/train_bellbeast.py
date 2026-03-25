import argparse
import os
import time

import pandas as pd
import torch
import torch.distributed as dist
import yaml

# jigsaw imports
from jigsaw import utils
from torch.nn.parallel import DistributedDataParallel

# Import evaluation
from evaluation import validate

# Import utility functions
from gb.config import setup_config_dict

# Import Dataloaders
from gb.dataloaders.dataloader_0_25_degree import get_dataloader_0_25_deg
from gb.dataloaders.dataloader_1_5_degree import get_dataloader_1_5_deg
from gb.losses.up_loss import UP_AllGaussLoss
from gb.model.expert import Expert

# Import model
from gb.model.model_gb import Bellbeast
from gb.model.model_gb_tiny import TinyBellbeast
from gb.utils import (
    convert_jigsaw_vi,
    count_flops,
    count_model_parameters,
    get_process_group,
    init_gb_process_group,
    save_checkpoint,
    set_all_seeds,
    set_dtype,
)


def training_loop(config_dict, prof=None):
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")
    dtp_group = get_process_group("DTP")
    up_group = get_process_group("SP")
    dp_group = get_process_group("DP")
    ddp_group = get_process_group("DDP")
    ep_group = get_process_group("Expert")

    dtype = set_dtype(config_dict["training"]["dtype"])
    autocast = config_dict["training"]["autocast"]

    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    # Define the Model
    initialization_rank = (
        channel_group.rank() + ep_group.rank() * channel_group.size() + up_group.size()
    )
    set_all_seeds(initialization_rank)

    ########### Tiny Bellbeast
    if config_dict["data"]["xlat"] == 120 and config_dict["data"]["xlon"] == 240:
        model = TinyBellbeast(
            xlat=config_dict["data"]["xlat"],
            xlon=config_dict["data"]["xlon"],
            in_channels=config_dict["model"]["n_input_channels"],
            embed_dim=config_dict["model"]["embedding_dimension"],
            out_channels=config_dict["model"]["n_output_channels"],
            heads=config_dict["model"][
                "n_attn_heads"
            ],  # Extend this to also configure Transformer hidden dimension; Currently embed_dim is kept
            noise_dim=config_dict["model"]["noise_dim"],
            kernel_size=config_dict["model"]["patch_size"],
            window_size_outer=config_dict["model"]["window_size_outer"],
            window_size_inner=config_dict["model"]["window_size_inner"],
            n_full_blocks=config_dict["model"][
                "n_attn_blocks"
            ],  # Change to List of ints of the numbers of blocks in the outer - inner - outer SWIN
            device=device,
            channel_group=channel_group,
            ep_group=ep_group,
            up_group=up_group,
            bayesian=True,
            dtype=dtype,
        )
        # Get the data loaders
        train_dataloader = get_dataloader_1_5_deg(
            config_dict,
            dtp_process_group=dtp_group,
            dp_process_group=dp_group,
            mode="train",
            dtype=dtype,
        )

        valid_dataloader = get_dataloader_1_5_deg(
            config_dict,
            dtp_process_group=dtp_group,
            dp_process_group=dp_group,
            mode="validation",
            dtype=dtype,
        )
    ########### Grand Bellbeast
    else:
        model = Bellbeast(
            xlat=config_dict["data"]["xlat"],
            xlon=config_dict["data"]["xlon"],
            in_channels=config_dict["model"]["n_input_channels"],
            embed_dim=config_dict["model"]["embedding_dimension"],
            out_channels=config_dict["model"]["n_output_channels"],
            heads=config_dict["model"][
                "n_attn_heads"
            ],  # Extend this to also configure Transformer hidden dimension; Currently embed_dim is kept
            noise_dim=config_dict["model"]["noise_dim"],
            kernel_size=config_dict["model"]["patch_size"],
            window_size_outer=config_dict["model"]["window_size_outer"],
            window_size_inner=config_dict["model"]["window_size_inner"],
            n_full_blocks=config_dict["model"][
                "n_attn_blocks"
            ],  # Change to List of ints of the numbers of blocks in the outer - inner - outer SWIN
            dtp_rank=dtp_group.rank(),
            device=device,
            dtp_group=dtp_group,
            spatial_group=spatial_group,
            channel_group=channel_group,
            ep_group=ep_group,
            up_group=up_group,
            bayesian=True,
            parallelism=config_dict["model_parallel"]["parallelism"],
            dtype=dtype,
        )
        # Get the data loaders
        train_dataloader = get_dataloader_0_25_deg(
            config_dict,
            dtp_process_group=dtp_group,
            dp_process_group=dp_group,
            mode="train",
            dtype=dtype,
        )

        valid_dataloader = get_dataloader_0_25_deg(
            config_dict,
            dtp_process_group=dtp_group,
            dp_process_group=dp_group,
            mode="validation",
            dtype=dtype,
        )

    model = model.to(dtype).to(device)

    if dist.get_rank(group=None) == 0:
        print(
            f"Parameters: {count_model_parameters(model, config_dict['model_parallel']['parallelism'], bayesian=True)}"
        )

    # Make model expert
    model = Expert(model, ep_group)
    expert_loss_weighting = model.get_loss_weights()

    # Make model Bayesian
    convert_jigsaw_vi(model)
    model.return_log_probs = False

    # Make model data parallel
    model = DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
        process_group=ddp_group,
    )

    loss_fn = UP_AllGaussLoss(
        model,
        expert_loss_weighting,
        len(train_dataloader.dataset),
        up_group,
        dp_group,
        ep_group,
    )

    n_samples = config_dict["training"]["n_samples"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    utils.setup_precision(
        config_dict["training"]["tf32"]
    )  # Check if this works with Torch-blue

    torch.cuda.empty_cache()
    set_all_seeds(42 + up_group.rank())

    # Per-step forward, backward, and total flops
    # These numbers are already summed across all ranks (not just the dtp group)
    forward_flops, backward_flops, total_flops = count_flops(
        model, train_dataloader, loss_fn, n_samples, device, dtp_group
    )

    ## Validation, profiling and timings
    results_path = os.path.join(
        os.getenv("OUTPUT_DIR"),
        os.getenv("SLURM_JOB_PARTITION"),
        os.getenv("SLURM_JOB_ID", "interactive"),
    )

    validation_results_path = os.path.join(results_path, "validation")
    checkpoint_path = os.path.join(results_path, "checkpoints")
    timings_path = os.path.join(results_path, "timings")
    profile_dir = os.getenv("PROFILE_DIR")
    if dist.get_rank(group=None) == 0:
        os.makedirs(validation_results_path, exist_ok=True)
        with open(f"{results_path}/config.yaml", "w") as file:
            yaml.dump(config_dict, file)

        os.makedirs(timings_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(profile_dir, exist_ok=True)

    time_log = []

    ### Run Training
    for epoch in range(config_dict["training"]["n_epochs"]):
        start_time = time.perf_counter()
        avg_loss, timings = train_one_epoch(
            config_dict=config_dict,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            n_samples=n_samples,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            dtp_group=dtp_group,
            autocast=autocast,
            checkpoint_path=checkpoint_path,
            prof=prof,
        )
        end_time = time.perf_counter()

        if not config_dict["skip_validation"]:
            data_fit_loss, prior_matching_loss = validate(
                config_dict=config_dict,
                model=model,
                expert_weights=expert_loss_weighting,
                n_samples=n_samples,
                dataloader=valid_dataloader,
                device=device,
                epoch=epoch,
                results_path=validation_results_path,
            )

            # Save checkpoint after validation to have the validation loss in the checkpoint
            save_checkpoint(
                validation_results_path,
                model,
                optimizer,
                epoch,
                step=None,
                loss=data_fit_loss,
            )

        for t in timings:
            time_log.append(t)

        if dist.get_rank(group=None) == 0:
            print(
                f"Epoch {epoch} time [min]: {(end_time - start_time) / 60.0}, average loss {avg_loss}, average TFLOPs = {total_flops * len(train_dataloader) / (end_time - start_time) / 1e12}"
            )

    if config_dict["timings"]:
        timings_path = os.path.join(results_path, "timings")
        time_df = pd.DataFrame(
            time_log,
            columns=[
                "#epoch",
                "#batch",
                "step_start",
                "data_loading_start",
                "data_loading_end",
                "gpu_time",
                "step_end",
            ],
        )
        time_df.to_csv(
            os.path.join(
                timings_path,
                f"timings_rank{dist.get_rank(group=None)}_gpu{dtp_group.rank()}.csv",
            ),
            sep=";",
            index=False,
        )


def train_one_epoch(
    config_dict,
    epoch,
    dataloader,
    model,
    n_samples,
    optimizer,
    loss_fn,
    device,
    dtp_group,
    autocast,
    checkpoint_path,
    prof=None,
):
    timings = []

    model.train()
    loss_sum = 0.0

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    step_start = time.time_ns()
    cum_batch_time = 0
    for i, (x, y, dl_start, dl_end) in enumerate(dataloader):
        x = x.to(device)  # ToDO: Run test if loading data directly to GPU
        y = y.to(device)
        gpu_time = time.time_ns()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            pred = model(x, samples=n_samples)
            loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        optimizer.step()

        loss_sum = loss_sum + float(loss.detach())
        step_end = time.time_ns()
        timings.append((epoch, i, step_start, dl_start, dl_end, gpu_time, step_end))
        cum_batch_time += step_end - step_start

        if i % config_dict["training"]["checkpoint_interval"] == 0:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                i,
                loss_sum,
            )

        if i % 100 == 0:
            # Average loss over all four processes for exactness
            dist.all_reduce(loss, group=dtp_group)
            if dist.get_rank(group=None) == 0:
                print(
                    f"{epoch}, after Step {i}/{len(dataloader)}. loss: {float(loss.detach())}, rmse: {torch.nn.functional.mse_loss(pred.mean(dim=(0)), y)} Average time per batch {cum_batch_time / 100000000} ms"
                )
            cum_batch_time = 0

        step_start = time.time_ns()
        if prof:
            prof.step()

    loss_sum_all = torch.tensor(loss_sum, requires_grad=False, device=device)
    dist.all_reduce(loss_sum_all, group=dtp_group)
    average_loss = loss_sum_all.item() / max(1, len(dataloader))
    return average_loss, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")

    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("configargs", nargs="*")

    parser.add_argument("--scaling", default=False)
    parser.add_argument("--dp_processes")
    parser.add_argument("--up_processes")
    parser.add_argument("--nodes", default=2)

    args = parser.parse_args()

    with open(f"{args.config_file}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    config_dict = setup_config_dict(config_dict, args)

    ### Distributed Stuff ###
    mesh_dims = config_dict["mesh_dims"]
    init_gb_process_group(mesh_dims)
    # dist.barrier(group=None)

    global_rank = dist.get_rank(group=None)
    # global_world_size = dist.get_world_size(group=None)

    if global_rank == 0:
        print(f"Train data path: {config_dict['data']['train_data_path']}")
        print(f"Valid data path: {config_dict['data']['valid_data_path']}")
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    if config_dict["profile"] and global_rank == 3:
        import torch.profiler as tprofiler

        def trace_handler(p):
            # Export in Chrome trace format (Perfetto can open this)
            path = os.getenv("PROFILE_DIR")
            p.export_chrome_trace(f"{path}/trace-{global_rank}.json")

        print("Profiling ... Please wait")
        with tprofiler.profile(
            activities=[
                tprofiler.ProfilerActivity.CPU,
                tprofiler.ProfilerActivity.CUDA,
            ],
            # record_shapes=True,
            # profile_memory=True,
            schedule=tprofiler.schedule(skip_first=1, wait=2, warmup=2, active=3),
            # with_stack=True,
            # with_flops=True,
            on_trace_ready=trace_handler,
        ) as prof:
            training_loop(config_dict, prof)

    else:
        training_loop(config_dict)

    if dist.get_rank(group=None) == 0:
        print("FINITE!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

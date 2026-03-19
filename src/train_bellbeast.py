import argparse
import os
import time

import pandas as pd
import torch
import torch.distributed as dist

# torch blue imports
import yaml

# jigsaw imports
from jigsaw import utils
from torch.nn.parallel import DistributedDataParallel

# Import evaluation
from evaluation import validate
from gb.dataloaders.dataloader_0_25_degree import get_dataloader
from gb.losses.up_loss import UP_AllGaussLoss
from gb.model.expert import Expert

# Import model
from gb.model.model_gb import Bellbeast
from gb.utils import (
    convert_jigsaw_vi,
    count_model_parameters,
    get_process_group,
    init_gb_process_group,
    set_all_seeds,
)


def training_loop(config_dict):
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")
    dtp_group = get_process_group("DTP")
    up_group = get_process_group("SP")
    dp_group = get_process_group("DP")
    ddp_group = get_process_group("DDP")
    ep_group = get_process_group("Expert")

    time_log = []

    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    # Define the Model
    initialization_rank = dtp_group.rank() + ep_group.rank() * dtp_group.size()
    set_all_seeds(initialization_rank)

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
    )

    model = model.to(torch.float32).to(device)

    # Make model expert
    model = Expert(model, ep_group)
    expert_loss_weighting = model.get_loss_weights()

    if dist.get_rank(group=None) == 0:
        print(
            f"Parameters: {count_model_parameters(model, config_dict['model_parallel']['parallelism'])}"
        )

        if not os.path.exists(os.path.join(os.getenv("WORKDIR"), "logs/timings")):
            os.makedirs(os.path.join(os.getenv("WORKDIR"), "logs/timings"))

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

    # Get the data loaders
    train_dataloader = get_dataloader(
        config_dict,
        dtp_process_group=dtp_group,
        dp_process_group=dp_group,
        mode="train",
    )

    valid_dataloader = get_dataloader(
        config_dict,
        dtp_process_group=dtp_group,
        dp_process_group=dp_group,
        mode="validation",
    )

    validation_results_path = os.path.join(
        os.getenv("WORKDIR"),
        "results",
        os.getenv("SLURM_JOB_PARTITION"),
        os.getenv("SLURM_JOB_ID"),
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
    )  ##Check if this works with Torch-blue

    torch.cuda.empty_cache()
    set_all_seeds(42 + up_group.rank())

    ### Run Training
    for epoch in range(config_dict["training"]["n_epochs"]):
        start_time = time.perf_counter()
        avg_loss, timings = train_one_epoch(
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            n_samples=n_samples,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            dtp_group=dtp_group,
        )

        # data_fit_loss, prior_matching_loss = validate(
        validate(
            config_dict=config_dict,
            model=model,
            expert_weights=expert_loss_weighting,
            n_samples=n_samples,
            dataloader=valid_dataloader,
            device=device,
            epoch=epoch,
            results_path=validation_results_path,
        )

        end_time = time.perf_counter()

        for t in timings:
            time_log.append(t)

        if dist.get_rank(group=None) == 0:
            print(
                f"Epoch {epoch} time [min]: {(end_time - start_time) / 60.0}, average loss {avg_loss}"
            )
            # print(
            #     f"Validation loss components: Data fitting: {data_fit_loss}, Prior matching: {prior_matching_loss}"
            # )

    time_df = pd.DataFrame(
        time_log, columns=["#epoch", "#batch", "batch_start", "batch_end"]
    )
    time_df.to_csv(
        os.path.join(
            os.getenv("WORKDIR"),
            f"logs/timings/timings_rank{dist.get_rank(group=None)}_gpu{dtp_group.rank()}.csv",
        ),
        sep=";",
        index=False,
    )


def train_one_epoch(
    epoch, dataloader, model, n_samples, optimizer, loss_fn, device, dtp_group
):
    timings = []
    timings.append((epoch, -1, time.time_ns(), time.time_ns()))

    model.train()
    loss_sum = 0.0

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    for i, (x, y, start, end) in enumerate(dataloader):
        timings.append((epoch, i, start, end))

        x = x.to(device)  # ToDO: Run test if loading data directly to GPU
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x, samples=n_samples)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        optimizer.step()

        loss_sum = loss_sum + float(loss.detach())

        if i % 100 == 0:
            # # Average loss over all four processes for exactness
            dist.all_reduce(loss, group=dtp_group)
            if dist.get_rank(group=None) == 0:
                print(
                    f"{epoch}, after Step {i}/{len(dataloader)}. loss: {float(loss.detach())}"
                )
    loss_sum_all = torch.tensor(loss_sum, requires_grad=False, device=device)
    dist.all_reduce(loss_sum_all, group=dtp_group)
    average_loss = loss_sum_all.item() / max(1, len(dataloader))
    return average_loss, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(f"{args.config_file}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    ### Distributed Stuff ###
    mesh_dims = config_dict["mesh_dims"]
    init_gb_process_group(mesh_dims)

    global_rank = dist.get_rank(group=None)
    # global_world_size = dist.get_world_size(group=None)

    ### Data Stuff ###
    WORKDIR = os.getenv("WORKDIR")

    if os.getenv("DATA_DIR") is not None:
        config_dict["data"]["train_data_path"] = (
            os.getenv("DATA_DIR") + config_dict["data"]["train_data_path"]
        )
        config_dict["data"]["valid_data_path"] = (
            os.getenv("DATA_DIR") + config_dict["data"]["valid_data_path"]
        )
        config_dict["data"]["constant_masks_path"] = (
            os.getenv("DATA_DIR") + config_dict["data"]["constant_masks_path"]
        )

    # append n_masks for odd number of masks (the dataset does this, too)
    n_masks = len(config_dict["data"]["constant_masks"])
    if n_masks % 2 == 1:
        n_masks += 1

    # default case: load only one timestep
    if config_dict["data"]["n_in_timesteps"] not in [1, 2]:
        config_dict["data"]["n_in_timesteps"] = 1

    n_input_vars = (
        config_dict["data"]["n_in_timesteps"] * config_dict["data"]["n_variables"]
    )

    # we'll set this automatically, to avoid confusion when defining the config,
    # as n_input_channels depends on how many timesteps we use as concatenated input
    config_dict["model"]["n_input_channels"] = n_input_vars + n_masks
    config_dict["model"]["embedding_dimension"] = (
        config_dict["model"]["embedding_dimension"]
        * config_dict["data"]["n_in_timesteps"]
    )

    if "normalization_dir" in config_dict["data"]:
        config_dict["data"]["normalization_dir"] = (
            WORKDIR + "/" + config_dict["data"]["normalization_dir"]
        )

    if global_rank == 0:
        print(f"Train data path: {config_dict['data']['train_data_path']}")
        print(f"Valid data path: {config_dict['data']['valid_data_path']}")
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    training_loop(config_dict)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

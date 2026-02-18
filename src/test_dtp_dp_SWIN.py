import argparse
import os
import random
import time

import torch
import torch.distributed as dist

# torch blue imports
import torch_blue
import yaml

# Jigsaw imports
from Jigsaw import Normalization, utils
from torch.nn.parallel import DistributedDataParallel
from torch_blue.vi import convert_to_vimodule

from gb.dataloaders.dataloader_0_25_degree import get_dataloader
from gb.losses.UP_Loss import UP_AllGaussLoss
from gb.model.NICE_SWIN_Jigsaw_025 import NICE_SWIN

# Import model
from gb.utils import get_process_group, init_gb_process_group


def set_all_seeds(seed_value: int) -> None:
    """Set random seeds."""
    random.seed(seed_value)  # Python random module
    torch.manual_seed(seed_value)  # pytorch random number generator for CPU
    torch.cuda.manual_seed(seed_value)  # pytorch random number generator for all GPUs
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU.
    # torch.backends.cudnn.deterministic = True  # use deterministic algorithms.
    # torch.backends.cudnn.benchmark = False  # disable to be deterministic.
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # python hash seed.


def training_loop(config_dict):
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")
    dtp_group = get_process_group("DTP")
    up_group = get_process_group("SP")
    dp_group = get_process_group("DP")

    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)
    print(f"Rank: {dist.get_rank()}, local rank: {slurm_localid}, device: {device}")

    # Define the Model
    set_all_seeds(dtp_group.rank())

    model = NICE_SWIN(
        xlat=config_dict["data"]["xlat"],
        xlon=config_dict["data"]["xlon"],
        in_channels=config_dict["model"]["n_input_channels"],
        embed_dim=config_dict["model"]["embedding_dimension"],
        out_channels=config_dict["model"]["n_output_channels"],
        heads=config_dict["model"][
            "n_attn_heads"
        ],  # Extend this to also configure Transformer hidden dimension; Currently embed_dim is kept
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
        bayesian=True,
    )

    model = model.to(torch.float32).to(device)

    # Make model Bayesian
    torch_blue.vi.utils.convert.ban_convert(Normalization.DistributedLayerNorm)
    # torch_blue.vi.utils.convert.ban_convert(Normalization.DistributedNoParamLayerNorm)
    torch_blue.vi.utils.convert.ban_convert(
        Normalization.DistributedNICENorm, ban_mode="submodule"
    )

    convert_to_vimodule(model)
    # predictive_distribution = Normal()

    model.return_log_probs = False

    # Make model data parallel
    model = DistributedDataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
        process_group=dp_group,
    )

    # Get the data loaders
    train_dataloader = get_dataloader(
        config_dict,
        dtp_process_group=dtp_group,
        dp_process_group=dp_group,
        mode="train",
    )
    # valid_dataloader = get_dataloader(
    #     config_dict,
    #       dtp_process_group = dtp_group,
    #        dp_process_group=dp_group,
    #     mode="validation"
    # )
    loss_fn = UP_AllGaussLoss(model, len(train_dataloader.dataset), up_group)
    n_samples = config_dict["training"]["n_samples"]
    # loss_fn = vi.KullbackLeiblerLoss(
    #        predictive_distribution, dataset_size=len(train_dataloader.dataset)
    #    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    utils.setup_precision(
        config_dict["training"]["tf32"]
    )  ##Check if this works with Torch-blue
    torch.cuda.empty_cache()
    set_all_seeds(42 + up_group.rank())

    ### Run Training
    for epoch in range(config_dict["training"]["n_epochs"]):
        start_time = time.perf_counter()
        avg_loss = train_one_epoch(
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            n_samples=n_samples,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            dtp_group=dtp_group,
        )
        end_time = time.perf_counter()

        if dist.get_rank(group=None) == 0:
            print(
                f"Epoch {epoch} time [min]: {(end_time - start_time) / 60.0}, average loss {avg_loss}"
            )


def train_one_epoch(
    epoch, dataloader, model, n_samples, optimizer, loss_fn, device, dtp_group
):
    model.train()
    loss_sum = 0.0

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)  # ToDO: Run test if loading data directly to GPU
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(
            x, samples=n_samples
        )  # Add samples = config_dict["training"]["n_samples"]
        loss = loss_fn(pred, y)

        # # Average loss over all four processes for exactness
        dist.all_reduce(loss, group=dtp_group)

        # Backpropagation
        loss.backward()

        optimizer.step()

        loss_sum = loss_sum + float(loss.detach())
        if dist.get_rank(group=None) == 0:
            print(
                f"{epoch}, after Step {i}/{len(dataloader)}. loss: {float(loss.detach())}"
            )

    average_loss = loss_sum / max(1, len(dataloader))
    return average_loss


def validate_epoch():
    pass


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

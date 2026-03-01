import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml

# Jigsaw imports
from jigsaw import utils

# torch blue imports
from torch_blue import vi
from torch_blue.vi import convert_to_vimodule
from torch_blue.vi.distributions import Normal

# Import model
from gb.model.test_models.vit import ViT
from gb.model.test_models.weather_mixer import WeatherMixer
from gb.utils import _create_process_mesh_dict


def training_loop(
    config_dict: dict, device: torch.device, localid: int, rank: int
) -> None:
    """Training loop for MLP Mixer model."""
    parallelism = config_dict["model_parallel"]["parallelism"]
    # Set all seeds in the same model group to be different
    utils.set_all_seeds(1 + rank % parallelism)

    # Set up model  groups
    pg_dict = _create_process_mesh_dict(
        config_dict["mesh_dims"],
        device_type="cuda",
    )
    model_group, row_subgroup, col_subgroup, _ = (
        pg_dict["DTP"],
        pg_dict["JRow"],
        pg_dict["JCol"],
        pg_dict["DDP"],
    )
    n_vars_local = 69 if parallelism == 1 else 36

    # Define model and load model
    if config_dict["model"]["type"] == "Mixer":
        model = (
            WeatherMixer(
                config_dict, rank, device, parallelism, model_group, col_subgroup
            )
            .to(torch.float32)
            .to(device)
        )
    elif config_dict["model"]["type"] == "ViT":
        match parallelism:
            case 1:
                model = (
                    ViT.Transformer_Sequential(
                        config_dict,
                        rank,
                        device,
                        parallelism,
                        model_group,
                        row_subgroup,
                        col_subgroup,
                    )
                    .to(torch.float32)
                    .to(device)
                )
            case _:
                model = (
                    ViT.ViT(
                        config_dict,
                        rank,
                        device,
                        parallelism,
                        model_group,
                        row_subgroup,
                        col_subgroup,
                    )
                    .to(torch.float32)
                    .to(device)
                )

    if rank == 0:
        print(
            f"model has {utils.count_parameters(model) * parallelism / 1e6} million parameters"
        )
    utils.set_all_seeds_model_group(1, rank=rank, parallelism=parallelism)
    train_dataloader, valid_dataloader = utils.build_dataloaders(
        config_dict, parallelism, rank
    )

    # Synchronize parameters
    for name, param in model.named_parameters():
        if param is not None:
            if "bias" in name:
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG, group=col_subgroup)

    # Convert to VI module
    convert_to_vimodule(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    predictive_distribution = Normal()
    loss_fn = vi.KullbackLeiblerLoss(
        predictive_distribution, dataset_size=config_dict["data"]["train_subset"]
    )

    utils.setup_precision(config_dict["training"]["tf32"])

    n_epochs = config_dict["training"]["n_epochs"]
    # Train for 10 epochs
    torch.cuda.empty_cache()
    for epoch in range(n_epochs):
        _ = train_one_epoch(
            config_dict,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            n_vars_local=n_vars_local,
            rank=rank,
            col_group=col_subgroup,
        )

        torch.cuda.synchronize()

        # model.eval()

        # val_loss = evaluation.validate(
        #    config_dict,
        #    model,
        #    valid_dataloader,
        #    valid_dataset_info,
        #    rank,
        #    device,
        #    epoch,
        #    config_dict["training"]["save_dir"],
        #    mse_file_path="rmse",
        #    acc_file_path="acc",
        #    world_size=world_size,
        #    model_subgroup_col=col_subgroup,
        #    ddp_group=ddp_group,
        #    model_group=model_group,
        # )

        # if rank < parallelism:
        #    utils.save_model(
        #        model,
        #        optimizer,
        #        None,
        #        config_dict["training"]["save_dir"],
        #        rank,
        #        epoch,
        #    )
        # if rank == 0:
        #    print(
        #        f"Epoch {epoch}. Average training loss {avg_loss}. Average validation loss: {val_loss}"
        #    )


def train_one_epoch(
    config_dict: dict,
    epoch: int,
    dataloader,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: torch.nn.modules.loss,
    device: torch.device,
    n_vars_local: int,
    rank: int,
    col_group: dist.ProcessGroup,
) -> float:
    """
    Train for one epoch.

    Tensor shapes:
        input_data: [B, T, H, W, C] or as produced by Dataloaders
        output:     [B, 720, 720, 36] -> reshaped to [B, xlat, xlon, 36]
        target_data: list-like per rollout; target_data = data[1][rollout]
    """
    model.train()
    epoch_loss = 0.0

    start_time = time.perf_counter()

    utils.set_all_seeds(1)

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y[0].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(
            x, samples=config_dict["training"]["n_samples"]
        )  # [B, 720, 720, 36]

        loss = loss_fn(pred, y.flatten(start_dim=1, end_dim=2))

        # Backpropagation
        loss.backward()
        # # Average loss over all four processes for exactness
        # dist.all_reduce(loss)
        loss = loss / config_dict["model_parallel"]["parallelism"]

        optimizer.step()

        epoch_loss = epoch_loss + float(loss.detach())
        if rank == 0:
            print(
                f"{epoch}, after Step {i}/{len(dataloader)}. loss: {float(loss.detach())}"
            )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    epoch_average_loss = epoch_loss / max(1, len(dataloader))
    if rank == 0:
        print(
            f"Epoch {epoch} time [min]: {(end_time - start_time) / 60.0}, average loss {epoch_average_loss}"
        )

    # if config_dict["model_parallel"]["parallelism"] == 2:
    #     pred = pred.mean(dim=0).reshape(1, config_dict["data"]["xlat"], config_dict["data"]["xlon"], config_dict["data"]["n_global_variables"]//2)

    return epoch_average_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    WORKDIR = os.getenv("WORKDIR")

    with open(f"{args.config_file}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_cuda()

    # Override configuration with environment variables
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

    print(f"Train data path: {config_dict['data']['train_data_path']}")
    print(f"Valid data path: {config_dict['data']['valid_data_path']}")

    if "normalization_dir" in config_dict["data"]:
        config_dict["data"]["normalization_dir"] = (
            WORKDIR + "/" + config_dict["data"]["normalization_dir"]
        )
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    training_loop(config_dict, device, localid, rank)
    dist.destroy_process_group()

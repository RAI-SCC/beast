import argparse
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml

import eval
import utils

import Model_GB as Model # Import model

# torch blue imports
from torch_blue.vi import convert_to_vimodule
from torch_blue import vi
from torch_blue.vi.distributions import Normal


def training_loop(
    config_dict: dict, device: torch.device, localid: int, rank: int
) -> None:
    """Training loop for MLP Mixer model."""
    n_parallel = config_dict["model_parallel"]["parallelism"]
    # Set all seeds in the same model group to be different
    utils.set_all_seeds(1 + rank % n_parallel)

    # Set up model  groups
    match n_parallel:
        case 1:
            model_group, ddp_group, _, model_subgroup_col = utils.setup_groups(
                rank, n_parallel, 1, 1
            )
            n_vars_local = 69
        case 2:
            model_group, ddp_group, row_subgroup, col_subgroup = utils.setup_groups(
                rank, n_parallel, 1, 2
            )
            n_vars_local = 36
        case 4:
            model_group, ddp_group, _, col_subgroup = utils.setup_groups(
                rank, n_parallel, 2, 2
            )
            n_vars_local = 36

    # Define model and load model
    if n_parallel == 2:
        model = (
            Model.Model_twoway(config_dict, rank, device, n_parallel, model_group)
            .to(device)
            .to(torch.float32)
        )
    elif n_parallel == 4:
        model = (
            Model.Model(
                config_dict, rank, device, n_parallel, model_group, col_subgroup
            )
            .to(torch.float32)
            .to(device)
        )
    if rank == 0:
        print(
            f"model has {utils.count_parameters(model) * n_parallel / 1e6} million parameters"
        )

    train_dataloader, valid_dataloader = utils.build_dataloaders(
        config_dict, n_parallel, rank
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

    epoch = 0
    # Train for 10 epochs
    torch.cuda.empty_cache()
    for epoch in range(epoch, 100):
        train_one_epoch(
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

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y[0].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x, samples=4)  # [B, 720, 720, 36]

        loss = loss_fn(pred, y.flatten(start_dim=1, end_dim=2))

        # Backpropagation
        loss.backward()
        # # Average loss over all four processes for exactness
        dist.all_reduce(loss)
        loss = loss / 4

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

    if config_dict['model_parallel']['parallelism'] == 2:
        pred = pred.mean(dim=0).reshape(1, 720, 1440, 36)
        eval.gen_plots(config_dict, x, pred, y, rank, ".", epoch, True)

    return epoch_average_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_cuda()

    training_loop(config_dict, device, localid, rank)
    dist.destroy_process_group()

import sys

sys.path.append("./src/")  # for utils import

import argparse

import jigsaw
import torch
import torch.distributed as dist
import yaml
from jigsaw import convolutions

from gb.comm import get_process_group, init_gb_process_group
from gb.utils import (
    convert_jigsaw_vi,
    set_all_seeds,
)


def main():
    # start_time = time.perf_counter()

    # train_subset = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="path to config file",
    )
    args = parser.parse_args()

    # wall_time = float(args.wall_time)

    with open(f"{args.config_file}", "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Set seeds for reproducibility
    set_all_seeds(2)

    mesh_dims = config_dict["mesh_dims"]  # [1, 4, 1, 1, 1]
    # group_dict: dict[str, dist.ProcessGroup] =
    init_gb_process_group(mesh_dims)
    # init_gb_process_group([1, 4, 1, 1, 1])  # ["Expert", "DP", "SP", "JCol", "JRow"]

    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    rank = get_process_group("DDP").rank()
    # world_size = get_process_group("DDP").size()

    conversion_test(device)

    if dist.is_initialized():
        dist.destroy_process_group()
        if rank == 0:
            print("Process group destroyed")


def sum_square_parameters(model):
    return sum(
        (p.detach() ** 2).sum().detach().item()
        for p_name, p in model.named_parameters()
        if p.requires_grad and "log_" not in p_name
    )


def conversion_test(device):
    full_dtp_group = get_process_group("DTP")
    dtp_col_group = get_process_group("JSpatial")  # get_process_group("JCol")
    dtp_row_group = get_process_group("JChannel")  # get_process_group("JRow")

    local_dtp_col = dtp_col_group.rank()
    local_dtp_row = dtp_row_group.rank()
    total_dtp_cols = dtp_col_group.size()

    # mlp = jigsaw.MLP.DistributedMLP(
    #     in_dim=16,
    #     hidden_dim=32,
    #     out_dim=16,
    #     rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
    #     device=device,
    #     n_parallel=full_dtp_group.size(),
    #     model_group=full_dtp_group,
    # )

    xwt = jigsaw.linear.DistributedXWT(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xw = jigsaw.linear.DistributedXW(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xtw = jigsaw.linear.DistributedXTW(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xwt_sync = jigsaw.linear.DistributedXWTSync(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
        sync_group=dtp_col_group,
    )

    # conv = convolutions.DistributedConv2D(
    #     in_channels=16,
    #     out_channels=64,
    #     kernel_size=3,
    #     stride=1,
    #     padding=0,
    #     dilation=1,
    #     groups=1,
    #     rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
    #     spatial_group=dtp_col_group,
    #     channel_group=dtp_row_group,
    #     device=device,
    #     parallelism=full_dtp_group.size(),
    #     bias=True,
    #     bayesian=True,
    # )

    trp_conv = convolutions.DistributedConvTranspose2D(
        in_channels=16,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        spatial_group=dtp_col_group,
        channel_group=dtp_row_group,
        device=device,
        parallelism=full_dtp_group.size(),
        bias=True,
        bayesian=True,
    )

    # convert_jigsaw_vi(mlp)
    # print(f"Rank {full_dtp_group.rank()}, MLP Targets: linear1: ~{18}, linear2: ~{8.5} \\ Found:")

    convert_jigsaw_vi(xwt)
    print(
        f"Rank {full_dtp_group.rank()}, XWT Target: ~{68} \n Found: {sum_square_parameters(xwt)}"
    )

    convert_jigsaw_vi(xw)
    print(
        f"Rank {full_dtp_group.rank()}, XW Target: ~{68} \n Found: {sum_square_parameters(xw)}"
    )

    convert_jigsaw_vi(xtw)
    print(
        f"Rank {full_dtp_group.rank()}, XTW Target: ~{68} \n Found: {sum_square_parameters(xtw)}"
    )

    convert_jigsaw_vi(xwt_sync)
    print(
        f"Rank {full_dtp_group.rank()}, XWTSync Target: ~{68} \n Found: {sum_square_parameters(xwt_sync)}"
    )

    # convert_jigsaw_vi(conv)
    # print(f"Rank {full_dtp_group.rank()}, Conv2d Target: unknown")

    convert_jigsaw_vi(trp_conv)
    print(
        f"Rank {full_dtp_group.rank()}, ConvTranspose2d Target: ~{16.1} \n Found: {sum_square_parameters(trp_conv)}"
    )


if __name__ == "__main__":
    main()

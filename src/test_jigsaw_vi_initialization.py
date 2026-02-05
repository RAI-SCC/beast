import argparse
import os

import Jigsaw
import torch
import torch.distributed as dist
import yaml

from gb.utils import convert_jigsaw_vi, get_process_group, init_gb_process_group


def set_all_seeds(seed: int) -> None:
    """Set random seeds."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    rank = get_process_group("DDP").rank()
    device = f"cuda:{slurm_localid}"
    # world_size = get_process_group("DDP").size()

    conversion_test(device)

    if dist.is_initialized():
        dist.destroy_process_group()
        if rank == 0:
            print("Process group destroyed")


def conversion_test(device):
    full_dtp_group = get_process_group("DTP")
    dtp_col_group = get_process_group("JCol")
    dtp_row_group = get_process_group("JRow")

    local_dtp_col = dtp_col_group.rank()
    local_dtp_row = dtp_row_group.rank()
    total_dtp_cols = dtp_col_group.size()

    mlp = Jigsaw.MLP.DistributedMLP(
        in_dim=16,
        hidden_dim=32,
        out_dim=16,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        n_parallel=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xwt = Jigsaw.Linear.DistributedXWT(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xw = Jigsaw.Linear.DistributedXW(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xtw = Jigsaw.Linear.DistributedXTW(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xwt_sync = Jigsaw.Linear.DistributedXWTSync(
        in_dim=16,
        out_dim=64,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        parallelism=full_dtp_group.size(),
        model_group=full_dtp_group,
        sync_group=dtp_col_group,
    )

    conv = Jigsaw.Convolutions.DistributedConv2D(
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

    trp_conv = Jigsaw.Convolutions.DistributedConvTranspose2D(
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

    convert_jigsaw_vi(mlp)
    print(f"Targets: linear1: ~{18}, linear2: ~{8.5}")

    convert_jigsaw_vi(xwt)
    print(f"Target: ~{68}")

    convert_jigsaw_vi(xw)
    print(f"Target: ~{68}")

    convert_jigsaw_vi(xtw)
    print(f"Target: ~{68}")

    convert_jigsaw_vi(xwt_sync)
    print(f"Target: ~{68}")

    convert_jigsaw_vi(conv)
    # print(f"Target: ~{68}")

    convert_jigsaw_vi(trp_conv)
    # print(f"Target: ~{68}")


if __name__ == "__main__":
    main()

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

    xw = Jigsaw.Linear.DistributedXW(
        in_dim=16,
        out_dim=32,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        n_parallel=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    xtw = Jigsaw.Linear.DistributedXTW(
        in_dim=16,
        out_dim=32,
        rank=(local_dtp_col + local_dtp_row * total_dtp_cols),
        device=device,
        n_parallel=full_dtp_group.size(),
        model_group=full_dtp_group,
    )

    convert_jigsaw_vi(mlp)
    convert_jigsaw_vi(xw)
    convert_jigsaw_vi(xtw)


if __name__ == "__main__":
    main()

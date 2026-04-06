import argparse

import torch
import torch.distributed as dist
import yaml

from gb.config import setup_config_dict
from gb.setup import (
    get_process_group,
    init_gb_process_group,
    load_configuration_yaml,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--resume")
    parser.add_argument("--transplant")
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--profile", action="store_true")

    parser.add_argument("configargs", nargs="*")

    args = parser.parse_args()

    resume_dir = None if args.resume is None else f"{args.resume}"

    if args.config_file is not None:
        with open(f"{args.config_file}", "r") as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)
        config_dict = setup_config_dict(config_dict, args)
    else:
        if args.transplant is not None:
            raise RuntimeError(
                "You cannot transplant weights without giving a config file. Please set --config_file"
            )
        config_dict = load_configuration_yaml(resume_dir)
        config_dict = setup_config_dict(config_dict, args, resume=True)

    ### Distributed Stuff ###
    mesh_dims = config_dict["mesh_dims"]
    init_gb_process_group(mesh_dims)
    dist.barrier(group=None)
    global_rank = dist.get_rank(group=None)

    if global_rank == 0:
        print(f"dist is initialized: {dist.is_initialized()}")

    device_id = torch.cuda.current_device()
    device = torch.device("cuda", device_id)

    warmup_all = torch.tensor(1.0).to(device)
    warmup_ddp = torch.tensor(1.0).to(device)

    dist.all_reduce(warmup_all, op=dist.ReduceOp.SUM)
    dist.all_reduce(warmup_ddp, op=dist.ReduceOp.SUM, group=get_process_group("DDP"))

    # print(f'Rank {global_rank} finished torch.distributed warmup with {warmup_all} ranks and {warmup_ddp} ddp ranks')
    # global_world_size = dist.get_world_size(group=None)

    if global_rank == 0:
        print(f"Train data path: {config_dict['data']['train_data_path']}")
        print(f"Mask path: {config_dict['data']['constant_masks_path']}")
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    from training import training_loop

    training_loop(config_dict)

    if dist.get_rank(group=None) == 0:
        print("FINITE!")

    dist.barrier(group=None)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

import argparse
import os

import torch.distributed as dist
import yaml

from gb.comm import (
    init_gb_process_group,
)
from gb.config import load_configuration_yaml, setup_config_dict


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
        print(f"Train data path: {config_dict['data']['train_data_path']}")
        print(f"Mask path: {config_dict['data']['constant_masks_path']}")
        print(f"Normalization dir: {config_dict['data']['normalization_dir']}")

    from training import training_loop

    if config_dict["profile"] and global_rank == 3:
        import torch.profiler as tprofiler

        print("Profiling ... Please wait")
        with tprofiler.profile(
            activities=[
                tprofiler.ProfilerActivity.CPU,
                tprofiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            schedule=tprofiler.schedule(skip_first=1, wait=2, warmup=2, active=3),
            with_stack=True,
            # with_flops=True,
            on_trace_ready=tprofiler.tensorboard_trace_handler(
                f"{os.getenv('PROFILE_DIR')}/tb_logs_rank_{global_rank}"
            ),
        ) as prof:
            training_loop(config_dict, prof)

    else:
        training_loop(config_dict)

    if dist.get_rank(group=None) == 0:
        print("FINITE!")

    dist.barrier(group=None)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

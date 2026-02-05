import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml

# Jigsaw imports
from Jigsaw import utils

# torch blue imports
from torch_blue import vi
from torch_blue.vi import convert_to_vimodule
from torch_blue.vi.distributions import Normal

# Import model
from gb.model.SWIN_Jigsaw_025 import SWIN 
from gb.utils import init_gb_process_group, get_process_group
from gb.losses.UP_Loss import UP_AllGaussLoss

def set_all_seeds(seed: int) -> None:
    """Set random seeds."""
    #os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def training_loop(config_dict):
    spatial_group = get_process_group("JSpatial")
    channel_group = get_process_group("JChannel")
    dtp_group = get_process_group("DTP")
    #up_group = get_process_group("SP")
    dp_group = get_process_group("DP")

    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    device = f'cuda:{slurm_localid}'

    # Define the Model
    set_all_seeds(dtp_group.rank())
    
    model = SWIN(
        xlat = config_dict['data']['xlat'],
        xlon = config_dict['data']['xlon'],
        in_channels = config_dict['model']['n_input_channels'],
        embed_dim = config_dict['model']['embedding_dimension'],
        out_channels = config_dict['model']['n_output_channels'],
        heads = config_dict['model']['n_attn_heads'], # Extend this to also configure Transformer hidden dimension; Currently embed_dim is kept 
        kernel_size = config_dict['model']['patch_size'],
        window_size_outer = config_dict['model']['window_size_outer'],
        window_size_inner = config_dict['model']['window_size_inner'],
        n_full_blocks = config_dict['model']['n_attn_blocks'], # Change to List of ints of the numbers of blocks in the outer - inner - outer SWIN
        dtp_rank = dtp_group.rank(),
        device = device,
        dtp_group = dtp_group,
        spatial_group = spatial_group,
        channel_group = channel_group
    ) 

    model = model.to(torch.float32).to(device)
    # Synchronize parameters; ToDO: Check this @deifilia :D
    for name, param in model.named_parameters():
        if param is not None:
            if "bias" in name:
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG, group=spatial_group)

    set_all_seeds(dp_group.rank())# Check if needed, or which PG needs the same seeds here?!?! Should maybe be UP?
    
    ### Get Data Loaders
    train_dataloader, valid_dataloader = utils.build_dataloaders(
        config_dict, 4, dtp_group.rank()
    )
    # train_subset = config_dict["data"]["train_subset"]
    # valid_subset = config_dict["data"]["valid_subset"]
    # train_subset = -1 if train_subset < 0 else train_subset
    # valid_subset = -1 if valid_subset < 0 else valid_subset

    # train_dataloader, _ = get_dataloader(
    #     config_dict,
    #     config_dict["data"]["train_data_path"],
    #     mode="train",
    #     distributed=(not model_parallel),
    #     batch_size=config_dict["training"]["train_batch"],
    #     subset_size=train_subset,
    #     rank=rank,
    # )
    # valid_dataloader, _ = get_dataloader(
    #     config_dict,
    #     config_dict["data"]["valid_data_path"],
    #     mode="validation",
    #     distributed=(not model_parallel),
    #     batch_size=config_dict["training"]["valid_batch"],
    #     subset_size=valid_subset,
    #     rank=rank,
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # Make Model Bayesian
    #torch_blue.vi.utils.convert.ban_convert(Normalization.DistributedLayerNorm)
    #convert_to_vimodule(model)
    #predictive_distribution = Normal()
    #loss_fn = vi.KullbackLeiblerLoss(
    #    predictive_distribution, dataset_size=config_dict["data"]["train_subset"]
    #)
    # model_ddp = DistributedDataParallel(
    #     model,
    #     device_ids=[slurm_localid],
    #     output_device=slurm_localid,
    #     process_group=dp_group,
    # )

    utils.setup_precision(config_dict["training"]["tf32"]) ##Check if this works with Torch-blue
    torch.cuda.empty_cache()

    ### Run Training
    for epoch in range(config_dict["training"]["n_epochs"]):
        start_time = time.perf_counter()
        avg_loss = train_one_epoch(
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            dtp_group=dtp_group,
        )
        end_time = time.perf_counter()

        if dist.get_rank(group=None) == 0:
            print(f"Epoch {epoch} time [min]: {(end_time - start_time) / 60.0}, average loss {avg_loss}")



def train_one_epoch(
    epoch,
    dataloader,
    model,
    optimizer,
    loss_fn,
    device,
    dtp_group
):
    model.train()
    loss_sum = 0.0

    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device) # ToDO: Run test if loading data directly to GPU
        y = y[0].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)

        loss = loss_fn(pred, y)

        # # Average loss over all four processes for exactness
        dist.all_reduce(loss, group=dtp_group)
        loss = loss / 4

        # Backpropagation
        loss.backward()

        optimizer.step()

        loss_sum = loss_sum + float(loss.detach())
        if dist.get_rank(group=None) == 0:
            print(f"{epoch}, after Step {i}/{len(dataloader)}. loss: {float(loss.detach())}")

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
    global_world_size = dist.get_world_size(group=None)

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
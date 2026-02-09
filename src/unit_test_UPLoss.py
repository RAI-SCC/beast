import os

import torch
import torch_blue

from gb.model.UP_Loss import UP_AllGaussLoss
from gb.utils import get_process_group, init_gb_process_group


def set_all_seeds(seed: int) -> None:
    """Set random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DummyModule(torch_blue.vi.VIModule):
    def __init__(self):
        super().__init__()

        self.linear = torch_blue.vi.VILinear(3, 3)

    def forward(self, x):
        return self.linear(x)


def main():
    set_all_seeds(42)

    mesh_dims = [1, 1, 2, 1, 1]  # two-way UP
    # mesh_dims = [1, 1, 4, 1, 1] # four-way UP

    init_gb_process_group(mesh_dims)
    # init_gb_process_group([1, 4, 1, 1, 1])  # ["Expert", "DP", "SP", "JCol", "JRow"]

    dataset_size = 10

    up_group = get_process_group("SP")

    slurm_localid = int(os.getenv("SLURM_LOCALID") or 0)
    rank = up_group.rank()
    device = f"cuda:{slurm_localid}"
    world_size = up_group.size()

    # model output
    # Changed to floats to increase randomness
    list_of_samples = [
        [[1.1, 2.2, 3.8]],
        [[2.2, 3.1, 1.9]],
        [[3.3, 1.3, 2.6]],
        [[3.4, 2.4, 1.5]],
        [[2.5, 3.5, 1.6]],
        [[1.6, 3.6, 2.4]],
        [[1.7, 1.7, 1.3]],
        [[2.8, 2.8, 2.4]],
    ]
    
    
    n_samples_total = 8

    target = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    list_of_tensors = [
        torch.tensor(entry, device=device, dtype=torch.float32)
        for entry in list_of_samples
    ]

    dummy_module = DummyModule().to(device)

    up_loss_no_prior = UP_AllGaussLoss(dummy_module, dataset_size, up_group, heat=0.0)

    up_loss_with_prior = UP_AllGaussLoss(dummy_module, dataset_size, up_group, heat=1.0)

    full_samples = torch.stack(list_of_tensors, dim=0)
    print(f"Rank {rank}: all samples mean: {full_samples.mean(dim=0)}")
    print(f"Rank {rank}: all samples std: {full_samples.std(dim=0)}")

    # Create separate tensor of total samples
    total_samples = torch.tensor(list_of_samples).detach().to(device).requires_grad_(True)
    
    ideal_value_no_prior = (
        (total_samples.mean(dim=0) - target) ** 2 / (2 * total_samples.std(dim=0) ** 2)
        + torch.log(total_samples.std(dim=0))
    ).sum() * dataset_size
    ideal_value_with_prior = (
        ideal_value_no_prior + up_loss_with_prior.kl_loss.prior_matching()
    )
    
    print(f"Rank {rank}: ideal value without prior {ideal_value_no_prior.item()}")
    print(f"Rank {rank}: ideal value with prior {ideal_value_with_prior.item()}")

    # Split up sample data into different processes
    local_samples = torch.stack(
        list_of_tensors[
            (rank * (n_samples_total // world_size)) : (
                (rank + 1) * (n_samples_total // world_size)
            )
        ],
        dim=0,
    ).detach().to(device).requires_grad_(True)
    
    print(f"Rank {rank}: local samples mean: {local_samples.mean(dim=0)}")
    print(f"Rank {rank}: local samples std: {local_samples.std(dim=0)}")

    local_value_no_prior = up_loss_no_prior(local_samples, target)
    # local_value_with_prior = up_loss_with_prior(local_samples, target)

    print(f"Rank {rank}: local value without prior {local_value_no_prior.item()}")
    # print(f"Rank {rank}: local value with prior {local_value_with_prior.item()}")

    assert torch.all(torch.isclose(local_value_no_prior, ideal_value_no_prior)), (
        "Local value without prior differs from ideal value"
    )
    # assert torch.allclose(local_value_with_prior, ideal_value_with_prior), (
    #     "Local value with prior differs from ideal value"
    # )
    
    # Call backward pass
    ideal_value_no_prior.backward()
    local_value_no_prior.backward()
    
    
    ideal_grad = total_samples.grad[
        rank * (n_samples_total // world_size) : (rank + 1) * (n_samples_total // world_size)
    ]
    
    # Print statements for local 
    print(f'Local samples grad {local_samples.grad}')
    if rank == 0:
        print(f'Ideal samples grad {total_samples.grad}')
    
    assert torch.allclose(local_samples.grad, ideal_grad), (
        "gradient wrt input data differs from ideal value"
    )
    
    
    torch.distributed.destroy_process_group()
if __name__ == "__main__":
    main()

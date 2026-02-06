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

    list_of_samples = [
        [[1.0, 2.0, 3.0]],
        [[2.0, 3.0, 1.0]],
        [[3.0, 1.0, 2.0]],
        [[3.0, 2.0, 1.0]],
        [[2.0, 3.0, 1.0]],
        [[1.0, 3.0, 2.0]],
        [[1.0, 1.0, 1.0]],
        [[2.0, 2.0, 2.0]],
    ]

    total_samples = 8

    target = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

    list_of_tensors = [
        torch.tensor(entry, device=device, dtype=torch.float32)
        for entry in list_of_samples
    ]

    dummy_module = DummyModule()

    up_loss_no_prior = UP_AllGaussLoss(dummy_module, dataset_size, up_group, heat=0.0)

    up_loss_with_prior = UP_AllGaussLoss(dummy_module, dataset_size, up_group, heat=1.0)

    full_samples = torch.stack(list_of_tensors, dim=0)
    print(f"Rank {rank}: all samples mean: {full_samples.mean(dim=0)}")
    print(f"Rank {rank}: all samples std: {full_samples.std(dim=0)}")

    ideal_value_no_prior = (
        full_samples.mean(dim=0) ** 2 / (2 * full_samples.std(dim=0) ** 2)
        + torch.log(full_samples.std(dim=0))
    ).sum() * dataset_size
    ideal_value_with_prior = (
        ideal_value_no_prior + up_loss_with_prior.kl_loss.prior_matching()
    )

    print(f"Rank {rank}: ideal value without prior {ideal_value_no_prior.item()}")
    print(f"Rank {rank}: ideal value with prior {ideal_value_with_prior.item()}")

    local_samples = torch.stack(
        list_of_tensors[
            (rank * (total_samples // world_size)) : (
                (rank + 1) * (total_samples // world_size)
            )
        ],
        dim=0,
    )

    print(f"Rank {rank}: local samples mean: {local_samples.mean(dim=0)}")
    print(f"Rank {rank}: local samples std: {local_samples.std(dim=0)}")

    local_value_no_prior = up_loss_no_prior(local_samples, target)
    local_value_with_prior = up_loss_with_prior(local_samples, target)

    print(f"Rank {rank}: local value without prior {local_value_no_prior.item()}")
    print(f"Rank {rank}: local value with prior {local_value_with_prior.item()}")

    assert torch.all(torch.isclose(local_value_no_prior, ideal_value_no_prior)), (
        "Local value without prior differs from ideal value"
    )
    assert torch.allclose(local_value_with_prior, ideal_value_with_prior), (
        "Local value with prior differs from ideal value"
    )


if __name__ == "__main__":
    main()

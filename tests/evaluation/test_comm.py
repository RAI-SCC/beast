from typing import cast

import pytest
import torch
import torch.distributed as dist

from beast.evaluation import gather_along_dimension


@pytest.mark.dist
def test_gather_along_dimension_fixed_size(device: torch.device) -> None:
    """Test gathering tensors with identical sizes across ranks."""
    # Setup: each rank has a tensor of shape (2, 3)
    # Global shape should be (2 * world_size, 3) if dim=0 or (2, 3 * world_size) if dim=1
    x = torch.randn(2, 3, device=device)

    # Test dim=0
    result_dim0 = gather_along_dimension(
        x, group=dist.group.WORLD, dim=0, variable_size=False
    )
    assert result_dim0.shape == (2 * dist.get_world_size(), 3)

    # Test dim=1
    result_dim1 = gather_along_dimension(
        x, group=dist.group.WORLD, dim=1, variable_size=False
    )
    assert result_dim1.shape == (2, 3 * dist.get_world_size())

    max_size = 10
    x2 = torch.zeros(max_size, max_size, device=device, dtype=torch.float32)
    local_random_index1 = torch.randint(0, max_size, (1,)).item()
    local_random_index2 = torch.randint(0, max_size, (1,)).item()
    x2[cast(int, local_random_index1), cast(int, local_random_index2)] = 1.0

    local_indices = torch.tensor(
        [local_random_index1, local_random_index2], device=device, dtype=torch.int32
    )
    all_indices = [
        torch.zeros(2, device=device, dtype=torch.int32)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_indices, local_indices, group=dist.group.WORLD)

    result2 = gather_along_dimension(
        x2, group=dist.group.WORLD, dim=0, variable_size=False
    )

    target2 = torch.zeros_like(result2)
    for i, indices in enumerate(all_indices):
        target2[
            cast(int, indices[0].item() + i * max_size), cast(int, indices[1].item())
        ] = 1.0

    assert torch.allclose(result2, target2), (
        "Gathered tensor does not match expected result for fixed size case."
    )


@pytest.mark.skip(reason="This crashes the process group.")
@pytest.mark.dist
def test_gather_along_dimension_variable_size(device: torch.device) -> None:
    """Test gathering tensors with different sizes along the gather dimension."""
    rank = dist.get_rank()
    # Each rank has a different size along dim=0: rank 0 has 1, rank 1 has 2, etc.
    size_at_rank = rank + 1
    x = torch.randn(size_at_rank, 3, device=device)

    result = gather_along_dimension(
        x, group=dist.group.WORLD, dim=0, variable_size=True
    )

    # Total size should be sum of sizes: 1 + 2 + ... + world_size
    expected_total_size = sum(range(1, dist.get_world_size() + 1))
    assert result.shape == (expected_total_size, 3)

    if rank < dist.get_world_size() // 2:
        x2 = torch.zeros(size_at_rank, 3, device=device)
    else:
        x2 = torch.ones(size_at_rank, 3, device=device)

    result2 = gather_along_dimension(
        x2, group=dist.group.WORLD, dim=0, variable_size=True
    )

    expected_result2 = torch.cat(
        [
            torch.zeros(i + 1, 3, device=device)
            if i < dist.get_world_size() // 2
            else torch.ones(i + 1, 3, device=device)
            for i in range(dist.get_world_size())
        ],
        dim=0,
    )

    assert torch.allclose(result2, expected_result2), (
        "Gathered tensor does not match expected result."
    )

    x3 = x2.mT.clone()
    result3 = gather_along_dimension(
        x3, group=dist.group.WORLD, dim=1, variable_size=True
    )

    expected_result3 = expected_result2.mT.clone()

    assert torch.allclose(result3, expected_result3), (
        "Gathered tensor does not match expected result for dim=1 case."
    )


def test_gather_along_dimension_no_group(device: torch.device) -> None:
    """Test that the function returns the input as-is when no group is provided."""
    x = torch.randn(2, 3, device=device)
    result = gather_along_dimension(x, group=None, dim=0)
    torch.testing.assert_close(result, x)

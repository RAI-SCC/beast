import pytest
import torch
import torch.distributed as dist

from beast.evaluation.comm import gather_along_dimension


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


def test_gather_along_dimension_no_group(device: torch.device) -> None:
    """Test that the function returns the input as-is when no group is provided."""
    x = torch.randn(2, 3, device=device)
    result = gather_along_dimension(x, group=None, dim=0)
    torch.testing.assert_close(result, x)

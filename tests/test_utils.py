"""Tests for beast utility functions."""

import pytest
import torch

from beast.utils import set_dtype


@pytest.mark.parametrize(
    "name, expected",
    [
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float16", torch.float16),
        ("fp16", torch.float16),
    ],
)
def test_set_dtype(name: str, expected: torch.dtype) -> None:
    """Test that set_dtype maps full and short dtype names to torch dtypes."""
    assert set_dtype(name) == expected


def test_set_dtype_rejects_unknown() -> None:
    """Test that set_dtype raises on an unknown dtype instead of falling back to float32."""
    with pytest.raises(ValueError, match="Unknown dtype"):
        set_dtype("float64")

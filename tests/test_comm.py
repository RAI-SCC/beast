import math

import pytest
import torch.distributed as dist
from conftest import dist_st
from hypothesis import assume, given
from hypothesis import strategies as st

from beast.comm import create_process_mesh_dict, get_process_group, shutdown_comm


class TestProcessorMesh:
    @pytest.mark.dist()
    @given(
        dist_st(
            st.lists(st.integers(min_value=1, max_value=4), min_size=4, max_size=4)
        ),
    )
    def test_process_group_unmatched_mesh_dims(self, mesh_dims: list[int]) -> None:
        """Test that creating a process mesh with incorrect dimensions raises an error."""
        assume(math.prod(mesh_dims) != dist.get_world_size())

        with pytest.raises(
            ValueError, match="Product of mesh dimensions must equal world size."
        ):
            create_process_mesh_dict(mesh_dims)  # Assuming world size is not 32

        shutdown_comm()

    @pytest.mark.dist(4)
    def test_create_process_mesh_incorrect_lenght(self) -> None:
        """Test that creating a process mesh with incorrect dimensions raises an error."""

        with pytest.raises(ValueError, match="mesh_dims must be of length 4"):
            create_process_mesh_dict(mesh_dims=[4])  # Assuming world size is not 32
        with pytest.raises(ValueError, match="mesh_dims must be of length 4"):
            create_process_mesh_dict(
                mesh_dims=[2, 2, 1]
            )  # Assuming world size is not 32
        with pytest.raises(ValueError, match="mesh_dims must be of length 4"):
            create_process_mesh_dict(
                mesh_dims=[
                    1,
                    4,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ]
            )  # Assuming world size is not 32

        shutdown_comm()

    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "mesh_dims",
        [
            [1, 1, 2, 2],
            [2, 2, 1, 1],
            [2, 1, 2, 1],
        ],
    )
    def test_create_process_mesh_dict(self, mesh_dims: list[int]) -> None:
        """Test that the process mesh dictionary is created correctly."""

        process_mesh_dict = create_process_mesh_dict(mesh_dims=mesh_dims)
        assert isinstance(process_mesh_dict, dict)
        expected_keys = {"DP", "SP", "JSpatial", "JChannel", "DTP", "DDP"}
        assert set(process_mesh_dict.keys()) == expected_keys

        get_process_group("DP")
        get_process_group("SP")
        get_process_group("JSpatial")
        get_process_group("JChannel")
        get_process_group("DTP")
        get_process_group("DDP")

        with pytest.raises(ValueError, match="Process group 'InvalidGroup' not found."):
            get_process_group("InvalidGroup")

        shutdown_comm()

        with pytest.raises(ValueError, match="Process group 'DP' not found."):
            get_process_group("DP")

import os
import sys
import warnings
from collections.abc import Callable, Generator
from typing import Any

import mpi4py.MPI as MPI
import pytest
import torch
import torch.distributed as dist
from hypothesis import strategies as st

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        help="Device to run tests on (cpu or cuda)",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Register the custom mark to avoid PytestUnknownMarkWarning
    config.addinivalue_line(
        "markers",
        "dist(n_ranks): Mark test to run only when running parallel tests, and optionaly, only with the specified ranks",
    )
    config.addinivalue_line(
        "markers",
        "gpu: Mark test to run only when a CUDA device is available and selected",
    )

    """Configure rank 0 to be the only printing rank."""
    if mpi_rank != 0:
        config.option.verbose = -1
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def pytest_runtest_setup(item: pytest.Item) -> None:
    # Check if the test has the 'dist' marker or the 'gpu' marker, and skip if the conditions are not met
    gpu_marker = item.get_closest_marker("gpu")
    if gpu_marker and (
        not torch.cuda.is_available()
        or item.config.getoption("--device") not in ["cuda", "gpu"]
    ):
        pytest.skip("Skipping GPU test since CUDA is not available or not selected.")

    skip_marker = item.get_closest_marker("dist")
    if skip_marker:
        ranks = int(skip_marker.args[0]) if len(skip_marker.args) == 1 else None
        if mpi_size <= 1 or (ranks is not None and ranks != mpi_size):
            pytest.skip(
                f"Skipping distributed test, only {mpi_size} ranks available, but {ranks if ranks else '> 1'} required"
            )


@pytest.fixture(scope="session")
def device(request: Any) -> torch.device:
    device = request.config.getoption("--device")
    torch.set_default_device("cpu")
    if device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    else:
        local_id = mpi_rank % torch.cuda.device_count()
        device = torch.device("cuda", local_id)
        return device


@pytest.fixture(autouse=True, scope="session")
def dist_env(device: torch.device) -> Generator[None, None, None]:
    """Initialize the distributed process group once for the entire test session."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
        assert dist.is_nccl_available()
    else:
        assert dist.is_gloo_available()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo",
        rank=mpi_rank,
        world_size=mpi_size,
        device_id=device.index if device.type == "cuda" else None,
    )
    if not dist.is_initialized():
        raise RuntimeError("Process group initialization failed.")
    yield None
    dist.destroy_process_group()


@st.composite
def dist_st(
    draw: Callable[[st.SearchStrategy], Any], strategy: st.SearchStrategy
) -> Any:
    """Decorator to make a strategy MPI-aware."""
    # Get the strategy
    data = draw(strategy)

    # Broadcast the strategy to all ranks
    sync_data = comm.bcast(data, root=0)

    return sync_data

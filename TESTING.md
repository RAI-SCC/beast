# How to test the beast

## Requirenments

- pytest
- hypothesis
- coverage
- mpi4py

## Disk space: set `TMPDIR` before you start

`tests/data/test_dataset.py` and the evaluation tests build real zarr stores as
fixtures. **One full run writes about 8 GB into `TMPDIR`**, and pytest keeps the
last three runs, so repeated runs accumulate to roughly 24 GB.

Many Linux distributions mount `/tmp` as a tmpfs sized at half of RAM. On such a
machine a plain `pytest` fills `/tmp` partway through and reports around ten
failures that all read `OSError: [Errno 28] No space left on device`, most of them
in `tests/data/test_dataset.py`. **These are not real test failures.** Point
`TMPDIR` at disk-backed storage:

```console
λ TMPDIR=/var/tmp pytest
```

With a disk-backed `TMPDIR`, a single-rank CPU run is **155 passed, 73 skipped**
in about 70 seconds. The 73 skips are the GPU and multi-rank tests, which require
`--device cuda` and `mpirun` respectively.

If a test report shows `No space left on device`, check `df -h /tmp` before
investigating anything else, and clear stale fixtures with:

```console
λ rm -rf "${TMPDIR:-/tmp}"/pytest-of-*
```

## Structure

Everything goes in the test folder, and the structure should mirror the source code directory, with filenames called `test_<module>.py`

## QoL

### GPU test

GPU tests can be marked by decorating the method with the `gpu` mark. A fixture with the correct device is also available. See example bellow:

```python
@pytest.mark.gpu
def test_gpu_marker(self, device: torch.device) -> None:
    """Test that the gpu marker correctly selects a CUDA device."""
    assert device.type == "cuda"
```

To activate GPU tests, pass the `--device cuda` option on the command line:

```console
λ pytest --device cuda
```

If no GPUs are available, the test will be skipped. The device fixture can still be used, in case the tests also need to run on CPU.


### Distributed tests

If the tests need to be run on more than 1 rank, use the `dist` marker.

```python
@pytest.mark.dist()
def test_dist_marker_no_ranks(self) -> None:
    assert (
        dist.get_world_size() > 1
    )  # This test should only run if there are more than 1 ranks
```

If a specific number of ranks is needed, add the number of ranks as an argument to the marker.

```python
@pytest.mark.dist(4)
def test_dist_marker_with_ranks(self) -> None:
    """Test that the dist marker with rank specification works."""
    assert dist.is_initialized()
    assert dist.get_world_size() == 4
```

To parallelize the tests, start pytest with mpi.

```console
mpirun -n 4 pytest
```

### Fixtures for distributed tests

Two fixtures in `conftest.py` exist specifically to keep ranks in agreement.

`shared_tmp_path` creates the directory on rank 0 and broadcasts it to everyone.
Use it instead of `tmp_path` whenever more than one rank touches the same files —
with plain `tmp_path`, each rank gets a *different* directory and the test either
fails confusingly or passes for the wrong reason.

```python
@pytest.mark.dist()
def test_checkpoint_roundtrip(self, shared_tmp_path: pathlib.Path) -> None: ...
```

`dist_st` wraps a Hypothesis strategy so that every rank draws the same value.

```python
from tests.conftest import dist_st


@given(shape=dist_st(st.integers(min_value=1, max_value=8)))
def test_something_collective(shape: int) -> None: ...
```

Any Hypothesis test whose generated data feeds a collective **must** use it.
Without it, ranks draw different values, disagree on tensor shapes, and the test
*hangs* inside the collective rather than failing with a useful message.

### Small models

Use the `BEASTLING` configuration in tests:

```python
model = get_model("BEASTLING", device=device)
```

It is small (14M parameters), deterministic, and runs on a 120×240 grid. The
production configurations (`BEAST-JUPITER-*`, `BEAST-LUMI-*`) will not fit on a
single test device.

### Note on the session process group

`conftest.py` initializes a `torch.distributed` process group for the whole test
session, via an autouse session fixture. Even a plain single-rank `pytest` run
therefore has `dist.is_initialized() == True`, and code under test may call
collectives freely. It uses `nccl` on CUDA and `gloo` on CPU, with `MASTER_PORT`
fixed at `12345` — so two test runs on one machine will collide on that port.

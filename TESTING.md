# How to test the beast

## Requirenments

- pytest
- hypothesis
- coverage
- mpi4py

## What a run costs

Measured on CPU, 2026-09-13:

| Invocation | Result | Time | Scratch in `TMPDIR` |
|------------|--------|------|---------------------|
| `pytest` | 155 passed, 81 skipped | ~25 s | 490 MB |
| `mpirun -n 4 pytest --device cpu` | 182 passed, 45 skipped, 9 xfailed | ~46 s | 2.1 GB |

`tests/data/test_dataset.py` builds real zarr stores as fixtures, which is where
the scratch space goes. The 4-rank figure is higher because each rank gets its own
temporary directory.

Pytest retains the last three runs, so scratch accumulates. If a report shows
`OSError: [Errno 28] No space left on device`, that is disk exhaustion rather than
a real failure. Check `df -h /tmp`, then either clear stale fixtures:

```console
λ rm -rf "${TMPDIR:-/tmp}"/pytest-of-*
```

or point `TMPDIR` at disk-backed storage, which is worth doing on any machine that
mounts `/tmp` as a tmpfs:

```console
λ TMPDIR=/var/tmp pytest
```

### Reuse the zarr fixtures

`generate_mock_zarr_data` reseeds numpy per timestep, so a store is fully
determined by `(n_time, n_features, n_lat, n_lon)`. The session-scoped
`cached_mock_zarr` fixture exploits that: it builds each distinct shape once and
hands the same path to every test that asks for it.

```python
def test_something(self, cached_mock_zarr, tmp_path_factory):
    zarr_path = cached_mock_zarr(48, 8, 120, 240)
```

**Use the fixture rather than calling `generate_mock_zarr_data` directly.** A
direct call writes a private copy; before the fixture existed, twenty-two
byte-identical copies accounted for roughly a gigabyte per run. Masks and
normalization files are tiny and still go to each test's own directory, so
nothing shared is ever written to.

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
λ mpirun -n 4 pytest --device cpu
```

Pass `--device cpu` unless the machine has at least as many GPUs as ranks. The
`device` fixture selects CUDA whenever it is available and assigns
`rank % device_count`, so `mpirun -n 4` on a single-GPU machine puts four ranks on
one device and every test errors out with a NCCL `DistBackendError`.

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

Use `tests/small_models.py` for model tests:

```python
from small_models import SMALL_BEAST_OVERRIDES, small_beast_config

# Build the model directly from a full config dict...
config = small_beast_config(bayesian=True)
model = Beast(**config, device=device)

# ...or splat the overrides into the factory.
model = get_model("BEASTLING", **SMALL_BEAST_OVERRIDES, bayesian=True, device=device)
```

Note the import is `from small_models import ...`, not `from tests.small_models`:
`tests/` is the pytest rootdir and is already on `sys.path`.

`small_beast_config` shrinks `BEASTLING` from 14 M parameters to 378 K while
keeping the block structure, attention and patch embedding intact — enough for the
shape, checkpoint and parameter-sharing assertions the tests actually make.

The window sizes are not arbitrary: `Beast._build_model` asserts divisibility
relations between the grid, the spatial group size, the kernel size and both
window extents. The module docstring spells them out. Read it before changing any
of those numbers, or the model will fail to build on some mesh layouts and not
others.

### Note on the session process group

`conftest.py` initializes a `torch.distributed` process group for the whole test
session, via an autouse session fixture. Even a plain single-rank `pytest` run
therefore has `dist.is_initialized() == True`, and code under test may call
collectives freely. It uses `nccl` on CUDA and `gloo` on CPU, with `MASTER_PORT`
fixed at `12345` — so two test runs on one machine will collide on that port.

# Testing

BEAST is tested with `pytest`, `hypothesis` and `mpi4py`, which are part of the
`dev` extra (`pip install -e ".[dev]"`).

## Running the tests

```console
pytest                                # single process, CPU
pytest --device cuda                  # also run the GPU tests
mpirun -n 4 pytest --device cpu       # also run the 4-rank distributed tests
```

Use `--device cpu` with `mpirun` unless the machine has at least as many GPUs as
ranks. Otherwise several ranks share one GPU and the tests fail with a NCCL
`DistBackendError`.

## Writing tests

Tests live in `tests/`, which mirrors the layout of `beast/`. The tests of
`beast/<package>/<module>.py` go into `tests/<package>/test_<module>.py`.

### GPU tests

Mark tests that need a GPU with `gpu`. They are skipped unless pytest runs with
`--device cuda` on a machine with a GPU. The `device` fixture gives the device to
use, so the same test can also run on the CPU.

```python
@pytest.mark.gpu
def test_on_gpu(device: torch.device) -> None:
    assert device.type == "cuda"
```

### Distributed tests

Mark tests that need several processes with `dist`, optionally with the number of
ranks they need. They are skipped unless pytest runs under `mpirun` with enough
ranks.

```python
@pytest.mark.dist(4)
def test_with_four_ranks() -> None:
    assert dist.get_world_size() == 4
```

A process group is always initialized, also for a single-process run, so the code
under test can call collectives. Two fixtures in `tests/conftest.py` keep the
ranks in agreement:

- `shared_tmp_path` is a temporary directory that is the same on all ranks. Use
  it whenever several ranks read or write the same files; `tmp_path` differs
  between ranks.
- `dist_st` wraps a Hypothesis strategy so that all ranks draw the same value.
  Use it for every generated value that reaches a collective, otherwise the
  ranks disagree on shapes and the test hangs.

`tests/` is on the import path, so helpers are imported by module name:

```python
from conftest import dist_st


@given(n=dist_st(st.integers(min_value=1, max_value=8)))
def test_collective(n: int) -> None: ...
```

### Data and models

- Get mock ERA5 stores from the `cached_mock_zarr` fixture in
  `tests/data/test_dataset.py`. It builds each shape once per session and reuses
  it, instead of writing a new store for every test.
- Build models from `tests/small_models.py`. `small_beast_config` shrinks
  `BEASTLING` to about 400 K parameters but keeps its structure. Its grid and
  window sizes satisfy the
  [grid requirements](configuration.md#model-presets) on every mesh the tests
  use, so change them with care.

```python
from small_models import SMALL_BEAST_OVERRIDES

model = get_model("BEASTLING", **SMALL_BEAST_OVERRIDES, device=device)
```

## What a test run costs

A single-process run takes about half a minute on a laptop CPU. The dataset tests
write zarr stores to `TMPDIR`: about 0.5 GB for a single process and about 2 GB for
four ranks, and pytest keeps the last three runs. If tests fail with
`OSError: [Errno 28] No space left on device`, remove old runs or move `TMPDIR` to
a larger disk:

```console
rm -rf "${TMPDIR:-/tmp}"/pytest-of-*
TMPDIR=/var/tmp pytest
```

The session process group uses the fixed port `12345`, so two test runs on the
same machine at the same time interfere with each other.

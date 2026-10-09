# Pitfalls

Things in BEAST that fail quietly, fail confusingly, or fail somewhere other than
where you made the mistake. Grouped by when you are likely to meet them.

The common thread: this is distributed, probabilistic code, so a mistake often
produces *plausible numbers* or a *hang* rather than an exception. Several entries
below are about telling those apart.

---

## Setup

### `import beast.model` fails without the jigsaw submodule

`libs/jigsaw` is a git submodule and a hard import dependency. A plain
`git clone` gives you an empty directory and an `ImportError` that does not
mention submodules.

```console
git submodule update --init --recursive
pip install -e libs/jigsaw/
```

### Two dependencies track moving targets

`pyproject.toml` pins `torch-blue` to a git *branch*
(`@stable-beast`), and `.gitmodules` tracks jigsaw at `branch = main`. Neither is
pinned to a tag or SHA. An unexplained breakage after a clean reinstall — with no
change in this repository — is plausibly one of them having moved.

---

## Running the tests

### `mpirun -n 4 pytest` fails with a NCCL error on a machine with fewer than 4 GPUs

The `device` fixture selects CUDA whenever it is available and assigns
`rank % torch.cuda.device_count()`. With four ranks and one GPU, all four land on
the same device and every test errors with `DistBackendError`. It looks like the
distributed code is broken; it is not.

```console
mpirun -n 4 pytest --device cpu      # forces gloo
```

### `No space left on device` is not a test failure

A single-rank run writes ~490 MB of zarr fixtures to `TMPDIR`; four ranks write
~2.1 GB, and pytest retains the last three runs. On a tmpfs `/tmp` this
accumulates into `OSError: [Errno 28] No space left on device`, reported as ~10
failures concentrated in `tests/data/test_dataset.py`.

Check `df -h /tmp` before investigating anything else.

```console
rm -rf "${TMPDIR:-/tmp}"/pytest-of-*    # clear stale fixtures
TMPDIR=/var/tmp pytest                  # or use disk-backed scratch
```

### A Hypothesis test that feeds a collective will *hang*, not fail

Each rank draws its own value from a strategy. If that value determines a tensor
shape that then enters a collective, the ranks disagree and block forever inside
the collective. There is no error message and no traceback — the suite simply
stops.

Wrap the strategy in `dist_st`, which broadcasts rank 0's draw to everyone:

```python
@given(shape=dist_st(st.integers(min_value=1, max_value=8)))
def test_something_collective(shape: int) -> None: ...
```

### `tmp_path` gives each rank a *different* directory

Use `shared_tmp_path` in any test where more than one rank touches the same
files. It creates the directory on rank 0 and broadcasts it. With plain
`tmp_path`, a checkpoint written by rank 0 is invisible to rank 1, and the test
either fails confusingly or passes for the wrong reason.

### Passing pre-commit does not mean passing CI

The pre-commit mypy hook sets `pass_filenames: true`, so it only type-checks the
files in your commit. CI runs `mypy` over the whole `beast` package. A change that
breaks a type somewhere it did not touch passes locally and fails in CI. Run
`mypy` with no arguments before pushing.

Relatedly, pre-commit runs the **full pytest suite** as a local hook, so commits
take a couple of minutes. That is deliberate.

### The process group is always initialized

`conftest.py` initializes `torch.distributed` for the whole session via an autouse
fixture, so even a single-rank `pytest` has `dist.is_initialized() == True`. Code
under test may call collectives freely — but a single-rank pass proves very little
about sharded behaviour. Run distributed changes under MPI at least once.

`MASTER_PORT` is hard-coded to `12345`, so two test runs on one machine collide.

### Don't leave a process mesh standing

`beast.comm` keeps the mesh in module-level state. A test that builds one leaks it
into every test that follows, making outcomes depend on collection order. The
autouse `_reset_process_mesh` fixture calls `shutdown_comm()` after each test —
don't work around it.

### Build fixtures through the helpers

- `cached_mock_zarr`, not `generate_mock_zarr_data` directly — the fixture shares
  stores between tests, a direct call writes a private copy.
- `small_beast_config()` from `tests/small_models.py`, not a production config;
  the real models will not fit on a test device. Import it as
  `from small_models import ...` — `tests/` is the pytest rootdir and already on
  `sys.path`, so `from tests.small_models import ...` fails.
- The window sizes in `small_models.py` are not arbitrary. `Beast._build_model`
  asserts divisibility relations between the grid, the spatial group size, the
  kernel size and both window extents. Change them and the model fails to build
  on *some* mesh layouts and not others. The module docstring spells them out.

---

## Configuration

### A missing data path silently trains on random noise

This is the most expensive mistake available in this codebase.

`get_dataloader` checks `os.path.exists(config_dict["data"]["data_path"])`. If the
path is missing it emits a `UserWarning` **on rank 0 only** and substitutes
`ERA5Dummy`, which returns random tensors of the correct shape. Training then runs
to completion and produces a model that has learned nothing. On a large job that
one warning is easy to lose.

If a loss curve is flat, grep the rank-0 log for `not found. Running dummy dataset`
before debugging anything else.

### `config_dict` has no schema and no validation

A misspelled key is silently ignored; a missing one surfaces as a `KeyError` deep
inside dataloader construction. [configuration.md](configuration.md) is the only
complete record of the keys the library reads — keep it current when you add one.

Two specific traps:

- `n_variables`, `xlat` and `xlon` are read **only** on the dummy-data path. A
  config that omits them works fine against real data, then raises `KeyError` on
  the day the data path becomes unreachable — i.e. exactly when you least want a
  second failure.
- `num_workers` cannot be `0`. `persistent_workers=True` is hard-coded in
  `get_dataloader`, and PyTorch rejects the combination with
  `ValueError: persistent_workers option needs num_workers > 0`.

### `set_dtype` silently falls back to float32

`beast.utils.set_dtype` has a catch-all `case _: return torch.float32`. A typo
like `"bf16"` instead of `"bfloat16"` costs you mixed precision with no error and
no warning — just an unexplained slowdown and more memory.

### `mode="test"` is not rejected

`get_dataloader` handles `"train"` and `"validation"`. Anything else — including
`"test"` — falls through to `years = None`, which means *the entire time range*
rather than an error.

### Environment variables are parsed strictly and read at import time

`BEAST_DL_DSYNC_INIT` and `BEAST_USE_TE_DPA` are evaluated as
`os.getenv(...) in ["True", "1"]`. Only those two exact strings work: `true`,
`TRUE` and `yes` are all read as false.

They are also read at module import. Setting them from Python after importing
`beast` has no effect; set them in the environment before the process starts.

---

## Distributed execution

### Swapping the DTP and DP ranks trains on the wrong data, without erroring

```python
loader = get_dataloader(config, dtp_rank=..., dtp_size=..., dp_rank=..., dp_size=...)
```

`dtp_rank`/`dtp_size` select which *shard* of each sample this rank loads.
`dp_rank`/`dp_size` select which *samples* it sees. Swap them and the job runs to
completion, at full speed, on the wrong data.

### DDP averages gradients over its group

Whether averaging is the correct reduction depends on which mesh axes the group
spans. This is why `wrap_ddp` lives in `comm.py` next to the mesh definition
rather than in a training script. Change the mesh layout without revisiting it and
you can silently train a differently-weighted model.

`tests/test_grad_reduction.py` asserts gradients are invariant to mesh layout and
is the test most likely to catch a mistake here. Run it under MPI.

### The mesh is exactly four dimensions

`create_process_mesh_dict` takes `(DP, SP, JSpatial, JChannel)` and validates the
arity.

### Building a mesh is a collective call that prints

`create_process_mesh_dict` calls `visualize_process_mesh()` unconditionally, and
that function runs `all_gather_object`. Every rank must reach it or the job
deadlocks. It also prints a rank table from rank 0 on every mesh construction.

### `shutdown_comm` skips the world group on purpose

A mesh dimension spanning the whole world may *be* the default process group, and
`DTP`/`DDP` are flattened views that can alias a dimension group. Destroying the
world group would break every later collective, so `shutdown_comm` skips it and
de-duplicates by object identity. Read the docstring before changing it.

---

## The model and the API

### `Beast.__init__` is keyword-only, and `device` has no default

Every argument comes after `*`. `get_model("BEASTLING")` raises
`TypeError: Beast.__init__() missing 1 required keyword-only argument: 'device'`.

```python
model = get_model("BEASTLING", device=torch.device("cpu"))
```

### `get_model` copies the config shallowly

It calls `.copy()` on the entry from `CONFIGS` before applying your keyword
arguments, so nested mutable values — the `heads` list, for instance — are
**shared** with the module-level `BEAST_CONFIGURATIONS`. Mutating one in place
corrupts the registry for every later call in the process. Pass replacements;
never mutate in place.

### Dropout applies to the deterministic path only

Not the Bayesian one. This was a deliberate fix (commit `dfaaaea`) with a
performance impact; do not "restore" dropout in the Bayesian branch.

### The model is stochastic by design

A Bayesian configuration draws a weight sample per forward pass, so two calls on
identical input give different outputs. That is the mechanism that produces the
forecast ensemble — it is not nondeterminism to be debugged. Use `BEASTLING`
(`bayesian: False`) when you need repeatable output.

### Evaluation metrics are latitude-weighted

Grid cells near the poles cover less area and are down-weighted by
`cos(latitude)`. Comparing `beast.evaluation` metrics against an unweighted
reference implementation shows a systematic discrepancy that is correct behaviour.

### `EWBEra5` is not substitutable for `ERA5Distributed`

It subclasses `ERA5Distributed` to reuse the sharding and normalization setup, but
its `__getitem__` returns `(event, lead_time, inputs)` where the parent returns an
`(input, target)` training pair. It is an evaluation dataset, not a training one.
The override carries a documented `# type: ignore[override]`; do not pass an
`EWBEra5` to code written against the `ERA5Distributed` interface.

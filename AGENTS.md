# AGENTS.md

Guidance for AI coding agents working in the BEAST repository. Humans are welcome
to read it too — it is the fastest orientation to the codebase.

## What this repository is

BEAST (**B**ayesian **E**xascale **A**tmospheric **S**win **T**ransformer) is a
distributed, probabilistic weather-prediction model built on PyTorch. It predicts
future atmospheric states from ERA5 reanalysis data on a 720×1440 lat/lon grid.

Two properties drive almost every design decision in the code, and you should keep
both in mind before changing anything:

1. **The model is Bayesian.** Weights are distributions, not point estimates. The
   variational-inference machinery comes from `torch-blue`, and the loss in
   `beast/losses/up_loss.py` is a KL-divergence objective, not a plain MSE.
2. **The model is sharded across a 4-D process mesh.** A single forward pass spans
   many GPUs. Tensors you see in a module are usually *local shards*, not whole
   tensors. Spatial sharding is provided by the `jigsaw` submodule.

This repository is a **library**, not an application. It exposes models, layers,
losses, datasets and evaluation metrics. There is no training entry point here —
training loops live in the downstream projects that consume this package.

## Layout

```
beast/
  __init__.py         Re-exports get_model
  comm.py             Process-mesh construction, process groups, DDP wrapping
  checkpoint.py       Distributed checkpoint save/load and path parsing
  utils.py            Seeding, dtype parsing, latitude weights
  data/
    era5/dataset.py   ERA5Distributed, ERA5Dummy, EWBEra5, Event (largest data file)
    dataloader.py     BeastDataLoader + get_dataloader factory (config-driven)
    sampler.py        ResumableDistributedSampler (mid-epoch restart)
    transforms.py     Normalization / de-normalization transforms
  layers/
    transformer.py    Transformer blocks, incl. distributed and NICE variants
    mlp.py            MLP blocks
    nice_norm.py      NICE normalization
  losses/up_loss.py   UPAllGaussLoss and Gaussian prior matching
  model/
    model_factory.py  get_model(model_id, **kwargs)
    model_configs.py  CONFIGS registry (currently only BEAST_CONFIGURATIONS)
    beast_model.py    The Beast module
    beast_config.py   Named configurations (BEAST-JUPITER-*, BEAST-LUMI-*, BEASTLING)
  evaluation/         Metrics (mae, mse, rmse, crps, spread_skill_ratio), plotting
libs/jigsaw/          Git submodule: domain-decomposed distributed primitives
tests/                Mirrors beast/ layout
docs/                 Architecture, configuration schema, publishing checklist
```

## Setup

The `jigsaw` submodule is a hard dependency; `import beast.model` fails without it.

```console
git submodule update --init --recursive
python -m venv .venv && source .venv/bin/activate
pip install -e libs/jigsaw/
pip install -e ".[dev]"
pre-commit install
```

## Commands

| Task | Command |
|------|---------|
| Run tests (single rank) | `pytest` |
| Run tests on GPU | `pytest --device cuda` |
| Run distributed tests | `mpirun -n 4 pytest --device cpu` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Type check | `mypy` |
| All checks | `pre-commit run --all-files` |

`pyproject.toml` configures mypy to check the `beast` package only; `tests/` is
excluded. `pre-commit` runs the full pytest suite as
a local hook, so commits are slow — this is intentional, do not disable it.

## Conventions

- **Python ≥ 3.12.** Modern syntax is expected: `X | None` over `Optional[X]`,
  built-in generics, `match` statements where they read well.
- **Type annotations are mandatory** on function definitions in `beast/`
  (`disallow_untyped_defs = true`). `warn_return_any` is on, so do not let `Any`
  leak out of a public function's return type.
- **NumPy-style docstrings**, enforced by ruff's `D` rules with
  `convention = "numpy"`. Every public function needs Parameters/Returns sections.
  Tests are exempt (`"tests/*" = ["D"]`).
- **Line length is not enforced** (`E501` is ignored), but match the surrounding
  code rather than writing very long lines.
- Ruff rule set is `E, F, I, D, D417, N`. Import sorting (`I`) is automatic —
  run `ruff check --fix` rather than reordering imports by hand.

## Testing

### Scratch space

`tests/data/test_dataset.py` builds real zarr stores as fixtures. A single-rank
run writes **about 490 MB** into `TMPDIR`; a 4-rank `mpirun` writes about 2.1 GB,
since each rank gets its own directory. Pytest keeps the last three runs.

That fits a tmpfs `/tmp` on most machines, but pytest retaining three runs plus a
multi-rank run can still fill a small one. If a test report shows
`OSError: [Errno 28] No space left on device`, that is disk exhaustion rather than
a real failure — check `df -h /tmp`, clear stale fixtures with
`rm -rf "${TMPDIR:-/tmp}"/pytest-of-*`, or point `TMPDIR` at disk-backed storage:

```console
TMPDIR=/var/tmp pytest
```

The zarr stores are cached across tests by the session-scoped `cached_mock_zarr`
fixture, keyed on `(n_time, n_features, n_lat, n_lon)`. `generate_mock_zarr_data`
reseeds numpy per timestep, so stores with equal parameters are byte-identical and
sharing them is safe. **Use that fixture rather than calling
`generate_mock_zarr_data` directly** — a direct call reintroduces a private copy.

### Verified numbers

Measured on 2026-09-13 on CPU:

| Invocation | Result | Time | Scratch |
|------------|--------|------|---------|
| `pytest` | 155 passed, 81 skipped | ~25 s | 490 MB |
| `mpirun -n 4 pytest --device cpu` | 182 passed, 45 skipped, 9 xfailed | ~46 s | 2.1 GB |

The skips in the single-rank run are the GPU and multi-rank tests.

Note that `mpirun -n 4 pytest` without `--device cpu` fails with a NCCL error on a
machine with fewer than 4 GPUs: `conftest.py` selects CUDA whenever it is
available and assigns `rank % device_count`. Pass `--device cpu` to force gloo.

### The rest

Read `TESTING.md` for the full story. The essentials:

- Test files mirror the source tree: `beast/layers/mlp.py` → `tests/layer/test_mlp.py`.
- `conftest.py` initializes a **real** `torch.distributed` process group for the whole
  session, via MPI. Even a single-rank `pytest` run has `dist.is_initialized() == True`.
  Code under test may therefore call collectives freely.
- `@pytest.mark.gpu` skips unless CUDA is available *and* `--device cuda` was passed.
- `@pytest.mark.dist` skips unless running under MPI with >1 rank.
  `@pytest.mark.dist(4)` requires exactly 4 ranks.
- The `device` fixture gives the right device and assigns GPUs round-robin by MPI rank.
- `shared_tmp_path` broadcasts one rank-0 tmpdir to all ranks — use it instead of
  `tmp_path` in distributed tests, or ranks will write to different directories.
- `dist_st` wraps a Hypothesis strategy so all ranks draw the *same* value. Any
  Hypothesis test whose data feeds a collective must use it, otherwise ranks
  disagree on shapes and the test hangs instead of failing.
- `_reset_process_mesh` is an autouse fixture that calls `shutdown_comm()` after
  every test. `beast.comm` holds the mesh in module-level state, so without it a
  test that builds a mesh leaks it into every test that follows and outcomes
  depend on collection order. Do not build a mesh and leave it up.
- Use `tests/small_models.py` for model tests: `small_beast_config(**overrides)`
  returns the `BEASTLING` config shrunk to 378 K parameters (from 14 M) while
  keeping the block structure, attention and patch embedding. Its module docstring
  records the divisibility constraints the window sizes must satisfy — read it
  before changing any of those numbers. Import it as `from small_models import ...`
  (`tests/` is the pytest rootdir and already on `sys.path`).
- Use `cached_mock_zarr` rather than calling `generate_mock_zarr_data` directly,
  so stores are shared instead of duplicated per parametrisation.

**When adding a test that touches distributed code, run it under MPI at least once**
(`mpirun -n 4 pytest --device cpu tests/path/test_x.py`). A test that passes on one
rank proves very little about sharded code.

## Working with the process mesh

`beast/comm.py` builds a 4-D mesh with named dimensions and derives combined groups:

| Group | Meaning |
|-------|---------|
| `DP` | Data parallel — distinct batches |
| `SP` | Sample parallel — distinct ensemble members / noise draws |
| `JSpatial` | Jigsaw spatial decomposition — tiles of the lat/lon grid |
| `JChannel` | Jigsaw channel decomposition — slices of the feature dimension |
| `DTP` | Flattened `JSpatial × JChannel` — the ranks sharing one data item |
| `DDP` | Flattened `DP × SP × JSpatial` — the gradient-averaging group |

Two rules that are easy to get wrong:

1. `create_process_mesh_dict` must be called before `get_process_group`. It takes
   **exactly four** dimensions, in the order `(DP, SP, JSpatial, JChannel)`, and
   their product must equal the world size.
2. **DDP averages gradients over its group.** Whether averaging is the correct
   reduction depends on which mesh axes the group spans. This is why `wrap_ddp`
   lives in `comm.py` next to the mesh definition rather than in a training script.
   If you change the mesh layout, re-check `tests/test_grad_reduction.py` — it
   asserts gradients are invariant to mesh layout, and it is the test most likely
   to catch a mistake here.

## Configuration

`get_dataloader` is driven by a nested `config_dict` that is **not** defined by a
schema anywhere in the code. `docs/configuration.md` documents every key the
library reads. If you add a new `config_dict[...]` lookup, update that file in the
same change, otherwise the only record of the key is the line that reads it.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `BEAST_DL_DSYNC_INIT` | `0` | Initialize datasets in synchronized waves — safer at large scale |
| `BEAST_DS_HOURLY_SUBSAMPLING` | `1` | Keep only samples where `hour % factor == 0` |
| `BEAST_USE_TE_DPA` | `0` | Use Transformer Engine `DotProductAttention` instead of flash attention |

The boolean ones are read as `os.getenv(...) in ["True", "1"]`, so `true`,
`yes` and `TRUE` do **not** enable them.

## Gotchas

**[docs/pitfalls.md](docs/pitfalls.md) is the complete list — read it before
debugging anything surprising.** Because this code is distributed and
probabilistic, most mistakes here produce plausible numbers or a hang rather than
an exception, and a good share of "failures" are actually environment problems.

The ones you are most likely to hit:

- **`get_dataloader` silently falls back to random data** when
  `data["data_path"]` does not exist, warning on rank 0 only. Training completes
  having learned nothing. Check for `not found. Running dummy dataset` first.
- **Hypothesis + collectives = a hang, not a failure.** Wrap the strategy in
  `dist_st` so every rank draws the same value.
- **`mpirun -n 4 pytest` needs `--device cpu`** unless the machine has ≥4 GPUs;
  otherwise every test errors with a NCCL `DistBackendError`.
- **`No space left on device` in a test report is disk exhaustion**, not a real
  failure. Check `df -h /tmp`.
- **Passing pre-commit does not mean passing CI.** The pre-commit mypy hook sets
  `pass_filenames`, so it only checks files in your commit; CI checks the whole
  package. Run bare `mypy` before pushing.
- **`get_model` copies the config shallowly.** Nested mutables — the `heads` list —
  are shared with the module-level registry. Pass replacements, never mutate.
- **`Beast.__init__` is keyword-only** and `device` has no default.
- **Dropout applies to the deterministic path only,** not the Bayesian one. A
  deliberate fix (commit `dfaaaea`); do not "restore" it in the Bayesian branch.
- **Swapping `dtp_rank` and `dp_rank`** in `get_dataloader` trains on the wrong
  data at full speed without erroring.
- **`shutdown_comm()` deliberately skips `dist.group.WORLD`**, because a mesh
  dimension spanning the world may *be* the default group and `DTP`/`DDP` are
  flattened views that can alias one. Read its docstring before changing it.
- Metrics in `beast/evaluation/` are latitude-weighted; an unweighted reference
  will disagree systematically, and that is correct.

## Pull requests

- Branch off `main`. Never commit to `main` directly.
- `pre-commit run --all-files` must pass — and separately, bare `mypy`, which CI
  runs over the whole package where the hook does not.
- Public API changes should touch `docs/architecture.md` in the same PR.
- New `config_dict` keys should touch `docs/configuration.md` in the same PR.
- A new way to fail quietly belongs in `docs/pitfalls.md` in the same PR.

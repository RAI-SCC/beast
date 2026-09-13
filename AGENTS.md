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
    dataloader.py     GBDataLoader + get_dataloader factory (config-driven)
    sampler.py        ResumableDistributedSampler (mid-epoch restart)
    transforms.py     Normalization / de-normalization transforms
  layers/
    transformer.py    Transformer blocks, incl. distributed and NICE variants
    expert.py         Expert + ExpertWeightingLayer
    mlp.py            MLP blocks
    nice_norm.py      NICE normalization
  losses/up_loss.py   UPAllGaussLoss and Gaussian prior matching
  model/
    model_factory.py  get_model(model_id, **kwargs)
    model_configs.py  CONFIGS registry (currently only BEAST_CONFIGURATIONS)
    gb/beast_model.py The Beast module — the production model
    gb/beast_config.py Named configurations (BEAST-JUPITER-*, BEAST-LUMI-*, BEASTLING)
    test_models/      Historical/experimental architectures. See caveat below.
  evaluation/         Metrics (mae, mse, rmse, crps, spread_skill_ratio), plotting
libs/jigsaw/          Git submodule: domain-decomposed distributed primitives
tests/                Mirrors beast/ layout
docs/                 Architecture, configuration schema, publishing checklist
```

### `beast/model/test_models/` is not what it sounds like

Despite the name it holds **historical and experimental architectures**
(ViT, Swin variants, a weather mixer), not tests for models. It is excluded from
both mypy and ruff in `pyproject.toml`. Do not hold it to the standards of the
rest of the codebase, and do not refactor it unless asked — but also do not treat
it as a template for new code. Tests for the real model live in `tests/model/`.

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

> **Before your first `pytest`, read the disk-space warning below.** A single run
> writes ~8 GB to `TMPDIR`, which fails outright on a tmpfs-backed `/tmp`.

| Task | Command |
|------|---------|
| Run tests (single rank) | `TMPDIR=/var/tmp pytest` |
| Run tests on GPU | `pytest --device cuda` |
| Run distributed tests | `mpirun -n 4 pytest` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Type check | `mypy` |
| All checks | `pre-commit run --all-files` |

`pyproject.toml` configures mypy to check the `beast` package only; `tests/` and
`beast/model/test_models/` are excluded. `pre-commit` runs the full pytest suite as
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

### The suite needs ~8 GB of scratch space

`tests/data/test_dataset.py` and `tests/evaluation/` build real zarr stores as
fixtures. One full run writes **about 8 GB** into `TMPDIR`, and pytest keeps the
last three runs, so repeated runs accumulate to ~24 GB.

Many Linux distributions mount `/tmp` as a tmpfs sized at half of RAM. On such a
machine a plain `pytest` fills `/tmp` partway through and produces around ten
failures that all read `OSError: [Errno 28] No space left on device` — most of them
in `tests/data/test_dataset.py`. **These are not real test failures.** Point
`TMPDIR` at disk-backed storage and they disappear:

```console
TMPDIR=/var/tmp pytest
```

Verified on 2026-09-13: with a disk-backed `TMPDIR` the suite is **155 passed,
73 skipped, 0 failed** in about 70 s on CPU. The 73 skips are the GPU and
multi-rank tests, which need `--device cuda` and `mpirun` respectively.

If you see `No space left on device` in a test report, check `df -h /tmp` before
investigating anything else, and clear stale fixtures with
`rm -rf "${TMPDIR:-/tmp}"/pytest-of-*`.

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
- Use the `BEASTLING` configuration for tests. The production configurations
  (`BEAST-JUPITER-*`, `BEAST-LUMI-*`) will not fit on a single test device.

**When adding a test that touches distributed code, run it under MPI at least once**
(`mpirun -n 4 pytest tests/path/test_x.py`). A test that passes on one rank proves
very little about sharded code.

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

1. `create_process_mesh_dict` must be called before `get_process_group`, and the
   product of the mesh dimensions must equal the world size.
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

- **`get_model` mutates nothing but copies shallowly.** It `.copy()`s the config
  dict from `CONFIGS` before applying kwargs; nested mutable values are shared.
- **`Beast.__init__` is keyword-only.** All arguments after `*` — positional calls fail.
- **Dropout applies to the deterministic path only,** not the Bayesian one. This was
  a deliberate fix (commit `dfaaaea`); do not "restore" dropout in the Bayesian branch.
- **`get_dataloader` silently falls back to a dummy dataset** of random numbers when
  `data["data_path"]` does not exist. It warns on rank 0 only. If results look like
  noise, check that warning first.
- `shutdown_comm()` clears the group dict but leaves the module-level `_mesh` bound.
- Metrics in `beast/evaluation/` are latitude-weighted. Comparing them against an
  unweighted reference implementation will show a systematic discrepancy that is
  not a bug.

## Pull requests

- Branch off `main`. Never commit to `main` directly.
- `pre-commit run --all-files` must pass before you open a PR.
- Public API changes should touch `docs/architecture.md` in the same PR.
- New `config_dict` keys should touch `docs/configuration.md` in the same PR.

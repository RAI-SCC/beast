# BEAST architecture

This document describes how the pieces of the `beast` package fit together. It is
aimed at someone — human or agent — who needs to make a change and wants to know
which file to open and what will break.

## The short version

BEAST maps a stack of input weather fields to a predicted future atmospheric state:

```
ERA5 fields  ─▶  patch embed  ─▶  encoder  ─▶  bottleneck  ─▶  decoder  ─▶  output fields
(88 channels)    (conv2d)        (transformer blocks, windowed attention)     (84 channels)
```

Two things distinguish it from a standard vision transformer:

1. **It is Bayesian.** Layers carry weight *distributions*. A forward pass draws a
   sample, so two calls with identical inputs give different outputs, and an
   ensemble is produced by sampling repeatedly. The training objective is a
   KL-regularized likelihood, not a point-estimate regression loss.
2. **It is sharded.** A single forward pass is split across a mesh of GPUs along
   both spatial and channel axes. Modules operate on local shards and insert
   collectives where the maths requires a global view.

## Module map

### `beast/model/`

`get_model(model_id, **kwargs)` is the public entry point, re-exported from the
package root. It looks `model_id` up in `CONFIGS` (`model_configs.py`), shallow-copies
the stored config, applies `kwargs` as overrides, and instantiates the class.

`CONFIGS` currently aggregates only `BEAST_CONFIGURATIONS` from `beast_config.py`:

| Model ID | Grid | `embed_dim` | Bayesian | Purpose |
|----------|------|-------------|----------|---------|
| `BEAST-JUPITER-2.6B-64K` | 720×1440 | 3072 | yes | Largest configuration, JUPITER |
| `BEAST-JUPITER-0.7B-1.8K` | 720×1440 | 1024 | yes | Smaller JUPITER run |
| `BEAST-LUMI-1.2B-1.8K` | 720×1440 | 2048 | yes | LUMI run |
| `BEASTLING` | 120×240 | 256 | no | Small, non-Bayesian; development and tests |

`BEASTLING` is the one you want when developing locally — it is deterministic,
uses a coarse grid, and fits on a single device.

`beast_model.py` defines `Beast`. Its `__init__` is **keyword-only** and takes
the architecture parameters plus four optional process groups (`channel_group`,
`spatial_group`, `dtp_group`, `up_group`). Passing `None` for all of them yields a
single-device model, which is how most tests run it.

`model/utils.py` provides `count_model_parameters` and `convert_jigsaw_vi`, which
bridges jigsaw's distributed layers and torch-blue's variational-inference layers.

### `beast/layers/`

- `transformer.py` — `TransformerBlock`, `NiceTransformer`, and the distributed
  variants `DistributedTransformer` / `DistributedNICETransformer`. Attention
  backend is flash attention by default; setting `BEAST_USE_TE_DPA=1` switches to
  Transformer Engine's `DotProductAttention` when jigsaw exposes it.
- `nice_norm.py` — NICE normalization, an alternative to LayerNorm used when a
  configuration sets `nice_norm: true`.
- `mlp.py` — feed-forward blocks.

Windowed attention is controlled by `window_size_outer` and `window_size_inner`
together with `n_full_blocks`, which says how many *full* (non-windowed) attention
blocks appear in each of the three stages.

### `beast/losses/up_loss.py`

`UPAllGaussLoss` extends `torch_blue`'s `AnalyticalKullbackLeiblerLoss`. It combines
a data-fit term with an analytic KL divergence between the variational posterior and
a Gaussian prior. `MeanStdPriorMatchingAggregation` is a custom
`torch.autograd.Function` that aggregates prior-matching statistics across ranks
with a hand-written backward pass — if you touch it, verify gradients under MPI,
because a single-rank run exercises none of the interesting paths.

### `beast/data/`

- `era5/dataset.py` (~800 lines, the largest module) — `ERA5Distributed` reads
  zarr stores and returns the shard belonging to this rank's DTP position.
  `ERA5Dummy` generates random tensors of the same shape. `EWBEra5` targets the
  WeatherBench-style layout, and `Event` describes a named case study.
- `dataloader.py` — `get_dataloader(config_dict, ...)` builds the dataset, applies
  optional subsetting, wraps it in a `ResumableDistributedSampler`, and returns a
  `BeastDataLoader`. `BeastDataLoader` is a plain `DataLoader` whose iterator also yields
  the perf-counter timestamps bracketing each fetch, for I/O profiling.
- `sampler.py` — `ResumableDistributedSampler` slices the shuffled index list at
  `resume_idx`, so a job restarted mid-epoch does not repeat samples.
- `transforms.py` — normalization against precomputed statistics.

**The dummy-data fallback is the single biggest foot-gun in this package.** If
`config_dict["data"]["data_path"]` does not exist, `get_dataloader` warns on rank 0
and silently substitutes `ERA5Dummy`. Training proceeds happily on random noise.

### `beast/comm.py`

Owns the distributed topology. `create_process_mesh_dict(mesh_dims)` builds a
4-D `DeviceMesh` and caches the resulting process groups in a module-level dict,
retrieved by name with `get_process_group`. It checks that exactly four mesh
dimensions are given.

| Group | Axis | Meaning |
|-------|------|---------|
| `DP` | mesh dim 0 | Data parallel — different batches |
| `SP` | mesh dim 1 | Sample parallel — different ensemble members |
| `JSpatial` | mesh dim 2 | Jigsaw spatial tiles of the lat/lon grid |
| `JChannel` | mesh dim 3 | Jigsaw channel slices of the feature dimension |
| `DTP` | `JSpatial × JChannel` | The ranks that jointly hold one data item |
| `DDP` | `DP × SP × JSpatial` | The gradient-averaging group |

`DTP` determines data loading: all ranks in a DTP group receive shards of the
*same* sample, which is why `get_dataloader` takes `dtp_rank`/`dtp_size` separately
from `dp_rank`/`dp_size`.

`DDP` determines gradient reduction. `wrap_ddp` lives here rather than in a
training script precisely because DDP *averages* over its group, and whether
averaging is correct depends on which mesh axes the group spans. Changing the mesh
layout without revisiting this is the most likely way to silently train a wrong
model. `tests/test_grad_reduction.py` guards it by asserting gradients are
invariant to mesh layout.

`execute_in_waves(func, wave_size)` staggers execution across ranks in batches,
used to avoid stampeding a shared filesystem at large scale — the same concern
behind `BEAST_DL_DSYNC_INIT`.

`visualize_process_mesh()` prints a rank-by-group table from rank 0. It is a
collective call: every rank must reach it or the job deadlocks. Note
`create_process_mesh_dict` calls it unconditionally, so building a mesh is itself
a collective operation that prints.

`shutdown_comm()` destroys the groups and drops the mesh. It skips
`dist.group.WORLD` and de-duplicates by object identity, because a dimension
spanning the whole world may *be* the default group, and `DTP`/`DDP` are flattened
views that can alias a dimension group. Destroying the world group would break
every later collective.

### `beast/checkpoint.py`

`save_checkpoint` / `load_checkpoint` handle sharded state. `parse_checkpoint_path`
recovers metadata (step, epoch) from checkpoint
filenames, which is how a restarted job finds where it left off and what to pass as
the sampler's `resume_idx`.

### `beast/evaluation/`

Metrics operate on distributed tensors and are **latitude-weighted** — grid cells
near the poles cover less area and are down-weighted by `cos(latitude)`, computed
in `utils.latitude_weighted_average` and `beast.utils.get_spatial_weights`.
Comparing these against an unweighted reference will show a systematic
discrepancy that is correct behaviour, not a bug.

- `metrics.py` — `mae`, `mse`
- `derived_metrics.py` — `rmse`, `spread_skill_ratio`
- `special_metrics.py` — `crps` (Continuous Ranked Probability Score), the natural
  metric for a probabilistic forecast
- `comm.py` — `gather_along_dimension`, reassembling sharded tensors for scoring
- `utils.py` — variable naming and selection helpers
- `plot.py` — `pred_target_bias_plot`

## External dependencies that shape the design

| Dependency | Role |
|------------|------|
| `jigsaw` (submodule, `libs/jigsaw`) | Domain-decomposed distributed conv, linear, roll, halo exchange |
| `torch-blue` (pinned to `stable-beast`) | Variational-inference layers and KL losses |
| `zarr` / `xarray` / `h5py` | ERA5 storage formats |
| `mpi4py` | Test-suite launcher and rank discovery |

Both `jigsaw` and `torch-blue` are pinned to moving targets — a submodule branch
and a git branch respectively. An unexplained breakage after a clean reinstall is
plausibly one of them moving rather than a change in this repository.

## Where things are *not*

- **No training loop.** This is a library; training scripts live downstream.
- **No config files.** `get_dataloader` takes a `config_dict` the caller supplies.
  See [configuration.md](configuration.md) for the keys it reads.
- **No pretrained weights**, and no data. ERA5 must be obtained separately.

# Configuration reference

`beast.data.get_dataloader` is driven by a nested dictionary (`config_dict`) that
the caller supplies. There is **no schema class and no validation** in the code: a
missing key surfaces as a `KeyError` deep inside dataloader construction, and a
misspelled key is silently ignored.

This file is the only complete record of which keys the library reads. **If you add
a `config_dict[...]` lookup, add it here in the same change.**

The ways this bites in practice are collected in [pitfalls.md](pitfalls.md).

## Sections

The data loading code reads two sections, `data` and `training`. The `Beast` model
is configured through `get_model` keyword arguments instead (see below).

## `data`

Read by `get_dataloader` in `beast/data/dataloader.py`.

| Key | Type | Description |
|-----|------|-------------|
| `data_path` | `str` | Path to the ERA5 zarr store. **If this path does not exist, a dummy random dataset is used instead** — see the warning below. |
| `constant_masks_path` | `str` | Directory holding the constant mask files (orography, land-sea mask, …). |
| `constant_masks` | `list[str]` | Filenames of the masks to load, relative to `constant_masks_path`. |
| `normalization_dir` | `str` | Directory containing `norm_mean.npy` and `norm_std.npy`. |
| `train_time_range` | `tuple[str, str]` | Training period as `("%Y-%m-%d", "%Y-%m-%d")`. Used when `mode="train"`. |
| `valid_time_range` | `tuple[str, str]` | Validation period, same format. Used when `mode="validation"`. |
| `dt` | `int` | Temporal stride in hours between consecutive timesteps. |
| `n_in_timesteps` | `int` | Number of input timesteps, concatenated along the channel dimension. |
| `n_out_timesteps` | `int` | Number of target timesteps, concatenated along the channel dimension. |
| `train_subset` | `int` | Cap the training set to this many samples. `<= 0` means use all. |
| `valid_subset` | `int` | Cap the validation set to this many samples. `<= 0` means use all. |
| `num_workers` | `int` | `DataLoader` worker processes. Must be `> 0`: `persistent_workers=True` is hard-coded, and PyTorch rejects the combination with `ValueError: persistent_workers option needs num_workers > 0`. |
| `prefetch_factor` | `int` | Batches prefetched per worker. |
| `n_variables` | `int` | Channel count. **Only read on the dummy-dataset path**, to shape the random tensors. |
| `xlat` | `int` | Latitude grid points. **Dummy path only.** |
| `xlon` | `int` | Longitude grid points. **Dummy path only.** |

> **The dummy-data fallback is silent by default.**
> `get_dataloader` checks `os.path.exists(data_path)`. If the path is missing it
> emits a `UserWarning` **on rank 0 only** and substitutes `ERA5Dummy`, which
> returns random numbers of the right shape. Training then runs to completion and
> produces a model that has learned nothing. On a large multi-rank job that single
> warning is easy to lose in the log. If your loss curve is flat, grep the rank-0
> log for `not found. Running dummy dataset` before debugging anything else.
>
> Note that `n_variables`, `xlat` and `xlon` are read *only* on this fallback path,
> so a config that omits them works fine against real data and then raises
> `KeyError` the day the data path becomes unreachable.

## `training`

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | `int` | Per-`DataLoader` batch size, i.e. per data-parallel rank. |
| `dtype` | `str` | One of `"float32"`, `"bfloat16"`, `"float16"`. Parsed by `beast.utils.set_dtype`. |

> `set_dtype` has a catch-all `case _: return torch.float32`. An unrecognised or
> misspelled dtype silently yields float32 rather than raising, so a typo such as
> `"bf16"` costs you mixed-precision performance with no error.

## Example

A minimal config that exercises the real code path:

```python
config = {
    "data": {
        "data_path": "/scratch/era5/era5.zarr",
        "constant_masks_path": "/scratch/era5/masks",
        "constant_masks": ["land_sea_mask.npy", "orography.npy"],
        "normalization_dir": "/scratch/era5/norm",
        "train_time_range": ("1979-01-01", "2015-12-31"),
        "valid_time_range": ("2016-01-01", "2017-12-31"),
        "dt": 6,
        "n_in_timesteps": 1,
        "n_out_timesteps": 1,
        "train_subset": 0,
        "valid_subset": 0,
        "num_workers": 4,
        "prefetch_factor": 2,
        # Only consulted if data_path is missing, but cheap insurance:
        "n_variables": 88,
        "xlat": 720,
        "xlon": 1440,
    },
    "training": {
        "batch_size": 1,
        "dtype": "bfloat16",
    },
}
```

Used together with the process mesh:

```python
from beast.comm import create_process_mesh_dict, get_process_group
from beast.data.dataloader import get_dataloader
import torch.distributed as dist

create_process_mesh_dict([dp, sp, jspatial, jchannel])
dtp = get_process_group("DTP")
dp_group = get_process_group("DP")

loader = get_dataloader(
    config,
    dtp_rank=dist.get_rank(dtp),
    dtp_size=dist.get_world_size(dtp),
    dp_rank=dist.get_rank(dp_group),
    dp_size=dist.get_world_size(dp_group),
    mode="train",
)
```

`dtp_rank`/`dtp_size` select which *shard* of each sample this rank loads;
`dp_rank`/`dp_size` select which *samples* it sees. Swapping the two produces a
job that runs without error and trains on the wrong data.

## Unsupported `mode`

`get_dataloader` accepts `mode` values `"train"` and `"validation"`. Any other
value — including `"test"` — falls through to `years = None`, meaning **the entire
time range** rather than an error. `"test"` is documented as not yet supported.

## Model configuration

The production `Beast` model is **not** configured through `config_dict`. Use the
factory:

```python
from beast import get_model

model = get_model("BEASTLING", device=torch.device("cpu"))
```

`get_model` takes a named configuration from `beast/model/gb/beast_config.py` and
applies any keyword arguments as overrides. `device` has no default and must be
supplied. The available IDs are `BEAST-JUPITER-2.6B-64K`,
`BEAST-JUPITER-0.7B-1.8K`, `BEAST-LUMI-1.2B-1.8K` and `BEASTLING`.

> `get_model` shallow-copies the stored config (`.copy()`). Nested mutable values
> such as the `heads` list are **shared** with the module-level `BEAST_CONFIGURATIONS`
> dict. Mutating `model_config["heads"]` in place would corrupt the registry for
> every later call in the process. Pass replacements, never mutate in place.

## Environment variables

These sit outside `config_dict` and are read at import time, so setting them from
Python after importing `beast` has no effect.

| Variable | Default | Read in | Effect |
|----------|---------|---------|--------|
| `BEAST_DL_DSYNC_INIT` | `0` | `data/dataloader.py` | Initialize datasets in synchronized waves. Recommended beyond 512 nodes to avoid stampeding the filesystem. |
| `BEAST_DS_HOURLY_SUBSAMPLING` | `1` | `data/era5/dataset.py` | Keep only samples where `hour % factor == 0`. |
| `BEAST_USE_TE_DPA` | `0` | `layers/transformer.py` | Use Transformer Engine `DotProductAttention` instead of flash attention. Also requires jigsaw to expose TE support. |

`BEAST_DL_DSYNC_INIT` and `BEAST_USE_TE_DPA` are parsed as
`os.getenv(...) in ["True", "1"]`. Only the exact strings `True` and `1` enable
them — `true`, `TRUE` and `yes` are all read as false.

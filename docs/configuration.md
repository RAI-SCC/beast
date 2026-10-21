# Configuration

BEAST is configured in three places: a dictionary for the data loader, keyword
arguments for the model, and a few environment variables.

## Data loader

{func}`~beast.data.dataloader.get_dataloader` reads the sections `data` and
`training` of a nested dictionary. The keys are not validated: a missing key
raises a `KeyError` when the loader is built, and unknown keys are ignored.

### `data`

| Key | Type | Description |
|-----|------|-------------|
| `data_path` | `str` | Path to the zarr store, see [Data format](#data-format). If the path does not exist, {func}`~beast.data.dataloader.get_dataloader` raises a `FileNotFoundError`. Not read by {func}`~beast.data.dataloader.get_dummy_dataloader`. |
| `constant_masks_path` | `str` | Directory with the constant mask files. |
| `constant_masks` | `list[str]` | File names of the constant masks, appended to every input. |
| `normalization_dir` | `str` | Directory with `norm_mean.npy` and `norm_std.npy`. |
| `train_time_range` | `[str, str]` | Start (inclusive) and end (exclusive) of the training data, as `%Y-%m-%d`. Used with `mode="train"`. |
| `valid_time_range` | `[str, str]` | The same for the validation data. Used with `mode="validation"`. |
| `dt` | `int` | Hours between consecutive timesteps. |
| `n_in_timesteps` | `int` | Number of input timesteps, stacked along the channels. |
| `n_out_timesteps` | `int` | Number of target timesteps, stacked along the channels. |
| `train_subset` | `int` | Use only the first this many training samples. `0` or less uses all. |
| `valid_subset` | `int` | The same for the validation samples. |
| `num_workers` | `int` | Worker processes of the data loader. Must be at least 1. |
| `prefetch_factor` | `int` | Batches loaded in advance by each worker. |
| `n_variables`, `xlat`, `xlon` | `int` | Shape of the random data of {func}`~beast.data.dataloader.get_dummy_dataloader`. Default to the ERA5 sizes 84, 721 and 1440. `dt` defaults to 6 there. |

Any other `mode` than `"train"` and `"validation"` uses the whole time range of the
data.

### `training`

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | `int` | Batch size of each data-parallel rank. |
| `dtype` | `str` | `"float32"`, `"bfloat16"` or `"float16"`. Any other value gives `float32`. |

### Example

```python
config_dict = {
    "data": {
        "data_path": "/data/era5/era5.zarr",
        "constant_masks_path": "/data/era5/masks",
        "constant_masks": ["land_sea_mask.npy", "orography.npy"],
        "normalization_dir": "/data/era5/normalization",
        "train_time_range": ["1979-01-01", "2016-01-01"],
        "valid_time_range": ["2016-01-01", "2018-01-01"],
        "dt": 6,
        "n_in_timesteps": 1,
        "n_out_timesteps": 1,
        "train_subset": 0,
        "valid_subset": 0,
        "num_workers": 4,
        "prefetch_factor": 2,
    },
    "training": {
        "batch_size": 1,
        "dtype": "bfloat16",
    },
}
```

## Data format

{class}`~beast.data.era5.ERA5Distributed` reads a zarr store with one array,
`fields`, with the dimensions `time`, `feature`, `latitude` and `longitude`. It
should be chunked along `time`. If the number of latitudes or longitudes is odd,
the last row or column is dropped.

- **Channel order.** The evaluation helpers expect the surface variables first,
  followed by every pressure-level variable at each of its levels.
- **Normalization.** `norm_mean.npy` and `norm_std.npy` hold one value per
  feature, in the order of the `feature` dimension.
- **Constant masks.** Each mask is a `.npy` array of shape `(latitude,
  longitude)`.
- **Land-sea mask for SST.** The loss weights from
  {meth}`~beast.data.era5.ERA5Distributed.get_feature_weights` read
  `sst_mask.npy` from `constant_masks_path` (1 over land) and expect sea surface
  temperature to be variable 4.

## Model

The model is configured with keyword arguments to {func}`~beast.get_model`, not
with the dictionary. `get_model` takes the name of a configuration and accepts
any argument of {class}`~beast.model.beast_model.Beast` as an override:

```python
model = get_model("BEASTLING", device=torch.device("cpu"), nice_norm=True)
```

`device` has no default and must always be given.

### Model presets

| Name | Intended for |
|------|--------------|
| `BEAST-JUPITER-2.6B-64K` | The largest run, on JUPITER |
| `BEAST-JUPITER-0.7B-1.8K` | JUPITER |
| `BEAST-LUMI-1.2B-1.8K` | LUMI |
| `BEASTLING` | Development and tests on one device |

The presets are the configurations of the BEAST runs on JUPITER and LUMI: 88
input channels (84 ERA5 variables and 4 constant masks), 84 output channels, and
a 720 x 1440 grid (120 x 240 for `BEASTLING`). Only `BEASTLING` is deterministic. The full values
are in {mod}`beast.model.beast_config`. For other data, override `height`,
`width`, `in_channels` and `out_channels`.

The grid has to fit the windows. With `stride` equal to `kernel_size`, as in all
presets:

- `window_size_outer` and `window_size_inner` must be even.
- `height` must be divisible by `kernel_size * window_size_outer[0]` and by
  `2 * kernel_size * window_size_inner[0]`.
- `width` must be divisible by `n * kernel_size * window_size_outer[1]` and by
  `n * 2 * kernel_size * window_size_inner[1]`, where `n` is the size of the
  `JSpatial` process group.

## Environment variables

These are read when `beast` is imported, so set them before the import. The
switches accept only `1` and `True`.

| Variable | Default | Effect |
|----------|---------|--------|
| `BEAST_DL_DSYNC_INIT` | `0` | Open the datasets in waves of ranks, to spread out file system access. Recommended beyond 512 nodes. |
| `BEAST_DS_HOURLY_SUBSAMPLING` | `1` | Use only timesteps whose hour is a multiple of this value. Applied only when a time range is set. |
| `BEAST_USE_TE_DPA` | `0` | Use Transformer Engine attention, if it is installed. |

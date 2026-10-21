# Architecture

## The model

{class}`~beast.model.beast_model.Beast` is a Swin-style transformer with a
U-shaped layout:

1. A convolution embeds patches of `kernel_size` x `kernel_size` grid points.
2. The first stage applies transformer blocks within attention windows of
   `window_size_outer`.
3. A convolution halves the resolution and doubles the channels, and the second
   stage works on windows of `window_size_inner`.
4. A transposed convolution restores the resolution. Its output is added to the
   output of the first stage, and the third stage works on windows of
   `window_size_outer` again.
5. A transposed convolution maps the result, together with the patch embedding,
   back to the output variables.

Each stage consists of `n_full_blocks` pairs of blocks. The second block of a
pair shifts the windows by half a window size, so that information can flow
between neighbouring windows. With `nice_norm=True`, the blocks use NICE
normalization: layer norms whose scale and shift are computed from a random
noise vector drawn for every forward pass.

Inputs and outputs can hold several timesteps, stacked along the channels. The
dataset sets how many with `n_in_timesteps` and `n_out_timesteps`, and the model
has to match: `in_channels` is the number of variables times `n_in_timesteps`
plus the number of constant masks, and `out_channels` is the number of
variables times `n_out_timesteps`. The figure shows the setup of the paper, with
the states at {math}`t - \Delta t` and {math}`t` as input and the states at
{math}`t + \Delta t` and {math}`t + 2\Delta t` as output.

```{figure} images/beast_detailed.png
:alt: The layout of the BEAST model, and a transformer block sharded over two GPUs.

Upper part: the layout of BEAST. Lower part: a transformer block sharded over two
GPUs, see [Parallelism](#parallelism). From Figure 2 of the
[BEAST paper](https://arxiv.org/abs/2609.12815).
```

## Bayesian layers

The model is built from ordinary layers. {func}`~beast.model.convert_jigsaw_vi`
makes it Bayesian afterwards: linear and convolution layers become variational
layers from [torch-blue](https://github.com/RAI-SCC/torch_blue), whose weights
are normal distributions. A forward pass with `samples=n` draws `n` sets of
weights and returns `n` predictions, stacked along a new first dimension.

{class}`~beast.losses.up_loss.UPAllGaussLoss` trains such a model. It fits a
normal distribution to the predictions of all ranks in the `SP` process group
and adds the KL divergence between the weight distributions and their prior.

## Parallelism

BEAST splits both the work and the memory over four dimensions at once. The
paper groups them into three levels of parallelism, shown in the figure from
left to right. In the code, each level corresponds to dimensions of the process
mesh that {func}`~beast.comm.create_process_mesh_dict` builds, described in the
[Quickstart](quickstart.md#1-build-the-process-mesh):

| Paper | Mesh groups | What the ranks of a group do |
|-------|-------------|------------------------------|
| Domain–tensor parallelism (DTP) | `JSpatial` x `JChannel`, together `DTP` | Work on one data item together. Each rank holds a tile of the grid (`JSpatial`) and a slice of the channels (`JChannel`). |
| Uncertainty parallelism (UP) | `SP` | Compute different ensemble members for the same input. Each `DTP` instance draws its own weight samples and noise. |
| Data parallelism (DP) | `DP` | Work on different data items. |

```{figure} images/beast_parallel.png
:alt: Domain-tensor, uncertainty and data parallelism, each grouping the level before it.

The parallelism of BEAST. A `DTP` instance splits one data item over several
GPUs, a `UP` instance runs several `DTP` instances that draw different weight
and noise samples for the same input, and a `DP` instance runs several `UP`
instances on different data. From Figure 1 of the
[BEAST paper](https://arxiv.org/abs/2609.12815).
```

Within a `DTP` group, every rank holds part of the same sample:

- **Data.** {class}`~beast.data.era5.ERA5Distributed` loads one of two channel
  halves and a slice of the longitudes on each rank.
- **Model.** If the `DTP` group has more than one rank, `Beast` uses the
  distributed layers of [jigsaw](https://github.com/RAI-SCC/jigsaw) and of
  {mod}`beast.layers`. They work on the local shard and communicate where they
  need values from other ranks, for example at the edges of the shifted windows.
- **Attention.** In each transformer block, distributed linear layers compute
  the queries, keys and values, as in the lower part of the
  [model figure](#the-model). Each rank then holds them for its own grid tile
  and a share of the attention heads, so the attention itself needs no
  communication. It uses the scaled dot-product attention of PyTorch, or
  Transformer Engine if `BEAST_USE_TE_DPA` is set. Another distributed linear
  layer maps the result back to the local channels.
- **Parameters.** Parameters are sharded over the `JChannel` group and
  replicated over the `JSpatial` group. DDP averages their gradients over the
  `DDP` group.

## Package layout

| Module | Contents |
|--------|----------|
| {mod}`beast.model` | `Beast`, its named configurations, `get_model` and `convert_jigsaw_vi` |
| {mod}`beast.layers` | Transformer blocks, MLPs and NICE normalization, single-process and distributed |
| {mod}`beast.losses` | The UP loss for Bayesian models |
| {mod}`beast.data` | ERA5 datasets, the data loader and its sampler, downsampling transforms |
| {mod}`beast.evaluation` | Metrics (MSE, MAE, RMSE, CRPS, spread-skill ratio), latitude weighting, variable selection and plots |
| {mod}`beast.comm` | The process mesh and data parallelism |
| {mod}`beast.checkpoint` | Saving and loading sharded checkpoints |
| {mod}`beast.utils` | Seeding, dtype parsing and latitude weights for the loss |

The evaluation metrics are computed per grid point. Use
{func}`~beast.evaluation.latitude_weighted_average` to average them over the
globe with weights proportional to the area of each grid cell.

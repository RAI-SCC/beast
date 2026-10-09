# Overview

BEAST predicts future atmospheric states from current ones. It reads gridded
weather fields, such as ERA5 reanalysis, and returns the fields one or more
timesteps later.

## What makes it different

**It is probabilistic.** BEAST can be made Bayesian: its weights become
distributions, and every forward pass draws a new sample of them. Running the
same input several times gives an ensemble of forecasts, and the spread of the
ensemble shows how certain the forecast is. The training loss,
{class}`~beast.losses.up_loss.UPAllGaussLoss`, rewards a well-calibrated
ensemble, not only an accurate mean.

**It is built to scale.** One forward pass can be split over many GPUs along four
axes at once: different data items, different ensemble members, tiles of the
grid and slices of the channels. Splitting the grid and the channels is done by
the [jigsaw](https://github.com/RAI-SCC/jigsaw) library, and the Bayesian layers
come from [torch-blue](https://github.com/RAI-SCC/torch_blue).

## What you get

- {class}`~beast.model.beast_model.Beast`, a Swin-style transformer for gridded
  fields, and named configurations of it through {func}`beast.get_model`.
- The layers it is built from, single-process and distributed.
- {class}`~beast.losses.up_loss.UPAllGaussLoss` for training Bayesian models.
- ERA5 datasets and a data loader that shards each sample over the GPUs.
- Evaluation metrics, including the CRPS of an ensemble.
- Helpers to build the process mesh, wrap the model for data parallelism and
  save checkpoints.

BEAST does not include a training loop, trained weights or data.

## Next steps

- [Install BEAST](installation.md).
- Follow the [Quickstart](quickstart.md) for the steps of a training run.
- See [Architecture](architecture.md) for how the model and the process mesh
  work.

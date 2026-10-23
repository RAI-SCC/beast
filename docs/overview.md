# Overview

BEAST predicts future atmospheric states from current ones. It reads gridded
weather fields, such as ERA5 reanalysis, and returns the fields one or more
timesteps later. The model and its parallelization are described in the paper
[4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling](https://arxiv.org/abs/2609.12815).

## What makes it different

**It is probabilistic.** A forecast from BEAST is an ensemble, and the spread of
the ensemble shows how certain the forecast is. BEAST captures two kinds of
uncertainty:

- **Epistemic uncertainty** is the uncertainty of the model itself, which comes
  from limited training data. A Bayesian BEAST learns a distribution for each of
  its weights instead of a single value, and every forward pass draws a new
  sample of the weights.
- **Aleatoric uncertainty** comes from the chaotic nature of the atmosphere.
  With NICE normalization, every forward pass also draws a random noise vector,
  from which the scale and shift of the layer norms are computed.

Running the same input several times therefore gives different forecasts. The
training loss, {class}`~beast.losses.up_loss.UPAllGaussLoss`, rewards an
ensemble that is well calibrated, not only an accurate ensemble mean.

```{figure} images/beast_idea.png
:alt: One input state goes into a network with a sampled weight set and a noise sample, which produces an ensemble of forecasts.

Each ensemble member is a forecast of the network with its own weight sample
{math}`\theta_m`, drawn from the learned weight distribution
{math}`q(\theta \mid D)`, and its own noise sample {math}`\epsilon_m`. The weight
samples capture the epistemic uncertainty and the noise samples the aleatoric
uncertainty.
```

**It is built to scale.** One forward pass can be split over many GPUs along four
axes at once: different data items, different ensemble members, tiles of the
grid and slices of the channels, as described in
[Architecture](architecture.md#parallelism). Splitting the grid and the channels
is done by the [jigsaw](https://github.com/RAI-SCC/jigsaw) library, and the
Bayesian layers come from [torch-blue](https://github.com/RAI-SCC/torch_blue).

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

## Citing BEAST

If you use BEAST in your research, please cite the paper:

```bibtex
@misc{kieckhefen2026beast,
  title         = {4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling},
  author        = {Kieckhefen, Deifilia and Guti{\'e}rrez Hermosillo Muriedas, Juan Pedro and Heyen, Lars Helge and Bode, Mathis and Hakulinen, Iida and Herten, Andreas and John, Chelsea Maria and Kurth, Thorsten and Moisala, Anni and {\"O}zdemir, Asena Karolin and Phipps, Kaleb and Taubert, Oskar and Weyrauch, Arvid and G{\"o}tz, Markus and Debus, Charlotte},
  year          = {2026},
  eprint        = {2609.12815},
  archivePrefix = {arXiv},
  primaryClass  = {physics.ao-ph},
  doi           = {10.48550/arXiv.2609.12815},
  url           = {https://arxiv.org/abs/2609.12815}
}
```

## Next steps

- [Install BEAST](installation.md).
- Follow the [Quickstart](quickstart.md) for the steps of a training run.
- See [Architecture](architecture.md) for how the model and the process mesh
  work.

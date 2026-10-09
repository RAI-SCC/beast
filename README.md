# Beast - Bayesian Exascale Atmospheric Swin Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Description

BEAST is a distributed, probabilistic weather-prediction model built on PyTorch.
It predicts future atmospheric states from ERA5 reanalysis data on a 720×1440
latitude/longitude grid, taking 88 input channels to 84 output channels.

Two properties set it apart from a conventional vision transformer:

- **It is Bayesian.** Layers carry weight *distributions* rather than point
  estimates, using variational inference from
  [torch-blue](https://github.com/RAI-SCC/torch_blue). A forward pass draws a
  sample, so repeated calls on the same input yield a forecast *ensemble* —
  giving calibrated uncertainty rather than a single deterministic trajectory.
- **It is sharded for exascale.** A single forward pass is split across a
  four-dimensional process mesh, along data, ensemble-sample, spatial and channel
  axes. Spatial domain decomposition is provided by the
  [jigsaw](https://github.com/RAI-SCC/jigsaw) library.

This repository is a **library**, not a training application. It provides the
model, layers, losses, datasets and evaluation metrics; training loops live in the
downstream projects that consume it.

Named configurations ship for the JUPITER and LUMI supercomputers, alongside
`BEASTLING` — a small, single-device, deterministic configuration (14M parameters)
intended for development and testing.

### Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Module map, process mesh, how the pieces fit together |
| [docs/configuration.md](docs/configuration.md) | Every configuration key and environment variable |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common errors and how to fix them |
| [TESTING.md](TESTING.md) | Running the test suite, GPU and distributed markers |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and pull-request expectations |
| [AGENTS.md](AGENTS.md) | Orientation for AI coding agents |

## Installation

### From source

1. Clone the repository

```console
git clone https://github.com/RAI-SCC/beast.git
cd beast
```

2. Initialize and update the submodules

`jigsaw` is a hard dependency — `import beast.model` fails without it.

```console
git submodule update --init --recursive
```

3. Create a virtual environment and activate it

```console
python -m venv .venv
source .venv/bin/activate
```

4. Install jigsaw

```console
pip install -e libs/jigsaw/
```

5. Install with pip or your preferred package manager

```console
pip install .
```

For development:

```console
pip install -e ".[dev]"
pre-commit install
```

Python 3.12 or newer is required.

## Usage

### Building a model

```python
import torch
from beast import get_model

model = get_model("BEASTLING", device=torch.device("cpu"))
```

`device` is required and has no default. Available configuration IDs:

| Model ID | Grid | Embed dim | Bayesian | Intended for |
|----------|------|-----------|----------|--------------|
| `BEAST-JUPITER-2.6B-64K` | 720×1440 | 3072 | yes | JUPITER, largest run |
| `BEAST-JUPITER-0.7B-1.8K` | 720×1440 | 1024 | yes | JUPITER |
| `BEAST-LUMI-1.2B-1.8K` | 720×1440 | 2048 | yes | LUMI |
| `BEASTLING` | 120×240 | 256 | no | Development and tests |

Any architecture parameter can be overridden with a keyword argument:

```python
model = get_model("BEASTLING", device=torch.device("cuda"), bayesian=True)
```

### Distributed setup

Build the process mesh before constructing the model or the dataloader. The
product of the mesh dimensions must equal the world size.

```python
import torch.distributed as dist
from beast.comm import create_process_mesh_dict, get_process_group, wrap_ddp

# mesh dims: (DP, SP, JSpatial, JChannel)
create_process_mesh_dict([2, 2, 2, 2])  # 16 ranks

dtp_group = get_process_group("DTP")  # ranks sharing one data item
ddp_group = get_process_group("DDP")  # gradient-averaging group

model = wrap_ddp(model, device_id=local_rank, ddp_group=ddp_group)
```

### Loading data

```python
from beast.data.dataloader import get_dataloader

loader = get_dataloader(
    config_dict,
    dtp_rank=dist.get_rank(dtp_group),
    dtp_size=dist.get_world_size(dtp_group),
    dp_rank=dist.get_rank(dp_group),
    dp_size=dist.get_world_size(dp_group),
    mode="train",
)
```

See [docs/configuration.md](docs/configuration.md) for the full `config_dict`
schema.

> **Note:** if `config_dict["data"]["data_path"]` does not exist, `get_dataloader`
> warns on rank 0 and falls back to a dataset of random numbers. Training then
> completes normally having learned nothing. If a loss curve looks flat, check the
> rank-0 log for `not found. Running dummy dataset` first.

### Evaluation

```python
from beast.evaluation import crps, rmse, spread_skill_ratio
```

All metrics are latitude-weighted by `cos(latitude)`, since grid cells near the
poles cover less area. Comparing them against an unweighted reference
implementation will show a systematic — and correct — difference.

### Configuration

#### ENV VARIABLES

| Variable | Default | Description |
|----------|---------|-------------|
| BEAST_DL_DSYNC_INIT | 0 | If set to 1, initializes the datasets using wave synchronization for safer data access at large scales. If set to 0, initializes the datasets normaly. |
| BEAST_DS_HOURLY_SUBSAMPLING | 1 | The dataset will only consider entries that match `hour % subsampling_factor` for training/evaluating |
| BEAST_USE_TE_DPA | 0 | If set to 1, uses the transformer engine `DotProductAttention` module for the attention layers. Otherwise it will use flash attention. |

The boolean variables are parsed as `os.getenv(...) in ["True", "1"]`, so only the
exact strings `True` and `1` enable them.

## Testing

```console
pytest                             # 155 passed, 81 skipped, ~25 s
pytest --device cuda               # include GPU tests
mpirun -n 4 pytest --device cpu    # include 4-rank distributed tests
```

See [TESTING.md](TESTING.md) for the GPU and distributed test markers, the
fixtures that keep ranks in agreement, and what a run costs in scratch space.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please open an issue before starting
substantial work, and run `pre-commit run --all-files` before opening a PR.

## Acknowledgments

BEAST is developed at the [Scientific Computing Center](https://www.scc.kit.edu/)
of the [Karlsruhe Institute of Technology](https://www.kit.edu/), with
contributions from [CSC – IT Center for Science](https://www.csc.fi/) and
[NVIDIA](https://www.nvidia.com/).

It builds on two companion libraries from the same group:
[jigsaw](https://github.com/RAI-SCC/jigsaw) for distributed domain decomposition
and [torch-blue](https://github.com/RAI-SCC/torch_blue) for variational inference
in PyTorch.

Training and development used compute resources on the JUPITER and LUMI
supercomputers.

<!-- TODO before release: add the specific grant numbers and compute allocation
     IDs required by the funding bodies and by the EuroHPC access agreements. -->

## Cite this

If you use BEAST in your research, please cite it. Machine-readable metadata is in
[CITATION.cff](CITATION.cff); GitHub renders a formatted citation from it via the
"Cite this repository" button.

<!-- TODO before release: replace with the paper citation once published. -->

## License

MIT — see [LICENSE](LICENSE).

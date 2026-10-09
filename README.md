# BEAST - Bayesian Exascale Atmospheric Swin Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

BEAST is a PyTorch library for building and training probabilistic weather
models that run across thousands of GPUs. It provides a Swin-style transformer
for gridded weather fields, its layers and loss, ERA5 datasets and evaluation
metrics.

- **Probabilistic.** BEAST models can be Bayesian: every forward pass draws new
  weights, so repeated passes give an ensemble of forecasts and a measure of
  their uncertainty.
- **Built to scale.** One forward pass can be split over many GPUs along the data,
  the ensemble members, the grid and the channels, using
  [jigsaw](https://github.com/RAI-SCC/jigsaw) and
  [torch-blue](https://github.com/RAI-SCC/torch_blue).

The [documentation](docs/index.md) covers installation, a quickstart for
distributed training, the configuration and the architecture.

## Installation

```console
git clone --recurse-submodules https://github.com/RAI-SCC/beast.git
cd beast
pip install -e libs/jigsaw/
pip install .
```

Python 3.12 or newer is required. See the
[installation guide](docs/installation.md) for development installs.

## Example

Build the small `BEASTLING` configuration and predict the next timestep on the
CPU:

```python
import torch
from beast import get_model

model = get_model("BEASTLING", device=torch.device("cpu"))
x = torch.randn(1, 88, 120, 240)  # (batch, input channels, latitude, longitude)
with torch.no_grad():
    y = model(x)  # (1, 84, 120, 240)
```

The [quickstart](docs/quickstart.md) sets up a Bayesian model on several GPUs,
with its data loader and loss.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set
up a development environment and open a pull request, and the
[testing guide](docs/testing.md) for running the tests. Everyone taking part is
expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

## Acknowledgments

BEAST is developed at the [Scientific Computing Center](https://www.scc.kit.edu/)
of the [Karlsruhe Institute of Technology](https://www.kit.edu/), with
contributions from [CSC – IT Center for Science](https://www.csc.fi/) and
[NVIDIA](https://www.nvidia.com/). Training and development used compute
resources on the JUPITER and LUMI supercomputers.

<!-- TODO before release: add the specific grant numbers and compute allocation
     IDs required by the funding bodies and by the EuroHPC access agreements. -->

## Citation

If you use BEAST in your research, please cite it. The citation metadata is in
[CITATION.cff](CITATION.cff), and GitHub shows a formatted citation through the
"Cite this repository" button.

<!-- TODO before release: replace with the paper citation once published. -->

## License

MIT, see [LICENSE](LICENSE).

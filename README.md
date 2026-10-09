# BEAST - Bayesian Exascale Atmospheric Swin Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.12815-b31b1b.svg)](https://arxiv.org/abs/2609.12815)

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

The model and its parallelization are described in the paper
[4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling](https://arxiv.org/abs/2609.12815).
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
of the [Karlsruhe Institute of Technology](https://www.kit.edu/), together with
[Forschungszentrum Jülich](https://www.fz-juelich.de/),
[CSC – IT Center for Science](https://www.csc.fi/) and
[NVIDIA](https://www.nvidia.com/).

This research is supported by:

- the German Federal Ministry of Research, Technology and Space (BMFTR) under
  the grant 01LK2313A SMARTWEATHER21-SCC-2,
- the Carl Zeiss Foundation under the Breakthroughs 2025 project "WOW – a World
  model of Our World",
- the European High Performance Computing Joint Undertaking (EuroHPC JU), the
  BMFTR, the Ministry of Culture and Science of North Rhine-Westphalia
  (MKW NRW) and the Hessian Ministry of Science and Research, Arts and Culture
  (HMWK) under grant agreement No 101250682 (JUPITER AI Factory, JAIF).

The JUPITER supercomputer was used through the JUPITER Research and Early Access
Program (JUREAP). JUPITER is funded by the EuroHPC Joint Undertaking, the BMFTR
and the Ministry of Culture and Science of the German state of North
Rhine-Westphalia. Computations on the LUMI supercomputer used resources provided
by CSC – IT Center for Science, Finland. LUMI is co-funded by the EuroHPC Joint
Undertaking and the LUMI Consortium.

## Citation

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

The citation metadata is also in [CITATION.cff](CITATION.cff), and GitHub shows
it through the "Cite this repository" button.

## License

MIT, see [LICENSE](LICENSE).

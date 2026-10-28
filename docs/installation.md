# Installation

BEAST needs Python 3.12 or newer and PyTorch. Distributed runs need a GPU build of
PyTorch with NCCL; everything else also runs on the CPU.

## From source

Clone the repository together with the jigsaw library it depends on:

```console
git clone https://github.com/RAI-SCC/beast.git
cd beast
```

Create a virtual environment and install BEAST into it:

```console
python -m venv .venv
source .venv/bin/activate
pip install .
```

## For development

Install BEAST in editable mode with the development tools, and set up the
pre-commit hooks:

```console
pip install -e ".[dev]"
pre-commit install
```

To build this documentation locally:

```console
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

## Optional: Transformer Engine attention

On NVIDIA GPUs, attention can use Transformer Engine instead of PyTorch's
scaled dot-product attention. Install Transformer Engine and set
`BEAST_USE_TE_DPA=1`, see [Configuration](configuration.md#environment-variables).

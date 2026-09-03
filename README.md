# Beast - Bayesian Exascale Atmospheric Swin Transformer

## Description

## Installation

### From source

1. Clone the repository

```console
git clone https://github.com/RAI-SCC/beast.git
```

2. Initialize and update the submodules

```console
git submodule init
git submodule update --remote --recursive
```

3. Create a virtual environment and activate it

```console
python -m venv .venv
source .venv/bin/activate
```

3. Download and install jigsaw

```console
git submodule init
git submodule update --remote --recursive
pip install -e libs/jigsaw/
```

4. Install with pip or your prefered package manager

```console
pip install .
```

For development:

```console
pip install -e .[dev]
```

## Usage

### Configuration

#### ENV VARIABLES

| Variable | Default | Description |
|----------|---------|-------------|
| BEAST_DL_DSYNC_INIT | 0 | If set to 1, initializes the datasets using wave synchronization for safer data access at large scales. If set to 0, initializes the datasets normaly. |
| BEAST_DS_HOURLY_SUBSAMPLING | 1 | The dataset will only consider entries that match `hour % subsampling_factor` for training/evaluating |
| BEAST_USE_TE_DPA | 0 | If set to 1, uses the transformer engine `DotProductAttention` module for the attention layers. Otherwise it will use flash attention. |

## Acknowledgments

## Cite this

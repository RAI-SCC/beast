# GrandBellbeast

## Project Structure

- systems
    - System specific stuff (mostly env variables, convenient for accounts, reservations, special container configurations)
- experiments
    - bash scripts (srun/sbatch, data download, ...)
    - python scripts
- src
    - Python files, model related
- results
    - pkl files, checkpoints
    - anything plotable
    - mlflow
- logs
    - where to dump SLURM output files, in case heavy debugging is needed
- configs
    - model configurations
- utils
    - utility scripts, e.g., for logging, plotting, data loading

## Setup

### Dev tools

**pre-commit**

1) Install pre-commit

   ```pip install pre-commit```

2) Install hooks

   ```pre-commit install```

3) Test on all files

   ```pre-commit run --all-files```

### Update torch_blue and Jigsaw submodules

```console
git submodule update --remote --recursive
```

### Python env

Create a new python environment with v 3.11 or above.

```console
pip install -r requirements.txt
```

### Starting an interactive enroot session on HK cluster

Important to initialize the container, install the dependencies, etc. Needs to be run by anyone running things inside the container.

```console
./utils/start_interactive_enroot_hk.sh <system_env.sh> [additional salloc parameters]
```

## Example commands

### Testing distributed process group

```console
sbatch -A hk-project-pai00058 -p accelerated --nodes 8 --tasks-per-node 4 --gres=gpu:4 experiments/test_dist.sh systems/hk_acc_ml.sh
```

### Running ```train_bnn_weather.py``` with ```configs/weatherMixerTwoway.yaml```

```console
sbatch -A hk-project-pai00058 -p accelerated --nodes 1 --tasks-per-node 2 --gres=gpu:2 experiments/weatherMixer/train_bnn.sh systems/hk_acc_ml.sh configs/weatherMixerTwoway.yaml
```
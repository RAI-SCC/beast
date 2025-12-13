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

```git submodule init```

```console
git submodule update --remote --recursive
```

### Python env

Create a new python environment with v 3.11 or above.

```console
pip install -r requirements.txt
```

### Horeka & FTP

**Get images**

```console
apptainer build $CONTAINER_HOME/pytorch25.10.sif docker://nvcr.io/nvidia/pytorch:25.10-py3
apptainer build $CONTAINER_HOME/pytorch-rocm6.4.4.sif docker://rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
```

**Create a writable (sandbox) container**

```console
apptainer build --sandbox $CONTAINER_HOME/gb_pytorch25.10/  $CONTAINER_HOME/pytorch25.10.sif
```

**Enter the container (to install dependencies)**

```console
apptainer shell --writable $CONTAINER_HOME/gb_pytorch25.10/
```

### LUMI

**Setup PyTorch with Easybuild in LUMI**

1) Load Easybuild modules

```console
ml load LUMI partition/container EasyBuild-user/LUMI
```

2) Build module

```console
eb PyTorch-2.7.1-rocm-6.2.4-python-3.12-singularity-20250827.eb
```

3) Go inside container and install dependencies

```console
ml load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
start-shell
pip install -r requirements.txt
```

4) Done, now you can use the experiment scripts, making sure to load the `systems/lumi_pG_eb_pytorch.sh` system file.

## Example commands

### Testing distributed process group

```console
sbatch -A hk-project-pai00058 -p accelerated --nodes 8 --tasks-per-node 4 --gres=gpu:4 experiments/test_dist.sh systems/hk_acc_ml.sh
```

### Running ```train_bnn_weather.py``` with ```configs/weatherMixerTwoway.yaml```

```console
sbatch -A hk-project-pai00058 -p accelerated --nodes 1 --tasks-per-node 2 --gres=gpu:2 experiments/weatherMixer/train_bnn.sh systems/hk_acc_ml.sh configs/weatherMixerTwoway.yaml
```

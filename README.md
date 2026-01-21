# GrandBellbeast

## Project Structure

- configs
    - model configurations
- data
    - Small example, prototype datasets
- experiments
    - bash scripts (srun/sbatch, data download, ...)
- logs
    - Slurm output logs, debugging logs, training logs
    - Good for debugging, sanity check, etc.
    - Can be large, consider cleaning up periodically
- results
    - pkl files, checkpoints
    - anything plotable
    - mlflow
- src
    - libs
        - Jigsaw
        - torch_blue
    - models
        - model architectures
    - dataloaders
        - data loading scripts
    - python scripts, training scripts, utils
- systems
    - System specific stuff (mostly env variables, convenient for accounts, reservations, special container configurations)
- utils
    - bash utility scripts (Download data, setup environment, print info, etc.)

## Setup

### Dev tools (for your working environment, and code quality)

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

> [!WARNING]
> If you authenticate to github using HTTPs with username and password/token, you might get an authentification error. If that is the case, try this command:
>
> ```console
> git config --local url."https://github.com/".insteadOf "git@github.com:"
> ```
>
> Otherwise, make sure your SSH keys are correctly set up. Or contact @JuanPedroGHM for help.


### Horeka & FTP

1) **Get images**

```console
ml purge
export CONTAINER_HOME=$HOME/containers
mkdir $CONTAINER_HOME
apptainer build $CONTAINER_HOME/pytorch25.06.sif docker://nvcr.io/nvidia/pytorch:25.06-py3
```

2) **Create a writable (sandbox) container**

```console
apptainer build --sandbox $CONTAINER_HOME/gb_pytorch25.06/  $CONTAINER_HOME/pytorch25.06.sif
```

3) **Install dependencies in the container**

```console
apptainer exec --writable --no-home --bind $(pwd):/mnt --cwd /mnt $CONTAINER_HOME/gb_pytorch25.06/ pip install --upgrade pip
apptainer exec --writable --no-home --bind $(pwd):/mnt --cwd /mnt $CONTAINER_HOME/gb_pytorch25.06/ pip install -r requirements.txt
```

**All-in-one**

```console
ml purge
export CONTAINER_HOME=$HOME/containers
mkdir $CONTAINER_HOME
apptainer build $CONTAINER_HOME/pytorch25.06.sif docker://nvcr.io/nvidia/pytorch:25.06-py3
apptainer build --sandbox $CONTAINER_HOME/gb_pytorch25.06/  $CONTAINER_HOME/pytorch25.06.sif
apptainer exec --writable --no-home --bind $(pwd):/mnt --cwd /mnt $CONTAINER_HOME/gb_pytorch25.06/ pip install --upgrade pip
apptainer exec --writable --no-home --bind $(pwd):/mnt --cwd /mnt $CONTAINER_HOME/gb_pytorch25.06/ pip install -r requirements.txt
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

3) Go inside container and install dependencies (only squash once we have the final setup)

```console
ml load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
start-shell
pip update --upgrade pip
pip install --force-reinstall aiter
pip install -r requirements.txt
exit
make-squashfs
rm -rf $CONTAINERROO/user-software
```

4) Done, now you can use the experiment scripts, making sure to load the `systems/lumi_pG_eb_pytorch.sh` system file.

## Example commands

### Testing distributed process group

```console
sbatch -A hk-project-pai00058 -p accelerated --nodes 8 --tasks-per-node 4 --gres=gpu:4 experiments/test_dist.sh systems/hk_acc_ml.sh
```

### Testing Bayesian domain-tensor-parallel (DTP) ViT model on HoreKa
For now, without data parallelism (DDP), so only 4 GPUs in a node:

```console
sbatch -A hk-project-pai00071 -p accelerated --nodes 1 --tasks-per-node 4 --gres=gpu:4 experiments/test_ViT.sh systems/hk_container.sh
```

## Wiki

### Distributed Process Groups in GrandBellbeast

Parallel Process Groups keys:
 - JCol: Jigsaw Column Tensor Parallel group (part of DTP)
 - JRow: Jigsaw Row Tensor Parallel group (part of DTP)
 - DTP: Domain Tensor Parallel group, equal to Column x Row
 - SP: Sample Parallel Group
 - DP: Data Parallel Group
 - DDP: Distributed Data Parallel Group (DP x SP), to be pased to the DDP module
 - Expert: Top group for Mixture of Experts

Mesh Dimensions: [Expert, DP, SP, JCol, JRow]

See `src/util.py` for implementation details.

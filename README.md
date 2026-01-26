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

Base image: ```/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif```


0) Source ```systems/lumi_singularity.sh```

1) Create virtual env (shared env in project dir, only run when resetting the env)

```console
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
singularity exec $SIF bash -c '$WITH_CONDA && python -m venv $VENV_DIR --system-site-packages'
singularity exec $SIF bash -c '$WITH_CONDA && source $VENV_DIR/bin/activate && pip install -r requirenments.txt'
```

## Example commands

### Testing distributed process group

**Don't forget to adjust the mesh inside the ```src/test_process_mesh.py``` to match the number of nodes and gpus per node.**

```console
sbatch -A hk-project-pai00058 -p accelerated --nodes 8 --tasks-per-node 4 --gres=gpu:4 experiments/test_dist.sh systems/hk_acc_ml.sh
```

On lumi:
```console
sbatch -A project_462001248 -p standard-g --nodes 4 --tasks-per-node 4 --gres=gpu:4 experiments/test_dist.sh systems/lumi_singularity.sh
```

### Testing Bayesian domain-tensor-parallel (DTP) ViT model on HoreKa
For now, without data parallelism (DDP), so only 4 GPUs in a node:

```console
sbatch -A hk-project-pai00071 -p accelerated --nodes 1 --tasks-per-node 4 --gres=gpu:4 experiments/test_dtp_ViT.sh systems/hk_container.sh
```

### Testing Bayesian SWIN Transformer model (Kangu) on HoreKa
For now only includes data parallelism (DDP):

```console
sbatch -A hk-project-pai00071 -p accelerated --nodes 2 --tasks-per-node 4 --gres=gpu:4 experiments/test_dp_SWIN.sh systems/hk_container.sh
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

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

```console
git submodule init
git submodule update --remote --recursive
```

> [!WARNING]
> If you authenticate to github using HTTPs with username and password/token, you might get an authentification error. If that is the case, try this commands:
>
> ```console
> sed -i -e "s/git@github.com:/https:\/\/github.com\//g" .gitmodules
> git submodule deinit --all
> git submodule init
> git submodule update --remote --recursive
> git restore .gitmodules
> ```
>
> Otherwise, make sure your SSH keys are correctly set up. Or contact @JuanPedroGHM for help.


### System setup


1) Source system image

```console
source systems/<system_file>.sh
```

2) Ensure the container image is there. If not, then download it (not for lumi, for now):

```console
apptainer pull $SIF docker://nvcr.io/nvidia/pytorch:25.06-py3
```

3) Run init script

```console
init_system
```

4) Done

#### Troubleshooting

 * Make sure the container image exists
 * Make sure the virtual environemnt exists (if not, run the init_system function)
 * Library pointing to the wrong path (should either be a folder inside the container, the virtual environment, or the code repository for editable libraries), either restart the environment (`rm -rf $VENV_DIR`, and run `init_system` again), or install the library manually (`apptainer exec $SIF systems/venv_entry.sh pip installl <library>`)
 * Anything else, you know where my office is.

## Example commands

### Testing distributed process group

**Don't forget to adjust the mesh inside the ```src/test_process_mesh.py``` to match the number of nodes and gpus per node.**

```console
sbatch -A hk-project-pai00058 -p accelerated --nodes 4 --tasks-per-node 4 --gres=gpu:4 --job-name=test_pg experiments/run_gb_script.sh systems/hk_container.sh src/test_process_mesh.py
```

On lumi:
```console
sbatch -A project_462001248 -p standard-g --nodes 4 --tasks-per-node 4 --gres=gpu:4 --job-name=test_pg experiments/run_gb_script.sh systems/lumi_singularity.sh src/test_process_mesh.py
```

On jupiter:
```console
sbatch -A jureap1 -p booster --nodes 4 --tasks-per-node 4 --gres=gpu:4 --job-name=test_pg experiments/run_gb_script.sh systems/jupiter_container.sh src/test_process_mesh.py
```

### Testing Bayesian domain-tensor-parallel (DTP) ViT model on HoreKa
For now, without data parallelism (DDP), so only 4 GPUs in a node:

```console
sbatch -A hk-project-pai00071 -p accelerated --nodes 1 --tasks-per-node 4 --gres=gpu:4 experiments/run_gb_script.sh systems/hk_container.sh src/test_dtp_ViT.py --config_file configs/ViTFourway.yaml
```

### Performance monitoring with Perun

```console
sbatch -A jureap1 -p booster --nodes 2 --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=exact_up_dp_SWIN experiments/run_gb_script.sh --perun systems/jupiter_container.sh src/test_exact_up_dp_SWIN.py --config_file configs/up_dp_SWIN_config.yaml
```
### Current sota model with dtp, sp and dp + torch_blue on SWIN, 0.25 degree
```console
sbatch -A jureap1 -p booster --nodes 4 --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=dtp_sp_dp_SWIN experiments/run_gb_script.sh systems/jupiter_container.sh src/test_dtp_dp_SWIN.py --config_file configs/dtp_sp_dp_SWIN_config.yaml
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

### Model Library

For the list and descriptions of currently implemented models, see the [model documentation](src/gb/model/README.md)

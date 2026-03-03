# GrandBellbeast

## Project Structure

- ``configs/`` : model training configurations in yaml
    - ``bellbeast_config.yaml`` : **main model configuration**
    - ``tests/`` : old configs or those for test runs
- ``data/`` : Small example, prototype datasets
    - ``constant_masks/`` : constant masks for 1.5degree data
    - ``normalization/`` : variable normalization values
        - ``norms.nc`` : normalization values for Bellbeast dataset with 84 variables
- ``experiments/`` : bash scripts (srun/sbatch, data download, ...)
    - ``run_gb_script.sh`` : your main bash script to launch jobs (system independent)
- ``logs/``: Slurm output logs, debugging logs, training logs, timings; Good for debugging, sanity check, etc.; Can be large, consider cleaning up periodically
- ``results/`` : pkl files, checkpoints, anything plotable; mlflow
- ``src/`` : main source code
    - ``data/`` : scripts regarding data download (1.5 degree) and conversion, nomalization calculation, etc.
    - ``libs/`` : external libraries
        - jigsaw
        - torch_blue
    - ``test/`` : test training scripts for sub-models and developments
    - ``gb/:`` Bellbeast codes
        - ``dataloaders/`` : all code regarding data IO
            - ``dataset_0_25_degree.py`` : torch.Dataset for high-resolution data
            - ``dataset_1_5_degree.py`` : torch.Dataset for low-resolution data
            - ``dataloader_0_25_degree.py`` : custom GB torch.dataloader and utility function
        - ``losses/`` : lossfunctions
            - ``up_loss.py:`` Bellbeast loss function (uncertainty parallel and experts included)
        - ``models/`` : torch.nn.modeule definictions
            - ``test_models/`` : test models of various stages and sub-developments
            - ``expert.py`` : model wrapper for Ensemble of Experts
            - ``model_gb.py`` : **The torch.nn Bellbeast model definition**
        - ``utils.py`` : utility functions for process mesh and other stuff
    - ``evaluation.py`` : evaluation loop
    - ``train_bellbeast.py`` : **training script to train the Bellbeast (model_gb)**
- ``systems/`` : System specific container setups (mostly env variables, convenient for accounts, reservations, special container configurations)
- ``utils/`` : bash utility scripts (Download data, setup environment, print info, etc.)

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

## Training the Bellbeast
Current model is ``Bellbeast`` in ``model_gb.py`` and includes :
 * 4-way domain- and tensor-parallelism (DTP) (via Jigsaw); 8-way (for LUMI) is wip
 * Uncertainty parallelism (UP), i.e. sample parallelsim (for variational inference) and noise injection conditioning for ensembles (NICE, for diffusion)
 * Data parallelism (DP) via torch DDP
 * Up to 4 Experts (EP) with different loss weightings

 * Training on hourly subsampled 0.25 degree ERA5 data (chunked stored) with 84 variables and 3 constant masks
     * Either one or two timesteps can serve as input (concatenated along variable dimension) to predict the next (6h) timestep
     * Normalization values are in ``data/normalization/norms.nc``

To run training, configure in ``configs/bellbeast_config.yaml``
* Adjust the process mesh (``mesh_dims``) to match your number of GPUs/Nodes
    * The last two dims are "fixed" to ``[2,2]`` = 4 GPUs on Jupiter and ``[4,2]`` GPUs on LUMI
    * Distribute your nodes to the first three dims, such that their product equals the job node count ``<N_NODES>``; Order is ``[num_experts, num_data_parallel, num_sample_parallel]``
* Adjust other parameters in the training configuration, e.g.
    * Where the training data and constant masks lie (originating from ``DATA_DIR="/e/scratch/jureap1/GrandBellbeast/data/"`` in ``systems/<system_file.sh>``
    * Which constant masks to use (list of file names)
    * How much data to use (date format in ``train_time_range``)
    * How many timesteps to use as input (``n_in_timesteps``)
    * How big the model should be: **``embedding_dimension``**, ``n_attn_heads``, ``patch_size``, etc.
    * How many epochs to train (``n_epochs``)
    * How many Bayesian samples per SP process (``n_samples``)

### On Jupiter
```console
sbatch -A jureap1 -p booster --nodes <N_NODES> --tasks-per-node 4 --gres=gpu:4 -t 00:06:00 --job-name=train_bellbeast experiments/run_gb_script.sh systems/jupiter_container.sh src/train_bellbeast.py --config_file configs/bellbeast_config.yaml
```

### On LUMI: (8 tasks per node is work in progress)
```console
sbatch -A project_462001248 -p standard-g --nodes <N_NODES> --tasks-per-node 4 --gres=gpu:4 --job-name=train_bellbeast experiments/run_gb_script.sh systems/lumi_singularity.sh src/train_bellbeast.py --config_file configs/bellbeast_config.yaml
```
Go grab a cup of coffe and stare at loss values


## Other Example commands

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

### Performance monitoring with Perun

```console
sbatch -A jureap1 -p booster --nodes 2 --tasks-per-node 4 --gres=gpu:4 -t 30:00 --job-name=exact_up_dp_SWIN experiments/run_gb_script.sh --perun systems/jupiter_container.sh src/test_exact_up_dp_SWIN.py --config_file configs/up_dp_SWIN_config.yaml
```


## Wiki

### Distributed Process Groups in GrandBellbeast

Parallel Process Groups keys:
 - JChannel: Jigsaw Row Tensor Parallel group (part of DTP)
 - JSpatial: Jigsaw Column Tensor Parallel group (part of DTP)
 - DTP: Domain Tensor Parallel group, equal to Column x Row
 - SP: Sample Parallel Group
 - DP: Data Parallel Group
 - DDP: Distributed Data Parallel Group (DP x SP), to be pased to the DDP module
 - Expert: Top group for Mixture of Experts

Mesh Dimensions: [Expert, DP, SP, JSpatial, JChannel]

See `src/util.py` for implementation details.

### Model Library

For the list and descriptions of currently implemented models, see the [model documentation](src/gb/model/README.md)

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
    - mlflow?
 - logs
    - where to dump SLURM output files, in case heavy debugging is needed
 - configs
    - model configurations
 - data
    - data

## Setting up environment
Create a new python environment with v 3.11 or above. 
```
cd /path/to/Jigsaw
git checkout vi
pip install -e .
cd /path/to/torch-blue
git checkout vi_convert
pip install -e .
```

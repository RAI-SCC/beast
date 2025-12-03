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

## Setup

### Dev tools

**pre-commit**

1) Install pre-commit

   ```pip install pre-commit```

2) Install hooks

   ```pre-commit install```

3) Test on all files

   ```pre-commit run --all-files```

### Python env

Create a new python environment with v 3.11 or above.
```
pip install -e src/lib/torch_blue
pip install -e src/lib/Jigsaw
```

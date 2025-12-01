# GrandBellbeast

## Project Structure

 - system
    - System specific stuff (mostly env variables, convenient for accounts, reservations, special container configurations)
 - experiments
    - bash scripts (srun/sbatch, data download, ...)
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

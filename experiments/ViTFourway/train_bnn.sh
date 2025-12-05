#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --account='hk-project-pai00071'
#SBATCH --output='ViT_weather.out'
#SBATCH --job-name='ViT_bnn'
#SBATCH --constraint=LSDF

home_directory='../..'

source $home_directory/.venv/bin/activate
which python

export MASTER_PORT=12342
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun python -u $home_directory/src/train_bnn_weather.py --config_file config.yaml

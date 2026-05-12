#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name='gb_data_conversion'
#SBATCH --output='data_conversion_final.out'
#SBATCH --constraint=LSDF

source /home/hk-project-scc/ih5525/venv/GrandBellbeast/bin/activate

python -u calculate_normalization.py --data_path "/lsdf/kit/scc/projects/bellbeast/era5_preprocessed.zarr" --output_path "/home/hk-project-scc/ih5525/GrandBellbeast/data/norms.nc" --checkpoint_path "/home/hk-project-scc/ih5525/GrandBellbeast/data/checkpoint1"

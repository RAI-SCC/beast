#!/bin/bash
#SBATCH --job-name=Ablation_Train
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=hk-project-p0026382
#SBATCH --exclusive
#SBATCH --output='logs/%x_%j.out'

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12355
echo "MASTER_ADDR="$MASTER_ADDR

source /hkfs/work/workspace/scratch/bl7340-beast_ablation/beast/.venv/bin/activate

srun python -u /hkfs/work/workspace/scratch/bl7340-beast_ablation/beast/ablation_train.py



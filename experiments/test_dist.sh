#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --job-name='gb_test_dist'
#SBATCH --output='logs/%x_%j.out'

cd /workspace

source utils/utils.sh
print_slurm_env

echo "$0 $@"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <system_env>"
    echo "  system_env: Path to the system environment file"
    exit 1
fi

SYSTEM_ENV_FILE="$(basename $1 .env)"

echo "System Env File: $SYSTEM_ENV_FILE"

echo "Sourcing system environment..."
source $1

setup_system

export MASTER_PORT=12342
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "Running cmd:"
echo "  srun --cpu-bind=v ${SYSTEM_SLURM_PARAMS[@]} python -u $WORKDIR/src/test_process_mesh.py" "$WORKDIR/configs/mesh_dims.yaml"
echo ""

srun --cpu-bind=v ${SYSTEM_SLURM_PARAMS[@]} python -u $WORKDIR/src/test_process_mesh.py

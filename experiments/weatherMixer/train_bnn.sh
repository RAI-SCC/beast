#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --constraint=LSDF
#SBATCH --job-name='gb_bnn_weather'
#SBATCH --output='logs/%x_%j.out'

source utils/utils.sh
print_slurm_env

echo "$0 $@"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <system_env> <config_file>"
    echo "  system_env: Path to the system environment file"
    echo "  config_file: Path to the configuration file for training"
    echo "  Optional arguments can be added after the config_file"
    exit 1
fi

SYSTEM_ENV_FILE="$(basename $1 .env)"
CONFIG_ID="$(basename $2 .yaml)"

echo "System Env File: $SYSTEM_ENV_FILE"
echo "Config ID:       $CONFIG_ID"

echo "Sourcing system environment..."
source $1

setup_system

export MASTER_PORT=12342
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun --cpu-bind=v python -u $WORKDIR/src/train_bnn_weather.py --config_file $2 "${@:3}"

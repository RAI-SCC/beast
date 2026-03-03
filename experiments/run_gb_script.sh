#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=30:00
#SBATCH --job-name='gb_run'
#SBATCH --output='logs/%x_%j.out'

echo "$0 $@"

perun=false
function print_help() {
    cat <<EOF
Usage: $0 [OPTIONS] <system_env> <python_script> [ARGS...]

Arguments:
  system_env       Path to system environment file (.env)
  python_script    Path to Python script or config file (.yaml)
  ARGS             Additional arguments passed to the Python script

Options:
  --perun          Enable Perun monitoring
  -h, --help       Display this help message

EOF
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_help
    exit 0
fi
while [[ $# -gt 0 ]]; do
    case $1 in
        --perun)
            perun=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "Error: Missing required arguments (system_env and python script)" >&2
    exit 1
fi

source utils/utils.sh
print_slurm_env

system_env=$1
python_script=$2
shift 2
remaining_args=("$@")

SYSTEM_ENV_FILE="$(basename $system_env .sh)"
SCRIPT_FILE="$(basename $python_script .py)"

echo "System Env File: $SYSTEM_ENV_FILE"
echo "Script File:     $SCRIPT_FILE"
echo "Sourcing system environment..."
source $system_env

setup_system

export MASTER_PORT=12342
master_addr=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

if [[ $perun == true ]]; then
    PYTHON_CMD="perun --log_file=logs/perun_${SCRIPT_FILE}_${SYSTEM_ENV_FILE}_${SLURM_JOB_ID}.log monitor"
else
    PYTHON_CMD="python -u"
fi

echo "Running cmd:"
echo "  srun ${SYSTEM_SLURM_PARAMS[@]} $PYTHON_CMD $WORKDIR/$python_script ${remaining_args[@]}"
echo ""

srun ${SYSTEM_SLURM_PARAMS[@]} $PYTHON_CMD $WORKDIR/$python_script "${remaining_args[@]}"

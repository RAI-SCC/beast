#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name='gb_run'
#SBATCH --output='logs/%x_%j.out'
#SBATCH --error='logs/%x_%j.err'

echo "$0 $@"

perun=false
profsys=false
function print_help() {
    cat <<EOF
Usage: $0 [OPTIONS] <system_env> <python_script> [ARGS...]

Arguments:
  system_env       Path to system environment file (.env)
  python_script    Path to Python script or config file (.yaml)
  ARGS             Additional arguments passed to the Python script

Options:
  --perun          Enable Perun monitoring
  --profsys        Enable Nsight Systems profiling/rocprof-systems
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
        --profsys)
            profsys=true
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
# Validate that only one profiling tool is enabled
if [[ ($profsys == true && $perun == true) ]]; then
    echo "Error: Cannot enable multiple profiling tools at the same time" >&2
    exit 1
fi

if [[ $perun == true ]]; then
    PYTHON_CMD="perun --log_file=logs/perun_${SCRIPT_FILE}_${SYSTEM_ENV_FILE}_${SLURM_JOB_ID}.log monitor"
else
    PYTHON_CMD="python -u"
fi

export PROFSYS=$profsys

# 1.) Copy image file to dev/shm on each node
SHM_DIR=/dev/shm/${SLURM_JOB_ID}
export IMAGE_SHM=${SHM_DIR}/${CONTAINER_NAME}
export IMAGE_REGISTRY=$SIF
srun -n $SLURM_NNODES --ntasks-per-node 1 bash -c 'mkdir -p $(dirname ${IMAGE_SHM}) && echo "[${SLURM_PROCID}|${SLURM_NTASKS}] copying ${IMAGE_REGISTRY} to ${IMAGE_SHM}" && cp ${IMAGE_REGISTRY} ${IMAGE_SHM}'

# 2.) Set apptainer temporary directories and pycache to dev shm as well
export APPTAINER_TMPDIR=${SHM_DIR}/apptainer/tmp/
export APPTAINER_CACHEDIR=${SHM_DIR}/apptainer/cache/
export PYTHONPYCACHEPREFIX=${SHM_DIR}/pycache/
# Create directories
srun -n $SLURM_NNODES --ntasks-per-node 1 bash -c 'mkdir -p ${APPTAINER_TMPDIR} ${PYTHONPYCACHEPREFIX} ${APPTAINER_CACHE_DIR}'

# Run acutal job
echo "Running cmd:"
echo "  srun ${SYSTEM_SLURM_PARAMS[@]} $PYTHON_CMD $WORKDIR/$python_script ${remaining_args[@]}"
echo ""


srun ${SYSTEM_SLURM_PARAMS[@]} ${IMAGE_SHM} systems/venv_entry.sh $PYTHON_CMD $WORKDIR/$python_script "${remaining_args[@]}"

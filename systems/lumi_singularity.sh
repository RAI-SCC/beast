#!/usr/bin/env bash
export PROJECT_DIR=/projappl/project_462001248
export SCRATCH_DIR=/scratch/project_462001248

export VENV_DIR=$HOME/.venvs/gbb
export SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"

export WORKDIR="$( pwd )"
export LOCAL_WORKDIR="$( pwd )"

export DATA_DIR=$SCRATCH_DIR/data/

export SYSTEM_SLURM_PARAMS=(
    singularity
    exec
    --bind "$DATA_DIR"
    $SIF
    systems/venv_entry.sh
)

setup_system(){
    # Load modules
    module purge
    module use /appl/local/containers/ai-modules
    module load singularity-AI-bindings
}

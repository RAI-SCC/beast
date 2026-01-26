#!/usr/bin/env bash
export WORKDIR="$( pwd )"
export LOCAL_WORKDIR="$( pwd )"

export SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"
export PROJECT_DIR=/projappl/project_462001248
export SCRACTH_DIR=/scratch/project_462001248
export VENV_DIR=$PROJECT_DIR/venv
export DATA_DIR=$SCRATCH_DIR/data

export SYSTEM_SLURM_PARAMS=(
    singularity
    exec
    $SIF
    systems/lumi_entry.sh
)

setup_system(){
	# Load modules
    module purge
	module use /appl/local/containers/ai-modules
	module load singularity-AI-bindings
}

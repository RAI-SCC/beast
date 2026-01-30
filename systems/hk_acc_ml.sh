#!/usr/bin/env bash
export WORKDIR="$( pwd )"
export LOCAL_WORKDIR="$( pwd )"
export DATA_DIR="/hkfs/work/workspace/scratch/ke4365-era5_dataset/"
export SYSTEM_SLURM_PARAMS=()

setup_system() {

    echo "Setting up modules..."
    ml restore

    echo "Sourcing virtual environment..."
    load_env $WORKDIR/.venv $WORKDIR/requirements.txt
    source $WORKDIR/.venv/bin/activate
    which python

}

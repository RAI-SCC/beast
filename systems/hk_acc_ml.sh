#!/usr/bin/env bash
export WORKDIR="$( pwd )"
export LOCAL_WORKDIR="$( pwd )"
export DATA_DIR="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/"
export MASKS_DIR="/lsdf/kit/scc/projects/"
export SYSTEM_SLURM_PARAMS=()

setup_system() {

    echo "Setting up modules..."
    ml restore

    echo "Sourcing virtual environment..."
    load_env $WORKDIR/.venv $WORKDIR/requirements.txt
    source $WORKDIR/.venv/bin/activate
    which python

}

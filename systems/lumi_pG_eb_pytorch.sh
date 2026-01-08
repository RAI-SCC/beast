#!/usr/bin/env bash
export WORKDIR="$( pwd )"
export LOCAL_WORKDIR="$( pwd )"
# export DATA_DIR="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/"
# export MASKS_DIR="/lsdf/kit/scc/projects/"
export SYSTEM_SLURM_PARAMS=()

setup_system() {

    echo "Setting up modules..."
    ml load LUMI EasyBuild-user/LUMI PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827

}

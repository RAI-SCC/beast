#!/usr/bin/env bash
export PROJECT_DIR="${PROJECT_jureap1}/GBB"
export LOCAL_WORKDIR="$( pwd )"

export CONTAINER_HOME="${PROJECT_DIR}/images"
export CONTAINER_IMAGE="gb_pytorch25.06"

# Container mnt points
export WORKDIR="/mnt" # For source code
export ALT_MNT="/workspace" # For data and additional stuff (probably just mount $PROJECT_DIR)

export SYSTEM_SLURM_PARAMS=(
    apptainer
    exec
    --nv
    --no-home
    #--bind "/scratch"
    #--bind "$TMP"
    --bind "$LOCAL_WORKDIR":"$WORKDIR"
    $CONTAINER_HOME/$CONTAINER_IMAGE
)

setup_system() {
    echo "Purging modules"
    ml purge
}

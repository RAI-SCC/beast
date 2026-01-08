#!/usr/bin/env bash
export LOCAL_WORKDIR="$( pwd )"
export WORKDIR="/mnt"
export DATA_DIR="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/"
export MASKS_DIR="/lsdf/kit/scc/projects/"
export CONTAINER_HOME="$HOME/containers"
export CONTAINER_IMAGE="gb-pytorch25.10"
export SYSTEM_SLURM_PARAMS=(
    apptainer
    exec
    --nv
    --no-home
    --bind "/scratch"
    --bind "$TMP"
    --bind "$LOCAL_WORKDIR":"$WORKDIR"
    # --bind "$DATA_DIR":"$DATA_DIR"
    # --bind "$MASKS_DIR":"$MASKS_DIR"
    $CONTAINER_HOME/$CONTAINER_IMAGE
)

setup_system() {
    echo "Purging modules"
    ml purge
}

#!/usr/bin/env bash
export LOCAL_WORKDIR="$( pwd )"
export WORKDIR="/mnt"
export DATA_DIR="/hkfs/work/workspace/scratch/ke4365-era5_dataset/"
export MASKS_DIR="/lsdf/kit/scc/projects/SmartWeater21/"
export CONTAINER_HOME="$HOME/containers"
export CONTAINER_IMAGE="gb_pytorch25.06"
export SYSTEM_SLURM_PARAMS=(
    apptainer
    exec
    --nv
    --no-home
    --bind "/scratch"
    --bind "$TMP"
    --bind "$LOCAL_WORKDIR":"$WORKDIR"
    --bind "$DATA_DIR":"$DATA_DIR"
    --bind "$MASKS_DIR":"$MASKS_DIR"
    --bind "$LOCAL_WORKDIR/src/libs"
    $CONTAINER_HOME/$CONTAINER_IMAGE
)

setup_system() {
    echo "Purging modules"
    ml purge
}

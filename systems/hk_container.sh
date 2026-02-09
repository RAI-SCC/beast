#!/usr/bin/env bash
export LOCAL_WORKDIR="$( pwd )"
export WORKDIR="/mnt"
#export DATA_DIR="/hkfs/work/workspace/scratch/ke4365-era5_dataset/"
export DATA_DIR="/lsdf/kit/scc/projects/bellbeast/"
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
    --bind "$LOCAL_WORKDIR/src/libs"
    --cwd "$WORKDIR"
    $CONTAINER_HOME/$CONTAINER_IMAGE
)

setup_system() {
    echo "Purging modules"
    ml purge
}

#!/usr/bin/env bash
# =============================================================================
# Container configuration and paths
# =============================================================================
export WORKDIR="$( pwd )" # For source code

export DATA_DIR="/lsdf/kit/scc/projects/bellbeast/"
export CONTAINER_HOME="$HOME/containers"
export SIF="$CONTAINER_HOME/pytorch25.06.sif"

export VENV_DIR="$HOME/.venvs/gbb"

# =============================================================================
# Environment configuration
# =============================================================================
# Locale
export LC_ALL=C
export LANG=C
export PYTHONNOUSERSITE=1

# OpenMPI configuration for UCX transport
export OMPI_MCA_btl="^openib"
export OMPI_MCA_pml="ucx"
export OMPI_MCA_osc="ucx"

# # NCCL configuration for multi-node GPU communication
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="ibs1"

# PMIx for Slurm integration
export PMIX_MCA_gds="hash"

# =============================================================================
# Slurm configuration
# =============================================================================
# CPU_BIND="cores"
# MEM_BIND="local"

export SYSTEM_SLURM_PARAMS=(
    -c18
    --cpu-bind=v,core
    apptainer
    exec
    --nv
    --bind "/scratch"
    --bind "$TMP"
    --bind "$DATA_DIR":"$DATA_DIR"
    $SIF
    systems/venv_entry.sh
)

setup_system() {
    echo "Purging modules"
    ml purge
}

init_system() {
    setup_system
    apptainer exec --nv $SIF python -m venv $VENV_DIR --system-site-packages
    apptainer exec --nv $SIF systems/venv_entry.sh pip install --upgrade pip
    apptainer exec --nv $SIF systems/venv_entry.sh pip install -r requirements.txt
    apptainer exec --nv $SIF systems/venv_entry.sh pip install --upgrade h5py
}

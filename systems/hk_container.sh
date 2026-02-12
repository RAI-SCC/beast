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
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_IB_HCA="mlx5"
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# PMIx for Slurm integration
export PMIX_MCA_gds="hash"

# =============================================================================
# Slurm configuration
# =============================================================================
# CPU_BIND="cores"
# MEM_BIND="local"

export SYSTEM_SLURM_PARAMS=(
    # --cpu-bind "$CPU_BIND"
    # --mem-bind "$MEM_BIND"
    apptainer
    exec
    --nv
    --bind "/scratch"
    --bind "$TMP"
    --bind "$DATA_DIR":"$DATA_DIR"
    $SIF
    systems/venv_entrypoint.sh
)

setup_system() {
    echo "Purging modules"
    ml purge
    ml load 
}

#!/usr/bin/env bash
# =============================================================================
# Container configuration and paths
# =============================================================================
export PROJECT_DIR="${PROJECT_jureap1}/GBB"

export VENV_DIR="${HOME}/.venvs/gbb"
export CONTAINER_HOME="${PROJECT_DIR}/images"
export SIF="${CONTAINER_HOME}/pytorch25.06.sif"

# Container mnt points
export WORKDIR="$( pwd )" # For source code

#export DATA_DIR="${PROJECT_DIR}/data"
export DATA_DIR="/e/scratch/jureap1/GrandBellbeast/data/"

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

# UCX configuration for InfiniBand NDR200
export UCX_TLS="rc,cuda_copy,cuda_ipc"
export UCX_MEMTYPE_CACHE="n"
export UCX_RNDV_FRAG_MEM_TYPE="cuda"
export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"

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
    --bind "$DATA_DIR":"/workspace/"
    $SIF
    systems/venv_entry.sh
)

setup_system() {
    echo "Purging modules"
    ml purge
    ml load Stages/2026 GCC/14.3.0 OpenMPI/5.0.8 MPI-settings/CUDA
    
    echo "Loaded modules:"
    module list
}

init_system() {
    setup_system
    apptainer exec --nv $SIF python -m venv $VENV_DIR --system-site-packages
    apptainer exec --nv $SIF systems/venv_entry.sh pip install --upgrade pip
    apptainer exec --nv $SIF systems/venv_entry.sh pip install -r requirements.txt
    apptainer exec --nv $SIF systems/venv_entry.sh pip install --upgrade h5py
}

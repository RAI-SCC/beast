#!/usr/bin/env bash
export PROJECT_DIR="${PROJECT_jureap1}/GBB"

export VENV_DIR="${HOME}/.venvs/gbb"
export CONTAINER_HOME="${PROJECT_DIR}/images"
export SIF="${CONTAINER_HOME}/pytorch25.06.sif"

# Container mnt points
export WORKDIR="/mnt" # For source code
export LOCAL_WORKDIR="$( pwd )"
export ALT_MNT="/workspace" # For data and additional stuff (probably just mount $PROJECT_DIR)

#export DATA_DIR="${PROJECT_DIR}/data"
export DATA_DIR="/e/scratch/jureap1/GrandBellbeast/data/"

# Set environment variables for the container
export LC_ALL=C
export LANG=C

# Prevent user site-packages from interfering
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

export SYSTEM_SLURM_PARAMS=(
    apptainer
    exec
    --nv
    --bind "$DATA_DIR":"/workspace/"
    --bind "$LOCAL_WORKDIR":"$WORKDIR"
    $SIF
    systems/venv_entry.sh
)

setup_system() {
    echo "Purging modules"
    ml purge
    ml load Stages/2026 GCC/14.3.0 OpenMPI/5.0.8 MPI-settings/CUDA
}

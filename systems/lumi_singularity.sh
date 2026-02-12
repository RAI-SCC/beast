#!/usr/bin/env bash
# =============================================================================
# Container configuration and paths
# =============================================================================
export PROJECT_DIR=/projappl/project_462001248
export SCRATCH_DIR=/scratch/project_462001248

export VENV_DIR=$HOME/.venvs/gbb
export SIF="/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif"

export WORKDIR="$( pwd )"
export LOCAL_WORKDIR="$( pwd )"

export DATA_DIR=$SCRATCH_DIR/data/

# =============================================================================
# Environment configuration 
# =============================================================================
# Locale
export LC_ALL=C
export LANG=C
export PYTHONNOUSERSITE=1

# Cray MPICH Configuration
export MPICH_GPU_SUPPORT_ENABLED=1          # Enable GPU-aware MPI
export MPICH_OFI_NIC_POLICY="GPU"           # Use GPU-mapped NICs
export MPICH_RANK_REORDER_METHOD=1          # Optimize rank placement
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

# RCCL Configuration 
export RCCL_DEBUG="${RCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"  
export NCCL_NET_GDR_LEVEL=3                     

# Libfabric/CXI for Slingshot 
export FI_CXI_ATS=0
export FI_CXI_DISABLE_CQ_HUGETLB=1
export FI_MR_CACHE_MONITOR=userfaultfd

# ROCm / HIP Settings
export ROCR_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 8 GCDs per node
export HIP_FORCE_DEV_KERNARG=1

# =============================================================================
# Slurm configuration
# =============================================================================

# CPU mask for optimal GPU-CPU affinity on LUMI-G

# Each MI250X has 2 GCDs. The CPU-GPU mapping on LUMI-G is:
#   GCD 4,5 -> NUMA 0 (cores 0-15)  -> CCD 0,1
#   GCD 2,3 -> NUMA 1 (cores 16-31) -> CCD 2,3
#   GCD 6,7 -> NUMA 2 (cores 32-47) -> CCD 4,5
#   GCD 0,1 -> NUMA 3 (cores 48-63) -> CCD 6,7
#
# For 8 tasks (one per GCD), bind each task to 7 cores (mask 0xfe = 11111110)
# Leaving core 0 of each group for system tasks

CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

# Memory binding to NUMA nodes
MEM_BIND="map_mem:3,3,1,1,0,0,2,2"

export SYSTEM_SLURM_PARAMS=(
    --cpu-bind="${CPU_BIND}"
    --mem-bind="${MEM_BIND}"
    singularity
    exec
    --rocm
    --bind "$DATA_DIR"
    $SIF
    systems/venv_entry.sh
)

setup_system(){
    module purge
    echo "Setting up LUMI-G environment..."
    
    # Load LUMI software stack
    module load LUMI/24.03
    module load partition/G
    
    # Load ROCm
    module load rocm/6.0.3
    
    # Load Cray MPICH with GPU support
    module load cray-mpich
    
    # Load singularity bindings for MPI-aware containers
    module load singularity-bindings
    
    # AWS OFI RCCL plugin for GPU-aware MPI over Slingshot
    module load aws-ofi-rccl

    echo "Loaded modules:"
    module list
}

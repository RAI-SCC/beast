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
export OUTPUT_DIR=$PROJECT_DIR/output

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
export RCCL_DEBUG="${RCCL_DEBUG:-INFO}"
export NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"
export NCCL_NET_GDR_LEVEL="PHB"

# Libfabric/CXI for Slingshot
# export FI_CXI_ATS=0
# export FI_CXI_DISABLE_CQ_HUGETLB=1
# export FI_MR_CACHE_MONITOR=userfaultfd

# ROCm / HIP Settings
# export ROCM_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 8 GCDs per node
#
# =============================================================================
# Profiling configuration
# =============================================================================

export PROFILE_DIR="${OUTPUT_DIR}/${SLURM_JOB_PARTITION}/${SLURM_JOB_ID}/profiles"

profsys_cmd=(
    rocprofv3
    --sys-trace
    --output-format pftrace
    -d "$PROFILE_DIR"
    --
)
export PROFSYS_CMD="${profsys_cmd[@]}"

# =============================================================================
# Slurm configuration
# =============================================================================

# CPU mask for optimal GPU-CPU affinity on LUMI-G

# Each MI250X has 2 GCDs. The CPU-GPU mapping on LUMI-G is:
#   GCD 4,5 -> NUMA 0 (cores 0-15)  -> GCD 0,1
#   GCD 2,3 -> NUMA 1 (cores 16-31) -> GCD 2,3
#   GCD 6,7 -> NUMA 2 (cores 32-47) -> GCD 4,5
#   GCD 0,1 -> NUMA 3 (cores 48-63) -> GCD 6,7
#
# For 8 tasks (one per GCD), bind each task to 7 cores (mask 0xfe = 11111110)
# Leaving core 0 of each group for system tasks

# Set up the CPU bind masks
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

export SYSTEM_SLURM_PARAMS=(
    --cpu-bind="mask_cpu=$CPU_BIND_MASKS"
    --cpus-per-task=7
    --mpi=pmi2
    singularity
    exec
    --bind "$DATA_DIR"
    $SIF
    systems/venv_entry.sh
)

setup_system(){
    # module purge
    echo "Setting up LUMI-G environment..."
    module use /appl/local/containers/ai-modules
    module load singularity-AI-bindings

    # Load LUMI software stack
    module load LUMI/25.03
    module load partition/G
    module load cpe/25.03
    module load amd/6.3.4
    module load craype
    module load rocm/6.3.4
    module load cray-mpich/8.1.32

    echo "Loaded modules:"
    module list
}

init_system(){
    module use /appl/local/containers/ai-modules
    module load singularity-AI-bindings
    singularity exec $SIF python -m venv $VENV_DIR --system-site-packages
    singularity exec $SIF systems/venv_entry.sh pip install --upgrade pip
    singularity exec $SIF systems/venv_entry.sh pip install -r requirements.txt
    singularity exec $SIF systems/venv_entry.sh pip install --upgrade h5py
    singularity exec $SIF systems/venv_entry.sh pip install --force-reinstall aiter
    singularity exec $SIF systems/venv_entry.sh python -c 'import aiter'
}

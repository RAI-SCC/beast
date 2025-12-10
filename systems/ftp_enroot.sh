#!/usr/bin/env bash
export LOCAL_WORKDIR="$( pwd )"
export WORKDIR="/workspace"
export DATA_DIR="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/"
export MASKS_DIR="/lsdf/kit/scc/projects/"
export SYSTEM_SLURM_PARAMS=(
   --container-image=rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1
   --container-name=gb-pytorch-rocm6.4.4
   # --container-mounts=/etc/slurm/task_prolog.ftp-x86:/etc/slurm/task_prolog.ftp-x86,/scratch:/scratch,/usr/lib64/slurm/libslurmfull.so,/usr/lib64/libhwloc.so.15,$LOCAL_WORKDIR:$WORKDIR,$DATA_DIR:$DATA_DIR,$MASKS_DIR:$MASKS_DIR
   --container-mounts=/etc/slurm/task_prolog.ftp-x86:/etc/slurm/task_prolog.ftp-x86,/scratch:/scratch,/usr/lib64/slurm/libslurmfull.so,/usr/lib64/libhwloc.so.15,$LOCAL_WORKDIR:$WORKDIR
   --container-writable
)

setup_system() {

    echo "Inside the container, nothing to do ..."
    
}
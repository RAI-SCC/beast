#!/usr/bin/env bash
export LOCAL_WORKDIR="$( pwd )"
export WORKDIR="/workspace"
export DATA_DIR="/lsdf/kit/imk-tro/projects/Gruppe_Quinting/"
export MASKS_DIR="/lsdf/kit/scc/projects/"
export SYSTEM_SLURM_PARAMS=(
   --container-image=nvcr.io#nvidia/pytorch:25.10-py3
   --container-name=gb-pytorch25.10
   # --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm/libslurmfull.so,/usr/lib64/libhwloc.so.15,$LOCAL_WORKDIR:$WORKDIR,$DATA_DIR:$DATA_DIR,$MASKS_DIR:$MASKS_DIR
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm/libslurmfull.so,/usr/lib64/libhwloc.so.15,$LOCAL_WORKDIR:$WORKDIR
   --container-writable
)

setup_system() {

    echo "Inside the container, nothing to do ..."
    
}
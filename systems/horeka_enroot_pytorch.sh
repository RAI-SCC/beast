#!/usr/bin/env bash
export WORKDIR="/workspace"
export SYSTEM_ID="horeka_enroot_pytorch"
export SYSTEM_SBATCH_PARAMS=(
   --container-image=nvcr.io#nvidia/pytorch:25.10-py3
   --container-name=redist-bench-pytorch25.10
   --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/usr/lib64/slurm/libslurmfull.so,/usr/lib64/libhwloc.so.15,$LOCAL_WORKDIR:$WORKDIR
   --container-writable
)

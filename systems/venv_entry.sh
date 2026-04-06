#!/usr/bin/env bash
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "VENV_DIR: $VENV_DIR"
    echo "DATA_DIR: $DATA_DIR"
    echo "Runnning cmd: $@"
fi

source $VENV_DIR/bin/activate

if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "PYTHON ENVIRONMENT SOURCED"
fi

if [[ "$SLURM_PROCID" =~ ^([0-3]|64)$ ]]; then
    if [[ ${PROFSYS} == true ]]; then
        echo "Enabling program tracing (PROF_TRACE=true) for process $SLURM_PROCID"
        echo "Profiling command: $PROFSYS_CMD $@"
        $PROFSYS_CMD $@
    else
        exec "$@"
    fi
else
    exec "$@"
fi

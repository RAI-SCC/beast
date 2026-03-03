#!/usr/bin/env bash
source $VENV_DIR/bin/activate
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "VENV_DIR: $VENV_DIR"
    echo "DATA_DIR: $DATA_DIR"
    echo "Runnning cmd: $@"
fi
exec "$@"

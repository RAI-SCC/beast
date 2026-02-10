#!/usr/bin/env bash
echo "Setting up container venv..."
echo "VENV_DIR: $VENV_DIR"
echo "DATA_DIR: $DATA_DIR"
$WITH_CONDA
source $VENV_DIR/bin/activate
echo "Runnning cmd: $@"
exec "$@"

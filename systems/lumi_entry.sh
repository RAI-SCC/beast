#!/bin/bash
echo "Setting up container venv..."
$WITH_CUDA
source $VENV_DIR/bin/activate
echo "Runnning cmd: $@"
exec "$@"

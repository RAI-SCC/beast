#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <system_env>"
    echo "  system_env: Path to the system environment file"
    exit 1
fi

SYSTEM_ENV_FILE="$(basename $1 .env)"

echo "System Env File: $SYSTEM_ENV_FILE"

echo "Sourcing system environment..."
source $1
shift 1

SALLOC_PARAMS+=("${SYSTEM_SLURM_PARAMS[@]}")
SALLOC_PARAMS+=("$@")

# Clear environment modules
ml purge

echo "Full cmd:
salloc ${SALLOC_PARAMS[*]}"
echo ""

# Launch interactive SLURM job with enroot based on the system slurm params and other command line parameters
salloc "${SALLOC_PARAMS[@]}" 
#!/usr/bin/env bash
need() { command -v "$1" >/dev/null 2>&1 || return 1; }

print_slurm_env() {
    echo "Slurm variables:"
    echo "  SLURM_JOB_ID:         ${SLURM_JOB_ID:-N/A}"
    echo "  SLURM_PARTITION:      ${SLURM_JOB_PARTITION:-N/A}"
    echo "  SLURM_NTASKS:         ${SLURM_NTASKS:-N/A}"
    echo "  SLURM_NNODES:         ${SLURM_NNODES:-N/A}"
    echo "  SLURM_TASKS_PER_NODE: ${SLURM_TASKS_PER_NODE:-N/A}"
    echo "  SLURM_NODELIST:       ${SLURM_NODELIST:-N/A}"
}

# Create env function: pass as first argument the env path, second argument is a requirenments file. If the environment already exists, it will be reused.
load_env() {
    local PYTHON_BIN=${PYTHON_BIN:-python3.12}
    local ENV_PATH=$1
    shift
    local REQS_FILE=$1
    shift

    if [[ -d "$ENV_PATH" ]]; then
        echo "Loading environment at $ENV_PATH."
        source "$ENV_PATH/bin/activate"
    else
        echo "Creating environment at $ENV_PATH"
        $PYTHON_BIN -m venv "$ENV_PATH"
        source "$ENV_PATH/bin/activate"
        pip install --upgrade pip
        if [[ -f "$REQS_FILE" ]]; then
            pip install -r "$REQS_FILE"
        else
            echo "Requirements file $REQS_FILE not found! Creating empty environment."
        fi
        deactivate
    fi
}

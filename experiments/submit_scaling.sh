#!/bin/bash
# submit_weak_lsn.sh
set -euo pipefail
SYSTEM_NAME="lumi"

# ── Sweep parameters ──────────────────────────────────────────────────────────
# EXP_NAME="weak_model_lsn"
# VAR_NAME="lsn"
# VARS=(1 2 4 8 16)

# EXP_NAME="weak_model_lbs"
# VAR_NAME="lbs"
# VARS=(1 2 4 8 16)

EXP_NAME="weak_model_dim"
VAR_NAME="embed_dim"
VARS=(128 256 512)

# ── Defaults (can be overridden via flags) ────────────────────────────────────
ACCOUNT="project_462001248"
PARTITION="standard-g"
NODES=2
TASKS_PER_NODE=8
CPUS_PER_TASK=7
GPUS=8
TIME="01:00:00"
PLOT_TIME="00:10:00"
CONTAINER_SETUP="lumi_singularity.sh"

# ── Parse remaining args, intercept SLURM overrides ──────────────────────────
remaining_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account|-A)        ACCOUNT="$2";        shift 2 ;;
        --partition|-p)      PARTITION="$2";      shift 2 ;;
        --nodes|-N)          NODES="$2";           shift 2 ;;
        --tasks-per-node)    TASKS_PER_NODE="$2"; shift 2 ;;
        --gpus)              GPUS="$2";            shift 2 ;;
        --time)              TIME="$2";            shift 2 ;;
        --plot-time)         PLOT_TIME="$2";       shift 2 ;;
        *)                   remaining_args+=("$1"); shift ;;
    esac
done

REPO_ROOT="$(pwd)"

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_DIR="${REPO_ROOT}/logs/${SYSTEM_NAME}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
CONFIG_DIR="${EXPERIMENT_DIR}/configs"
LOG_DIR="${EXPERIMENT_DIR}/logs"
ANALYSIS_DIR="${REPO_ROOT}/results/${SYSTEM_NAME}"


mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$ANALYSIS_DIR"

# ── Sweep metadata ────────────────────────────────────────────────────────────
cat > "${EXPERIMENT_DIR}/sweep_config.yaml" <<EOF
sweep:
  submitted_at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git_commit: "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
  git_branch: "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
  ${VAR_NAME}: [${VARS[*]}]
EOF

# ── Submit sweep jobs ─────────────────────────────────────────────────────────
RUN_ID=0
JOB_IDS=()

for VAR in "${VARS[@]}"; do

    RUN_NAME=$(printf "run_%03d_${VAR}-%04d" "$RUN_ID" "$VAR")

    CONFIG_FILE="${CONFIG_DIR}/${RUN_NAME}.yaml"
    cat > "$CONFIG_FILE" <<EOF
run:
  id: ${RUN_ID}
  name: "${RUN_NAME}"
  submitted_at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
model:
  ${VAR_NAME}: ${VAR}
output:
  log_file: "${LOG_DIR}/${RUN_NAME}.out"
EOF

    RAW=$(sbatch \
        -A "$ACCOUNT" -p "$PARTITION" \
        --nodes "$NODES" --tasks-per-node "$TASKS_PER_NODE" --gres=gpu:${GPUS} \
        --time ${TIME} \
        --job-name="scale_${RUN_NAME}" \
        --output="${LOG_DIR}/${RUN_NAME}.out" \
        --error="${LOG_DIR}/${RUN_NAME}.err" \
        experiments/run_gb_script.sh \
        systems/${CONTAINER_SETUP} \
        src/train_bellbeast.py \
        --config_file "configs/bellbeast_config.yaml" \
        --train_subset 10 \
        --"${VAR_NAME}" ${VAR} )

    JOB_ID=$(echo "$RAW" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")

    echo "  slurm_job_id: ${JOB_ID}" >> "$CONFIG_FILE"
    cp configs/bellbeast_config.yaml "${CONFIG_DIR}/"

    echo "Submitted ${RUN_NAME} → SLURM job ${JOB_ID}"
    RUN_ID=$(( RUN_ID + 1 ))
done

echo ""
echo "All sweep jobs submitted. Experiment root: ${EXPERIMENT_DIR}"

# ── Build afterany dependency string  job1:job2:job3 … ───────────────────────
DEPENDENCY=$(IFS=:; echo "${JOB_IDS[*]}")
SWEEP_TAG=$(basename "${EXPERIMENT_DIR}")
PLOT_PATH="${ANALYSIS_DIR}/scaling_${SWEEP_TAG}.png"

# ── Write self-contained plotting job script ──────────────────────────────────
PLOT_SCRIPT="${EXPERIMENT_DIR}/plot_job.sh"
cat > "${PLOT_SCRIPT}" <<PLOTEOF
#!/bin/bash
#SBATCH -A "$ACCOUNT"
#SBATCH -p "$PARTITION"
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --time "$PLOT_TIME"
#SBATCH --job-name="plot_${SWEEP_TAG}"
#SBATCH --output="${LOG_DIR}/plot_${SWEEP_TAG}.out"
#SBATCH --error="${LOG_DIR}/plot_${SWEEP_TAG}.err"

# /etc/bashrc on Jupiter sets BASHRCSOURCED without exporting it first,
# which trips up 'set -u'.  Source it with nounset temporarily disabled.
set +u
source /etc/bashrc 2>/dev/null || true
set -u
set -eo pipefail

echo "========================================================"
echo " Scaling plot job"
echo " Sweep     : ${SWEEP_TAG}"
echo " Logs dir  : ${LOG_DIR}"
echo " Output    : ${PLOT_PATH}"
echo " Started   : \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================"

# Activate the project virtualenv (same path used by training jobs)

python3 "${REPO_ROOT}/scaling_experiments/analysis/parse_scaling.py" \\
    --logs_dir  "${LOG_DIR}" \\
    --out_path  "${PLOT_PATH}" \\
    --sweep_id  "${SWEEP_TAG}"

echo ""
echo "Plot written → ${PLOT_PATH}"
echo "Finished    : \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PLOTEOF

chmod +x "${PLOT_SCRIPT}"

# ── Submit plotting job with afterany dependency ──────────────────────────────
# afterany fires after every listed job ends, regardless of exit status.
PLOT_JOB_RAW=$(sbatch \
    --dependency="afterany:${DEPENDENCY}" \
    --kill-on-invalid-dep=yes \
    "${PLOT_SCRIPT}")

PLOT_JOB_ID=$(echo "$PLOT_JOB_RAW" | awk '{print $NF}')

# Record the plotting job in the sweep metadata
cat >> "${EXPERIMENT_DIR}/sweep_config.yaml" <<EOF
  plot_job_id: ${PLOT_JOB_ID}
  plot_dependency: "${DEPENDENCY}"
  plot_output: "${PLOT_PATH}"
EOF

echo ""
echo "Plot job submitted → SLURM job ${PLOT_JOB_ID}"
echo "  dependency : afterany:${DEPENDENCY}"
echo "  output     : ${PLOT_PATH}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${PLOT_JOB_ID}"
echo "  squeue -u \$USER"

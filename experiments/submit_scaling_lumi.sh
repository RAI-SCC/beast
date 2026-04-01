#!/bin/bash
# submit_weak_lsn.sh
set -euo pipefail

# ── Experiments ──────────────────────────────────────────────────────────
# Uncomment the one that you want to run

# ---------------------------
# SINGLE-NODE EXPERIMENTS
# ---------------------------
# Exp 2.1: Increase the local sampling number
# Model size stays the same
# ---------------------------
# EXP_NAME="weak_model_lsn"
# VAR_NAME="training_n_samples"
# VARS=(1 2 4 6 8)
# MESH_STRING="1,1,2,4,2"
# LOCAL_SAMPLES=1 #will be overwritten by VARS
# EMBED_DIM=128
# NODES=(2) # stays constant
# X_AXIS_NAME=${VAR_NAME}
# SUBSET_SIZE=500

# ---------------------------
# Exp 2.2: Increase the local batch size
# Model size stays the same
# Total dataset size increases with the local batch size
# ---------------------------
# EXP_NAME="weak_model_lbs"
# VAR_NAME="training_batch_size"
# VARS=(1 2 4 6 8)
# MESH_STRING="1,1,2,4,2"
# LOCAL_SAMPLES=1
# EMBED_DIM=128
# NODES=(2) # Stays constant
# X_AXIS_NAME=${VAR_NAME}
# SUBSET_SIZE=500

# ---------------------------
# Exp 2.3: Increase the model size
# Local batch size and sampling number stay the same
# ---------------------------

# EXP_NAME="weak_model_dim"
# VAR_NAME="model_embedding_dimension"
# VARS=(128 256 512 1024)
# MESH_STRING="1,1,2,4,2"
# LOCAL_SAMPLES=1
# EMBED_DIM=128 # will be overwritten by VARS
# NODES=(2) # Stays constant
# X_AXIS_NAME=${VAR_NAME}
# SUBSET_SIZE=500

# ---------------------------
# MULTI-NODE EXPERIMENTS
# ---------------------------
# Exp 3.1: Increase global sampling number by increasing node count
# Local sampling number and batch size stay the same
# Total dataset size stays the same
# ---------------------------

EXP_NAME="up_weak_scaling_baseline"
VAR_NAME="training_batch_size"
VARS=(1)
EMBED_DIM=512
LOCAL_SAMPLES=2
MESH_STRING="1,1,-1,4,2"
NODES=(1 4 8 16 32 64 128 256 512 1024)
X_AXIS_NAME="nodes"
SUBSET_SIZE=500

# Exp 3.2: Increase global sampling number by increasing node count
# Local sampling number and batch size stay the same
# Total dataset size stays the same
# ---------------------------

# EXP_NAME="up_weak_scaling_realistic"
# VAR_NAME="training_batch_size"
# VARS=(1)
# EMBED_DIM=512
# LOCAL_SAMPLES=2
# MESH_STRING="1,16,-1,4,2"
# NODES=(16 32 64 128 256 512 1024)
# X_AXIS_NAME="nodes"
# SUBSET_SIZE=500

# ---------------------------
# Exp 4.1: Increase global batch size by increasing node count
# Local sampling number and batch size stay the same
# Total dataset is scaled with node count to ensure constant work per node
# ---------------------------

# EXP_NAME="dp_strong_scaling_baseline"
# VAR_NAME="training_batch_size"
# VARS=(1)
# EMBED_DIM=512
# LOCAL_SAMPLES=2
# MESH_STRING="1,-1,1,4,2"
# NODES=(1 4 8 16 32 64 128 256 512 1024)
# SUBSET_SIZE=4096
# X_AXIS_NAME="nodes"

# ---------------------------
# Exp 4.2: Increase global batch size by increasing node count
# Local sampling number and batch size stay the same
# Total dataset is scaled with node count to ensure constant work per node
# ---------------------------

# EXP_NAME="dp_strong_scaling_realistic"
# VAR_NAME="training_batch_size"
# VARS=(1)
# EMBED_DIM=512
# LOCAL_SAMPLES=2
# MESH_STRING="1,-1,8,4,2"
# NODES=(8 16 32 64 128 256 512 1024)
# SUBSET_SIZE=4096
# X_AXIS_NAME="nodes"

# ── System (can be overridden via flags) ────────────────────────────────────
SYSTEM_NAME="lumi"
ACCOUNT="project_462001248"
PARTITION="standard-g"
NODE=2
TASKS_PER_NODE=8
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
        --node|-N)           NODE="$2";           shift 2 ;;
        --tasks-per-node)    TASKS_PER_NODE="$2"; shift 2 ;;
        --gpus)              GPUS="$2";            shift 2 ;;
        --time)              TIME="$2";            shift 2 ;;
        --plot-time)         PLOT_TIME="$2";       shift 2 ;;
        *)                   remaining_args+=("$1"); shift ;;
    esac
done

REPO_ROOT="$(pwd)"

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENT_DIR="${PROJECT_DIR}/output/${SYSTEM_NAME}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
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

for NODE in "${NODES[@]}"; do
for VAR in "${VARS[@]}"; do

    RUN_NAME=$(printf "run_%03d_${VAR_NAME}-%04d_nodes-%04d" "$RUN_ID" "$VAR" "$NODE")

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
        --nodes "${NODE}" --tasks-per-node "$TASKS_PER_NODE" --gres=gpu:${GPUS} \
        --time ${TIME} \
        --job-name="scale_${RUN_NAME}" \
        --output="${LOG_DIR}/${RUN_NAME}.out" \
        --error="${LOG_DIR}/${RUN_NAME}.err" \
        experiments/run_gb_script.sh \
        systems/${CONTAINER_SETUP} \
        src/train_bellbeast.py \
        --config_file "configs/bellbeast_config_lumi.yaml" \
        --skip_validation \
        mesh_dims=${MESH_STRING} \
        data_train_subset=${SUBSET_SIZE} \
        training_n_samples=${LOCAL_SAMPLES} \
        model_embedding_dimension=${EMBED_DIM} \
        ${VAR_NAME}=${VAR}

    JOB_ID=$(echo "$RAW" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")

    echo "  slurm_job_id: ${JOB_ID}" >> "$CONFIG_FILE"
    cp configs/bellbeast_config_lumi.yaml "${CONFIG_DIR}/"

    echo "Submitted ${RUN_NAME} → SLURM job ${JOB_ID}"
    RUN_ID=$(( RUN_ID + 1 ))
done
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

module use /appl/local/csc/modulefiles
module load pytorch

python3 "${REPO_ROOT}/experiments/parse_scaling.py" \\
    --logs_dir  "${LOG_DIR}" \\
    --out_path  "${PLOT_PATH}" \\
    --sweep_id  "${SWEEP_TAG}" \\
    --x_axis "${X_AXIS_NAME}"

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

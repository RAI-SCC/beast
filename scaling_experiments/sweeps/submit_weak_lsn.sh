#!/bin/bash
# submit_weak_scaling.sh

# ── Sweep parameters ──────────────────────────────────────────────────────────
SAMPLES=(1 2 4 8 16 32 64)

# ── Experiment root ───────────────────────────────────────────────────────────
EXPERIMENT_DIR="$(pwd)/scaling_experiments/experiments/weak_model_lsn_$(date +%Y%m%d_%H%M%S)"
CONFIG_DIR="${EXPERIMENT_DIR}/configs"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "$CONFIG_DIR" "$LOG_DIR"

# ── Save the sweep metadata itself ────────────────────────────────────────────
cat > "${EXPERIMENT_DIR}/sweep_config.yaml" <<EOF
sweep:
  submitted_at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  git_commit: "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
  git_branch: "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
  sample: [${SAMPLES[*]}]
EOF

RUN_ID=0

for SAMPLE in "${SAMPLES[@]}"; do
  RUN_NAME=$(printf "run_%03d_sample-%02d" "$RUN_ID" "$SAMPLE")

  # ── Dump the per-run YAML ──────────────────────────────────────────────
  CONFIG_FILE="${CONFIG_DIR}/${RUN_NAME}.yaml"
  cat > "$CONFIG_FILE" <<EOF
    run:
    id: ${RUN_ID}
    name: "${RUN_NAME}"
    submitted_at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    model:
    dims: ${DIM}
    output:
    log_file: "${LOG_DIR}/${RUN_NAME}.out"
EOF

  # ── Submit ─────────────────────────────────────────────────────────────
  JOB_ID=$(sbatch -A jureap1 -p booster --nodes 2 --tasks-per-node 4 --gres=gpu:4 --time 01:00:00 \
    --job-name="scale_${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}.out" \
    --error="${LOG_DIR}/${RUN_NAME}.err" \
    experiments/run_gb_script.sh systems/jupiter_container.sh src/train_bellbeast.py --config_file "configs/bellbeast_config.yaml" \
    --n_samples ${SAMPLE} )

  # ── Record job ID back into the config ────────────────────────────────
  echo "  slurm_job_id: ${JOB_ID}" >> "$CONFIG_FILE"

  echo "Submitted ${RUN_NAME} → SLURM job ${JOB_ID}"
  (( RUN_ID++ ))

done

echo ""
echo "All jobs submitted. Experiment root: ${EXPERIMENT_DIR}"

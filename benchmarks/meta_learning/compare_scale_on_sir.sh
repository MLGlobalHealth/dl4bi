#!/usr/bin/env bash

# Do NOT exit on error
set +e

LOG_DIR="logs"
FAIL_LOG="${LOG_DIR}/failures.log"
mkdir -p "$LOG_DIR"

# ----------------------------
# Sweeps
# ----------------------------

MODELS=(
  bsa_tnp
  tnp_d
  te_tnp
  convcnp
)
DATASETS=(
  spatial_128x128
  spatial_256x256
  spatial_512x512
  spatial_1024x1024
)
SEEDS=(46) # other seeds don't matter for timing/scaling

# ----------------------------
# Fixed args
# ----------------------------

VALID_NUM_STEPS=10 # number of samples since batch_size=1
PROJECT="AISTATS BSA-TNP - SIR"
EVAL_ONLY=True

# ----------------------------
# Dataset-specific convcnp params
# ----------------------------

get_convcnp_bounds () {
  case "$1" in
    spatial_128x128)
      echo "model.s_lower=[-4.5,-4.5] model.s_upper=[4.5,4.5]"
      ;;
    spatial_256x256)
      echo "model.s_lower=[-8.5,-8.5] model.s_upper=[8.5,8.5]"
      ;;
    spatial_512x512)
      echo "model.s_lower=[-16.5,-16.5] model.s_upper=[16.5,16.5]"
      ;;
    spatial_1024x1024)
      echo "model.s_lower=[-32.5,-32.5] model.s_upper=[32.5,32.5]"
      ;;
    *)
      echo ""
      ;;
  esac
}

# ----------------------------
# Sweep
# ----------------------------

for MODEL in "${MODELS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      EXTRA_ARGS=""
      [[ "$MODEL" == "convcnp" ]] && EXTRA_ARGS=$(get_convcnp_bounds "$DATA")

      RUN_ID="model=${MODEL}_data=${DATA}_seed=${SEED}"
      RUN_LOG="${LOG_DIR}/${RUN_ID}.log"

      echo "=============================================="
      echo "Running $RUN_ID"
      [[ -n "$EXTRA_ARGS" ]] && echo "Extra args: $EXTRA_ARGS"
      echo "Log: $RUN_LOG"
      echo "=============================================="

      python sir.py \
        model="$MODEL" \
        data="$DATA" \
        data.batch_size=1 \
        seed="$SEED" \
        valid_num_steps="$VALID_NUM_STEPS" \
        project="$PROJECT" \
        evaluate_only="$EVAL_ONLY" \
        $EXTRA_ARGS \
        > "$RUN_LOG" 2>&1

      EXIT_CODE=$?

      # Detect the specific OOM signature in the log (works whether or not exit code is 137)
      OOM_TAG=""
      if grep -Fq "RESOURCE_EXHAUSTED: Out of memory" "$RUN_LOG"; then
        OOM_TAG=" [OOM:RESOURCE_EXHAUSTED]"
      fi

      if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[FAIL] $RUN_ID (exit=$EXIT_CODE)${OOM_TAG}" | tee -a "$FAIL_LOG"
      else
        # Optional: still record OOMs even if exit code is 0 (rare, but possible if caught)
        if [[ -n "$OOM_TAG" ]]; then
          echo "[WARN] $RUN_ID (exit=$EXIT_CODE)${OOM_TAG}" | tee -a "$FAIL_LOG"
        else
          echo "[OK]   $RUN_ID"
        fi
      fi

    done
  done
done

echo "Sweep finished."
echo "Failures (if any) logged to: $FAIL_LOG"

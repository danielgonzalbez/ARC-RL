#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -x .venv/bin/python ]]; then
  default_python=.venv/bin/python
else
  default_python=python3
fi

PYTHON_BIN=${PYTHON_BIN:-$default_python}
CHECKPOINT=${CHECKPOINT:-checkpoints/qwen35_arc_full8192_sft60_bs16_v2_best.safetensors}
MANIFEST=${MANIFEST:-data_splits/qwen35_rl_diagnostic_16tasks_aug_holdout.json}
PAIR_WEIGHT=${PAIR_WEIGHT:-0.75}
STRUCTURED_F1_WEIGHT=${STRUCTURED_F1_WEIGHT:-0.25}
LOSS_REDUCTION=${LOSS_REDUCTION:-token}
EOS_STRUCTURE_WEIGHT=${EOS_STRUCTURE_WEIGHT:-3.0}
DUPLICATE_COORDINATE_WEIGHT=${DUPLICATE_COORDINATE_WEIGHT:-2.0}
EOS_MISSING_PENALTY=${EOS_MISSING_PENALTY:-8.0}
EOS_DUPLICATE_PENALTY=${EOS_DUPLICATE_PENALTY:-4.0}
EOS_INVALID_PENALTY=${EOS_INVALID_PENALTY:-4.0}
EOS_MALFORMED_PENALTY=${EOS_MALFORMED_PENALTY:-2.0}
NO_EOS_PENALTY=${NO_EOS_PENALTY:-2.0}
RANKING_PAIR_F1_WEIGHT=${RANKING_PAIR_F1_WEIGHT:-0.35}
RANKING_COORDINATE_COVERAGE_WEIGHT=${RANKING_COORDINATE_COVERAGE_WEIGHT:-0.65}
RANKING_DUPLICATE_COORDINATE_PENALTY=${RANKING_DUPLICATE_COORDINATE_PENALTY:-0.50}
RANKING_INVALID_COORDINATE_PENALTY=${RANKING_INVALID_COORDINATE_PENALTY:-0.50}

mkdir -p logs checkpoints

if [[ -z ${RUN_NAME+x} ]]; then
  base_run_name=qwen35_rl_diagnostic_16tasks_mix_v1
  RUN_NAME=$base_run_name
  restart=1
  while [[ -e "logs/${RUN_NAME}.jsonl" || -e "checkpoints/${RUN_NAME}_best" ]]; do
    printf -v RUN_NAME '%s_restart%02d' "$base_run_name" "$restart"
    ((restart += 1))
  done
  echo "run_name=${RUN_NAME}"
fi

TRITON_PRINT_AUTOTUNING=0 \
VLLM_LOGGING_LEVEL=ERROR \
WANDB_SILENT=true \
VLLM_WORK_DIR="${VLLM_WORK_DIR:-/tmp/arc_vllm_work}" \
"$PYTHON_BIN" -m arc_rl.rl_qwen35_native_vllm \
  --tokenizer-mode arc \
  --checkpoint "$CHECKPOINT" \
  --output-checkpoint "checkpoints/${RUN_NAME}_best" \
  --rl-train-source training-partition \
  --train-task-partition data_splits/qwen35_arc_full8192_sft60_rl40_seed3407.json \
  --train-partition-role rl \
  --augment-unseen-long \
  --num-color-swaps 4 \
  --diagnostic-example-split "$MANIFEST" \
  --max-sequence-length 8192 \
  --max-new-tokens 4500 \
  --eval-examples 32 \
  --eval-max-new-tokens 4184 \
  --problems-per-step 4 \
  --rollouts-per-problem 16 \
  --retained-rollouts-per-problem 4 \
  --bottom-rollouts-per-problem 4 \
  --training-microbatch-size 4 \
  --sample-logical-tasks-without-replacement \
  --steps 48 \
  --learning-rate 1e-6 \
  --lr-schedule cosine \
  --warmup-steps 2 \
  --weight-decay 0.005 \
  --loss-reduction "$LOSS_REDUCTION" \
  --dense-token-reward \
  --pair-correctness-weight "$PAIR_WEIGHT" \
  --structured-f1-weight "$STRUCTURED_F1_WEIGHT" \
  --eos-structure-weight "$EOS_STRUCTURE_WEIGHT" \
  --duplicate-coordinate-weight "$DUPLICATE_COORDINATE_WEIGHT" \
  --eos-missing-penalty "$EOS_MISSING_PENALTY" \
  --eos-duplicate-penalty "$EOS_DUPLICATE_PENALTY" \
  --eos-invalid-penalty "$EOS_INVALID_PENALTY" \
  --eos-malformed-penalty "$EOS_MALFORMED_PENALTY" \
  --no-eos-penalty "$NO_EOS_PENALTY" \
  --ranking-pair-f1-weight "$RANKING_PAIR_F1_WEIGHT" \
  --ranking-coordinate-coverage-weight "$RANKING_COORDINATE_COVERAGE_WEIGHT" \
  --ranking-duplicate-coordinate-penalty "$RANKING_DUPLICATE_COORDINATE_PENALTY" \
  --ranking-invalid-coordinate-penalty "$RANKING_INVALID_COORDINATE_PENALTY" \
  --supervised-anchor-weight 0 \
  --gradient-checkpointing \
  --eval-selection pair_f1 \
  --min-pair-f1-improvement 0.005 \
  --eval-every 4 \
  --eval-patience 3 \
  --min-steps-before-early-stop 12 \
  --eval-detail \
  --sampled-eval-rollouts 4 \
  --sampled-eval-temperature 1.0 \
  --vllm-max-model-len 10496 \
  --vllm-max-num-seqs 64 \
  --vllm-gpu-memory-utilization 0.25 \
  --seed 3407 \
  --run-name "$RUN_NAME" \
  --log-file "logs/${RUN_NAME}.jsonl" \
  2>&1 | tee "logs/${RUN_NAME}.console.log"

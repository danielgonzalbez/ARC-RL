#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# The first segment stopped immediately after its step-15 evaluation and wrote
# this selected checkpoint. Override CHECKPOINT or STEPS when continuing a
# later segment.
checkpoint="${CHECKPOINT:-checkpoints/rl_qwen35_arc_long8192_pair_v1_best/model.safetensors}"
steps="${STEPS:-105}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="${RUN_NAME:-rl_qwen35_arc_long8192_pair_v1_resume_${timestamp}}"

if [[ ! -f "$checkpoint" ]]; then
  printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
  exit 1
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
# vLLM otherwise enables verbose Triton autotuning diagnostics through its JIT
# monitor. These settings hide setup chatter but preserve runner summaries and
# errors. They do not disable autotuning or its disk cache.
export TRITON_PRINT_AUTOTUNING=0
export VLLM_LOGGING_LEVEL=ERROR
export WANDB_SILENT=true
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore:tl.make_block_ptr is deprecated:UserWarning}"

exec "${ARC_PYTHON:-.venv/bin/python}" -u -m arc_rl.rl_qwen35_native_vllm \
  --tokenizer-mode arc \
  --checkpoint "$checkpoint" \
  --output-checkpoint checkpoints/rl_qwen35_arc_long8192_pair_v1_best \
  --data-dir minimal-arc-main \
  --rl-train-source training-unseen-long \
  --augment-unseen-long \
  --max-sequence-length 8192 \
  --min-train-sequence-length 2187 \
  --pretraining-max-sequence-length 2186 \
  --pretraining-max-target-tokens 2186 \
  --num-color-swaps 2 \
  --max-new-tokens 4500 \
  --vllm-max-model-len 10496 \
  --vllm-max-num-seqs 32 \
  --vllm-gpu-memory-utilization 0.15 \
  --problems-per-step 4 \
  --rollouts-per-problem 8 \
  --training-microbatch-size 8 \
  --dense-token-reward \
  --pair-correctness-weight 0.25 \
  --learning-rate 2e-7 \
  --supervised-anchor-weight 0.4 \
  --curriculum-steps 0 \
  --gradient-checkpointing \
  --steps "$steps" \
  --eval-examples 150 \
  --eval-every 15 \
  --eval-patience 3 \
  --min-steps-before-early-stop 30 \
  --run-name "$run_name" \
  --log-file "logs/${run_name}.jsonl"

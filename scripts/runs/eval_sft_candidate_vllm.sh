#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s CHECKPOINT RUN_NAME\n' "$0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

checkpoint="$1"
run_name="$2"
python_bin="${ARC_PYTHON:-.venv/bin/python}"
partition="data_splits/qwen35_arc_full8192_sft60_rl40_seed3407.json"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_PRINT_AUTOTUNING=0
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_WORK_DIR="${VLLM_WORK_DIR:-/tmp/arc_vllm_work}"
export WANDB_SILENT=true
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "$python_bin" -u -m arc_rl.rl_qwen35_native_vllm \
  --eval-only \
  --tokenizer-mode arc \
  --checkpoint "$checkpoint" \
  --data-dir minimal-arc-main \
  --rl-train-source training-partition \
  --augment-unseen-long \
  --train-task-partition "$partition" \
  --train-partition-role rl \
  --max-sequence-length 8192 \
  --max-new-tokens 4500 \
  --eval-examples 150 \
  --eval-max-sequence-length 2816 \
  --eval-max-new-tokens 2228 \
  --vllm-max-model-len 10496 \
  --vllm-max-num-seqs 64 \
  --vllm-gpu-memory-utilization 0.70 \
  --vllm-enforce-eager \
  --run-name "$run_name" \
  --log-file "logs/${run_name}.jsonl"

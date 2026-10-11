#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
RUN_NAME="${RUN_NAME:-qwen35_4b_arc_rslora_r16_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/logs/$RUN_NAME.jsonl}"
CONSOLE_LOG="${CONSOLE_LOG:-$ROOT_DIR/logs/$RUN_NAME.console.log}"
PARTITION="${PARTITION:-$ROOT_DIR/data_splits/qwen35_arc_full8192_sft60_rl40_seed3407.json}"

MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2816}"
EPOCHS="${EPOCHS:-6}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"

mkdir -p "$OUTPUT_DIR" "$(dirname -- "$LOG_FILE")" "$(dirname -- "$CONSOLE_LOG")"

# expandable_segments helps avoid allocator fragmentation on a nearly full GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" -m arc_rl.pretrain_qwen35_rslora \
  --model-id Qwen/Qwen3.5-4B-Base \
  --output-dir "$OUTPUT_DIR" \
  --train-task-partition "$PARTITION" \
  --train-partition-role sft \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --bucket-size 32 \
  --pad-to-multiple-of 64 \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --eval-examples 32 \
  --eval-max-sequence-length "$MAX_SEQUENCE_LENGTH" \
  --num-color-swaps 4 \
  --loss-scope full \
  --learning-rate 1e-4 \
  --token-learning-rate 3e-4 \
  --weight-decay 0.01 \
  --warmup-fraction 0.05 \
  --max-grad-norm 1.0 \
  --early-stop-patience 2 \
  --min-eval-improvement 1e-4 \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout 0.05 \
  --seed 3407 \
  --run-name "$RUN_NAME" \
  --log-file "$LOG_FILE" \
  2>&1 | tee "$CONSOLE_LOG"

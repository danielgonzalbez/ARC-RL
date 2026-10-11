# Qwen3.5 ARC-tokenizer full-8,192 SFT, effective batch 16

## Lineage

The canonical data manifest is
`data_splits/qwen35_arc_full8192_sft60_rl40_seed3407.json`. The SFT side has
216 logical task families and 1,296 augmented examples; the disjoint RL side
has 144 families and 864 augmented examples. Evaluation remains the fixed 150
unaugmented official-evaluation examples admitted at 2,816 tokens.

```text
Qwen3.5-0.8B pretrained text body + newly initialized 24-token ARC interface
  -> qwen35_arc_full8192_sft60_bs16_v2_best.safetensors
     selected epoch 4 / optimizer update 324
     fixed-150 vLLM: NLL 0.053251, exact 2/150,
     precision 0.559231, recall 0.679183, mean length 522.9, EOS 0.9667
  -> qwen35_arc_full8192_sft60_bs16_v2_cont2_*.safetensors
     two-epoch low-LR branch from the selected epoch-4 weights
```

The first run continued internally through epoch 6, but only the best epoch-4
weights were saved. Epochs 5 and 6 reached evaluation NLL 0.053994 and
0.057656. No optimizer/scheduler state or epoch-6 model weights exist, so the
two-epoch follow-up is a fresh optimizer branch from epoch 4, not an exact
epoch-7/8 resume.

## Initial run

The run used microbatch 16, accumulation 1, full causal loss, LR 1.5e-5,
weight decay 0.1, and cosine decay. It peaked at 38.04 GiB reserved VRAM. Its
best epoch reduced held-out NLL substantially relative to the effective-batch
64 predecessor (0.053251 versus 0.069721) while using the same data split.

## Pre-continuation training-partition audit

Before launching the continuation, the selected epoch-4 checkpoint is greedily
evaluated with vLLM on all 1,296 examples from the SFT partition, including the
same rotations and color permutations used during supervised training. Results
are written to
`logs/qwen35_arc_full8192_sft60_bs16_v2_train_eval_vllm.jsonl`.

| NLL | BPC | Exact | Pair precision | Pair recall | Mean length | EOS |
|---:|---:|---:|---:|---:|---:|---:|
| 0.011681 | 0.016852 | 317/1,296 | 0.738705 | 0.841271 | 834.7 | 0.978 |

The checkpoint has learned the supervised partition substantially better than
the fixed evaluation cohort (train/eval recall 0.8413/0.6792), but it has not
simply memorized the augmented rows: greedy exact accuracy is 24.46% even on
the SFT data. The remaining gap means extra SFT could still fit training rows,
while the rising held-out NLL after epoch 4 warns that doing so may worsen
generalization.

## Continuation artifact policy

- The original epoch-4 checkpoint remains the incumbent best.
- `qwen35_arc_full8192_sft60_bs16_v2_cont2_best.safetensors` is created only if
  a continuation epoch beats the epoch-4 baseline NLL.
- `qwen35_arc_full8192_sft60_bs16_v2_cont2_latest.safetensors` always records
  the terminal continuation weights.
- Obsolete vLLM export directories and the dominated effective-batch-64 model
  may be removed only after their log-to-checkpoint mappings are recorded.

The dominated effective-batch-64 checkpoint
`qwen35_arc_full8192_sft60_v1_best.safetensors` was removed after recording its
epoch-7 mapping and fixed-150 metrics in
`docs/qwen35_arc_long8192_sft_rl_split.md` and `docs/checkpoint_log_mapping.md`.
Its training and evaluation logs remain. Generated `vllm_work` exports were
also removed because they are reproducible copies, not source checkpoints.

Continuation and post-continuation metrics will be appended after completion.

## Two-epoch low-LR branch result

The branch used effective batch 16, full causal loss, peak LR 3e-6, 5% warmup,
cosine decay, and weight decay 0.1. It started from the selected epoch-4
checkpoint whose independently reloaded baseline NLL was 0.053295.

| Branch epoch | Local update | Train loss | Eval NLL |
|---:|---:|---:|---:|
| 1 | 81 | 0.069437 | 0.053373 |
| 2 | 162 | 0.059534 | 0.058489 |

Neither epoch beat the parent, so no `cont2_best` file was created. The
incumbent best remains `qwen35_arc_full8192_sft60_bs16_v2_best.safetensors`.
The terminal epoch-2 state is retained as
`qwen35_arc_full8192_sft60_bs16_v2_cont2_latest.safetensors` for audit and a
post-run vLLM comparison. This branch confirms that additional conventional
SFT continues to fit the training distribution but does not recover held-out
NLL.

### Order-independent vLLM comparison

NLL is auxiliary for this task because it penalizes an otherwise correct pair
set emitted in a different sequence order. Final model selection therefore
also compares greedy, order-independent set metrics:

| Checkpoint | Exact | Precision | Recall | Pair F1 | Mean length | EOS |
|---|---:|---:|---:|---:|---:|---:|
| Parent epoch 4 | 2/150 | 0.559231 | **0.679183** | 0.613398 | 522.9 | 0.967 |
| Branch terminal | 2/150 | **0.620151** | 0.664008 | **0.641331** | **454.5** | **0.993** |

The terminal checkpoint is not uniformly worse: it gives up 1.52 recall
points but gains 6.09 precision points, improves pair F1 by 2.79 points, emits
shorter answers, and terminates more reliably. The checkpoints form a Pareto
tradeoff, so both are retained. Use the parent when target-pair coverage is the
priority and the terminal branch when balanced pair F1/precision is the
priority. The terminal vLLM trace is
`logs/qwen35_arc_full8192_sft60_bs16_v2_cont2_latest_eval_vllm.jsonl`.

Future SFT runs should retain every epoch candidate until vLLM evaluation is
complete. Selecting and deleting epochs solely by teacher-forced NLL can lose
a better order-independent policy.

## Color-token-NLL continuation search

The next continuation search uses teacher-forced color-token NLL for cheap
per-epoch ranking and early stopping. A color token is the value digit
immediately following `:` in each serialized `row-column:value` pair. This
excludes coordinate digits, punctuation, spaces, and EOS from the selection
loss, so arbitrary pair order contributes less directly to the stopping
signal. Total target NLL remains logged as a diagnostic.

Color-token NLL is not a replacement for generated evaluation: it conditions
on the golden prefix and therefore cannot detect omitted/extra coordinates,
repetition, or failure to emit EOS. Each run retains its five lowest
color-token-NLL epoch checkpoints plus the terminal checkpoint. vLLM is used
only after supervised training finishes to compare those retained candidates
on the fixed 150 tasks by exact accuracy, pair precision, recall, F1, length,
and EOS rate.

Two update-budget-matched branches start from the same epoch-4 parent:

- control: original + three rotations + two color permutations, 1,296
  examples for at most eight epochs;
- color augmentation: original + three rotations + four color permutations,
  1,728 examples for at most six epochs.

Both schedules contain 1,296 optimizer updates at batch size 8, peak LR
`2e-6`, weight decay `0.01`, and 5% warmup. Evaluation remains unaugmented and
the fixed SFT/RL logical-task manifest is unchanged. The launchers are
`run_qwen35_arc_full8192_sft60_color_nll_control.sh` and
`run_qwen35_arc_full8192_sft60_color_nll_aug4.sh`.

### Result

Both branches were stopped manually after epoch 2 because color-token NLL
worsened sharply. This follows the intended stopping rule rather than waiting
for the launchers' conservative five-evaluation safety patience. Consequently,
each run has two valid completed epoch checkpoints rather than five; both are
retained, and the epoch-2 file is also hard-linked as the terminal checkpoint.
One uncheckpointed partial epoch was discarded in each branch.

| Branch | Epoch | Update | Target NLL | Color NLL | Exact | Precision | Recall | Pair F1 | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control, 2 colors | 1 | 162 | 0.053134 | **0.304962** | 1/150 | 0.600861 | **0.678599** | 0.637369 | 0.973 |
| control, 2 colors | 2 / last | 324 | 0.059159 | 0.338462 | 2/150 | 0.600538 | 0.672957 | 0.634689 | 0.987 |
| augmented, 4 colors | **1** | 216 | 0.053598 | **0.305516** | 1/150 | **0.630667** | 0.672568 | **0.650944** | **0.993** |
| augmented, 4 colors | 2 / last | 432 | 0.059538 | 0.338953 | 0/150 | 0.594443 | 0.661868 | 0.626346 | 0.980 |

The prior low-LR continuation incumbent had pair F1 `0.641331` with 2/150
exact tasks. Under the requested F1-first SFT criterion, the four-color epoch-1
checkpoint is the new selected SFT state: it gains `0.009613` absolute F1 and
raises precision by `0.10564` relative to the original epoch-4 parent, while
giving up one exact task and `0.006615` recall relative to the low-LR
continuation. The color-NLL increase at epoch 2 correctly predicts generative
degradation in both branches, so it is useful as the inexpensive stopping
signal. Additional color permutations improve the precision/F1 tradeoff at
the first epoch but do not postpone overfitting.

Selected checkpoint:

```text
checkpoints/qwen35_arc_full8192_sft60_color_nll_aug4_best.safetensors
```

Retained epoch candidates and terminal aliases are under:

```text
checkpoints/qwen35_arc_full8192_sft60_color_nll_control_top5/
checkpoints/qwen35_arc_full8192_sft60_color_nll_control_last.safetensors
checkpoints/qwen35_arc_full8192_sft60_color_nll_aug4_top5/
checkpoints/qwen35_arc_full8192_sft60_color_nll_aug4_last.safetensors
```

Training traces are
`logs/qwen35_arc_full8192_sft60_color_nll_control.jsonl` and
`logs/qwen35_arc_full8192_sft60_color_nll_aug4.jsonl`. The four vLLM traces
use the same basename with `_epoch1_eval_vllm` or `_epoch2_eval_vllm`.

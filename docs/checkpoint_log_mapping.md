# Checkpoint and log map

This repository trains compact language models to emit ARC test grids as
unordered `row-column:value` pairs followed by EOS. It first adapts Qwen models
with supervised ARC data and then applies grouped, PPO-style policy-gradient
updates. The environment, reward, and optimizer semantics are documented in
[`rl_environment.md`](rl_environment.md) and [`loss.md`](loss.md).

## Archive layout

The supplied extensionless ZIP has been preserved as `checkpoints.zip`. Its 32
members account for 11 model artifacts (six supervised and five RL). They were
initially extracted to `checkpoints/`; dominated or currently unused extracted
copies were later removed under disk pressure, but remain recoverable from the
ZIP. The latest cleanup removed the extracted native-tokenizer SFT/RL lineage
and disposable `vllm_work` exports to make room for the official pristine
Qwen3.5 base snapshot and the new SFT output. Paths
written into the logs sometimes begin
with the historical absolute prefix `/workspace/workspace/`; in this checkout,
drop that prefix and use the repository-relative path shown below.

A log's `run_start` event records both its input (`checkpoint`, `model_id`, or
`resume_checkpoint`) and intended `output_checkpoint`. A `best_checkpoint`
event identifies the state actually written. For RL runs, the final `result.best`
object identifies the selected evaluation step. Consequently, matching a log
and checkpoint by basename is necessary but not sufficient: several `*_best`
directories contain step 0 because no RL update beat the starting policy.

## Best checkpoint and continuation lineage

The strongest historical checkpoint preserved in `checkpoints.zip` is:

```text
checkpoints/rl_qwen35_native_unseen_long_aug_v1_best/
```

It is Qwen3.5-0.8B with the native tokenizer, selected at RL step 5. On the
fixed 150-example evaluation cohort it has 7 exact tasks, pair recall
0.719358, pair precision 0.658103, golden-token NLL 0.041874, and 0.060412 bits
per target character. Its extracted directory is currently absent but can be
restored from the ZIP; the corrected full-dataset experiment intentionally
does not use it.

Its complete preserved lineage is:

```text
Qwen/Qwen3.5-0.8B
  -> compare_qwen35_pretrained_native_v3 (SFT epoch 3 / step 141)
  -> rl_qwen35_native_arc_v2_best (RL selection remained at step 0)
  -> rl_qwen35_native_unseen_long_v1_best (RL step 30)
  -> rl_qwen35_native_unseen_long_aug_v1_best (RL step 5; current best)
```

The `rl_qwen35_native_arc_v2_best` directory is a text-only exported policy of
the selected step-0 SFT model, not a claim that its 20 RL updates improved the
model. The two later stages trained on long official-training tasks whose
logical IDs were excluded from pretraining; the final stage added rotations
and color permutations.

After re-extraction, another native-tokenizer run can use the directory itself
as `--checkpoint`, including its tokenizer/config files:

```text
--checkpoint checkpoints/rl_qwen35_native_unseen_long_aug_v1_best
```

Do not initialize from the step-25 terminal weights of the augmented run: the
saved directory is the promoted step-5 policy, while step 25 merely triggered
early stopping.

## Archive checkpoint-to-log mapping

The mapping below describes the preserved ZIP. The local copies removed after
review were `compare_qwen3_pretrained_custom_v2`,
`compare_qwen3_scratch_custom`, `compare_qwen35_pretrained_custom`,
`extend_qwen3_pretrained_custom_to_plateau`, and
`rl_qwen35_custom_early_dense_v1_best`. None was the best checkpoint in its
tokenizer family or an input required by the selected continuation.

| Preserved checkpoint | Producing log | Selected state | Role and held-out result |
|---|---|---:|---|
| `checkpoints/compare_qwen3_pretrained_custom_v2.safetensors` | `logs/compare_qwen3_pretrained_custom_v2.jsonl` | SFT epoch 5 / step 235 | Qwen3 + 24-token ARC tokenizer; 0/150 exact, recall 0.534728. |
| `checkpoints/compare_qwen3_scratch_custom.safetensors` | `logs/compare_qwen3_scratch_custom.jsonl` | SFT epoch 5 / step 235 | Scratch control; 0/150 exact, recall 0.013424; not a continuation base. |
| `checkpoints/compare_qwen35_pretrained_custom.safetensors` | `logs/compare_qwen35_pretrained_custom.jsonl` | SFT epoch 4 / step 188 | Early Qwen3.5 + ARC-tokenizer model; 2/150 exact, recall 0.661770. Parent of the dense-RL ablation. |
| `checkpoints/compare_qwen35_pretrained_native_v3/` | `logs/compare_qwen35_pretrained_native_v3.jsonl` | SFT epoch 3 / step 141 | Best supervised native-tokenizer model; corrected reevaluation is in `logs/qwen35_best_final_reeval.jsonl`: 7/150 exact, recall 0.704767, precision 0.660264. |
| `checkpoints/extend_qwen3_pretrained_custom_to_plateau.safetensors` | `logs/extend_qwen3_pretrained_custom_to_plateau.jsonl` | Branch epoch 2 / step 94 | Continuation of `compare_qwen3_pretrained_custom_v2`; 0/150 exact, recall 0.536868. |
| `checkpoints/optimize_qwen35_custom_fresh_full_reg_v2.safetensors` | `logs/optimize_qwen35_custom_fresh_full_reg_v2.jsonl` | SFT epoch 4 / step 372 | Best supervised ARC-tokenizer model; 6/150 exact, recall 0.688521, precision 0.681626. |
| `checkpoints/rl_qwen35_native_arc_v2_best/` | `logs/rl_qwen35_native_arc_v2.jsonl` | RL step 0 | Valid native-tokenizer RL run, but all updates lost to the initial policy; 7/150 exact, recall 0.709241. Parent of unseen-long RL. |
| `checkpoints/rl_qwen35_custom_arc_v3_best/` | `logs/rl_qwen35_custom_arc_v3.jsonl` | RL step 0 | Valid ARC-tokenizer RL run, but all updates lost to `optimize_qwen35_custom_fresh_full_reg_v2`; 6/150 exact, recall 0.689689. |
| `checkpoints/rl_qwen35_custom_early_dense_v1_best/` | `logs/rl_qwen35_custom_early_dense_v1.jsonl` | RL step 30 | Dense-token-reward ablation from the earlier custom SFT model; 2/150 exact, recall 0.674416. |
| `checkpoints/rl_qwen35_native_unseen_long_v1_best/` | `logs/rl_qwen35_native_unseen_long_v1.jsonl` | RL step 30 | First disjoint unseen-long stage; 7/150 exact, recall 0.717121, NLL 0.041904. |
| `checkpoints/rl_qwen35_native_unseen_long_aug_v1_best/` | `logs/rl_qwen35_native_unseen_long_aug_v1.jsonl` | RL step 5 | Augmented unseen-long continuation and current best; 7/150 exact, recall 0.719358, NLL 0.041874. |

The two step-0 RL artifacts are still useful lineage boundaries: they are
runner-ready, text-only exports with the exact tokenizer/configuration expected
by `arc_rl.rl_qwen35_native_vllm`.

## Logs without a checkpoint in the supplied archive

These logs remain useful evidence, but their intended output path is absent
from `checkpoints.zip`.

| Log(s) | Input or relationship | Why there is no preserved output here |
|---|---|---|
| `compare_qwen3_pretrained_custom.jsonl` | Qwen3 base | Aborted during dataset setup; superseded by `compare_qwen3_pretrained_custom_v2`. |
| `compare_qwen35_pretrained_native.jsonl`, `compare_qwen35_pretrained_native_v2.jsonl` | Qwen3.5 base | Incomplete early attempts; superseded by the completed `v3` run. |
| `optimize_qwen35_custom_fresh_full_reg.jsonl` | Qwen3.5 base | Incomplete first attempt (ends after epoch 2); superseded by `v2`. |
| `extend_qwen35_native_to_plateau.jsonl` | `compare_qwen35_pretrained_native_v3` | Completed continuation, but never beat its input, so the epoch-3 SFT checkpoint remained selected. |
| `optimize_qwen35_custom_stage2_full_reg.jsonl` | `compare_qwen35_pretrained_custom.safetensors` | Losing stage-2 continuation. The experiment report says this checkpoint was deleted after disk pressure; its log was retained. |
| `optimized_qwen35_native_target_reg.jsonl` | Qwen3.5 base | Losing target-only ablation. The experiment report says this checkpoint was deleted after disk pressure; its log was retained. |
| `qwen35_best_final_reeval.jsonl` | Reads `compare_qwen35_pretrained_native_v3` | Reevaluation-only run; it corrects pair-precision accounting and does not produce a new model. |
| `rl_qwen35_native_arc_v1.jsonl`, `rl_qwen35_custom_arc_v1.jsonl` | Native SFT / optimized custom SFT | Diagnostic runs affected by identical sibling rollout seeds; `custom_arc_v1` also ends before a result. Superseded by native `v2` and custom `v3`. |
| `rl_qwen35_native_smoke_v1.jsonl`, `rl_qwen35_native_smoke_v2.jsonl`, `rl_qwen35_native_smoke_v3.jsonl`, `rl_qwen35_native_smoke_v4.jsonl`, `rl_qwen35_native_smoke_v5.jsonl`, `rl_qwen35_native_smoke_v6.jsonl`, `rl_qwen35_native_smoke_v7.jsonl` | `compare_qwen35_pretrained_native_v3` | Startup and one/two-step integration smokes; no smoke artifact was included. Versions 1--5 are incomplete traces. |
| `rl_qwen35_native_diversity_smoke.jsonl`, `rl_qwen35_native_diversity_smoke_v2.jsonl`, `rl_qwen35_native_diversity_smoke_v3.jsonl` | `compare_qwen35_pretrained_native_v3` | Two-step diversity/seed diagnostics; not model-selection runs. |
| `rl_qwen35_custom_smoke_v1.jsonl` | `optimize_qwen35_custom_fresh_full_reg_v2.safetensors` | Two-step ARC-tokenizer smoke; not preserved. |
| `rl_qwen35_custom_early_dense_smoke.jsonl` | `compare_qwen35_pretrained_custom.safetensors` | Two-step dense-reward smoke preceding the preserved full run. |
| `rl_qwen35_native_unseen_long_smoke_v1.jsonl` | `rl_qwen35_native_arc_v2_best` | Two-step long-context smoke preceding the preserved full run. |
| `rl_qwen35_native_unseen_long_aug_mb2_probe.jsonl`, `rl_qwen35_native_unseen_long_aug_mb4_probe.jsonl` | `rl_qwen35_native_unseen_long_v1_best` | One-step memory probes used to select training microbatch 4; neither is a model candidate. |

## Other checkpoint-like file

`checkpoints_run_vs5edt1s_model.safetensors` is outside the archive and has no
matching JSONL in `logs/`. It is the legacy small ARC policy used as the
default starting checkpoint by the older vLLM/static-cache runner. It is unrelated to the
Qwen3.5 checkpoint lineage selected for further solver work.

## Selection caveat for the next iteration

The reported 150-example cohort has repeatedly served as validation for epoch,
hyperparameter, and RL-checkpoint selection. Continue from the current best,
but reserve a separate untouched final cohort. The next run should preserve
the native tokenizer, disjoint unseen-long source, supervised anchor, rollout
diversity checks, and exact-first selection semantics unless the experiment is
explicitly testing one of those choices.

## Active ARC-tokenizer continuation

The historical `run_qwen35_arc_long8192_pair_v1.sh` launcher
starts from `checkpoints/optimize_qwen35_custom_fresh_full_reg_v2.safetensors`,
writes its trace to `logs/rl_qwen35_arc_long8192_pair_v1.jsonl`, and promotes
selected states to `checkpoints/rl_qwen35_arc_long8192_pair_v1_best/`. This run
uses the custom tokenizer because the auxiliary exact-pair reward must be
restricted to numeric token positions. Its 1,038 augmented examples represent
173 logical training tasks, have total lengths 2,201--8,041, and have zero
logical-task overlap with the data admitted by the 2,186-token pretraining
boundary. The fixed evaluation mapping remains the same 150 examples used by
the prior Qwen3.5 runs.

The first segment stopped after its step-15 evaluation. It improved pair recall
from 0.685700 to 0.688911 and NLL from 0.050809 to 0.050725 while retaining
6/150 exact tasks, so its selected step-15 weights were written to the output
directory. [`resume_qwen35_arc_long8192_pair_v1.sh`](../scripts/runs/resume_qwen35_arc_long8192_pair_v1.sh)
continues from that `model.safetensors` for the remaining 105-update ceiling.
It creates a timestamped run name and log so restart attempts cannot collide
with the original vLLM work directory or append to the original trace.

That resume was stopped after update 34 when evaluation precision, EOS rate,
and rollout length showed coordinated degradation. Its selected resumed-step-15
weights remain in `rl_qwen35_arc_long8192_pair_v1_best`. The successor maps:

```text
checkpoints/rl_qwen35_arc_long8192_pair_v1_best/model.safetensors
  -> logs/rl_qwen35_arc_long8192_pair_curriculum_v2.jsonl
  -> checkpoints/rl_qwen35_arc_long8192_pair_curriculum_v2_best/
```

The v2 rationale and configuration were recorded separately in the original
experiment workspace.

The v2 run completed normally through update 45 and stopped after its second
consecutive evaluation miss. Its selected checkpoint is update 15, not the
terminal update: 6/150 exact, 0.675899 precision, 0.693191 recall, 0.050708
validation NLL, 431.6 mean output tokens, and 1.000 EOS rate. The directory
`checkpoints/rl_qwen35_arc_long8192_pair_curriculum_v2_best` therefore contains
the update-15 weights corresponding to
`logs/rl_qwen35_arc_long8192_pair_curriculum_v2.jsonl`.

## Long-context 60% SFT / 40% RL branch

The corrected staged branch initializes a new ARC-tokenizer interface on the
pretrained Qwen3.5 body; it does not resume an ARC checkpoint. Its explicit
whole-<=8,192-token task split is
`data_splits/qwen35_arc_full8192_sft60_rl40_seed3407.json`. The SFT output and
log are intended to be `checkpoints/qwen35_arc_full8192_sft60_v1_best.safetensors`
and `logs/qwen35_arc_full8192_sft60_v1.jsonl`; the subsequent disjoint-RL output
and log are intended to be `checkpoints/qwen35_arc_full8192_sft60_rl40_v1_best/`
and `logs/qwen35_arc_full8192_sft60_rl40_v1.jsonl`. The earlier long-only
attempt was stopped before any optimizer update and is diagnostic only.

The SFT stage completed at epoch 8 and selected epoch 7. The selected file maps
to `logs/qwen35_arc_full8192_sft60_v1.jsonl` for training/NLL selection and to
`logs/qwen35_arc_full8192_sft60_v1_eval_vllm.jsonl` for its final fixed-150
greedy metrics: 2 exact, 0.403156 pair precision and 0.611284 pair recall. The
incomplete Transformers final-evaluation attempt has no `final` event and must
not be treated as the checkpoint's generative result.

That effective-batch-64 checkpoint was later deleted after the batch-16 run
dominated it in NLL, precision, recall, length, EOS rate, and memory use. Its
training and evaluation logs remain the canonical record.

The replacement lineage is documented in
[`qwen35_arc_full8192_sft60_bs16_v2.md`](qwen35_arc_full8192_sft60_bs16_v2.md):

```text
checkpoints/qwen35_arc_full8192_sft60_bs16_v2_best.safetensors
  selected initial epoch 4 / update 324
  -> checkpoints/qwen35_arc_full8192_sft60_bs16_v2_cont2_latest.safetensors
     two-epoch low-LR terminal branch; no continuation epoch beat parent NLL
```

Both files are intentionally retained. On order-independent fixed-150 vLLM
metrics, the parent has higher recall (0.679183 versus 0.664008), while the
terminal branch has higher precision (0.620151 versus 0.559231), pair F1
(0.641331 versus 0.613398), and EOS rate. The training-partition audit of the
parent is in
`logs/qwen35_arc_full8192_sft60_bs16_v2_train_eval_vllm.jsonl`; the terminal
fixed-150 evaluation is in
`logs/qwen35_arc_full8192_sft60_bs16_v2_cont2_latest_eval_vllm.jsonl`.

The subsequent color-token-NLL search compared two versus four supervised
color permutations at matched optimizer-update budget. Both branches worsened
on color NLL at epoch 2 and were stopped. The F1-selected state is:

```text
checkpoints/qwen35_arc_full8192_sft60_color_nll_aug4_best.safetensors
  -> logs/qwen35_arc_full8192_sft60_color_nll_aug4.jsonl
  -> logs/qwen35_arc_full8192_sft60_color_nll_aug4_epoch1_eval_vllm.jsonl
```

It is the four-color epoch-1 / update-216 model: 1/150 exact, precision
0.630667, recall 0.672568, pair F1 0.650944, and EOS rate 0.993. It improves
F1 over the previous low-LR terminal state (0.641331) but has one fewer exact
task. Both completed epochs from each branch and explicit terminal aliases are
retained; see `qwen35_arc_full8192_sft60_bs16_v2.md` for the full table.

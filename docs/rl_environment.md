# ARC RL environment

This document describes the RL environment implemented in
`arc_rl.rl_static_cache` and `arc_rl.rl_qwen35_native_vllm`: how ARC tasks become
observations, which actions the policy can take, when an episode terminates,
and how rewards are assigned. Sections about the 24-token action space and
dense syntax reward apply only to the ARC-tokenizer path; Qwen3.5's native
tokenizer uses the structured sequence reward documented below.

## Dataset and observations

Training tasks are loaded from:

```text
arc-agi_training_challenges.json
arc-agi_training_solutions.json
```

Only tasks with one test input and one test output are used. Each task is
serialized with the checkpoint's compact character format:

```text
I<train input>O<train output>S ... T<test input>F
```

The matrices use NumPy's array text representation. The symbols mean:

- `I`: training input starts;
- `O`: training output starts;
- `S`: training example separator;
- `T`: test input starts;
- `F`: the policy must start producing the test output.

The target grid is represented as coordinate/value pairs:

```text
0-0:7 0-1:0 0-2:7 ... E
```

`E` is the EOS token. Target pairs are currently constructed in row-major
order, although sequence correctness is order-independent.

Prompts that cannot fit together with the requested generation allowance in
the configured context are skipped. With the default 512-token rollout limit
and 2048-token largest bucket/vLLM context, prompt length must be at most 1536
tokens.

`--num-eval-examples` limits usable examples after prompt, target, and total
length filtering. The loader continues scanning evaluation tasks until it has
that many valid single-test examples or exhausts the split; it does not stop
after inspecting that many source files.

## Action space

With the custom tokenizer, the model chooses one token from the 24-token
vocabulary:

```text
I O T S F [ ] newline space - : 0 1 2 3 4 5 6 7 8 9 E pad unused
```

The intended completion language uses only:

```text
digits, hyphen, colon, space, EOS
```

Other tokens remain possible actions, but the dense reward treats them as
invalid output characters.

With Qwen3.5's native tokenizer, actions instead come from its full pretrained
vocabulary. Decoded completions are still interpreted as the same coordinate-
pair language, but syntax roles cannot be assigned reliably per subword token.

The legacy runner samples directly from the policy with:

```python
do_sample=True
temperature=1.0
top_p=1.0
```

There is no exploration bonus or external reference policy.

The Qwen3.5 experiments use temperature 1.2 and top-p 0.98, independent sibling
seeds, and adaptive hotter retries only if rollout diversity is insufficient.
Retry temperature is capped by `--max-rollout-temperature` and never exceeds
vLLM's accepted maximum of 2.0.

## Episode termination

An episode terminates when either:

1. the policy generates EOS (`E`); or
2. `max_new_tokens` is reached.

The default rollout allowance is 512 tokens. The unseen-long Qwen3.5 runs
explicitly increase it to 2,048 and use an 8,192-token vLLM context; training
examples are admitted at 2,817--6,144 serialized ARC characters. Generated
tensors are trimmed immediately after the first EOS, including that EOS token.

## Dense token reward

The environment maintains a small parser state:

- whether the policy is writing a row number;
- whether it is writing a column number;
- whether it is writing a predicted value;
- whether a malformed fragment has put the parser in a `wrong` state;
- whether the current coordinate is invalid;
- whether EOS has already occurred.

Most valid structural or correct-content actions receive `+1`. Malformed or
incorrect actions normally receive `-1`. A severely out-of-range coordinate
can receive `-2`.

### EOS

EOS receives:

```text
+1 if the predicted and target pair sets are equal
-1 otherwise
```

Set equality makes pair order irrelevant but rejects missing pairs, additional
incorrect pairs and malformed fragments parsed as extra pairs.

Tokens after EOS would receive zero, although normal generation is trimmed at
the first EOS.

### Invalid characters

A token outside digits, `-`, `:`, and space receives `-1` and puts the parser
in the wrong state. Examples include `I`, `[`, and newline.

The first output character must be a digit. A different first character
receives `-1`.

Repeated leading-zero forms such as `00` receive `-1`.

### Row and column syntax

The intended coordinate prefix is:

```text
row-column:
```

- A hyphen while writing a row receives `+1` and transitions to the column.
- A hyphen in another state receives `-1`.
- A colon after a valid column receives `+1` and transitions to the value.
- A colon at the start, in a row, or while already writing a value receives
  `-1`.
- In-range row and column digits generally receive `+1`.
- Duplicate coordinates and attempts to repeat an already completed row can
  receive `-1`.
- Out-of-range positions receive `-1`; an extremely large number, more than
  ten times a nonzero relevant dimension, receives `-2`.

### Predicted values

While writing the value after `:`, a digit receives:

- `-1` if that value does not occur anywhere in the true grid;
- `+1` if the recovered `row-column:value` fragment occurs in the target;
- `0` in one invalid-position recovery case;
- otherwise `-1`.

The first check is deliberately only a value-membership test. Passing it does
not prove that the value is correct at the chosen coordinate; the subsequent
full-fragment check handles that distinction.

### Spaces and parser recovery

A space is intended to separate complete coordinate/value pairs.

- A space before a value is complete receives `-1`.
- A space after a correct pair receives `+1` when it is a valid separator and
  otherwise `-1`.
- A space after another syntactically complete but incorrect pair currently
  receives `+1` and resets the parser for the next row.
- When the parser is already in the `wrong` state, the next space receives
  `+1` as a recovery/reset action.
- Other tokens generated while in the wrong state generally receive `0` until
  that reset.

These recovery rewards are part of the inherited reward design. They encourage
the policy to stop extending a malformed pair and resume at the next pair.

## Sequence reward

In addition to dense token rewards, every completion receives either dense
target-pair coverage or a binary sequence reward. The binary form is:

```text
1 if predicted and target coordinate/value pair sets are equal
0 otherwise
```

Comparison removes EOS, splits on whitespace, and uses sets. Pair order and
duplicate pairs do not matter.

Complete correctness uses set equality:

```python
target_pairs == predicted_pairs
```

Pair order and duplicates do not matter, while an additional incorrect pair
does make a completion incorrect. Partial coverage remains a separate dense
training signal rather than the complete-correctness definition.

RL training defaults to a dense sequence score equal to the fraction of target
pairs covered by the prediction. Binary complete coverage remains selectable.
The score is normalized across sibling rollouts for the same ARC problem and
then repeated across valid completion tokens.

## Rollout grouping and generation backends

For every chosen ARC problem, the training loop creates
`rollouts_per_problem` copies of the prompt. Multiple samples are important
because reward normalization needs variation to produce useful advantages. The
default statistical batch is four problems times eight rollouts, or 32
requests.

Problem selection defaults to a shuffled cycle over every loaded augmented
example. Each example is selected once before the order is reshuffled for the
next cycle. Independent random sampling remains available, but the shuffled
cycle gives sustained runs predictable coverage of the full filtered dataset.
Length-curriculum sampling sorts examples by combined prompt and golden-target
length, samples from a growing prefix that begins at the configured shortest
fraction, and linearly expands eligibility to the whole dataset. Each update
logs eligible example count and maximum length.

The Qwen3.5 runner additionally supports
`--rl-train-source training-unseen-long`. It reconstructs the original
pretraining admission set, strips rotation/color suffixes to recover logical
task IDs, and selects only longer base examples from the official training
split whose logical ID was never admitted to pretraining. These RL examples are
unaugmented by default. `--augment-unseen-long` enables the original plus
rotations and the configured color permutations, but selection remains grouped
by logical task: an update samples distinct base IDs first and only then picks
one admissible variant for each. This prevents near-duplicate variants from
occupying separate problem groups in one update. The runner asserts zero
logical overlap with both the reconstructed pretraining IDs and the complete
official evaluation split, then logs the full train/evaluation ID lists and
length ranges. `--min-train-sequence-length`, `--max-sequence-length`, and
`--pretraining-max-sequence-length` make the split boundary explicit and
reproducible.

Every backend validates:

```text
required capacity = prompt length + max_new_tokens
```

The Transformers backend selects the smallest static-cache bucket that can hold
that capacity. Rollouts assigned to the same bucket are generated together, and
the last microbatch is padded with dummy rows to preserve the fixed batch
dimension. A static cache is keyed by:

```text
(fixed rollout batch size, cache length bucket)
```

The allocation is reused, but cache contents are reset before every independent
microbatch.

The default vLLM backend receives raw prompt token IDs, stops on token 21, and
returns each sampled token ID plus its chosen-action log-probability. Its
context length, GPU-memory allocation, and active-sequence limit are
configurable; the defaults are 2048 tokens, 10%, and 96 sequences. After an optimizer update, BF16 policy weights are
atomically published and reloaded by the persistent vLLM workers. Prefix
caching is disabled so K/V values from older weights cannot be reused.

## One RL iteration

One training iteration performs:

```text
select ARC problems
    -> repeat prompts for multiple rollouts
    -> sample completions with vLLM or static-cache generation
    -> calculate token and sequence rewards
    -> calculate per-problem reward baselines and standard deviations
    -> reject a population-level long-rollout/no-EOS collapse
    -> score sampled actions in bounded differentiable microbatches
    -> calculate the clipped policy loss with token- or sequence-level reduction
    -> accumulate BF16 gradients and clip their norm
    -> update FP32 AdamW master weights and copy them to BF16
    -> reload persistent vLLM workers when selected
```

The rollout log-probabilities are retained as the old-policy probabilities.
With vLLM, the generator owns a small second BF16 inference copy and KV cache;
the training process owns BF16 working weights plus FP32 master weights and
optimizer state. The configured phases coexist within the documented GPU
memory allocation.

Before an optimizer update, the default collapse guard aborts when mean rollout
length is at least 90% of `max_new_tokens` and fewer than 25% of rollouts
contain EOS. Individual long rollouts remain valid; the guard targets a batch-
wide loss of termination behavior.

When `--save-best-eval` is selected, the initial evaluation establishes the
promotion baseline. By default, policies are ordered by exact-solution count,
then target-pair recall, then golden-token NLL. Recall must exceed the previous
best by the configured threshold (default 0.002 absolute) to count as a gain;
within that tolerance lower NLL breaks the tie. `exact_nll` selects the former
exact-count/NLL ordering. The final policy is therefore not automatically
allowed to replace a better early checkpoint.

The Qwen3.5 runner can instead use `--eval-selection pair_f1`. In that mode,
pair F1 is the primary promotion and early-stopping metric (with its own
minimum-improvement threshold), followed by exact count and golden-token NLL
only inside the F1 tolerance. The 60/40 F1-oriented continuation generates 40
vLLM siblings per problem and retains the best eight *distinct token
sequences* by structured reward. Reward baselines and standard deviations are
computed over the complete 40-rollout sibling group before filtering, so
retained candidates remain normalized against the full sampled population.
If a problem yields fewer distinct candidates than requested, the complete
rollout batch is retried at a higher temperature. After the configured retries
are exhausted, training continues instead of aborting. The JSONL trace records
`rollout_diversity_warning` with the per-group unique counts. When top/bottom
filtering cannot fill its quota with distinct token sequences, it fills the
remaining slots with the best or worst remaining rollout rows and records
`rollout_selection_duplicate_warning`. Repeated sequences can therefore carry
extra weight, but the exception is explicit and auditable.

For partitioned RL, `--sample-logical-tasks-without-replacement` defines an
epoch over the manifest's logical task families. Every family is visited once
before reshuffling, and one of its original/rotation/color views is selected
when visited. With four problems per update, the 144 RL families require 36
updates per complete pass. The original continuation disabled early stopping
until update 72 (two complete passes). The later full-split `v9` experiment
permits stopping from update 36 onward, but never before every logical family
has been visited.

Small environment diagnostics can use `--diagnostic-example-split`. The JSON
manifest names exact augmented official-training examples for train and eval;
the fixed official evaluation cohort is not loaded into the returned eval
dataset. This mode is intended to test whether rewards and updates improve
held-out transformations of a few known task families before committing to a
long disjoint-split experiment. Train/eval example IDs must be non-empty,
disjoint, and available after the configured logical-task partition filter.
`--diagnostic-allow-overlap` relaxes only the disjointness check for an explicit
memorization test in which the exact same example is trained and evaluated. It
must not be used as evidence of held-out generalization.
When an exact diagnostic manifest is present, length filtering only has to
retain the examples named by that manifest; unrelated logical families from
the enclosing partition may be ineligible.

### Staged diagnostic protocol

RL changes should pass the following stages in order. A failure at an earlier
stage is investigated there instead of being hidden by a larger or less
controlled dataset:

1. **One exact view.** Train and evaluate the identical example using
   `--diagnostic-allow-overlap`. This is a memorization/capacity test for the
   reward, gradient, optimizer, BF16 update, weight synchronization, and vLLM
   evaluation path. It is not a generalization result.
2. **The same four exact views.** Train and evaluate a fixed four-example set,
   chosen to span short through long targets. This adds cross-example
   interference while keeping train and evaluation inputs identical.
3. **Held-out transformations of four task families.** Train on selected
   rotations/color permutations and evaluate disjoint exact views from those
   same logical families. This tests transformation generalization, not
   generalization to unseen ARC tasks.
4. **Untouched 150-task evaluation cohort.** Only configurations that show a
   clear improvement in the controlled stages are evaluated on the fixed
   official cohort. This cohort remains outside diagnostic training and is the
   comparison that supports claims about the real experiment.

Greedy F1 alone is discontinuous and can change when the output switches
between sequence modes. Each diagnostic evaluation should therefore combine
the aggregate `eval` record with these JSONL events:

- `eval_example` records deterministic greedy length/EOS, exactness, pair
  precision/recall/F1, covered and omitted target pairs, incorrect and
  duplicate pairs, and a prediction hash/preview. The hash makes a discrete
  output change explicit.
- `sampled_eval_example` records mean and best F1, exact rollout count, mean
  length, and EOS fraction for one task.
- `sampled_eval` aggregates the sampled pass into micro precision/recall/F1,
  macro mean/best F1, exact `pass@k`, structured reward, length, and EOS rate.
  Sample seeds change with the training step, so small step-to-step changes are
  estimates rather than paired deterministic comparisons.

`--no-save-checkpoints` suppresses both the initial checkpoint copy and later
best-policy writes while retaining `best_eval` events. It should be used for
metric-only diagnostics, especially when several branches share a constrained
workspace. A checkpoint must be enabled explicitly for any branch whose
weights need to be reused.

The advancement criteria are stage-specific. Stage 1 must produce a
deterministic exact solution and a nonzero sampled exact `pass@k`. From Stage 2
onward, the minimum material change is `+0.01` absolute greedy pair F1 from the
stage baseline, with no regression in exact count or EOS behavior, corroborated
by either sampled macro-mean F1 or exact `pass@k`. When multiple configurations
pass, preference goes to the one that reaches the criterion earlier, preserves
it through the final evaluation, and has the stronger sampled distribution.
The untouched 150-task cohort is not opened until the held-out-transformation
stage also passes this rule.

### Completed diagnostic results

Stage 1 used one identical train/evaluation view (`e179c5f4`) for 30 updates.
Both variants started at greedy F1 `0.72`, exact `0/1`:

| Objective | First greedy exact | Best greedy | Final greedy | Final sampled evaluation |
| --- | ---: | ---: | ---: | --- |
| Dense control, `structured_f1_weight=0` | step 6 | F1 `1.0`, `1/1` exact | F1 `0.80`, `0/1` exact | micro F1 `0.8696`, `2/8` exact |
| Structured F1, `structured_f1_weight=1` | step 5 | F1 `1.0`, `1/1` exact | F1 `1.0`, `1/1` exact | micro F1 `0.9697`, `4/8` exact |

The control remained greedy-exact from steps 6--28 but switched at steps
29--30 to an overlong output with recall `1.0` and precision `0.6667`. The
structured-F1 branch remained greedy-exact through step 30. Thus both pass the
capacity test, while the structured term reaches it one update earlier and is
more stable at the end.

Stage 2 trained and evaluated the same four exact views, spanning target
lengths 18--3,928, for 16 updates. The shared baseline was micro F1 `0.486411`,
macro F1 `0.458753`, exact `0/4`, and EOS fraction `0.75`.

| Objective | First/best step | Greedy micro F1 | Greedy macro F1 | Exact | EOS | Sampled micro F1 at step 16 | Sampled `pass@4` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense control, `structured_f1_weight=0` | 16 | `0.899148` | `0.912536` | `2/4` | `1.0` | `0.810800` | `0.25` |
| Structured F1, `structured_f1_weight=1` | 10 | `0.899148` | `0.912536` | `2/4` | `1.0` | `0.892406` | `0.50` |

The structured-F1 branch reached the same final greedy solution six updates
earlier and held it through step 16. At that final sampled evaluation it also
had macro-mean F1 `0.769055`, macro best-of-four F1 `0.958897`, and `4/16`
exact rollouts, versus `0.609602`, `0.708897`, and `2/16` for the control. Both
branches exceed the Stage 2 threshold, but these speed, stability, and sampled
results select `structured_f1_weight=1` for the next stage. This is a controlled
diagnostic choice, not yet evidence of held-out ARC improvement.

Stage 3 trained on 24 augmented views from four task families and evaluated on
eight disjoint recolorings of those same families. The baseline was micro F1
`0.738739`, macro F1 `0.578237`, exact `0/8`, and EOS fraction `0.75`.

| Objective | Best step | Best micro F1 | Best macro F1 | Final micro F1 | Final macro F1 | Final exact/EOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense control, `structured_f1_weight=0` | 18 | `0.827458` | `0.661877` | `0.814441` | `0.811976` | `2/8`, `1.0` |
| Structured F1, `structured_f1_weight=1` | 14 | `0.824490` | `0.598723` | `0.806827` | `0.759660` | `2/8`, `1.0` |

Both branches pass the held-out-transformation gate by a wide margin. The
control has the slightly higher best micro F1, while the structured branch has
the stronger final sampled micro F1 (`0.816742` versus `0.798317`). This stage
tests equivariance within four known families and does not select a deployable
checkpoint by itself.

Stage 4 evaluated the saved Stage 3 checkpoints on the untouched fixed cohort
of 150 official tasks, using deterministic vLLM generation with the same
`12,288`-token context setting for all three policies:

| Policy | Precision | Recall | Micro F1 | Delta F1 | Macro F1 | Exact | EOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical SFT baseline | `0.678761` | `0.724023` | `0.700662` | - | `0.628351` | `3/150` | `149/150` |
| Stage 3 dense control | `0.666113` | `0.726215` | `0.694867` | `-0.005795` | `0.630762` | `3/150` | `148/150` |
| Stage 3 structured F1 | `0.661590` | `0.728789` | `0.693566` | `-0.007096` | `0.632562` | `3/150` | `148/150` |

The diagnostic policies trade precision for recall and make small macro-F1
gains, but both regress on the primary micro-F1 criterion and neither improves
exact accuracy. They are therefore not promoted. Passing Stage 3 is a useful
environment/update check, not evidence that the fixed150 distribution will
improve. The next experiment returns to all 144 RL families and uses a milder
mixture (`pair_correctness_weight=0.75`, `structured_f1_weight=0.25`, golden
anchor `0.02`) to counter the observed over-generation.

### Diagnostic artifacts and storage

The reproducibility artifacts are:

| Stage/branch | Launcher | Structured log |
| --- | --- | --- |
| Stage 1 dense | `run_qwen35_rl_diagnostic_exact_control_v4.sh` | `logs/qwen35_rl_diagnostic_exact_control_v4b.jsonl` |
| Stage 1 structured F1 | `run_qwen35_rl_diagnostic_exact_f1_v5.sh` | `logs/qwen35_rl_diagnostic_exact_f1_v5.jsonl` |
| Stage 2 both branches | `run_qwen35_rl_diagnostic_exact_4tasks.sh` | `logs/qwen35_rl_diagnostic_exact_4tasks_{control,f1}_v6.jsonl` |
| Stage 3 held-out transformations | `run_qwen35_rl_diagnostic_aug_holdout.sh` | `logs/qwen35_rl_diagnostic_aug_holdout_{control,f1}_v7.jsonl` |
| Stage 4 fixed150 | `run_qwen35_rl_fixed150_eval.sh` | `logs/qwen35_rl_fixed150_{base,aug_control,aug_f1}_v8.jsonl` |
| Full 144-family RL | `run_qwen35_rl_full_aug_top8_mix_v9.sh` | `logs/qwen35_arc_full8192_rl_full_aug_top8_mix_v9.jsonl` |

Each structured log has a same-stem `.console.log`. Stage 1 and Stage 2 were
metric-only and used `--no-save-checkpoints`; their apparent output checkpoint
paths in `run_start` are therefore not reusable models. The original Stage 1
attempt terminated at step 3 while saving with `Disk quota exceeded`; its
partial `v4` logs remain for audit, while the completed rerun is `v4b`.

Checkpoint cleanup removed six obsolete or duplicate RL/diagnostic model
copies, freeing approximately 6 GB, without deleting their logs or launchers.
The canonical start point remains
`checkpoints/qwen35_arc_full8192_sft60_rl40_f1_v1_best/model.safetensors`; the
recovered long-run step-38 artifact also remains at
`checkpoints/qwen35_arc_full8192_sft60_rl40_f1_v1_cont1_step038_recovered.safetensors`.
The reusable Stage 3 checkpoints are
`checkpoints/qwen35_rl_diagnostic_aug_holdout_control_v7_best/model.safetensors`
and
`checkpoints/qwen35_rl_diagnostic_aug_holdout_f1_v7_best/model.safetensors`.

Long runs can set evaluation patience together with a minimum number of
updates. Patience counts consecutive evaluations that fail to improve the
configured selection criterion and cannot stop training before the
configured minimum. This makes early stopping depend on held-out behavior,
not the non-stationary scalar policy loss.

Qwen3.5 evaluation is hybrid. Golden-token NLL and bits per character are
computed with teacher forcing by the differentiable Transformers model. Exact
accuracy, pair precision/recall, length, and EOS rate are computed from a greedy
vLLM rollout pass with `do_sample=False`. This permits the full long-context
evaluation cohort to be used without constructing one sequence-length-padded
tensor for the entire cohort.

Diagnostics can additionally log greedy metrics for every example and sample
multiple vLLM completions per evaluation prompt. The sampled summary reports
mean and best pair F1, exact `pass@k`, length, EOS rate, and structured reward.
Per-example records include omissions, incorrect unique pairs, duplicates, and
an output hash so discrete greedy changes are visible without storing every
long completion in the JSONL log.
Short diagnostics that only compare metrics can use `--no-save-checkpoints` to
avoid writing redundant initial and promoted model copies; best-evaluation
events are still recorded in JSONL.

Supervised ARC-tokenizer searches may additionally log color-token NLL: the
teacher-forced NLL of only the value digit immediately following each `:` in
the target pair serialization. It is useful for epoch ranking because it
excludes coordinate/order syntax, but it does not measure autonomous sequence
construction. Retained SFT candidates must therefore still be compared with
greedy vLLM exact and pair metrics after training.

The augmented experiment evaluated every five updates to guard against the
degradation seen in earlier runs. That interval is only 0.48 nominal logical
epochs for 84 tasks and eight selected tasks per update. Future sustained runs
should evaluate the full cohort every 10--11 updates, use roughly three such
misses for patience, and reserve a separate untouched final cohort to limit
checkpoint-selection bias.

Evaluation defaults to the ARC evaluation split. `--eval-split training` is
available only as an explicit memorization/capacity diagnostic: it evaluates
unaugmented training tasks and must never be reported as held-out
generalization.

## Current environment limitations

- Sequence success uses exact-set equality: output order is ignored, but an
  extra unique output pair makes the sequence incorrect.
- Dense pair correctness uses string-fragment membership in the serialized
  target, rather than parsing both outputs into canonical dictionaries.
- Group normalization produces no signal when every sibling rollout has the
  same reward; adequate rollouts per problem remain essential.
- A rollout that reaches the configured token limit without EOS is valid but expensive,
  especially during autoregressive generation.

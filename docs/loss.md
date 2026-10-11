# RL loss

This document describes the shared clipped policy-gradient loss used by
`arc_rl.rl_static_cache` and `arc_rl.rl_qwen35_native_vllm`. It is inspired by
PPO. There is no learned value model, entropy bonus, or KL penalty. The reward
path and several defaults differ between the legacy 24-token model, Qwen3.5
with the ARC tokenizer, and Qwen3.5 with its native tokenizer; those differences
are called out below.

## Inputs to the loss

For each generated completion token, the training loop stores:

- `rollout_logprobs`: the log-probability assigned when the action was sampled;
- `current_logprobs`: the log-probability assigned by the differentiable
  training forward pass;
- a dense ARC token reward when the 24-token tokenizer path is enabled;
- a coverage/binary sequence reward, or the native-tokenizer structured score;
- a mask identifying real completion tokens rather than padding.

The legacy runner uses `temperature=1.0` and `top_p=1.0`. The sustained Qwen3.5
runs use temperature 1.2 and top-p 0.98, with a distinct deterministic seed per
sibling rollout and hotter retry temperatures if diversity is too low. vLLM's
sampled-action log-probabilities remain the old policy; they are not substituted
with values from the differentiable model. The script logs mean and maximum
absolute disagreement between backends and aborts by default if the mean
exceeds 0.1. Small nonzero differences are expected from distinct BF16
attention kernels.


## Causal-logit alignment

Suppose a sequence consists of a prompt of length `P` followed by completion
tokens `y[0], y[1], ...`:

```text
[prompt[0], ..., prompt[P-1], y[0], y[1], ...]
```

A causal-model logit at position `i` predicts the token at position `i + 1`.
The log-probability of `y[j]` must therefore be read from position:

```text
P - 1 + j
```

The implementation is:

```python
positions = (
    torch.arange(completion.numel(), device=device)
    + prompt.numel()
    - 1
)

selected = log_distributions[row, positions].gather(
    -1,
    completion.unsqueeze(-1),
).squeeze(-1)
```

Right padding is excluded by the attention mask and by a separate completion
mask. A generated token with ID zero is still a real action and is not mistaken
for padding.

## Token-reward statistics

This section applies to the 24-token ARC tokenizer. Native-tokenizer Qwen3.5
cannot reliably classify subword tokens into these syntax roles, so its token
reward tensor is zero and its learning signal comes from the structured
sequence score described next.

Every output token is classified into one of seven roles:

1. space;
2. hyphen;
3. colon;
4. row number;
5. column number;
6. predicted grid value;
7. EOS.

For each role `k`, each ARC problem rollout group supplies a mean and population
standard deviation over the valid tokens of that role:

```text
mu_k    = mean(r_i for tokens i with type k)
sigma_k = population_std(r_i for tokens i with type k)
```

The token advantage is:

```text
A_token_i = (r_i - mu_type(i)) / (sigma_type(i) + 1e-8)
```

If all rewards of a type are equal, its standard deviation is zero and its
normalized advantage is zero. That token type contributes no policy-gradient
signal for that batch.

## Sequence-reward statistics

For the 24-token path, the sequence reward is target-pair coverage in `[0, 1]`;
binary complete coverage remains selectable. Native-tokenizer Qwen3.5 instead
uses:

```text
score = 0.75 * pair_F1 + 0.25 * exact_set_match
        - 0.05 * missing_EOS
```

Its precision denominator counts whitespace-delimited candidate fragments,
while recall counts unique target pairs covered. Duplicating a correct pair
therefore lowers F1, although exact-set equality itself ignores duplicates.
Statistics are computed once over the rollout scores for each ARC problem and
broadcast to valid token positions:

```text
A_sequence_i = (R_sequence_i - mu_sequence) / (sigma_sequence + 1e-8)
```

This is a per-rollout, per-problem baseline. Completion length does not change
the sequence-reward mean or standard deviation, and one ARC problem cannot
change the advantages of another problem.

## Combined advantage

The two advantages are mixed using `sequence_weight`:

```text
A_i = (1 - sequence_weight) * A_token_i
      + sequence_weight * A_sequence_i
```

In code:

```python
advantages = (
    (1.0 - sequence_weight) * token_advantages
    + sequence_weight * sequence_advantages
).detach()
advantages = advantages * mask
```

Advantages are detached because rewards and batch statistics are treated as
fixed targets, not differentiable model outputs.

The native-tokenizer Qwen3.5 runs use `sequence_weight=1.0`, because their token
advantage is zero. The dense ARC-tokenizer ablation used `sequence_weight=0.25`
so that token-level credit supplied 75% of the combined advantage.

### Numeric exact-pair objective

The long-context ARC-tokenizer continuation replaces the broadcast sequence
term with an exact-pair term. Each whitespace-delimited predicted fragment is
checked against the target `row-column:value` set. Its binary correctness is
assigned only to digit tokens belonging to that fragment. Spaces, hyphens,
colons, EOS, and invalid nonnumeric tokens receive no pair-correctness credit;
they continue to learn from the dense syntax/token reward.

For this mode the advantages are additive rather than a convex sequence mix:

```text
A_i = A_token_i + pair_correctness_weight * A_pair_i * is_number_i
```

The experiment uses `pair_correctness_weight=0.25`, so individual token/syntax
credit has unit weight and exact-pair credit is smaller. Both token-role
statistics and exact-pair statistics are computed separately inside each
sibling-rollout group for one ARC problem. No reward mean or standard
deviation is shared between unrelated problems in the outer rollout batch.

The 60/40 F1-oriented RL run raises this coefficient to `0.50`. Its continuation
generates 40 siblings per problem, computes token and pair baselines over all
40, then retains the eight distinct token sequences with the highest structured
reward for the differentiable policy update. Computing advantages only inside
the selected top eight would recenter the winners and assign negative relative
advantages to some of them. Keeping the full sampling population as the
reference preserves the intended best-of-40 signal while deduplication avoids
overweighting repeated generations.

The later full-split `v9` run retains the same full-40 normalization and
best-eight distinct selection, but uses `pair_correctness_weight=0.75` and
`structured_f1_weight=0.25`. This is a precision-oriented response to the
fixed150 diagnostic result: a stronger F1 term raised recall while reducing
precision and primary micro F1.

The parallel contrast ablation keeps that same 40-rollout reference but trains
on both the eight best and eight worst distinct sequences. The bottom eight
retain their negative full-population advantages, providing explicit
suppression of poor generations. It differs from recomputing statistics over
the selected 16, which would change the reference distribution and weaken the
controlled comparison with the top-eight-only branch.

Distinctness is preferred rather than an unconditional runtime requirement.
Current runs retry an under-diverse population at a higher temperature. If the
retry budget is exhausted, top/bottom selection fills any remaining quota with
repeated token sequences in ranking order and emits explicit diversity and
selection-warning events. Advantage statistics still come from the full
pre-filter sibling population.

### Optional structured-F1 advantage

Diagnostics can add `--structured-f1-weight`. Pair F1 is computed independently
for every sampled completion, normalized across the complete sibling group,
and broadcast over the real tokens in that completion before top/bottom
filtering:

```text
A_i = A_token_i + lambda_pair * A_pair_i * is_number_i
      + lambda_f1 * A_f1_rollout
```

The default `lambda_f1` is zero and therefore preserves previous runs. Unlike
numeric pair credit, this rollout-level term can reward coverage and penalize
omissions even though an omitted pair has no generated token of its own.

The staged exact-view diagnostics selected `structured_f1_weight=1.0` for the
held-out-transformation test. On one exact view, it reached deterministic
exactness at update 5 instead of update 6 and remained exact through update 30,
whereas the zero-weight control regressed to F1 `0.80`. On the four-view test,
both objectives reached greedy micro F1 `0.899148` and `2/4` exact tasks, but
the structured branch reached that state at update 10 rather than update 16.
At update 16 its sampled micro F1/pass@4 were `0.892406`/`0.50`, compared with
`0.810800`/`0.25` for the control. On held-out recolorings, however, the dense
control obtained a slightly higher best micro F1 (`0.827458` versus
`0.824490`). On fixed150, both saved policies regressed from the canonical
baseline `0.700662`: control reached `0.694867` and structured F1 `0.693566`.
Thus `1.0` was useful for the capacity tests but is not the selected weight for
the full-split run. The CLI default remains zero for backward compatibility.

## Policy ratio and clipping

For each valid action:

```text
ratio_i = exp(current_logprob_i - rollout_logprob_i)
```

The log-ratio is first limited to `[-20, 20]` for numerical safety. The PPO
surrogate objective, with default `clip_epsilon=0.2`, is:

```text
unclipped_i = ratio_i * A_i
clipped_i   = clamp(ratio_i, 0.8, 1.2) * A_i
objective_i = min(unclipped_i, clipped_i)
```

By default, the final loss is the negative mean over every valid generated
token (`--loss-reduction token`):

```text
loss = -sum(objective_i * mask_i) / max(sum(mask_i), 1)
```

The denominator counts valid actions, not positive rewards. Negative-reward
actions therefore contribute normally, and a batch without positive rewards
does not divide by zero.

`--loss-reduction sequence` instead averages within every retained rollout and
then averages those rollout losses:

```text
loss_i = -sum(objective_i,t * mask_i,t) / valid_tokens_i
loss   = sum(loss_i) / number_of_nonempty_rollouts
```

Every retained rollout then has equal total weight regardless of its generated
length. Microbatch accumulation is weighted by the number of nonempty rollouts
rather than valid tokens, so changing `training_microbatch_size` does not
change the objective. Reward and advantage normalization are unchanged. The
default remains `token` for backward compatibility with completed runs.

## Optional supervised anchor

Sustained RL runs can add a golden-target negative-log-likelihood anchor over
the unique ARC problems selected for the update. Each problem is scored once,
not once per sibling rollout, in bounded anchor microbatches:

```text
loss_total = loss_policy + anchor_weight * loss_golden_nll
```

The anchor does not change rollout rewards or advantages. It regularizes the
same policy parameters toward valid target completions so a dense reward proxy
cannot freely erode the pretrained distribution over a long run. Its default
weight is zero in the legacy runner and 0.1 in the Qwen3.5 runner; the
unseen-long experiments explicitly used 0.4. Both component losses and the
combined loss are logged.

The 60% SFT / 40% RL split experiment sets the RL anchor weight to zero. This
is intentional: teacher-forcing golden targets from the nominally RL-only 40%
would turn that stage into mixed SFT/RL and invalidate the staged comparison.
The subsequent full-split `v9` optimization intentionally uses a small `0.02`
anchor as a regularizer; it is a different experiment and must not be compared
as a pure RL-only continuation.

## Optimizer update

One rollout batch receives one optimizer update, but its differentiable scoring
is split into training microbatches. The legacy default is 16 sequences and the
Qwen3.5 default is 2; the verified augmented unseen-long run used microbatch 4
for an effective rollout batch of 32 and peaked at 45.87 GiB on an A100-80GB.
Reward baselines and standard deviations are computed once over the complete
statistical batch.
With token reduction, each microbatch mean loss is multiplied by its valid-token
count divided by the full valid-token count before backward. With sequence
reduction, the corresponding weight is its nonempty-rollout count divided by
the full nonempty-rollout count. The accumulated gradient is therefore
identical to the selected full-batch objective, up to floating-point order:

```python
optimizer.zero_grad(set_to_none=True)
for microbatch in rollout_batch:
    current_logprobs = current_policy_logprobs(model, prompts, completions)
    loss, metrics = compute_clipped_policy_loss(...)
    (loss * microbatch_reduction_units / total_reduction_units).backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Policy parameters and their gradients are BF16. `FP32MasterAdamW` converts
the completed gradients to FP32, updates FP32 master weights with FP32 first and
second moments, then copies the updated values back to the BF16 working model.
The legacy default learning rate is `1e-5`; Qwen3.5 defaults to `2e-6`, while
the augmented unseen-long continuation explicitly used `2e-7`. Weight decay is
`0.01`. The parameter-wise BF16 copy used by forward/backward is distinct from
the FP32 master copy used for the update.

The CLI can keep that rate constant or apply linear warmup followed by cosine
decay to a configurable fraction of the peak rate. The rate is updated before
each rollout/update and is recorded in every local training event. The cosine
horizon is the configured maximum update count; evaluation patience may end a
run before the end of that horizon.

The legacy runner also logs the absolute and relative L2 norm of the effective BF16
parameter change and the fraction of working parameters whose BF16 value
changed. These are measured after the FP32 optimizer step and before copying
the new BF16 values into the working model. The log separately reports mean
absolute token, sequence and combined advantages, plus the fraction of nonzero
combined advantages. Together with the pre-clipping gradient norm, these
metrics distinguish a near-zero scalar policy loss from a genuinely
signal-free or BF16-quantized-away update.

The Qwen3.5 diagnostic runner can probe representative early, middle, late, and
final-normalization parameters. At the configured interval it separately
computes dense-token, weighted numeric-pair, raw structured-F1, and combined
gradient norms and pairwise cosine similarities on the first training
microbatch. It also logs the relative BF16 parameter update and changed-value
fraction for those probes. These measurements are attribution diagnostics, not
a replacement for the full accumulated gradient used by the optimizer.

The corresponding JSONL event is `gradient_diagnostic`. Its component names
have the following precise meanings:

- `token` is the dense ARC-token advantage alone;
- `pair_weighted` includes `pair_correctness_weight` and applies only where the
  numeric-pair mask is active;
- `structured_f1_raw` is the normalized rollout-F1 direction before applying
  `structured_f1_weight`;
- `combined` is the actual policy loss for that microbatch, including the
  configured pair and structured-F1 coefficients.

For every component, `probe_grad_norm` and `parameter_grad_norms` report norms
over the selected probe tensors. `cosines` reports directional agreement: a
positive value means two objectives tend to update the probes in the same
direction, a negative value exposes conflict, and a value near zero indicates
weak alignment. The `update` block is measured after the optimizer step and
reports the relative L2 change and fraction of BF16 values that actually
changed, both overall across the probes and per tensor.

These are deliberately bounded diagnostics: they use the first retained
training microbatch and representative tensors rather than every parameter.
They can distinguish missing signal, objective conflict, clipping, and an
update too small to survive BF16 quantization, but they do not by themselves
prove that the full update improves behavior. In particular, the current event
does not compute a golden-target NLL gradient; teacher-forced NLL remains an
evaluation metric unless the supervised anchor is enabled.

## Interpreting the first update

The rollout and differentiable forward use synchronized model weights. Before
the optimizer step, their ratio should therefore be approximately one. vLLM
and Transformers can still differ slightly because they use distinct BF16
attention implementations:

```text
ratio_i ~= 1
```

Normalized advantages also sum to approximately zero. The displayed scalar
loss can consequently be near zero while gradients remain nonzero:

```text
gradient = -sum(A_i * gradient(log pi(action_i)))
```

Different actions have different log-probability gradients, so this weighted
gradient need not vanish even when the scalar advantages sum to zero.

With exactly one optimizer update per rollout batch, clipping is normally
inactive because the ratio starts at one. PPO clipping becomes consequential
if the same rollout batch is reused for multiple policy epochs or if minibatch
updates change the policy before all stored rollouts have been consumed.

## Evaluation and interpreting reward trends

Qwen3.5 evaluation is deliberately hybrid. Golden-token NLL and bits per
character are computed by the differentiable Transformers policy with teacher
forcing. Exact tasks, pair precision/recall, output length, and EOS rate come
from deterministic greedy vLLM generation (`do_sample=False`) after the latest
weights have been synchronized.

For diagnostics, `eval_example` explains the aggregate greedy result with
per-task omissions, incorrect/duplicate pairs, EOS, output length, and a stable
output hash. `sampled_eval` measures distributional movement with mean and
best-of-k pair F1 and exact `pass@k`; it can improve before the greedy argmax
changes. Conversely, a one-step greedy F1 jump without a corresponding sampled
trend may be a mode switch rather than stable progress. Sampled seeds are
step-dependent, so this event is interpreted as a noisy distributional probe,
not as a paired evaluation on identical random draws.

Improvement on held-out transformations of a few training families is not a
surrogate for improvement on fixed150. In the four-family diagnostic, both
objectives improved held-out recoloring micro F1 by more than `0.08`, yet their
saved checkpoints reduced fixed150 micro F1 by `0.005795` and `0.007096`.
Promotion therefore always uses the untouched fixed cohort; the smaller stages
validate mechanics, capacity, and equivariance only.

Training reward is not a train-set loss curve: every update samples different
tasks and variants, and within-problem normalization makes the scalar PPO loss
hovering near zero normal. The augmented unseen-long run showed a weak positive
reward trend (0.3706 over its first five updates versus 0.4000 over its last
five), accompanied by higher sampled pair precision/recall, stable EOS, shorter
outputs, and no growth in correct-pair duplicates on evaluation. This is not
evidence of reward hacking, but random batches prevent a paired before/after
claim. A fixed greedy train diagnostic is required for that stronger test.

# 120M accumulated pretraining and RL

The 120M preset has 120,478,912 parameters. `--gradient-accumulation-steps`
is an alias for `--minibatches`; effective batch size is microbatch size times
accumulation steps. Each update is the mean over all supervised tokens in its
accumulation group, including a partial final group. Empty padding rows have
ignored labels and never duplicate training examples. The scheduler advances
once per optimizer update. FP32 weights/AdamW state use BF16 CUDA autocast.

Epoch selection uses held-out target NLL. `--skip-greedy-eval` allows sweeps to
measure baseline/final NLL without expensive random-policy decoding; it makes
no claims about generated solutions. The selected checkpoint is reloaded in
BF16 for final NLL. RL subsequently evaluates generated solutions.

Checkpoints saved by `save_model` now include a sibling `.config.json` file.
RL strict loading and vLLM export read this architecture; old checkpoints
without a sidecar retain the original small-model default. Keep the sidecar
with the weights when moving a checkpoint. Token IDs, rewards, causal scoring,
and the vLLM RoPE bridge remain as documented in the environment/loss docs.

## Experiment protocol

Start each candidate from the same seed (3407), train for 15 epochs with all
1,476 eligible augmented examples at total context 2816, and evaluate all 150
eligible untouched evaluation examples (requested limit 400). Same dataset,
augmentation, full-sequence objective and cosine schedule across candidates.
Compare effective batch 32 at LR 3e-5 and 1e-4, then effective batch 64 at
LR 1e-4. Microbatch 4 initially, with a memory-tested alternative if needed.
Select the lowest final BF16 held-out NLL among candidate best checkpoints.
These are best among tested configurations, not a global optimum. Evaluation
is used for tuning, so results are validation estimates, not an untouched test.

Commands, failures, epoch curves and RL outcomes are recorded in
`logs/accum120m/` and `docs/experiments_120m.txt`.

## Initial RL follow-up

`run_120m_rl.py` reads the completed ranking and starts the winner with vLLM.
The planned run has 100 updates, evaluation every 20, four problems with eight
rollouts each, scoring microbatch two, LR 1e-6 with warmup/cosine decay, and
the existing supervised anchor at weight 0.5. Context 5632 accommodates the
same 2816-total-token validation examples with 2816 rollout tokens. The vLLM
allocation is 20%, with 32 concurrent sequences and CUDA graphs enabled.
This first run used the original exact-first/NLL-second evaluation rule and
collapse/parity guards. Patience was three evaluations with at least 60
updates. The later search below replaces model selection with exact count,
pair recall and NLL, in that order.

## Measured pretraining results

All candidates completed 15 epochs. NLL below is measured after BF16 reload
of each candidate's best checkpoint on the same 150 validation examples.

| Candidate | Microbatch | Accumulation | Effective batch | Peak LR | Best epoch | BF16 NLL |
| --- | --- | --- | --- | --- | --- | --- |
| A | 4 | 8 | 32 | 3e-5 | 14 | 0.350571 |
| B | 4 | 8 | 32 | 1e-4 | 13 | 0.252273 |
| C | 2 | 32 | 64 | 1e-4 | 15 | 0.328059 |

B is the selected configuration among those tested. Its epochs 14 and 15
were worse than epoch 13, despite declining training loss. A and C also had
isolated upward validation transitions but recovered. These curves show why
checkpoint selection matters; they do not establish sustained overfitting for
every configuration. The raw curves and commands remain in the experiment logs.

## Initial measured RL result

The run stopped successfully at step 60 after three non-improving evaluations.
Complete solutions stayed at 1/150. Target-pair coverage increased from
43.05% to 45.06%, but NLL worsened from 0.252200 to 0.260524. The existing
exact-first/NLL-second selection rule retained pretrained B; consequently
`rl_best.safetensors` was not written. Recommended weights remain
`checkpoints/accum120m/b.safetensors` with `b.config.json`.

All 1920 training rollouts terminated with EOS. Backend logprob MAE averaged
0.003843 (worst batch mean 0.006053), and peak allocated/reserved CUDA memory
was 15.66/22.92 GiB. vLLM CUDA graphs, policy weight reloads and the unchanged
reward/loss pipeline ran successfully with the 120M architecture.

## Curriculum RL search

The exact counts below were measured with the then-current target-set inclusion
criterion. Later short-task experiments changed semantic correctness to
unordered equality of predicted and target pair sets; pair recall, precision
and NLL below are unaffected by that historical distinction.

Six 100-update runs then started independently from pretrained B and compared
length curriculum versus uniform sampling, 2x16/4x8/8x4 rollout grouping, a
64-sequence batch, and a conservative LR/anchor variant. Selection prioritizes
exact solutions, then held-out target-pair recall, with NLL as a tie-breaker.

The best tested coverage configuration is curriculum 4x8 at update 80:
1/150 exact, recall 0.462549, precision 0.260072 and NLL 0.253799. It is saved
at `checkpoints/rl_120m_curriculum/d_curriculum_4x8.safetensors` with
its architecture sidecar. Independent reload reproduced NLL 0.253799 and
recall 0.462646. All rollouts retained EOS and lengths decreased, so no policy
collapse was observed.

Use D when pair coverage is the priority. Use pretrained B when golden-token
NLL is the priority: RL gained +0.033269 absolute recall but gave up 0.001600
NLL and did not add an exact solution. Full configurations, trial comparisons
and the loss diagnosis are in `docs/rl_120m_curriculum.txt`; raw JSONL logs are
under `logs/rl_120m_curriculum/`.

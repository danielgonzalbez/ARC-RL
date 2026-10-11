# Repository guidance

This file applies to the entire repository.

Before changing `src/arc_rl/rl_static_cache.py`, read the documentation relevant to the behavior you will touch:

- Read [`docs/rl_environment.md`](docs/rl_environment.md) for changes involving ARC prompt or target serialization, the action space, rollout sampling or termination, token and sequence rewards, rollout grouping, static-cache selection, or the overall RL iteration. Use it when modifying generation backends (including vLLM), because rollout outputs must preserve the environment's token, EOS, reward, and grouping semantics.
- Read [`docs/loss.md`](docs/loss.md) for changes involving rollout/current log-probabilities, causal-logit alignment, masks, reward normalization, advantages, PPO-style ratios or clipping, loss reduction, or optimizer updates. Use it when a generation change affects which log-probabilities are retained or compared during training.
- Read both documents when a change crosses the rollout/training boundary, alters completion representation or padding, or changes data passed from generation into reward and loss computation.

Treat the files under `docs/` as the canonical references. Keep them synchronized with intentional behavior changes; do not infer environment or loss semantics solely from similarly named top-level notes.

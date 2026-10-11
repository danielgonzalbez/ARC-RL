"""Plot the held-out Qwen3.5 RL comparison from JSONL traces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Native tokenizer": ROOT / "logs/rl_qwen35_native_arc_v2.jsonl",
    "ARC tokenizer": ROOT / "logs/rl_qwen35_custom_arc_v3.jsonl",
    "ARC early + dense reward": (
        ROOT / "logs/rl_qwen35_custom_early_dense_v1.jsonl"
    ),
    "Native unseen-long train": (
        ROOT / "logs/rl_qwen35_native_unseen_long_v1.jsonl"
    ),
    "Native unseen-long + augmentation": (
        ROOT / "logs/rl_qwen35_native_unseen_long_aug_v1.jsonl"
    ),
}
OUTPUT = ROOT / "docs/qwen35_rl_tokenizer_comparison.png"


def events(path: Path, event: str) -> list[dict]:
    return [
        row
        for line in path.read_text().splitlines()
        if (row := json.loads(line))["event"] == event
    ]


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("#1769aa", "#d1495b", "#2a9d8f", "#7b2cbf", "#f28e2b")
    for (label, path), color in zip(RUNS.items(), colors):
        evaluation = events(path, "eval")
        training = events(path, "train")
        eval_steps = [row["step"] for row in evaluation]
        train_steps = [row["step"] for row in training]
        axes[0, 0].plot(
            eval_steps, [row["loss"] for row in evaluation],
            marker="o", linewidth=2, label=label, color=color,
        )
        axes[0, 1].plot(
            eval_steps, [row["num_correct"] for row in evaluation],
            marker="o", linewidth=2, label=label, color=color,
        )
        axes[1, 0].plot(
            eval_steps, [row["pair_recall"] for row in evaluation],
            marker="o", linewidth=2, label=f"{label} recall", color=color,
        )
        axes[1, 0].plot(
            eval_steps, [row["pair_precision"] for row in evaluation],
            marker="x", linestyle="--", linewidth=1.5,
            label=f"{label} precision", color=color, alpha=0.8,
        )
        axes[1, 1].plot(
            train_steps, [row["diverse_group_fraction"] for row in training],
            linewidth=1.8, label=f"{label} diverse groups", color=color,
        )
        axes[1, 1].plot(
            train_steps, [row["advantage_nonzero_fraction"] for row in training],
            linestyle="--", linewidth=1.3,
            label=f"{label} nonzero advantages", color=color, alpha=0.75,
        )

    axes[0, 0].set(title="Held-out golden-token NLL", xlabel="RL update", ylabel="NLL")
    axes[0, 1].set(title="Exact ARC tasks (out of 150)", xlabel="RL update", ylabel="Exact tasks")
    axes[1, 0].set(title="Pair metrics", xlabel="RL update", ylabel="Fraction", ylim=(0.60, 0.73))
    axes[1, 1].set(title="Rollout diversity / RL signal", xlabel="RL update", ylabel="Fraction", ylim=(0, 1.05))
    for axis in axes.flat:
        axis.legend(fontsize=8)
    figure.suptitle(
        "Qwen3.5-0.8B ARC RL: tokenizer and train-set ablations", fontsize=15
    )
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()

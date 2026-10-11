"""Plot the native-tokenizer unseen-long augmentation ablation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Unaugmented": ROOT / "logs/rl_qwen35_native_unseen_long_v1.jsonl",
    "Rotations + 2 color swaps": (
        ROOT / "logs/rl_qwen35_native_unseen_long_aug_v1.jsonl"
    ),
}
OUTPUT = ROOT / "docs/qwen35_rl_unseen_long_augmentation.png"


def events(path: Path, event: str) -> list[dict]:
    return [
        row
        for line in path.read_text().splitlines()
        if (row := json.loads(line))["event"] == event
    ]


def rolling_mean(values: list[float], window: int = 5) -> list[float]:
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / min(index + 1, window)
        for index in range(len(values))
    ]


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ("#7b2cbf", "#f28e2b")
    for (label, path), color in zip(RUNS.items(), colors):
        evaluation = events(path, "eval")
        training = events(path, "train")
        eval_steps = [row["step"] for row in evaluation]
        train_steps = [row["step"] for row in training]
        axes[0, 0].plot(
            eval_steps,
            [row["loss"] for row in evaluation],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        axes[0, 1].plot(
            eval_steps,
            [row["pair_recall"] for row in evaluation],
            marker="o",
            linewidth=2,
            label=f"{label}: recall",
            color=color,
        )
        axes[0, 1].plot(
            eval_steps,
            [row["pair_precision"] for row in evaluation],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            label=f"{label}: precision",
            color=color,
            alpha=0.8,
        )
        reward = [row["reward_mean"] for row in training]
        exact = [row["num_correct"] / row["num_samples"] for row in training]
        axes[1, 0].plot(
            train_steps,
            rolling_mean(reward),
            linewidth=2,
            label=f"{label}: reward (5-step mean)",
            color=color,
        )
        axes[1, 0].plot(
            train_steps,
            rolling_mean(exact),
            linestyle="--",
            linewidth=1.5,
            label=f"{label}: exact fraction (5-step mean)",
            color=color,
            alpha=0.8,
        )
        axes[1, 1].plot(
            train_steps,
            [row["advantage_nonzero_fraction"] for row in training],
            linewidth=1.7,
            label=f"{label}: nonzero advantages",
            color=color,
        )
        axes[1, 1].plot(
            train_steps,
            [row["diverse_group_fraction"] for row in training],
            linestyle="--",
            linewidth=1.3,
            label=f"{label}: diverse groups",
            color=color,
            alpha=0.8,
        )

    axes[0, 0].set(title="Held-out golden-token NLL", xlabel="RL update", ylabel="NLL")
    axes[0, 1].set(title="Held-out pair metrics", xlabel="RL update", ylabel="Fraction")
    axes[1, 0].set(title="Training reward / exact rate", xlabel="RL update", ylabel="Fraction")
    axes[1, 1].set(title="Remaining RL signal", xlabel="RL update", ylabel="Fraction", ylim=(0, 1.05))
    for axis in axes.flat:
        axis.legend(fontsize=8)
    figure.suptitle("Qwen3.5-0.8B native ARC RL: unseen-long augmentation", fontsize=15)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()

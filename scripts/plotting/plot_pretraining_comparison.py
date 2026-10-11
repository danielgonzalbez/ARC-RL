"""Plot the three controlled ARC pretraining comparison runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNS = (
    (
        "Qwen3 0.6B pretrained + ARC tokenizer",
        PROJECT_ROOT / "logs/compare_qwen3_pretrained_custom_v2.jsonl",
        True,
        "#3366cc",
    ),
    (
        "Qwen3 0.6B scratch + ARC tokenizer",
        PROJECT_ROOT / "logs/compare_qwen3_scratch_custom.jsonl",
        True,
        "#dc3912",
    ),
    (
        "Qwen3.5 0.8B pretrained + native tokenizer",
        PROJECT_ROOT / "logs/compare_qwen35_pretrained_native_v3.jsonl",
        False,
        "#109618",
    ),
)


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs/qwen_arc_pretraining_comparison.png",
    )
    args = parser.parse_args()

    figure, (train_axis, eval_axis) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, constrained_layout=True
    )
    for label, path, character_tokenizer, color in RUNS:
        records = read_records(path)
        steps = [record for record in records if record["event"] == "step"]
        epochs = [record for record in records if record["event"] == "epoch"]
        baseline = next(record for record in records if record["event"] == "baseline")

        train_frame = pd.DataFrame(steps)
        smoothed = train_frame["train_loss"].rolling(7, min_periods=1).mean()
        train_axis.plot(
            train_frame["optimizer_step"],
            smoothed,
            label=label,
            color=color,
            linewidth=2,
        )

        eval_steps = [0] + [record["optimizer_step"] for record in epochs]
        if character_tokenizer:
            eval_bpc = [baseline["loss"] / math.log(2)] + [
                record["eval_loss"] / math.log(2) for record in epochs
            ]
        else:
            eval_bpc = [baseline["bits_per_character"]] + [
                record["eval_bits_per_character"] for record in epochs
            ]
        eval_axis.plot(
            eval_steps,
            eval_bpc,
            marker="o",
            label=label,
            color=color,
            linewidth=2,
        )

    continuation = read_records(
        PROJECT_ROOT / "logs/extend_qwen35_native_to_plateau.jsonl"
    )
    continuation_steps = [
        record for record in continuation if record["event"] == "step"
    ]
    continuation_epochs = [
        record for record in continuation if record["event"] == "epoch"
    ]
    continuation_baseline = next(
        record for record in continuation if record["event"] == "baseline"
    )
    # The continued checkpoint is epoch 3 / optimizer step 141 of the initial
    # native-tokenizer run. Later initial-run epochs were evaluated but not kept.
    branch_step = 141
    continuation_frame = pd.DataFrame(continuation_steps)
    continuation_smoothed = continuation_frame["train_loss"].rolling(
        7, min_periods=1
    ).mean()
    train_axis.plot(
        branch_step + continuation_frame["optimizer_step"],
        continuation_smoothed,
        label="Qwen3.5 continuation from best step 141",
        color="#990099",
        linestyle="--",
        linewidth=2,
    )
    eval_axis.plot(
        [branch_step]
        + [branch_step + record["optimizer_step"] for record in continuation_epochs],
        [continuation_baseline["bits_per_character"]]
        + [record["eval_bits_per_character"] for record in continuation_epochs],
        marker="x",
        label="Qwen3.5 continuation to plateau",
        color="#990099",
        linestyle="--",
        linewidth=2,
    )

    train_axis.set_title("ARC pretraining: loss by optimizer step")
    train_axis.set_ylabel("Training NLL (7-step moving average)")
    train_axis.grid(alpha=0.25)
    train_axis.legend()
    eval_axis.set_title("Held-out ARC: tokenizer-normalized evaluation")
    eval_axis.set_xlabel("Optimizer step")
    eval_axis.set_ylabel("Bits per target character (lower is better)")
    eval_axis.set_yscale("log")
    eval_axis.grid(alpha=0.25)
    eval_axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

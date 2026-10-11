"""Plot the ARC tokenizer and regularization follow-up experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def records(path: str) -> list[dict]:
    with (PROJECT_ROOT / path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eval_curve(path: str, *, character: bool, offset: int = 0):
    data = records(path)
    baseline = next(row for row in data if row["event"] == "baseline")
    epochs = [row for row in data if row["event"] == "epoch"]
    steps = [offset] + [offset + row["optimizer_step"] for row in epochs]
    if character:
        values = [baseline["loss"] / math.log(2)] + [
            row["eval_loss"] / math.log(2) for row in epochs
        ]
    else:
        values = [baseline["bits_per_character"]] + [
            row["eval_bits_per_character"] for row in epochs
        ]
    return steps, values


def final_record(path: str, event: str = "final") -> dict:
    return next(row for row in records(path) if row["event"] == event)


def main() -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
    comparison, continuations, generative = axes

    comparison_runs = (
        (
            "Qwen3.5 native · full loss · batch 32",
            "logs/compare_qwen35_pretrained_native_v3.jsonl",
            False,
            "#109618",
            "-",
        ),
        (
            "Qwen3.5 native · target loss · batch 16",
            "logs/optimized_qwen35_native_target_reg.jsonl",
            False,
            "#66aa55",
            "--",
        ),
        (
            "Qwen3.5 ARC · full loss · batch 32",
            "logs/compare_qwen35_pretrained_custom.jsonl",
            True,
            "#990099",
            "-",
        ),
        (
            "Qwen3.5 ARC · full loss · batch 16 + WD 0.1",
            "logs/optimize_qwen35_custom_fresh_full_reg_v2.jsonl",
            True,
            "#ff9900",
            "--",
        ),
    )
    for label, path, character, color, style in comparison_runs:
        steps, values = eval_curve(path, character=character)
        comparison.plot(
            steps, values, marker="o", label=label, color=color,
            linestyle=style, linewidth=2,
        )
    comparison.set_title("Fresh pretraining configurations")
    comparison.set_ylabel("Bits per target character (lower is better)")
    comparison.set_yscale("log")
    comparison.grid(alpha=0.25)
    comparison.legend(fontsize=9)

    branch_runs = (
        (
            "Qwen3 ARC initial",
            "logs/compare_qwen3_pretrained_custom_v2.jsonl",
            0,
            "#3366cc",
            "-",
        ),
        (
            "Qwen3 ARC continuation from step 235",
            "logs/extend_qwen3_pretrained_custom_to_plateau.jsonl",
            235,
            "#3366cc",
            "--",
        ),
        (
            "Qwen3.5 ARC initial",
            "logs/compare_qwen35_pretrained_custom.jsonl",
            0,
            "#990099",
            "-",
        ),
        (
            "Qwen3.5 ARC continuation from step 188",
            "logs/optimize_qwen35_custom_stage2_full_reg.jsonl",
            188,
            "#990099",
            "--",
        ),
    )
    for label, path, offset, color, style in branch_runs:
        steps, values = eval_curve(path, character=True, offset=offset)
        continuations.plot(
            steps, values, marker="o", label=label, color=color,
            linestyle=style, linewidth=2,
        )
    continuations.set_title("Continuation branches from retained checkpoints")
    continuations.set_xlabel("Optimizer step along lineage")
    continuations.set_ylabel("Bits per target character (lower is better)")
    continuations.set_yscale("log")
    continuations.grid(alpha=0.25)
    continuations.legend(fontsize=9)

    final_runs = (
        ("Q3 ARC\noriginal", "logs/compare_qwen3_pretrained_custom_v2.jsonl", "final"),
        ("Q3 ARC\nextended", "logs/extend_qwen3_pretrained_custom_to_plateau.jsonl", "final"),
        ("Q3.5 ARC\noriginal", "logs/compare_qwen35_pretrained_custom.jsonl", "final"),
        ("Q3.5 ARC\nstage 2", "logs/optimize_qwen35_custom_stage2_full_reg.jsonl", "final"),
        ("Q3.5 ARC\nfresh reg", "logs/optimize_qwen35_custom_fresh_full_reg_v2.jsonl", "final"),
        ("Q3.5 native\ntarget reg", "logs/optimized_qwen35_native_target_reg.jsonl", "final"),
        ("Q3.5 native\nfull", "logs/qwen35_best_final_reeval.jsonl", "baseline"),
    )
    labels, recalls, precisions, exacts = [], [], [], []
    for label, path, event in final_runs:
        row = final_record(path, event)
        labels.append(label)
        recalls.append(100 * row["pair_recall"])
        precisions.append(100 * row["pair_precision"])
        exacts.append(row["num_correct"])
    positions = np.arange(len(labels))
    width = 0.36
    generative.bar(positions - width / 2, recalls, width, label="Pair recall")
    generative.bar(positions + width / 2, precisions, width, label="Pair precision")
    for position, recall, exact in zip(positions, recalls, exacts):
        generative.text(
            position - width / 2, recall + 1.2, f"exact={exact}",
            ha="center", va="bottom", fontsize=8, rotation=90,
        )
    generative.set_title("Greedy evaluation on the same 150 ARC tasks")
    generative.set_ylabel("Percent")
    generative.set_xticks(positions, labels)
    generative.set_ylim(0, 90)
    generative.grid(axis="y", alpha=0.25)
    generative.legend()

    output = PROJECT_ROOT / "docs/qwen_arc_followup_optimization.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output.resolve())


if __name__ == "__main__":
    main()

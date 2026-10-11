"""Build reproducible few-family ARC RL train/eval manifests.

Training and evaluation use the same logical task families but never the same
serialized view.  By default, train receives the original, three rotations,
and two recolorings; eval receives two independent recolorings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_ANCHORS = ("f8ff0b80", "e179c5f4", "e48d4e1a", "c444b776")


def matrix_to_sequence(matrix: Sequence[Sequence[int]]) -> str:
    pairs = [
        f"{row}-{column}:{value}"
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
    ]
    return " ".join(pairs) + "E"


def format_arc_prompt(challenge: dict) -> str:
    pieces: list[str] = []
    for example in challenge["train"]:
        pieces.extend(
            ("I", str(np.asarray(example["input"])), "O", str(np.asarray(example["output"])), "S")
        )
    pieces.extend(("T", str(np.asarray(challenge["test"][0]["input"])), "F"))
    return "".join(pieces)


def transform_task(challenge: dict, solution: list[list[int]], transform: Callable):
    transformed = {
        "train": [
            {"input": transform(item["input"]), "output": transform(item["output"])}
            for item in challenge["train"]
        ],
        "test": [{"input": transform(item["input"])} for item in challenge["test"]],
    }
    return transformed, transform(solution)


def rotate(grid: Sequence[Sequence[int]], angle: int) -> list[list[int]]:
    return np.rot90(np.asarray(grid), k=angle // 90).tolist()


def recolor(grid: Sequence[Sequence[int]], permutation: Sequence[int]) -> list[list[int]]:
    return [[int(permutation[value]) for value in row] for row in grid]


def serialized_fingerprint(challenge: dict, solution: list[list[int]]) -> str:
    payload = format_arc_prompt(challenge) + "\0" + matrix_to_sequence(solution)
    return hashlib.sha256(payload.encode()).hexdigest()


def task_lengths(challenge: dict, solution: list[list[int]]) -> dict[str, int]:
    prompt = format_arc_prompt(challenge)
    target = matrix_to_sequence(solution)
    return {"prompt": len(prompt), "target": len(target), "total": len(prompt) + len(target)}


def choose_spread_tasks(
    eligible: dict[str, dict[str, int]], count: int, anchors: Sequence[str]
) -> list[str]:
    if count < len(anchors):
        raise ValueError("num_tasks cannot be smaller than the number of anchors")
    missing = set(anchors) - set(eligible)
    if missing:
        raise ValueError(f"anchors are not eligible RL tasks: {sorted(missing)}")
    selected = list(dict.fromkeys(anchors))
    remaining = [task_id for task_id in eligible if task_id not in selected]
    while len(selected) < count:
        # Farthest-point sampling over total serialized length gives coverage of
        # the range without depending on JSON insertion order.
        candidate = max(
            remaining,
            key=lambda task_id: (
                min(
                    abs(eligible[task_id]["total"] - eligible[item]["total"])
                    for item in selected
                ),
                eligible[task_id]["total"],
                task_id,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return sorted(selected, key=lambda task_id: (eligible[task_id]["total"], task_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "minimal-arc-main"
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=PROJECT_ROOT
        / "data_splits/qwen35_arc_full8192_sft60_rl40_seed3407.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-color-swaps", type=int, default=2)
    parser.add_argument("--eval-color-swaps", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--max-target-length", type=int, default=4500)
    parser.add_argument("--anchor-task-ids", nargs="*", default=list(DEFAULT_ANCHORS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_tasks <= 0 or args.train_color_swaps < 0 or args.eval_color_swaps <= 0:
        raise ValueError("task and color-swap counts must be positive")
    challenges = json.loads(
        (args.data_dir / "arc-agi_training_challenges.json").read_text()
    )
    solutions = json.loads(
        (args.data_dir / "arc-agi_training_solutions.json").read_text()
    )
    partition = json.loads(args.partition.read_text())
    allowed = set(partition["rl_task_ids"])
    eligible: dict[str, dict[str, int]] = {}
    for task_id in sorted(allowed & challenges.keys() & solutions.keys()):
        challenge = challenges[task_id]
        solution_rows = solutions[task_id]
        if len(challenge.get("test", ())) != 1 or len(solution_rows) != 1:
            continue
        lengths = task_lengths(challenge, solution_rows[0])
        if lengths["total"] <= args.max_sequence_length and lengths["target"] <= args.max_target_length:
            eligible[task_id] = lengths
    if args.num_tasks > len(eligible):
        raise ValueError(f"requested {args.num_tasks} tasks but only {len(eligible)} are eligible")

    selected = choose_spread_tasks(eligible, args.num_tasks, args.anchor_task_ids)
    train_ids: list[str] = []
    eval_ids: list[str] = []
    train_hashes: dict[str, str] = {}
    eval_hashes: dict[str, str] = {}
    color_count = args.train_color_swaps + args.eval_color_swaps
    permutations = {
        args.seed + index + 1: np.random.default_rng(args.seed + index + 1)
        .permutation(10)
        .tolist()
        for index in range(color_count)
    }
    for task_id in selected:
        challenge = challenges[task_id]
        solution = solutions[task_id][0]
        base_variants = [(task_id, challenge, solution)]
        for angle in (90, 180, 270):
            transformed = transform_task(
                challenge, solution, lambda grid, angle=angle: rotate(grid, angle)
            )
            base_variants.append((f"{task_id}_rotation_{angle}", *transformed))
        for variant_id, variant_challenge, variant_solution in base_variants:
            train_ids.append(variant_id)
            train_hashes[variant_id] = serialized_fingerprint(variant_challenge, variant_solution)
        for index, (color_seed, permutation) in enumerate(permutations.items()):
            transformed = transform_task(
                challenge,
                solution,
                lambda grid, permutation=permutation: recolor(grid, permutation),
            )
            variant_id = f"{task_id}_colors_0_seed_{color_seed}"
            destination_ids, destination_hashes = (
                (train_ids, train_hashes)
                if index < args.train_color_swaps
                else (eval_ids, eval_hashes)
            )
            destination_ids.append(variant_id)
            destination_hashes[variant_id] = serialized_fingerprint(*transformed)

    collisions = set(train_hashes.values()) & set(eval_hashes.values())
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} serialized train/eval views collide; choose different color seeds"
        )
    manifest = {
        "version": 1,
        "description": (
            "Few-family RL diagnostic with original/rotation/train-color views and "
            "disjoint held-out color views from the same logical ARC families."
        ),
        "seed": args.seed,
        "partition": str(args.partition),
        "partition_role": "rl",
        "selection": "fixed anchors plus farthest-point coverage of serialized total length",
        "logical_task_ids": selected,
        "logical_tasks": len(selected),
        "task_lengths": {task_id: eligible[task_id] for task_id in selected},
        "augmentation": {
            "train": f"original+3-rotations+{args.train_color_swaps}-color-swaps",
            "eval": f"{args.eval_color_swaps}-held-out-color-swaps",
            "all_color_seeds": list(permutations),
        },
        "train_example_ids": train_ids,
        "eval_example_ids": eval_ids,
        "train_examples": len(train_ids),
        "eval_examples": len(eval_ids),
        "serialized_train_eval_overlap": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Create the audited 60/40 logical-task split for long Qwen3.5 ARC training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from arc_rl.task_partition import (
    filter_partition_examples,
    length_summary,
    logical_task_id,
    rank_stratified_split,
)
from arc_rl.pretrain_qwen3 import load_pretraining_examples
from arc_rl.rl_qwen35_native_vllm import select_unseen_long_examples
from arc_rl.rl_static_cache import ArcCharacterTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "minimal-arc-main"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sft-fraction", type=float, default=0.6)
    parser.add_argument("--min-sequence-length", type=int, default=2187)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4500)
    parser.add_argument("--pretraining-max-sequence-length", type=int, default=2186)
    parser.add_argument("--pretraining-max-target-tokens", type=int, default=2186)
    parser.add_argument("--num-color-swaps", type=int, default=2)
    parser.add_argument("--no-rotations", action="store_true")
    parser.add_argument(
        "--include-all-training",
        action="store_true",
        help="Split every eligible official-training task instead of excluding prior pretraining IDs.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = ArcCharacterTokenizer()
    candidates, _ = load_pretraining_examples(args, tokenizer)
    if args.include_all_training:
        eligible = [
            example
            for example in candidates
            if example.prompt_ids.numel() + example.target_ids.numel()
            >= args.min_sequence_length
        ]
        seen_pretraining_ids: set[str] = set()
    else:
        pretraining_args = argparse.Namespace(**vars(args))
        pretraining_args.max_sequence_length = args.pretraining_max_sequence_length
        pretraining_args.max_new_tokens = args.pretraining_max_target_tokens
        pretraining = load_pretraining_examples(pretraining_args, tokenizer)[0]
        eligible, seen_pretraining_ids = select_unseen_long_examples(
            candidates,
            pretraining,
            min_total_tokens=args.min_sequence_length,
        )

    by_task: dict[str, list] = defaultdict(list)
    for example in eligible:
        by_task[logical_task_id(example.task_id)].append(example)
    task_lengths = {
        task_id: sum(
            int(item.prompt_ids.numel() + item.target_ids.numel()) for item in variants
        )
        / len(variants)
        for task_id, variants in by_task.items()
    }
    sft_ids, rl_ids = rank_stratified_split(
        task_lengths, sft_fraction=args.sft_fraction, seed=args.seed
    )
    if (set(sft_ids) | set(rl_ids)) & seen_pretraining_ids:
        raise RuntimeError("long split overlaps reconstructed pretraining task families")
    sft_examples = filter_partition_examples(eligible, set(sft_ids))
    rl_examples = filter_partition_examples(eligible, set(rl_ids))
    manifest = {
        "version": 1,
        "description": (
            "Logical-task-disjoint, length-rank-stratified split of every eligible "
            "augmented ARC training example."
            if args.include_all_training
            else
            "Logical-task-disjoint, length-rank-stratified split of augmented "
            "ARC training examples not admitted to <=2186-token pretraining."
        ),
        "include_all_training": args.include_all_training,
        "seed": args.seed,
        "sft_fraction": args.sft_fraction,
        "min_sequence_length": args.min_sequence_length,
        "max_sequence_length": args.max_sequence_length,
        "max_new_tokens": args.max_new_tokens,
        "pretraining_max_sequence_length": args.pretraining_max_sequence_length,
        "pretraining_max_target_tokens": args.pretraining_max_target_tokens,
        "augmentation": f"original+3-rotations+{args.num_color_swaps}-color-swaps",
        "eligible_logical_tasks": len(by_task),
        "eligible_examples": len(eligible),
        "sft_logical_tasks": len(sft_ids),
        "rl_logical_tasks": len(rl_ids),
        "sft_summary": length_summary(sft_examples),
        "rl_summary": length_summary(rl_examples),
        "sft_task_ids": sft_ids,
        "rl_task_ids": rl_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in manifest.items() if not key.endswith("_task_ids")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

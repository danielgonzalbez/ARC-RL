"""Deterministic logical-task partitions for staged ARC SFT and RL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


PARTITION_ROLES = ("sft", "rl")


def logical_task_id(task_id: str) -> str:
    """Collapse rotation and color augmentation suffixes to the ARC task ID."""

    return task_id.split("_rotation_", 1)[0].split("_colors_", 1)[0]


def load_partition_ids(path: Path, role: str) -> tuple[set[str], dict]:
    """Load and validate one role from a persisted task-family partition."""

    if role not in PARTITION_ROLES:
        raise ValueError(f"partition role must be one of {PARTITION_ROLES}, got {role!r}")
    path = path.expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    sft_ids = set(manifest.get("sft_task_ids", ()))
    rl_ids = set(manifest.get("rl_task_ids", ()))
    if not sft_ids or not rl_ids:
        raise ValueError(f"partition must contain non-empty sft_task_ids and rl_task_ids: {path}")
    overlap = sft_ids & rl_ids
    if overlap:
        raise ValueError(f"partition roles overlap for {len(overlap)} logical tasks")
    listed_total = manifest.get("eligible_logical_tasks")
    if listed_total is not None and listed_total != len(sft_ids | rl_ids):
        raise ValueError(
            "partition eligible_logical_tasks does not match the two ID lists: "
            f"{listed_total} != {len(sft_ids | rl_ids)}"
        )
    return (sft_ids if role == "sft" else rl_ids), manifest


def filter_partition_examples(examples: Sequence, allowed_ids: set[str]) -> list:
    """Keep variants belonging to allowed logical tasks, preserving input order."""

    return [item for item in examples if logical_task_id(item.task_id) in allowed_ids]


def rank_stratified_split(
    task_lengths: dict[str, float], *, sft_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split exact task counts while interleaving roles across the length ranking.

    The cumulative-rounding allocation keeps every local region of the sorted
    length distribution close to the requested ratio. A seeded hash only breaks
    equal-length ties, making the manifest stable across Python versions.
    """

    if not task_lengths:
        raise ValueError("cannot split an empty task collection")
    if not 0.0 < sft_fraction < 1.0:
        raise ValueError("sft_fraction must be strictly between zero and one")

    def tie_break(task_id: str) -> str:
        return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()

    ranked = sorted(task_lengths, key=lambda item: (task_lengths[item], tie_break(item)))
    sft_ids: list[str] = []
    rl_ids: list[str] = []
    previous_sft_count = 0
    for rank, task_id in enumerate(ranked, start=1):
        cumulative_sft_count = round(rank * sft_fraction)
        if cumulative_sft_count > previous_sft_count:
            sft_ids.append(task_id)
            previous_sft_count = cumulative_sft_count
        else:
            rl_ids.append(task_id)
    return sorted(sft_ids), sorted(rl_ids)


def length_summary(examples: Iterable) -> dict[str, int | float]:
    lengths = [int(item.prompt_ids.numel() + item.target_ids.numel()) for item in examples]
    if not lengths:
        raise ValueError("cannot summarize an empty example collection")
    return {
        "examples": len(lengths),
        "min_total_length": min(lengths),
        "max_total_length": max(lengths),
        "mean_total_length": sum(lengths) / len(lengths),
    }

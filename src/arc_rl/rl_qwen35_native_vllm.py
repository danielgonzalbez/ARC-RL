"""RL fine-tuning for Qwen3.5 ARC policies with vLLM rollouts.

The runner deliberately keeps rollout generation and policy optimization
separate: vLLM samples raw token IDs and records their old log-probs,
then Transformers recomputes differentiable log-probs for the sampled IDs.
Native and ARC-specific tokenizers are supported. Only the text backbone is
loaded from multimodal Qwen3.5 checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from .task_partition import (
    filter_partition_examples,
    load_partition_ids,
    logical_task_id,
)
from .pretrain_qwen3 import load_pretraining_examples
from .pretrain_qwen35_custom import load_custom_checkpoint
from .pretrain_qwen35_native import (
    NativeARCExample,
    _convert_examples,
    load_datasets,
)
from .rl_static_cache import (
    ARCExample,
    DEFAULT_ARC_DATA,
    PAD_TOKEN_ID,
    ArcCharacterTokenizer,
    ExperimentLogger,
    FP32MasterAdamW,
    RewardBatch,
    build_reward_batch,
    compute_clipped_policy_loss,
    gradient_mean_and_variance,
    load_arc_examples,
    pad_token_logprobs,
)


DEFAULT_CHECKPOINT = Path("checkpoints/compare_qwen35_pretrained_native_v3")
DEFAULT_CUSTOM_CHECKPOINT = Path(
    "checkpoints/optimize_qwen35_custom_fresh_full_reg_v2.safetensors"
)


def select_unseen_long_examples(
    candidates: Sequence[ARCExample],
    pretraining_examples: Sequence[ARCExample],
    *,
    min_total_tokens: int,
) -> tuple[list[ARCExample], set[str]]:
    """Select candidates whose logical task never entered pretraining."""

    seen = {logical_task_id(example.task_id) for example in pretraining_examples}
    selected = [
        example
        for example in candidates
        if logical_task_id(example.task_id) not in seen
        and example.prompt_ids.numel() + example.target_ids.numel()
        >= min_total_tokens
    ]
    return selected, seen


@dataclass(frozen=True)
class NativeRolloutBatch:
    token_ids: list[torch.Tensor]
    token_logprobs: list[torch.Tensor]
    ended_with_eos: list[bool]


@dataclass(frozen=True)
class PairMetrics:
    exact: int
    good_pairs: int
    candidate_pairs: int
    covered_target_pairs: int
    target_pairs: int

    @property
    def precision(self) -> float:
        return self.good_pairs / max(self.candidate_pairs, 1)

    @property
    def recall(self) -> float:
        return self.covered_target_pairs / max(self.target_pairs, 1)

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return 2.0 * precision * recall / max(precision + recall, 1e-12)


@dataclass(frozen=True)
class NativeEvalMetrics:
    step: int
    loss: float
    bits_per_character: float
    num_correct: int
    num_samples: int
    pair_precision: float
    pair_recall: float
    pair_f1: float
    macro_pair_precision: float
    macro_pair_recall: float
    macro_pair_f1: float
    good_pairs: int
    candidate_pairs: int
    covered_target_pairs: int
    target_pairs: int
    mean_length: float
    min_length: int
    max_length: int
    eos_fraction: float


@dataclass(frozen=True)
class NativeStepMetrics:
    step: int
    loss: float
    policy_loss: float
    supervised_anchor_loss: float
    learning_rate: float
    grad_norm: float
    grad_mean: float
    grad_variance: float
    rollout_seconds: float
    rollout_tokens_per_second: float
    weight_sync_seconds: float
    rollout_logprob_mae: float
    rollout_logprob_max_error: float
    reward_mean: float
    reward_std: float
    retained_reward_mean: float
    rollouts_retained_per_problem: int
    top_rollouts_retained_per_problem: int
    bottom_rollouts_retained_per_problem: int
    logical_task_epoch: int
    logical_tasks_seen_in_epoch: int
    token_reward_mean: float
    sequence_reward_mean: float
    pair_correctness_reward_mean: float
    diverse_group_fraction: float
    minimum_unique_rollouts_per_group: int
    rollout_temperature: float
    rollout_resamples: int
    advantage_abs_mean: float
    advantage_nonzero_fraction: float
    structured_f1_advantage_abs_mean: float
    eos_structure_advantage_abs_mean: float
    coordinate_structure_advantage_abs_mean: float
    eos_structure_reward_mean: float
    shown_coordinate_fraction_mean: float
    missing_coordinates_mean: float
    duplicate_coordinates_mean: float
    invalid_coordinates_mean: float
    malformed_candidates_mean: float
    ratio_mean: float
    clip_fraction: float
    num_correct: int
    num_samples: int
    groups_with_exact_rollout: int
    groups_with_exact_rollout_fraction: float
    exact_rollout_pair_advantage_mean: float
    pair_precision: float
    pair_recall: float
    problem_total_length_mean: float
    problem_total_length_min: int
    problem_total_length_max: int
    mean_length: float
    eos_fraction: float
    cuda_peak_allocated_gib: float
    cuda_peak_reserved_gib: float


def _pieces(text: str) -> list[str]:
    """Return whitespace-delimited candidates before an optional ARC E marker."""

    return [piece for piece in text.split("E", 1)[0].split() if piece]


_ARC_PAIR_PATTERN = re.compile(r"^(\d+)-(\d+):(\d+)$")


@dataclass(frozen=True)
class CoordinateStructureDetail:
    expected_coordinates: int
    shown_coordinates: int
    missing_coordinates: int
    duplicate_coordinates: int
    invalid_coordinates: int
    malformed_candidates: int
    syntactically_complete: bool
    # Character spans containing row/column digits for structurally useful or
    # harmful candidates. ARC character-token positions equal character offsets.
    coordinate_spans: tuple[tuple[int, int, bool], ...]

    @property
    def shown_fraction(self) -> float:
        return self.shown_coordinates / max(self.expected_coordinates, 1)

    @property
    def structurally_complete(self) -> bool:
        return (
            self.syntactically_complete
            and self.missing_coordinates == 0
            and self.duplicate_coordinates == 0
            and self.invalid_coordinates == 0
            and self.malformed_candidates == 0
        )


@dataclass(frozen=True)
class CoordinateStructureAdvantages:
    terminal: torch.Tensor
    coordinate: torch.Tensor
    terminal_mask: torch.Tensor
    coordinate_mask: torch.Tensor
    terminal_scores: torch.Tensor
    details: tuple[CoordinateStructureDetail, ...]


def coordinate_ranking_score(
    pair_f1: float,
    detail: CoordinateStructureDetail,
    *,
    pair_f1_weight: float,
    coverage_weight: float,
    duplicate_penalty: float,
    invalid_penalty: float,
) -> float:
    """Rank a rollout by correctness and unique coordinate coverage."""

    denominator = max(detail.expected_coordinates, 1)
    return (
        pair_f1_weight * pair_f1
        + coverage_weight * detail.shown_fraction
        - duplicate_penalty * detail.duplicate_coordinates / denominator
        - invalid_penalty * detail.invalid_coordinates / denominator
    )


def coordinate_structure_detail(
    target_text: str,
    prediction_text: str,
    *,
    ended_with_eos: bool,
) -> CoordinateStructureDetail:
    """Count unique expected coordinates independently of predicted values."""

    expected: set[tuple[int, int]] = set()
    for piece in _pieces(target_text):
        match = _ARC_PAIR_PATTERN.fullmatch(piece)
        if match is None:
            raise ValueError(f"target contains malformed ARC pair: {piece!r}")
        coordinate = (int(match.group(1)), int(match.group(2)))
        if coordinate in expected:
            raise ValueError(f"target repeats ARC coordinate: {coordinate}")
        expected.add(coordinate)
    if not expected:
        raise ValueError("target must contain at least one ARC coordinate")

    body = prediction_text.split("E", 1)[0]
    seen: set[tuple[int, int]] = set()
    duplicates = invalid = malformed = 0
    spans: list[tuple[int, int, bool]] = []
    candidates = list(re.finditer(r"\S+", body))
    for candidate in candidates:
        match = _ARC_PAIR_PATTERN.fullmatch(candidate.group())
        if match is None:
            malformed += 1
            continue
        coordinate = (int(match.group(1)), int(match.group(2)))
        # Mark only row/column digits. The value has its own correctness reward.
        colon = candidate.group().index(":")
        coordinate_positions = [
            candidate.start() + offset
            for offset, character in enumerate(candidate.group()[:colon])
            if character.isdigit()
        ]
        if coordinate not in expected:
            invalid += 1
            useful = False
        elif coordinate in seen:
            duplicates += 1
            useful = False
        else:
            seen.add(coordinate)
            useful = True
        spans.extend((position, position + 1, useful) for position in coordinate_positions)

    syntactically_complete = bool(candidates) and malformed == 0
    return CoordinateStructureDetail(
        expected_coordinates=len(expected),
        shown_coordinates=len(seen),
        missing_coordinates=len(expected - seen),
        duplicate_coordinates=duplicates,
        invalid_coordinates=invalid,
        malformed_candidates=malformed,
        syntactically_complete=ended_with_eos and syntactically_complete,
        coordinate_spans=tuple(spans),
    )


def build_coordinate_structure_advantages(
    target_texts: Sequence[str],
    prediction_texts: Sequence[str],
    token_ids: Sequence[torch.Tensor],
    ended_with_eos: Sequence[bool],
    *,
    group_size: int,
    width: int,
    missing_penalty: float,
    duplicate_penalty: float,
    invalid_penalty: float,
    malformed_penalty: float,
    no_eos_penalty: float,
    device: torch.device,
) -> CoordinateStructureAdvantages:
    """Build hard terminal and local coordinate advantages over full groups.

    The normalized terminal return is broadcast over every sampled token. This
    gives structural completeness sequence-level credit without concentrating a
    length-scaled advantage of thousands on one EOS position. Coordinate
    advantages reward first visits and penalize repeated or out-of-grid
    row/column combinations immediately.
    """

    size = len(token_ids)
    if not size or not (
        size == len(target_texts) == len(prediction_texts) == len(ended_with_eos)
    ):
        raise ValueError("coordinate-structure inputs must have equal non-zero length")
    if group_size <= 1 or size % group_size:
        raise ValueError("group_size must exceed one and divide structure rows")

    terminal_scores = torch.empty(size, dtype=torch.float32, device=device)
    terminal_mask = torch.zeros((size, width), dtype=torch.bool, device=device)
    coordinate_raw = torch.zeros((size, width), dtype=torch.float32, device=device)
    coordinate_mask = torch.zeros((size, width), dtype=torch.bool, device=device)
    details: list[CoordinateStructureDetail] = []

    for row, (target, prediction, ids, ended) in enumerate(
        zip(target_texts, prediction_texts, token_ids, ended_with_eos)
    ):
        detail = coordinate_structure_detail(
            target, prediction, ended_with_eos=ended
        )
        details.append(detail)
        denominator = detail.expected_coordinates
        if detail.structurally_complete:
            score = 2.0
        else:
            score = -2.0
            score -= missing_penalty * detail.missing_coordinates / denominator
            score -= duplicate_penalty * detail.duplicate_coordinates / denominator
            score -= invalid_penalty * detail.invalid_coordinates / denominator
            score -= malformed_penalty * detail.malformed_candidates / denominator
        if not ended:
            score -= no_eos_penalty
        terminal_scores[row] = score

        ids_list = ids.flatten().tolist()
        if not ids_list or len(ids_list) > width:
            raise ValueError("cannot assign a terminal reward to an empty rollout")
        terminal_mask[row, : len(ids_list)] = True

        for start, end, useful in detail.coordinate_spans:
            for position in range(start, min(end, len(ids_list), width)):
                coordinate_mask[row, position] = True
                coordinate_raw[row, position] = 1.0 if useful else -1.0

    terminal = torch.zeros_like(coordinate_raw)
    coordinate = torch.zeros_like(coordinate_raw)
    for start in range(0, size, group_size):
        end = start + group_size
        group_scores = terminal_scores[start:end]
        normalized = (group_scores - group_scores.mean()) / (
            group_scores.std(unbiased=False) + 1e-8
        )
        for offset, row in enumerate(range(start, end)):
            terminal[row, : len(token_ids[row])] = normalized[offset]

        group_coordinate_mask = coordinate_mask[start:end]
        if group_coordinate_mask.any():
            values = coordinate_raw[start:end][group_coordinate_mask]
            normalized_values = (values - values.mean()) / (
                values.std(unbiased=False) + 1e-8
            )
            group_coordinate = coordinate[start:end]
            group_coordinate[group_coordinate_mask] = normalized_values

    return CoordinateStructureAdvantages(
        terminal=terminal,
        coordinate=coordinate,
        terminal_mask=terminal_mask,
        coordinate_mask=coordinate_mask,
        terminal_scores=terminal_scores,
        details=tuple(details),
    )


def structured_reward(target_text: str, prediction_text: str, ended_with_eos: bool) -> float:
    """Reward exact sets first and otherwise use duplicate-sensitive pair F1."""

    target = set(_pieces(target_text))
    candidates = _pieces(prediction_text)
    predicted = set(candidates)
    correct_unique = len(target & predicted)
    precision = correct_unique / max(len(candidates), 1)
    recall = correct_unique / max(len(target), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    exact = target == predicted
    score = 0.75 * f1 + 0.25 * float(exact)
    if not ended_with_eos:
        score -= 0.05
    return score


def pair_metrics(targets: Sequence[str], predictions: Sequence[str]) -> PairMetrics:
    exact = good = candidates_count = covered = target_count = 0
    for target_text, prediction_text in zip(targets, predictions):
        target = set(_pieces(target_text))
        candidates = _pieces(prediction_text)
        predicted = set(candidates)
        exact += int(target == predicted)
        good += sum(candidate in target for candidate in candidates)
        candidates_count += len(candidates)
        covered += len(predicted & target)
        target_count += len(target)
    return PairMetrics(exact, good, candidates_count, covered, target_count)


def pair_detail(target_text: str, prediction_text: str) -> dict[str, Any]:
    """Return per-example error counts without logging potentially huge grids."""

    target = set(_pieces(target_text))
    candidates = _pieces(prediction_text)
    predicted = set(candidates)
    metrics = pair_metrics([target_text], [prediction_text])
    return {
        "exact": metrics.exact,
        "good_pairs": metrics.good_pairs,
        "candidate_pairs": metrics.candidate_pairs,
        "covered_target_pairs": metrics.covered_target_pairs,
        "target_pairs": metrics.target_pairs,
        "pair_precision": metrics.precision,
        "pair_recall": metrics.recall,
        "pair_f1": metrics.f1,
        "omitted_unique_pairs": len(target - predicted),
        "incorrect_unique_pairs": len(predicted - target),
        "duplicate_pairs": len(candidates) - len(predicted),
        "prediction_sha256": hashlib.sha256(
            prediction_text.encode("utf-8")
        ).hexdigest(),
        "prediction_preview": prediction_text[:240],
    }


def group_normalized_sequence_advantages(
    scores: torch.Tensor,
    lengths: Sequence[int],
    *,
    group_size: int,
    width: int,
) -> torch.Tensor:
    """Broadcast per-rollout normalized scores using full sibling groups."""

    if scores.ndim != 1 or len(scores) != len(lengths):
        raise ValueError("scores and lengths must describe the same rollout rows")
    if group_size <= 1 or len(lengths) % group_size:
        raise ValueError("group_size must exceed one and divide the rollout rows")
    result = torch.zeros(
        (len(lengths), width), dtype=torch.float32, device=scores.device
    )
    for start in range(0, len(lengths), group_size):
        group = scores[start : start + group_size]
        normalized = (group - group.mean()) / (group.std(unbiased=False) + 1e-8)
        for offset, length in enumerate(lengths[start : start + group_size]):
            result[start + offset, :length] = normalized[offset]
    return result


def reward_advantage_components(batch: RewardBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw dense-token and numeric-pair normalized advantages."""

    token = (batch.token_rewards - batch.token_baselines) / (batch.token_stds + 1e-8)
    pair = (batch.sequence_rewards - batch.sequence_baselines) / (
        batch.sequence_std + 1e-8
    )
    if batch.sequence_mask is not None:
        pair = pair * batch.sequence_mask
    return token * batch.mask, pair * batch.mask


def on_policy_component_loss(
    current_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_reduction: str = "token",
) -> torch.Tensor:
    """Unclipped near-on-policy objective used only for gradient attribution."""

    ratio = (current_logprobs - rollout_logprobs).clamp(-20.0, 20.0).exp()
    objective = ratio * advantages.detach() * mask
    if loss_reduction == "token":
        return -objective.sum() / mask.sum().clamp_min(1)
    if loss_reduction != "sequence":
        raise ValueError("loss_reduction must be 'token' or 'sequence'")
    token_counts = mask.sum(dim=1)
    per_sequence = objective.sum(dim=1) / token_counts.clamp_min(1)
    return -per_sequence.sum() / (token_counts > 0).sum().clamp_min(1)


def select_gradient_probe_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    """Select small representative early/middle/late tensors for diagnostics."""

    named = dict(model.named_parameters())
    layer_ids = sorted(
        {
            int(name.split(".")[2])
            for name in named
            if name.startswith("model.layers.")
            and name.endswith(".mlp.down_proj.weight")
        }
    )
    if not layer_ids:
        return []
    wanted = {
        f"model.layers.{layer_ids[0]}.mlp.down_proj.weight",
        f"model.layers.{layer_ids[len(layer_ids) // 2]}.mlp.down_proj.weight",
        f"model.layers.{layer_ids[-1]}.mlp.down_proj.weight",
        "model.norm.weight",
    }
    return [(name, named[name]) for name in sorted(wanted) if name in named]


def component_gradient_diagnostics(
    losses: dict[str, torch.Tensor],
    probe_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    """Measure component gradient norms/cosines on representative parameters."""

    if not probe_parameters:
        return {"probe_parameters": [], "components": {}, "cosines": {}}
    parameters = [parameter for _, parameter in probe_parameters]
    gradients: dict[str, list[torch.Tensor | None]] = {}
    components: dict[str, Any] = {}
    for component, loss in losses.items():
        grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        detached = [None if grad is None else grad.detach().float() for grad in grads]
        gradients[component] = detached
        norm_sq = sum(
            float(grad.square().sum()) for grad in detached if grad is not None
        )
        components[component] = {
            "loss": float(loss.detach()),
            "probe_grad_norm": math.sqrt(norm_sq),
            "parameter_grad_norms": {
                name: (0.0 if grad is None else float(grad.norm()))
                for (name, _), grad in zip(probe_parameters, detached)
            },
        }
    cosines: dict[str, float] = {}
    names = list(gradients)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            dot = left_norm_sq = right_norm_sq = 0.0
            for left_grad, right_grad in zip(gradients[left], gradients[right]):
                if left_grad is None or right_grad is None:
                    continue
                dot += float((left_grad * right_grad).sum())
                left_norm_sq += float(left_grad.square().sum())
                right_norm_sq += float(right_grad.square().sum())
            denominator = math.sqrt(left_norm_sq * right_norm_sq)
            cosines[f"{left}__{right}"] = dot / denominator if denominator else 0.0
    return {
        "probe_parameters": [name for name, _ in probe_parameters],
        "components": components,
        "cosines": cosines,
    }


def build_native_reward_batch(
    predictions: Sequence[str],
    targets: Sequence[str],
    lengths: Sequence[int],
    ended_with_eos: Sequence[bool],
    *,
    group_size: int,
    device: torch.device,
) -> tuple[RewardBatch, torch.Tensor]:
    """Broadcast one structured sequence score and normalize within each problem."""

    size = len(predictions)
    if not size or not (size == len(targets) == len(lengths) == len(ended_with_eos)):
        raise ValueError("rollout reward inputs must have the same non-zero length")
    if group_size <= 1 or size % group_size:
        raise ValueError("group_size must exceed one and evenly divide the rollout batch")
    if min(lengths) <= 0:
        raise ValueError("vLLM returned an empty rollout")
    width = max(lengths)
    mask = torch.zeros((size, width), dtype=torch.bool, device=device)
    scores = torch.tensor(
        [
            structured_reward(target, prediction, ended)
            for target, prediction, ended in zip(targets, predictions, ended_with_eos)
        ],
        dtype=torch.float32,
        device=device,
    )
    sequence_rewards = torch.zeros((size, width), dtype=torch.float32, device=device)
    sequence_baselines = torch.zeros_like(sequence_rewards)
    sequence_stds = torch.zeros_like(sequence_rewards)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
        sequence_rewards[row, :length] = scores[row]
    for start in range(0, size, group_size):
        group = scores[start : start + group_size]
        mean = group.mean()
        std = group.std(unbiased=False)
        for row in range(start, start + group_size):
            length = lengths[row]
            sequence_baselines[row, :length] = mean
            sequence_stds[row, :length] = std
    zeros = torch.zeros_like(sequence_rewards)
    return (
        RewardBatch(
            token_rewards=zeros,
            sequence_rewards=sequence_rewards,
            token_baselines=zeros,
            token_stds=zeros,
            sequence_baselines=sequence_baselines,
            sequence_std=sequence_stds,
            mask=mask,
            sequence_mask=mask,
        ),
        scores,
    )


def slice_reward_batch(batch: RewardBatch, start: int, end: int, width: int) -> RewardBatch:
    return RewardBatch(
        token_rewards=batch.token_rewards[start:end, :width],
        sequence_rewards=batch.sequence_rewards[start:end, :width],
        token_baselines=batch.token_baselines[start:end, :width],
        token_stds=batch.token_stds[start:end, :width],
        sequence_baselines=batch.sequence_baselines[start:end, :width],
        sequence_std=batch.sequence_std[start:end, :width],
        mask=batch.mask[start:end, :width],
        sequence_mask=(
            None
            if batch.sequence_mask is None
            else batch.sequence_mask[start:end, :width]
        ),
    )


def select_reward_batch(batch: RewardBatch, indices: Sequence[int]) -> RewardBatch:
    """Keep selected rollouts without recomputing full-group reward statistics."""

    index = torch.tensor(indices, dtype=torch.long, device=batch.mask.device)
    return RewardBatch(
        token_rewards=batch.token_rewards.index_select(0, index),
        sequence_rewards=batch.sequence_rewards.index_select(0, index),
        token_baselines=batch.token_baselines.index_select(0, index),
        token_stds=batch.token_stds.index_select(0, index),
        sequence_baselines=batch.sequence_baselines.index_select(0, index),
        sequence_std=batch.sequence_std.index_select(0, index),
        mask=batch.mask.index_select(0, index),
        sequence_mask=(
            None
            if batch.sequence_mask is None
            else batch.sequence_mask.index_select(0, index)
        ),
    )


def native_policy_logprobs(
    model: Qwen3_5ForCausalLM,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
) -> torch.Tensor:
    """Score completion tokens without materializing prompt-position vocabulary logits."""

    if not prompts or len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same non-zero length")
    prompt_lengths = {int(prompt.numel()) for prompt in prompts}
    if len(prompt_lengths) != 1:
        raise ValueError("a native scoring microbatch must contain one shared prompt length")
    prompt_length = prompt_lengths.pop()
    max_completion = max(int(completion.numel()) for completion in completions)
    device = next(model.parameters()).device
    total_length = prompt_length + max_completion
    input_ids = torch.full(
        (len(prompts), total_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    target_ids = torch.full(
        (len(prompts), max_completion), pad_token_id, dtype=torch.long, device=device
    )
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        full = torch.cat((prompt, completion)).to(device)
        input_ids[row, : full.numel()] = full
        attention_mask[row, : full.numel()] = 1
        target_ids[row, : completion.numel()] = completion.to(device)
    positions = torch.arange(
        prompt_length - 1,
        prompt_length + max_completion - 1,
        dtype=torch.long,
        device=device,
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=positions,
        ).logits
        selected = -F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(len(prompts), max_completion)
    return selected


def load_text_policy(
    checkpoint: str | Path, *, device: torch.device, dtype: torch.dtype
) -> Qwen3_5ForCausalLM:
    """Load only the text policy, including weights from a multimodal checkpoint."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    outer_config = AutoConfig.from_pretrained(checkpoint)
    text_config = getattr(outer_config, "text_config", outer_config)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint,
        config=text_config,
        dtype=dtype,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    missing = set(loading.get("missing_keys", ()))
    unexpected = set(loading.get("unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            f"text checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.config.use_cache = False
    return model


def load_policy(
    checkpoint: Path,
    *,
    tokenizer_mode: str,
    device: torch.device,
    dtype: torch.dtype,
):
    if tokenizer_mode == "native":
        return load_text_policy(checkpoint, device=device, dtype=dtype)
    model = load_custom_checkpoint(checkpoint)
    model.to(device=device, dtype=dtype)
    model.config.use_cache = False
    return model


def load_mode_datasets(args, tokenizer):
    if args.rl_train_source in {"training-unseen-long", "training-partition"}:
        character_tokenizer = ArcCharacterTokenizer()
        evaluation, eval_summary = load_arc_examples(
            args.data_dir,
            split="evaluation",
            tokenizer=character_tokenizer,
            usable_example_limit=args.eval_examples,
            max_total_tokens=args.eval_max_sequence_length,
            augment=False,
            seed=args.seed,
        )
        if args.augment_unseen_long:
            candidates, candidate_summary = load_pretraining_examples(
                args, character_tokenizer
            )
        else:
            candidates, raw_candidate_summary = load_arc_examples(
                args.data_dir,
                split="training",
                tokenizer=character_tokenizer,
                max_target_tokens=args.max_new_tokens,
                max_total_tokens=args.max_sequence_length,
                augment=False,
                seed=args.seed,
            )
            candidate_summary = asdict(raw_candidate_summary)
        pretraining_summary = None
        seen_pretraining_tasks: set[str] = set()
        if args.rl_train_source == "training-unseen-long":
            pretraining_args = argparse.Namespace(**vars(args))
            pretraining_args.max_new_tokens = args.pretraining_max_target_tokens
            pretraining_args.max_sequence_length = args.pretraining_max_sequence_length
            pretraining_examples, pretraining_summary = load_pretraining_examples(
                pretraining_args, character_tokenizer
            )
            train, seen_pretraining_tasks = select_unseen_long_examples(
                candidates,
                pretraining_examples,
                min_total_tokens=args.min_train_sequence_length,
            )
        else:
            train = [
                example
                for example in candidates
                if example.prompt_ids.numel() + example.target_ids.numel()
                >= args.min_train_sequence_length
            ]
        partition_summary = None
        if args.train_task_partition is not None:
            allowed_ids, partition = load_partition_ids(
                args.train_task_partition, args.train_partition_role
            )
            train = filter_partition_examples(train, allowed_ids)
            selected_ids = {logical_task_id(example.task_id) for example in train}
            missing_ids = allowed_ids - selected_ids
            if missing_ids and args.diagnostic_example_split is None:
                raise ValueError(
                    f"partition has {len(missing_ids)} {args.train_partition_role} "
                    "task IDs without an eligible RL example"
                )
            partition_summary = {
                "path": str(args.train_task_partition.resolve()),
                "role": args.train_partition_role,
                "seed": partition.get("seed"),
                "logical_tasks": len(selected_ids),
                "eligible_missing_logical_tasks": len(missing_ids),
            }
        diagnostic_summary = None
        if args.diagnostic_example_split is not None:
            split_path = args.diagnostic_example_split.expanduser().resolve()
            with split_path.open(encoding="utf-8") as handle:
                diagnostic = json.load(handle)
            train_ids = list(diagnostic.get("train_example_ids", ()))
            eval_ids = list(diagnostic.get("eval_example_ids", ()))
            if not train_ids or not eval_ids:
                raise ValueError(
                    "diagnostic split requires non-empty train_example_ids and "
                    "eval_example_ids"
                )
            overlap = set(train_ids) & set(eval_ids)
            if overlap and not args.diagnostic_allow_overlap:
                raise ValueError(
                    f"diagnostic train/eval examples overlap: {sorted(overlap)}"
                )
            by_id = {example.task_id: example for example in train}
            missing = (set(train_ids) | set(eval_ids)) - set(by_id)
            if missing:
                raise ValueError(
                    f"diagnostic split references unavailable examples: {sorted(missing)}"
                )
            train = [by_id[task_id] for task_id in train_ids]
            evaluation = [by_id[task_id] for task_id in eval_ids]
            train_families = {logical_task_id(example.task_id) for example in train}
            eval_families = {logical_task_id(example.task_id) for example in evaluation}
            if eval_families != train_families:
                raise ValueError(
                    "diagnostic train/eval must contain the same logical task "
                    f"families: train_only={sorted(train_families - eval_families)}, "
                    f"eval_only={sorted(eval_families - train_families)}"
                )

            def content_fingerprint(example: ARCExample) -> str:
                digest = hashlib.sha256()
                digest.update(example.prompt_ids.numpy().tobytes())
                digest.update(b"\0")
                digest.update(example.target_ids.numpy().tobytes())
                return digest.hexdigest()

            train_content = {content_fingerprint(example) for example in train}
            eval_content = {content_fingerprint(example) for example in evaluation}
            content_overlap = train_content & eval_content
            if content_overlap and not args.diagnostic_allow_overlap:
                raise ValueError(
                    "diagnostic train/eval contain identical serialized views "
                    f"despite distinct IDs ({len(content_overlap)} collisions)"
                )
            diagnostic_summary = {
                "source": (
                    "overlapping official-training memorization examples"
                    if overlap
                    else "held-out augmented official-training examples"
                ),
                "path": str(split_path),
                "loaded_examples": len(evaluation),
                "logical_tasks": len(
                    {logical_task_id(example.task_id) for example in evaluation}
                ),
                "task_ids": eval_ids,
                "official_evaluation_examples": 0,
                "train_eval_overlap": sorted(overlap),
                "serialized_train_eval_overlap": len(content_overlap),
            }
            eval_summary = diagnostic_summary
        pretraining_overlap = seen_pretraining_tasks & {
            logical_task_id(example.task_id) for example in train
        }
        evaluation_path = Path(args.data_dir) / "arc-agi_evaluation_challenges.json"
        with evaluation_path.open() as handle:
            official_evaluation_ids = set(json.load(handle))
        evaluation_split_overlap = official_evaluation_ids & {
            logical_task_id(example.task_id) for example in train
        }
        if pretraining_overlap or evaluation_split_overlap:
            raise RuntimeError(
                "RL tasks violate the configured pretraining/evaluation separation"
            )
        if not train:
            raise ValueError("no training tasks fit the requested partition and lengths")
        train_summary = {
            "source": args.rl_train_source,
            "source_summary": candidate_summary,
            "pretraining_summary": pretraining_summary,
            "loaded_examples": len(train),
            "logical_tasks": len({logical_task_id(example.task_id) for example in train}),
            "seen_pretraining_logical_tasks": len(seen_pretraining_tasks),
            "pretraining_task_overlap": 0,
            "evaluation_split_overlap": 0,
            "augmentation": (
                f"rotations+{args.num_color_swaps}-color-swaps"
                if args.augment_unseen_long
                else "none"
            ),
            "min_prompt_length": min(example.prompt_ids.numel() for example in train),
            "max_prompt_length": max(example.prompt_ids.numel() for example in train),
            "min_target_length": min(example.target_ids.numel() for example in train),
            "max_target_length": max(example.target_ids.numel() for example in train),
            "min_total_length": min(
                example.prompt_ids.numel() + example.target_ids.numel()
                for example in train
            ),
            "max_total_length": max(
                example.prompt_ids.numel() + example.target_ids.numel()
                for example in train
            ),
            "task_ids": [example.task_id for example in train],
            "task_partition": partition_summary,
            "diagnostic_example_split": diagnostic_summary,
        }
        if args.tokenizer_mode == "native":
            return (
                _convert_examples(train, character_tokenizer, tokenizer),
                _convert_examples(evaluation, character_tokenizer, tokenizer),
                train_summary,
                eval_summary,
            )
        return train, evaluation, train_summary, eval_summary
    if args.tokenizer_mode == "native":
        return load_datasets(args, tokenizer)
    train, train_summary = load_pretraining_examples(args, tokenizer)
    evaluation, eval_summary = load_arc_examples(
        args.data_dir,
        split="evaluation",
        tokenizer=tokenizer,
        usable_example_limit=args.eval_examples,
        max_total_tokens=args.eval_max_sequence_length,
        augment=False,
        seed=args.seed,
    )
    return train, evaluation, train_summary, eval_summary


def example_target_text(example: NativeARCExample | ARCExample, tokenizer) -> str:
    if hasattr(example, "target_text"):
        return example.target_text
    return tokenizer.decode(example.target_ids)


def prepare_vllm_checkpoint(
    model, tokenizer, path: Path, *, tokenizer_mode: str
) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    model.config.architectures = ["Qwen3_5ForCausalLM"]
    model.config.save_pretrained(path)
    if tokenizer_mode == "native":
        tokenizer.save_pretrained(path)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if key != "lm_head.weight" or not model.config.tie_word_embeddings
    }
    save_file(state, path / "model.safetensors")
    return path


def next_vllm_export_path(path: Path) -> Path:
    """Return a free export path without deleting a previous vLLM snapshot."""

    path = path.expanduser()
    if not path.exists():
        return path
    for restart in range(1, 10_000):
        candidate = path.with_name(f"{path.name}_restart{restart:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free vLLM export path beside {path}")


class NativeVLLMRolloutGenerator:
    """Persistent native-tokenizer vLLM engine with atomic disk weight reloads."""

    def __init__(
        self,
        model_dir: Path,
        *,
        eos_token_id: int,
        vocab_size: int,
        dtype: str,
        max_model_len: int,
        max_num_seqs: int,
        gpu_memory_utilization: float,
        seed: int,
        enforce_eager: bool,
        tokenizer_mode: str,
    ) -> None:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        self.model_dir = model_dir.resolve()
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = int(vocab_size)
        self.max_model_len = int(max_model_len)
        tokenizer_kwargs = (
            {"tokenizer": str(self.model_dir), "skip_tokenizer_init": False}
            if tokenizer_mode == "native"
            else {"tokenizer": None, "skip_tokenizer_init": True}
        )
        self.llm = LLM(
            model=str(self.model_dir),
            **tokenizer_kwargs,
            dtype=dtype,
            max_model_len=self.max_model_len,
            max_num_seqs=int(max_num_seqs),
            max_num_batched_tokens=max(8192, self.max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            enable_prefix_caching=False,
            seed=int(seed),
            enforce_eager=bool(enforce_eager),
            trust_remote_code=False,
        )
        weights = list(self.model_dir.glob("*.safetensors"))
        if len(weights) != 1:
            raise RuntimeError("native vLLM weight sync requires one safetensors file")
        self.weights_path = weights[0]
        with safe_open(self.weights_path, framework="pt", device="cpu") as handle:
            self.weight_keys = list(handle.keys())

    def sync_from_model(self, model: Qwen3_5ForCausalLM) -> float:
        started = time.perf_counter()
        state = model.state_dict()
        unknown = set(self.weight_keys) - set(state)
        if unknown:
            raise RuntimeError(f"cannot publish missing policy tensors: {sorted(unknown)}")
        cpu_state = {
            key: state[key].detach().cpu().contiguous() for key in self.weight_keys
        }
        pending = self.weights_path.with_suffix(".safetensors.pending")
        save_file(cpu_state, pending)
        os.replace(pending, self.weights_path)
        del cpu_state
        self.llm.collective_rpc("reload_weights")
        return time.perf_counter() - started

    def generate(
        self,
        prompts: Sequence[torch.Tensor],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> NativeRolloutBatch:
        from vllm import SamplingParams

        prompt_ids = [prompt.flatten().tolist() for prompt in prompts]
        if any(not ids for ids in prompt_ids):
            raise ValueError("prompts must not be empty")
        if any(max(ids) >= self.vocab_size or min(ids) < 0 for ids in prompt_ids):
            raise ValueError("a prompt contains IDs outside the native vocabulary")
        if any(len(ids) + max_new_tokens > self.max_model_len for ids in prompt_ids):
            raise ValueError("a prompt plus max_new_tokens exceeds vLLM max_model_len")
        # vLLM deliberately gives requests with the same explicit seed the same
        # random stream.  Every rollout therefore needs its own deterministic
        # seed; otherwise all siblings of a GRPO-style group are identical and
        # the normalized advantage (and hence the RL gradient) is exactly zero.
        params = [
            SamplingParams(
                temperature=temperature if do_sample else 0.0,
                top_p=top_p,
                max_tokens=max_new_tokens,
                logprobs=1,
                stop_token_ids=[self.eos_token_id],
                detokenize=False,
                seed=None if seed is None else seed + request_index,
            )
            for request_index in range(len(prompt_ids))
        ]
        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_ids],
            params,
            use_tqdm=False,
        )
        token_ids: list[torch.Tensor] = []
        token_logprobs: list[torch.Tensor] = []
        eos_flags: list[bool] = []
        for request in outputs:
            sample = request.outputs[0]
            ids = list(sample.token_ids)
            values = []
            for token_id, candidates in zip(ids, sample.logprobs or ()):
                chosen = candidates.get(token_id)
                if chosen is None:
                    raise RuntimeError("vLLM omitted the sampled token log-probability")
                values.append(float(chosen.logprob))
            if len(values) != len(ids):
                raise RuntimeError("vLLM token/log-probability lengths differ")
            token_ids.append(torch.tensor(ids, dtype=torch.long))
            token_logprobs.append(torch.tensor(values, dtype=torch.float32))
            eos_flags.append(
                self.eos_token_id in ids
                or sample.stop_reason == self.eos_token_id
            )
        return NativeRolloutBatch(token_ids, token_logprobs, eos_flags)


def decode_rollouts(tokenizer, rollouts: NativeRolloutBatch) -> list[str]:
    predictions = []
    for ids in rollouts.token_ids:
        clean = [
            token_id
            for token_id in ids.tolist()
            if token_id not in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        ]
        if isinstance(tokenizer, ArcCharacterTokenizer):
            predictions.append(tokenizer.decode(clean, skip_special_tokens=True))
        else:
            predictions.append(
                tokenizer.decode(
                    clean,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    return predictions


def golden_nll(model, examples, pad_token_id: int) -> tuple[float, float]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_logprob = 0.0
    total_tokens = 0
    total_characters = sum(
        len(example.target_text)
        if hasattr(example, "target_text")
        else int(example.target_ids.numel())
        for example in examples
    )
    try:
        with torch.inference_mode():
            for example in examples:
                logprobs = native_policy_logprobs(
                    model,
                    [example.prompt_ids],
                    [example.target_ids],
                    pad_token_id=pad_token_id,
                )[0, : example.target_ids.numel()]
                total_logprob += float(logprobs.double().sum())
                total_tokens += example.target_ids.numel()
    finally:
        model.train(was_training)
    nll = -total_logprob / total_tokens
    bpc = -total_logprob / total_characters / math.log(2.0)
    return nll, bpc


def evaluate(
    model,
    tokenizer,
    examples: Sequence[NativeARCExample],
    generator: NativeVLLMRolloutGenerator,
    *,
    step: int,
    max_new_tokens: int,
    logger=None,
    log_details: bool = False,
    sampled_rollouts: int = 0,
    sampled_temperature: float = 1.0,
    sampled_top_p: float = 0.98,
    sampled_seed: int = 0,
) -> NativeEvalMetrics:
    loss, bpc = golden_nll(model, examples, tokenizer.pad_token_id)
    rollouts = generator.generate(
        [example.prompt_ids for example in examples],
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    predictions = decode_rollouts(tokenizer, rollouts)
    target_texts = [example_target_text(example, tokenizer) for example in examples]
    metrics = pair_metrics(target_texts, predictions)
    individual_metrics = [
        pair_metrics([target], [prediction])
        for target, prediction in zip(target_texts, predictions)
    ]
    lengths = [len(ids) for ids in rollouts.token_ids]
    if logger is not None and log_details:
        for example, target, prediction, length, ended in zip(
            examples,
            target_texts,
            predictions,
            lengths,
            rollouts.ended_with_eos,
        ):
            logger.log(
                "eval_example",
                {
                    "task_id": example.task_id,
                    "logical_task_id": logical_task_id(example.task_id),
                    "length": length,
                    "ended_with_eos": ended,
                    **pair_detail(target, prediction),
                },
                step=step,
            )
    if sampled_rollouts > 0:
        sampled_prompts = [
            example.prompt_ids
            for example in examples
            for _ in range(sampled_rollouts)
        ]
        sampled_targets = [
            target
            for target in target_texts
            for _ in range(sampled_rollouts)
        ]
        sampled = generator.generate(
            sampled_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=sampled_temperature,
            top_p=sampled_top_p,
            seed=sampled_seed + 10_000 * step,
        )
        sampled_predictions = decode_rollouts(tokenizer, sampled)
        sampled_pair_metrics = pair_metrics(sampled_targets, sampled_predictions)
        per_task_best_f1 = []
        per_task_mean_f1 = []
        tasks_with_exact = 0
        for task_index, example in enumerate(examples):
            start = task_index * sampled_rollouts
            end = start + sampled_rollouts
            details = [
                pair_detail(target_texts[task_index], prediction)
                for prediction in sampled_predictions[start:end]
            ]
            best_f1 = max(item["pair_f1"] for item in details)
            mean_f1 = sum(item["pair_f1"] for item in details) / sampled_rollouts
            exact_count = sum(item["exact"] for item in details)
            per_task_best_f1.append(best_f1)
            per_task_mean_f1.append(mean_f1)
            tasks_with_exact += int(exact_count > 0)
            if logger is not None and log_details:
                logger.log(
                    "sampled_eval_example",
                    {
                        "task_id": example.task_id,
                        "logical_task_id": logical_task_id(example.task_id),
                        "rollouts": sampled_rollouts,
                        "mean_pair_f1": mean_f1,
                        "best_pair_f1": best_f1,
                        "exact_rollouts": exact_count,
                        "mean_length": sum(
                            len(ids) for ids in sampled.token_ids[start:end]
                        )
                        / sampled_rollouts,
                        "eos_fraction": sum(sampled.ended_with_eos[start:end])
                        / sampled_rollouts,
                    },
                    step=step,
                )
        if logger is not None:
            logger.log(
                "sampled_eval",
                {
                    "rollouts_per_example": sampled_rollouts,
                    "num_examples": len(examples),
                    "num_rollouts": len(sampled_predictions),
                    "exact_rollouts": sampled_pair_metrics.exact,
                    "tasks_with_exact_rollout": tasks_with_exact,
                    "pass_at_k": tasks_with_exact / len(examples),
                    "micro_pair_precision": sampled_pair_metrics.precision,
                    "micro_pair_recall": sampled_pair_metrics.recall,
                    "micro_pair_f1": sampled_pair_metrics.f1,
                    "macro_mean_pair_f1": sum(per_task_mean_f1) / len(examples),
                    "macro_best_pair_f1": sum(per_task_best_f1) / len(examples),
                    "mean_structured_reward": sum(
                        structured_reward(target, prediction, ended)
                        for target, prediction, ended in zip(
                            sampled_targets,
                            sampled_predictions,
                            sampled.ended_with_eos,
                        )
                    )
                    / len(sampled_predictions),
                    "mean_length": sum(len(ids) for ids in sampled.token_ids)
                    / len(sampled.token_ids),
                    "eos_fraction": sum(sampled.ended_with_eos)
                    / len(sampled.ended_with_eos),
                },
                step=step,
            )
    return NativeEvalMetrics(
        step=step,
        loss=loss,
        bits_per_character=bpc,
        num_correct=metrics.exact,
        num_samples=len(examples),
        pair_precision=metrics.precision,
        pair_recall=metrics.recall,
        pair_f1=metrics.f1,
        macro_pair_precision=sum(item.precision for item in individual_metrics)
        / len(individual_metrics),
        macro_pair_recall=sum(item.recall for item in individual_metrics)
        / len(individual_metrics),
        macro_pair_f1=sum(item.f1 for item in individual_metrics)
        / len(individual_metrics),
        good_pairs=metrics.good_pairs,
        candidate_pairs=metrics.candidate_pairs,
        covered_target_pairs=metrics.covered_target_pairs,
        target_pairs=metrics.target_pairs,
        mean_length=sum(lengths) / len(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        eos_fraction=sum(rollouts.ended_with_eos) / len(rollouts.ended_with_eos),
    )


def scheduled_lr(args, step: int) -> float:
    if args.lr_schedule == "constant":
        return args.learning_rate
    if step <= args.warmup_steps:
        return args.learning_rate * step / max(args.warmup_steps, 1)
    progress = (step - args.warmup_steps - 1) / max(
        args.steps - args.warmup_steps - 1, 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return args.learning_rate * (
        args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
    )


def evaluation_is_better(candidate, best, args) -> bool:
    if args.eval_selection == "pair_f1":
        f1_delta = candidate.pair_f1 - best.pair_f1
        if f1_delta >= args.min_pair_f1_improvement:
            return True
        if abs(f1_delta) < args.min_pair_f1_improvement:
            if candidate.num_correct != best.num_correct:
                return candidate.num_correct > best.num_correct
            return candidate.loss < best.loss
        return False
    if candidate.num_correct != best.num_correct:
        return candidate.num_correct > best.num_correct
    recall_delta = candidate.pair_recall - best.pair_recall
    if recall_delta >= args.min_pair_recall_improvement:
        return True
    if abs(recall_delta) < args.min_pair_recall_improvement:
        return candidate.loss < best.loss
    return False


class LogicalTaskEpochSampler:
    """Visit every logical ARC task once before reshuffling augmented views."""

    def __init__(self, examples, rng: random.Random):
        self.rng = rng
        self.by_task: dict[str, list] = {}
        for example in examples:
            self.by_task.setdefault(logical_task_id(example.task_id), []).append(example)
        self.task_ids = sorted(self.by_task)
        self.order: list[str] = []
        self.position = 0
        self.epoch = 0

    @property
    def seen_in_epoch(self) -> int:
        return self.position

    def choose(self, count: int):
        if count > len(self.task_ids):
            raise ValueError("problems_per_step exceeds the number of logical tasks")
        if not self.order or self.position + count > len(self.order):
            if self.order and self.position != len(self.order):
                raise ValueError(
                    "logical task count must be divisible by problems_per_step "
                    "to avoid crossing epoch boundaries"
                )
            self.order = self.task_ids.copy()
            self.rng.shuffle(self.order)
            self.position = 0
            self.epoch += 1
        selected = self.order[self.position : self.position + count]
        self.position += count
        return [self.rng.choice(self.by_task[task_id]) for task_id in selected]


def choose_examples(
    examples,
    args,
    step: int,
    rng: random.Random,
    logical_task_sampler: LogicalTaskEpochSampler | None = None,
):
    if logical_task_sampler is not None:
        return logical_task_sampler.choose(args.problems_per_step)
    ordered = sorted(
        examples, key=lambda item: item.prompt_ids.numel() + item.target_ids.numel()
    )
    if args.curriculum_steps > 0 and step < args.curriculum_steps:
        progress = (step - 1) / max(args.curriculum_steps - 1, 1)
        fraction = args.curriculum_start_fraction + (
            1.0 - args.curriculum_start_fraction
        ) * progress
        ordered = ordered[: max(1, math.ceil(len(ordered) * fraction))]
    if args.rl_train_source == "training-unseen-long":
        by_task: dict[str, list] = {}
        for example in ordered:
            by_task.setdefault(logical_task_id(example.task_id), []).append(example)
        task_ids = list(by_task)
        if args.prefer_longest_problems:
            task_ids.sort(
                key=lambda task_id: max(
                    item.prompt_ids.numel() + item.target_ids.numel()
                    for item in by_task[task_id]
                ),
                reverse=True,
            )
            selected_task_ids = task_ids[: args.problems_per_step]
            chosen = [
                max(
                    by_task[task_id],
                    key=lambda item: item.prompt_ids.numel() + item.target_ids.numel(),
                )
                for task_id in selected_task_ids
            ]
        else:
            selected_task_ids = (
                rng.sample(task_ids, args.problems_per_step)
                if len(task_ids) >= args.problems_per_step
                else rng.choices(task_ids, k=args.problems_per_step)
            )
            chosen = [rng.choice(by_task[task_id]) for task_id in selected_task_ids]
        if len(task_ids) >= args.problems_per_step:
            assert len({logical_task_id(item.task_id) for item in chosen}) == len(chosen)
        return chosen
    return (
        rng.sample(ordered, args.problems_per_step)
        if len(ordered) >= args.problems_per_step
        else rng.choices(ordered, k=args.problems_per_step)
    )


def save_checkpoint(model, tokenizer, path: Path, *, tokenizer_mode: str) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    model.config.architectures = ["Qwen3_5ForCausalLM"]
    model.config.save_pretrained(path)
    if tokenizer_mode == "native":
        tokenizer.save_pretrained(path)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if key != "lm_head.weight" or not model.config.tie_word_embeddings
    }
    pending = path / "model.safetensors.pending"
    save_file(state, pending)
    os.replace(pending, path / "model.safetensors")
    return path


def train(model, tokenizer, train_examples, eval_examples, generator, optimizer, args, logger):
    device = next(model.parameters()).device
    rng = random.Random(args.seed)
    gradient_probe_parameters = select_gradient_probe_parameters(model)
    logical_task_sampler = (
        LogicalTaskEpochSampler(train_examples, rng)
        if args.sample_logical_tasks_without_replacement
        else None
    )
    best = evaluate(
        model,
        tokenizer,
        eval_examples,
        generator,
        step=0,
        max_new_tokens=args.eval_max_new_tokens,
        logger=logger,
        log_details=args.eval_detail,
        sampled_rollouts=args.sampled_eval_rollouts,
        sampled_temperature=args.sampled_eval_temperature,
        sampled_top_p=args.sampled_eval_top_p,
        sampled_seed=args.seed + 700_000,
    )
    logger.log("eval", asdict(best), step=0)
    if not args.no_save_checkpoints:
        save_checkpoint(
            model,
            tokenizer,
            args.output_checkpoint,
            tokenizer_mode=args.tokenizer_mode,
        )
    print(
        f"eval step=0 loss={best.loss:.6f} bpc={best.bits_per_character:.6f} "
        f"exact={best.num_correct}/{best.num_samples} "
        f"precision={best.pair_precision:.6f} recall={best.pair_recall:.6f} "
        f"f1={best.pair_f1:.6f} "
        f"length={best.mean_length:.1f} eos={best.eos_fraction:.3f}",
        flush=True,
    )
    no_improvement = 0
    history = []
    for step in range(1, args.steps + 1):
        run_gradient_diagnostics = (
            args.gradient_diagnostics_every > 0
            and (step == 1 or step % args.gradient_diagnostics_every == 0)
        )
        gradient_diagnostic = None
        probe_before = (
            {
                name: parameter.detach().clone()
                for name, parameter in gradient_probe_parameters
            }
            if run_gradient_diagnostics
            else {}
        )
        chosen = choose_examples(
            train_examples, args, step, rng, logical_task_sampler
        )
        chosen_total_lengths = [
            example.prompt_ids.numel() + example.target_ids.numel()
            for example in chosen
        ]
        prompts = [
            example.prompt_ids
            for example in chosen
            for _ in range(args.rollouts_per_problem)
        ]
        target_examples = [
            example
            for example in chosen
            for _ in range(args.rollouts_per_problem)
        ]
        lr = scheduled_lr(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        rollout_temperature = args.rollout_temperature
        rollout_resamples = 0
        while True:
            rollouts = generator.generate(
                prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=rollout_temperature,
                top_p=args.rollout_top_p,
                seed=args.seed + step + 100_000 * rollout_resamples,
            )
            unique_rollouts_per_group = [
                len(
                    {
                        tuple(ids.tolist())
                        for ids in rollouts.token_ids[
                            start : start + args.rollouts_per_problem
                        ]
                    }
                )
                for start in range(
                    0, len(rollouts.token_ids), args.rollouts_per_problem
                )
            ]
            diverse_groups = sum(count > 1 for count in unique_rollouts_per_group)
            diverse_group_fraction = diverse_groups / args.problems_per_step
            required_unique = (
                args.retained_rollouts_per_problem
                + args.bottom_rollouts_per_problem
                if args.retained_rollouts_per_problem
                else 2
            )
            if (
                (
                    diverse_group_fraction >= args.min_diverse_group_fraction
                    and min(unique_rollouts_per_group) >= required_unique
                )
                or rollout_resamples >= args.max_rollout_resamples
            ):
                break
            rollout_resamples += 1
            rollout_temperature = min(
                rollout_temperature * args.rollout_temperature_multiplier,
                args.max_rollout_temperature,
            )
        torch.cuda.synchronize(device)
        rollout_seconds = time.perf_counter() - started
        predictions = decode_rollouts(tokenizer, rollouts)
        target_texts = [
            example_target_text(example, tokenizer) for example in target_examples
        ]
        lengths = [len(ids) for ids in rollouts.token_ids]
        diversity_insufficient = (
            diverse_group_fraction < args.min_diverse_group_fraction
            or min(unique_rollouts_per_group) < required_unique
        )
        if diversity_insufficient:
            diversity_warning = {
                "unique_rollouts_per_group": unique_rollouts_per_group,
                "diverse_group_fraction": diverse_group_fraction,
                "required_unique": required_unique,
                "rollout_resamples": rollout_resamples,
                "final_temperature": rollout_temperature,
                "duplicates_allowed": True,
            }
            logger.log("rollout_diversity_warning", diversity_warning, step=step)
            print(
                "WARNING rollout diversity is below the requested threshold; "
                f"continuing with duplicates: unique_per_group="
                f"{unique_rollouts_per_group} required_unique={required_unique} "
                f"resamples={rollout_resamples} temperature={rollout_temperature:.3f}",
                flush=True,
            )
        if args.dense_token_reward:
            rewards = build_reward_batch(
                rollouts.token_ids,
                [example.target_ids for example in target_examples],
                tokenizer,
                group_size=args.rollouts_per_problem,
                sequence_reward_mode="pair_correctness",
                device=device,
            )
            sequence_scores = torch.tensor(
                [
                    pair_metrics([target], [prediction]).precision
                    for target, prediction in zip(target_texts, predictions)
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            rewards, sequence_scores = build_native_reward_batch(
                predictions,
                target_texts,
                lengths,
                rollouts.ended_with_eos,
                group_size=args.rollouts_per_problem,
                device=device,
            )
        mean_length = sum(lengths) / len(lengths)
        eos_fraction = sum(rollouts.ended_with_eos) / len(rollouts.ended_with_eos)
        if (
            not args.smoke_test
            and mean_length >= args.max_mean_rollout_fraction * args.max_new_tokens
            and eos_fraction < 0.25
        ):
            raise RuntimeError(
                f"rollout collapse: mean_length={mean_length:.1f}, eos={eos_fraction:.3f}"
            )
        rollout_pairs = pair_metrics(target_texts, predictions)
        exact_flags = [
            set(_pieces(target)) == set(_pieces(prediction))
            for target, prediction in zip(target_texts, predictions)
        ]
        groups_with_exact_rollout = sum(
            any(exact_flags[start : start + args.rollouts_per_problem])
            for start in range(0, len(exact_flags), args.rollouts_per_problem)
        )
        exact_pair_advantages = []
        if args.dense_token_reward and rewards.sequence_mask is not None:
            pair_advantages = (
                (rewards.sequence_rewards - rewards.sequence_baselines)
                / (rewards.sequence_std + 1e-8)
            )
            for row, is_exact in enumerate(exact_flags):
                exact_mask = rewards.sequence_mask[row]
                if is_exact and exact_mask.any():
                    exact_pair_advantages.append(pair_advantages[row][exact_mask].mean())

        top_per_problem = (
            args.retained_rollouts_per_problem or args.rollouts_per_problem
        )
        bottom_per_problem = args.bottom_rollouts_per_problem
        retained_per_problem = top_per_problem + bottom_per_problem
        pair_f1_scores = [
            pair_metrics([target], [prediction]).f1
            for target, prediction in zip(target_texts, predictions)
        ]
        f1_scores = torch.tensor(
            pair_f1_scores,
            dtype=torch.float32,
            device=device,
        )
        structured_f1_advantages = group_normalized_sequence_advantages(
            f1_scores,
            lengths,
            group_size=args.rollouts_per_problem,
            width=rewards.mask.shape[1],
        )
        structure_enabled = (
            args.eos_structure_weight > 0
            or args.duplicate_coordinate_weight > 0
            or args.ranking_coordinate_coverage_weight > 0
            or args.ranking_duplicate_coordinate_penalty > 0
            or args.ranking_invalid_coordinate_penalty > 0
        )
        if structure_enabled:
            structure = build_coordinate_structure_advantages(
                target_texts,
                predictions,
                rollouts.token_ids,
                rollouts.ended_with_eos,
                group_size=args.rollouts_per_problem,
                width=rewards.mask.shape[1],
                missing_penalty=args.eos_missing_penalty,
                duplicate_penalty=args.eos_duplicate_penalty,
                invalid_penalty=args.eos_invalid_penalty,
                malformed_penalty=args.eos_malformed_penalty,
                no_eos_penalty=args.no_eos_penalty,
                device=device,
            )
            eos_structure_advantages = structure.terminal
            coordinate_structure_advantages = structure.coordinate
            terminal_structure_mask = structure.terminal_mask
            coordinate_structure_mask = structure.coordinate_mask
            all_structure_details = structure.details
            all_terminal_structure_scores = structure.terminal_scores
        else:
            eos_structure_advantages = torch.zeros_like(structured_f1_advantages)
            coordinate_structure_advantages = torch.zeros_like(
                structured_f1_advantages
            )
            terminal_structure_mask = torch.zeros_like(
                structured_f1_advantages, dtype=torch.bool
            )
            coordinate_structure_mask = torch.zeros_like(
                structured_f1_advantages, dtype=torch.bool
            )
            all_structure_details = ()
            all_terminal_structure_scores = torch.zeros(
                len(predictions), dtype=torch.float32, device=device
            )
        composite_ranking_enabled = (
            args.ranking_coordinate_coverage_weight > 0
            or args.ranking_duplicate_coordinate_penalty > 0
            or args.ranking_invalid_coordinate_penalty > 0
        )
        if composite_ranking_enabled:
            ranking_scores = [
                coordinate_ranking_score(
                    pair_f1,
                    detail,
                    pair_f1_weight=args.ranking_pair_f1_weight,
                    coverage_weight=args.ranking_coordinate_coverage_weight,
                    duplicate_penalty=args.ranking_duplicate_coordinate_penalty,
                    invalid_penalty=args.ranking_invalid_coordinate_penalty,
                )
                for pair_f1, detail in zip(pair_f1_scores, all_structure_details)
            ]
        else:
            ranking_scores = [
                structured_reward(target, prediction, ended)
                for target, prediction, ended in zip(
                    target_texts, predictions, rollouts.ended_with_eos
                )
            ]
        if (
            top_per_problem < args.rollouts_per_problem
            or bottom_per_problem > 0
        ):
            retained_indices = []
            duplicate_fallbacks_per_group = []
            for start in range(0, len(prompts), args.rollouts_per_problem):
                group_indices = list(range(start, start + args.rollouts_per_problem))
                rng.shuffle(group_indices)
                group_indices.sort(key=lambda row: ranking_scores[row], reverse=True)
                ranked_unique_indices = []
                seen_sequences = set()
                for row in group_indices:
                    sequence = tuple(rollouts.token_ids[row].tolist())
                    if sequence in seen_sequences:
                        continue
                    seen_sequences.add(sequence)
                    ranked_unique_indices.append(row)
                top_indices = ranked_unique_indices[:top_per_problem]
                duplicate_fallbacks = 0
                if len(top_indices) < top_per_problem:
                    for row in group_indices:
                        if row not in top_indices:
                            top_indices.append(row)
                            duplicate_fallbacks += 1
                            if len(top_indices) == top_per_problem:
                                break
                top_set = set(top_indices)
                bottom_indices = [
                    row
                    for row in reversed(ranked_unique_indices)
                    if row not in top_set
                ][:bottom_per_problem]
                if len(bottom_indices) < bottom_per_problem:
                    selected_set = top_set | set(bottom_indices)
                    for row in reversed(group_indices):
                        if row not in selected_set:
                            bottom_indices.append(row)
                            selected_set.add(row)
                            duplicate_fallbacks += 1
                            if len(bottom_indices) == bottom_per_problem:
                                break
                unique_indices = top_indices + bottom_indices
                if len(unique_indices) != retained_per_problem:
                    raise RuntimeError(
                        f"could retain only {len(unique_indices)} rollout rows; "
                        f"required {retained_per_problem}"
                    )
                duplicate_fallbacks_per_group.append(duplicate_fallbacks)
                retained_indices.extend(unique_indices)
            if any(duplicate_fallbacks_per_group):
                selection_warning = {
                    "duplicate_fallbacks_per_group": duplicate_fallbacks_per_group,
                    "top_rollouts_per_problem": top_per_problem,
                    "bottom_rollouts_per_problem": bottom_per_problem,
                    "duplicates_allowed": True,
                }
                logger.log(
                    "rollout_selection_duplicate_warning", selection_warning, step=step
                )
                print(
                    "WARNING top/bottom selection used repeated token sequences: "
                    f"duplicate_fallbacks_per_group={duplicate_fallbacks_per_group}",
                    flush=True,
                )
            prompts = [prompts[row] for row in retained_indices]
            target_examples = [target_examples[row] for row in retained_indices]
            rollouts = NativeRolloutBatch(
                token_ids=[rollouts.token_ids[row] for row in retained_indices],
                token_logprobs=[rollouts.token_logprobs[row] for row in retained_indices],
                ended_with_eos=[rollouts.ended_with_eos[row] for row in retained_indices],
            )
            rewards = select_reward_batch(rewards, retained_indices)
            sequence_scores = sequence_scores[retained_indices]
            structured_f1_advantages = structured_f1_advantages.index_select(
                0, torch.tensor(retained_indices, dtype=torch.long, device=device)
            )
            retained_index = torch.tensor(
                retained_indices, dtype=torch.long, device=device
            )
            eos_structure_advantages = eos_structure_advantages.index_select(
                0, retained_index
            )
            coordinate_structure_advantages = (
                coordinate_structure_advantages.index_select(0, retained_index)
            )
            terminal_structure_mask = terminal_structure_mask.index_select(
                0, retained_index
            )
            coordinate_structure_mask = coordinate_structure_mask.index_select(
                0, retained_index
            )
            retained_ranking_scores = [ranking_scores[row] for row in retained_indices]
        else:
            retained_ranking_scores = ranking_scores
        old_logprobs = pad_token_logprobs(
            rollouts.token_logprobs, multiple_of=1
        ).to(device)
        total_tokens = int(rewards.mask.sum())
        total_sequences = int(rewards.mask.any(dim=1).sum())
        total_reduction_units = (
            total_tokens if args.loss_reduction == "token" else total_sequences
        )
        eos_structure_advantage_abs_mean = (
            float(eos_structure_advantages[terminal_structure_mask].abs().mean())
            if terminal_structure_mask.any()
            else 0.0
        )
        structured_f1_advantage_abs_mean = float(
            structured_f1_advantages[rewards.mask].abs().mean()
        )
        coordinate_structure_advantage_abs_mean = (
            float(
                coordinate_structure_advantages[coordinate_structure_mask]
                .abs()
                .mean()
            )
            if coordinate_structure_mask.any()
            else 0.0
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        policy_loss_sum = error_sum = 0.0
        error_count = 0
        max_error = 0.0
        metric_sums = {
            "advantage_abs_mean": 0.0,
            "advantage_nonzero_fraction": 0.0,
            "extra_advantage_abs_mean": 0.0,
            "ratio_mean": 0.0,
            "clip_fraction": 0.0,
            "token_reward_mean": 0.0,
            "sequence_reward_mean": 0.0,
            "pair_correctness_reward_mean": 0.0,
        }
        for start in range(0, len(prompts), args.training_microbatch_size):
            end = min(start + args.training_microbatch_size, len(prompts))
            completion_chunk = rollouts.token_ids[start:end]
            width = max(len(item) for item in completion_chunk)
            reward_chunk = slice_reward_batch(rewards, start, end, width)
            current = native_policy_logprobs(
                model,
                prompts[start:end],
                completion_chunk,
                pad_token_id=tokenizer.pad_token_id,
            )
            errors = (current.detach() - old_logprobs[start:end, :width])[
                reward_chunk.mask
            ].abs()
            error_sum += float(errors.sum())
            error_count += errors.numel()
            max_error = max(max_error, float(errors.max()))
            extra_advantages = (
                args.structured_f1_weight
                * structured_f1_advantages[start:end, :width]
                + args.eos_structure_weight
                * eos_structure_advantages[start:end, :width]
                + args.duplicate_coordinate_weight
                * coordinate_structure_advantages[start:end, :width]
            )
            batch_loss, batch_metrics = compute_clipped_policy_loss(
                current,
                old_logprobs[start:end, :width],
                reward_chunk,
                clip_epsilon=args.clip_epsilon,
                sequence_weight=args.sequence_weight,
                pair_correctness_weight=(
                    args.pair_correctness_weight
                    if args.dense_token_reward
                    else None
                ),
                extra_advantages=extra_advantages,
                loss_reduction=args.loss_reduction,
            )
            if run_gradient_diagnostics and gradient_diagnostic is None:
                token_advantage, pair_advantage = reward_advantage_components(
                    reward_chunk
                )
                f1_advantage = structured_f1_advantages[start:end, :width]
                eos_advantage = eos_structure_advantages[start:end, :width]
                coordinate_advantage = coordinate_structure_advantages[
                    start:end, :width
                ]
                component_losses = {
                    "token": on_policy_component_loss(
                        current,
                        old_logprobs[start:end, :width],
                        token_advantage,
                        reward_chunk.mask,
                        loss_reduction=args.loss_reduction,
                    ),
                    "pair_weighted": args.pair_correctness_weight
                    * on_policy_component_loss(
                        current,
                        old_logprobs[start:end, :width],
                        pair_advantage,
                        reward_chunk.mask,
                        loss_reduction=args.loss_reduction,
                    ),
                    "structured_f1_raw": on_policy_component_loss(
                        current,
                        old_logprobs[start:end, :width],
                        f1_advantage,
                        reward_chunk.mask,
                        loss_reduction=args.loss_reduction,
                    ),
                    "eos_structure_weighted": args.eos_structure_weight
                    * on_policy_component_loss(
                        current,
                        old_logprobs[start:end, :width],
                        eos_advantage,
                        reward_chunk.mask,
                        loss_reduction=args.loss_reduction,
                    ),
                    "coordinate_structure_weighted": args.duplicate_coordinate_weight
                    * on_policy_component_loss(
                        current,
                        old_logprobs[start:end, :width],
                        coordinate_advantage,
                        reward_chunk.mask,
                        loss_reduction=args.loss_reduction,
                    ),
                    "combined": batch_loss,
                }
                gradient_diagnostic = component_gradient_diagnostics(
                    component_losses, gradient_probe_parameters
                )
                gradient_diagnostic.update(
                    {
                        "microbatch_start": start,
                        "microbatch_end": end,
                        "microbatch_valid_tokens": int(reward_chunk.mask.sum()),
                        "pair_correctness_weight": args.pair_correctness_weight,
                        "structured_f1_weight": args.structured_f1_weight,
                        "eos_structure_weight": args.eos_structure_weight,
                        "duplicate_coordinate_weight": args.duplicate_coordinate_weight,
                    }
                )
            valid = int(reward_chunk.mask.sum())
            reduction_units = (
                valid
                if args.loss_reduction == "token"
                else int(reward_chunk.mask.any(dim=1).sum())
            )
            (batch_loss * reduction_units / total_reduction_units).backward()
            policy_loss_sum += float(batch_loss.detach()) * reduction_units
            for key in metric_sums:
                metric_sums[key] += float(batch_metrics[key].detach()) * valid
        policy_loss = policy_loss_sum / total_reduction_units
        anchor_loss = 0.0
        if args.supervised_anchor_weight > 0:
            anchor_tokens = sum(example.target_ids.numel() for example in chosen)
            anchor_sum = 0.0
            for example in chosen:
                logprobs = native_policy_logprobs(
                    model,
                    [example.prompt_ids],
                    [example.target_ids],
                    pad_token_id=tokenizer.pad_token_id,
                )[0, : example.target_ids.numel()]
                loss = -logprobs.mean()
                weight = example.target_ids.numel() / anchor_tokens
                (args.supervised_anchor_weight * loss * weight).backward()
                anchor_sum += float(loss.detach()) * example.target_ids.numel()
            anchor_loss = anchor_sum / anchor_tokens
        logprob_mae = error_sum / max(error_count, 1)
        if logprob_mae > args.max_rollout_logprob_mae:
            raise RuntimeError(
                f"vLLM/current logprob MAE {logprob_mae:.6f} exceeds "
                f"{args.max_rollout_logprob_mae:.6f}"
            )
        grad_mean, grad_variance = gradient_mean_and_variance(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if gradient_diagnostic is not None:
            update_norm_sq = parameter_norm_sq = 0.0
            changed = total = 0
            per_parameter_updates = {}
            for name, parameter in gradient_probe_parameters:
                before = probe_before[name]
                after = parameter.detach()
                delta = after.float() - before.float()
                delta_norm = float(delta.norm())
                before_norm = float(before.float().norm())
                update_norm_sq += delta_norm**2
                parameter_norm_sq += before_norm**2
                changed_count = int((after != before).sum())
                changed += changed_count
                total += before.numel()
                per_parameter_updates[name] = {
                    "update_l2": delta_norm,
                    "relative_update_l2": delta_norm / max(before_norm, 1e-30),
                    "bf16_changed_fraction": changed_count / before.numel(),
                }
            update_norm = math.sqrt(update_norm_sq)
            parameter_norm = math.sqrt(parameter_norm_sq)
            gradient_diagnostic["update"] = {
                "probe_update_l2": update_norm,
                "probe_relative_update_l2": update_norm
                / max(parameter_norm, 1e-30),
                "probe_bf16_changed_fraction": changed / max(total, 1),
                "parameter_updates": per_parameter_updates,
            }
            logger.log("gradient_diagnostic", gradient_diagnostic, step=step)
        sync_seconds = generator.sync_from_model(model)
        metrics = NativeStepMetrics(
            step=step,
            loss=policy_loss + args.supervised_anchor_weight * anchor_loss,
            policy_loss=policy_loss,
            supervised_anchor_loss=anchor_loss,
            learning_rate=lr,
            grad_norm=float(grad_norm),
            grad_mean=grad_mean,
            grad_variance=grad_variance,
            rollout_seconds=rollout_seconds,
            rollout_tokens_per_second=sum(lengths) / rollout_seconds,
            weight_sync_seconds=sync_seconds,
            rollout_logprob_mae=logprob_mae,
            rollout_logprob_max_error=max_error,
            reward_mean=float(sequence_scores.mean()),
            reward_std=float(sequence_scores.std(unbiased=False)),
            retained_reward_mean=(
                sum(retained_ranking_scores) / len(retained_ranking_scores)
            ),
            rollouts_retained_per_problem=retained_per_problem,
            top_rollouts_retained_per_problem=top_per_problem,
            bottom_rollouts_retained_per_problem=bottom_per_problem,
            logical_task_epoch=(
                logical_task_sampler.epoch if logical_task_sampler is not None else 0
            ),
            logical_tasks_seen_in_epoch=(
                logical_task_sampler.seen_in_epoch
                if logical_task_sampler is not None
                else 0
            ),
            token_reward_mean=metric_sums["token_reward_mean"] / total_tokens,
            sequence_reward_mean=metric_sums["sequence_reward_mean"] / total_tokens,
            pair_correctness_reward_mean=(
                metric_sums["pair_correctness_reward_mean"] / total_tokens
            ),
            diverse_group_fraction=diverse_group_fraction,
            minimum_unique_rollouts_per_group=min(unique_rollouts_per_group),
            rollout_temperature=rollout_temperature,
            rollout_resamples=rollout_resamples,
            advantage_abs_mean=metric_sums["advantage_abs_mean"] / total_tokens,
            advantage_nonzero_fraction=metric_sums["advantage_nonzero_fraction"] / total_tokens,
            structured_f1_advantage_abs_mean=structured_f1_advantage_abs_mean,
            eos_structure_advantage_abs_mean=eos_structure_advantage_abs_mean,
            coordinate_structure_advantage_abs_mean=(
                coordinate_structure_advantage_abs_mean
            ),
            eos_structure_reward_mean=float(all_terminal_structure_scores.mean()),
            shown_coordinate_fraction_mean=(
                sum(detail.shown_fraction for detail in all_structure_details)
                / max(len(all_structure_details), 1)
            ),
            missing_coordinates_mean=(
                sum(detail.missing_coordinates for detail in all_structure_details)
                / max(len(all_structure_details), 1)
            ),
            duplicate_coordinates_mean=(
                sum(detail.duplicate_coordinates for detail in all_structure_details)
                / max(len(all_structure_details), 1)
            ),
            invalid_coordinates_mean=(
                sum(detail.invalid_coordinates for detail in all_structure_details)
                / max(len(all_structure_details), 1)
            ),
            malformed_candidates_mean=(
                sum(detail.malformed_candidates for detail in all_structure_details)
                / max(len(all_structure_details), 1)
            ),
            ratio_mean=metric_sums["ratio_mean"] / total_tokens,
            clip_fraction=metric_sums["clip_fraction"] / total_tokens,
            num_correct=rollout_pairs.exact,
            num_samples=len(exact_flags),
            groups_with_exact_rollout=groups_with_exact_rollout,
            groups_with_exact_rollout_fraction=(
                groups_with_exact_rollout / args.problems_per_step
            ),
            exact_rollout_pair_advantage_mean=(
                float(torch.stack(exact_pair_advantages).mean())
                if exact_pair_advantages
                else 0.0
            ),
            pair_precision=rollout_pairs.precision,
            pair_recall=rollout_pairs.recall,
            problem_total_length_mean=(
                sum(chosen_total_lengths) / len(chosen_total_lengths)
            ),
            problem_total_length_min=min(chosen_total_lengths),
            problem_total_length_max=max(chosen_total_lengths),
            mean_length=mean_length,
            eos_fraction=eos_fraction,
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        )
        history.append(metrics)
        logger.log("train", asdict(metrics), step=step)
        print(
            f"train step={step} loss={metrics.loss:.6f} lr={lr:.3e} "
            f"reward={metrics.reward_mean:.4f}+/-{metrics.reward_std:.4f} "
            f"retained={metrics.rollouts_retained_per_problem}/"
            f"{args.rollouts_per_problem} retained_reward="
            f"{metrics.retained_reward_mean:.4f} "
            f"top={metrics.top_rollouts_retained_per_problem} "
            f"bottom={metrics.bottom_rollouts_retained_per_problem} "
            f"task_epoch={metrics.logical_task_epoch} "
            f"tasks_seen={metrics.logical_tasks_seen_in_epoch} "
            f"diverse_groups={metrics.diverse_group_fraction:.2f} "
            f"min_unique={metrics.minimum_unique_rollouts_per_group} "
            f"temperature={metrics.rollout_temperature:.3f} "
            f"token_reward={metrics.token_reward_mean:.3f} "
            f"shown={metrics.shown_coordinate_fraction_mean:.3f} "
            f"missing={metrics.missing_coordinates_mean:.1f} "
            f"duplicate={metrics.duplicate_coordinates_mean:.1f} "
            f"invalid={metrics.invalid_coordinates_mean:.1f} "
            f"eos_struct={metrics.eos_structure_reward_mean:.3f} "
            f"exact={metrics.num_correct}/{metrics.num_samples} "
            f"exact_groups={metrics.groups_with_exact_rollout}/"
            f"{args.problems_per_step} "
            f"exact_pair_adv={metrics.exact_rollout_pair_advantage_mean:.3f} "
            f"precision={metrics.pair_precision:.4f} recall={metrics.pair_recall:.4f} "
            f"problem_len={metrics.problem_total_length_min}-"
            f"{metrics.problem_total_length_max} "
            f"length={mean_length:.1f} eos={eos_fraction:.3f} "
            f"logprob_mae={logprob_mae:.5f} grad_norm={float(grad_norm):.4f} "
            f"rollout_tok_s={metrics.rollout_tokens_per_second:.1f} "
            f"sync_s={sync_seconds:.1f} peak_gib={metrics.cuda_peak_reserved_gib:.2f}",
            flush=True,
        )
        if args.eval_every > 0 and step % args.eval_every == 0:
            candidate = evaluate(
                model,
                tokenizer,
                eval_examples,
                generator,
                step=step,
                max_new_tokens=args.eval_max_new_tokens,
                logger=logger,
                log_details=args.eval_detail,
                sampled_rollouts=args.sampled_eval_rollouts,
                sampled_temperature=args.sampled_eval_temperature,
                sampled_top_p=args.sampled_eval_top_p,
                sampled_seed=args.seed + 700_000,
            )
            logger.log("eval", asdict(candidate), step=step)
            print(
                f"eval step={step} loss={candidate.loss:.6f} "
                f"bpc={candidate.bits_per_character:.6f} "
                f"exact={candidate.num_correct}/{candidate.num_samples} "
                f"precision={candidate.pair_precision:.6f} "
                f"recall={candidate.pair_recall:.6f} "
                f"f1={candidate.pair_f1:.6f} "
                f"length={candidate.mean_length:.1f} eos={candidate.eos_fraction:.3f}",
                flush=True,
            )
            if evaluation_is_better(candidate, best, args):
                best = candidate
                no_improvement = 0
                if not args.no_save_checkpoints:
                    saved = save_checkpoint(
                        model,
                        tokenizer,
                        args.output_checkpoint,
                        tokenizer_mode=args.tokenizer_mode,
                    )
                    logger.log(
                        "best_checkpoint", {"path": str(saved), **asdict(best)}, step=step
                    )
                else:
                    logger.log("best_eval", asdict(best), step=step)
            else:
                no_improvement += 1
            if (
                args.eval_patience > 0
                and step >= args.min_steps_before_early_stop
                and no_improvement >= args.eval_patience
            ):
                logger.log("early_stop", {"evaluations_without_improvement": no_improvement}, step=step)
                print(f"early_stop step={step} patience={no_improvement}", flush=True)
                break
    return best, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-mode", choices=("native", "arc"), default="native")
    parser.add_argument(
        "--rl-train-source",
        choices=("pretraining", "training-unseen-long", "training-partition"),
        default="pretraining",
        help=(
            "Use the original pretraining examples, longer tasks excluded from "
            "pretraining, or an explicit partition of all eligible training tasks."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-checkpoint", type=Path, default=Path("checkpoints/rl_qwen35_native_best"))
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Run diagnostics without writing initial or promoted policy checkpoints.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ARC_DATA)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--problems-per-step", type=int, default=4)
    parser.add_argument("--rollouts-per-problem", type=int, default=8)
    parser.add_argument(
        "--retained-rollouts-per-problem",
        type=int,
        default=0,
        help=(
            "Train on the top K structured-reward rollouts from each full sibling "
            "group; zero retains every rollout. Baselines still use the full group."
        ),
    )
    parser.add_argument(
        "--bottom-rollouts-per-problem",
        type=int,
        default=0,
        help=(
            "Additionally train on the bottom K distinct structured-reward "
            "rollouts; advantages remain normalized over the full sibling group."
        ),
    )
    parser.add_argument("--training-microbatch-size", type=int, default=2)
    parser.add_argument(
        "--sample-logical-tasks-without-replacement",
        action="store_true",
        help=(
            "Shuffle logical task families and visit each exactly once per epoch; "
            "choose one augmented view whenever a family is visited."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--rollout-temperature", type=float, default=1.2)
    parser.add_argument(
        "--max-rollout-temperature",
        type=float,
        default=2.0,
        help="Upper bound for adaptive diversity retries (vLLM accepts at most 2).",
    )
    parser.add_argument("--rollout-top-p", type=float, default=0.98)
    parser.add_argument("--rollout-temperature-multiplier", type=float, default=1.25)
    parser.add_argument("--max-rollout-resamples", type=int, default=2)
    parser.add_argument("--min-diverse-group-fraction", type=float, default=0.25)
    parser.add_argument("--eval-max-new-tokens", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=2816)
    parser.add_argument("--min-train-sequence-length", type=int, default=0)
    parser.add_argument("--pretraining-max-sequence-length", type=int, default=2816)
    parser.add_argument("--pretraining-max-target-tokens", type=int, default=2816)
    parser.add_argument(
        "--augment-unseen-long",
        action="store_true",
        help="Add configured rotations and color swaps to unseen long tasks.",
    )
    parser.add_argument("--eval-max-sequence-length", type=int, default=2816)
    parser.add_argument("--eval-examples", type=int, default=150)
    parser.add_argument("--num-color-swaps", type=int, default=2)
    parser.add_argument("--no-rotations", action="store_true")
    parser.add_argument(
        "--train-task-partition",
        type=Path,
        help="JSON manifest containing disjoint sft_task_ids and rl_task_ids.",
    )
    parser.add_argument(
        "--diagnostic-example-split",
        type=Path,
        help=(
            "JSON manifest with exact train_example_ids and eval_example_ids. "
            "Both sides are drawn from the loaded official-training candidates; "
            "the official evaluation set is not used."
        ),
    )
    parser.add_argument(
        "--diagnostic-allow-overlap",
        action="store_true",
        help="Allow identical train/eval IDs only for explicit memorization tests.",
    )
    parser.add_argument(
        "--train-partition-role",
        choices=("sft", "rl"),
        default="rl",
        help="Logical-task side to load from --train-task-partition.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--loss-reduction",
        choices=("token", "sequence"),
        default="token",
        help=(
            "Average the policy objective over all valid tokens, or average "
            "within each rollout first so every retained sequence has equal weight."
        ),
    )
    parser.add_argument("--sequence-weight", type=float, default=1.0)
    parser.add_argument(
        "--pair-correctness-weight",
        type=float,
        default=0.25,
        help=(
            "Additive exact-pair advantage weight for numeric ARC tokens only; "
            "dense individual-token advantages retain unit weight."
        ),
    )
    parser.add_argument(
        "--structured-f1-weight",
        type=float,
        default=0.0,
        help=(
            "Add a full-sibling normalized pair-F1 advantage broadcast across "
            "each completion; zero preserves the original dense objective."
        ),
    )
    parser.add_argument(
        "--eos-structure-weight",
        type=float,
        default=0.0,
        help=(
            "Post-normalization weight for the hard terminal coordinate-coverage "
            "advantage; zero disables its contribution to the policy loss."
        ),
    )
    parser.add_argument(
        "--duplicate-coordinate-weight",
        type=float,
        default=0.0,
        help=(
            "Post-normalization weight for rewarding first coordinate visits and "
            "penalizing repeated or out-of-grid row/column combinations."
        ),
    )
    parser.add_argument(
        "--ranking-pair-f1-weight",
        type=float,
        default=1.0,
        help="Pair-F1 contribution to composite top/bottom rollout ranking.",
    )
    parser.add_argument(
        "--ranking-coordinate-coverage-weight",
        type=float,
        default=0.0,
        help=(
            "Unique expected-coordinate coverage contribution to rollout ranking; "
            "a positive value enables composite structural ranking."
        ),
    )
    parser.add_argument(
        "--ranking-duplicate-coordinate-penalty",
        type=float,
        default=0.0,
        help="Penalty on repeated-coordinate rate in composite rollout ranking.",
    )
    parser.add_argument(
        "--ranking-invalid-coordinate-penalty",
        type=float,
        default=0.0,
        help="Penalty on out-of-grid-coordinate rate in composite rollout ranking.",
    )
    parser.add_argument("--eos-missing-penalty", type=float, default=8.0)
    parser.add_argument("--eos-duplicate-penalty", type=float, default=4.0)
    parser.add_argument("--eos-invalid-penalty", type=float, default=4.0)
    parser.add_argument("--eos-malformed-penalty", type=float, default=2.0)
    parser.add_argument("--no-eos-penalty", type=float, default=2.0)
    parser.add_argument("--dense-token-reward", action="store_true")
    parser.add_argument("--supervised-anchor-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-diagnostics-every",
        type=int,
        default=0,
        help=(
            "Log token/pair/structured-F1 probe-gradient norms, cosines, and "
            "effective BF16 updates at step 1 and every N steps; zero disables."
        ),
    )
    parser.add_argument("--max-rollout-logprob-mae", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--curriculum-start-fraction", type=float, default=0.25)
    parser.add_argument("--curriculum-steps", type=int, default=20)
    parser.add_argument(
        "--prefer-longest-problems",
        action="store_true",
        help="Select the longest distinct logical tasks (for memory probes).",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--eval-detail",
        action="store_true",
        help="Log per-example greedy error counts and output hashes.",
    )
    parser.add_argument("--sampled-eval-rollouts", type=int, default=0)
    parser.add_argument("--sampled-eval-temperature", type=float, default=1.0)
    parser.add_argument("--sampled-eval-top-p", type=float, default=0.98)
    parser.add_argument("--eval-patience", type=int, default=4)
    parser.add_argument("--min-steps-before-early-stop", type=int, default=15)
    parser.add_argument("--min-pair-recall-improvement", type=float, default=0.002)
    parser.add_argument(
        "--eval-selection",
        choices=("exact_recall_nll", "pair_f1"),
        default="exact_recall_nll",
    )
    parser.add_argument("--min-pair-f1-improvement", type=float, default=0.002)
    parser.add_argument("--max-mean-rollout-fraction", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", type=int, default=6144)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.15)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run-name")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--wandb-project", default="ARC-vLLM-RL")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the checkpoint with vLLM and exit without optimizer updates.",
    )
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        help=(
            "With --eval-only, evaluate the loaded training partition instead "
            "of the official evaluation cohort."
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    positive = (
        args.steps,
        args.problems_per_step,
        args.rollouts_per_problem,
        args.training_microbatch_size,
        args.max_new_tokens,
        args.eval_examples,
        args.vllm_max_model_len,
        args.vllm_max_num_seqs,
    )
    if min(positive) <= 0:
        raise ValueError("steps, batch sizes, lengths, and eval_examples must be positive")
    if args.rollouts_per_problem % args.training_microbatch_size:
        raise ValueError("training_microbatch_size must divide rollouts_per_problem")
    retained = args.retained_rollouts_per_problem
    bottom = args.bottom_rollouts_per_problem
    if retained < 0 or retained > args.rollouts_per_problem:
        raise ValueError("retained_rollouts_per_problem must be in [0, rollouts_per_problem]")
    if bottom < 0 or bottom > args.rollouts_per_problem:
        raise ValueError("bottom_rollouts_per_problem must be in [0, rollouts_per_problem]")
    if bottom and not retained:
        raise ValueError("bottom rollout selection requires retained_rollouts_per_problem")
    selected = (retained or args.rollouts_per_problem) + bottom
    if selected > args.rollouts_per_problem:
        raise ValueError("top and bottom retained rollouts cannot exceed the full group")
    if selected % args.training_microbatch_size:
        raise ValueError(
            "training_microbatch_size must divide total selected rollouts per problem"
        )
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("vLLM GPU utilization must be in (0, 1]")
    if not 0 < args.rollout_temperature <= 2:
        raise ValueError("rollout_temperature must be in (0, 2]")
    if not args.rollout_temperature <= args.max_rollout_temperature <= 2:
        raise ValueError(
            "max_rollout_temperature must be between rollout_temperature and 2"
        )
    if not 0 < args.rollout_top_p <= 1:
        raise ValueError("rollout_top_p must be in (0, 1]")
    if args.rollout_temperature_multiplier <= 1:
        raise ValueError("rollout_temperature_multiplier must exceed one")
    if args.max_rollout_resamples < 0:
        raise ValueError("max_rollout_resamples must be non-negative")
    if not 0 < args.min_diverse_group_fraction <= 1:
        raise ValueError("min_diverse_group_fraction must be in (0, 1]")
    if not 0 <= args.supervised_anchor_weight or not 0 <= args.weight_decay:
        raise ValueError("anchor weight and weight decay must be non-negative")
    if not 0 <= args.sequence_weight <= 1:
        raise ValueError("sequence_weight must be in [0, 1]")
    if not 0 <= args.pair_correctness_weight < 1:
        raise ValueError("pair_correctness_weight must be in [0, 1)")
    if args.structured_f1_weight < 0:
        raise ValueError("structured_f1_weight must be non-negative")
    structure_values = (
        args.eos_structure_weight,
        args.duplicate_coordinate_weight,
        args.ranking_pair_f1_weight,
        args.ranking_coordinate_coverage_weight,
        args.ranking_duplicate_coordinate_penalty,
        args.ranking_invalid_coordinate_penalty,
        args.eos_missing_penalty,
        args.eos_duplicate_penalty,
        args.eos_invalid_penalty,
        args.eos_malformed_penalty,
        args.no_eos_penalty,
    )
    if min(structure_values) < 0:
        raise ValueError("coordinate-structure weights and penalties must be non-negative")
    if (
        (
            args.eos_structure_weight > 0
            or args.duplicate_coordinate_weight > 0
            or args.ranking_coordinate_coverage_weight > 0
            or args.ranking_duplicate_coordinate_penalty > 0
            or args.ranking_invalid_coordinate_penalty > 0
        )
        and args.tokenizer_mode != "arc"
    ):
        raise ValueError("coordinate-structure rewards currently require tokenizer_mode=arc")
    if args.sampled_eval_rollouts < 0:
        raise ValueError("sampled_eval_rollouts must be non-negative")
    if args.gradient_diagnostics_every < 0:
        raise ValueError("gradient_diagnostics_every must be non-negative")
    if not 0 < args.sampled_eval_temperature <= 2:
        raise ValueError("sampled_eval_temperature must be in (0, 2]")
    if not 0 < args.sampled_eval_top_p <= 1:
        raise ValueError("sampled_eval_top_p must be in (0, 1]")
    if args.dense_token_reward and args.tokenizer_mode != "arc":
        raise ValueError("dense_token_reward currently requires tokenizer_mode=arc")
    if args.augment_unseen_long and args.rl_train_source not in {
        "training-unseen-long", "training-partition"
    }:
        raise ValueError(
            "augment_unseen_long requires an explicit official-training source"
        )
    if args.train_task_partition is not None and args.rl_train_source not in {
        "training-unseen-long", "training-partition"
    }:
        raise ValueError(
            "train_task_partition requires an explicit official-training source"
        )
    if args.min_train_sequence_length < 0:
        raise ValueError("min_train_sequence_length must be non-negative")
    if args.min_train_sequence_length > args.max_sequence_length:
        raise ValueError(
            "min_train_sequence_length cannot exceed max_sequence_length"
        )
    if (
        args.pretraining_max_sequence_length <= 0
        or args.pretraining_max_target_tokens <= 0
    ):
        raise ValueError("pretraining length limits must be positive")


def main() -> None:
    args = parse_args()
    if args.tokenizer_mode == "arc" and args.checkpoint == DEFAULT_CHECKPOINT:
        args.checkpoint = DEFAULT_CUSTOM_CHECKPOINT
    if (
        args.tokenizer_mode == "arc"
        and args.output_checkpoint == Path("checkpoints/rl_qwen35_native_best")
    ):
        args.output_checkpoint = Path("checkpoints/rl_qwen35_custom_best")
    validate_args(args)
    # Keep the venv's console scripts (not the symlink target's /usr/local/bin)
    # visible to FlashInfer JIT subprocesses.
    interpreter_bin = str(Path(sys.executable).absolute().parent)
    os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.tokenizer_mode == "native":
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    else:
        tokenizer = ArcCharacterTokenizer()
    train_examples, eval_examples, train_summary, eval_summary = load_mode_datasets(
        args, tokenizer
    )
    if args.eval_on_train:
        if not args.eval_only:
            raise ValueError("--eval-on-train requires --eval-only")
        eval_examples = train_examples
        eval_summary = {
            "source": "loaded_training_partition",
            "loaded_examples": len(eval_examples),
            "logical_tasks": len(
                {logical_task_id(example.task_id) for example in eval_examples}
            ),
        }
    if args.smoke_test:
        args.steps = 2
        args.problems_per_step = 1
        args.rollouts_per_problem = 8
        args.training_microbatch_size = 1
        args.max_new_tokens = min(args.max_new_tokens, 32)
        eval_examples = eval_examples[:2]
        args.eval_every = 1
        args.eval_patience = 0
    eval_target_max = max(example.target_ids.numel() for example in eval_examples)
    if args.eval_max_new_tokens <= 0:
        args.eval_max_new_tokens = eval_target_max + 256
    required_context = max(
        max(example.prompt_ids.numel() for example in train_examples) + args.max_new_tokens,
        max(example.prompt_ids.numel() for example in eval_examples) + args.eval_max_new_tokens,
    )
    if required_context > args.vllm_max_model_len:
        raise ValueError(
            f"vLLM max_model_len={args.vllm_max_model_len} is below required {required_context}"
        )

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        f"rl_qwen35_{args.tokenizer_mode}_%Y%m%dT%H%M%SZ"
    )
    log_file = args.log_file or Path("logs") / f"{run_name}.jsonl"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "run_name": run_name,
            "rollout_batch_size": args.problems_per_step * args.rollouts_per_problem,
            "reward": (
                "dense_arc_token+numeric_exact_pair"
                if args.dense_token_reward
                else "0.75_pair_f1+0.25_exact-0.05_missing_eos"
            ),
            "policy_sequence_weight": args.sequence_weight,
            "pair_correctness_weight": args.pair_correctness_weight,
            "dense_token_reward": args.dense_token_reward,
            "tokenizer_mode": args.tokenizer_mode,
            "rl_train_source": args.rl_train_source,
            "gpu": torch.cuda.get_device_name(device),
            "vllm_version_required": ">=0.27,<0.28",
        }
    )
    logger = ExperimentLogger(
        log_file,
        config=config,
        run_name=run_name,
        wandb_project=args.wandb_project,
    )
    logger.log("dataset_train", train_summary)
    logger.log(
        "dataset_eval",
        eval_summary if isinstance(eval_summary, dict) else asdict(eval_summary),
    )
    logger.log(
            "qwen35_dataset",
        {
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "max_native_train_prompt": max(e.prompt_ids.numel() for e in train_examples),
            "max_native_eval_prompt": max(e.prompt_ids.numel() for e in eval_examples),
            "max_native_eval_target": eval_target_max,
            "required_vllm_context": required_context,
            "rl_train_source": args.rl_train_source,
            "train_task_ids": [example.task_id for example in train_examples],
            "eval_task_ids": [example.task_id for example in eval_examples],
        },
    )

    model = load_policy(
        args.checkpoint,
        tokenizer_mode=args.tokenizer_mode,
        device=device,
        dtype=torch.bfloat16,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    vllm_work_root = Path(
        os.environ.get("VLLM_WORK_DIR", "checkpoints/vllm_work")
    ).expanduser()
    requested_vllm_dir = vllm_work_root / run_name
    vllm_dir = next_vllm_export_path(requested_vllm_dir)
    if vllm_dir != requested_vllm_dir:
        logger.log(
            "vllm_export_restart",
            {
                "occupied_path": str(requested_vllm_dir.resolve()),
                "selected_path": str(vllm_dir.resolve()),
            },
        )
        print(
            f"vllm_export_exists={requested_vllm_dir} "
            f"using={vllm_dir}",
            flush=True,
        )
    prepare_vllm_checkpoint(
        model, tokenizer, vllm_dir, tokenizer_mode=args.tokenizer_mode
    )
    print(
        f"model_parameters={sum(p.numel() for p in model.parameters()):,} "
        f"train_examples={len(train_examples)} eval_examples={len(eval_examples)} "
        f"eval_max_new_tokens={args.eval_max_new_tokens} vllm_dir={vllm_dir}",
        flush=True,
    )
    generator = NativeVLLMRolloutGenerator(
        vllm_dir,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=model.config.vocab_size,
        dtype="bfloat16",
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        seed=args.seed,
        enforce_eager=args.vllm_enforce_eager,
        tokenizer_mode=args.tokenizer_mode,
    )
    if args.eval_only:
        result = evaluate(
            model,
            tokenizer,
            eval_examples,
            generator,
            step=0,
            max_new_tokens=args.eval_max_new_tokens,
            logger=logger,
            log_details=args.eval_detail,
            sampled_rollouts=args.sampled_eval_rollouts,
            sampled_temperature=args.sampled_eval_temperature,
            sampled_top_p=args.sampled_eval_top_p,
            sampled_seed=args.seed + 700_000,
        )
        logger.log("eval", asdict(result), step=0)
        logger.log("result", {"eval_only": True, "metrics": asdict(result)})
        print(
            f"eval step=0 loss={result.loss:.6f} "
            f"bpc={result.bits_per_character:.6f} "
            f"exact={result.num_correct}/{result.num_samples} "
            f"precision={result.pair_precision:.6f} "
            f"recall={result.pair_recall:.6f} "
            f"f1={result.pair_f1:.6f} "
            f"length={result.mean_length:.1f} eos={result.eos_fraction:.3f}",
            flush=True,
        )
        logger.close()
        return
    optimizer = FP32MasterAdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best, history = train(
        model,
        tokenizer,
        train_examples,
        eval_examples,
        generator,
        optimizer,
        args,
        logger,
    )
    logger.log(
        "result",
        {
            "best": asdict(best),
            "updates_completed": len(history),
            "output_checkpoint": str(args.output_checkpoint.resolve()),
        },
    )
    print(f"best_eval={json.dumps(asdict(best), sort_keys=True)}", flush=True)
    logger.close()


if __name__ == "__main__":
    main()

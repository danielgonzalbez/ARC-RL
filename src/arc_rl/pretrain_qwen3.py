"""Compiled supervised continuation pretraining for the small ARC policy.

The script preserves the prompt/target representation used by
``arc_rl.rl_static_cache`` and optimizes causal cross entropy only on target
tokens.  It evaluates the input checkpoint before training and the best saved
checkpoint after training with the same greedy rollout allowance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from .task_partition import (
    filter_partition_examples,
    load_partition_ids,
    logical_task_id,
)
from .rl_static_cache import (
    ARC_CHARACTERS,
    ARCExample,
    ArcCharacterTokenizer,
    BucketedRolloutGenerator,
    DEFAULT_ARC_DATA,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    completion_is_correct,
    current_policy_logprobs,
    evaluate_policy,
    load_arc_examples,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
COLON_TOKEN_ID = ARC_CHARACTERS.index(":")
DIGIT_TOKEN_IDS = frozenset(ARC_CHARACTERS.index(str(value)) for value in range(10))


def build_model_config() -> Qwen3Config:
    """Define Qwen3-0.6B with the small ARC vocabulary and new I/O layers."""

    return Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=40_960,
        attention_bias=False,
        use_cache=True,
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
    )


def build_model() -> Qwen3ForCausalLM:
    """Construct the custom-vocabulary Qwen3-0.6B model."""

    return Qwen3ForCausalLM(build_model_config())


def build_seeded_model(seed: int) -> Qwen3ForCausalLM:
    """Construct reproducible scratch weights without advancing global RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build_model()


def _check_source_architecture(source_config: Qwen3Config) -> None:
    """Fail clearly if model_id no longer describes the architecture above."""

    custom = build_model_config()
    fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    mismatches = {
        field: (getattr(source_config, field), getattr(custom, field))
        for field in fields
        if getattr(source_config, field) != getattr(custom, field)
    }
    if mismatches:
        raise RuntimeError(f"Pretrained model does not match the custom config: {mismatches}")


def load_pretrained_body(model_id: str) -> Qwen3ForCausalLM:
    """Initialize new ARC I/O layers and inherit the pretrained transformer.

    Loading happens on CPU. Only ``model.embed_tokens.weight`` and
    ``lm_head.weight`` are omitted; all transformer blocks and the final RMS
    normalization must load exactly.
    """

    source = Qwen3ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    _check_source_architecture(source.config)
    model = build_model()
    replaced = {"model.embed_tokens.weight", "lm_head.weight"}
    inherited = {
        name: tensor
        for name, tensor in source.state_dict().items()
        if name not in replaced
    }
    incompatible = model.load_state_dict(inherited, strict=False)
    missing = set(incompatible.missing_keys)
    if missing != replaced or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected pretrained-weight mismatch: "
            f"missing={sorted(missing)}, unexpected={incompatible.unexpected_keys}"
        )
    del source, inherited
    model.generation_config.eos_token_id = EOS_TOKEN_ID
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    return model


def load_custom_checkpoint(path: Path) -> Qwen3ForCausalLM:
    """Load a checkpoint previously produced by this script."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_model()
    incompatible = model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.generation_config.eos_token_id = EOS_TOKEN_ID
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    return model


def save_custom_checkpoint(model: Qwen3ForCausalLM, path: Path) -> Path:
    """Save weights without relying on the small RL model's architecture."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path))
    return path


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    optimizer_step: int
    train_loss: float
    eval_loss: float
    eval_color_loss: float
    eval_selection_loss: float
    eval_color_tokens: int
    learning_rate: float
    tokens_seen: int
    seconds: float
    tokens_per_second: float
    grad_norm: float
    cuda_peak_allocated_gib: float
    cuda_peak_reserved_gib: float


@dataclass(frozen=True)
class GoldenEvalMetrics:
    target_loss: float
    color_loss: float
    target_tokens: int
    color_tokens: int


class Logger:
    def __init__(self, path: Path, config: dict, run_name: str, project: str) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.wandb_run = None
        # This consumes .env through python-dotenv without displaying or changing it.
        load_dotenv()
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=project, name=run_name, config=config
                )
            except Exception as exc:
                print(f"wandb disabled after initialization failure: {exc}", flush=True)
        self.log("run_start", config)

    def log(self, event: str, values: dict) -> None:
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log({event: values} if event != "epoch" else values)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Initialize the custom-vocabulary Qwen3 architecture randomly.",
    )
    parser.add_argument("--output-checkpoint", type=Path, default=Path("checkpoints/qwen3_0.6b_arc_pretrained_best.safetensors"))
    parser.add_argument(
        "--latest-checkpoint",
        type=Path,
        help="Optionally save the terminal model weights separately from the best checkpoint.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ARC_DATA)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--minibatches", "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Microbatches accumulated before each optimizer update.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Stop after this many epochs without a material eval improvement; zero disables it.",
    )
    parser.add_argument(
        "--min-eval-improvement", type=float, default=0.0,
        help="Absolute eval-loss decrease required to reset plateau patience.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("target_nll", "color_nll"),
        default="target_nll",
        help=(
            "Metric used for checkpoint ranking and early stopping. color_nll "
            "scores only the value digit immediately following each colon."
        ),
    )
    parser.add_argument(
        "--top-k-checkpoints",
        type=int,
        default=0,
        help="Retain this many lowest-selection-loss epoch checkpoints; zero disables.",
    )
    parser.add_argument(
        "--top-k-checkpoint-dir",
        type=Path,
        help="Directory for retained epoch checkpoints when --top-k-checkpoints is set.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-scope", choices=("full", "target"), default="full")
    parser.add_argument("--max-sequence-length", type=int, default=2816)
    parser.add_argument(
        "--min-train-sequence-length",
        type=int,
        default=0,
        help="Discard augmented training examples shorter than this total length.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2816)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--eval-max-sequence-length", type=int, default=2816)
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=0,
        help="Greedy evaluation allowance; zero reuses --max-new-tokens.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-color-swaps", type=int, default=2)
    parser.add_argument("--no-rotations", action="store_true")
    parser.add_argument(
        "--train-task-partition",
        type=Path,
        help="JSON manifest containing disjoint sft_task_ids and rl_task_ids.",
    )
    parser.add_argument(
        "--train-partition-role",
        choices=("sft", "rl"),
        default="sft",
        help="Logical-task side to load from --train-task-partition.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="max-autotune")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Recompute layer activations during backward to reduce peak VRAM.",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--wandb-project", default="ARC-Pretraining")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--skip-baseline-greedy-eval",
        action="store_true",
        help="Compute baseline golden NLL without an expensive random-head rollout.",
    )
    parser.add_argument(
        "--skip-final-greedy-eval",
        action="store_true",
        help=(
            "Report only golden-target NLL for the selected checkpoint; use "
            "when a separate backend performs final generation."
        ),
    )
    parser.add_argument("--save-last-candidate", action="store_true")
    return parser.parse_args()


def padded_batch(
    examples: Sequence[ARCExample],
    max_length: int,
    device: torch.device,
    *,
    loss_scope: str = "target",
    batch_size: int | None = None,
):
    # A single fixed shape gives torch.compile/CUDA graphs one stable graph.
    batch = batch_size or len(examples)
    input_ids = torch.full((batch, max_length), PAD_TOKEN_ID, dtype=torch.long)
    labels = torch.full((batch, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_length), dtype=torch.long)
    target_tokens = 0
    for row, example in enumerate(examples):
        sequence = torch.cat((example.prompt_ids, example.target_ids))
        if sequence.numel() > max_length:
            raise ValueError(f"{example.task_id} has {sequence.numel()} tokens, limit={max_length}")
        length = sequence.numel()
        prompt_length = example.prompt_ids.numel()
        input_ids[row, :length] = sequence
        attention_mask[row, :length] = 1
        if loss_scope == "full":
            labels[row, :length] = sequence
            target_tokens += length - 1
        else:
            labels[row, prompt_length:length] = example.target_ids
            target_tokens += example.target_ids.numel()
    return (
        input_ids.to(device, non_blocking=True),
        attention_mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        target_tokens,
    )


def load_pretraining_examples(args, tokenizer):
    """Load all eligible tasks with rotations and independent color variants."""
    examples, base_summary = load_arc_examples(
        args.data_dir, split="training", tokenizer=tokenizer,
        max_target_tokens=args.max_new_tokens,
        max_total_tokens=args.max_sequence_length, augment=True,
        add_rotations=not args.no_rotations, num_color_swaps=0, seed=args.seed,
    )
    color_summaries = []
    for color_index in range(args.num_color_swaps):
        color_examples, color_summary = load_arc_examples(
            args.data_dir, split="training", tokenizer=tokenizer,
            max_target_tokens=args.max_new_tokens,
            max_total_tokens=args.max_sequence_length, augment=True,
            add_rotations=False, num_color_swaps=1,
            seed=args.seed + color_index + 1,
        )
        # Each call also contains the unmodified original. Keep only its color
        # variant and give it a globally unique ID for logs/debugging.
        for example in color_examples:
            if example.task_id.endswith("_colors_0"):
                examples.append(
                    ARCExample(
                        task_id=f"{example.task_id}_seed_{args.seed + color_index + 1}",
                        prompt_ids=example.prompt_ids,
                        target_ids=example.target_ids,
                    )
                )
        color_summaries.append(asdict(color_summary))
    return examples, {"base_and_rotations": asdict(base_summary), "color_loads": color_summaries, "loaded_examples": len(examples)}


def apply_training_partition(examples, args):
    """Apply length and optional logical-task filters to augmented examples."""

    selected = [
        example
        for example in examples
        if example.prompt_ids.numel() + example.target_ids.numel()
        >= args.min_train_sequence_length
    ]
    summary = {
        "min_train_sequence_length": args.min_train_sequence_length,
        "before_partition_examples": len(examples),
    }
    if args.train_task_partition is not None:
        allowed_ids, manifest = load_partition_ids(
            args.train_task_partition, args.train_partition_role
        )
        selected = filter_partition_examples(selected, allowed_ids)
        selected_ids = {logical_task_id(example.task_id) for example in selected}
        missing_ids = allowed_ids - selected_ids
        if missing_ids:
            raise ValueError(
                f"partition has {len(missing_ids)} {args.train_partition_role} task IDs "
                "without an eligible augmented example"
            )
        summary.update(
            {
                "partition_path": str(args.train_task_partition.resolve()),
                "partition_role": args.train_partition_role,
                "partition_seed": manifest.get("seed"),
                "partition_logical_tasks": len(selected_ids),
            }
        )
    if not selected:
        raise ValueError("no supervised training examples remain after filtering")
    summary.update(
        {
            "loaded_examples": len(selected),
            "logical_tasks": len({logical_task_id(item.task_id) for item in selected}),
            "min_total_length": min(
                item.prompt_ids.numel() + item.target_ids.numel() for item in selected
            ),
            "max_total_length": max(
                item.prompt_ids.numel() + item.target_ids.numel() for item in selected
            ),
            "task_ids": [item.task_id for item in selected],
        }
    )
    return selected, summary


def target_loss(model, input_ids, attention_mask, labels):
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def golden_eval_metrics(model, examples, batch_size, max_length, device) -> GoldenEvalMetrics:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_sum = 0
    color_loss_sum = 0.0
    color_token_sum = 0
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                chunk = list(examples[start : start + batch_size])
                prompts = [example.prompt_ids for example in chunk]
                targets = [example.target_ids for example in chunk]
                logprobs = current_policy_logprobs(model, prompts, targets)
                for row, target in enumerate(targets):
                    target_logprobs = logprobs[row, : target.numel()]
                    loss_sum -= float(target_logprobs.sum())
                    token_sum += target.numel()
                    color_mask = torch.zeros_like(target, dtype=torch.bool)
                    color_mask[1:] = target[:-1] == COLON_TOKEN_ID
                    color_ids = target[color_mask].tolist()
                    if any(token_id not in DIGIT_TOKEN_IDS for token_id in color_ids):
                        raise ValueError("a token after ':' is not an ARC color digit")
                    color_loss_sum -= float(target_logprobs[color_mask].sum())
                    color_token_sum += int(color_mask.sum())
    finally:
        model.train(was_training)
    if color_token_sum == 0:
        raise ValueError("evaluation set contains no ARC color-value tokens")
    return GoldenEvalMetrics(
        target_loss=loss_sum / token_sum,
        color_loss=color_loss_sum / color_token_sum,
        target_tokens=token_sum,
        color_tokens=color_token_sum,
    )


def golden_eval_loss(model, examples, batch_size, max_length, device) -> float:
    """Backward-compatible total target NLL wrapper."""

    return golden_eval_metrics(
        model, examples, batch_size, max_length, device
    ).target_loss


def greedy_eval(model, tokenizer, examples, args, device, step):
    max_new_tokens = args.eval_max_new_tokens or args.max_new_tokens
    max_prompt = max(example.prompt_ids.numel() for example in examples)
    required = max_prompt + max_new_tokens
    bucket = int(math.ceil(required / 256) * 256)
    generator = BucketedRolloutGenerator(
        model,
        buckets=(bucket,),
        rollout_batch_size=args.eval_batch_size,
        compile_decode=True,
        compile_mode=args.compile_mode,
    )
    return evaluate_policy(
        model, tokenizer, examples, generator, step=step,
        max_new_tokens=max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    if args.from_scratch and args.resume_checkpoint is not None:
        raise ValueError("--from-scratch and --resume-checkpoint are mutually exclusive")
    if args.max_sequence_length <= 1 or args.max_new_tokens <= 0:
        raise ValueError("sequence and rollout limits must be positive")
    if not 0 <= args.min_train_sequence_length <= args.max_sequence_length:
        raise ValueError("min_train_sequence_length must be in [0, max_sequence_length]")
    if args.eval_max_new_tokens < 0:
        raise ValueError("eval_max_new_tokens must be non-negative")
    if (
        args.batch_size <= 0
        or args.minibatches <= 0
        or args.eval_batch_size <= 0
        or args.epochs <= 0
    ):
        raise ValueError("batch sizes, minibatches, and epochs must be positive")
    if args.early_stop_patience < 0 or args.min_eval_improvement < 0:
        raise ValueError("plateau patience and minimum improvement must be non-negative")
    if args.top_k_checkpoints < 0:
        raise ValueError("top_k_checkpoints must be non-negative")
    if args.num_color_swaps < 0:
        raise ValueError("num_color_swaps must be non-negative")
    if args.top_k_checkpoints and args.top_k_checkpoint_dir is None:
        raise ValueError(
            "--top-k-checkpoint-dir is required when --top-k-checkpoints is positive"
        )
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("arc_pretrain_%Y%m%dT%H%M%SZ")
    log_file = args.log_file or Path("logs") / f"{run_name}.jsonl"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"run_name": run_name, "gpu": torch.cuda.get_device_name(device), "compute_dtype": "bfloat16", "training_weight_dtype": "float32", "fullgraph": not args.no_compile, "dynamic": False, "effective_batch_size": args.batch_size * args.minibatches})
    logger = Logger(log_file, config, run_name, args.wandb_project)
    tokenizer = ArcCharacterTokenizer()
    train_examples, augmentation_summary = load_pretraining_examples(args, tokenizer)
    train_examples, train_summary = apply_training_partition(train_examples, args)
    train_summary["augmentation_source"] = augmentation_summary
    eval_examples, eval_summary = load_arc_examples(
        args.data_dir, split="evaluation", tokenizer=tokenizer,
        usable_example_limit=args.eval_examples,
        max_total_tokens=args.eval_max_sequence_length,
        augment=False, seed=args.seed,
    )
    logger.log("dataset_train", train_summary)
    logger.log("dataset_eval", asdict(eval_summary))
    print(f"train_examples={len(train_examples)} eval_examples={len(eval_examples)}", flush=True)

    if args.from_scratch:
        base_model = build_seeded_model(args.seed)
    elif args.resume_checkpoint is None:
        base_model = load_pretrained_body(args.model_id)
    else:
        base_model = load_custom_checkpoint(args.resume_checkpoint)
    base_model.to(device=device, dtype=torch.bfloat16)
    baseline_golden = golden_eval_metrics(
        base_model,
        eval_examples,
        args.eval_batch_size,
        args.eval_max_sequence_length,
        device,
    )
    if args.skip_baseline_greedy_eval:
        from types import SimpleNamespace

        baseline = SimpleNamespace(
            loss=baseline_golden.target_loss,
            color_loss=baseline_golden.color_loss,
            num_correct=0,
        )
        logger.log(
            "baseline",
            {
                "loss": baseline.loss,
                "color_loss": baseline.color_loss,
                "color_tokens": baseline_golden.color_tokens,
                "selection_metric": args.selection_metric,
                "greedy_evaluated": False,
            },
        )
        print(
            f"baseline loss={baseline.loss:.6f} "
            f"color_loss={baseline.color_loss:.6f} greedy_evaluated=False",
            flush=True,
        )
    else:
        baseline = greedy_eval(base_model, tokenizer, eval_examples, args, device, 0)
        baseline_values = asdict(baseline)
        baseline_values.update(
            {
                "color_loss": baseline_golden.color_loss,
                "color_tokens": baseline_golden.color_tokens,
                "selection_metric": args.selection_metric,
            }
        )
        logger.log("baseline", baseline_values)
        print(
            f"baseline loss={baseline.loss:.6f} "
            f"color_loss={baseline_golden.color_loss:.6f} "
            f"exact={baseline.num_correct}/{baseline.num_samples} "
            f"reward={baseline.mean_reward:.6f} recall={baseline.pair_recall:.6f}",
            flush=True,
        )
    if args.eval_only:
        logger.close()
        return

    del base_model
    torch.cuda.empty_cache()
    # FP32 trainable weights and optimizer state preserve small continuation
    # updates; CUDA matrix compute remains BF16 under autocast below.
    if args.from_scratch:
        model = build_seeded_model(args.seed)
    elif args.resume_checkpoint is None:
        model = load_pretrained_body(args.model_id)
    else:
        model = load_custom_checkpoint(args.resume_checkpoint)
    model.to(device=device, dtype=torch.float32)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    # Qwen3/Transformers 5.10 creates an inference tensor that PyTorch 2.8
    # cannot mutate while recording a training CUDA graph. Keep Inductor full
    # graph compilation and autotuning, but disable only the replay layer.
    compiled_model = model
    if not args.no_compile:
        torch._inductor.config.triton.cudagraphs = False
        compile_options = {"triton.cudagraphs": False}
        if args.compile_mode == "max-autotune":
            compile_options.update(
                {"max_autotune": True, "coordinate_descent_tuning": True}
            )
        compiled_model = torch.compile(
            model, fullgraph=True, dynamic=False, options=compile_options
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        fused=True,
    )
    microbatches_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    steps_per_epoch = math.ceil(microbatches_per_epoch / args.minibatches)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    order_rng = random.Random(args.seed)
    baseline_selection_loss = (
        baseline_golden.color_loss
        if args.selection_metric == "color_nll"
        else baseline_golden.target_loss
    )
    best_loss = baseline.loss
    best_color_loss = baseline_golden.color_loss
    best_selection_loss = baseline_selection_loss
    plateau_best_loss = baseline_selection_loss
    best_checkpoint_path = args.output_checkpoint
    top_candidates: list[tuple[float, int, Path]] = []
    if args.top_k_checkpoint_dir is not None:
        args.top_k_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs_without_material_improvement = 0
    best_epoch = 0
    global_step = 0
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        completed_epoch = epoch
        order_rng.shuffle(train_examples)
        torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        tokens_seen = 0
        last_grad_norm = 0.0
        for group_start in range(0, microbatches_per_epoch, args.minibatches):
            group_size = min(
                args.minibatches, microbatches_per_epoch - group_start
            )
            optimizer.zero_grad(set_to_none=True)
            group_examples = train_examples[
                group_start * args.batch_size :
                (group_start + group_size) * args.batch_size
            ]
            group_tokens = sum(
                example.prompt_ids.numel() + example.target_ids.numel() - 1
                if args.loss_scope == "full"
                else example.target_ids.numel()
                for example in group_examples
            )
            group_loss_sum = 0.0
            for group_offset in range(group_size):
                microbatch_index = group_start + group_offset
                start = microbatch_index * args.batch_size
                chunk = list(train_examples[start : start + args.batch_size])
                input_ids, mask, labels, target_tokens = padded_batch(
                    chunk,
                    args.max_sequence_length,
                    device,
                    loss_scope=args.loss_scope,
                    batch_size=args.batch_size,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = target_loss(compiled_model, input_ids, mask, labels)
                (loss * target_tokens / group_tokens).backward()
                weighted_loss = float(loss.detach()) * target_tokens
                group_loss_sum += weighted_loss
                loss_sum += weighted_loss
                tokens_seen += target_tokens
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            global_step += 1
            step_values = {
                "epoch": epoch,
                "optimizer_step": global_step,
                "train_loss": group_loss_sum / group_tokens,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "tokens": group_tokens,
                "tokens_seen_epoch": tokens_seen,
                "grad_norm": last_grad_norm,
                "seconds_since_epoch_start": time.perf_counter() - epoch_start,
            }
            logger.log("step", step_values)
            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"train_loss={step_values['train_loss']:.6f} "
                    f"lr={step_values['learning_rate']:.3e}",
                    flush=True,
                )
        seconds = time.perf_counter() - epoch_start
        eval_golden = golden_eval_metrics(
            model,
            eval_examples,
            args.eval_batch_size,
            args.eval_max_sequence_length,
            device,
        )
        eval_loss = eval_golden.target_loss
        eval_color_loss = eval_golden.color_loss
        eval_selection_loss = (
            eval_color_loss if args.selection_metric == "color_nll" else eval_loss
        )
        metrics = EpochMetrics(
            epoch=epoch, optimizer_step=global_step,
            train_loss=loss_sum / tokens_seen, eval_loss=eval_loss,
            eval_color_loss=eval_color_loss,
            eval_selection_loss=eval_selection_loss,
            eval_color_tokens=eval_golden.color_tokens,
            learning_rate=optimizer.param_groups[0]["lr"], tokens_seen=tokens_seen,
            seconds=seconds, tokens_per_second=tokens_seen / seconds,
            grad_norm=last_grad_norm,
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        )
        logger.log("epoch", asdict(metrics))
        print(
            f"epoch={epoch} train_loss={metrics.train_loss:.6f} "
            f"eval_loss={eval_loss:.6f} color_loss={eval_color_loss:.6f} "
            f"selection_loss={eval_selection_loss:.6f} "
            f"tok_s={metrics.tokens_per_second:.1f} "
            f"peak_gib={metrics.cuda_peak_reserved_gib:.2f}",
            flush=True,
        )
        candidate_checkpoint_path = None
        if args.top_k_checkpoints > 0:
            qualifies = (
                len(top_candidates) < args.top_k_checkpoints
                or eval_selection_loss < top_candidates[-1][0]
            )
            if qualifies:
                metric_label = args.selection_metric.replace("_nll", "")
                candidate_checkpoint_path = args.top_k_checkpoint_dir / (
                    f"epoch_{epoch:03d}_step_{global_step:06d}_"
                    f"{metric_label}_{eval_selection_loss:.6f}.safetensors"
                )
                save_custom_checkpoint(model, candidate_checkpoint_path)
                top_candidates.append(
                    (eval_selection_loss, epoch, candidate_checkpoint_path)
                )
                top_candidates.sort(key=lambda item: (item[0], item[1]))
                while len(top_candidates) > args.top_k_checkpoints:
                    removed_loss, removed_epoch, removed_path = top_candidates.pop()
                    removed_path.unlink(missing_ok=True)
                    logger.log(
                        "top_k_checkpoint_removed",
                        {
                            "epoch": removed_epoch,
                            "selection_loss": removed_loss,
                            "path": str(removed_path.resolve()),
                        },
                    )
                logger.log(
                    "top_k_checkpoint",
                    {
                        "epoch": epoch,
                        "optimizer_step": global_step,
                        "selection_metric": args.selection_metric,
                        "selection_loss": eval_selection_loss,
                        "eval_loss": eval_loss,
                        "eval_color_loss": eval_color_loss,
                        "path": str(candidate_checkpoint_path.resolve()),
                        "ranked_paths": [
                            str(path.resolve()) for _, _, path in top_candidates
                        ],
                    },
                )
        if eval_selection_loss < best_selection_loss:
            best_selection_loss = eval_selection_loss
            best_loss = eval_loss
            best_color_loss = eval_color_loss
            best_epoch = epoch
            if candidate_checkpoint_path is None:
                save_custom_checkpoint(model, args.output_checkpoint)
                best_checkpoint_path = args.output_checkpoint
            else:
                best_checkpoint_path = candidate_checkpoint_path
            logger.log(
                "best_checkpoint",
                {
                    "epoch": epoch,
                    "eval_loss": eval_loss,
                    "eval_color_loss": eval_color_loss,
                    "selection_metric": args.selection_metric,
                    "selection_loss": eval_selection_loss,
                    "path": str(best_checkpoint_path.resolve()),
                },
            )
        if eval_selection_loss < plateau_best_loss - args.min_eval_improvement:
            plateau_best_loss = eval_selection_loss
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1
        logger.log(
            "plateau",
            {
                "epoch": epoch,
                "optimizer_step": global_step,
                "plateau_best_loss": plateau_best_loss,
                "selection_metric": args.selection_metric,
                "epochs_without_material_improvement": epochs_without_material_improvement,
                "min_eval_improvement": args.min_eval_improvement,
            },
        )
        if (
            args.early_stop_patience > 0
            and epochs_without_material_improvement >= args.early_stop_patience
        ):
            print(
                f"early_stop plateau epoch={epoch} "
                f"best_selection_loss={best_selection_loss:.6f} "
                f"epochs_without_material_improvement={epochs_without_material_improvement}",
                flush=True,
            )
            logger.log(
                "early_stop",
                {
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "best_loss": best_loss,
                    "best_color_loss": best_color_loss,
                    "best_selection_loss": best_selection_loss,
                    "selection_metric": args.selection_metric,
                    "reason": "eval_loss_plateau",
                },
            )
            break

    if args.latest_checkpoint is not None:
        save_custom_checkpoint(model, args.latest_checkpoint)
        logger.log(
            "latest_checkpoint",
            {
                "epoch": completed_epoch,
                "optimizer_step": global_step,
                "path": str(args.latest_checkpoint.resolve()),
            },
        )
    if top_candidates:
        logger.log(
            "top_k_summary",
            {
                "selection_metric": args.selection_metric,
                "checkpoints": [
                    {
                        "rank": rank,
                        "selection_loss": selection_loss,
                        "epoch": epoch,
                        "path": str(path.resolve()),
                    }
                    for rank, (selection_loss, epoch, path) in enumerate(
                        top_candidates, start=1
                    )
                ],
            },
        )
    if best_epoch == 0 and args.save_last_candidate:
        best_epoch = completed_epoch
        save_custom_checkpoint(model, args.output_checkpoint)
        logger.log("candidate_checkpoint", {"epoch": best_epoch, "path": str(args.output_checkpoint.resolve()), "reason": "matched_bfloat16_reload_evaluation"})
    if best_epoch == 0:
        print(
            f"No epoch improved baseline {args.selection_metric}; "
            "no replacement checkpoint promoted",
            flush=True,
        )
        logger.log(
            "result",
            {
                "improved": False,
                "baseline_loss": baseline.loss,
                "baseline_color_loss": baseline_golden.color_loss,
                "best_loss": best_loss,
                "best_color_loss": best_color_loss,
                "selection_metric": args.selection_metric,
                "top_k_retained": len(top_candidates),
            },
        )
        logger.close()
        return
    del compiled_model, model, optimizer
    torch.cuda.empty_cache()
    best_model = load_custom_checkpoint(best_checkpoint_path)
    best_model.to(device=device, dtype=torch.bfloat16)
    if args.skip_final_greedy_eval:
        final_golden = golden_eval_metrics(
            best_model,
            eval_examples,
            args.eval_batch_size,
            args.eval_max_sequence_length,
            device,
        )
        logger.log(
            "result",
            {
                "improved": best_selection_loss < baseline_selection_loss,
                "best_epoch": best_epoch,
                "best_checkpoint": str(best_checkpoint_path.resolve()),
                "baseline_loss": baseline.loss,
                "baseline_color_loss": baseline_golden.color_loss,
                "best_loss": best_loss,
                "best_color_loss": best_color_loss,
                "final_loss": final_golden.target_loss,
                "final_color_loss": final_golden.color_loss,
                "selection_metric": args.selection_metric,
                "best_selection_loss": best_selection_loss,
                "greedy_evaluated": False,
            },
        )
        print(
            f"final loss={final_golden.target_loss:.6f} "
            f"color_loss={final_golden.color_loss:.6f} "
            "greedy_evaluated=False",
            flush=True,
        )
        logger.close()
        return
    final = greedy_eval(
        best_model,
        tokenizer,
        eval_examples,
        args,
        device,
        best_epoch * steps_per_epoch,
    )
    logger.log("final", asdict(final))
    promoted = final.loss < baseline.loss and final.num_correct >= baseline.num_correct
    logger.log("result", {"improved": promoted, "best_epoch": best_epoch, "baseline_loss": baseline.loss, "final_loss": final.loss, "baseline_exact": baseline.num_correct, "final_exact": final.num_correct})
    print(f"final loss={final.loss:.6f} exact={final.num_correct}/{final.num_samples} reward={final.mean_reward:.6f} recall={final.pair_recall:.6f}", flush=True)
    logger.close()


if __name__ == "__main__":
    main()

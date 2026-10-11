"""Memory-conscious ARC SFT for Qwen3.5-4B-Base with RSLoRA.

This starts a fresh ARC experiment from the pretrained Qwen text/vision
checkpoint. The text transformer body is frozen in BF16, selected projections
receive rank-stabilized LoRA adapters, and the original 248k-token interface is
replaced by a freshly initialized, trainable 24-token ARC embedding/lm_head.
The two token layers remain tied, matching the source Qwen architecture.

The defaults target a single 16 GiB BF16-capable NVIDIA GPU: microbatch one,
dynamic length padding, length bucketing, gradient accumulation and activation
checkpointing and an 8-bit optimizer. Outputs are PEFT adapter directories containing the RSLoRA
weights plus the small trainable token interface.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from . import pretrain_qwen3 as arc_trainer
from .task_partition import (
    filter_partition_examples,
    load_partition_ids,
    logical_task_id,
)
from .rl_static_cache import (
    ARC_CHARACTERS,
    ARCExample,
    DEFAULT_ARC_DATA,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    ArcCharacterTokenizer,
    load_arc_examples,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
DEFAULT_PARTITION = Path("data_splits/qwen35_arc_full8192_sft60_rl40_seed3407.json")
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
COLON_TOKEN_ID = ARC_CHARACTERS.index(":")


@dataclass(frozen=True)
class EvalMetrics:
    target_nll: float
    color_nll: float
    target_tokens: int
    color_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/qwen35_4b_arc_rslora")
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ARC_DATA)
    parser.add_argument("--train-task-partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--train-partition-role", choices=("sft", "rl"), default="sft")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--bucket-size", type=int, default=32)
    parser.add_argument("--pad-to-multiple-of", type=int, default=64)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--min-train-sequence-length", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2816)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--eval-max-sequence-length", type=int, default=4096)
    parser.add_argument("--num-color-swaps", type=int, default=4)
    parser.add_argument("--no-rotations", action="store_true")
    parser.add_argument("--loss-scope", choices=("full", "target"), default="full")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--token-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--min-eval-improvement", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default=",".join(DEFAULT_LORA_TARGETS),
        help="Comma-separated text-module suffixes. Do not use all-linear: it hits vision.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run-name", default="qwen35_4b_arc_rslora_16gb")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--wandb-project", default="ARC-Qwen35-4B-RSLoRA")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.bucket_size,
        args.pad_to_multiple_of,
        args.max_sequence_length,
        args.max_new_tokens,
        args.eval_examples,
        args.eval_max_sequence_length,
        args.lora_rank,
        args.lora_alpha,
    )
    if min(positive) <= 0:
        raise ValueError("epochs, sizes, lengths and LoRA dimensions must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if args.learning_rate <= 0 or args.token_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("weight_decay must be non-negative and max_grad_norm positive")
    if args.early_stop_patience < 0 or args.min_eval_improvement < 0:
        raise ValueError("early-stop settings must be non-negative")
    if args.num_color_swaps < 0:
        raise ValueError("num_color_swaps must be non-negative")
    if not 0 <= args.min_train_sequence_length <= args.max_sequence_length:
        raise ValueError("min_train_sequence_length must be within the training limit")
    if not [target for target in args.lora_targets.split(",") if target.strip()]:
        raise ValueError("lora_targets cannot be empty")


def initialize_arc_token_interface(model, seed: int) -> None:
    """Initialize the already resized, tied 24-token interface deterministically."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.nn.init.normal_(
            model.get_input_embeddings().weight,
            mean=0.0,
            std=float(model.config.initializer_range),
        )
    model.tie_weights()
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.shape != (VOCAB_SIZE, model.config.hidden_size):
        raise RuntimeError(f"unexpected ARC embedding shape: {tuple(input_weight.shape)}")
    if output_weight.data_ptr() != input_weight.data_ptr():
        raise RuntimeError("Qwen ARC embedding and lm_head must remain tied")


def build_model(args: argparse.Namespace):
    # The published checkpoint is multimodal. Copy only the pretrained text
    # body so the unused vision tower consumes neither VRAM nor adapter slots.
    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    text_config = deepcopy(source.config.text_config)
    text_config.vocab_size = VOCAB_SIZE
    text_config.pad_token_id = PAD_TOKEN_ID
    text_config.eos_token_id = EOS_TOKEN_ID
    text_config.bos_token_id = None
    text_config.tie_word_embeddings = True
    text_config.use_cache = False
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Qwen3_5ForCausalLM(text_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    inherited = {
        f"model.{name}": tensor
        for name, tensor in source.model.language_model.state_dict().items()
        if name != "embed_tokens.weight"
    }
    incompatible = model.load_state_dict(inherited, strict=False)
    expected_missing = {"model.embed_tokens.weight", "lm_head.weight"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "unexpected pretrained text-weight mismatch: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    initialize_arc_token_interface(model, args.seed)
    model.config.use_cache = False
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    model.generation_config.eos_token_id = EOS_TOKEN_ID
    del source, inherited
    gc.collect()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora_targets = [target.strip() for target in args.lora_targets.split(",") if target.strip()]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        use_rslora=True,
        ensure_weight_tying=True,
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT produced no trainable parameters")
    if not any("lora_" in name for name, _ in trainable):
        raise RuntimeError("no RSLoRA parameters are trainable")
    if not any(
        "embed_tokens" in name or "lm_head" in name for name, _ in trainable
    ):
        raise RuntimeError("the tied ARC embedding/lm_head is not trainable")
    model.to(device=torch.device("cuda:0"), dtype=torch.bfloat16)
    return model


def length_bucketed_batches(
    examples: Sequence[ARCExample],
    *,
    batch_size: int,
    bucket_size: int,
    rng: random.Random,
) -> list[list[ARCExample]]:
    ordered = sorted(
        examples, key=lambda item: item.prompt_ids.numel() + item.target_ids.numel()
    )
    buckets = [ordered[start : start + bucket_size] for start in range(0, len(ordered), bucket_size)]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    return [
        bucket[start : start + batch_size]
        for bucket in buckets
        for start in range(0, len(bucket), batch_size)
    ]


def dynamic_batch(
    examples: Sequence[ARCExample],
    *,
    max_length: int,
    multiple_of: int,
    loss_scope: str,
    device: torch.device,
):
    actual = max(item.prompt_ids.numel() + item.target_ids.numel() for item in examples)
    width = min(max_length, int(math.ceil(actual / multiple_of) * multiple_of))
    return arc_trainer.padded_batch(
        examples,
        width,
        device,
        loss_scope=loss_scope,
        batch_size=len(examples),
    )


def apply_length_compatible_partition(examples, args):
    """Apply the logical split while allowing a shorter context than its manifest.

    The canonical manifest was built at 8192 tokens. A 4096-token 16 GiB run
    necessarily omits families whose every augmentation is longer; this is
    reported rather than treated as a corrupt partition.
    """

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
        omitted_ids = sorted(allowed_ids - selected_ids)
        summary.update(
            {
                "partition_path": str(args.train_task_partition.resolve()),
                "partition_role": args.train_partition_role,
                "partition_seed": manifest.get("seed"),
                "partition_requested_logical_tasks": len(allowed_ids),
                "partition_logical_tasks": len(selected_ids),
                "length_omitted_logical_tasks": len(omitted_ids),
                "length_omitted_task_ids": omitted_ids,
            }
        )
        if omitted_ids:
            print(
                f"length_filter omitted {len(omitted_ids)}/{len(allowed_ids)} "
                f"{args.train_partition_role} task families at "
                f"max_sequence_length={args.max_sequence_length}",
                flush=True,
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


def causal_loss_and_sums(model, input_ids, attention_mask, labels):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        shifted_logits = logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        losses = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shifted_labels)
    valid = shifted_labels != -100
    color = valid & (input_ids[:, :-1] == COLON_TOKEN_ID)
    return (
        losses[valid].mean(),
        losses[valid].sum().detach(),
        int(valid.sum()),
        losses[color].sum().detach(),
        int(color.sum()),
    )


@torch.no_grad()
def evaluate(model, examples, args, device) -> EvalMetrics:
    was_training = model.training
    model.eval()
    target_sum = color_sum = 0.0
    target_tokens = color_tokens = 0
    for start in range(0, len(examples), args.batch_size):
        chunk = examples[start : start + args.batch_size]
        input_ids, attention_mask, labels, _ = dynamic_batch(
            chunk,
            max_length=args.eval_max_sequence_length,
            multiple_of=args.pad_to_multiple_of,
            loss_scope="target",
            device=device,
        )
        _, target_loss_sum, valid, color_loss_sum, colors = causal_loss_and_sums(
            model, input_ids, attention_mask, labels
        )
        target_sum += float(target_loss_sum)
        color_sum += float(color_loss_sum)
        target_tokens += valid
        color_tokens += colors
    model.train(was_training)
    if target_tokens == 0 or color_tokens == 0:
        raise RuntimeError("evaluation produced no supervised ARC tokens")
    return EvalMetrics(
        target_nll=target_sum / target_tokens,
        color_nll=color_sum / color_tokens,
        target_tokens=target_tokens,
        color_tokens=color_tokens,
    )


def optimizer_groups(model, args) -> list[dict]:
    lora_parameters = []
    token_parameters = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if "lora_" in name:
            lora_parameters.append(parameter)
        elif "embed_tokens" in name or "lm_head" in name:
            token_parameters.append(parameter)
        else:
            raise RuntimeError(f"unexpected trainable parameter: {name}")
    if not lora_parameters or not token_parameters:
        raise RuntimeError("optimizer requires both RSLoRA and ARC token parameters")
    return [
        {
            "params": lora_parameters,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "group_name": "rslora",
        },
        {
            "params": token_parameters,
            "lr": args.token_learning_rate,
            "weight_decay": args.weight_decay,
            "group_name": "arc_token_interface",
        },
    ]


def save_adapter(model, path: Path, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    (path / "arc_rslora_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "arc_character_vocabulary.json").write_text(
        json.dumps(
            {
                "characters": list(ARC_CHARACTERS),
                "pad_token_id": PAD_TOKEN_ID,
                "eos_token_id": EOS_TOKEN_ID,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def train(model, train_examples, eval_examples, args, logger, device) -> None:
    import bitsandbytes as bnb

    groups = optimizer_groups(model, args)
    optimizer = bnb.optim.PagedAdamW8bit(groups)
    batches_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    updates_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = int(total_updates * args.warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )
    rng = random.Random(args.seed)
    trainable_parameters = [parameter for group in groups for parameter in group["params"]]
    best_nll = math.inf
    epochs_without_improvement = 0
    optimizer_step = 0

    baseline = evaluate(model, eval_examples, args, device)
    logger.log("baseline", asdict(baseline))
    print(
        f"baseline target_nll={baseline.target_nll:.6f} "
        f"color_nll={baseline.color_nll:.6f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        batches = length_bucketed_batches(
            train_examples,
            batch_size=args.batch_size,
            bucket_size=args.bucket_size,
            rng=rng,
        )
        loss_sum = 0.0
        tokens_seen = 0
        last_grad_norm = 0.0
        for group_start in range(0, len(batches), args.gradient_accumulation_steps):
            accumulation = batches[
                group_start : group_start + args.gradient_accumulation_steps
            ]
            optimizer.zero_grad(set_to_none=True)
            group_tokens = sum(
                sum(
                    item.prompt_ids.numel() + item.target_ids.numel() - 1
                    if args.loss_scope == "full"
                    else item.target_ids.numel()
                    for item in batch
                )
                for batch in accumulation
            )
            for batch in accumulation:
                input_ids, attention_mask, labels, supervised_tokens = dynamic_batch(
                    batch,
                    max_length=args.max_sequence_length,
                    multiple_of=args.pad_to_multiple_of,
                    loss_scope=args.loss_scope,
                    device=device,
                )
                loss, _, _, _, _ = causal_loss_and_sums(
                    model, input_ids, attention_mask, labels
                )
                (loss * supervised_tokens / group_tokens).backward()
                loss_sum += float(loss.detach()) * supervised_tokens
                tokens_seen += supervised_tokens
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
            if optimizer_step == 1 or optimizer_step % 10 == 0:
                logger.log(
                    "step",
                    {
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "train_loss": loss_sum / max(tokens_seen, 1),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "token_learning_rate": optimizer.param_groups[1]["lr"],
                        "grad_norm": last_grad_norm,
                        "tokens_seen_epoch": tokens_seen,
                    },
                )
                print(
                    f"step={optimizer_step} epoch={epoch} "
                    f"loss={loss_sum / max(tokens_seen, 1):.6f} "
                    f"grad_norm={last_grad_norm:.3f}",
                    flush=True,
                )

        metrics = evaluate(model, eval_examples, args, device)
        seconds = time.perf_counter() - epoch_started
        epoch_values = {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "train_loss": loss_sum / max(tokens_seen, 1),
            **asdict(metrics),
            "seconds": seconds,
            "tokens_per_second": tokens_seen / seconds,
            "grad_norm": last_grad_norm,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        logger.log("epoch", epoch_values)
        print(
            f"epoch={epoch} train_loss={epoch_values['train_loss']:.6f} "
            f"eval_nll={metrics.target_nll:.6f} color_nll={metrics.color_nll:.6f} "
            f"peak_gib={epoch_values['cuda_peak_reserved_gib']:.2f}",
            flush=True,
        )
        metadata = {
            "base_model_id": args.model_id,
            "tokenizer": "ArcCharacterTokenizer",
            "vocab_size": VOCAB_SIZE,
            "tie_word_embeddings": True,
            "use_rslora": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": args.lora_targets.split(","),
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "eval": asdict(metrics),
        }
        save_adapter(model, args.output_dir / "latest", metadata)
        if metrics.target_nll < best_nll - args.min_eval_improvement:
            best_nll = metrics.target_nll
            epochs_without_improvement = 0
            save_adapter(model, args.output_dir / "best", metadata)
            logger.log("best_checkpoint", {"path": str(args.output_dir / "best"), **metadata})
        else:
            epochs_without_improvement += 1
        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            logger.log(
                "early_stop",
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "epochs_without_improvement": epochs_without_improvement,
                    "best_target_nll": best_nll,
                },
            )
            print(f"early_stop epoch={epoch} best_nll={best_nll:.6f}", flush=True)
            break


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a CUDA GPU with BF16 support is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_file = args.log_file or Path("logs") / f"{args.run_name}.jsonl"
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "frozen_body_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "optimizer": "PagedAdamW8bit",
            "use_rslora": True,
            "fresh_arc_token_interface": True,
            "gpu": torch.cuda.get_device_name(device),
        }
    )
    logger = arc_trainer.Logger(log_file, config, args.run_name, args.wandb_project)
    tokenizer = ArcCharacterTokenizer()
    train_examples, augmentation = arc_trainer.load_pretraining_examples(args, tokenizer)
    train_examples, train_summary = apply_length_compatible_partition(train_examples, args)
    train_summary["augmentation_source"] = augmentation
    eval_examples, eval_summary = load_arc_examples(
        args.data_dir,
        split="evaluation",
        tokenizer=tokenizer,
        usable_example_limit=args.eval_examples,
        max_total_tokens=args.eval_max_sequence_length,
        augment=False,
        seed=args.seed,
    )
    logger.log("dataset_train", train_summary)
    logger.log("dataset_eval", asdict(eval_summary))
    print(
        f"train_examples={len(train_examples)} eval_examples={len(eval_examples)}",
        flush=True,
    )
    model = build_model(args)
    model.print_trainable_parameters()
    logger.log(
        "model",
        {
            "all_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    )
    try:
        train(model, train_examples, eval_examples, args, logger, device)
    finally:
        logger.close()


if __name__ == "__main__":
    main()

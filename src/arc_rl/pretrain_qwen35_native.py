"""ARC continuation pretraining for native-tokenizer Qwen3.5.

The underlying ARC text serialization and train/evaluation task selection are
the same as in ``arc_rl.pretrain_qwen3``. Unlike that experiment, this file
keeps the complete pretrained vocabulary, tokenizer, embeddings, and LM head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from .pretrain_qwen3 import Logger, load_pretraining_examples
from .rl_static_cache import (
    ARCExample,
    ArcCharacterTokenizer,
    DEFAULT_ARC_DATA,
    load_arc_examples,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


@dataclass(frozen=True)
class NativeARCExample:
    task_id: str
    prompt_ids: torch.Tensor
    target_ids: torch.Tensor
    target_text: str


@dataclass(frozen=True)
class NativeEvalMetrics:
    step: int
    loss: float
    bits_per_character: float
    num_correct: int
    num_samples: int
    mean_length: float
    min_length: int
    max_length: int
    good_pairs: int
    candidate_pairs: int
    pair_precision: float
    covered_target_pairs: int
    target_pairs: int
    pair_recall: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-checkpoint", type=Path, default=Path("checkpoints/qwen35_0.8b_arc_native_best"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ARC_DATA)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4, help="Examples per GPU microbatch.")
    parser.add_argument(
        "--minibatches", "--gradient-accumulation-steps",
        type=int, default=8,
        help="Microbatches accumulated before each optimizer update.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without a material eval improvement; zero disables it.",
    )
    parser.add_argument(
        "--min-eval-improvement",
        type=float,
        default=0.0,
        help="Absolute eval-loss decrease required to reset plateau patience.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-scope", choices=("full", "target"), default="full")
    parser.add_argument("--max-sequence-length", type=int, default=2816, help="ARC character-token filter, shared with the custom-tokenizer runs.")
    parser.add_argument("--max-new-tokens", type=int, default=2816, help="ARC character-token target filter, shared with the custom-tokenizer runs.")
    parser.add_argument("--eval-examples", type=int, default=150)
    parser.add_argument("--eval-max-sequence-length", type=int, default=2816)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-color-swaps", type=int, default=2)
    parser.add_argument("--no-rotations", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--compile", action="store_true", help="Compile the text training forward with torch.compile.")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute layer activations during backward to reduce peak VRAM.",
    )
    parser.add_argument(
        "--skip-baseline-greedy-eval",
        action="store_true",
        help="Compute baseline golden NLL without an expensive pre-SFT rollout.",
    )
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="max-autotune")
    parser.add_argument("--run-name")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--wandb-project", default="ARC-Pretraining")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.minibatches <= 0:
        raise ValueError("epochs, batch size, and accumulation steps must be positive")
    if args.eval_examples <= 0 or args.eval_batch_size <= 0:
        raise ValueError("evaluation sizes must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not 0 <= args.warmup_fraction <= 1:
        raise ValueError("warmup fraction must be between zero and one")
    if args.max_grad_norm <= 0 or args.num_color_swaps < 0:
        raise ValueError("gradient norm must be positive and color swaps non-negative")
    if args.early_stop_patience < 0 or args.min_eval_improvement < 0:
        raise ValueError("plateau patience and minimum improvement must be non-negative")


def _convert_examples(
    examples: Sequence[ARCExample],
    character_tokenizer: ArcCharacterTokenizer,
    native_tokenizer,
) -> list[NativeARCExample]:
    converted = []
    eos_id = native_tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The native tokenizer must define eos_token_id")
    for example in examples:
        prompt_text = character_tokenizer.decode(example.prompt_ids)
        target_text = character_tokenizer.decode(example.target_ids)
        if not target_text.endswith(character_tokenizer.eos_token):
            raise ValueError(f"ARC target lacks its terminal marker: {example.task_id}")
        # Replace the ARC character-level E terminator with Qwen's native EOS.
        target_body = target_text[: -len(character_tokenizer.eos_token)]
        prompt_ids = native_tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = native_tokenizer.encode(target_body, add_special_tokens=False)
        target_ids.append(eos_id)
        converted.append(
            NativeARCExample(
                task_id=example.task_id,
                prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
                target_ids=torch.tensor(target_ids, dtype=torch.long),
                target_text=target_text,
            )
        )
    return converted


def load_datasets(args, native_tokenizer):
    character_tokenizer = ArcCharacterTokenizer()
    train_character, train_summary = load_pretraining_examples(args, character_tokenizer)
    eval_character, eval_summary = load_arc_examples(
        args.data_dir,
        split="evaluation",
        tokenizer=character_tokenizer,
        usable_example_limit=args.eval_examples,
        max_total_tokens=args.eval_max_sequence_length,
        augment=False,
        seed=args.seed,
    )
    train = _convert_examples(train_character, character_tokenizer, native_tokenizer)
    evaluation = _convert_examples(eval_character, character_tokenizer, native_tokenizer)
    return train, evaluation, train_summary, eval_summary


def padded_batch(
    examples: Sequence[NativeARCExample],
    padded_length: int,
    pad_token_id: int,
    device: torch.device,
    *,
    loss_scope: str,
    batch_size: int | None = None,
):
    rows = batch_size or len(examples)
    input_ids = torch.full((rows, padded_length), pad_token_id, dtype=torch.long)
    labels = torch.full((rows, padded_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((rows, padded_length), dtype=torch.long)
    loss_tokens = 0
    for row, example in enumerate(examples):
        sequence = torch.cat((example.prompt_ids, example.target_ids))
        if sequence.numel() > padded_length:
            raise ValueError(
                f"{example.task_id} has {sequence.numel()} native tokens, "
                f"padded length is {padded_length}"
            )
        prompt_length = example.prompt_ids.numel()
        length = sequence.numel()
        input_ids[row, :length] = sequence
        attention_mask[row, :length] = 1
        if loss_scope == "full":
            labels[row, :length] = sequence
            loss_tokens += length - 1
        else:
            labels[row, prompt_length:length] = example.target_ids
            loss_tokens += example.target_ids.numel()
    return (
        input_ids.to(device, non_blocking=True),
        attention_mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        loss_tokens,
    )


def causal_loss(model, input_ids, attention_mask, labels):
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    # Keeping logits in BF16 avoids a multi-gigabyte FP32 copy for the native
    # 248k-token vocabulary. Cross entropy still uses its stable fused kernel.
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _pair_list(text: str) -> list[str]:
    text = text.split("E", 1)[0]
    return [piece for piece in text.split() if piece]


def _prediction_stats(targets: Sequence[str], predictions: Sequence[str]):
    correct = good = candidates = covered = target_count = 0
    for target_text, predicted_text in zip(targets, predictions):
        target = set(_pair_list(target_text))
        predicted_list = _pair_list(predicted_text)
        predicted = set(predicted_list)
        correct += target == predicted
        good += sum(piece in target for piece in predicted_list)
        candidates += len(predicted_list)
        covered += len(predicted & target)
        target_count += len(target)
    return correct, good, candidates, covered, target_count


def golden_eval(
    model,
    examples: Sequence[NativeARCExample],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
):
    was_training = model.training
    model.eval()
    nll_sum = 0.0
    token_count = 0
    character_count = sum(len(example.target_text) for example in examples)
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                chunk = list(examples[start : start + batch_size])
                length = max(
                    example.prompt_ids.numel() + example.target_ids.numel()
                    for example in chunk
                )
                padded_length = math.ceil(length / 16) * 16
                ids, mask, labels, tokens = padded_batch(
                    chunk,
                    padded_length,
                    pad_token_id,
                    device,
                    loss_scope="target",
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = causal_loss(model, ids, mask, labels)
                nll_sum += float(loss) * tokens
                token_count += tokens
    finally:
        model.train(was_training)
    loss = nll_sum / token_count
    bits_per_character = nll_sum / character_count / math.log(2)
    return loss, bits_per_character


def greedy_predictions(
    model,
    tokenizer,
    examples: Sequence[NativeARCExample],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
):
    was_training = model.training
    model.eval()
    predictions: list[str] = []
    lengths: list[int] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                chunk = list(examples[start : start + batch_size])
                prompt_length = max(example.prompt_ids.numel() for example in chunk)
                input_ids = torch.full(
                    (len(chunk), prompt_length),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row, example in enumerate(chunk):
                    length = example.prompt_ids.numel()
                    input_ids[row, prompt_length - length :] = example.prompt_ids.to(device)
                    attention_mask[row, prompt_length - length :] = 1
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )[:, prompt_length:]
                for row in generated:
                    ids = row.tolist()
                    if tokenizer.eos_token_id in ids:
                        ids = ids[: ids.index(tokenizer.eos_token_id)]
                    ids = [item for item in ids if item != tokenizer.pad_token_id]
                    lengths.append(len(ids))
                    predictions.append(tokenizer.decode(ids, skip_special_tokens=True))
    finally:
        model.train(was_training)
    return predictions, lengths


def evaluate(
    model,
    tokenizer,
    examples: Sequence[NativeARCExample],
    *,
    step: int,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> NativeEvalMetrics:
    loss, bits_per_character = golden_eval(
        model, examples, batch_size, tokenizer.pad_token_id, device
    )
    predictions, lengths = greedy_predictions(
        model, tokenizer, examples, batch_size, max_new_tokens, device
    )
    targets = [example.target_text for example in examples]
    correct, good, candidates, covered, target_count = _prediction_stats(
        targets, predictions
    )
    return NativeEvalMetrics(
        step=step,
        loss=loss,
        bits_per_character=bits_per_character,
        num_correct=correct,
        num_samples=len(examples),
        mean_length=sum(lengths) / len(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        good_pairs=good,
        candidate_pairs=candidates,
        pair_precision=good / max(candidates, 1),
        covered_target_pairs=covered,
        target_pairs=target_count,
        pair_recall=covered / max(target_count, 1),
    )


def load_model(model_id_or_path: str | Path, device: torch.device, dtype: torch.dtype):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_id_or_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    return model


def save_checkpoint(model, tokenizer, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        path,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(path)
    return path


def main() -> None:
    args = parse_args()
    validate_args(args)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_examples, eval_examples, train_summary, eval_summary = load_datasets(
        args, tokenizer
    )
    training_length = max(
        example.prompt_ids.numel() + example.target_ids.numel()
        for example in train_examples
    )
    training_padded_length = math.ceil(training_length / 128) * 128
    eval_target_max = max(example.target_ids.numel() for example in eval_examples)
    generation_max_new_tokens = min(args.max_new_tokens, eval_target_max + 256)

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "arc_qwen35_native_%Y%m%dT%H%M%S_%fZ"
    )
    log_file = args.log_file or Path("logs") / f"{run_name}.jsonl"
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "run_name": run_name,
            "effective_batch_size": args.batch_size * args.minibatches,
            "native_training_padded_length": training_padded_length,
            "generation_max_new_tokens": generation_max_new_tokens,
            "gpu": torch.cuda.get_device_name(device),
            "compute_dtype": "bfloat16",
            "training_weight_dtype": "float32",
            "tokenizer": args.model_id,
        }
    )
    logger = Logger(log_file, config, run_name, args.wandb_project)
    logger.log("dataset_train", train_summary)
    logger.log("dataset_eval", asdict(eval_summary))
    logger.log(
        "native_dataset",
        {
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "training_padded_length": training_padded_length,
            "eval_max_target_tokens": eval_target_max,
            "generation_max_new_tokens": generation_max_new_tokens,
            "eval_task_ids": [example.task_id for example in eval_examples],
        },
    )
    print(
        f"train_examples={len(train_examples)} eval_examples={len(eval_examples)} "
        f"native_padded_length={training_padded_length} "
        f"generation_max_new_tokens={generation_max_new_tokens}",
        flush=True,
    )

    base_model = load_model(args.model_id, device, torch.bfloat16)
    parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
    logger.log("model", {"parameters": parameter_count, "model_id": args.model_id})
    if args.skip_baseline_greedy_eval:
        from types import SimpleNamespace

        baseline_loss, baseline_bpc = golden_eval(
            base_model,
            eval_examples,
            args.eval_batch_size,
            tokenizer.pad_token_id,
            device,
        )
        baseline = SimpleNamespace(
            loss=baseline_loss,
            bits_per_character=baseline_bpc,
            num_correct=0,
        )
        logger.log(
            "baseline",
            {
                "loss": baseline.loss,
                "bits_per_character": baseline.bits_per_character,
                "greedy_evaluated": False,
            },
        )
        print(
            f"baseline loss={baseline.loss:.6f} "
            f"bpc={baseline.bits_per_character:.6f} greedy_evaluated=False",
            flush=True,
        )
    else:
        baseline = evaluate(
            base_model,
            tokenizer,
            eval_examples,
            step=0,
            batch_size=args.eval_batch_size,
            max_new_tokens=generation_max_new_tokens,
            device=device,
        )
        logger.log("baseline", asdict(baseline))
        print(
            f"baseline loss={baseline.loss:.6f} bpc={baseline.bits_per_character:.6f} "
            f"exact={baseline.num_correct}/{baseline.num_samples} "
            f"recall={baseline.pair_recall:.6f}",
            flush=True,
        )
    if args.eval_only:
        logger.close()
        return
    del base_model
    torch.cuda.empty_cache()

    model = load_model(args.model_id, device, torch.float32)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    training_model = model
    if args.compile:
        torch._inductor.config.triton.cudagraphs = False
        compile_options = {"triton.cudagraphs": False}
        if args.compile_mode == "max-autotune":
            compile_options.update(
                {"max_autotune": True, "coordinate_descent_tuning": True}
            )
        training_model = torch.compile(
            model, fullgraph=True, dynamic=False, options=compile_options
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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
    global_step = 0
    best_loss = baseline.loss
    plateau_best_loss = baseline.loss
    epochs_without_material_improvement = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        order_rng.shuffle(train_examples)
        torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        for group_start in range(0, microbatches_per_epoch, args.minibatches):
            group_size = min(args.minibatches, microbatches_per_epoch - group_start)
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
            for offset in range(group_size):
                start = (group_start + offset) * args.batch_size
                chunk = list(train_examples[start : start + args.batch_size])
                ids, mask, labels, tokens = padded_batch(
                    chunk,
                    training_padded_length,
                    tokenizer.pad_token_id,
                    device,
                    loss_scope=args.loss_scope,
                    batch_size=args.batch_size,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = causal_loss(training_model, ids, mask, labels)
                (loss * tokens / group_tokens).backward()
                weighted_loss = float(loss.detach()) * tokens
                group_loss_sum += weighted_loss
                epoch_loss_sum += weighted_loss
                epoch_tokens += tokens
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            values = {
                "epoch": epoch,
                "optimizer_step": global_step,
                "train_loss": group_loss_sum / group_tokens,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "tokens": group_tokens,
                "tokens_seen_epoch": epoch_tokens,
                "grad_norm": grad_norm,
                "seconds_since_epoch_start": time.perf_counter() - epoch_start,
            }
            logger.log("step", values)
            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"train_loss={values['train_loss']:.6f} "
                    f"lr={values['learning_rate']:.3e}",
                    flush=True,
                )

        eval_loss, eval_bpc = golden_eval(
            model,
            eval_examples,
            args.eval_batch_size,
            tokenizer.pad_token_id,
            device,
        )
        seconds = time.perf_counter() - epoch_start
        epoch_values = {
            "epoch": epoch,
            "optimizer_step": global_step,
            "train_loss": epoch_loss_sum / epoch_tokens,
            "eval_loss": eval_loss,
            "eval_bits_per_character": eval_bpc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "tokens_seen": epoch_tokens,
            "seconds": seconds,
            "tokens_per_second": epoch_tokens / seconds,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        logger.log("epoch", epoch_values)
        print(
            f"epoch={epoch} train_loss={epoch_values['train_loss']:.6f} "
            f"eval_loss={eval_loss:.6f} eval_bpc={eval_bpc:.6f} "
            f"peak_gib={epoch_values['cuda_peak_reserved_gib']:.2f}",
            flush=True,
        )
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            saved = save_checkpoint(model, tokenizer, args.output_checkpoint)
            logger.log(
                "best_checkpoint",
                {"epoch": epoch, "eval_loss": eval_loss, "path": str(saved)},
            )
        if eval_loss < plateau_best_loss - args.min_eval_improvement:
            plateau_best_loss = eval_loss
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1
        logger.log(
            "plateau",
            {
                "epoch": epoch,
                "optimizer_step": global_step,
                "plateau_best_loss": plateau_best_loss,
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
                f"best_loss={best_loss:.6f} "
                f"epochs_without_material_improvement="
                f"{epochs_without_material_improvement}",
                flush=True,
            )
            logger.log(
                "early_stop",
                {
                    "epoch": epoch,
                    "optimizer_step": global_step,
                    "best_loss": best_loss,
                    "reason": "eval_loss_plateau",
                },
            )
            break

    if best_epoch == 0:
        logger.log(
            "result",
            {"improved": False, "baseline_loss": baseline.loss, "best_loss": best_loss},
        )
        logger.close()
        return

    del training_model, model, optimizer
    torch.cuda.empty_cache()
    best_model = load_model(args.output_checkpoint.resolve(), device, torch.bfloat16)
    final = evaluate(
        best_model,
        tokenizer,
        eval_examples,
        step=best_epoch * steps_per_epoch,
        batch_size=args.eval_batch_size,
        max_new_tokens=generation_max_new_tokens,
        device=device,
    )
    logger.log("final", asdict(final))
    logger.log(
        "result",
        {
            "improved": final.loss < baseline.loss,
            "best_epoch": best_epoch,
            "baseline_loss": baseline.loss,
            "final_loss": final.loss,
            "baseline_exact": baseline.num_correct,
            "final_exact": final.num_correct,
        },
    )
    print(
        f"final loss={final.loss:.6f} bpc={final.bits_per_character:.6f} "
        f"exact={final.num_correct}/{final.num_samples} "
        f"recall={final.pair_recall:.6f}",
        flush=True,
    )
    logger.close()


if __name__ == "__main__":
    main()

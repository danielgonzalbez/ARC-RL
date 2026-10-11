"""Compiled supervised pretraining for the ARC policy.

The script preserves the prompt/target representation used by
``arc_rl.rl_static_cache`` and optimizes causal cross entropy over either
full sequences or target tokens. It can continue from a matching checkpoint or
initialize the 120M preset from scratch, and evaluates the policy before and
after training with the same greedy rollout allowance.
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
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from .rl_static_cache import (
    ARCExample,
    ArcCharacterTokenizer,
    BucketedRolloutGenerator,
    DEFAULT_ARC_DATA,
    DEFAULT_CHECKPOINT,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    build_model_config as build_small_model_config,
    current_policy_logprobs,
    evaluate_policy,
    load_arc_examples,
    save_model,
)


MODEL_CONFIG_CHOICES = ("small", "120m")
DEFAULT_OUTPUT_CHECKPOINT = Path("checkpoints/pretrained_best.safetensors")
WANDB_SWEEP_ARGUMENTS = {
    "learning-rate": ("learning_rate", float),
    "weight-decay": ("weight_decay", float),
    "warmup-fraction": ("warmup_fraction", float),
    "max-grad-norm": ("max_grad_norm", float),
    "batch-size": ("batch_size", int),
    "minibatches": ("minibatches", int),
    "gradient-accumulation-steps": ("minibatches", int),
    "loss-scope": ("loss_scope", str),
}


def build_pretraining_model_config(name: str) -> Qwen3Config:
    """Build one of the supported pretraining model architectures."""

    if name == "small":
        return build_small_model_config()
    if name == "120m":
        # Scale the small model while retaining its architectural ratios:
        # a 4x gated MLP, one query/KV head per 64 residual channels, equal
        # query and KV head counts, and 128-dimensional attention heads.
        # Thirty layers puts the resulting model at 120,478,912 parameters.
        return Qwen3Config(
            vocab_size=VOCAB_SIZE,
            hidden_size=448,
            intermediate_size=1792,
            num_hidden_layers=30,
            num_attention_heads=7,
            num_key_value_heads=7,
            head_dim=128,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=7,
            linear_num_value_heads=7,
            max_position_embeddings=32768,
            bos_token_id=None,
            eos_token_id=EOS_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            use_cache=True,
        )
    raise ValueError(
        f"Unknown model config {name!r}; expected one of {MODEL_CONFIG_CHOICES}"
    )


def load_or_initialize_model(
    config_name: str,
    checkpoint_path: Path | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Qwen3ForCausalLM:
    """Build a selected architecture and optionally load matching weights."""

    # Repeated construction (baseline and trainable copies) must produce the
    # same initial weights when the 120m preset is trained from scratch.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(build_pretraining_model_config(config_name))

    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state_dict = load_file(str(checkpoint_path), device="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint does not match the {config_name!r} model config: "
                f"{checkpoint_path}"
            ) from exc

    model.to(device=device, dtype=dtype)
    model.generation_config.eos_token_id = EOS_TOKEN_ID
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    return model


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    eval_loss: float
    learning_rate: float
    tokens_seen: int
    seconds: float
    tokens_per_second: float
    grad_norm: float
    cuda_peak_allocated_gib: float
    cuda_peak_reserved_gib: float


class Logger:
    def __init__(self, path: Path, config: dict, run_name: str, project: str) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.wandb_run = None
        # This consumes .env through python-dotenv without displaying or changing it.
        load_dotenv()
        if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_SWEEP_ID"):
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=project, name=run_name, config=config
                )
            except Exception as exc:
                print(f"wandb disabled after initialization failure: {exc}", flush=True)

    def apply_sweep_overrides(self, args: argparse.Namespace) -> dict:
        """Apply values selected by a W&B sweep to argparse attributes."""

        if self.wandb_run is None:
            return {}
        wandb_config = dict(self.wandb_run.config)
        overrides = {}
        for sweep_name, (argument_name, converter) in WANDB_SWEEP_ARGUMENTS.items():
            if sweep_name not in wandb_config:
                continue
            value = converter(wandb_config[sweep_name])
            setattr(args, argument_name, value)
            overrides[argument_name] = value
        if overrides:
            self.wandb_run.config.update(overrides, allow_val_change=True)
        return overrides

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
    parser.add_argument(
        "--model-config",
        choices=MODEL_CONFIG_CHOICES,
        default="small",
        help="Model architecture. The 120m preset starts from scratch unless --checkpoint is supplied.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Matching initial weights (defaults to the supplied checkpoint for the small preset).",
    )
    parser.add_argument("--output-checkpoint", type=Path, default=DEFAULT_OUTPUT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ARC_DATA)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Examples per GPU microbatch.",
    )
    parser.add_argument(
        "--minibatches", "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Microbatches accumulated before each optimizer update.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-scope", choices=("full", "target"), default="full")
    parser.add_argument("--max-sequence-length", type=int, default=2816)
    parser.add_argument("--max-new-tokens", type=int, default=2816)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--eval-max-sequence-length", type=int, default=2816)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-color-swaps", type=int, default=2)
    parser.add_argument("--no-rotations", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="max-autotune")
    parser.add_argument("--run-name")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--wandb-project", default="ARC-Pretraining")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-greedy-eval", action="store_true", help="Use golden-target NLL only for baseline and final sweep evaluation.")
    parser.add_argument("--save-last-candidate", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_sequence_length <= 1 or args.max_new_tokens <= 0:
        raise ValueError("sequence and rollout limits must be positive")
    if (
        args.batch_size <= 0
        or args.minibatches <= 0
        or args.eval_batch_size <= 0
        or args.epochs <= 0
    ):
        raise ValueError("batch sizes, minibatches, and epochs must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not 0 <= args.warmup_fraction <= 1:
        raise ValueError("warmup fraction must be between 0 and 1")
    if args.max_grad_norm <= 0:
        raise ValueError("maximum gradient norm must be positive")
    if args.num_color_swaps < 0:
        raise ValueError("number of color swaps must be non-negative")


def padded_batch(examples: Sequence[ARCExample], max_length: int, device: torch.device, *, loss_scope: str = "target", batch_size: int | None = None):
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


def target_loss(model, input_ids, attention_mask, labels):
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def golden_eval_loss(model, examples, batch_size, max_length, device) -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_sum = 0
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                chunk = list(examples[start : start + batch_size])
                prompts = [example.prompt_ids for example in chunk]
                targets = [example.target_ids for example in chunk]
                logprobs = current_policy_logprobs(model, prompts, targets)
                for row, target in enumerate(targets):
                    loss_sum -= float(logprobs[row, : target.numel()].sum())
                    token_sum += target.numel()
    finally:
        model.train(was_training)
    return loss_sum / token_sum


def greedy_eval(model, tokenizer, examples, args, device, step):
    max_prompt = max(example.prompt_ids.numel() for example in examples)
    required = max_prompt + args.max_new_tokens
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
        max_new_tokens=args.max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint is None and args.model_config == "small":
        args.checkpoint = DEFAULT_CHECKPOINT
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

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "arc_pretrain_%Y%m%dT%H%M%S_%fZ"
    )
    if (
        os.environ.get("WANDB_SWEEP_ID")
        and args.output_checkpoint == DEFAULT_OUTPUT_CHECKPOINT
    ):
        args.output_checkpoint = Path("checkpoints/sweeps") / f"{run_name}.safetensors"
    log_file = args.log_file or Path("logs") / f"{run_name}.jsonl"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    architecture = build_pretraining_model_config(args.model_config)
    config.update({
        "run_name": run_name,
        "gpu": torch.cuda.get_device_name(device),
        "compute_dtype": "bfloat16",
        "training_weight_dtype": "float32",
        "fullgraph": True,
        "dynamic": False,
        "hidden_size": architecture.hidden_size,
        "intermediate_size": architecture.intermediate_size,
        "num_hidden_layers": architecture.num_hidden_layers,
        "num_attention_heads": architecture.num_attention_heads,
        "num_key_value_heads": architecture.num_key_value_heads,
        "head_dim": architecture.head_dim,
        "effective_batch_size": args.batch_size * args.minibatches,
    })
    logger = Logger(log_file, config, run_name, args.wandb_project)
    sweep_overrides = logger.apply_sweep_overrides(args)
    validate_args(args)
    if sweep_overrides:
        config.update(sweep_overrides)
        config["effective_batch_size"] = args.batch_size * args.minibatches
        print(f"wandb_sweep_overrides={sweep_overrides}", flush=True)
    logger.log("run_start", config)
    tokenizer = ArcCharacterTokenizer()
    train_examples, train_summary = load_pretraining_examples(args, tokenizer)
    eval_examples, eval_summary = load_arc_examples(
        args.data_dir, split="evaluation", tokenizer=tokenizer,
        usable_example_limit=args.eval_examples,
        max_total_tokens=args.eval_max_sequence_length,
        augment=False, seed=args.seed,
    )
    logger.log("dataset_train", train_summary)
    logger.log("dataset_eval", asdict(eval_summary))
    print(f"train_examples={len(train_examples)} eval_examples={len(eval_examples)}", flush=True)

    base_model = load_or_initialize_model(
        args.model_config, args.checkpoint, device=device,
        dtype=torch.bfloat16, seed=args.seed,
    )
    parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
    effective_batch_size = args.batch_size * args.minibatches
    logger.log("model", {
        "config": args.model_config,
        "parameters": parameter_count,
        "microbatch_size": args.batch_size,
        "minibatches": args.minibatches,
        "effective_batch_size": effective_batch_size,
    })
    print(f"model_config={args.model_config} parameters={parameter_count:,}", flush=True)
    print(
        f"microbatch_size={args.batch_size} minibatches={args.minibatches} "
        f"effective_batch_size={effective_batch_size}",
        flush=True,
    )
    if args.skip_greedy_eval:
        from types import SimpleNamespace
        baseline = SimpleNamespace(loss=golden_eval_loss(base_model, eval_examples, args.eval_batch_size, args.eval_max_sequence_length, device))
        logger.log("baseline", {"loss": baseline.loss, "greedy_evaluated": False})
    else:
        baseline = greedy_eval(base_model, tokenizer, eval_examples, args, device, 0)
        logger.log("baseline", asdict(baseline))
    print(f"baseline loss={baseline.loss:.6f}", flush=True)
    if args.eval_only:
        logger.close()
        return

    del base_model
    torch.cuda.empty_cache()
    # FP32 trainable weights and optimizer state preserve small continuation
    # updates; CUDA matrix compute remains BF16 under autocast below.
    model = load_or_initialize_model(
        args.model_config, args.checkpoint, device=device,
        dtype=torch.float32, seed=args.seed,
    )
    model.config.use_cache = False
    model.train()
    # Qwen3/Transformers 5.10 creates an inference tensor that PyTorch 2.8
    # cannot mutate while recording a training CUDA graph. Keep Inductor full
    # graph compilation and autotuning, but disable only the replay layer.
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
    best_loss = baseline.loss
    best_epoch = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
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
            group_examples = train_examples[group_start * args.batch_size : (group_start + group_size) * args.batch_size]
            group_tokens = sum(
                example.prompt_ids.numel() + example.target_ids.numel() - 1
                if args.loss_scope == "full" else example.target_ids.numel()
                for example in group_examples
            )
            for group_offset in range(group_size):
                microbatch_index = group_start + group_offset
                start = microbatch_index * args.batch_size
                chunk = list(train_examples[start : start + args.batch_size])
                input_ids, mask, labels, target_tokens = padded_batch(
                    chunk, args.max_sequence_length, device,
                    loss_scope=args.loss_scope, batch_size=args.batch_size,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = target_loss(compiled_model, input_ids, mask, labels)
                (loss * target_tokens / group_tokens).backward()
                loss_sum += float(loss.detach()) * target_tokens
                tokens_seen += target_tokens
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            global_step += 1
        seconds = time.perf_counter() - epoch_start
        eval_loss = golden_eval_loss(
            model, eval_examples, args.eval_batch_size,
            args.max_sequence_length, device,
        )
        metrics = EpochMetrics(
            epoch=epoch, train_loss=loss_sum / tokens_seen, eval_loss=eval_loss,
            learning_rate=optimizer.param_groups[0]["lr"], tokens_seen=tokens_seen,
            seconds=seconds, tokens_per_second=tokens_seen / seconds,
            grad_norm=last_grad_norm,
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        )
        logger.log("epoch", asdict(metrics))
        print(f"epoch={epoch} train_loss={metrics.train_loss:.6f} eval_loss={eval_loss:.6f} tok_s={metrics.tokens_per_second:.1f} peak_gib={metrics.cuda_peak_reserved_gib:.2f}", flush=True)
        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            save_model(model, args.output_checkpoint)
            logger.log("best_checkpoint", {"epoch": epoch, "eval_loss": eval_loss, "path": str(args.output_checkpoint.resolve())})

    if best_epoch == 0 and args.save_last_candidate:
        best_epoch = args.epochs
        save_model(model, args.output_checkpoint)
        logger.log("candidate_checkpoint", {"epoch": best_epoch, "path": str(args.output_checkpoint.resolve()), "reason": "matched_bfloat16_reload_evaluation"})
    if best_epoch == 0:
        print("No epoch improved baseline golden NLL; no replacement checkpoint promoted", flush=True)
        logger.log("result", {"improved": False, "baseline_loss": baseline.loss, "best_loss": best_loss})
        logger.close()
        return
    del compiled_model, model, optimizer
    torch.cuda.empty_cache()
    best_model = load_or_initialize_model(
        args.model_config, args.output_checkpoint, device=device,
        dtype=torch.bfloat16, seed=args.seed,
    )
    if args.skip_greedy_eval:
        final_loss = golden_eval_loss(best_model, eval_examples, args.eval_batch_size, args.eval_max_sequence_length, device)
        logger.log("result", {"improved": final_loss < baseline.loss, "best_epoch": best_epoch, "baseline_loss": baseline.loss, "best_loss": best_loss, "final_loss": final_loss, "greedy_evaluated": False})
        logger.close()
        return
    final = greedy_eval(best_model, tokenizer, eval_examples, args, device, best_epoch)
    logger.log("final", asdict(final))
    promoted = final.loss < baseline.loss and final.num_correct >= baseline.num_correct
    logger.log("result", {"improved": promoted, "best_epoch": best_epoch, "baseline_loss": baseline.loss, "final_loss": final.loss, "baseline_exact": baseline.num_correct, "final_exact": final.num_correct})
    print(f"final loss={final.loss:.6f} exact={final.num_correct}/{final.num_samples} reward={final.mean_reward:.6f} recall={final.pair_recall:.6f}", flush=True)
    logger.close()


if __name__ == "__main__":
    main()

"""Static-cache RL training for the small custom Qwen3 ARC policy.

This module trains the custom policy with either the original Hugging Face
static-cache generator or a persistent local vLLM rollout engine:

* ``build_model`` defines the 2.63M-parameter Qwen3 variant.
* ``load_model`` loads the safetensors checkpoint into that exact model.
* ``StaticCachePool`` lazily owns one cache per ``(batch_size, total_length)``
  bucket and resets cache contents between independent rollout batches.
* ``BucketedRolloutGenerator`` groups prompts by total-length capacity, pads
  the final microbatch to a stable batch size, and returns sampled tokens plus
  the log-probability assigned to every sampled token.
* ``compute_clipped_policy_loss`` implements the clipped policy objective from
  ``rl_small.py`` with token-format and sequence-level advantages.
* ``train_rl`` performs rollout -> reward -> differentiable forward -> update
  iterations and synchronizes vLLM after each optimizer update when selected.
* ``--benchmark-vllm`` compares that generator with vLLM on real short and
  long ARC prompts loaded from the checkpoint's training distribution.

The cache pool may safely survive optimizer updates because the buffers are
reset before every independent rollout.  Their K/V contents are recomputed
using the policy's current weights; only their allocations are reused.
"""

from __future__ import annotations

if __name__ == "__main__":
    print("startup phase=import_dependencies", flush=True)

import argparse
import gc
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import CompileConfig, Qwen3Config, Qwen3ForCausalLM, StaticCache

if __name__ == "__main__":
    print("startup phase=imports_ready", flush=True)


# The vocabulary used to train checkpoints_run_vs5edt1s_model.safetensors.
# IDs 22 and 23 were unused during training; ID 22 is used as padding here.
ARC_CHARACTERS = (
    "I",
    "O",
    "T",
    "S",
    "F",
    "[",
    "]",
    "\n",
    " ",
    "-",
    ":",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "E",
    "<pad>",
    "<unused>",
)

EOS_TOKEN_ID = 21
PAD_TOKEN_ID = 22
DUMMY_TOKEN_ID = 0
VOCAB_SIZE = len(ARC_CHARACTERS)

DEFAULT_BUCKETS = (1024, 3072, 4096)
DEFAULT_ROLLOUT_BATCH_SIZE = 32
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_CHECKPOINT = Path("checkpoints/checkpoints_run_vs5edt1s_model.safetensors")
DEFAULT_ARC_DATA = Path("minimal-arc-main")


class ExperimentLogger:
    """Line-buffered JSONL logger with optional W&B scalar mirroring."""

    def __init__(self, path: Path, *, config: dict, run_name: str, wandb_project: str):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.wandb_run = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project, name=run_name, config=config
                )
            except Exception as exc:
                print(f"wandb disabled after initialization failure: {exc}", flush=True)
        self.log("run_start", config)

    def log(self, event: str, values: dict, *, step: int | None = None) -> None:
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        if step is not None:
            record["step"] = step
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.wandb_run is not None:
            payload = values if event in ("train", "eval") else {event: values}
            self.wandb_run.log(payload, step=step)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
        self.handle.close()


class FP32MasterAdamW:
    """AdamW over FP32 masters for a BF16 working parameter tree."""

    def __init__(self, params, *, lr: float, weight_decay: float):
        self.working_params = [parameter for parameter in params if parameter.requires_grad]
        self.master_params = [
            torch.nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
            for parameter in self.working_params
        ]
        self.optimizer = torch.optim.AdamW(
            self.master_params,
            lr=lr,
            weight_decay=weight_decay,
            fused=torch.cuda.is_available(),
        )
        self.param_groups = self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.working_params:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        for working, master in zip(self.working_params, self.master_params):
            master.grad = (
                None if working.grad is None else working.grad.detach().float()
            )
        self.optimizer.step()
        for working, master in zip(self.working_params, self.master_params):
            working.copy_(master.to(dtype=working.dtype))


class ArcCharacterTokenizer:
    """Minimal tokenizer matching the checkpoint's 24-symbol vocabulary."""

    eos_token_id = EOS_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    eos_token = "E"
    pad_token = "<pad>"

    _tag_replacements = (
        ("<test_output>", "F"),
        ("<test_input>", "T"),
        ("<output>", "O"),
        ("<input>", "I"),
        ("<sep>", "S"),
        ("<eos>", "E"),
    )

    def __init__(self) -> None:
        self.char_to_id = {char: idx for idx, char in enumerate(ARC_CHARACTERS)}

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        for tag, replacement in self._tag_replacements:
            text = text.replace(tag, replacement)
        if add_eos and not text.endswith(self.eos_token):
            text += self.eos_token

        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"Character {exc.args[0]!r} is not in the ARC vocabulary") from exc

    def decode(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        pieces: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if not 0 <= token_id < VOCAB_SIZE:
                raise ValueError(f"Token id {token_id} is outside [0, {VOCAB_SIZE})")
            if skip_special_tokens and token_id in (EOS_TOKEN_ID, PAD_TOKEN_ID, 23):
                continue
            pieces.append(ARC_CHARACTERS[token_id])
        return "".join(pieces)

    def batch_decode(
        self,
        sequences: Iterable[Sequence[int] | torch.Tensor],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [
            self.decode(sequence, skip_special_tokens=skip_special_tokens)
            for sequence in sequences
        ]


def build_model_config() -> Qwen3Config:
    """Return the exact architecture used by the supplied small checkpoint."""

    return Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        max_position_embeddings=32768,
        bos_token_id=None,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        use_cache=True,
    )


def build_model() -> Qwen3ForCausalLM:
    """Construct an uninitialized small Qwen3 policy."""

    return Qwen3ForCausalLM(build_model_config())


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3ForCausalLM:
    """Construct the policy, strictly load its checkpoint, and move it once."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model()
    state_dict = load_file(str(checkpoint_path), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the configured Qwen3 model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    model.to(device=device, dtype=dtype)
    model.generation_config.eos_token_id = EOS_TOKEN_ID
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    return model


def parameter_summary(model: torch.nn.Module) -> str:
    count = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return f"parameters={count:,}, trainable={trainable:,}"


class StaticCachePool:
    """Lazily allocate and reset static caches for stable shape buckets.

    A cache is keyed by both batch size and total sequence capacity.  In normal
    use ``BucketedRolloutGenerator`` pads its last microbatch, so only one batch
    size is used and each configured length produces exactly one allocation.
    """

    def __init__(self, model: Qwen3ForCausalLM, buckets: Sequence[int]) -> None:
        normalized = tuple(sorted({int(bucket) for bucket in buckets}))
        if not normalized or normalized[0] <= 0:
            raise ValueError("At least one positive cache bucket is required")
        self.model = model
        self.buckets = normalized
        self._caches: dict[tuple[int, int], StaticCache] = {}

    def select_bucket(self, required_total_length: int) -> int:
        for bucket in self.buckets:
            if required_total_length <= bucket:
                return bucket
        raise ValueError(
            f"Required total length {required_total_length} exceeds the largest "
            f"configured cache bucket ({self.buckets[-1]})"
        )

    def get(self, *, batch_size: int, max_cache_len: int) -> StaticCache:
        if max_cache_len not in self.buckets:
            raise ValueError(f"{max_cache_len} is not a configured cache bucket")
        key = (int(batch_size), int(max_cache_len))
        cache = self._caches.get(key)
        if cache is None:
            # Transformers 5.x initializes the device, dtype and batch-shaped
            # backing tensors lazily during the first prefill for this cache.
            cache = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
            self._caches[key] = cache
        else:
            cache.reset()
        return cache

    def reset_all(self) -> None:
        for cache in self._caches.values():
            cache.reset()

    @property
    def allocated_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._caches))


@dataclass(frozen=True)
class RolloutBatch:
    """Rollout results restored to the same order as the input prompts."""

    token_ids: list[torch.Tensor]
    token_logprobs: list[torch.Tensor]
    bucket_by_prompt: list[int]


class BucketedRolloutGenerator:
    """Generate rollouts with lazily allocated static-cache length buckets."""

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        buckets: Sequence[int] = DEFAULT_BUCKETS,
        rollout_batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE,
        compile_decode: bool = True,
        compile_mode: str = "reduce-overhead",
    ) -> None:
        if rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        self.model = model
        self.rollout_batch_size = int(rollout_batch_size)
        self.cache_pool = StaticCachePool(model, buckets)
        self.compile_decode = bool(compile_decode)
        self.compile_config = CompileConfig(mode=compile_mode) if compile_decode else None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(
        self,
        prompts: Sequence[Sequence[int] | torch.Tensor],
        *,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> RolloutBatch:
        """Generate and return per-token sampling log-probabilities.

        Bucketing is based on ``len(prompt) + max_new_tokens``.  Actual future
        EOS positions are unknowable, so the caller should choose a realistic
        generation budget rather than route using observed completion length.
        """

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not prompts:
            return RolloutBatch([], [], [])

        normalized_prompts = [self._normalize_prompt(prompt) for prompt in prompts]
        grouped: dict[int, list[tuple[int, torch.Tensor]]] = {}
        bucket_by_prompt = [0] * len(normalized_prompts)
        for original_index, prompt in enumerate(normalized_prompts):
            bucket = self.cache_pool.select_bucket(prompt.numel() + max_new_tokens)
            bucket_by_prompt[original_index] = bucket
            grouped.setdefault(bucket, []).append((original_index, prompt))

        ordered_tokens: list[torch.Tensor | None] = [None] * len(prompts)
        ordered_logprobs: list[torch.Tensor | None] = [None] * len(prompts)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for bucket, indexed_prompts in grouped.items():
                    for start in range(0, len(indexed_prompts), self.rollout_batch_size):
                        microbatch = indexed_prompts[start : start + self.rollout_batch_size]
                        results = self._generate_microbatch(
                            microbatch,
                            bucket=bucket,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        for original_index, token_ids, token_logprobs in results:
                            ordered_tokens[original_index] = token_ids
                            ordered_logprobs[original_index] = token_logprobs
        finally:
            self.model.train(was_training)

        if any(item is None for item in ordered_tokens + ordered_logprobs):
            raise RuntimeError("Internal error: one or more rollout results were not restored")

        return RolloutBatch(
            # clone outside inference_mode so these are ordinary tensors that
            # can safely be used as gather indices in the training forward.
            token_ids=[item.clone() for item in ordered_tokens if item is not None],
            token_logprobs=[item.clone() for item in ordered_logprobs if item is not None],
            bucket_by_prompt=bucket_by_prompt,
        )

    def _normalize_prompt(self, prompt: Sequence[int] | torch.Tensor) -> torch.Tensor:
        prompt_tensor = torch.as_tensor(prompt, dtype=torch.long).flatten().cpu()
        if prompt_tensor.numel() == 0:
            raise ValueError("Prompts must contain at least one token")
        if prompt_tensor.min().item() < 0 or prompt_tensor.max().item() >= self.model.config.vocab_size:
            raise ValueError("A prompt contains token ids outside the model vocabulary")
        return prompt_tensor

    def _generate_microbatch(
        self,
        indexed_prompts: Sequence[tuple[int, torch.Tensor]],
        *,
        bucket: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        real_batch_size = len(indexed_prompts)
        fixed_batch_size = self.rollout_batch_size
        prompt_width = max(prompt.numel() for _, prompt in indexed_prompts)
        if prompt_width + max_new_tokens > bucket:
            raise RuntimeError("Selected cache bucket is too small after batch padding")

        input_ids = torch.full(
            (fixed_batch_size, prompt_width),
            PAD_TOKEN_ID,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for row, (_, prompt) in enumerate(indexed_prompts):
            prompt = prompt.to(self.device)
            input_ids[row, -prompt.numel() :] = prompt
            attention_mask[row, -prompt.numel() :] = 1

        # Fill unused rows with a valid one-token dummy prompt.  The outputs are
        # discarded, but keeping the batch dimension fixed avoids extra cache
        # allocations and compilation variants for final partial microbatches.
        if real_batch_size < fixed_batch_size:
            input_ids[real_batch_size:, -1] = DUMMY_TOKEN_ID
            attention_mask[real_batch_size:, -1] = 1

        cache = self.cache_pool.get(batch_size=fixed_batch_size, max_cache_len=bucket)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": PAD_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
            "return_dict_in_generate": True,
            "output_scores": True,
            "disable_compile": not self.compile_decode,
        }
        if self.compile_config is not None:
            generation_kwargs["compile_config"] = self.compile_config
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        outputs = self.model.generate(**generation_kwargs)
        generated_ids = outputs.sequences[:, prompt_width:]
        if outputs.scores:
            processed_scores = torch.stack(outputs.scores, dim=1).float()
            all_logprobs = processed_scores.log_softmax(dim=-1).gather(
                -1, generated_ids.unsqueeze(-1)
            ).squeeze(-1)
        else:
            all_logprobs = torch.empty(
                (fixed_batch_size, 0), dtype=torch.float32, device=self.device
            )

        results: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for row, (original_index, _) in enumerate(indexed_prompts):
            token_ids, token_logprobs = self._trim_after_eos(
                generated_ids[row], all_logprobs[row]
            )
            results.append(
                (original_index, token_ids.detach().cpu(), token_logprobs.detach().cpu())
            )
        return results

    @staticmethod
    def _trim_after_eos(
        token_ids: torch.Tensor, token_logprobs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eos_positions = (token_ids == EOS_TOKEN_ID).nonzero(as_tuple=False)
        length = int(eos_positions[0].item()) + 1 if eos_positions.numel() else token_ids.numel()
        return token_ids[:length], token_logprobs[:length]

class VLLMRolloutGenerator:
    """Generate ARC rollouts from raw token IDs with a local vLLM engine."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        dtype: str,
        max_model_len: int,
        max_num_seqs: int,
        gpu_memory_utilization: float,
        seed: int,
        enforce_eager: bool = False,
    ) -> None:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is required for the selected rollout operation; install requirements "
                "in the Python environment used to run this script"
            ) from exc

        self.model_dir = Path(model_dir).resolve()
        self.max_model_len = int(max_model_len)
        self.llm = LLM(
            model=str(self.model_dir),
            tokenizer=None,
            skip_tokenizer_init=True,
            dtype=dtype,
            max_model_len=self.max_model_len,
            max_num_seqs=int(max_num_seqs),
            gpu_memory_utilization=float(gpu_memory_utilization),
            seed=int(seed),
            enforce_eager=bool(enforce_eager),
            trust_remote_code=False,
        )

    def sync_from_model(self, model: Qwen3ForCausalLM) -> float:
        """Atomically publish trained weights and reload the persistent worker."""
        started = time.perf_counter()
        destination = self.model_dir / "model.safetensors"
        temporary = self.model_dir / "model.safetensors.pending"
        state_dict = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
        save_file(state_dict, str(temporary))
        os.replace(temporary, destination)
        self.llm.collective_rpc("reload_weights")
        return time.perf_counter() - started

    def generate(
        self,
        prompts: Sequence[Sequence[int] | torch.Tensor],
        *,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> RolloutBatch:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not prompts:
            return RolloutBatch([], [], [])

        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM became unavailable after engine creation") from exc

        prompt_token_ids: list[list[int]] = []
        for prompt in prompts:
            normalized = torch.as_tensor(prompt, dtype=torch.long).flatten().tolist()
            if not normalized:
                raise ValueError("Prompts must contain at least one token")
            if min(normalized) < 0 or max(normalized) >= VOCAB_SIZE:
                raise ValueError("A prompt contains token ids outside the model vocabulary")
            if len(normalized) + max_new_tokens > self.max_model_len:
                raise ValueError("A prompt plus its rollout exceeds vLLM max_model_len")
            prompt_token_ids.append(normalized)

        sampling_kwargs = {
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "logprobs": 1,
            "stop_token_ids": [EOS_TOKEN_ID],
            "detokenize": False,
        }
        if seed is not None:
            sampling_kwargs["seed"] = int(seed)
        params = SamplingParams(**sampling_kwargs)
        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompt_token_ids],
            sampling_params=params,
            use_tqdm=False,
        )

        token_ids: list[torch.Tensor] = []
        token_logprobs: list[torch.Tensor] = []
        for request in outputs:
            sample = request.outputs[0]
            ids = torch.tensor(sample.token_ids, dtype=torch.long)
            values: list[float] = []
            for token_id, candidates in zip(sample.token_ids, sample.logprobs or ()):
                chosen = candidates.get(token_id)
                if chosen is None:
                    raise RuntimeError(
                        "vLLM did not return the sampled token's log-probability"
                    )
                values.append(float(chosen.logprob))
            if len(values) != len(sample.token_ids):
                raise RuntimeError(
                    "vLLM returned incomplete sampled-token log-probabilities"
                )
            token_ids.append(ids)
            token_logprobs.append(torch.tensor(values, dtype=torch.float32))
        return RolloutBatch(
            token_ids, token_logprobs, [self.max_model_len] * len(outputs)
        )



def pad_token_logprobs(
    token_logprobs: Sequence[torch.Tensor], *, multiple_of: int = 16
) -> torch.Tensor:
    """Pad rollout log-probabilities to the tensor shape expected by RL loss code."""

    if not token_logprobs:
        return torch.empty((0, 0), dtype=torch.float32)
    max_length = max(item.numel() for item in token_logprobs)
    padded_length = ((max_length + multiple_of - 1) // multiple_of) * multiple_of
    result = torch.zeros((len(token_logprobs), padded_length), dtype=torch.float32)
    for row, item in enumerate(token_logprobs):
        result[row, : item.numel()] = item
    return result


@dataclass(frozen=True)
class ARCExample:
    """One ARC prompt and its coordinate/value completion."""

    task_id: str
    prompt_ids: torch.Tensor
    target_ids: torch.Tensor


def matrix_to_sequence(matrix: Sequence[Sequence[int]]) -> str:
    """Use the coordinate format on which the checkpoint was trained."""

    pairs = [
        f"{row}-{column}:{value}"
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
    ]
    return " ".join(pairs) + "E"


def format_arc_prompt(challenge: dict) -> str:
    """Reproduce the compact I/O/S/T/F prompt format from the training notebook."""

    pieces: list[str] = []
    for example in challenge["train"]:
        pieces.extend(
            (
                "I",
                str(np.asarray(example["input"])),
                "O",
                str(np.asarray(example["output"])),
                "S",
            )
        )
    pieces.extend(("T", str(np.asarray(challenge["test"][0]["input"])), "F"))
    return "".join(pieces)


@dataclass(frozen=True)
class ARCLoadSummary:
    """Counts and token-length bounds produced while loading one ARC split."""

    split: str
    base_tasks: int
    candidate_examples: int
    loaded_examples: int
    skipped_prompt_too_long: int
    skipped_target_too_long: int
    skipped_total_too_long: int
    max_prompt_length: int
    max_total_length: int
    max_target_length: int


def rotate_grid(grid: Sequence[Sequence[int]], angle: int) -> list[list[int]]:
    """Rotate an ARC grid counter-clockwise by 90, 180, or 270 degrees."""

    if angle not in (90, 180, 270):
        raise ValueError("Rotation angle must be 90, 180, or 270 degrees")
    return np.rot90(np.asarray(grid), k=angle // 90).tolist()


def swap_grid_colors(
    grid: Sequence[Sequence[int]], permutation: Sequence[int]
) -> list[list[int]]:
    """Apply one permutation of ARC colors 0..9 to a grid."""

    if len(permutation) != 10:
        raise ValueError("A color permutation must contain exactly 10 entries")
    return [[int(permutation[value]) for value in row] for row in grid]


def _transform_arc_challenge(challenge: dict, transform) -> dict:
    """Apply the same geometric/color transform to every grid in a task."""

    transformed_train = [
        {
            "input": transform(example["input"]),
            "output": transform(example["output"]),
        }
        for example in challenge["train"]
    ]
    transformed_test = [
        {"input": transform(example["input"])}
        for example in challenge["test"]
    ]
    return {"train": transformed_train, "test": transformed_test}


def load_arc_examples(
    data_dir: str | Path = DEFAULT_ARC_DATA,
    *,
    split: str,
    tokenizer: ArcCharacterTokenizer | None = None,
    base_task_limit: int | None = None,
    usable_example_limit: int | None = None,
    max_prompt_tokens: int | None = None,
    max_target_tokens: int | None = None,
    max_total_tokens: int | None = 2048,
    augment: bool = False,
    add_rotations: bool = True,
    num_color_swaps: int = 1,
    seed: int = 3407,
) -> tuple[list[ARCExample], ARCLoadSummary]:
    """Load one ARC split, optionally augmenting training tasks only.

    Augmentation follows the relevant part of rl.py: three rotations plus
    independently permuted colors. The same transform is applied to all
    demonstration grids, the test input, and its golden output. Evaluation
    callers pass augment=False and therefore remain untouched.
    """

    if split not in ("training", "evaluation"):
        raise ValueError("split must be 'training' or 'evaluation'")
    if augment and split != "training":
        raise ValueError("Data augmentation is restricted to the training split")
    if num_color_swaps not in (0, 1):
        raise ValueError("num_color_swaps must be 0 or 1")

    data_dir = Path(data_dir).expanduser().resolve()
    challenges_path = data_dir / f"arc-agi_{split}_challenges.json"
    solutions_path = data_dir / f"arc-agi_{split}_solutions.json"
    if not challenges_path.is_file() or not solutions_path.is_file():
        raise FileNotFoundError(
            f"Expected {challenges_path.name} and {solutions_path.name} in {data_dir}"
        )

    with challenges_path.open() as handle:
        challenges = json.load(handle)
    with solutions_path.open() as handle:
        solutions = json.load(handle)

    tokenizer = tokenizer or ArcCharacterTokenizer()
    color_rng = np.random.default_rng(seed)
    color_permutations = [
        color_rng.permutation(10).tolist() for _ in range(num_color_swaps)
    ]

    examples: list[ARCExample] = []
    base_tasks = 0
    candidate_examples = 0
    skipped_prompt = 0
    skipped_target = 0
    skipped_total = 0

    for task_id, challenge in challenges.items():
        if len(challenge.get("test", ())) != 1 or len(solutions.get(task_id, ())) != 1:
            continue
        if base_task_limit is not None and base_tasks >= base_task_limit:
            break
        base_tasks += 1

        variants = [(task_id, challenge, solutions[task_id][0])]
        if augment and add_rotations:
            for angle in (90, 180, 270):
                transform = lambda grid, angle=angle: rotate_grid(grid, angle)
                variants.append(
                    (
                        f"{task_id}_rotation_{angle}",
                        _transform_arc_challenge(challenge, transform),
                        transform(solutions[task_id][0]),
                    )
                )
        if augment:
            for color_index, permutation in enumerate(color_permutations):
                transform = (
                    lambda grid, permutation=permutation:
                    swap_grid_colors(grid, permutation)
                )
                variants.append(
                    (
                        f"{task_id}_colors_{color_index}",
                        _transform_arc_challenge(challenge, transform),
                        transform(solutions[task_id][0]),
                    )
                )

        for variant_id, variant_challenge, variant_solution in variants:
            candidate_examples += 1
            example = ARCExample(
                task_id=variant_id,
                prompt_ids=torch.tensor(
                    tokenizer.encode(format_arc_prompt(variant_challenge)),
                    dtype=torch.long,
                ),
                target_ids=torch.tensor(
                    tokenizer.encode(matrix_to_sequence(variant_solution)),
                    dtype=torch.long,
                ),
            )
            total_tokens = example.prompt_ids.numel() + example.target_ids.numel()
            if max_total_tokens is not None and total_tokens > max_total_tokens:
                skipped_total += 1
                continue
            if (
                max_prompt_tokens is not None
                and example.prompt_ids.numel() > max_prompt_tokens
            ):
                skipped_prompt += 1
                continue
            if (
                max_target_tokens is not None
                and example.target_ids.numel() > max_target_tokens
            ):
                skipped_target += 1
                continue
            examples.append(example)
            if (
                usable_example_limit is not None
                and len(examples) >= usable_example_limit
            ):
                break
        if usable_example_limit is not None and len(examples) >= usable_example_limit:
            break

    if not examples:
        raise ValueError(
            f"No usable single-test ARC examples were found in {data_dir} for {split}"
        )

    summary = ARCLoadSummary(
        split=split,
        base_tasks=base_tasks,
        candidate_examples=candidate_examples,
        loaded_examples=len(examples),
        skipped_prompt_too_long=skipped_prompt,
        skipped_target_too_long=skipped_target,
        skipped_total_too_long=skipped_total,
        max_prompt_length=max(example.prompt_ids.numel() for example in examples),
        max_target_length=max(example.target_ids.numel() for example in examples),
        max_total_length=max(example.prompt_ids.numel() + example.target_ids.numel() for example in examples),
    )
    return examples, summary


def load_arc_training_examples(
    data_dir: str | Path = DEFAULT_ARC_DATA,
    *,
    tokenizer: ArcCharacterTokenizer | None = None,
    limit: int | None = None,
    max_prompt_tokens: int | None = None,
    max_target_tokens: int | None = None,
    max_total_tokens: int | None = 2048,
) -> list[ARCExample]:
    """Compatibility wrapper for loading unaugmented ARC training examples."""

    examples, _ = load_arc_examples(
        data_dir,
        split="training",
        tokenizer=tokenizer,
        base_task_limit=limit,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        max_total_tokens=max_total_tokens,
        augment=False,
    )
    return examples


def _target_pairs(text: str, eos_token: str) -> set[str]:
    return {piece.replace(eos_token, "") for piece in text.split() if piece}


def completion_is_correct(
    target_ids: Sequence[int] | torch.Tensor,
    predicted_ids: Sequence[int] | torch.Tensor,
    tokenizer: ArcCharacterTokenizer,
) -> bool:
    """Return whether predicted and target coordinate/value pair sets are equal."""

    target = _target_pairs(tokenizer.decode(target_ids), tokenizer.eos_token)
    predicted = _target_pairs(tokenizer.decode(predicted_ids), tokenizer.eos_token)
    return target == predicted


def _grid_corner(target_text: str, eos_token: str) -> tuple[int, int]:
    pairs = _target_pairs(target_text, eos_token)
    rows = [int(pair.split("-")[0]) for pair in pairs]
    columns = [int(pair.split("-")[1].split(":")[0]) for pair in pairs]
    return max(rows), max(columns)


def _already_predicted(prediction: str, new_character: str) -> bool:
    pieces = prediction.split()
    if len(pieces) == 1:
        return False
    return pieces[-1] + new_character in " ".join(pieces[:-1])


def _has_all_target_pairs(prediction: str, target: str, eos_token: str) -> bool:
    return _target_pairs(target, eos_token).issubset(_target_pairs(prediction, eos_token))


def arc_token_rewards(
    target_ids: Sequence[int] | torch.Tensor,
    predicted_ids: Sequence[int] | torch.Tensor,
    tokenizer: ArcCharacterTokenizer,
    *,
    pair_correctness_mode: bool = False,
) -> torch.Tensor:
    """Dense ARC rewards ported from ``setup/utils.py:get_rewards``.

    Valid grid syntax and target coordinate/value pairs receive +1; malformed
    syntax and impossible/wrong predictions receive -1. In pair-correctness
    mode, pair quality is handled separately on numeric tokens, so EOS rewards
    syntactically complete termination and a separator after an incorrect pair
    is penalized. Tokens after EOS get zero, although normal generation is
    trimmed at its first EOS.
    """

    target_text = tokenizer.decode(target_ids)
    predicted = torch.as_tensor(predicted_ids, dtype=torch.long).flatten().cpu()
    true_values = {
        int(pair.split(":")[-1].replace(tokenizer.eos_token, ""))
        for pair in target_text.split()
    }
    max_row, max_column = _grid_corner(target_text, tokenizer.eos_token)

    rewards: list[float] = []
    in_row = True
    in_prediction = False
    wrong = False
    invalid_position = False
    ended = False

    for index, token in enumerate(predicted.tolist()):
        prefix = tokenizer.decode(predicted[:index])
        character = tokenizer.decode([token])

        if token == tokenizer.eos_token_id:
            ended = True
            if pair_correctness_mode:
                syntactically_complete = (
                    not wrong
                    and bool(prefix)
                    and (
                        (in_prediction and prefix[-1].isdigit())
                        or (in_row and prefix[-1] == " ")
                    )
                )
                rewards.append(1.0 if syntactically_complete else -1.0)
            else:
                rewards.append(
                    1.0
                    if _has_all_target_pairs(prefix, target_text, tokenizer.eos_token)
                    else -1.0
                )
            continue
        if character == " " and wrong:
            wrong = False
            in_row = True
            in_prediction = False
            rewards.append(1.0)
            continue
        if ended:
            rewards.append(0.0)
            continue
        if character not in "0123456789-: ":
            rewards.append(-1.0)
            wrong = True
            continue
        if index == 0 and not character.isdigit():
            rewards.append(-1.0)
            wrong = True
            continue
        if character == "0" and index > 0 and prefix[-1] == "0":
            rewards.append(-1.0)
            wrong = True
            continue
        if not in_prediction and character == " ":
            rewards.append(-1.0)
            continue
        if character == ":" and (index == 0 or in_row or in_prediction):
            rewards.append(-1.0)
            wrong = True
            continue

        if character.isdigit():
            previous_digits = ""
            for previous in reversed(prefix):
                if not previous.isdigit():
                    break
                previous_digits = previous + previous_digits
            recovered_number = previous_digits + character
            number = int(recovered_number)
            if number > max_row and number > max_column and number not in true_values:
                rewards.append(-1.0)
                if (in_row and number > max_row) or (
                    not in_row and not in_prediction and number > max_column
                ):
                    wrong = True
                continue
            if number == 0 and len(recovered_number) > 1:
                rewards.append(-1.0)
                continue

        if wrong:
            in_row = False
            in_prediction = False
            rewards.append(0.0)
            continue

        if character.isdigit():
            previous_digits = ""
            for previous in reversed(prefix):
                if not previous.isdigit():
                    break
                previous_digits = previous + previous_digits
            recovered_number = previous_digits + character
            number = int(recovered_number)

            if not in_prediction:
                upper_bound = max_row if in_row else max_column
                if number <= upper_bound:
                    row_complete = in_row and all(
                        f"{number}-{column}:" in " ".join(prefix.split()[:-1])
                        for column in range(max_column + 1)
                    )
                    if row_complete or (not in_row and _already_predicted(prefix, character)):
                        rewards.append(-1.0)
                        wrong = True
                    else:
                        rewards.append(1.0)
                else:
                    invalid_position = True
                    rewards.append(-2.0 if upper_bound > 0 and number / upper_bound > 10 else -1.0)
                    wrong = True
            elif number not in true_values:
                rewards.append(-1.0)
            else:
                recovered_prediction = ""
                for previous in reversed(prefix):
                    if previous not in "0123456789:-":
                        break
                    recovered_prediction = previous + recovered_prediction
                recovered_prediction += character
                if recovered_prediction in target_text:
                    rewards.append(1.0)
                elif invalid_position:
                    rewards.append(0.0)
                else:
                    rewards.append(-1.0)
            continue

        if character == ":":
            if index == 0 or in_row or in_prediction or not prefix[-1].isdigit():
                rewards.append(-1.0)
                wrong = True
            else:
                rewards.append(1.0)
            in_prediction = True
            in_row = False
            continue

        if character == " ":
            invalid_position = False
            if not in_prediction or (index > 0 and prefix[-1] == ":"):
                rewards.append(-1.0)
            else:
                recovered_prediction = ""
                for previous in reversed(prefix):
                    if previous not in "0123456789:-":
                        break
                    recovered_prediction = previous + recovered_prediction
                if recovered_prediction in target_text:
                    good_separator = (
                        recovered_prediction + " " in target_text
                        or recovered_prediction + tokenizer.eos_token in target_text
                    )
                    rewards.append(1.0 if good_separator else -1.0)
                else:
                    rewards.append(-1.0 if pair_correctness_mode else 1.0)
            wrong = False
            in_prediction = False
            in_row = True
            continue

        if character == "-":
            rewards.append(1.0 if in_row else -1.0)
            if not in_row:
                wrong = True
            in_row = False
            continue

        rewards.append(0.0)

    return torch.tensor(rewards, dtype=torch.float32)


def arc_token_types(token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Identify output-token roles for the same per-type baselines as rl_small.py."""

    tokens = torch.as_tensor(token_ids, dtype=torch.long).flatten().tolist()
    # 1 space, 2 hyphen, 3 colon, 4 row, 5 column, 6 value, 7 EOS.
    state = next_state = 4
    result: list[int] = []
    for token in tokens:
        character = ARC_CHARACTERS[token]
        if character == " ":
            token_type = 1
            state = next_state = 4
        elif character == "-":
            token_type = 2
            state = next_state = 5
        elif character == ":":
            token_type = 3
            state = next_state = 6
        elif character.isdigit():
            token_type = state
            next_state = 2 if state == 4 else (5 if state == 5 else 1)
        elif token == EOS_TOKEN_ID:
            token_type = 7
        else:
            token_type = next_state
        result.append(token_type)
    return torch.tensor(result, dtype=torch.long)


@dataclass(frozen=True)
class RewardBatch:
    token_rewards: torch.Tensor
    sequence_rewards: torch.Tensor
    token_baselines: torch.Tensor
    token_stds: torch.Tensor
    sequence_baselines: torch.Tensor
    sequence_std: torch.Tensor
    mask: torch.Tensor
    sequence_mask: torch.Tensor | None = None


def pair_correctness_token_rewards(
    target_ids: Sequence[int] | torch.Tensor,
    predicted_ids: Sequence[int] | torch.Tensor,
    tokenizer: ArcCharacterTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign exact-pair correctness only to numeric tokens in each candidate.

    A numeric token gets 1 when the whitespace-delimited candidate containing
    it is exactly one of the target ``row-column:value`` pairs, and 0 otherwise.
    Structural tokens never receive this auxiliary reward; their signal remains
    exclusively the dense syntax/token reward.
    """

    target_pairs = _target_pairs(tokenizer.decode(target_ids), tokenizer.eos_token)
    predicted = torch.as_tensor(predicted_ids, dtype=torch.long).flatten().cpu()
    rewards = torch.zeros(predicted.numel(), dtype=torch.float32)
    number_mask = torch.zeros(predicted.numel(), dtype=torch.bool)
    candidate_start = 0

    def finish_candidate(end: int) -> None:
        if end <= candidate_start:
            return
        candidate = tokenizer.decode(predicted[candidate_start:end])
        correct = float(candidate in target_pairs)
        for position in range(candidate_start, end):
            if tokenizer.decode([predicted[position]]).isdigit():
                number_mask[position] = True
                rewards[position] = correct

    for index, token_id in enumerate(predicted.tolist()):
        character = tokenizer.decode([token_id])
        if character == " " or token_id == tokenizer.eos_token_id:
            finish_candidate(index)
            candidate_start = index + 1
            if token_id == tokenizer.eos_token_id:
                break
    else:
        finish_candidate(predicted.numel())
    return rewards, number_mask


def build_reward_batch(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    tokenizer: ArcCharacterTokenizer,
    *,
    group_size: int | None = None,
    sequence_reward_mode: str = "binary",
    device: torch.device,
) -> RewardBatch:
    """Score rollouts and normalize reward statistics within each ARC problem."""

    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must have the same non-zero length")
    if group_size is None:
        group_size = len(predictions)
    if group_size <= 1 or len(predictions) % group_size:
        raise ValueError("group_size must exceed one and evenly divide the rollout batch")
    if sequence_reward_mode not in ("binary", "coverage", "pair_correctness"):
        raise ValueError(
            "sequence_reward_mode must be binary, coverage, or pair_correctness"
        )
    max_length = max(prediction.numel() for prediction in predictions)
    shape = (len(predictions), max_length)
    rewards = torch.zeros(shape, dtype=torch.float32, device=device)
    sequence_rewards = torch.zeros_like(rewards)
    token_types = torch.zeros(shape, dtype=torch.long, device=device)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    sequence_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    for row, (prediction, target) in enumerate(zip(predictions, targets)):
        length = prediction.numel()
        mask[row, :length] = True
        rewards[row, :length] = arc_token_rewards(
            target,
            prediction,
            tokenizer,
            pair_correctness_mode=(sequence_reward_mode == "pair_correctness"),
        ).to(device)
        token_types[row, :length] = arc_token_types(prediction).to(device)
        if sequence_reward_mode == "pair_correctness":
            pair_rewards, pair_mask = pair_correctness_token_rewards(
                target, prediction, tokenizer
            )
            sequence_rewards[row, :length] = pair_rewards.to(device)
            sequence_mask[row, :length] = pair_mask.to(device)
        else:
            target_pairs = _target_pairs(tokenizer.decode(target), tokenizer.eos_token)
            predicted_pairs = _target_pairs(
                tokenizer.decode(prediction), tokenizer.eos_token
            )
            score = (
                float(target_pairs == predicted_pairs)
                if sequence_reward_mode == "binary"
                else len(target_pairs & predicted_pairs) / max(len(target_pairs), 1)
            )
            sequence_rewards[row, :length] = score
            sequence_mask[row, :length] = True

    baselines = torch.zeros_like(rewards)
    stds = torch.zeros_like(rewards)
    sequence_baselines = torch.zeros_like(sequence_rewards)
    sequence_stds = torch.zeros_like(sequence_rewards)
    for start in range(0, len(predictions), group_size):
        end = start + group_size
        group_rows = torch.zeros_like(mask)
        group_rows[start:end] = True
        for token_type in range(1, 8):
            type_mask = group_rows & mask & (token_types == token_type)
            if type_mask.any():
                values = rewards[type_mask]
                baselines[type_mask] = values.mean()
                stds[type_mask] = values.std(unbiased=False)
        group_sequence_mask = group_rows & sequence_mask
        if group_sequence_mask.any():
            values = sequence_rewards[group_sequence_mask]
            sequence_baselines[group_sequence_mask] = values.mean()
            sequence_stds[group_sequence_mask] = values.std(unbiased=False)
    return RewardBatch(
        token_rewards=rewards,
        sequence_rewards=sequence_rewards,
        token_baselines=baselines,
        token_stds=stds,
        sequence_baselines=sequence_baselines,
        sequence_std=sequence_stds,
        mask=mask,
        sequence_mask=sequence_mask,
    )


def current_policy_logprobs(
    model: Qwen3ForCausalLM,
    prompts: Sequence[torch.Tensor],
    completions: Sequence[torch.Tensor],
    *,
    multiple_of: int = 16,
) -> torch.Tensor:
    """Differentiably score completion tokens with correct causal alignment."""

    if len(prompts) != len(completions) or not prompts:
        raise ValueError("prompts and completions must have the same non-zero length")
    device = next(model.parameters()).device
    full_lengths = [prompt.numel() + completion.numel() for prompt, completion in zip(prompts, completions)]
    padded_length = math.ceil(max(full_lengths) / multiple_of) * multiple_of
    input_ids = torch.full(
        (len(prompts), padded_length), PAD_TOKEN_ID, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        full_sequence = torch.cat((prompt, completion)).to(device)
        input_ids[row, : full_sequence.numel()] = full_sequence
        attention_mask[row, : full_sequence.numel()] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    log_distributions = outputs.logits.float().log_softmax(dim=-1)
    max_completion = max(completion.numel() for completion in completions)
    rows: list[torch.Tensor] = []
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        completion = completion.to(device)
        # Logit p-1 predicts the first completion token; rl_small.py used p,
        # which was an off-by-one error.
        positions = torch.arange(completion.numel(), device=device) + prompt.numel() - 1
        selected = log_distributions[row, positions].gather(
            -1, completion.unsqueeze(-1)
        ).squeeze(-1)
        rows.append(F.pad(selected, (0, max_completion - completion.numel())))
    return torch.stack(rows)


def compute_clipped_policy_loss(
    current_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    reward_batch: RewardBatch,
    *,
    clip_epsilon: float = 0.2,
    sequence_weight: float = 0.5,
    pair_correctness_weight: float | None = None,
    extra_advantages: torch.Tensor | None = None,
    loss_reduction: str = "token",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO-style loss from rl_small.py, with safe masking and normalization."""

    if not 0.0 <= sequence_weight <= 1.0:
        raise ValueError("sequence_weight must be in [0, 1]")
    if pair_correctness_weight is not None and not 0.0 <= pair_correctness_weight < 1.0:
        raise ValueError("pair_correctness_weight must be in [0, 1)")
    if loss_reduction not in {"token", "sequence"}:
        raise ValueError("loss_reduction must be 'token' or 'sequence'")
    mask = reward_batch.mask
    if current_logprobs.shape != mask.shape or rollout_logprobs.shape != mask.shape:
        raise ValueError("log-probability tensors and reward mask must have equal shapes")

    token_advantages = (
        (reward_batch.token_rewards - reward_batch.token_baselines)
        / (reward_batch.token_stds + 1e-8)
    )
    sequence_advantages = (
        (reward_batch.sequence_rewards - reward_batch.sequence_baselines)
        / (reward_batch.sequence_std + 1e-8)
    )
    if pair_correctness_weight is None:
        advantages = (
            (1.0 - sequence_weight) * token_advantages
            + sequence_weight * sequence_advantages
        )
    else:
        sequence_mask = reward_batch.sequence_mask
        if sequence_mask is None:
            raise ValueError("pair correctness loss requires a sequence_mask")
        # Dense token/syntax credit has unit weight. Exact-pair credit is a
        # smaller additive term and is zero on spaces, hyphens, colons and EOS.
        advantages = token_advantages + pair_correctness_weight * (
            sequence_advantages * sequence_mask
        )
    if extra_advantages is not None:
        if extra_advantages.shape != mask.shape:
            raise ValueError("extra_advantages and reward mask must have equal shapes")
        advantages = advantages + extra_advantages
    advantages = advantages.detach()
    advantages = advantages * mask

    log_ratio = (current_logprobs - rollout_logprobs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    objective = torch.minimum(unclipped, clipped) * mask
    if loss_reduction == "token":
        loss = -objective.sum() / mask.sum().clamp_min(1)
    else:
        token_counts = mask.sum(dim=1)
        sequence_objectives = objective.sum(dim=1) / token_counts.clamp_min(1)
        loss = -sequence_objectives.sum() / (token_counts > 0).sum().clamp_min(1)

    valid_ratio = ratio[mask]
    valid_advantages = advantages[mask]
    metrics = {
        "advantage_mean": valid_advantages.mean(),
        "advantage_abs_mean": valid_advantages.abs().mean(),
        "advantage_nonzero_fraction": (valid_advantages != 0).float().mean(),
        "extra_advantage_abs_mean": (
            torch.zeros((), dtype=advantages.dtype, device=advantages.device)
            if extra_advantages is None
            else extra_advantages[mask].abs().mean()
        ),
        "clip_fraction": (
            (valid_ratio < 1.0 - clip_epsilon)
            | (valid_ratio > 1.0 + clip_epsilon)
        ).float().mean(),
        "ratio_mean": valid_ratio.mean(),
        "token_reward_mean": reward_batch.token_rewards[mask].mean(),
        "sequence_reward_mean": reward_batch.sequence_rewards[mask].mean(),
        "pair_correctness_reward_mean": (
            reward_batch.sequence_rewards[
                reward_batch.sequence_mask
                if reward_batch.sequence_mask is not None
                else mask
            ].mean()
        ),
    }
    return loss, metrics


@dataclass(frozen=True)
class RLTrainingConfig:
    steps: int = 1
    problems_per_step: int = 2
    rollouts_per_problem: int = 4
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    clip_epsilon: float = 0.2
    sequence_weight: float = 0.5
    max_grad_norm: float = 1.0
    eval_every: int = 1
    seed: int = 3407


@dataclass(frozen=True)
class RLEvalMetrics:
    step: int
    loss: float
    mean_reward: float
    num_correct: int
    num_samples: int
    pair_precision: float
    pair_recall: float
    good_pairs: int
    candidate_pairs: int
    covered_target_pairs: int
    target_pairs: int
    mean_length: float
    min_length: int
    max_length: int


@dataclass(frozen=True)
class RLStepMetrics:
    step: int
    loss: float
    grad_norm: float
    grad_mean: float
    grad_variance: float
    rollout_seconds: float
    rollout_tokens_per_second: float
    weight_sync_seconds: float
    rollout_logprob_mae: float
    mean_length: float
    correct_fraction: float
    token_reward_mean: float
    num_correct: int
    num_samples: int
    samples_seen: int
    eval_metrics: RLEvalMetrics | None
    advantage_abs_mean: float
    ratio_mean: float
    clip_fraction: float


def pair_evaluation_counts(
    targets: Sequence[torch.Tensor],
    predictions: Sequence[torch.Tensor],
    tokenizer: ArcCharacterTokenizer,
) -> tuple[int, int, int, int]:
    """Count duplicate-sensitive precision and unique target-pair coverage."""

    if len(targets) != len(predictions):
        raise ValueError("evaluation targets and predictions must have equal length")
    good_pairs = candidate_pairs = covered_target_pairs = target_pairs = 0
    for target, prediction in zip(targets, predictions):
        target_set = _target_pairs(tokenizer.decode(target), tokenizer.eos_token)
        prediction_text = tokenizer.decode(prediction).split(tokenizer.eos_token, 1)[0]
        candidates = [piece for piece in prediction_text.split() if piece]
        predicted_set = set(candidates)
        good_pairs += sum(candidate in target_set for candidate in candidates)
        candidate_pairs += len(candidates)
        covered_target_pairs += len(predicted_set & target_set)
        target_pairs += len(target_set)
    return good_pairs, candidate_pairs, covered_target_pairs, target_pairs


def gradient_mean_and_variance(model: torch.nn.Module) -> tuple[float, float]:
    """Return element-wise mean and population variance of current gradients."""

    device = next(model.parameters()).device
    gradient_sum = torch.zeros((), dtype=torch.float32, device=device)
    gradient_square_sum = torch.zeros((), dtype=torch.float32, device=device)
    gradient_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        gradient_sum += gradient.sum()
        gradient_square_sum += gradient.square().sum()
        gradient_count += gradient.numel()

    if gradient_count == 0:
        return 0.0, 0.0
    gradient_mean = gradient_sum / gradient_count
    gradient_variance = (
        gradient_square_sum / gradient_count - gradient_mean.square()
    ).clamp_min(0.0)
    return float(gradient_mean), float(gradient_variance)


def evaluate_policy(
    model: Qwen3ForCausalLM,
    tokenizer: ArcCharacterTokenizer,
    examples: Sequence[ARCExample],
    generator: BucketedRolloutGenerator | VLLMRolloutGenerator,
    *,
    step: int,
    max_new_tokens: int,
) -> RLEvalMetrics:
    """Evaluate unaugmented ARC examples with golden NLL and greedy rollouts."""

    if not examples:
        raise ValueError("At least one evaluation example is required")

    prompts = [example.prompt_ids for example in examples]
    targets = [example.target_ids for example in examples]
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            golden_logprobs = current_policy_logprobs(model, prompts, targets)
            valid_golden_logprobs = torch.cat(
                [
                    golden_logprobs[row, : target.numel()]
                    for row, target in enumerate(targets)
                ]
            )
            eval_loss = -valid_golden_logprobs.mean()

        rollouts = generator.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    finally:
        model.train(was_training)

    token_rewards = torch.cat(
        [
            arc_token_rewards(target, prediction, tokenizer)
            for target, prediction in zip(targets, rollouts.token_ids)
        ]
    )
    num_correct = sum(
        completion_is_correct(target, prediction, tokenizer)
        for target, prediction in zip(targets, rollouts.token_ids)
    )
    lengths = [int(prediction.numel()) for prediction in rollouts.token_ids]
    generated_tokens = sum(lengths)
    good_pairs, candidate_pairs, covered_target_pairs, target_pairs = (
        pair_evaluation_counts(targets, rollouts.token_ids, tokenizer)
    )
    return RLEvalMetrics(
        step=step,
        loss=float(eval_loss),
        mean_reward=float(token_rewards.mean()),
        num_correct=num_correct,
        num_samples=len(examples),
        pair_precision=good_pairs / max(candidate_pairs, 1),
        pair_recall=covered_target_pairs / max(target_pairs, 1),
        good_pairs=good_pairs,
        candidate_pairs=candidate_pairs,
        covered_target_pairs=covered_target_pairs,
        target_pairs=target_pairs,
        mean_length=generated_tokens / len(examples),
        min_length=min(lengths),
        max_length=max(lengths),
    )


def log_step_metrics(metrics: RLStepMetrics) -> None:
    """Emit immediately visible train and optional evaluation metrics."""

    print(
        f"train step={metrics.step} samples_seen={metrics.samples_seen} "
        f"batch_samples={metrics.num_samples} loss={metrics.loss:.6f} "
        f"grad_mean={metrics.grad_mean:.6e} "
        f"grad_variance={metrics.grad_variance:.6e} "
        f"grad_norm={metrics.grad_norm:.6f} "
        f"mean_reward={metrics.token_reward_mean:.6f} "
        f"num_correct={metrics.num_correct}/{metrics.num_samples} "
        f"rollout_seconds={metrics.rollout_seconds:.3f} "
        f"rollout_tokens_per_second={metrics.rollout_tokens_per_second:.1f} "
        f"weight_sync_seconds={metrics.weight_sync_seconds:.3f} "
        f"rollout_logprob_mae={metrics.rollout_logprob_mae:.4f} "
        f"mean_length={metrics.mean_length:.1f} "
        f"ratio={metrics.ratio_mean:.4f} clipped={metrics.clip_fraction:.3f}",
        flush=True,
    )
    if metrics.eval_metrics is not None:
        evaluation = metrics.eval_metrics
        print(
            f"eval step={evaluation.step} samples={evaluation.num_samples} "
            f"loss={evaluation.loss:.6f} grad_mean=n/a grad_variance=n/a "
            f"mean_reward={evaluation.mean_reward:.6f} "
            f"num_correct={evaluation.num_correct}/{evaluation.num_samples} "
            f"mean_length={evaluation.mean_length:.1f}",
            flush=True,
        )


def train_rl(
    model: Qwen3ForCausalLM,
    tokenizer: ArcCharacterTokenizer,
    examples: Sequence[ARCExample],
    generator: BucketedRolloutGenerator | VLLMRolloutGenerator,
    config: RLTrainingConfig,
    *,
    eval_examples: Sequence[ARCExample] = (),
    optimizer: torch.optim.Optimizer | None = None,
) -> list[RLStepMetrics]:
    """Run on-policy ARC RL updates using the bucketed static-cache generator."""

    if config.steps < 0 or config.problems_per_step <= 0 or config.rollouts_per_problem <= 0:
        raise ValueError("steps must be non-negative and batch sizes must be positive")
    if config.eval_every < 0:
        raise ValueError("eval_every must be non-negative")
    if not examples and config.steps:
        raise ValueError("At least one training example is required")
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    device = next(model.parameters()).device
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    history: list[RLStepMetrics] = []
    model.train()
    if eval_examples and config.eval_every > 0:
        initial_eval = evaluate_policy(
            model,
            tokenizer,
            eval_examples,
            generator,
            step=0,
            max_new_tokens=config.max_new_tokens,
        )
        print(
            f"eval step=0 samples={initial_eval.num_samples} "
            f"loss={initial_eval.loss:.6f} grad_mean=n/a grad_variance=n/a "
            f"mean_reward={initial_eval.mean_reward:.6f} "
            f"num_correct={initial_eval.num_correct}/{initial_eval.num_samples} "
            f"mean_length={initial_eval.mean_length:.1f}",
            flush=True,
        )

    for step in range(1, config.steps + 1):
        chosen = (
            rng.sample(list(examples), config.problems_per_step)
            if len(examples) >= config.problems_per_step
            else rng.choices(list(examples), k=config.problems_per_step)
        )
        prompts = [
            example.prompt_ids
            for example in chosen
            for _ in range(config.rollouts_per_problem)
        ]
        targets = [
            example.target_ids
            for example in chosen
            for _ in range(config.rollouts_per_problem)
        ]

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rollout_started = time.perf_counter()
        rollouts = generator.generate(
            prompts,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rollout_seconds = time.perf_counter() - rollout_started
        generated_tokens = sum(item.numel() for item in rollouts.token_ids)
        max_completion = max(item.numel() for item in rollouts.token_ids)
        rollout_logprobs = pad_token_logprobs(
            rollouts.token_logprobs, multiple_of=1
        )[:, :max_completion].to(device)
        reward_batch = build_reward_batch(
            rollouts.token_ids, targets, tokenizer, device=device
        )

        optimizer.zero_grad(set_to_none=True)
        model.train()
        current_logprobs = current_policy_logprobs(model, prompts, rollouts.token_ids)
        rollout_logprob_mae = float(
            (current_logprobs.detach() - rollout_logprobs)[reward_batch.mask]
            .abs()
            .mean()
        )
        if isinstance(generator, VLLMRolloutGenerator):
            rollout_logprobs = current_logprobs.detach()
        loss, loss_metrics = compute_clipped_policy_loss(
            current_logprobs,
            rollout_logprobs,
            reward_batch,
            clip_epsilon=config.clip_epsilon,
            sequence_weight=config.sequence_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite RL loss at step {step}: {loss.item()}")
        loss.backward()
        grad_mean, grad_variance = gradient_mean_and_variance(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        weight_sync_seconds = 0.0
        if isinstance(generator, VLLMRolloutGenerator):
            weight_sync_seconds = generator.sync_from_model(model)

        correct = [
            completion_is_correct(target, prediction, tokenizer)
            for target, prediction in zip(targets, rollouts.token_ids)
        ]
        eval_metrics = None
        if (
            eval_examples
            and config.eval_every > 0
            and step % config.eval_every == 0
        ):
            eval_metrics = evaluate_policy(
                model,
                tokenizer,
                eval_examples,
                generator,
                step=step,
                max_new_tokens=config.max_new_tokens,
            )
        metrics = RLStepMetrics(
            step=step,
            loss=float(loss.detach()),
            grad_norm=float(grad_norm.detach()),
            grad_mean=grad_mean,
            grad_variance=grad_variance,
            rollout_seconds=rollout_seconds,
            rollout_tokens_per_second=generated_tokens / rollout_seconds,
            weight_sync_seconds=weight_sync_seconds,
            rollout_logprob_mae=rollout_logprob_mae,
            mean_length=sum(item.numel() for item in rollouts.token_ids) / len(prompts),
            correct_fraction=sum(correct) / len(correct),
            token_reward_mean=float(loss_metrics["token_reward_mean"].detach()),
            num_correct=sum(correct),
            num_samples=len(prompts),
            samples_seen=step * len(prompts),
            eval_metrics=eval_metrics,
            advantage_abs_mean=float(loss_metrics["advantage_abs_mean"].detach()),
            ratio_mean=float(loss_metrics["ratio_mean"].detach()),
            clip_fraction=float(loss_metrics["clip_fraction"].detach()),
        )
        history.append(metrics)
        log_step_metrics(metrics)
    return history


def save_model(model: Qwen3ForCausalLM, path: str | Path) -> Path:
    """Save only policy weights in the same strict-loadable safetensors format."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(state_dict, str(path))
    return path


def export_vllm_model(
    checkpoint_path: str | Path, output_dir: str | Path, *, dtype: str
) -> Path:
    """Create the local Hugging Face directory that vLLM expects."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_model_config()
    config.architectures = ["Qwen3ForCausalLM"]
    config.dtype = getattr(torch, dtype)
    config.save_pretrained(output_dir)
    shutil.copy2(checkpoint_path, output_dir / "model.safetensors")
    return output_dir


@dataclass(frozen=True)
class RolloutBenchmarkMeasurement:
    backend: str
    cohort: str
    prompt_count: int
    prompt_tokens: int
    generated_tokens: int
    seconds: float

    @property
    def generated_tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds


def select_benchmark_cohorts(
    examples: Sequence[ARCExample],
    *,
    prompts_per_cohort: int,
    rollouts_per_prompt: int,
) -> dict[str, list[torch.Tensor]]:
    """Select actual shortest and longest ARC prompts, then repeat rollouts."""

    if prompts_per_cohort <= 0 or rollouts_per_prompt <= 0:
        raise ValueError("Benchmark prompt and rollout counts must be positive")
    ordered = sorted(examples, key=lambda example: example.prompt_ids.numel())
    count = min(prompts_per_cohort, len(ordered) // 2)
    if count == 0:
        raise ValueError("At least two usable ARC examples are required")
    selected = {"short": ordered[:count], "long": ordered[-count:]}
    return {
        name: [
            example.prompt_ids
            for example in cohort
            for _ in range(rollouts_per_prompt)
        ]
        for name, cohort in selected.items()
    }


def measure_rollout_generator(
    generator,
    cohorts: dict[str, list[torch.Tensor]],
    *,
    backend: str,
    max_new_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    do_sample: bool,
    seed: int,
    device: torch.device,
) -> list[RolloutBenchmarkMeasurement]:
    """Warm and time one backend, counting only real generated tokens."""

    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    measurements: list[RolloutBenchmarkMeasurement] = []
    for cohort_name, prompts in cohorts.items():
        for run in range(warmup_runs):
            torch.manual_seed(seed + run)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": 1.0,
                "top_p": 1.0,
            }
            generator.generate(prompts, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        generated_tokens = 0
        for run in range(measured_runs):
            torch.manual_seed(seed + warmup_runs + run)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": 1.0,
                "top_p": 1.0,
            }
            output = generator.generate(prompts, **kwargs)
            generated_tokens += sum(tokens.numel() for tokens in output.token_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        measurements.append(
            RolloutBenchmarkMeasurement(
                backend=backend,
                cohort=cohort_name,
                prompt_count=len(prompts) * measured_runs,
                prompt_tokens=(
                    sum(prompt.numel() for prompt in prompts) * measured_runs
                ),
                generated_tokens=generated_tokens,
                seconds=seconds,
            )
        )
    return measurements


def run_vllm_benchmark(
    args: argparse.Namespace,
    tokenizer: ArcCharacterTokenizer,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Compare static-cache Transformers generation and vLLM on ARC prompts."""

    max_model_len = max(args.buckets)
    if args.benchmark_max_new_tokens <= 0:
        raise ValueError("benchmark_max_new_tokens must be positive")
    max_prompt_tokens = max_model_len - args.benchmark_max_new_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("Largest bucket must exceed benchmark_max_new_tokens")

    examples, _ = load_arc_examples(
        args.data_dir,
        split="training",
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=None,
        max_total_tokens=None,
        augment=False,
        seed=args.seed,
    )
    cohorts = select_benchmark_cohorts(
        examples,
        prompts_per_cohort=args.benchmark_prompts_per_cohort,
        rollouts_per_prompt=args.benchmark_rollouts_per_prompt,
    )
    for name, prompts in cohorts.items():
        lengths = sorted(prompt.numel() for prompt in prompts)
        print(
            f"benchmark cohort={name} requests={len(prompts)} "
            f"prompt_tokens_min={lengths[0]} "
            f"prompt_tokens_median={lengths[len(lengths) // 2]} "
            f"prompt_tokens_max={lengths[-1]}",
            flush=True,
        )

    model = load_model(args.checkpoint, device=device, dtype=dtype)
    torch_generator = BucketedRolloutGenerator(
        model,
        buckets=args.buckets,
        rollout_batch_size=args.rollout_batch_size,
        compile_decode=not args.no_compile,
    )
    torch_results = measure_rollout_generator(
        torch_generator,
        cohorts,
        backend="torch",
        max_new_tokens=args.benchmark_max_new_tokens,
        warmup_runs=args.benchmark_warmup_runs,
        measured_runs=args.benchmark_runs,
        do_sample=not args.benchmark_greedy,
        seed=args.seed,
        device=device,
    )

    with tempfile.TemporaryDirectory(prefix="arc-vllm-model-") as temporary_dir:
        model_dir = export_vllm_model(
            args.checkpoint, temporary_dir, dtype=args.dtype
        )
        del torch_generator
        del model
        gc.collect()
        torch.cuda.empty_cache()

        vllm_generator = VLLMRolloutGenerator(
            model_dir,
            dtype=args.dtype,
            max_model_len=max_model_len,
            max_num_seqs=max(
                args.rollout_batch_size,
                max(len(prompts) for prompts in cohorts.values()),
            ),
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            seed=args.seed,
            enforce_eager=args.vllm_enforce_eager,
        )
        vllm_results = measure_rollout_generator(
            vllm_generator,
            cohorts,
            backend="vllm",
            max_new_tokens=args.benchmark_max_new_tokens,
            warmup_runs=args.benchmark_warmup_runs,
            measured_runs=args.benchmark_runs,
            do_sample=not args.benchmark_greedy,
            seed=args.seed,
            device=device,
        )

    by_backend = {
        (measurement.backend, measurement.cohort): measurement
        for measurement in torch_results + vllm_results
    }
    for cohort in ("short", "long"):
        torch_result = by_backend[("torch", cohort)]
        vllm_result = by_backend[("vllm", cohort)]
        for result in (torch_result, vllm_result):
            print(
                f"benchmark backend={result.backend} cohort={result.cohort} "
                f"requests={result.prompt_count} prompt_tokens={result.prompt_tokens} "
                f"generated_tokens={result.generated_tokens} "
                f"seconds={result.seconds:.3f} "
                f"generated_tokens_per_second="
                f"{result.generated_tokens_per_second:.1f}",
                flush=True,
            )
        print(
            f"benchmark speedup cohort={cohort} "
            f"vllm_over_torch="
            f"{vllm_result.generated_tokens_per_second / torch_result.generated_tokens_per_second:.3f}x",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ARC_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--buckets", type=int, nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--rollout-batch-size", type=int, default=DEFAULT_ROLLOUT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-dataset-tokens", type=int, default=2048)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--num-examples", type=int, help="Optional base-task limit; default loads all training tasks")
    parser.add_argument("--num-eval-examples", type=int, default=8)
    parser.add_argument("--num-color-swaps", type=int, choices=(0, 1), default=1)
    parser.add_argument("--no-rotations", action="store_true")
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable train-only rotations and color permutations",
    )
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N updates; 0 disables evaluation")
    parser.add_argument("--problems-per-step", type=int, default=2)
    parser.add_argument("--rollouts-per-problem", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--check-samples", type=int, default=0)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--no-compile", action="store_true", help="Disable automatic decode compilation")
    parser.add_argument(
        "--rollout-backend",
        choices=("torch", "vllm"),
        default="vllm",
        help="Backend for training and evaluation rollouts",
    )
    parser.add_argument(
        "--benchmark-vllm",
        action="store_true",
        help="Benchmark torch.generate against vLLM and exit without training",
    )
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=256)
    parser.add_argument("--benchmark-prompts-per-cohort", type=int, default=8)
    parser.add_argument("--benchmark-rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--benchmark-warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument(
        "--benchmark-greedy",
        action="store_true",
        help="Use greedy decoding; default samples like RL rollouts",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graphs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_dataset_tokens <= 0:
        raise ValueError("max_dataset_tokens must be positive")
    if args.num_examples is not None and args.num_examples <= 0:
        raise ValueError("num_examples must be positive when supplied")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(
            "This training entry point is GPU-only; pass a CUDA device such as --device cuda:0"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required, but this PyTorch build cannot access a CUDA-capable GPU"
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device.index} is unavailable; "
            f"found {torch.cuda.device_count()} CUDA device(s)"
        )
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The selected CUDA GPU does not support bfloat16; pass --dtype float16"
        )

    tokenizer = ArcCharacterTokenizer()
    if args.benchmark_vllm:
        print(
            f"device={device}, gpu={torch.cuda.get_device_name(device)}, dtype={dtype}",
            flush=True,
        )
        run_vllm_benchmark(args, tokenizer, device=device, dtype=dtype)
        return

    model = load_model(args.checkpoint, device=device, dtype=dtype)
    print(
        f"device={device}, gpu={torch.cuda.get_device_name(device)}, dtype={dtype}, "
        f"{parameter_summary(model)}"
    )

    vllm_temp_dir = None
    if args.rollout_backend == "torch":
        generator = BucketedRolloutGenerator(
            model,
            buckets=args.buckets,
            rollout_batch_size=args.rollout_batch_size,
            compile_decode=not args.no_compile,
        )
    else:
        print("startup phase=initialize_vllm", flush=True)
        vllm_temp_dir = tempfile.TemporaryDirectory(prefix="arc-vllm-training-")
        model_dir = export_vllm_model(
            args.checkpoint, vllm_temp_dir.name, dtype=args.dtype
        )
        generator = VLLMRolloutGenerator(
            model_dir,
            dtype=args.dtype,
            max_model_len=max(args.buckets),
            max_num_seqs=max(args.rollout_batch_size, args.num_eval_examples),
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            seed=args.seed,
            enforce_eager=args.vllm_enforce_eager,
        )
        print("startup phase=vllm_ready", flush=True)
    max_prompt_tokens = max(args.buckets) - args.max_new_tokens
    examples, train_summary = load_arc_examples(
        args.data_dir,
        split="training",
        tokenizer=tokenizer,
        base_task_limit=args.num_examples,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=args.max_new_tokens,
        max_total_tokens=args.max_dataset_tokens,
        augment=not args.no_augmentation,
        add_rotations=not args.no_rotations,
        num_color_swaps=args.num_color_swaps,
        seed=args.seed,
    )
    eval_examples: list[ARCExample] = []
    eval_summary = None
    if args.eval_every > 0 and args.num_eval_examples > 0:
        eval_examples, eval_summary = load_arc_examples(
            args.data_dir,
            split="evaluation",
            tokenizer=tokenizer,
            base_task_limit=args.num_eval_examples,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=args.max_new_tokens,
            max_total_tokens=args.max_dataset_tokens,
            augment=False,
            seed=args.seed,
        )

    print(
        f"dataset split=train base_tasks={train_summary.base_tasks} "
        f"candidate_samples={train_summary.candidate_examples} "
        f"loaded_samples={train_summary.loaded_examples} "
        f"skipped_prompt_too_long={train_summary.skipped_prompt_too_long} "
        f"skipped_target_too_long={train_summary.skipped_target_too_long} "
        f"skipped_total_too_long={train_summary.skipped_total_too_long} "
        f"max_prompt_ids={train_summary.max_prompt_length} "
        f"max_golden_ids={train_summary.max_target_length} "
        f"max_total_ids={train_summary.max_total_length} "
        f"rotations={not args.no_augmentation and not args.no_rotations} "
        f"color_swaps={0 if args.no_augmentation else args.num_color_swaps}",
        flush=True,
    )
    if eval_summary is not None:
        print(
            f"dataset split=eval base_tasks={eval_summary.base_tasks} "
            f"candidate_samples={eval_summary.candidate_examples} "
            f"loaded_samples={eval_summary.loaded_examples} "
            f"skipped_prompt_too_long={eval_summary.skipped_prompt_too_long} "
            f"skipped_target_too_long={eval_summary.skipped_target_too_long} "
            f"skipped_total_too_long={eval_summary.skipped_total_too_long} "
            f"max_prompt_ids={eval_summary.max_prompt_length} "
            f"max_golden_ids={eval_summary.max_target_length} "
            f"max_total_ids={eval_summary.max_total_length} "
            f"rotations=False color_swaps=0",
            flush=True,
        )
    print(
        f"length_limits dataset_prompt_plus_golden_ids<={args.max_dataset_tokens} "
        f"cache_prompt_ids<={max_prompt_tokens} "
        f"golden_ids<={args.max_new_tokens} "
        f"rollout_new_tokens<={args.max_new_tokens}",
        flush=True,
    )

    history = train_rl(
        model,
        tokenizer,
        examples,
        generator,
        RLTrainingConfig(
            steps=args.train_steps,
            problems_per_step=args.problems_per_step,
            rollouts_per_problem=args.rollouts_per_problem,
            max_new_tokens=args.max_new_tokens,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            seed=args.seed,
        ),
        eval_examples=eval_examples,
    )

    if args.output_checkpoint is not None:
        print("saved_checkpoint:", save_model(model, args.output_checkpoint))

    check_examples = (eval_examples or examples)[: max(0, args.check_samples)]
    if check_examples:
        checks = generator.generate(
            [example.prompt_ids for example in check_examples],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        for example, prediction, bucket in zip(
            check_examples, checks.token_ids, checks.bucket_by_prompt
        ):
            print(
                f"check task={example.task_id} bucket={bucket} "
                f"length={prediction.numel()} "
                f"correct={completion_is_correct(example.target_ids, prediction, tokenizer)}"
            )
            print("  target:", tokenizer.decode(example.target_ids))
            print("  generated:", tokenizer.decode(prediction))
    if isinstance(generator, BucketedRolloutGenerator):
        print("allocated cache keys:", generator.cache_pool.allocated_keys)
    else:
        print("rollout backend: vllm", flush=True)


if __name__ == "__main__":
    main()

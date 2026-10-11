"""ARC-character-tokenizer pretraining for the Qwen3.5-0.8B text model.

This runner deliberately replaces the native 248k-token embedding/output
matrix with the 24-token ``ArcCharacterTokenizer`` while inheriting every
pretrained text-body weight.  It reuses the data, optimization, logging and
checkpoint format from :mod:`pretrain_arc_qwen06b`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
)

from . import pretrain_qwen3 as trainer
from .rl_static_cache import (
    ARCExample,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    ArcCharacterTokenizer,
    RolloutBatch,
    evaluate_policy,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def custom_text_config(model_id: str = DEFAULT_MODEL_ID) -> Qwen3_5TextConfig:
    """Copy the official text config, changing only its token interface."""

    source = AutoConfig.from_pretrained(model_id)
    config = deepcopy(source.text_config if hasattr(source, "text_config") else source)
    config.vocab_size = VOCAB_SIZE
    config.pad_token_id = PAD_TOKEN_ID
    config.eos_token_id = EOS_TOKEN_ID
    config.bos_token_id = None
    # Qwen3.5-0.8B ties its native embedding and output projection. Preserve
    # that architectural choice for the new ARC matrices.
    config.tie_word_embeddings = True
    config.use_cache = True
    return config


def build_model() -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM(custom_text_config())


def build_seeded_model(seed: int) -> Qwen3_5ForCausalLM:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build_model()


def load_pretrained_body(model_id: str) -> Qwen3_5ForCausalLM:
    """Load the official text body while replacing only token I/O weights."""

    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    source_config = source.config.text_config
    custom_config = deepcopy(source_config)
    custom_config.vocab_size = VOCAB_SIZE
    custom_config.pad_token_id = PAD_TOKEN_ID
    custom_config.eos_token_id = EOS_TOKEN_ID
    custom_config.bos_token_id = None
    custom_config.tie_word_embeddings = True
    custom_config.use_cache = True
    model = Qwen3_5ForCausalLM(custom_config)

    inherited = {
        f"model.{name}": tensor
        for name, tensor in source.model.language_model.state_dict().items()
        if name != "embed_tokens.weight"
    }
    incompatible = model.load_state_dict(inherited, strict=False)
    expected_missing = {"model.embed_tokens.weight", "lm_head.weight"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected Qwen3.5 pretrained-weight mismatch: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("Qwen3.5 ARC embedding and LM head are not tied")
    del source, inherited
    model.generation_config.eos_token_id = EOS_TOKEN_ID
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    return model


def load_custom_checkpoint(path) -> Qwen3_5ForCausalLM:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_model()
    incompatible = model.load_state_dict(
        load_file(str(path), device="cpu"), strict=False
    )
    # Hugging Face-style saves omit lm_head.weight when it is tied to the input
    # embedding. The original standalone SFT checkpoints contain the duplicate,
    # while RL output directories do not. Accept only this intentional omission
    # and restore the alias explicitly; every other mismatch remains fatal.
    allowed_missing = (
        {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != allowed_missing.intersection(missing) or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.tie_weights()
    model.generation_config.eos_token_id = EOS_TOKEN_ID
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    return model


class GenerateAdapter:
    """Expose HF greedy generation through the rollout interface used by eval."""

    def __init__(self, model, batch_size: int):
        self.model = model
        self.batch_size = batch_size

    def generate(
        self,
        prompts: Sequence[torch.Tensor],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        **_: object,
    ) -> RolloutBatch:
        device = next(self.model.parameters()).device
        outputs: list[torch.Tensor] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            width = max(prompt.numel() for prompt in chunk)
            input_ids = torch.full(
                (len(chunk), width), PAD_TOKEN_ID, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, prompt in enumerate(chunk):
                length = prompt.numel()
                input_ids[row, width - length :] = prompt.to(device)
                attention_mask[row, width - length :] = 1
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                eos_token_id=EOS_TOKEN_ID,
                pad_token_id=PAD_TOKEN_ID,
                use_cache=True,
            )[:, width:]
            for row in generated:
                ids = row.tolist()
                if EOS_TOKEN_ID in ids:
                    ids = ids[: ids.index(EOS_TOKEN_ID) + 1]
                else:
                    ids = [token_id for token_id in ids if token_id != PAD_TOKEN_ID]
                outputs.append(torch.tensor(ids, dtype=torch.long, device=device))
        return RolloutBatch(
            token_ids=outputs,
            token_logprobs=[torch.empty(0, device=device) for _ in outputs],
            bucket_by_prompt=[0] * len(outputs),
        )


def greedy_eval(model, tokenizer: ArcCharacterTokenizer, examples: Sequence[ARCExample], args, device, step):
    del device
    max_new_tokens = args.eval_max_new_tokens or args.max_new_tokens
    return evaluate_policy(
        model,
        tokenizer,
        examples,
        GenerateAdapter(model, args.eval_batch_size),
        step=step,
        max_new_tokens=max_new_tokens,
    )


def main() -> None:
    trainer.DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
    trainer.build_model = build_model
    trainer.build_seeded_model = build_seeded_model
    trainer.load_pretrained_body = load_pretrained_body
    trainer.load_custom_checkpoint = load_custom_checkpoint
    trainer.greedy_eval = greedy_eval
    trainer.main()


if __name__ == "__main__":
    main()

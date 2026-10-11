"""FastAPI service for batched vLLM generation with optional LoRA adapters."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest


class GenerateRequest(BaseModel):
    """Tokenized prompts and an optional adapter for one generation request."""

    request_id: int
    input_ids: list[list[int]] = Field(min_length=1)
    lora_path: str | None = None


class GenerateResponse(BaseModel):
    request_id: int
    output_ids: list[list[int]]
    logprobs: list[list[float | None]]


app = FastAPI(title="ARC vLLM inference", version="0.1.0")


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """Create the GPU-backed engine on first use, not during module import."""

    return LLM(
        model=os.environ.get("ARC_MODEL_ID", "Qwen/Qwen2.5-0.5B"),
        enable_lora=True,
        max_lora_rank=int(os.environ.get("ARC_MAX_LORA_RANK", "32")),
        gpu_memory_utilization=float(os.environ.get("ARC_GPU_MEMORY_UTILIZATION", "0.9")),
    )


def build_lora_request(path: str | None) -> LoRARequest | None:
    if path is None:
        return None

    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(resolved):
        raise ValueError(f"LoRA adapter directory does not exist: {resolved}")

    digest = hashlib.sha256(resolved.encode("utf-8")).digest()
    adapter_id = int.from_bytes(digest[:4], "big") or 1
    return LoRARequest(
        lora_name=os.path.basename(resolved),
        lora_int_id=adapter_id,
        lora_path=resolved,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate one completion per tokenized prompt."""

    prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in req.input_ids]
    sampling_params = SamplingParams(
        temperature=float(os.environ.get("ARC_TEMPERATURE", "0")),
        top_p=float(os.environ.get("ARC_TOP_P", "1")),
        max_tokens=int(os.environ.get("ARC_MAX_TOKENS", "4096")),
        logprobs=1,
    )
    outputs = get_llm().generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=build_lora_request(req.lora_path),
    )

    output_ids: list[list[int]] = []
    logprobs: list[list[float | None]] = []
    for request_output in outputs:
        sample = request_output.outputs[0]
        output_ids.append(list(sample.token_ids))
        logprobs.append(
            [
                None
                if token_logprobs is None
                else next(iter(token_logprobs.values())).logprob
                for token_logprobs in sample.logprobs
            ]
        )

    return GenerateResponse(
        request_id=req.request_id,
        output_ids=output_ids,
        logprobs=logprobs,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

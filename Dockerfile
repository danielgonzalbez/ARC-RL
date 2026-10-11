# syntax=docker/dockerfile:1.7

# Match the CUDA 12.9.1 runtime pulled in by the pinned PyTorch and vLLM
# wheels. A development image is required to compile the fused CUDA extensions.
ARG CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
ARG UV_VERSION=0.7.16

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG MAX_JOBS=8

ENV CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=${MAX_JOBS} \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv

RUN test "${TARGETARCH}" = "amd64" || \
    (echo "This image requires linux/amd64 because the pinned PyTorch and vLLM wheels are x86_64." >&2; exit 1)

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libnuma1 \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /workspace

# Keep the expensive CUDA-extension build cached when only application code
# changes. uv.toml makes causal-conv1d and flash-linear-attention build without
# isolation after the pinned PyTorch wheel has been installed.
COPY pyproject.toml uv.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /usr/bin/python3.12 "${VIRTUAL_ENV}" \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" --extra dev -r pyproject.toml

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps --editable .

COPY . .

# This smoke test does not need access to a GPU. GPU availability is checked
# when the container is started with Docker's --gpus option.
RUN python - <<'PY'
import causal_conv1d
import fla
import torch
import vllm

assert torch.version.cuda == "12.9", torch.version.cuda
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("causal-conv1d and flash-linear-attention: imported")
PY

CMD ["bash"]

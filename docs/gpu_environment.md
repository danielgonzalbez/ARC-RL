# CUDA 12.9 and vLLM environment

## Supported target

The supported environment is Linux x86_64 with CPython 3.12, an NVIDIA driver
575.57.08 or newer, and the CUDA 12.9.1 development toolkit. Qwen3.5 requires
vLLM 0.27 or newer. The repository uses the official vLLM 0.27.1 CUDA 12.9
wheel together with PyTorch 2.13.0 CUDA 12.9. The Docker image and the Python
wheels therefore use the same CUDA minor release.

The vLLM 0.27.1 release does not publish a CUDA 12.8 wheel. Its CUDA-12
artifact is built for CUDA 12.9, while the default artifact is CUDA 13.0. CUDA
12.9 is therefore the closest official build that remains in the CUDA-12 major
series and supports Qwen3.5.

Do not downgrade to an older CUDA-12.8 vLLM release: those releases predate
Qwen3.5 support. Do not install the unqualified PyPI `vllm==0.27.1`: it selects
the default CUDA 13.0 build, which does not match this CUDA-12 environment.

## Docker image

The repository's `Dockerfile` uses the official CUDA 12.9.1 development image.
A `devel` image is required because `causal-conv1d` and
`flash-linear-attention` compile CUDA code while the image is being built:

```bash
docker build --platform linux/amd64 -t arc-qwen-rl:cu129 .
```

The pinned PyTorch and vLLM artifacts are Linux x86_64 wheels, so the resulting
image targets `linux/amd64`. On an x86_64 GPU host, run it through the NVIDIA
Container Toolkit:

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v "$PWD/minimal-arc-main:/workspace/minimal-arc-main:ro" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/logs:/workspace/logs" \
  arc-qwen-rl:cu129
```

The data, checkpoints, logs, local virtual environment, and credentials are
excluded by `.dockerignore`; mount or inject them only at runtime. The host
needs the NVIDIA driver and Container Toolkit, but it does not need its own
CUDA toolkit or Python environment.

## Native installation

For a native installation, use a CUDA 12.9 toolkit so `nvcc`, PyTorch, vLLM,
and the compiled extensions share the same CUDA release. Create the environment
and install the project from the repository root. PyTorch and the extension
build tools are declared in `pyproject.toml`. `uv.toml` makes uv install
ordinary dependencies first and then compile the two Qwen CUDA extensions
without isolation. The direct PyTorch URL avoids exposing the CUDA wheel index
as a general package source, which previously made uv resolve an old
`packaging` incompatible with vLLM's FlashInfer dependency:

```bash
CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
MAX_JOBS=8 \
UV_CACHE_DIR=/tmp/arc_rl_uv_cache \
uv venv --python 3.12

CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
MAX_JOBS=8 \
UV_CACHE_DIR=/tmp/arc_rl_uv_cache \
uv pip install --python .venv/bin/python --link-mode=copy -e '.[dev]'
```

`causal-conv1d` and `flash-linear-attention` provide fused implementations for
the Qwen3.5 convolution and gated-delta-rule layers. Their installation needs
two phases, but uv performs both within the single command due to `uv.toml`.
Without that setting, `causal-conv1d` can pull a CUDA 13 PyTorch into a
temporary build environment and fail against the CUDA 12.9 compiler.

`causal-conv1d` 1.7.0 has no matching prebuilt wheel for this Python and
PyTorch combination, so it compiles from source. Its setup currently builds a
hard-coded set of CUDA architectures and does not honor
`TORCH_CUDA_ARCH_LIST`; on the recorded A100 host that build takes roughly 10
minutes.
If the main install was interrupted after installing the other packages,
resume just the fused packages with:

```bash
CUDA_HOME=/usr/local/cuda \
PATH=/usr/local/cuda/bin:$PATH \
MAX_JOBS=8 \
UV_CACHE_DIR=/tmp/arc_rl_uv_cache \
uv pip install --python .venv/bin/python --link-mode=copy --no-build-isolation \
  causal-conv1d==1.7.0 flash-linear-attention==0.5.2
```

A CUDA *major*-version mismatch is not expected and usually means that the
CUDA 13 vLLM/PyTorch stack was installed accidentally.

### CUDA 12.8 compatibility setup

The recorded A100 host originally provided a CUDA 12.8 toolkit (`nvcc
12.8.93`). It successfully compiled the optional extensions against the CUDA
12.9 PyTorch build using CUDA minor-version compatibility. This is retained as
a historical compatibility path, not the primary supported configuration.
It may emit a CUDA 12.8 versus 12.9 warning, and PTX/JIT workloads can require
a driver with full CUDA 12.9 support. Prefer the Docker image or a native CUDA
12.9 toolkit for new environments.

## Verification

```bash
.venv/bin/python - <<'PY'
import torch, vllm
import causal_conv1d
import fla

print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("GPU:", torch.cuda.get_device_name())
print("causal-conv1d:", causal_conv1d.__version__)
print("flash-linear-attention: imported")
PY
```

The expected core versions are `torch 2.13.0+cu129`, `torchvision
0.28.0+cu129`, `torchaudio 2.11.0+cu129`, `torchcodec 0.16.0+cu129`, and `vllm
0.27.1`, all from the CUDA-12 stack. Pinning the vision/audio companions
matters even for this text-only experiment: Transformers imports their modules
while constructing its loss and media registries, and the default PyPI wheels
target CUDA 13. Training should not print warnings that `causal_conv1d_fn` or
`chunk_gated_delta_rule` is unavailable.

### Remote-workspace I/O

On hosts where `/workspace` is a remote FUSE volume, importing the large
PyTorch/vLLM environment or writing temporary vLLM exports there can block on
storage for many minutes. `arc_rl.rl_qwen35_native_vllm` accepts
`VLLM_WORK_DIR=/tmp/arc_vllm_work` to place reproducible engine exports on
local storage while keeping source checkpoints and JSONL logs in the
workspace. A local venv can likewise be installed from the shared uv cache
with `--link-mode=hardlink`; `/tmp` environments are ephemeral and must not be
treated as the durable copy of a checkpoint or log.

### Quiet console output

vLLM's JIT monitor normally enables Triton's per-configuration autotuning
prints. They are diagnostics, not failures, but can produce hundreds of lines.
The training launchers set:

```bash
TRITON_PRINT_AUTOTUNING=0
VLLM_LOGGING_LEVEL=ERROR
WANDB_SILENT=true
```

Autotuning and its disk cache remain enabled; only their console messages are
hidden. The runner's concise `eval step=...` and `train step=...` summaries and
all fatal errors remain visible.

## Training-performance choices

vLLM remains resident between optimizer updates and reloads the changed policy
weights. Prefix caching stays disabled because cached activations would belong
to the previous policy. Rollouts use vLLM; differentiable current-policy and
anchor scoring use Transformers in BF16 with gradient checkpointing. Select
the largest safe Transformers microbatch with a longest-sequence one-step
probe before starting a full run; average sequence length is not a safe memory
estimate.

Official references:

- <https://github.com/vllm-project/vllm/releases/tag/v0.27.1>
- <https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.cuda.inc.md>
- <https://download.pytorch.org/whl/cu129/torch/>

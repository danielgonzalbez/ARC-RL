# ARC Qwen RL

Research code for supervised pretraining, evaluation, and reinforcement learning of
Qwen-based policies on ARC-AGI tasks. The repository contains both the custom
character-tokenizer pipeline and the native Qwen3.5/vLLM pipeline used by the
included experiment reports.

## Repository layout

```text
src/arc_rl/       Installable training, evaluation, and inference package
scripts/data/     Dataset split and diagnostic-manifest preparation
scripts/plotting/ Reproducible plots from JSONL experiment logs
scripts/runs/     GPU run configurations
data_splits/      Versioned task-ID manifests (not the ARC dataset itself)
docs/             Environment, algorithm, and experiment documentation
tests/            Reward, structure, and advantage diagnostics
```

Generated logs, model checkpoints, downloaded ARC data, archives, and local
credentials are intentionally excluded from version control.

## Requirements

The reproducible GPU environment targets:

- Linux x86_64
- CPython 3.12
- NVIDIA driver 575.57.08 or newer
- CUDA 12.9.1 toolkit for native builds (included in the Docker image)
- a CUDA-capable GPU with enough memory for the selected model and run script
- [uv](https://docs.astral.sh/uv/)

The dependency pins in `pyproject.toml` include CUDA 12.9 PyTorch and vLLM wheels.
See [the GPU environment notes](docs/gpu_environment.md) before
installing on another platform.

## Setup

```bash
git clone <repository-url>
cd arc-qwen-rl
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
```

### Docker

The supported image uses a CUDA 12.9.1 development toolkit, matching the
pinned CUDA 12.9 PyTorch and vLLM wheels. Build it on a Linux x86_64 host, or
request that platform explicitly from another architecture:

```bash
docker build --platform linux/amd64 -t arc-qwen-rl:cu129 .
```

Run the image with the NVIDIA Container Toolkit and mount datasets,
checkpoints, and logs rather than including them in the image:

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v "$PWD/minimal-arc-main:/workspace/minimal-arc-main:ro" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  -v "$PWD/logs:/workspace/logs" \
  arc-qwen-rl:cu129
```

The host must have a sufficiently recent NVIDIA driver, Docker, and the NVIDIA
Container Toolkit. It does not need a separate local Python environment or
CUDA toolkit for the containerized workflow.

Download or clone ARC-AGI so the repository contains the expected files:

```text
minimal-arc-main/
  data/
    training/
    evaluation/
  arc-agi_training_challenges.json
  arc-agi_training_solutions.json
  arc-agi_evaluation_challenges.json
  arc-agi_evaluation_solutions.json
```

The exact required location can also be supplied with each command's
`--data-dir` option. Checkpoints are not committed; put local weights in
`checkpoints/` or pass an explicit `--checkpoint` path.

## Common commands

Show the CLI for the native Qwen3.5 RL pipeline:

```bash
.venv/bin/arc-train-rl --help
```

Run the diagnostic test suite:

```bash
.venv/bin/pytest
```

Launch the inference service (the vLLM engine loads on the first generation
request):

```bash
.venv/bin/uvicorn arc_rl.inference_api:app --host 0.0.0.0 --port 8000
```

The scripts under `scripts/runs/` contain the recorded training configurations.
They resolve paths from the repository root and can be launched from any working
directory, for example:

```bash
scripts/runs/run_qwen35_rl_diagnostic_16tasks.sh
```

## Documentation

- [RL environment and reward semantics](docs/rl_environment.md)
- [Policy-loss semantics](docs/loss.md)
- [CUDA 12.9 and vLLM setup](docs/gpu_environment.md)
- [Qwen pretraining comparison](docs/qwen_arc_pretraining_comparison.md)
- [Checkpoint and log mapping](docs/checkpoint_log_mapping.md)

The experiment reports include historical checkpoint and log paths for
reproducibility. Those large/generated artifacts are deliberately not part of
the repository.

# Estimating CUDA memory for mixed-precision Transformer training

This note estimates **peak memory on one CUDA GPU** for a dense decoder-only
Transformer trained with BF16 computation and AdamW. It is an analytical
estimate, not an exact prediction: fused kernels, PyTorch's CUDA allocator,
attention implementation, activation checkpointing, and optimizer
implementation can materially change the measured peak.

## Notation and assumptions

To avoid using `L` for both sequence length and layer count:

- \(B\): microbatch size on this GPU
- \(T\): sequence length, including prompt and completion tokens
- \(H\): hidden width
- \(N\): number of Transformer layers
- \(A\): number of attention heads
- \(F\): MLP intermediate width
- \(V\): vocabulary size
- \(P\): number of trainable parameters

Assumptions:

- One GPU, without tensor, pipeline, or ZeRO/FSDP sharding.
- Dense Transformer; no mixture-of-experts layers.
- BF16 working parameters and gradients: 2 bytes per value.
- One FP32 master-weight copy: 4 bytes per value.
- AdamW first and second moments are FP32: 4 bytes each per value.
- \(B\) is the **microbatch** size, not the effective batch size after gradient
  accumulation. Gradient accumulation does not multiply activation memory by
  the number of accumulation steps.
- The activation formulas below assume a conventional GPT-like block with
  total attention projection width approximately \(H\). GQA, unusual head
  dimensions, gated MLP variants, dropout choices, and kernel fusion change the
  constants.

## 1. Model parameters

It is safest to measure \(P\) directly:

```python
P = sum(parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
```

For architectural planning, a bias-free decoder layer has approximately:

\[
P_{layer} \approx 4H^2 + c_{mlp}HF
\]

The \(4H^2\) term represents Q, K, V, and output projections. For an ordinary
two-matrix MLP, \(c_{mlp}=2\); for a gated MLP such as SwiGLU,
\(c_{mlp}=3\). Embeddings and the output head add approximately \(VH\) if tied
or \(2VH\) if untied. Norm and bias parameters are relatively small.

Thus:

\[
P \approx N(4H^2+c_{mlp}HF) + c_{vocab}VH.
\]

Here \(c_{vocab}=1\) for tied input/output embeddings and \(c_{vocab}=2\) for
untied embeddings.

A common GPT approximation is \(P\approx12NH^2\), but using the measured
parameter count avoids errors caused by gated MLPs, GQA, tied embeddings, or a
nonstandard head dimension.

## 2. Persistent training state

With the precision arrangement proposed here:

| State | Bytes per trainable parameter |
|---|---:|
| BF16 working weight | 2 |
| FP32 master weight | 4 |
| FP32 Adam first moment | 4 |
| FP32 Adam second moment | 4 |
| BF16 gradient | 2 |
| **Total once gradients exist** | **16** |

Before gradients are materialized, the persistent state is approximately
\(14P\) bytes. Once all gradients exist, it is approximately:

\[
M_{state+grad}=16P\quad\text{bytes}.
\]

Standard PyTorch autocast often keeps the model parameters themselves in FP32
and casts operands inside eligible operations instead of storing a separate
persistent BF16 model. Its accounting can therefore differ by implementation;
the table describes the explicit BF16-working-model plus FP32-master-model
design discussed here.

This matches NVIDIA's optimizer-held-master-weight arrangement: the model uses
low-precision weights while the optimizer retains FP32 master weights and
state. See [NVIDIA Transformer Engine's mixed-precision
description](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/introduction/introduction.html).

Two qualifications matter:

1. Casting the complete BF16 gradient tree to FP32 at once may temporarily keep
   both copies alive, adding as much as \(4P\) bytes. A fused or parameter-wise
   optimizer can avoid most of this overlap.
2. AdamW needs temporary tensors during the update. A fused AdamW kernel uses
   much less temporary memory than an implementation that creates full-sized
   intermediate tensor lists. Represent this unknown as \(M_{opt\_tmp}\).

## 3. Activations saved for backward

The simple estimate \(2NBTH\) bytes is **not** correct when all activations are
saved. It counts only one BF16 hidden-state tensor per layer. Backpropagation
normally needs several residual, normalization, Q/K/V, attention, and MLP
tensors.

For a GPT-like Transformer with ordinary attention and no activation
recomputation, Korthikanti et al. derive the following useful approximation:

\[
M_{act,ordinary}
\approx
N\left(34BTH + 5BAT^2\right)\quad\text{bytes}.
\]

The \(34BTH\) component covers the non-quadratic saved activations. The
\(5BAT^2\) component comes from attention scores, softmax, dropout, and related
attention tensors. The coefficients already represent a mixed-precision byte
estimate; they should not be multiplied by another 2 bytes. The derivation is
given in [Reducing Activation Recomputation in Large Transformer
Models](https://arxiv.org/abs/2205.05198).

The paper obtains the coefficients by listing the tensors retained for
backward in one classic GPT layer:

| Block | Saved tensors | Bytes per layer |
|---|---|---:|
| Attention, linear-sized | shared QKV input; Q and K; V; output-projection input; dropout mask | \(11BTH\) |
| Attention, quadratic | softmax output; softmax-dropout mask; dropout output | \(5BAT^2\) |
| MLP with width \(4H\) | inputs of both linear layers; GeLU input; dropout mask | \(19BTH\) |
| Two LayerNorms | the two normalization inputs | \(4BTH\) |

Therefore one layer requires approximately
\((11+19+4)BTH+5BAT^2=34BTH+5BAT^2\) bytes. The count assumes two-byte
activations and one-byte dropout masks. A SwiGLU MLP, zero dropout, fused
normalization, GQA, or a nonstandard projection width changes these constants.

### FlashAttention or memory-efficient SDPA

FlashAttention does not materialize the full \(T\times T\) attention matrix in
GPU memory, changing attention storage from quadratic to linear in sequence
length. See [FlashAttention](https://arxiv.org/abs/2205.14135).

A useful planning approximation, without full layer checkpointing, is:

\[
M_{act,flash}\approx C_{flash}NBTH,
\]

where \(C_{flash}\) is implementation-dependent. Using \(C_{flash}\approx34\)
bytes is a conservative first estimate for a GPT-like block because it removes
the quadratic term from the formula above while retaining the other saved
activations. It is not a universal constant.

### Full activation checkpointing

If each layer boundary is checkpointed and the interior of each layer is
recomputed during backward, only approximately one BF16 hidden state per layer
must remain resident:

\[
M_{checkpoints}\approx2NBTH\quad\text{bytes}.
\]

During backward, one recomputed layer's temporary activations are also live.
With a FlashAttention-like implementation, a conservative peak estimate is:

\[
M_{act,full\_ckpt\_peak}
\approx 2NBTH + C_{flash}BTH.
\]

The first term is stored checkpoints; the second is one active layer being
recomputed. Full recomputation usually costs additional compute; the cited
Megatron analysis reports a 30--40% execution-time overhead in its tested
configurations.

## 4. Logits and loss

The output logits have shape \([B,T,V]\). If fully materialized in BF16:

\[
M_{logits}=2BTV\quad\text{bytes}.
\]

If cross-entropy or log-softmax creates an FP32 copy, another \(4BTV\) bytes may
temporarily exist. A naive implementation can therefore consume 6 or more
bytes per logit, especially if it separately stores probabilities or
log-probabilities.

For a large vocabulary this can be a major peak. Fused or chunked
cross-entropy avoids retaining several full \([B,T,V]\) tensors. For a tiny
vocabulary it is usually negligible.

For example, with \(B=24\), \(T=2048\), and \(V=24\), a BF16 logits tensor is
only about 2.25 MiB. Keeping that tensor plus one FP32 copy is about 6.75 MiB.
Thus logits are small for a 24-token ARC vocabulary, although they would be a
major term for a conventional 50,000-token language-model vocabulary.

## 5. What temporary memory means

\(M_{temporary}\), abbreviated as \(M_{tmp}\), groups allocations that are
needed only while a CUDA operation or training phase is executing and whose
size cannot be inferred from \(B,T,H,N\) alone. Examples include:

- cuBLAS/cuBLASLt GEMM workspaces and attention-kernel workspaces;
- temporary forward and backward outputs that are not saved activations;
- reduction buffers for norms, softmax, and gradient clipping;
- an FP32 gradient copy if all BF16 gradients are cast at once;
- AdamW update intermediates, especially with a non-fused or `foreach`
  optimizer;
- distributed gradient buckets when DDP is used;
- buffers retained by compilation or CUDA Graph capture.

Allocator fragmentation and PyTorch's reserved-but-currently-unused CUDA
blocks are related capacity overheads, but are not live mathematical tensors.
They are why measured reserved memory can exceed the analytical total.

There is no reliable architecture-only closed form for this term. Minimize it
with FlashAttention, fused/chunked loss, fused AdamW, and parameter-wise or
fused gradient conversion; then measure it with CUDA peak-memory statistics.

## 6. Conservative peak formulas

Memory is not maximized at exactly the same instant for every category:
activations are progressively freed during backward, gradients progressively
appear, and the optimizer normally runs after the activation graph has been
released. Therefore, the true peak is the maximum of several phases rather
than the sum of every possible allocation.

A safe planning upper bound is nevertheless:

\[
\boxed{
M_{peak}\lesssim
16P + M_{act} + M_{logits,tmp} + M_{kernel\_tmp}
}
\]

where all quantities are bytes. More explicitly, estimate the phases as:

\[
M_{forward/backward}
\lesssim16P + M_{act} + M_{logits,tmp}+M_{backward\_tmp},
\]

\[
M_{optimizer}
\lesssim16P+M_{opt\_tmp},
\]

\[
M_{peak}=\max(M_{forward/backward},M_{optimizer}).
\]

This produces the following useful cases:

### Ordinary attention, no checkpointing

\[
\boxed{
M_{peak}\lesssim
16P + N(34BTH+5BAT^2)+M_{logits,tmp}+M_{tmp}
}
\]

### FlashAttention, no checkpointing

\[
\boxed{
M_{peak}\lesssim
16P+C_{flash}NBTH+M_{logits,tmp}+M_{tmp}
}
\]

with \(C_{flash}\approx34\) as an initial conservative estimate.

### FlashAttention with full layer checkpointing

\[
\boxed{
M_{peak}\lesssim
16P+2NBTH+C_{flash}BTH+M_{logits,tmp}+M_{tmp}
}
\]

## 7. CUDA allocator and practical headroom

The equations estimate live tensors, not total VRAM required for a reliable
run. CUDA kernels need workspaces, PyTorch reserves caching-allocator blocks,
and variable sequence lengths can cause fragmentation. Compilation systems may
also retain additional buffers.

After calculating the analytical estimate, reserve at least 10--20% VRAM
headroom. More may be needed for dynamic shapes or an unfused optimizer. Check
both:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

`max_memory_allocated` is closest to live tensor usage;
`max_memory_reserved` is closer to the CUDA memory capacity the process needs.

## 8. RL rollout memory is a separate phase

Autoregressive rollout generation typically uses a KV cache but does not store
a backward graph. For standard multi-head attention with BF16 K and V, its
rough size is:

\[
M_{KV}\approx4NBTH\quad\text{bytes}.
\]

For grouped-query attention with \(A_{kv}\) KV heads and \(A\) query heads:

\[
M_{KV}\approx4NBTH\frac{A_{kv}}{A}.
\]

If the rollout generator and training graph are released between phases, do
not add the complete rollout and training peaks together; take their maximum.
If both policy and reference models, retained KV caches, or rollout logits stay
on the GPU during the update, those allocations must be added explicitly.

## 9. What was correct in the proposed reasoning

The listed categories were correct:

- BF16 working weights plus FP32 master weights
- FP32 Adam first and second moments
- BF16 gradients
- BF16 activations

The missing details were:

- Activations contain many saved tensors per layer, not just one \(BTH\)
  tensor.
- Ordinary attention adds a potentially dominant \(BAT^2\) term.
- Logits can add a \(BTV\) term.
- Optimizer and CUDA-kernel temporaries can determine the peak.
- Peak memory depends on the microbatch size, while gradient accumulation can
  increase the effective batch size without multiplying activation memory.
- FlashAttention and activation checkpointing change the formula
  substantially.

The formulas should therefore be treated as capacity-planning estimates and
validated with a short CUDA run using the exact kernels, optimizer, and tensor
shapes intended for training.

## 10. Small planning calculator

This calculator intentionally returns the analytical part only. Add optimizer
temporaries, kernel workspaces, and 10--20% operational headroom afterward.

```python
def estimated_training_gib(
    *, P, B, T, H, N, A, V,
    attention="flash", checkpoint=False,
    flash_coefficient=34,
    fp32_loss_copy=True,
):
    # BF16 weights + FP32 master + two FP32 Adam moments + BF16 grads.
    model_state = 16 * P

    if checkpoint:
        # Stored layer-boundary checkpoints plus one recomputed layer.
        activations = 2 * N * B * T * H + flash_coefficient * B * T * H
    elif attention == "flash":
        activations = flash_coefficient * N * B * T * H
    elif attention == "ordinary":
        activations = N * (34 * B * T * H + 5 * B * A * T * T)
    else:
        raise ValueError("attention must be 'flash' or 'ordinary'")

    # One BF16 logits tensor and, optionally, one FP32 loss/log-softmax copy.
    logits = (2 + (4 if fp32_loss_copy else 0)) * B * T * V

    analytical_bytes = model_state + activations + logits
    return analytical_bytes / 2**30
```

## 11. Capacity combinations for the CUDA ARC model plus vLLM

The CUDA model configuration has:

- \(P=2{,}631{,}808\) parameters
- \(H=128\)
- \(N=8\)
- \(A=2\) query heads and 2 KV heads
- head dimension 128
- \(F=512\)
- \(V=24\)

Its persistent mixed-precision training state is only
\(16P\approx40.2\) MiB. Activations, CUDA/compiler overhead, and vLLM therefore
matter much more than parameters or logits.

The model is not exactly the classic GPT block assumed by the coefficient 34:
its total Q/K/V projection width and its gated MLP require more retained values.
For planning, this section uses the deliberately conservative FlashAttention
estimate:

\[
M_{train,analytical}\approx16P+50NBTH+6BTV.
\]

The \(6BTV\) term assumes BF16 logits plus one FP32 loss copy. With \(V=24\),
it remains very small.

### Sharing a GPU with the current vLLM configuration

The current vLLM configuration uses:

```python
max_model_len=2048
gpu_memory_utilization=0.1
max_num_seqs=96
```

If total GPU capacity is \(G\), vLLM is allowed approximately \(0.1G\) for its
model executor. vLLM uses the remaining part of that allowance after weights
and executor overhead for KV cache. The exact KV allocation is determined by
vLLM's startup profiling.

For the tables below, another 15% of total VRAM is intentionally left unused
for CUDA contexts, compilation, workspaces, allocator fragmentation, and
variation in tensor shapes. The analytical training allowance is therefore:

\[
M_{train,allowance}=0.75G.
\]

This is a planning convention, not a guarantee.

### Analytical training-microbatch ceilings

The entries below are the maximum \(B\) obtained from the conservative formula,
before empirical validation. They are ceilings, not suggested starting values.

| Total padded length \(T\) | 16-GiB GPU | 24-GiB GPU | 48-GiB GPU |
|---:|---:|---:|---:|
| 512 | 488 | 733 | 1468 |
| 1024 | 244 | 366 | 734 |
| 2048 | 122 | 183 | 367 |
| 4096 | 61 | 91 | 183 |
| 8192 | 30 | 45 | 91 |
| 16384 | 15 | 22 | 45 |

Only rows through \(T=2048\) apply while vLLM retains its current
`max_model_len`. The longer rows show what training could use if the context
configuration were increased.

Prudent initial training settings are roughly half these analytical ceilings:

| Total padded length \(T\) | 16-GiB start | 24-GiB start | 48-GiB start |
|---:|---:|---:|---:|
| 512 | 256 | 384 | 768 |
| 1024 | 128 | 192 | 384 |
| 2048 | 64 | 96 | 192 |
| 4096 | 32 | 48 | 96 |
| 8192 | 16 | 24 | 48 |
| 16384 | 8 | 12 | 24 |

Very large microbatches may be slower or less useful even when they fit. Tensor
Core utilization will saturate well before these memory ceilings for such a
small model. Increase from a smaller value while measuring throughput and peak
reserved memory.

These tables require the CUDA run to select FlashAttention or the
memory-efficient SDPA kernel. If attention falls back to the ordinary backend,
the quadratic term must be restored:

\[
M_{train,ordinary}\approx16P+50NBTH+5NBAT^2+6BTV.
\]

The corresponding analytical ceilings are much lower:

| Total padded length \(T\) | 16-GiB GPU | 24-GiB GPU | 48-GiB GPU |
|---:|---:|---:|---:|
| 512 | 271 | 408 | 817 |
| 1024 | 94 | 141 | 282 |
| 2048 | 29 | 43 | 87 |
| 4096 | 8 | 12 | 24 |
| 8192 | 2 | 3 | 6 |
| 16384 | does not fit | does not fit | 1 |

Do not use the FlashAttention starting table until the selected CUDA attention
backend has been verified. The current training microbatch \(B=16\) at
\(T=2048\) has an analytical footprint of roughly 1.64 GiB with FlashAttention
or 6.60 GiB with ordinary attention, including the 40.2-MiB model state but not
CUDA/compiler temporaries.

### vLLM rollout concurrency

For BF16 KV cache, this model uses approximately:

\[
4NA_{kv}d_{head}=4\cdot8\cdot2\cdot128=8192
\]

bytes of KV cache per cached token. At a full 2048-token context, that is about
16 MiB per active sequence.

If the full 10% vLLM allowance were KV cache, the upper limits would be about
102, 154, and 307 full-length sequences on 16-, 24-, and 48-GiB GPUs. Some of
the pool is needed for model/executor overhead, so safer initial concurrency
limits are:

| GPU | vLLM allowance | Suggested `max_num_seqs` at \(T=2048\) |
|---:|---:|---:|
| 16 GiB | 1.6 GiB | 64 |
| 24 GiB | 2.4 GiB | 96 |
| 48 GiB | 4.8 GiB | 192 |

At an average cached length of 1024 rather than 2048, approximately twice as
many sequences fit in the same KV cache. `max_num_batched_tokens` separately
limits tokens scheduled in one iteration; it does not enlarge the KV cache.

The total rollout request batch may exceed `max_num_seqs`: vLLM will schedule
the requests in waves. For 256 total rollouts, useful problem/rollout groupings
include:

| Problems | Rollouts per problem | Total rollout requests |
|---:|---:|---:|
| 256 | 1 | 256 |
| 128 | 2 | 256 |
| 64 | 4 | 256 |
| 32 | 8 | 256 |
| 16 | 16 | 256 |

For within-problem RL baselines, 64 problems × 4 rollouts or 32 problems × 8
rollouts is usually more useful than 256 problems × 1 rollout. Their vLLM KV
memory is the same when all 256 sequences have the same lengths; only the
statistical grouping differs.

### Current-code interpretation

The current CUDA defaults are:

```python
problems_per_step = 4
rollouts_per_problem = 8      # 32 rollout requests / statistical batch
training_microbatch_size = 16 # sequences in one forward/backward
max_new_tokens = 512
```

Consequently, \(B=16\), not 32, should be inserted into the
training-activation formula. Reward statistics are computed over all 32
requests, while two 16-sequence microbatches accumulate gradients with exact
valid-token weighting. Raising `training_microbatch_size` changes activation
memory; changing the problem or rollout counts changes vLLM request volume and
CPU/GPU rollout-result storage.

vLLM's length constraint applies to prompt plus completion:

\[
T_{prompt}+T_{generated}\le2048.
\]

With the default `max_new_tokens=512`, the full generation allowance is
possible for prompts of at most 1536 tokens. The loader also enforces a
2048-token prompt-plus-golden limit. Training microbatches are formed from the
rollout order and padded only to the longest full sequence in that microbatch;
the outer statistical batch is never materialized as one backward graph.

On the target RTX 4090, the tested 4-problem x 8-rollout configuration with a
16-sequence training microbatch, 2048-token vLLM context, 128-token rollout
allowance, 10% vLLM allocation, and `max_num_seqs=96` peaked at 8.988 GiB CUDA
reserved memory. This leaves substantial operational headroom within 24 GiB.

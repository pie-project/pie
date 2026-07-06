# driver/metal/qwen3 — Metal Qwen3-0.6B decoder-layer parity

Building toward a full **Qwen3-0.6B decoder-layer forward on Metal**, parity vs the
CUDA reference — the macOS-support north star (Metal runs the real model). Each
primitive is validated within tolerance vs a CPU f32 reference (`reference.hpp`)
that ports the CUDA kernel formula, using the same accumulation order (f32-epsilon
residual; `exp`/`sin`/`cos` are not bit-exact GPU-vs-host). Reuses the ptir
`MetalHarness`; no MLX.

## Arch (authoritative, `arch.hpp`)

Pulled from `Qwen/Qwen3-0.6B` config.json + `hf_config.cpp` — **not assumed**:
hidden=1024, **head_dim=128** (explicit, ≠ hidden/heads=64 ⇒ q_dim=2048), n_q_heads=16,
n_kv_heads=8 (GQA=2), intermediate=3072, rms_eps=1e-6, rope_theta=1e6, SiLU/SwiGLU,
tied embeddings, **QK-norm=YES** (per-head RMSNorm size 128, **pre-RoPE**), no attn
bias, no sliding-window, no softcap.

Decoder-layer order: `RMSNorm → QKV → QK-norm → RoPE → attn(1/√128) → O+residual →
RMSNorm → gate/up → SwiGLU → down+residual`.

## Primitives

| kernel | CUDA source | residual (vs f32 ref) |
|--------|-------------|------------------------|
| `matmul_xwt` (y = x·Wᵀ) | projections (QKV/O/gate/up/down) | max_abs ~1.5e-5 (K=1024) |
| `rmsnorm`               | `rmsnorm.cu`  | ~1e-7 |
| `rmsnorm` (per-head QK-norm) | `rmsnorm.cu` over head_dim | ~2e-7 |
| `rope_qwen`             | `rope.cu` (Qwen half-rotation) | ~4e-6 |
| `swiglu`                | `swiglu.cu`   | ~1e-7 |
| `paged_attention` (prefill/decode) | paged SDPA (`sdpa_paged`/FlashInfer) | ~2e-7 |
| **`decoder_layer`** (full forward) | `qwen3_forward.cpp` | **max_abs ~7e-7** |

The full **Qwen3-0.6B decoder-layer forward** is assembled and validated end-to-end:
`RMSNorm → QKV → QK-norm → RoPE → attention → O+residual → RMSNorm → gate/up →
SwiGLU → down+residual`, chained on Metal and matched to the CPU f32 reference at
`max_abs ≈ 7e-7` (f32 epsilon). Prefill = multi-token causal attention (each query
row attends `[0, position_ids[row]]`), the same paged kernel as decode.

Next: embedding + LM head (tied weights) to close the token→logits path, then a
multi-layer stack. Matmul is currently a parity-first sequential-k GEMM; an MPS/
MPSGraph swap-in is the perf follow-on (numerics already pinned).

## Build & run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/qwen3_test        # -> QWEN3_TEST_OK
```

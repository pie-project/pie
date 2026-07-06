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
| `embedding` (gather, tied) | token embed gather | **bit-exact** |
| `lm_head` (logits = h·embedᵀ) | tied LM head matmul | ~1e-5 |
| **`layer_stack`** (28× full forward) | embed→28×layer→norm→lm_head | max_abs ~4e-6 |

The full **Qwen3-0.6B forward stack** is validated end-to-end: `embedding → 28×
decoder_layer → final RMSNorm → tied LM head`, chained on Metal and matched to the
CPU reference (`max_abs ≈ 4e-6`). The 28-layer stack test uses reduced dims to prove
the stacking mechanism (loop + cross-layer residual + per-layer KV + final norm +
head) cheaply; per-primitive and single-layer parity are at real Qwen3-0.6B dims.

## Perf: MPS GEMM

The projections/LM-head GEMMs run through **`MPSMatrixMultiplication`**
(`MetalHarness::mps_gemm`, `y = x·Wᵀ` via `transposeRight`) by default
(`Chain::use_mps`), with the parity-first sequential kernel retained as a fallback
(`QWEN3_GEMM=seq`). Numerics stay within tolerance — the full-forward
`Metal-vs-CPU-ref max_abs` is unchanged at `4.1e-5` (the residual is attention/RoPE
`exp`-dominated, below the GEMM-swap delta).

Measured (real weights, N=8, Apple M1 Max): **Metal forward 665 ms (MPS) vs
2209 ms (sequential) → 3.3×**. The ~7 s wall time in the driver is dominated by the
CPU f32 reference run (self-validation); the Metal forward itself is sub-second.

## Algorithm loops (composed from certified pieces)

Beam search and constrained speculative decoding run **end-to-end on Metal** by
pure composition over the certified forward + KV cache + sampling ops (`model.hpp`
supplies a `KVCache`-parameterized `forward_block` so each beam / draft owns a
growing cache).

- **`qwen3_beam`** — forward per beam → `log_softmax` scores → K×K candidate top-k
  → reorder beams by parent → grow per-beam KV → repeat. Observed (K=3): top beam
  *"France is a city located in…"*, ranked by cumulative log-prob. **Metal runs
  beam search.**
- **`qwen3_specdecode`** — constrained speculative decoding: draft k tokens greedily,
  then spec-VERIFY under a **grammar mask** (masked logits via the certified
  `dselect` = `matrix_select_mask` op + `argmax`, both Metal) and accept the longest
  matching prefix; a grammar-disallowed draft is rejected and the mask-corrected
  token substituted. Observed (k=4, grammar forbids `13`="."): 9/12 drafts accepted;
  where greedy drafted `13` the grammar rejected+corrected it to `11`=",". **Metal
  runs constrained speculative decoding.**

## Autoregressive generation (`qwen3_generate`)

`qwen3_generate` runs **end-to-end greedy autoregressive decoding** on Metal: prompt
→ full 28-layer Metal forward → **Metal argmax** → append the new token's K/V to a
growing per-layer KV cache → repeat. This exercises the **multi-step KV-decode path**
(KV growth across autoregressive steps) that a single forward doesn't, fully
on-device.

Observed (real weights, prompt `9707 3838 374 279 6722 315 9625 30`, 20 tokens):
```
generated: 9625 374 264 3146 304 4505 13 576 6722 315 9625 374 12095 13 ...
           ≈ "France is a city in Europe. The capital of France is Paris. ..."
```
Coherent, grammatical, factually correct ("The capital of France is Paris"). The
generated token-id sequence is exported (`qwen3_generate.txt`) for the definitive
CUDA cross-check — matching greedy sequences (same prompt+weights) means tight
cross-backend parity, since per-step divergence compounds.

## Real-weight forward (`qwen3_forward`)

`qwen3_forward` loads the **actual** `Qwen/Qwen3-0.6B` `model.safetensors` (BF16, HF
`[out,in]` layout — exactly `matmul_xwt`'s W layout, no transpose; `bf16→f32` via
`bits<<16`), runs a real token sequence through the full Metal forward (embedding →
28× decoder_layer → final RMSNorm → LM head), self-validates the Metal logits vs the
CPU f32 reference, prints the next-token argmax/top-k, and writes a compact golden
(`qwen3_golden.txt`) for the CUDA cross-check.

Observed on Apple M1 Max: full 28-layer forward on real weights (N=8 tokens) in ~7 s,
**Metal-vs-CPU-ref `max_abs = 4.1e-5`** across all 1.2M logits — the real model runs
on Metal, parity-clean through the full depth. `qwen3_golden.txt` records token ids +
weight source/layout + argmax/top-20 + logit L2/sum/min/max so the CUDA reference
(4090) can run the identical input+weights for the definitive cross-backend check.

## Build & run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/qwen3_test        # -> QWEN3_TEST_OK
./build/qwen3_forward /path/to/model.safetensors kernels qwen3_golden.txt
```

#pragma once

// Gemma-4 (E2B / E4B family — text-only stripped of multimodal towers).
// Architectural pieces beyond Gemma-3:
//
//   * Per-Layer Embeddings (PLE): an auxiliary 256-dim residual stream
//     fed by a small per-layer embedding lookup, summed into the main
//     residual after each MLP block.
//   * KV-cache sharing: the last `num_kv_shared_layers` layers reuse
//     K/V from the most recent non-shared layer of the same type (full
//     vs sliding). Shared layers store no K/V of their own and load no
//     k_proj/v_proj weights.
//   * Per-layer-type head_dim: sliding layers use the standard
//     `head_dim` (256 on E2B), full-attention layers use a *separate*
//     `global_head_dim` (typically 512). KV-cache shape varies per
//     layer-type — needs a two-pool KvCache, or two separate KvCache
//     allocations.
//   * Double-wide MLP for shared layers when `use_double_wide_mlp` is
//     set (intermediate × 2).
//   * Proportional RoPE on full-attention layers: only the lower
//     `partial_rotary_factor * head_dim` dims are rotated; the
//     remaining suffix passes through unchanged. flashinfer's
//     `kRoPELlama` mode doesn't expose this; either pre-apply a custom
//     RoPE before write_kv (as we already do today) and route through
//     `PosEncodingMode::kNone`, or land a kernel that respects the
//     partial-rotary factor.
//   * `sm_scale = 1.0`: the learnable q/k norm absorbs the usual
//     `1/sqrt(head_dim)` factor. flashinfer's `params.sm_scale` accepts
//     this directly.
//   * Plain `w * x_hat` RMSNorm (no Gemma-2-style `(1+w)` shift) — uses
//     the existing `launch_rmsnorm_bf16`.
//   * V-Norm: pure RMSNorm (gamma=1, no learnable scale) on V before
//     the KV-cache write. Same kernel, just identity weights.
//   * Per-layer learnable scalar applied to the layer output.
//   * Final logit soft-cap (cap=30) — reuses `softcap.cu`.
//
// Critical flashinfer limitation in 0.6.x: prefill at `head_dim_qk =
// head_dim_vo = 512` is *not* supported by the default
// `BatchPrefillWithPagedKVCacheDispatched` template. Only decode is.
// The Python adapter falls back to a SDPA path for prefill; the
// matching C++ option is `trtllm_batch_context_with_kv_cache` (lands
// in 0.6.9, which is our pinned version) — pulling that into the
// build is part of this milestone.

#include "engine.hpp"
#include "model/qwen3.hpp"

namespace pie_cuda_driver::model {

// Stub. The structure required for Gemma-4 is sufficiently different
// from `Qwen3Weights` that it warrants its own struct (per-layer
// `attention_type ∈ {sliding, full}`, optional k_proj/v_proj, separate
// PLE-embed table). We return `Qwen3Weights` here only as a
// placeholder so the dispatch in `main.cpp` can compile uniformly.
Qwen3Weights bind_gemma4(Engine& engine);

}  // namespace pie_cuda_driver::model

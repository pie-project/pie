#pragma once

// Gated DeltaNet linear-attention, ported from driver/cuda's
// gated_delta_net.cu + causal_conv1d.cu (HF `torch_recurrent_gated_delta_rule`).
// Used by qwen3.6's `linear_attention` layers.
//
// Pure MLX: the decode step is single-token tensor ops; no custom Metal kernel
// (MLX runs the matmuls/reductions on GPU). Each sub-step is validated against
// a naive reference to ~1e-7.
//
// Pipeline per linear-attn layer (this op owns the middle; the graph owns the
// in_proj / out_proj linears):
//   mixed_qkv -> causal depthwise conv1d(+silu) -> split q/k/v -> L2norm(q,k)
//   -> g/beta gating -> recurrent gated-delta -> RMSNormGated(out, z) -> [N,V_dim]

#include <optional>
#include <vector>

#include "ops/tensor.hpp"

namespace pie_metal_driver {
class LinearStateCache;  // driver/metal/src/linear_state_cache.hpp (delta-owned)
}

namespace pie_metal_driver::ops {

struct GdnParams {
    int n_heads_k = 0;   // K_h: query/key heads
    int n_heads_v = 0;   // V_h: value heads (V_h % K_h == 0; GQA repeat of q/k)
    int head_k = 0;      // K_d: key head dim
    int head_v = 0;      // V_d: value head dim
    int conv_kernel = 0; // conv_K: depthwise conv width
    float norm_eps = 1e-6f;
};

// Per-request linear-attention state. Layouts match delta's LinearStateCache:
//   conv_state      : [R, conv_K, conv_dim]                  (conv_dim below)
//   recurrent_state : [R, V_h, K_d, V_d] fp32
// conv_dim = 2*K_h*K_d + V_h*V_d.
struct GdnState {
    Tensor conv_state;
    Tensor recurrent_state;
};

struct GdnResult {
    Tensor output;   // [R, V_dim]  (V_dim = V_h * V_d), post RMSNormGated
    GdnState state;  // updated conv + recurrent state to write back
};

// Single-token-per-request decode step. Cache-agnostic: state in -> state out,
// so it can be unit-tested and wired to any state container. R = number of
// requests, each contributing exactly one new token.
//   mixed_qkv   : [R, conv_dim]   (in_proj qkv output, pre-conv)
//   z           : [R, V_dim]      (gate for RMSNormGated)
//   a, b        : [R, V_h]        (in_proj gating params)
//   conv_w      : [conv_dim, conv_K]   conv_b: optional [conv_dim]
//   A_log       : [V_h]           dt_bias: [V_h]   (per-head, layer-shared)
//   gate_norm_w : [V_d]           (RMSNormGated weight, per value-head dim)
GdnResult gated_delta_net_decode(const Tensor& mixed_qkv,
                                 const Tensor& z,
                                 const Tensor& a,
                                 const Tensor& b,
                                 const Tensor& conv_w,
                                 const std::optional<Tensor>& conv_b,
                                 const Tensor& A_log,
                                 const Tensor& dt_bias,
                                 const Tensor& gate_norm_w,
                                 const GdnState& state_in,
                                 const GdnParams& params);

// Cache-bound decode entry: gather this layer's per-request state from delta's
// LinearStateCache, run the step, and scatter the updated state back. Returns
// the layer output [R, V_dim]. `lin_layer` is the ordinal among linear layers;
// `slot_ids` [R] selects the active request slots.
Tensor gated_delta_net(const Tensor& mixed_qkv,
                       const Tensor& z,
                       const Tensor& a,
                       const Tensor& b,
                       const Tensor& conv_w,
                       const std::optional<Tensor>& conv_b,
                       const Tensor& A_log,
                       const Tensor& dt_bias,
                       const Tensor& gate_norm_w,
                       LinearStateCache& cache,
                       int lin_layer,
                       const Tensor& slot_ids,
                       const GdnParams& params);

// Prefill / sequential scan for a SINGLE request of T tokens (T>=1). Causal
// conv uses `state_in.conv_state` as left context; the recurrence scans the T
// tokens carrying `state_in.recurrent_state`. Cache-agnostic (state in ->
// state out) so it can be unit-tested. Returns {output [T, V_dim], state}.
//   mixed_qkv : [T, conv_dim]   z : [T, V_dim]   a, b : [T, V_h]
GdnResult gated_delta_net_prefill(const Tensor& mixed_qkv,
                                  const Tensor& z,
                                  const Tensor& a,
                                  const Tensor& b,
                                  const Tensor& conv_w,
                                  const std::optional<Tensor>& conv_b,
                                  const Tensor& A_log,
                                  const Tensor& dt_bias,
                                  const Tensor& gate_norm_w,
                                  const GdnState& state_in,
                                  const GdnParams& params);

// ── Speculative-decode support: frozen-verify + commit-advance (GDN rewind) ──
// The recurrent state is an irreversible fold, so spec-decode must NOT advance
// the persisted state over uncommitted drafts. This variant scans the T-token
// window like `gated_delta_net_prefill` but (a) persists NOTHING, and (b) emits
// the per-token intermediate state CHECKPOINTS. The caller then commit-advances
// by selecting the checkpoint at the accepted-prefix boundary (S[commit_len]) —
// device gather-by-value (or a static slice when commit_len is host-known) — and
// writes only THAT back. See wiki `ptir-metal-gdn-rewind-design`.
//   recur_ckpts : [T+1, V_h, K_d, V_d] fp32 — S_0 (pre-window) .. S_T (full)
//   conv_ckpts  : [T+1, conv_K, conv_dim]   — C_0 .. C_T (conv window per boundary)
// Checkpoint m is bit-identical to the state a clean prefill of the first m
// window tokens would produce (same recurrent_step op sequence) — lossless.
struct GdnCheckpointResult {
    Tensor output;       // [T, V_dim]   (correct draft-position outputs)
    Tensor recur_ckpts;  // [T+1, V_h, K_d, V_d] fp32
    Tensor conv_ckpts;   // [T+1, conv_K, conv_dim]
};
GdnCheckpointResult gated_delta_net_prefill_checkpointed(
    const Tensor& mixed_qkv,
    const Tensor& z,
    const Tensor& a,
    const Tensor& b,
    const Tensor& conv_w,
    const std::optional<Tensor>& conv_b,
    const Tensor& A_log,
    const Tensor& dt_bias,
    const Tensor& gate_norm_w,
    const GdnState& state_in,
    const GdnParams& params);

// Cache-bound variable-length entry for a ragged batch (mixed prefill/decode).
// `qo_indptr` [R+1] gives each request's token span into the packed
// `mixed_qkv`/`z`/`a`/`b` (token-major, requests contiguous); `slot_ids` [R]
// selects each request's cache slot. Per request: gather state -> sequential
// scan -> scatter state back. Returns the packed layer output [N_total, V_dim]
// in qo_indptr token order. Decode (all T==1) is the R-token special case.
Tensor gated_delta_net_varlen(const Tensor& mixed_qkv,
                              const Tensor& z,
                              const Tensor& a,
                              const Tensor& b,
                              const Tensor& conv_w,
                              const std::optional<Tensor>& conv_b,
                              const Tensor& A_log,
                              const Tensor& dt_bias,
                              const Tensor& gate_norm_w,
                              LinearStateCache& cache,
                              int lin_layer,
                              const Tensor& slot_ids,
                              const std::vector<int>& qo_indptr,
                              const GdnParams& params);

}  // namespace pie_metal_driver::ops

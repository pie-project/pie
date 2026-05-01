#pragma once

// Qwen3.5 forward driver — hybrid linear-attention + full-attention.
// Produces last-token logits in `ws.logits` (or all-token logits when
// the workspace is full-width).

#include <cstdint>

#include "attention_workspace.hpp"
#include "device_buffer.hpp"
#include "kv_cache.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_forward.hpp"  // for Qwen3Workspace (reused)
#include "ops/gemm.hpp"
#include "qwen3_5_state_cache.hpp"

namespace pie_cuda_driver::model {

// Per-linear-attention-layer extra workspace. Allocated once and
// reused across layers (linear layers are processed sequentially).
struct Qwen3_5LinearAttnWorkspace {
    DeviceBuffer<std::uint16_t> mixed_qkv;     // [N, conv_dim] bf16
    DeviceBuffer<std::uint16_t> z;             // [N, V_dim]    bf16
    DeviceBuffer<std::uint16_t> a;             // [N, V_h]      bf16
    DeviceBuffer<std::uint16_t> b;             // [N, V_h]      bf16
    DeviceBuffer<std::uint16_t> mixed_qkv_post; // [N, conv_dim] bf16 — post-conv

    DeviceBuffer<float> q_norm;     // [N, V_h, K_d] fp32 — l2-normed + scaled
    DeviceBuffer<float> k_norm;     // [N, V_h, K_d] fp32 — l2-normed
    DeviceBuffer<float> v_fp32;     // [N, V_h, V_d] fp32
    DeviceBuffer<float> g_log;      // [N, V_h]       fp32
    DeviceBuffer<float> beta;       // [N, V_h]       fp32
    DeviceBuffer<float> core_out;   // [N, V_h, V_d]  fp32

    DeviceBuffer<std::uint16_t> core_out_bf16;  // [N, V_dim] bf16

    static Qwen3_5LinearAttnWorkspace allocate(
        int max_tokens, int conv_dim, int v_h, int k_d, int v_d);
};

// Forward pass over `total_tokens` tokens packed as a flat [N, H]
// stream. The state caches are SINGLE-REQUEST for now — `R == 1` is
// enforced; multi-request batching for the linear-attn path will land
// alongside Phase 5's chunked DeltaNet.
//
// Routes through:
//   * `is_pure_decode = true`: reads state, runs the recurrent step,
//     updates state. Conv/recurrent caches must be pre-warmed by a
//     prior prefill call.
//   * `is_pure_decode = false` (prefill): resets the state caches,
//     runs the sequential per-token recurrence (Phase-5 simplification),
//     persists final state.
void qwen3_5_forward_paged(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    KvCache& cache,
    Qwen3_5StateCache& state_cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens, int num_requests,
    bool is_pure_decode,
    const std::uint8_t* mask_d,
    const std::int32_t* mask_indptr_d);

}  // namespace pie_cuda_driver::model

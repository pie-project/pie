#pragma once

// Qwen3 forward pass — single-sequence prefill, no KV cache, no batching.
// **Parity-test path only.** The full BPIQ-driven path with paged KV lands
// in M1.3.

#include <cstdint>

#include <cuda_runtime.h>

#include "engine.hpp"
#include "model/qwen3.hpp"
#include "ops/gemm.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

// Reusable scratch buffers, sized once for `max_tokens`. The forward pass
// only writes prefixes of these, so reusing across calls is safe as long as
// you don't exceed `max_tokens`.
struct Qwen3Workspace {
    DeviceTensor y;          // [max_tokens, hidden]
    DeviceTensor norm_x;     // [max_tokens, hidden]
    DeviceTensor q;          // [max_tokens, h_q  * d]
    DeviceTensor k;          // [max_tokens, h_kv * d]
    DeviceTensor v;          // [max_tokens, h_kv * d]
    DeviceTensor attn_out;   // [max_tokens, h_q  * d]
    DeviceTensor norm_y;     // [max_tokens, hidden]
    DeviceTensor gate;       // [max_tokens, intermediate]
    DeviceTensor up;         // [max_tokens, intermediate]
    DeviceTensor logits;     // [max_tokens, vocab]
    DeviceTensor probs;      // [max_tokens, vocab] FP32 — softmax scratch for sampling

    static Qwen3Workspace allocate(const HfConfig& cfg, int max_tokens);
};

// Run prefill on `num_tokens` consecutive tokens starting at position 0.
// Inputs `token_ids` and `positions` are device pointers. Writes
// `ws.logits[:num_tokens, :]`.
void qwen3_forward_prefill(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,    // device, [num_tokens]
    const std::int32_t* positions,    // device, [num_tokens]
    int num_tokens);

}  // namespace pie_cuda_driver::model

// ── Paged variant ──────────────────────────────────────────────────────────
#include "attention_workspace.hpp"
#include "kv_cache.hpp"
#include "ops/attention_flashinfer.hpp"

namespace pie_cuda_driver::model {

// Same forward pass but routes K/V through the paged KV pool, matching the
// BPIQ contract. Each layer's K/V is written into `cache` at the locations
// described by the page-indptr arrays, and the attention call reads back
// from those same pages.
//
// `is_pure_decode` selects between the flashinfer paged decode kernel
// (when every request has qo_len == 1) and the naive reference path.
// Phase 2 will add the flashinfer prefill kernel; until then prefill
// stays on the naive path.
void qwen3_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,           // device, [total_tokens]
    const std::int32_t* positions,           // device, [total_tokens]
    const std::uint32_t* qo_indptr,          // device, [num_requests + 1]
    const std::uint32_t* kv_page_indices,    // device
    const std::uint32_t* kv_page_indptr,     // device, [num_requests + 1]
    const std::uint32_t* kv_last_page_lens,  // device, [num_requests]
    const std::uint32_t* qo_indptr_h,        // host pointer, [num_requests + 1]
    const std::uint32_t* kv_page_indptr_h,   // host pointer, [num_requests + 1]
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    // Optional custom mask. When non-null, the prefill path uses
    // flashinfer's MaskMode::kCustom; ignored on the decode path.
    const std::uint8_t*  custom_mask_d = nullptr,
    const std::int32_t*  custom_mask_indptr_d = nullptr,
    // Optional precomputed flashinfer decode plan. Non-null only on the
    // pure-decode + cuda-graph path: the request handler runs the plan
    // OUTSIDE cudaStreamBeginCapture so its host-side work (vector
    // allocation, work estimation, page-locked fill) doesn't get baked
    // into the captured graph and produce stale layouts on replay
    // (see forward_graph.hpp constraint #2). When null, the function
    // computes its own plan inline (eager path).
    const ops::DecodePlanCache* decode_plan_in = nullptr,
    // Stream to launch all forward-pass kernels on. Required for cuda
    // graph capture: cudaStreamBeginCapture only records ops on the
    // captured stream, so launches on the default (nullptr) stream
    // bypass capture and the resulting graph runs zero kernels on
    // replay. Default keeps the eager path on the legacy default stream.
    cudaStream_t stream = nullptr);

}  // namespace pie_cuda_driver::model

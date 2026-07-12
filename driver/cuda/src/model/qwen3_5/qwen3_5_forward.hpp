#pragma once

// Qwen3.5 forward driver — hybrid linear-attention + full-attention.
// Produces last-token logits in `ws.logits` (or all-token logits when
// the workspace is full-width).

#include <cstdint>

#include "ops/attention_workspace.hpp"
#include "device_buffer.hpp"
#include "distributed.hpp"
#include "store/kv_cache.hpp"
#include "model/llama_like/qwen3.hpp"
#include "model/qwen3_5/qwen3_5.hpp"
#include "model/workspace.hpp"  // for Workspace (reused)
#include "ops/attention_flashinfer.hpp"  // DecodePlanCachePtr
#include "ops/gemm.hpp"
#include "store/recurrent_state_cache.hpp"

namespace pie_cuda_driver::model {

// Per-linear-attention-layer extra workspace. Allocated once and
// reused across layers (linear layers are processed sequentially).
struct Qwen3_5LinearAttnWorkspace {
    DeviceBuffer<std::uint16_t> mixed_qkv;     // [N, conv_dim] bf16
    DeviceBuffer<std::uint16_t> mixed_qkvz;    // [N, conv_dim + V_dim] bf16
    DeviceBuffer<std::uint16_t> ba;            // [N, 2 * V_h] bf16
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

    // Hoisted from per-layer-call temp allocs. Sized to max_tokens
    // capacity once at startup; reused across layers + steps.
    DeviceBuffer<std::uint16_t> q_raw;   // [N, K_dim] bf16 (= K_h*K_d)
    DeviceBuffer<std::uint16_t> k_raw;   // [N, K_dim] bf16
    DeviceBuffer<std::uint16_t> v_raw;   // [N, V_dim] bf16
    DeviceBuffer<float>         q_pre;   // [N, K_h, K_d] fp32 (pre-repeat)
    DeviceBuffer<float>         k_pre;   // [N, K_h, K_d] fp32 (pre-repeat)

    // Full-attention layers (Qwen3.5 / 3.6-MoE) need a 2x-wide q+gate
    // packed buffer plus a separate gate buffer. Hoisted alongside the
    // linear-attn scratch so the same workspace covers both layer kinds.
    DeviceBuffer<std::uint16_t> fa_qg_packed;  // [N, 2*Hq] bf16
    DeviceBuffer<std::uint16_t> fa_gate;       // [N, Hq]   bf16

    static Qwen3_5LinearAttnWorkspace allocate(
        int max_tokens, int conv_dim, int v_h, int k_h, int k_d, int v_d,
        int hq);
};

// Per-fire knobs that control how the forward dispatches to flashinfer
// for the full-attention layers. `force_prefill_path = true` keeps every
// fire on the prefill kernel even when `is_pure_decode == true` — useful
// for parity-mode where we want the same dispatch shape across the run.
// In serving (executor) we leave it false so the graph-capturable
// decode kernel is picked when `is_pure_decode` fires, which is the
// only mode the graph cache key supports.
struct Qwen3_5ForwardCfg {
    bool force_prefill_path = false;

    // Small speculative-verification forwards (`N = D + 1`, `R = 1`)
    // need a graph-capturable full-attention path. FlashInfer prefill
    // planning is hoisted into Qwen3_5PlanState; this legacy-named knob
    // controls which short prefill-like batches use graph-friendly planning.
    int small_prefill_naive_attention_max_tokens = 0;

    // Tensor-parallel state. tp_size > 1 activates sharded dims for both
    // full-attention layers (Q/K/V column-parallel, O row-parallel) and
    // linear-attention layers (per-rank head shares of the conv1d, the
    // mixed_qkv mix, and the recurrent state). bind_qwen3_5 has
    // already produced per-rank tensors for the fused QKV/conv1d weights;
    // forward just needs to use the local-head dims.
    int tp_size = 1;
    NcclComm* tp_comm = nullptr;

    // Qwen MTP keeps the in-flight draft chain in local history. Its paged
    // cache lookup should stay pinned to the verified source prefix while
    // draft positions advance for RoPE and history masking.
    bool mtp_global_cache_uses_prefix_position = false;
};

// Persistent decode-plan cache. Owned by main.cpp's serving setup so
// the same cache pointer is read both by the per-fire `prepare` hook
// (which calls `prepare_qwen3_5_decode_plan`) and by `qwen3_5_forward_paged`
// itself when it dispatches the captured attention kernel. Plan() is the
// host-side work that breaks graph capture if folded into the body — by
// hoisting it here we let the executor refresh it once per fire,
// outside any cudaStream capture region.
struct Qwen3_5PlanState {
    ops::DecodePlanCachePtr decode_plan;
    ops::PrefillPlanCachePtr prefill_plan;
    bool use_prefill_plan = false;
};

// Refresh the decode plan for the current fire. Caller is expected to
// invoke this BEFORE either a direct forward call OR a graph replay,
// outside any capture region. No-op when `is_pure_decode == false`.
void prepare_qwen3_5_decode_plan(
    Qwen3_5PlanState& state,
    AttentionWorkspace& attn_ws,
    KvCache& cache,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    cudaStream_t stream = nullptr);

std::uint32_t qwen3_5_decode_graph_layout(
    const Qwen3_5PlanState& state);

// Forward pass over `total_tokens` tokens packed as a flat [N, H]
// stream. Multi-request batching: `slot_ids_h[r]` selects the
// per-request slab inside the linear-attn rs_cache; `is_fresh_h[r]`
// tells the body to zero that slab before consuming it (a request
// whose slot was just (re)assigned must be fed a prefill — guaranteed
// by the runtime's eviction → re-prefill invariant). `slot_ids_d` is
// the same array on device, consumed by the batched kernels.
//
// All three are nullable. nullptr → legacy single-request mode (slot 0,
// fresh on prefill, sticky on decode); used by the parity entry point.
// Multi-request callers must pass all three non-null.
//
// Routes through:
//   * `is_pure_decode = true`: batched recurrent step over (R, V_h)
//     blocks, slot-indexed per request.
//   * `is_pure_decode = false` (prefill): zeroes the slot for each
//     request flagged `is_fresh`, then runs the batched chunked
//     DeltaNet prefill — one block per (request, head) walking that
//     request's `qo_indptr` window.
void qwen3_5_forward_paged(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3_5PlanState& plan_state,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
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
    const std::int32_t* mask_indptr_d,
    const std::int32_t* slot_ids_h = nullptr,
    const std::uint8_t* is_fresh_h = nullptr,
    const std::int32_t* slot_ids_d = nullptr,
    const std::int32_t* logit_row_indices_d = nullptr,
    int num_logit_rows = 0,
    const std::int32_t* commit_advance_gather = nullptr,
    // Ph7 RS working-set fold-from-buffer (see batch/forward.hpp ForwardInputs):
    // per-request CSR of buffered-slab pool ids + direction flags.
    const std::uint32_t* rs_buffer_slot_ids_h = nullptr,
    const std::uint32_t* rs_buffer_slot_indptr_h = nullptr,
    bool rs_buffer_write = false,
    bool rs_buffer_fold = false);

void qwen3_5_mtp_process_cache(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::int32_t* slot_ids_d,
    const std::int32_t* source_row_indices,
    int total_tokens,
    int num_requests);

// Run one NEXTN step of Qwen3.6's one-layer MTP head. The
// `base_hidden_row_indices` select rows from the target model's last hidden
// states left in `ws.y` by `qwen3_5_forward_paged` (or from the previous MTP
// step); `token_ids` are the just accepted/drafted tokens at those rows' next
// positions. Writes logits to `ws.logits[0..num_tokens)`.
void qwen3_5_mtp_forward(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    KvCache& cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* position_ids,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    std::int32_t* sampled_token_ids,
    int num_tokens,
    int draft_step,
    int max_global_tokens);

// Workspace byte budget for the linear-attention path. Returns 0 if the
// model has no linear-attn layers (e.g. HF config absent).
std::size_t qwen3_5_la_workspace_bytes(const HfConfig& cfg,
                                       int N,
                                       int tp_size = 1);

}  // namespace pie_cuda_driver::model

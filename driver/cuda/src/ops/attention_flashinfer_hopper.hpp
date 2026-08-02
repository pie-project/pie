#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "ops/attention_workspace.hpp"

namespace pie_cuda_driver::ops {

struct HopperPrefillPlan {
    std::int64_t qo_tile_indices_offset = 0;
    std::int64_t qo_indptr_offset = 0;
    std::int64_t kv_indptr_offset = 0;
    std::int64_t qo_len_offset = 0;
    std::int64_t kv_len_offset = 0;
    std::int64_t head_indices_offset = 0;
    std::int64_t work_indptr_offset = 0;
    std::int64_t batch_indices_offset = 0;
    bool same_schedule_for_all_heads = false;
    int total_tokens = 0;
    int num_requests = 0;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int page_size = 0;
    int window_left = -1;
    bool causal = true;
    bool valid = false;
};

// Whether `merge_attention_states_bf16` below is BUILT on this
// architecture. The stub TU's comment asserts "nothing on this arch can
// reach a call to it, because the only caller runs after a dispatch that
// already threw" — true of the sm90 prefill's caller, and NOT true of a
// caller that merges partials produced by the ordinary decode dispatch.
// Such a caller has to ask.
bool merge_attention_states_supported();

bool hopper_prefill_supported(int head_dim,
                              int window_left,
                              int total_tokens,
                              int num_requests);

std::uint8_t hopper_prefill_graph_layout(const HopperPrefillPlan& plan);

void plan_attention_flashinfer_prefill_sm90_bf16(
    HopperPrefillPlan& plan,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool causal,
    int window_left,
    std::size_t int_base_bytes = 0);

void dispatch_attention_flashinfer_prefill_sm90_bf16(
    const HopperPrefillPlan& plan,
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap = 0.f,
    float sm_scale = -1.f,
    float* lse_out = nullptr,
    // Every request reads the same Q row. Used by the KV split, whose
    // pseudo-requests are one query against different slices of the pages --
    // the outputs stay per-request, only the input is shared.
    bool broadcast_q = false);

// Folds `num_index_sets` partial attention outputs and their log-sum-exps into
// one. Wraps flashinfer's cascade `MergeStates`; `v` is
// [num_index_sets, seq_len, num_heads, head_dim] bf16 and `s` the matching
// [num_index_sets, seq_len, num_heads] floats the dispatch writes to `lse_out`.
void merge_attention_states_bf16(
    const void* v, const float* s,
    void* v_merged, float* s_merged,
    int num_index_sets, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::ops

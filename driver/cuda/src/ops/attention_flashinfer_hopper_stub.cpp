#include "ops/attention_flashinfer_hopper.hpp"
#include <stdexcept>

namespace pie_cuda_driver::ops {

bool hopper_prefill_supported(int /*head_dim*/,
                              int /*window_left*/,
                              int /*total_tokens*/,
                              int /*num_requests*/) {
    return false;
}

std::uint8_t hopper_prefill_graph_layout(const HopperPrefillPlan& /*plan*/) {
    return 0;
}

void plan_attention_flashinfer_prefill_sm90_bf16(
    HopperPrefillPlan& /*plan*/,
    const std::uint32_t* /*qo_indptr_h*/,
    const std::uint32_t* /*kv_page_indptr_h*/,
    const std::uint32_t* /*kv_last_page_lens_h*/,
    int /*total_tokens*/,
    int /*num_requests*/,
    int /*num_q_heads*/,
    int /*num_kv_heads*/,
    int /*head_dim*/,
    int /*page_size*/,
    AttentionWorkspace& /*workspace*/,
    cudaStream_t /*stream*/,
    bool /*enable_cuda_graph*/,
    bool /*causal*/,
    int /*window_left*/,
    std::size_t /*int_base_bytes*/) {
    throw std::runtime_error("flashinfer sm90 prefill is not built for this CUDA architecture");
}

void dispatch_attention_flashinfer_prefill_sm90_bf16(
    const HopperPrefillPlan& /*plan*/,
    const void* /*q*/,
    void* /*k_pages*/,
    void* /*v_pages*/,
    void* /*o*/,
    const std::uint32_t* /*kv_page_indices_d*/,
    AttentionWorkspace& /*workspace*/,
    cudaStream_t /*stream*/,
    float /*logits_soft_cap*/,
    float /*sm_scale*/,
    float* /*lse_out*/,
    bool /*broadcast_q*/) {
    throw std::runtime_error("flashinfer sm90 prefill is not built for this CUDA architecture");
}

// Not Hopper-only in principle -- it wraps a cascade kernel that would build
// anywhere -- but it lives in the Hopper translation unit because the KV split
// that produces its inputs is the sm90 prefill's. Stubbed with the rest of
// that unit; nothing on this arch can reach a call to it, because the only
// caller runs after a dispatch that already threw.
bool merge_attention_states_supported() { return false; }

void merge_attention_states_bf16(
    const void* /*v*/, const float* /*s*/,
    void* /*v_merged*/, float* /*s_merged*/,
    int /*num_index_sets*/, int /*seq_len*/, int /*num_heads*/,
    int /*head_dim*/, cudaStream_t /*stream*/) {
    throw std::runtime_error(
        "flashinfer attention-state merge is not built for this CUDA architecture");
}

}  // namespace pie_cuda_driver::ops

namespace pie_cuda_driver::ops::detail {

void launch_attention_xqa_decode_bf16_gqa8_sm90(
    const void* /*q*/,
    void* /*k_pages*/,
    void* /*v_pages*/,
    void* /*o*/,
    const std::uint32_t* /*kv_page_indices_d*/,
    const std::uint32_t* /*kv_page_indptr_d*/,
    const std::uint32_t* /*kv_last_page_lens_d*/,
    int /*num_requests*/,
    int /*num_q_heads*/,
    int /*num_kv_heads*/,
    int /*head_dim*/,
    int /*page_size*/,
    int /*max_pages_per_seq*/,
    AttentionWorkspace& /*workspace*/,
    cudaStream_t /*stream*/,
    float /*sm_scale*/) {
    throw std::runtime_error("xqa gqa8 sm90 decode is not built for this CUDA architecture");
}

void launch_attention_xqa_decode_bf16_gqa8_sm90_prepared(
    const void* /*q*/,
    void* /*k_pages*/,
    void* /*v_pages*/,
    void* /*o*/,
    int /*num_requests*/,
    int /*num_q_heads*/,
    int /*num_kv_heads*/,
    int /*head_dim*/,
    int /*page_size*/,
    int /*max_pages_per_seq*/,
    AttentionWorkspace& /*workspace*/,
    cudaStream_t /*stream*/,
    float /*sm_scale*/) {
    throw std::runtime_error("xqa gqa8 sm90 decode is not built for this CUDA architecture");
}

void xqa_decode_bf16_gqa8_sm90_warmup_current_device() {}

}  // namespace pie_cuda_driver::ops::detail

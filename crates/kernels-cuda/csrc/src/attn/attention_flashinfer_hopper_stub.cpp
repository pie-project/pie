#include "attn/attention_flashinfer_hopper.hpp"
#include <stdexcept>

namespace pie_cuda_driver::kernels::attn {

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
    AttentionWorkspaceView /*workspace*/,
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
    AttentionWorkspaceView /*workspace*/,
    cudaStream_t /*stream*/,
    float /*logits_soft_cap*/,
    float /*sm_scale*/,
    float* /*lse_out*/,
    bool /*broadcast_q*/) {
    throw std::runtime_error("flashinfer sm90 prefill is not built for this CUDA architecture");
}

// `kernels::attn::merge_attention_states_bf16` is deliberately NOT stubbed here.
//
// It used to be, on the reasoning that the KV split producing its inputs was
// the sm90 prefill's, so the only caller ran after a dispatch that had
// already thrown. That was wrong: the DECODE KV-split path calls
// `kernels::attn::dispatch_attention_flashinfer_decode_bf16`, which is built on every
// architecture, and then merges. On sm_100 the dispatch succeeded and this
// stub threw on the first fire, poisoning the driver and taking gpt-oss and
// gemma-4 down with it.
//
// The real implementation now lives in `attention_merge_states.cu`, which is
// compiled unconditionally. Re-adding a stub here would be an ODR conflict,
// which is the desired outcome: it makes the mistake a link error rather
// than a runtime one.

}  // namespace pie_cuda_driver::kernels::attn

namespace pie_cuda_driver::kernels::attn::detail {

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
    AttentionWorkspaceView /*workspace*/,
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
    AttentionWorkspaceView /*workspace*/,
    cudaStream_t /*stream*/,
    float /*sm_scale*/) {
    throw std::runtime_error("xqa gqa8 sm90 decode is not built for this CUDA architecture");
}

void xqa_decode_bf16_gqa8_sm90_warmup_current_device() {}

}  // namespace pie_cuda_driver::kernels::attn::detail

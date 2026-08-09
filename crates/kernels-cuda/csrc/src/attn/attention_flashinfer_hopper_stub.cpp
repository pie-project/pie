// The non-SM90 arm of `PIE_CUDA_FLASHINFER_HOPPER_SOURCE`.
//
// WHAT THIS FILE IS FOR HAS CHANGED, and the name no longer describes it.
// `attention_flashinfer_hopper.cu` — the real FA3 prefill this stubbed — is
// DELETED, unreachable by call graph and not merely by a flag; the CMakeLists
// entry carries the argument. So the three `..._sm90_bf16` bodies below are
// now the ONLY definitions of those symbols anywhere, and only on non-sm90
// builds. That is harmless because nothing calls them (no table row, no
// `pie_k_*` entry, no `ffi` arm, and their one C++ caller went with
// `driver-cuda/csrc/`), and it is deliberate: a throw that nothing reaches is
// cheaper to keep than a declaration set to re-derive if FA3 comes back.
//
// The one body here that is still LIVE is the last one:
// `detail::launch_attention_xqa_decode_bf16_gqa8_sm90_prepared`, which
// `attention_xqa_gqa8.cu:156` calls unconditionally inside the GQA-8 arm of
// `attention_xqa_decode_bf16_prepared`. That is a real, rowed, model-reachable
// path on every architecture, and it is why this file cannot go until the six
// `attention_xqa*.cu` do.
#include "attn/attention_flashinfer_hopper.hpp"
#include <stdexcept>

namespace pie_cuda_driver::kernels::attn {

bool hopper_prefill_supported(int /*head_dim*/,
                              int /*window_left*/,
                              int /*total_tokens*/,
                              int /*num_requests*/) {
    return false;
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
// THE REAL IMPLEMENTATION IS ALSO GONE NOW. It lived in
// `attention_merge_states.cu`, compiled unconditionally, and that file is
// DELETED — not because the argument above stopped holding, but because
// `attn::merge_attention_states_bf16` lost its `table::attn` row to §38 and
// the definition had been unreachable ever since. So the symbol is absent
// from the archive on every architecture, and re-adding a stub HERE would be
// wrong twice: it would resurrect the sm90 gating this paragraph exists to
// refute, and it would answer a symbol whose only correct answer is a row
// with a Rust body over `flashinfer/attention/cascade.cuh`. The CMakeLists
// entry that was this file's neighbour carries the full consumer sweep.

}  // namespace pie_cuda_driver::kernels::attn

namespace pie_cuda_driver::kernels::attn::detail {
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
}  // namespace pie_cuda_driver::kernels::attn::detail

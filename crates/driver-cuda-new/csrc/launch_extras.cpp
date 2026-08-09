// The bridge's HAND-WRITTEN entries — object lifecycle the generated shim
// cannot express (retirement plan phase C).
//
// `emit_c_shim` forwards launcher calls, and a row states a signature. The
// FlashInfer plan caches are neither: `make_decode_plan()` returns a
// `unique_ptr` with a custom deleter, which is not an `extern "C"` shape,
// and the prepare takes the cache by MUTABLE reference. So these few
// entries are written by hand, own the release()/deleter dance, and speak
// raw pointers across the boundary. They are compiled by the same `cc`
// step as the generated shim, against the same headers — a drifted
// signature is still a compile error, which is the property that matters.
//
// Naming: `pie_x_*` (extras), never `pie_k_*` — the generated namespace
// stays the table's alone.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "attn/attention_flashinfer.hpp"
#include "attention_workspace_view.hpp"

using pie_cuda_driver::AttentionWorkspaceView;
using pie_cuda_driver::kernels::attn::DecodePlanCache;
using pie_cuda_driver::kernels::attn::DecodePlanCacheDeleter;
using pie_cuda_driver::kernels::attn::make_decode_plan;
using pie_cuda_driver::kernels::attn::PrefillPlanCache;
using pie_cuda_driver::kernels::attn::PrefillPlanCacheDeleter;
using pie_cuda_driver::kernels::attn::make_prefill_plan;

extern "C" DecodePlanCache* pie_x_make_decode_plan() {
    return make_decode_plan().release();
}

extern "C" void pie_x_destroy_decode_plan(DecodePlanCache* cache) {
    DecodePlanCacheDeleter{}(cache);
}

extern "C" void pie_x_set_decode_plan_int_base(DecodePlanCache* cache, std::size_t bytes) {
    pie_cuda_driver::kernels::attn::set_decode_plan_int_base(*cache, bytes);
}

extern "C" void pie_x_plan_attention_flashinfer_decode_bf16(
    DecodePlanCache* cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool full_attention_variant,
    bool hnd_layout,
    int window_left)
{
    pie_cuda_driver::kernels::attn::plan_attention_flashinfer_decode_bf16(
        *cache, kv_page_indptr_h, num_requests, num_q_heads, num_kv_heads,
        head_dim, page_size, workspace, stream, enable_cuda_graph,
        full_attention_variant, hnd_layout, window_left);
}

extern "C" PrefillPlanCache* pie_x_make_prefill_plan() {
    return make_prefill_plan().release();
}

extern "C" void pie_x_destroy_prefill_plan(PrefillPlanCache* cache) {
    PrefillPlanCacheDeleter{}(cache);
}

extern "C" void pie_x_plan_attention_flashinfer_prefill_bf16(
    PrefillPlanCache* cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    int window_left,
    bool full_attention_variant,
    bool hnd_layout,
    bool causal_mask,
    bool custom_mask,
    bool wants_prefill_score)
{
    pie_cuda_driver::kernels::attn::plan_attention_flashinfer_prefill_bf16(
        *cache, qo_indptr_h, kv_page_indptr_h, kv_last_page_lens_h,
        total_tokens, num_requests, num_q_heads, num_kv_heads, head_dim,
        page_size, workspace, stream, enable_cuda_graph, window_left,
        full_attention_variant, hnd_layout, causal_mask, custom_mask,
        wants_prefill_score);
}

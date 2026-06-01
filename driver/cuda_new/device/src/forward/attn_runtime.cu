#include "forward/attn_runtime.cuh"

#include <vector>

#include "ops/attention_naive_paged.cuh"
#include "workspace.hpp"

namespace pie_cuda_device::forward {

namespace {
constexpr bool tc_head_dim(int d) {
    return d == 64 || d == 128 || d == 256 || d == 512;
}
}  // namespace

AttnPlan plan_attention_for_fire(
    PieWorkspace* ws,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indptr_d,
    int num_tokens, int num_requests,
    int n_q_heads, int n_kv_heads, int head_dim, int page_size,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool full_attention_variant,
    int window_left,
    bool causal_mask)
{
    AttnPlan p;
    p.num_tokens = num_tokens;
    p.num_requests = num_requests;
    p.n_q_heads = n_q_heads;
    p.n_kv_heads = n_kv_heads;
    p.head_dim = head_dim;
    p.page_size = page_size;

    // Tensor-core templates only cover these head dims; anything else (e.g. 96,
    // or MLA's latent dims handled on their own path) stays on naive.
    if (!tc_head_dim(head_dim) || num_requests <= 0 || num_tokens <= 0) {
        return p;  // ws == nullptr → naive fallback
    }

    // Lazily stand up the plan scratch + caches on first use (archs that never
    // hit this path — MLA, Nemotron — don't pay the 136 MiB).
    if (!ws->attn_ready) {
        ws->attn_ws = ops::AttentionWorkspace::allocate();
        ws->prefill_plan = ops::make_prefill_plan();
        ws->decode_plan = ops::make_decode_plan();
        ws->attn_ready = true;
    }

    const int gqa = (n_kv_heads > 0) ? (n_q_heads / n_kv_heads) : 0;
    const bool is_pure_decode = (num_tokens == num_requests);
    p.use_decode = is_pure_decode && n_kv_heads > 0 &&
                   (n_q_heads % n_kv_heads == 0) && ops::decode_supports_gqa(gqa);

    const int R = num_requests;
    // The FlashInfer static-nonsplit decode plan (taken by default for R<=512 on
    // sm80+) produces an R-only schedule that ignores kv_page_indptr values
    // (request_indices=[0..R], o_indptr=[0..R], kv_chunk=page_size); the only
    // consumer of `kv_page_indptr_h` there is `num_pages_in_batch`, which the
    // raw-bf16 dispatch never reads (it fed the dropped quantized dequant-active
    // prologue). So for that regime we skip the per-fire synchronous D2H of the
    // index arrays entirely — critical for batched decode throughput (a D2H here
    // stalls the stream every fire and kills host-side fire pipelining). The
    // kernel reads the *device* page tables directly. Above 512 (the dynamic
    // planner, which does need real indptr) we fall back to the D2H.
    const bool decode_no_d2h = p.use_decode && R <= 512;
    if (decode_no_d2h) {
        std::vector<std::uint32_t> kvpp_h(R + 1, 0);
        ops::plan_attention_paged_decode_bf16(
            *ws->decode_plan, kvpp_h.data(), R, n_q_heads, n_kv_heads, head_dim,
            page_size, ws->attn_ws, stream, enable_cuda_graph,
            full_attention_variant, /*hnd_layout=*/false);
        p.decode = ws->decode_plan.get();
    } else if (p.use_decode) {
        std::vector<std::uint32_t> kvpp_h(R + 1);
        cudaMemcpy(kvpp_h.data(), kv_page_indptr_d,
                   sizeof(std::uint32_t) * (R + 1), cudaMemcpyDeviceToHost);
        ops::plan_attention_paged_decode_bf16(
            *ws->decode_plan, kvpp_h.data(), R, n_q_heads, n_kv_heads, head_dim,
            page_size, ws->attn_ws, stream, enable_cuda_graph,
            full_attention_variant, /*hnd_layout=*/false);
        p.decode = ws->decode_plan.get();
    } else {
        // PREFILL — the CTA-tile scheduler genuinely needs the host index
        // arrays. This is a one-shot fire (not the hot decode loop), so the D2H
        // sync is acceptable here.
        std::vector<std::uint32_t> qo_h(R + 1), kvpp_h(R + 1);
        cudaMemcpy(kvpp_h.data(), kv_page_indptr_d,
                   sizeof(std::uint32_t) * (R + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(qo_h.data(), qo_indptr_d,
                   sizeof(std::uint32_t) * (R + 1), cudaMemcpyDeviceToHost);
        ops::plan_attention_paged_prefill_bf16(
            *ws->prefill_plan, qo_h.data(), kvpp_h.data(), num_tokens, R,
            n_q_heads, n_kv_heads, head_dim, page_size, ws->attn_ws, stream,
            enable_cuda_graph, window_left, full_attention_variant,
            /*hnd_layout=*/false, causal_mask);
        p.prefill = ws->prefill_plan.get();
    }
    p.ws = &ws->attn_ws;
    return p;
}

void run_attention_layer(
    const AttnPlan& p,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    cudaStream_t stream,
    int window_left, float logits_soft_cap, float sm_scale, float* lse_out)
{
    if (p.ws == nullptr) {
        // Naive fallback (unsupported head_dim).
        ops::attention_naive_paged_bf16(
            q, k_pages, v_pages, o, qo_indptr_d, kv_page_indices_d,
            kv_page_indptr_d, kv_last_page_lens_d, p.num_tokens, p.num_requests,
            p.n_q_heads, p.n_kv_heads, p.head_dim, p.page_size, stream,
            window_left, sm_scale, logits_soft_cap, lse_out);
        return;
    }
    if (p.use_decode) {
        ops::dispatch_attention_paged_decode_bf16(
            *p.decode, q, k_pages, v_pages, o, kv_page_indices_d,
            kv_page_indptr_d, kv_last_page_lens_d, *p.ws, stream, window_left,
            logits_soft_cap, sm_scale, lse_out);
    } else {
        ops::dispatch_attention_paged_prefill_bf16(
            *p.prefill, q, k_pages, v_pages, o, qo_indptr_d, kv_page_indices_d,
            kv_page_indptr_d, kv_last_page_lens_d, *p.ws, stream,
            logits_soft_cap, sm_scale, lse_out);
    }
}

}  // namespace pie_cuda_device::forward

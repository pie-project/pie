#include "moe_forward.cuh"

#include "moe_mlp.cuh"

#include "../kernels/argmax.cuh"
#include "../kernels/embed.cuh"
#include "../kernels/kv_append.cuh"
#include "../kernels/residual_add.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../ops/attention_naive_paged.cuh"
#include "../ops/gemm.cuh"

#include <cstdint>

namespace pie_cuda_device::forward {

namespace {

// Attention-half scratch, sized for the token capacity and reused across
// layers (no per-layer alloc). Mirrors LayerScratch's attention slots.
struct AttnScratch {
    void* normed;   // [T, H]
    void* q;        // [T, Hq]
    void* k;        // [T, Hkv]
    void* v;        // [T, Hkv]
    void* attn;     // [T, Hq]
    void* o;        // [T, H]
    void* moe_out;  // [T, H]
};

// One MoE decoder layer in place on `hidden` [T, H]. Attention half is the
// standard llama pre-norm block (identical chaining to decoder_layer_inplace);
// the MLP half is replaced by the dense top-K MoE FFN.
void moe_decoder_layer_inplace(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const MoeLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    const AttnScratch& s, int num_tokens, int num_requests,
    const MoeForwardDims& d) {
    const int T = num_tokens;
    const int H = d.hidden_size;
    const int Hq = d.n_q_heads * d.head_dim;
    const int Hkv = d.n_kv_heads * d.head_dim;

    // --- attention block (GQA, rope, paged KV; causal) ---
    kernels::rmsnorm_bf16(hidden, w.attn_norm, s.normed, T, H, d.rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wq, s.q, T, Hq, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wk, s.k, T, Hkv, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wv, s.v, T, Hkv, H, 0.f);
    kernels::rope_bf16(s.q, s.k, positions, T, d.n_q_heads, d.n_kv_heads, d.head_dim,
                       d.rope_theta, /*interleaved=*/false, stream);
    kernels::write_kv_to_pages_bf16(k_pages, v_pages, s.k, s.v, qo_indptr, kv_page_indices,
                                    kv_page_indptr, kv_last_page_lens, T, num_requests,
                                    d.page_size, d.n_kv_heads, d.head_dim,
                                    /*hnd_layout=*/false, stream);
    ops::attention_naive_paged_bf16(
        s.q, k_pages, v_pages, s.attn, qo_indptr, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, T, num_requests, d.n_q_heads, d.n_kv_heads, d.head_dim,
        d.page_size, stream, /*window_left=*/-1, /*sm_scale=*/-1.f,
        /*logits_soft_cap=*/0.f, nullptr);
    ops::gemm_act_x_wt_bf16(cublas, s.attn, w.wo, s.o, T, H, Hq, 0.f);
    kernels::residual_add_bf16(hidden, s.o, (size_t)T * H, stream);

    // --- MoE FFN block (router -> top-K softmax -> per-expert swiglu -> combine) ---
    kernels::rmsnorm_bf16(hidden, w.ffn_norm, s.normed, T, H, d.rms_eps, stream);
    // moe_mlp_block_bf16 allocates its own MoE scratch and syncs the stream
    // internally; correctness baseline (dense over all E experts).
    moe_mlp_block_bf16(cublas, stream, s.normed, w.router_w, w.wgu, w.wdown, s.moe_out,
                       T, H, d.intermediate, d.num_experts, d.top_k);
    kernels::residual_add_bf16(hidden, s.moe_out, (size_t)T * H, stream);
}

}  // namespace

cudaError_t moe_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const MoeForwardWeights& w,
    const std::int32_t* positions,
    void* kv_k, void* kv_v,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, int num_kv_pages,
    const MoeForwardDims& d) {
    const int T = num_tokens;
    const int H = d.hidden_size;
    const int Hq = d.n_q_heads * d.head_dim;
    const int Hkv = d.n_kv_heads * d.head_dim;
    constexpr size_t es = sizeof(std::uint16_t);  // bf16

    // Residual stream + attention-half scratch, allocated once and reused
    // across layers. (The MoE FFN scratch is owned by moe_mlp_block_bf16.)
    enum { HIDDEN, NORMED, Q, K, V, ATTN, O, MOE_OUT, NBUF };
    const size_t sizes[NBUF] = {
        (size_t)T * H * es,    // hidden
        (size_t)T * H * es,    // normed
        (size_t)T * Hq * es,   // q
        (size_t)T * Hkv * es,  // k
        (size_t)T * Hkv * es,  // v
        (size_t)T * Hq * es,   // attn
        (size_t)T * H * es,    // o
        (size_t)T * H * es,    // moe_out
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i) {
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }
    }
    void* hidden = b[HIDDEN];
    const AttnScratch s{b[NORMED], b[Q], b[K], b[V], b[ATTN], b[O], b[MOE_OUT]};

    // embed: token_ids -> residual stream.
    kernels::embed_bf16(token_ids, w.embed, hidden, T, H, d.vocab, stream);

    // N decoder layers, with per-layer paged-KV stride.
    const size_t kv_layer_stride = (size_t)num_kv_pages * d.page_size * Hkv * es;
    for (int L = 0; L < d.n_layers; ++L) {
        void* kL = static_cast<char*>(kv_k) + (size_t)L * kv_layer_stride;
        void* vL = static_cast<char*>(kv_v) + (size_t)L * kv_layer_stride;
        moe_decoder_layer_inplace(cublas, stream, hidden, w.layers[L], positions, kL, vL,
                                  qo_indptr, kv_page_indices, kv_page_indptr,
                                  kv_last_page_lens, s, T, num_requests, d);
    }

    // final norm -> lm_head -> greedy argmax.
    kernels::rmsnorm_bf16(hidden, w.final_norm, b[NORMED], T, H, d.rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, b[NORMED], w.lm_head, out_logits, T, d.vocab, H, 0.f);
    kernels::argmax_bf16(out_logits, out_token_ids, T, d.vocab, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
    return e;
}

}  // namespace pie_cuda_device::forward

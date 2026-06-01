#include "llama_forward.cuh"

#include "../kernels/argmax.cuh"
#include "../kernels/embed.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../ops/gemm.cuh"
#include "../workspace.hpp"
#include "attn_runtime.cuh"

namespace pie_cuda_device::forward {

cudaError_t llama_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream, PieWorkspace* ws,
    const std::int32_t* token_ids, const LlamaWeights& w, const std::int32_t* positions,
    void* kv_k, void* kv_v,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, int num_kv_pages, int vocab, float rms_eps, float rope_theta) {
    const int T = num_tokens;
    const int H = hidden_size;
    const int Hkv = n_kv_heads * head_dim;
    constexpr size_t es = sizeof(std::uint16_t);  // bf16

    // embed: token_ids → residual stream.
    kernels::embed_bf16(token_ids, w.embed, ws->hidden_buf, T, H, vocab, stream);

    // Plan the fast paged attention ONCE for this fire (reused across all
    // layers). Falls back to naive inside run_attention_layer for head dims the
    // tensor-core templates don't instantiate. Llama/Qwen are full causal
    // attention (window_left=-1, no softcap, default 1/sqrt(d) scale).
    const AttnPlan attn = plan_attention_for_fire(
        ws, qo_indptr, kv_page_indptr, T, num_requests,
        n_q_heads, n_kv_heads, head_dim, page_size, stream);

    // N decoder layers, reusing the workspace scratch.
    const LayerScratch s{ws->normed, ws->q, ws->k, ws->v, ws->attn,
                         ws->o, ws->gate, ws->up, ws->mlp, ws->mlp_out};
    const size_t kv_layer_stride_bytes = (size_t)num_kv_pages * page_size * Hkv * es;
    for (int L = 0; L < w.n_layers; ++L) {
        void* kL = static_cast<char*>(kv_k) + (size_t)L * kv_layer_stride_bytes;
        void* vL = static_cast<char*>(kv_v) + (size_t)L * kv_layer_stride_bytes;
        decoder_layer_inplace(cublas, stream, ws->hidden_buf, w.layers[L], positions, kL, vL,
                              qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens, s,
                              T, num_requests, H, n_q_heads, n_kv_heads, head_dim, intermediate,
                              page_size, rms_eps, rope_theta, attn);
    }

    // final norm → lm_head → greedy argmax.
    kernels::rmsnorm_bf16(ws->hidden_buf, w.final_norm, ws->normed, T, H, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, ws->normed, w.lm_head, out_logits, T, vocab, H, 0.f);
    kernels::argmax_bf16(out_logits, out_token_ids, T, vocab, stream);

    return cudaStreamSynchronize(stream);
}

}  // namespace pie_cuda_device::forward

#include "gemma_layer.cuh"

#include "../kernels/gemma.cuh"          // rmsnorm_gemma_bf16, geglu_tanh_bf16
#include "../kernels/kv_append.cuh"
#include "../kernels/residual_add.cuh"
#include "../kernels/rope.cuh"
#include "../ops/attention_naive_paged.cuh"
#include "../ops/gemm.cuh"

namespace pie_cuda_device::forward {

void decoder_layer_gemma_inplace(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const GemmaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    const LayerScratch& s,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta,
    int window_left, float attn_logit_softcap,
    const AttnPlan& attn) {
    const int T = num_tokens;
    const int H = hidden_size;
    const int Hq = n_q_heads * head_dim;
    const int Hkv = n_kv_heads * head_dim;
    const int I = intermediate;

    // --- attention block (input-norm → attn → post-attn-norm → residual) ---
    kernels::rmsnorm_gemma_bf16(hidden, w.input_ln, s.normed, T, H, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wq, s.q, T, Hq, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wk, s.k, T, Hkv, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wv, s.v, T, Hkv, H, 0.f);
    kernels::rope_bf16(s.q, s.k, positions, T, n_q_heads, n_kv_heads, head_dim,
                       rope_theta, /*interleaved=*/false, stream);
    kernels::write_kv_to_pages_bf16(k_pages, v_pages, s.k, s.v, qo_indptr, kv_page_indices,
                                    kv_page_indptr, kv_last_page_lens, T, num_requests, page_size,
                                    n_kv_heads, head_dim, /*hnd_layout=*/false, stream);
    // Tensor-core paged attention (per-layer sliding window + Gemma attn
    // softcap), or naive fallback when attn.ws==nullptr.
    run_attention_layer(attn, s.q, k_pages, v_pages, s.attn, qo_indptr, kv_page_indices,
                        kv_page_indptr, kv_last_page_lens, stream,
                        window_left, attn_logit_softcap, /*sm_scale=*/-1.f, nullptr);
    ops::gemm_act_x_wt_bf16(cublas, s.attn, w.wo, s.o, T, H, Hq, 0.f);
    kernels::rmsnorm_gemma_bf16(s.o, w.post_attn_ln, s.o, T, H, rms_eps, stream); // post-norm (in place)
    kernels::residual_add_bf16(hidden, s.o, (size_t)T * H, stream);

    // --- MLP block (pre-ffn-norm → GeGLU → post-ffn-norm → residual) ---
    kernels::rmsnorm_gemma_bf16(hidden, w.pre_ffn_ln, s.normed, T, H, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.w_gate, s.gate, T, I, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.w_up, s.up, T, I, H, 0.f);
    kernels::geglu_tanh_bf16(s.gate, s.up, s.mlp, T * I, stream);
    ops::gemm_act_x_wt_bf16(cublas, s.mlp, w.w_down, s.mlp_out, T, H, I, 0.f);
    kernels::rmsnorm_gemma_bf16(s.mlp_out, w.post_ffn_ln, s.mlp_out, T, H, rms_eps, stream); // post-norm
    kernels::residual_add_bf16(hidden, s.mlp_out, (size_t)T * H, stream);
}

cudaError_t gemma_layer_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const GemmaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta,
    int window_left, float attn_logit_softcap) {
    const int T = num_tokens, H = hidden_size;
    const int Hq = n_q_heads * head_dim, Hkv = n_kv_heads * head_dim, I = intermediate;
    constexpr size_t es = sizeof(std::uint16_t);
    const size_t sizes[10] = {
        (size_t)T * H * es,  (size_t)T * Hq * es,  (size_t)T * Hkv * es, (size_t)T * Hkv * es,
        (size_t)T * Hq * es, (size_t)T * H * es,   (size_t)T * I * es,   (size_t)T * I * es,
        (size_t)T * I * es,  (size_t)T * H * es,
    };
    void* b[10] = {};
    for (int i = 0; i < 10; ++i) {
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }
    }
    LayerScratch s{b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]};
    // Standalone single-layer entry keeps naive attention (ws==nullptr, dims set).
    AttnPlan naive_plan;
    naive_plan.num_tokens = num_tokens;
    naive_plan.num_requests = num_requests;
    naive_plan.n_q_heads = n_q_heads;
    naive_plan.n_kv_heads = n_kv_heads;
    naive_plan.head_dim = head_dim;
    naive_plan.page_size = page_size;
    decoder_layer_gemma_inplace(cublas, stream, hidden, w, positions, k_pages, v_pages, qo_indptr,
                                kv_page_indices, kv_page_indptr, kv_last_page_lens, s, num_tokens,
                                num_requests, hidden_size, n_q_heads, n_kv_heads, head_dim,
                                intermediate, page_size, rms_eps, rope_theta, window_left,
                                attn_logit_softcap, naive_plan);
    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < 10; ++i) cudaFree(b[i]);
    return e;
}

}  // namespace pie_cuda_device::forward

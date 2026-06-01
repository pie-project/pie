#include "llama_layer.cuh"

#include "../kernels/add_bias.cuh"
#include "../kernels/kv_append.cuh"
#include "../kernels/qk_norm.cuh"
#include "../kernels/residual_add.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/swiglu.cuh"
#include "../ops/attention_naive_paged.cuh"
#include "../ops/gemm.cuh"

namespace pie_cuda_device::forward {

void decoder_layer_inplace(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const LlamaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    const LayerScratch& s,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta,
    const AttnPlan& attn,
    int window_left, float logits_soft_cap, float sm_scale) {
    const int T = num_tokens;
    const int H = hidden_size;
    const int Hq = n_q_heads * head_dim;
    const int Hkv = n_kv_heads * head_dim;
    const int I = intermediate;

    // --- attention block ---
    kernels::rmsnorm_bf16(hidden, w.attn_norm, s.normed, T, H, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wq, s.q, T, Hq, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wk, s.k, T, Hkv, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.wv, s.v, T, Hkv, H, 0.f);
    // Qwen2: additive q/k/v projection biases (skipped when null).
    if (w.q_bias != nullptr) kernels::add_bias_bf16(s.q, w.q_bias, T, Hq, stream);
    if (w.k_bias != nullptr) kernels::add_bias_bf16(s.k, w.k_bias, T, Hkv, stream);
    if (w.v_bias != nullptr) kernels::add_bias_bf16(s.v, w.v_bias, T, Hkv, stream);
    // Qwen3 / OLMo-3: per-head q/k RMSNorm before RoPE (skipped when null).
    if (w.q_norm != nullptr && w.k_norm != nullptr) {
        kernels::qk_norm_bf16(s.q, s.k, w.q_norm, w.k_norm, T, n_q_heads, n_kv_heads, head_dim,
                              rms_eps, stream);
    }
    kernels::rope_bf16(s.q, s.k, positions, T, n_q_heads, n_kv_heads, head_dim,
                       rope_theta, /*interleaved=*/false, stream);
    // KV append via the paged scatter — general over multi-page / multi-request
    // (uses the same CSR page lists attention reads below).
    kernels::write_kv_to_pages_bf16(k_pages, v_pages, s.k, s.v, qo_indptr, kv_page_indices,
                                    kv_page_indptr, kv_last_page_lens, T, num_requests, page_size,
                                    n_kv_heads, head_dim, /*hnd_layout=*/false, stream);
    // Tensor-core paged attention (decode/prefill plan), or naive fallback when
    // attn.ws==nullptr (head_dim the TC templates don't instantiate).
    run_attention_layer(attn, s.q, k_pages, v_pages, s.attn, qo_indptr, kv_page_indices,
                         kv_page_indptr, kv_last_page_lens, stream,
                         window_left, logits_soft_cap, sm_scale, nullptr);
    ops::gemm_act_x_wt_bf16(cublas, s.attn, w.wo, s.o, T, H, Hq, 0.f);
    kernels::residual_add_bf16(hidden, s.o, (size_t)T * H, stream);

    // --- MLP block ---
    kernels::rmsnorm_bf16(hidden, w.ffn_norm, s.normed, T, H, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.w_gate, s.gate, T, I, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, s.normed, w.w_up, s.up, T, I, H, 0.f);
    kernels::swiglu_bf16(s.gate, s.up, s.mlp, T * I, stream);
    ops::gemm_act_x_wt_bf16(cublas, s.mlp, w.w_down, s.mlp_out, T, H, I, 0.f);
    kernels::residual_add_bf16(hidden, s.mlp_out, (size_t)T * H, stream);
}

cudaError_t llama_layer_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const LlamaLayerWeights& w, const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    int num_tokens, int num_requests, int hidden_size,
    int n_q_heads, int n_kv_heads, int head_dim, int intermediate,
    int page_size, float rms_eps, float rope_theta) {
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
    // Standalone single-layer entry keeps the naive attention path (no plan):
    // an AttnPlan with ws==nullptr but dims populated routes run_attention_layer
    // to the naive fallback.
    AttnPlan naive_plan;
    naive_plan.num_tokens = num_tokens;
    naive_plan.num_requests = num_requests;
    naive_plan.n_q_heads = n_q_heads;
    naive_plan.n_kv_heads = n_kv_heads;
    naive_plan.head_dim = head_dim;
    naive_plan.page_size = page_size;
    decoder_layer_inplace(cublas, stream, hidden, w, positions, k_pages, v_pages, qo_indptr,
                          kv_page_indices, kv_page_indptr, kv_last_page_lens, s, num_tokens,
                          num_requests, hidden_size, n_q_heads, n_kv_heads, head_dim,
                          intermediate, page_size, rms_eps, rope_theta, naive_plan);
    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < 10; ++i) cudaFree(b[i]);
    return e;
}

}  // namespace pie_cuda_device::forward

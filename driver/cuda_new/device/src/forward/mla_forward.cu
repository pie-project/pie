#include "mla_forward.cuh"

#include "../kernels/argmax.cuh"
#include "../kernels/embed.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../ops/gemm.cuh"

#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

cudaError_t mla_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const MlaForwardWeights& w,
    const std::int32_t* positions,
    void* ckv_pages, void* kpe_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, const MlaForwardDims& dims) {
    const int T = num_tokens;
    const int H = dims.hidden;
    const int V = dims.vocab;
    constexpr std::size_t es = sizeof(std::uint16_t);  // bf16

    // Per-layer cache stride (in elements). ckv_pages / kpe_pages are single
    // contiguous allocations; layer L starts at L * <layer stride>. Matches
    // llama_forward's L*(num_pages*page_size*Hkv) offsetting.
    const std::size_t ckv_layer_stride_elems =
        (std::size_t)dims.num_pages * dims.page_size * dims.kv_lora_rank;
    const std::size_t kpe_layer_stride_elems =
        (std::size_t)dims.num_pages * dims.page_size * dims.qk_rope_head_dim;

    // Residual-stream scratch. mla_block_bf16 runs IN PLACE on `hidden` and
    // allocates its own per-block activation scratch, so we only need the
    // residual buffer + the final-norm output buffer here.
    void* hidden = nullptr;
    void* normed = nullptr;
    if (cudaError_t e = cudaMalloc(&hidden, (std::size_t)T * H * es); e != cudaSuccess)
        return e;
    if (cudaError_t e = cudaMalloc(&normed, (std::size_t)T * H * es); e != cudaSuccess) {
        cudaFree(hidden);
        return e;
    }

    // embed: token_ids -> residual stream (no scale for MLA, unlike Gemma).
    kernels::embed_bf16(token_ids, w.embed, hidden, T, H, V, stream);

    // N MLA decoder layers, each on its own slice of the per-layer paged cache.
    cudaError_t rc = cudaSuccess;
    for (int L = 0; L < dims.n_layers && rc == cudaSuccess; ++L) {
        void* ckv_L = static_cast<char*>(ckv_pages) +
                      (std::size_t)L * ckv_layer_stride_elems * es;
        void* kpe_L = static_cast<char*>(kpe_pages) +
                      (std::size_t)L * kpe_layer_stride_elems * es;
        rc = mla_block_bf16(
            cublas, stream, hidden, w.layers[L], positions, ckv_L, kpe_L,
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            T, num_requests, H, dims.num_heads, dims.q_lora_rank,
            dims.kv_lora_rank, dims.qk_nope_head_dim, dims.qk_rope_head_dim,
            dims.v_head_dim, dims.page_size, dims.rms_eps, dims.sm_scale,
            dims.rope_theta);
    }
    if (rc != cudaSuccess) {
        cudaFree(hidden);
        cudaFree(normed);
        return rc;
    }

    // final norm -> lm_head -> greedy argmax.
    kernels::rmsnorm_bf16(hidden, w.final_norm, normed, T, H, dims.rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, normed, w.lm_head, out_logits, T, V, H, 0.f);
    kernels::argmax_bf16(out_logits, out_token_ids, T, V, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    cudaFree(hidden);
    cudaFree(normed);
    return e;
}

}  // namespace pie_cuda_device::forward

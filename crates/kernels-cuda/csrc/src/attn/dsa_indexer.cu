// The host launchers, and nothing else. All three `__global__`s and the RoPE
// helper live in `attn/dsa_indexer.cuh` -- ONE definition, read by nvcc here
// and by NVRTC from the same text at run time.
//
// Only `index_knorm_rope` has a row; the other two are here for the AOT path
// and are waiting on a launch rule. `attn/dsa_indexer.cuh` says which rule
// each of them wants and why the ported ones do not fit.
#include "attn/dsa_indexer.cuh"
#include "attn/dsa_indexer.hpp"


namespace pie_cuda_driver::kernels::attn {

void dsa_index_knorm_rope_bf16(
    void* idx_k, const void* k_norm_weight, const void* k_norm_bias,
    const device::i32* positions, int tokens, int head_dim, int rope_dim,
    float theta, float eps, cudaStream_t stream)
{
    using bf16 = ::pie_cuda_driver::kernels::device::bf16;
    if (tokens <= 0) return;
    device::index_knorm_rope<bf16><<<tokens, device::kBlock, 0, stream>>>(
        static_cast<bf16*>(idx_k),
        static_cast<const bf16*>(k_norm_weight),
        static_cast<const bf16*>(k_norm_bias),
        positions, head_dim, rope_dim, theta, eps);
}

void dsa_index_q_rope_bf16(
    void* idx_q, const device::i32* positions, int tokens, int n_heads,
    int head_dim, int rope_dim, float theta, cudaStream_t stream)
{
    using bf16 = ::pie_cuda_driver::kernels::device::bf16;
    if (tokens <= 0) return;
    int block = ((n_heads + 31) / 32) * 32;
    if (block < 32) block = 32;
    device::index_q_rope<bf16><<<tokens, block, 0, stream>>>(
        static_cast<bf16*>(idx_q), positions,
        n_heads, head_dim, rope_dim, theta);
}

void dsa_index_topk_mask(
    const void* idx_q, const void* idx_k, const void* idx_w,
    device::u8* mask, int tokens, int n_heads, int head_dim, int topk,
    cudaStream_t stream)
{
    using bf16 = ::pie_cuda_driver::kernels::device::bf16;
    if (tokens <= 0) return;
    const std::size_t smem = static_cast<std::size_t>(tokens) * sizeof(float);
    device::index_topk_mask<bf16><<<tokens, device::kBlock, smem, stream>>>(
        static_cast<const bf16*>(idx_q),
        static_cast<const bf16*>(idx_k),
        static_cast<const bf16*>(idx_w),
        mask, tokens, n_heads, head_dim, topk);
}

}  // namespace pie_cuda_driver::kernels::attn

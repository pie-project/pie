#include "kernels/mla_paged.hpp"

#include <cuda_bf16.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels {
namespace {

__device__ __forceinline__ int find_request(const std::uint32_t* qo_indptr,
                                            int R,
                                            int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

__device__ __forceinline__ void resolve_dst(
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int token_idx,
    int& actual_page,
    int& offset_in_page)
{
    const int r = find_request(qo_indptr, R, token_idx);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = token_idx - qo_lo;
    const int pages_first = kv_page_indptr[r];
    const int pages_last = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after =
        (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;
    const int page_in_req = abs_kv_pos / page_size;
    offset_in_page = abs_kv_pos % page_size;
    actual_page = static_cast<int>(kv_page_indices[pages_first + page_in_req]);
}

__global__ void write_mla_kernel(
    const __nv_bfloat16* __restrict__ ckv_curr,
    const __nv_bfloat16* __restrict__ kpe_curr,
    __nv_bfloat16* __restrict__ ckv_pages,
    __nv_bfloat16* __restrict__ kpe_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim)
{
    const int t = blockIdx.x;
    int actual_page = 0;
    int offset_in_page = 0;
    resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                R, page_size, t, actual_page, offset_in_page);

    const long long ckv_src = static_cast<long long>(t) * kv_lora_rank;
    const long long ckv_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        kv_lora_rank;
    for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
        ckv_pages[ckv_dst + i] = ckv_curr[ckv_src + i];
    }

    const long long kpe_src = static_cast<long long>(t) * qk_rope_head_dim;
    const long long kpe_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        qk_rope_head_dim;
    for (int i = threadIdx.x; i < qk_rope_head_dim; i += blockDim.x) {
        kpe_pages[kpe_dst + i] = kpe_curr[kpe_src + i];
    }
}

}  // namespace

void launch_write_mla_to_pages_bf16(
    void* ckv_pages,
    void* kpe_pages,
    const void* ckv_curr,
    const void* kpe_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim,
    cudaStream_t stream)
{
    if (total_tokens <= 0) return;
    write_mla_kernel<<<total_tokens, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(ckv_curr),
        static_cast<const __nv_bfloat16*>(kpe_curr),
        static_cast<__nv_bfloat16*>(ckv_pages),
        static_cast<__nv_bfloat16*>(kpe_pages),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        num_requests, page_size, kv_lora_rank, qk_rope_head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void launch_write_mla_to_pages(
    MlaCacheLayerView layer,
    const void* ckv_curr,
    const void* kpe_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    cudaStream_t stream)
{
    launch_write_mla_to_pages_bf16(
        layer.ckv_pages, layer.kpe_pages, ckv_curr, kpe_curr,
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        total_tokens, num_requests, layer.page_size, layer.kv_lora_rank,
        layer.qk_rope_head_dim, stream);
}

}  // namespace pie_cuda_driver::kernels

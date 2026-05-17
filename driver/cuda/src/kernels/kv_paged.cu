#include "kernels/kv_paged.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// One block per current-step token. Threads stride over the (h_kv * head_dim)
// destination row.
//
// Linear scan to find the request index — `R` is small (≤ batch_size, which
// is bounded by max_forward_requests, ≤ a few hundred). A binary search would be
// nice, but the scan is fine in the M1.4 reference path.
__device__ __forceinline__ int find_request(const std::uint32_t* qo_indptr,
                                            int R, int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

__global__ void write_kv_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const __nv_bfloat16* __restrict__ v_curr,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int h_kv,
    int d)
{
    const int t = blockIdx.x;

    const int r = find_request(qo_indptr, R, t);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = t - qo_lo;

    const int pages_first = kv_page_indptr[r];
    const int pages_last  = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after = (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;

    const int page_in_req     = abs_kv_pos / page_size;
    const int offset_in_page  = abs_kv_pos % page_size;
    const int actual_page     = static_cast<int>(kv_page_indices[pages_first + page_in_req]);

    const long long row = h_kv * d;
    const long long src = static_cast<long long>(t) * row;
    const long long dst =
        ((static_cast<long long>(actual_page) * page_size) + offset_in_page) * row;

    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        k_pages[dst + i] = k_curr[src + i];
        v_pages[dst + i] = v_curr[src + i];
    }
}

__global__ void write_kv_decode_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const __nv_bfloat16* __restrict__ v_curr,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int page_size,
    int h_kv,
    int d)
{
    const int r = blockIdx.x;
    const int pages_first = kv_page_indptr[r];
    const int pages_last  = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int abs_kv_pos =
        (num_pages_r - 1) * page_size + static_cast<int>(kv_last_page_lens[r]) - 1;
    const int page_in_req = abs_kv_pos / page_size;
    const int offset_in_page = abs_kv_pos % page_size;
    const int actual_page =
        static_cast<int>(kv_page_indices[pages_first + page_in_req]);

    const long long row = h_kv * d;
    const long long src = static_cast<long long>(r) * row;
    const long long dst =
        ((static_cast<long long>(actual_page) * page_size) + offset_in_page) * row;

    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        k_pages[dst + i] = k_curr[src + i];
        v_pages[dst + i] = v_curr[src + i];
    }
}

}  // namespace

void launch_write_kv_to_pages_bf16(
    void* k_pages, void* v_pages,
    const void* k_curr, const void* v_curr,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    write_kv_kernel<<<total_tokens, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k_curr),
        static_cast<const __nv_bfloat16*>(v_curr),
        static_cast<__nv_bfloat16*>(k_pages),
        static_cast<__nv_bfloat16*>(v_pages),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        num_requests, page_size, num_kv_heads, head_dim);
}

void launch_write_kv_decode_to_pages_bf16(
    void* k_pages, void* v_pages,
    const void* k_curr, const void* v_curr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    write_kv_decode_kernel<<<num_requests, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k_curr),
        static_cast<const __nv_bfloat16*>(v_curr),
        static_cast<__nv_bfloat16*>(k_pages),
        static_cast<__nv_bfloat16*>(v_pages),
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        page_size, num_kv_heads, head_dim);
}

}  // namespace pie_cuda_driver::kernels

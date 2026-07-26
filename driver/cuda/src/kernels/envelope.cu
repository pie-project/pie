#include "kernels/envelope.hpp"

#include <cstdint>

#include <cuda_bf16.h>
#include <math_constants.h>

namespace pie_cuda_driver::kernels {

namespace {

// The per-(page, kv_head) min/max reduction, shared by full recompute and the
// page-list update so both paths are literally the same numerics (the full
// recompute is what `test_envelope_dot` parity-checks).
__device__ inline void envelope_reduce_page(
    const __nv_bfloat16* __restrict__ k_pages,
    int page,
    int kh,
    int live,
    int page_size,
    int num_kv_heads,
    int head_dim,
    float* __restrict__ env_min,
    float* __restrict__ env_max)
{
    const long token_stride = static_cast<long>(num_kv_heads) * head_dim;
    const long page_base = static_cast<long>(page) * page_size * token_stride +
                           static_cast<long>(kh) * head_dim;
    const long env_base =
        (static_cast<long>(page) * num_kv_heads + kh) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float mn = CUDART_INF_F;
        float mx = -CUDART_INF_F;
        for (int t = 0; t < live; ++t) {
            const float v = __bfloat162float(
                k_pages[page_base + static_cast<long>(t) * token_stride + d]);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        env_min[env_base + d] = mn;
        env_max[env_base + d] = mx;
    }
}

// One block per (page, kv_head); threads stride over head_dim, each reducing its
// dims' min/max across the page's live tokens. Streaming reads of the NHD layout.
__global__ void envelope_recompute_kernel(
    const __nv_bfloat16* __restrict__ k_pages,
    const std::int32_t* __restrict__ page_live_lens,
    float* __restrict__ env_min,
    float* __restrict__ env_max,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    const int page = blockIdx.x;
    const int kh = blockIdx.y;
    envelope_reduce_page(k_pages, page, kh, page_live_lens[page], page_size,
                         num_kv_heads, head_dim, env_min, env_max);
}

// Refresh exactly the pages this fire appended to, deriving that set on-device
// from the same CSR arithmetic `write_kv_kernel` uses. A request's post-append
// length is `(pages-1)*page_size + last_page_len`, so its new tokens occupy
// `[total_after - qo_len, total_after)` and the touched pages are that span
// divided by `page_size`. Rescanning the whole page list instead would cost a
// full KV read per layer -- as much as attention itself.
//
// One block per (touched slot, kv_head), where the grid's x extent is the host's
// worst-case bound `ceil(total_tokens/page_size) + num_requests`; blocks past
// the true count exit. Pages are append-only, so recomputing a touched page in
// full gives the same answer an incremental merge would.
__global__ void envelope_update_appended_kernel(
    const __nv_bfloat16* __restrict__ k_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    float* __restrict__ env_min,
    float* __restrict__ env_max,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    const int slot = blockIdx.x;
    const int kh = blockIdx.y;

    // Walk requests accumulating their touched-page counts until `slot` lands
    // inside one. R is the batch size, so a linear scan beats the divergence a
    // binary search would add.
    int seen = 0;
    for (int r = 0; r < num_requests; ++r) {
        const int pages_first = static_cast<int>(kv_page_indptr[r]);
        const int pages_last = static_cast<int>(kv_page_indptr[r + 1]);
        const int num_pages_r = pages_last - pages_first;
        if (num_pages_r <= 0) continue;

        const int qo_len =
            static_cast<int>(qo_indptr[r + 1]) - static_cast<int>(qo_indptr[r]);
        if (qo_len <= 0) continue;

        const int total_after =
            (num_pages_r - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
        const int pre_len = total_after - qo_len;
        if (total_after <= 0) continue;

        const int first_page = pre_len / page_size;
        const int last_page = (total_after - 1) / page_size;
        const int touched = last_page - first_page + 1;

        if (slot < seen + touched) {
            const int page_in_req = first_page + (slot - seen);
            if (page_in_req >= num_pages_r) return;
            const int live = (page_in_req == last_page)
                ? static_cast<int>(kv_last_page_lens[r])
                : page_size;
            if (live <= 0) return;
            envelope_reduce_page(
                k_pages,
                static_cast<int>(kv_page_indices[pages_first + page_in_req]),
                kh, live, page_size, num_kv_heads, head_dim, env_min, env_max);
            return;
        }
        seen += touched;
    }
}

// One block per (kv_head, page); threads reduce over the group·head_dim terms of
// `Σ max(q·min, q·max)`. Pages beyond `live_pages` are `-inf`.
template <int BLOCK>
__global__ void envelope_dot_kernel(
    const float* __restrict__ q,
    const float* __restrict__ env_min,
    const float* __restrict__ env_max,
    float* __restrict__ score,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int p_max,
    int live_pages)
{
    const int kh = blockIdx.y;
    const int p = blockIdx.x;
    float* out = &score[static_cast<long>(kh) * p_max + p];

    if (p >= live_pages) {
        if (threadIdx.x == 0) *out = -CUDART_INF_F;
        return;
    }

    const int group = num_q_heads / num_kv_heads;
    const long env_base =
        (static_cast<long>(p) * num_kv_heads + kh) * head_dim;
    const int terms = group * head_dim;

    float local = 0.f;
    for (int i = threadIdx.x; i < terms; i += BLOCK) {
        const int g = i / head_dim;
        const int d = i - g * head_dim;
        const int qh = kh * group + g;
        const float qd = q[static_cast<long>(qh) * head_dim + d];
        const float lo = qd * env_min[env_base + d];
        const float hi = qd * env_max[env_base + d];
        local += (lo > hi) ? lo : hi;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = buf[0];
}

}  // namespace

void launch_envelope_recompute_bf16(
    const std::uint16_t* k_pages,
    const std::int32_t* page_live_lens,
    float* env_min,
    float* env_max,
    int num_pages,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_pages <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    const dim3 grid(static_cast<unsigned>(num_pages),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    envelope_recompute_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_pages),
        page_live_lens, env_min, env_max,
        page_size, num_kv_heads, head_dim);
}

void launch_envelope_dot_f32(
    const float* q,
    const float* env_min,
    const float* env_max,
    float* score,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int p_max,
    int live_pages,
    cudaStream_t stream)
{
    if (p_max <= 0 || num_kv_heads <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(static_cast<unsigned>(p_max),
                    static_cast<unsigned>(num_kv_heads));
    envelope_dot_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        q, env_min, env_max, score,
        num_q_heads, num_kv_heads, head_dim, p_max, live_pages);
}

void launch_envelope_update_appended_bf16(
    const std::uint16_t* k_pages,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    float* env_min,
    float* env_max,
    int num_requests,
    int max_touched,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_requests <= 0 || max_touched <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(max_touched),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    envelope_update_appended_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_pages),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        env_min, env_max,
        num_requests, page_size, num_kv_heads, head_dim);
}

}  // namespace pie_cuda_driver::kernels

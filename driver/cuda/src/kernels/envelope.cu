#include "kernels/envelope.hpp"

#include <cstdint>

#include <cuda_bf16.h>
#include <math_constants.h>

namespace pie_cuda_driver::kernels {

namespace {

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
    const int live = page_live_lens[page];
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

}  // namespace pie_cuda_driver::kernels

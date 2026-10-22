#pragma once

#include "prim/device.cuh"

namespace pie::layout {

__device__ __forceinline__ float envelope_dot_thread_partial(
    const float* __restrict__ q,
    const bf16* __restrict__ env_min,
    const bf16* __restrict__ env_max,
    long env_base,
    int qh_base,
    int group,
    int head_dim,
    int thread,
    int nthreads) {
    float local = 0.f;
    const int terms = group * head_dim;
    for (int i = thread; i < terms; i += nthreads) {
        const int g = i / head_dim;
        const int d = i - g * head_dim;
        const float qd = q[static_cast<long>(qh_base + g) * head_dim + d];
        const float lo = qd * bf16_to_f32(env_min[env_base + d]);
        const float hi = qd * bf16_to_f32(env_max[env_base + d]);
        local += (lo > hi) ? lo : hi;
    }
    return local;
}

template <class T>
__device__ inline void reduce_page(
    const T* __restrict__ k_pages,
    int page,
    int kh,
    int live,
    int page_size,
    int num_kv_heads,
    int head_dim,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max)
{
    const long token_stride = static_cast<long>(num_kv_heads) * head_dim;
    const long page_base = static_cast<long>(page) * page_size * token_stride +
                           static_cast<long>(kh) * head_dim;
    const long env_base =
        (static_cast<long>(page) * num_kv_heads + kh) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float mn = pos_inf();
        float mx = -pos_inf();
        for (int t = 0; t < live; ++t) {
            const float v = Elem<T>::to_f32(
                k_pages[page_base + static_cast<long>(t) * token_stride + d]);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        env_min[env_base + d] = f32_to_bf16_rd(mn);
        env_max[env_base + d] = f32_to_bf16_ru(mx);
    }
}

template <class T>
__global__ void update_appended(
    const T* __restrict__ k_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim,
    const u32* __restrict__ win)
{
    const int slot = blockIdx.x;
    const int kh = blockIdx.y;

    if (win != nullptr && static_cast<int>(win[2]) < num_requests) num_requests = static_cast<int>(win[2]);

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
            reduce_page(
                k_pages,
                static_cast<int>(kv_page_indices[pages_first + page_in_req]),
                kh, live, page_size, num_kv_heads, head_dim, env_min, env_max);
            return;
        }
        seen += touched;
    }
}

template <int BLOCK>
__global__ void dot(
    const float* __restrict__ q,
    const bf16* __restrict__ env_min,
    const bf16* __restrict__ env_max,
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
        if (threadIdx.x == 0) *out = -pos_inf();
        return;
    }

    const int group = num_q_heads / num_kv_heads;
    const long env_base =
        (static_cast<long>(p) * num_kv_heads + kh) * head_dim;

    const float local = envelope_dot_thread_partial(
        q, env_min, env_max, env_base, kh * group, group, head_dim,
        static_cast<int>(threadIdx.x), BLOCK);

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = buf[0];
}

constexpr int kEnvelopeFuseMaxTokens = 128;

__device__ inline void atomic_min(bf16* addr, float value) {
    unsigned short* as_u16 = reinterpret_cast<unsigned short*>(addr);
    const unsigned short want = bf16_as_u16(f32_to_bf16_rd(value));
    unsigned short old = *as_u16;
    unsigned short assumed;
    do {
        if (bf16_to_f32(u16_as_bf16(old)) <= value) return;
        assumed = old;
        old = atomicCAS(as_u16, assumed, want);
    } while (assumed != old);
}

__device__ inline void atomic_max(bf16* addr, float value) {
    unsigned short* as_u16 = reinterpret_cast<unsigned short*>(addr);
    const unsigned short want = bf16_as_u16(f32_to_bf16_ru(value));
    unsigned short old = *as_u16;
    unsigned short assumed;
    do {
        if (bf16_to_f32(u16_as_bf16(old)) >= value) return;
        assumed = old;
        old = atomicCAS(as_u16, assumed, want);
    } while (assumed != old);
}

}

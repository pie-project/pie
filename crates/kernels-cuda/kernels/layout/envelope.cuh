#pragma once

#include "prelude/device.cuh"

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

template <int Tu = 0>
__global__ void recompute(
    const bf16* __restrict__ k_pages,
    const i32* __restrict__ page_live_lens,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    const int page = blockIdx.x;
    const int kh = blockIdx.y;
    reduce_page(k_pages, page, kh, page_live_lens[page], page_size,
                         num_kv_heads, head_dim, env_min, env_max);
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
    int head_dim)
{
    const int slot = blockIdx.x;
    const int kh = blockIdx.y;

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

template <int Tu = 0>
__global__ void seed_empty(
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    usize n)
{
    const usize i =
        static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    env_min[i] = f32_to_bf16(pos_inf());
    env_max[i] = f32_to_bf16(neg_inf());
}

constexpr int kEnvelopeFuseMaxTokens = 128;

template <int Tu = 0>
__global__ void merge_written_fused(
    const bf16* __restrict__ k_curr,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    __shared__ int s_mine[kEnvelopeFuseMaxTokens];
    __shared__ int s_count;
    __shared__ int s_started;
    __shared__ int s_taken;

    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;

    if (threadIdx.x == 0) {
        s_count = 0;
        s_started = 0;
        s_taken = 0;
    }
    __syncthreads();

    if (row_valid != nullptr && row_valid[token] == 0) return;
    const u32 page = w_page[token];

    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        if (row_valid != nullptr && row_valid[t] == 0) continue;
        if (w_page[t] != page) continue;
        if (t < token) {
            atomicOr(&s_taken, 1);
            continue;
        }
        s_mine[atomicAdd(&s_count, 1)] = t;
        if (w_off[t] == 0u) atomicOr(&s_started, 1);
    }
    __syncthreads();
    if (s_taken != 0) return;

    const long env_base =
        (static_cast<long>(page) * num_kv_heads + kh) * head_dim;
    const int count = s_count;
    const bool started = s_started != 0;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float lo = pos_inf();
        float hi = -pos_inf();
        for (int i = 0; i < count; ++i) {
            const long src =
                (static_cast<long>(s_mine[i]) * num_kv_heads + kh) * head_dim;
            const float v = bf16_to_f32(k_curr[src + d]);
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }

        if (!started) {
            lo = fminf(lo, bf16_to_f32(env_min[env_base + d]));
            hi = fmaxf(hi, bf16_to_f32(env_max[env_base + d]));
        }
        env_min[env_base + d] = f32_to_bf16_rd(lo);
        env_max[env_base + d] = f32_to_bf16_ru(hi);
    }
}

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

template <int Tu = 0>
__global__ void reset_started_pages(
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;
    if (row_valid != nullptr && row_valid[token] == 0) return;
    if (w_off[token] != 0u) return;

    const long env_base =
        (static_cast<long>(w_page[token]) * num_kv_heads + kh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        env_min[env_base + d] = f32_to_bf16(pos_inf());
        env_max[env_base + d] = f32_to_bf16(-pos_inf());
    }
}

template <int Tu = 0>
__global__ void merge_written(
    const bf16* __restrict__ k_curr,
    const u32* __restrict__ w_page,
    const u8* __restrict__ row_valid,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;
    if (row_valid != nullptr && row_valid[token] == 0) return;

    const long src_base =
        (static_cast<long>(token) * num_kv_heads + kh) * head_dim;
    const long env_base =
        (static_cast<long>(w_page[token]) * num_kv_heads + kh) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = bf16_to_f32(k_curr[src_base + d]);
        atomic_min(&env_min[env_base + d], v);
        atomic_max(&env_max[env_base + d], v);
    }
}

}

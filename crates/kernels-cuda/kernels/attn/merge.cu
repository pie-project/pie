#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include "prim/device.cuh"
#include "attn/fa2/cascade.cuh"

namespace pie::attn {

namespace merge_lse {

using DTypeIn = __nv_bfloat16;
using DTypeO = __nv_bfloat16;

using IdType = std::int32_t;

}

// The split kv's partial states folded one warp a (row, head): each lane
// holds HEAD_DIM / 32 columns, reads every partial's base-2 log-sum-exp and
// its normalized output straight from global memory, and lands the row's
// output and log-sum-exp. A row of no partials lands zeros and -inf, one
// partial lands as it stands. `win` is the staged `(count, start)` seat.
// With `sinks`, each head's output is scaled as `sink_rescale` scales it,
// by 1 / (1 + exp(sink - lse)), from the f32 sum before it rounds. Rows lie
// HEAD_DIM apart; LIVE of their columns are folded (the rest no one's).
template <int HEAD_DIM, class S = float, int LIVE = HEAD_DIM>
__global__ void __launch_bounds__(256) merge_partials_warp(
    const merge_lse::DTypeIn* __restrict__ v,
    const float* __restrict__ s,
    const merge_lse::IdType* __restrict__ indptr,
    merge_lse::DTypeO* __restrict__ v_merged,
    float* __restrict__ s_merged,
    u32 max_seq_len,
    const u32* __restrict__ seq_len_ptr,
    u32 num_heads,
    const S* __restrict__ sinks,
    const u32* __restrict__ win)
{
    static_assert(LIVE % 64 == 0 && LIVE <= HEAD_DIM, "two columns a lane at least");
    constexpr int PER = LIVE / 32;
    const u32 staged = seq_len_ptr != nullptr ? *seq_len_ptr : max_seq_len;
    const u32 rows = (win != nullptr && win[0] < staged) ? win[0] : staged;
    const u32 row_base = win != nullptr ? win[1] : 0;
    const u32 lane = threadIdx.x & 31;
    const u32 warps = (gridDim.x * blockDim.x) >> 5;
    const u32 col = lane * PER;
    for (u32 pair = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; pair < rows * num_heads;
         pair += warps) {
        const u32 pos = pair / num_heads;
        const u32 head = pair % num_heads;
        const int begin = indptr[pos];
        const int end = indptr[pos + 1];
        float m = -::pie::attn::fa2::math::inf;
        for (int k = begin + static_cast<int>(lane); k < end; k += 32) {
            m = fmaxf(m, s[static_cast<long long>(k) * num_heads + head]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        }
        float acc[PER];
#pragma unroll
        for (int j = 0; j < PER; ++j) acc[j] = 0.f;
        float d = 0.f;
        if (m != -::pie::attn::fa2::math::inf) {
#pragma unroll 4
            for (int k = begin; k < end; ++k) {
                const long long at = static_cast<long long>(k) * num_heads + head;
                const float w = exp2f(s[at] - m);
                d += w;
                const merge_lse::DTypeIn* row = v + at * HEAD_DIM + col;
#pragma unroll
                for (int j = 0; j < PER; j += 2) {
                    const u32 two = *reinterpret_cast<const u32*>(row + j);
                    acc[j] += w * Elem<bf16>::to_f32(bf16(static_cast<unsigned short>(two)));
                    acc[j + 1] += w * Elem<bf16>::to_f32(bf16(static_cast<unsigned short>(two >> 16)));
                }
            }
        }
        float inv = d > 0.f ? 1.f / d : 0.f;
        if (sinks != nullptr && d > 0.f) {
            constexpr float kLn2 = 0.69314718055994530942f;
            const float lse = m + log2f(d);
            inv *= 1.0f / (1.0f + __expf(-(lse * kLn2 - Elem<S>::to_f32(sinks[head]))));
        }
        const long long out = (static_cast<long long>(row_base) + pos) * num_heads + head;
        merge_lse::DTypeO* dst = v_merged + out * HEAD_DIM + col;
#pragma unroll
        for (int j = 0; j < PER; j += 2) {
            const u32 lo = Elem<bf16>::from_f32(acc[j] * inv).raw;
            const u32 hi = Elem<bf16>::from_f32(acc[j + 1] * inv).raw;
            *reinterpret_cast<u32*>(dst + j) = lo | (hi << 16);
        }
        if (s_merged != nullptr && lane == 0) {
            s_merged[out] = d > 0.f ? m + log2f(d) : -::pie::attn::fa2::math::inf;
        }
    }
}

template <class T>
__global__ void logit_softcap(T* __restrict__ x, float cap, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv_cap = 1.f / cap;
    const float v = Elem<T>::to_f32(x[i]);
    x[i] = Elem<T>::from_f32(cap * tanhf(v * inv_cap));
}

// `logit_softcap` eight elements a thread, read and written 16 bytes at a
// time: the same value for every element.
template <class T>
__global__ void logit_softcap_vec8(T* __restrict__ x, float cap, usize runs) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= runs) return;
    const float inv_cap = 1.f / cap;
    uint4 w = reinterpret_cast<const uint4*>(x)[i];
    T* e = reinterpret_cast<T*>(&w);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        e[k] = Elem<T>::from_f32(cap * tanhf(Elem<T>::to_f32(e[k]) * inv_cap));
    }
    reinterpret_cast<uint4*>(x)[i] = w;
}

template <class T, class S = T>
__global__ void sink_rescale(
    T* __restrict__ o,
    const float* __restrict__ lse,
    const S* __restrict__ sinks,
    i32 N,
    i32 num_q_heads,
    i32 head_dim,
    const u32* __restrict__ win)
{
    const i32 t = static_cast<i32>(blockIdx.x);

    if (win != nullptr && t >= static_cast<i32>(win[0])) return;

    const i32 t_row = win != nullptr ? t + static_cast<i32>(win[1]) : t;
    const i32 h = static_cast<i32>(blockIdx.y);
    if (t >= N || h >= num_q_heads) return;

    constexpr float kLn2 = 0.69314718055994530942f;
    const float lse_val = lse[t_row * num_q_heads + h];
    const float sink = Elem<S>::to_f32(sinks[h]);
    float r;
    if (!isfinite(lse_val)) {

        r = 1.0f;
    } else {
        const float diff = lse_val * kLn2 - sink;
        r = 1.0f / (1.0f + __expf(-diff));
    }

    const i32 row_stride = num_q_heads * head_dim;
    T* row = o + static_cast<long long>(t_row) * row_stride + h * head_dim;
    for (i32 d = static_cast<i32>(threadIdx.x); d < head_dim;
         d += static_cast<i32>(blockDim.x)) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * r);
    }
}

template <class T>
__global__ void merge_lse_combine(
    const T* __restrict__ o1,
    const float* __restrict__ lse1,
    const T* __restrict__ o2,
    const float* __restrict__ lse2,
    T* __restrict__ o_out,
    float* __restrict__ lse_out,
    int num_heads,
    int head_dim,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y;

    const float l1 = lse1[n_row * num_heads + h];
    const float l2 = lse2[n_row * num_heads + h];

    if (!isfinite(l2)) {

        if (o1 != o_out) {
            const long long off = (static_cast<long long>(n_row) * num_heads + h) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                o_out[off + d] = o1[off + d];
            }
        }
        if (lse_out != nullptr && lse_out != lse1) {
            if (threadIdx.x == 0) lse_out[n_row * num_heads + h] = l1;
        }
        return;
    }

    if (!isfinite(l1)) {
        const long long off = (static_cast<long long>(n_row) * num_heads + h) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            o_out[off + d] = o2[off + d];
        }
        if (lse_out != nullptr) {
            if (threadIdx.x == 0) lse_out[n_row * num_heads + h] = l2;
        }
        return;
    }

    const float lse_max = fmaxf(l1, l2);
    const float w1 = exp2f(l1 - lse_max);
    const float w2 = exp2f(l2 - lse_max);
    const float inv_total = 1.0f / (w1 + w2);

    const long long off = (static_cast<long long>(n_row) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v1 = Elem<T>::to_f32(o1[off + d]);
        const float v2 = Elem<T>::to_f32(o2[off + d]);
        o_out[off + d] = Elem<T>::from_f32((v1 * w1 + v2 * w2) * inv_total);
    }

    if (lse_out != nullptr && threadIdx.x == 0) {
        lse_out[n_row * num_heads + h] = lse_max + log2f(w1 + w2);
    }
}

}

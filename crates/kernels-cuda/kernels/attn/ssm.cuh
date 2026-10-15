#pragma once

#include <cuda_bf16.h>

#include "prelude/device.cuh"
#include "prelude/mma.cuh"

namespace pie::attn {

using f32 = float;
using state_bf16 = __nv_bfloat16;

constexpr int gqa_smem_bv = 128;

__device__ __forceinline__ float silu_f(float z) {
    return z / (1.f + __expf(-z));
}

// Whether lane `r` folds this fire's boundary: a fire that buffers its rows
// for a later commit asks every arm not to write the state, and a
// predicated fire names the lanes that do.
__device__ __forceinline__ bool row_persists(
    const u8* __restrict__ mask, int r) {
    return mask == nullptr || mask[r] != 0;
}

__device__ __forceinline__ bool folds(
    bool write_state, const u8* __restrict__ mask, int r) {
    return write_state && row_persists(mask, r);
}

template <bool SILU, bool RESIDUAL>
__device__ __forceinline__ float conv_out(float acc, float x_t) {
    if constexpr (RESIDUAL) {
        return acc + x_t;
    } else if constexpr (SILU) {
        return silu_f(acc);
    } else {
        return acc;
    }
}

template <class T, bool SILU>
__global__ void ssm_causal_conv1d_chunked(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out,
    int N, int C, int K)
{
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (c >= C) return;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;

    for (int t = tid; t < N; t += block_size) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1) + k;
            float xv = 0.f;
            if (src_t < 0) {
                if (state_out) {
                    xv = Elem<T>::to_f32(state_out[(K + src_t) * C + c]);
                }
            } else {
                xv = Elem<T>::to_f32(x[src_t * C + c]);
            }
            const float wv = Elem<T>::to_f32(weight[c * K + k]);
            acc += wv * xv;
        }
        y[t * C + c] = Elem<T>::from_f32(SILU ? silu_f(acc) : acc);
    }

    __syncthreads();

    if (state_out && tid == 0) {
        for (int s = 0; s < K; ++s) {
            const int src_t = N - K + s;
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state_out[(K + src_t) * C + c])
                : Elem<T>::to_f32(x[src_t * C + c]);
            state_out[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

template <class T>
__global__ void ssm_causal_conv1d_update(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state,
    T* __restrict__ y,
    int C, int K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    const float new_x  = Elem<T>::to_f32(x[c]);

    float acc = bias_v;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K) break;
        float xv;
        if (k < K - 1) {
            xv = Elem<T>::to_f32(state[(k + 1) * C + c]);
        } else {
            xv = new_x;
        }
        const float wv = Elem<T>::to_f32(weight[c * K + k]);
        acc += wv * xv;
    }
    y[c] = Elem<T>::from_f32(silu_f(acc));

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K - 1) break;
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[(K - 1) * C + c] = Elem<T>::from_f32(new_x);
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_causal_conv1d_chunked_batched(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    (void)lanes_stated;
    const int c = blockIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;

    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;

    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;

    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > Nr) b = Nr;
        if (b > 0) { t0 += b; Nr -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[rl];
        if (c < Nr) Nr = c;
    }
    if (Nr <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const T* x_r = x + (long long)(t0 + row0) * C;
    T* y_r = y + (long long)(t0 + row0) * C;
    T* state = state_out_base + (long long)slot * slot_stride_elems;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;

    for (int t = tid; t < Nr; t += block_size) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1 - k) * dil;
            float xv = 0.f;
            if (src_t < 0) {
                xv = Elem<T>::to_f32(state[((K - 1) * dil + 1 + src_t) * C + c]);
            } else {
                xv = Elem<T>::to_f32(x_r[src_t * C + c]);
            }
            const float wv = Elem<T>::to_f32(weight[c * K + k]);
            acc += wv * xv;
        }
        y_r[t * C + c] = Elem<T>::from_f32(
            conv_out<SILU, RESIDUAL>(acc, Elem<T>::to_f32(x_r[t * C + c])));
    }

    __syncthreads();

    if (state_out_base && write_state &&
        (write_state_mask == nullptr || write_state_mask[rl] != 0) &&
        tid == 0) {
        const int span = (K - 1) * dil + 1;
        for (int s = 0; s < span; ++s) {
            const int src_t = Nr - span + s;
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state[(span + src_t) * C + c])
                : Elem<T>::to_f32(x_r[src_t * C + c]);
            state[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_causal_conv1d_chunked_batched_channel_tile(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    (void)lanes_stated;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;

    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > Nr) b = Nr;
        if (b > 0) { t0 += b; Nr -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[rl];
        if (c < Nr) Nr = c;
    }
    if (Nr <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const T* x_r = x + static_cast<long long>(t0 + row0) * C;
    T* y_r = y + static_cast<long long>(t0 + row0) * C;
    T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    float wv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wv[k] = (k < K) ? Elem<T>::to_f32(weight[c * K + k]) : 0.f;
    }

    for (int t = 0; t < Nr; ++t) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k >= K) break;
            const int src_t = t - (K - 1 - k) * dil;
            float xv = 0.f;
            if (src_t < 0) {
                xv = Elem<T>::to_f32(state[((K - 1) * dil + 1 + src_t) * C + c]);
            } else {
                xv = Elem<T>::to_f32(x_r[src_t * C + c]);
            }
            acc += wv[k] * xv;
        }
        y_r[static_cast<long long>(t) * C + c] = Elem<T>::from_f32(
            conv_out<SILU, RESIDUAL>(acc, Elem<T>::to_f32(x_r[static_cast<long long>(t) * C + c])));
    }

    if (state_out_base && write_state &&
        (write_state_mask == nullptr || write_state_mask[rl] != 0)) {
        const int span = (K - 1) * dil + 1;
        for (int s = 0; s < span; ++s) {
            const int src_t = Nr - span + s;
            const float v = (src_t < 0)
                ? Elem<T>::to_f32(state[(span + src_t) * C + c])
                : Elem<T>::to_f32(x_r[src_t * C + c]);
            state[s * C + c] = Elem<T>::from_f32(v);
        }
    }
}

constexpr int CONV_TT = 32;

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_causal_conv1d_chunked_batched_row_tile(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(qo_indptr[lanes]);
    const int tile0 = static_cast<int>(blockIdx.y) * CONV_TT;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + CONV_TT, total);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int span = (K - 1) * dil + 1;
    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    float wv[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        wv[k] = (k < K) ? Elem<T>::to_f32(weight[c * K + k]) : 0.f;
    }
    int r = 0;
    {
        int lo = 0, hi = lanes - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (static_cast<int>(qo_indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
        }
        r = lo;
    }
    for (; r < lanes; ++r) {
        int t0 = static_cast<int>(qo_indptr[r]);
        int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
        if (t0 >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        if (begin_at != nullptr) {
            int b = begin_at[rl];
            if (b > Nr) b = Nr;
            if (b > 0) { t0 += b; Nr -= b; }
        }
        if (commit_len != nullptr) {
            const int cl = commit_len[rl];
            if (cl < Nr) Nr = cl;
        }
        if (Nr <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* x_r = x + static_cast<long long>(t0 + row0) * C;
        T* y_r = y + static_cast<long long>(t0 + row0) * C;
        T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems;
        const int t_lo = max(tile0, t0) - t0;
        const int t_hi = min(tile_end, t0 + Nr) - t0;
        for (int t = t_lo; t < t_hi; ++t) {
            float acc = bias_v;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (k >= K) break;
                const int src_t = t - (K - 1 - k) * dil;
                float xv = 0.f;
                if (src_t < 0) {
                    xv = Elem<T>::to_f32(state[(span + src_t) * C + c]);
                } else {
                    xv = Elem<T>::to_f32(x_r[static_cast<long long>(src_t) * C + c]);
                }
                acc += wv[k] * xv;
            }
            y_r[static_cast<long long>(t) * C + c] = Elem<T>::from_f32(
                conv_out<SILU, RESIDUAL>(acc, Elem<T>::to_f32(x_r[static_cast<long long>(t) * C + c])));
        }
    }
}

// The row-tiled conv eight channels a thread for a four-tap, undilated
// kernel: a thread reads each of its rows as one 16-byte vector, carries the
// three rows before the current one in registers, and reads a tile's rows
// eight at a time so their loads are in flight together.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_causal_conv1d_chunked_batched_row_tile_vec8(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    constexpr int TAPS = 4;
    constexpr int VEC = 8;
    constexpr int RUN = 8;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(qo_indptr[lanes]);
    const int tile0 = static_cast<int>(blockIdx.y) * CONV_TT;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + CONV_TT, total);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    auto unpack8 = [](const uint4& raw, float (&f)[VEC]) {
        const T* e = reinterpret_cast<const T*>(&raw);
        #pragma unroll
        for (int u = 0; u < VEC; ++u) f[u] = Elem<T>::to_f32(e[u]);
    };
    float wv[TAPS][VEC];
    {
        const uint4* wq = reinterpret_cast<const uint4*>(weight + static_cast<long long>(c) * TAPS);
        #pragma unroll
        for (int q = 0; q < TAPS; ++q) {
            float f[VEC];
            unpack8(wq[q], f);
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const int flat = q * VEC + e;
                wv[flat % TAPS][flat / TAPS] = f[e];
            }
        }
    }
    float bias_v[VEC];
    #pragma unroll
    for (int u = 0; u < VEC; ++u) bias_v[u] = bias ? Elem<T>::to_f32(bias[c + u]) : 0.f;
    int r = 0;
    {
        int lo = 0, hi = lanes - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (static_cast<int>(qo_indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
        }
        r = lo;
    }
    for (; r < lanes; ++r) {
        int t0 = static_cast<int>(qo_indptr[r]);
        int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
        if (t0 >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        if (begin_at != nullptr) {
            int b = begin_at[rl];
            if (b > Nr) b = Nr;
            if (b > 0) { t0 += b; Nr -= b; }
        }
        if (commit_len != nullptr) {
            const int cl = commit_len[rl];
            if (cl < Nr) Nr = cl;
        }
        if (Nr <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* x_r = x + static_cast<long long>(t0 + row0) * C + c;
        T* y_r = y + static_cast<long long>(t0 + row0) * C + c;
        const T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems + c;
        const int t_lo = max(tile0, t0) - t0;
        const int t_hi = min(tile_end, t0 + Nr) - t0;
        if (t_lo >= t_hi) continue;
        float hist[TAPS - 1][VEC];
        #pragma unroll
        for (int j = 0; j < TAPS - 1; ++j) {
            const int src = t_lo - (TAPS - 1) + j;
            const uint4 raw = src < 0
                ? *reinterpret_cast<const uint4*>(state + static_cast<long long>(TAPS + src) * C)
                : *reinterpret_cast<const uint4*>(x_r + static_cast<long long>(src) * C);
            unpack8(raw, hist[j]);
        }
        for (int tb = t_lo; tb < t_hi; tb += RUN) {
            uint4 run[RUN];
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = min(tb + j, t_hi - 1);
                run[j] = *reinterpret_cast<const uint4*>(x_r + static_cast<long long>(t) * C);
            }
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = tb + j;
                if (t >= t_hi) break;
                float cur[VEC];
                unpack8(run[j], cur);
                T out[VEC];
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    float acc = bias_v[u];
                    acc += wv[0][u] * hist[0][u];
                    acc += wv[1][u] * hist[1][u];
                    acc += wv[2][u] * hist[2][u];
                    acc += wv[3][u] * cur[u];
                    out[u] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc, cur[u]));
                    hist[0][u] = hist[1][u];
                    hist[1][u] = hist[2][u];
                    hist[2][u] = cur[u];
                }
                *reinterpret_cast<uint4*>(y_r + static_cast<long long>(t) * C) =
                    *reinterpret_cast<const uint4*>(out);
            }
        }
    }
}

// The row-tiled conv four channels a thread (one 8-byte vector a row) for
// a four-tap, undilated kernel: the 16-byte form's registers held a block
// to three an SM on sm_100, where four channels and a 64-register bound
// keep eight resident and the SiLU's divide hides under the loads. The sums
// run in the 16-byte form's order, so the two land the same bits.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void __launch_bounds__(128, 8) ssm_causal_conv1d_chunked_batched_row_tile_vec4(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil, bool write_state,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    using Run = uint2;
    constexpr int TAPS = 4;
    constexpr int VEC = 4;
    constexpr int RUN = 8;
    constexpr int TT = CONV_TT;
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(qo_indptr[lanes]);
    const int tile0 = static_cast<int>(blockIdx.y) * TT;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + TT, total);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    // The taps, channel-major: element (u * TAPS + q) is channel c + u's tap q.
    T wraw[VEC * TAPS];
    {
        const Run* wq = reinterpret_cast<const Run*>(weight + static_cast<long long>(c) * TAPS);
        #pragma unroll
        for (int q = 0; q < TAPS; ++q) reinterpret_cast<Run*>(wraw)[q] = wq[q];
    }
    float wf[VEC * TAPS];
    #pragma unroll
    for (int i = 0; i < VEC * TAPS; ++i) wf[i] = Elem<T>::to_f32(wraw[i]);
    auto wt = [&](int q, int u) -> float { return wf[u * TAPS + q]; };
    float bias_v[VEC];
    #pragma unroll
    for (int u = 0; u < VEC; ++u) bias_v[u] = bias ? Elem<T>::to_f32(bias[c + u]) : 0.f;
    int r = 0;
    {
        int lo = 0, hi = lanes - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (static_cast<int>(qo_indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
        }
        r = lo;
    }
    for (; r < lanes; ++r) {
        int t0 = static_cast<int>(qo_indptr[r]);
        int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
        if (t0 >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        if (begin_at != nullptr) {
            int b = begin_at[rl];
            if (b > Nr) b = Nr;
            if (b > 0) { t0 += b; Nr -= b; }
        }
        if (commit_len != nullptr) {
            const int cl = commit_len[rl];
            if (cl < Nr) Nr = cl;
        }
        if (Nr <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* x_r = x + static_cast<long long>(t0 + row0) * C + c;
        T* y_r = y + static_cast<long long>(t0 + row0) * C + c;
        const T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems + c;
        const int t_lo = max(tile0, t0) - t0;
        const int t_hi = min(tile_end, t0 + Nr) - t0;
        if (t_lo >= t_hi) continue;
        Run hist[TAPS - 1];
        #pragma unroll
        for (int j = 0; j < TAPS - 1; ++j) {
            const int src = t_lo - (TAPS - 1) + j;
            hist[j] = src < 0
                ? *reinterpret_cast<const Run*>(state + static_cast<long long>(TAPS + src) * C)
                : *reinterpret_cast<const Run*>(x_r + static_cast<long long>(src) * C);
        }
        for (int tb = t_lo; tb < t_hi; tb += RUN) {
            Run run[RUN];
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = min(tb + j, t_hi - 1);
                run[j] = *reinterpret_cast<const Run*>(x_r + static_cast<long long>(t) * C);
            }
            #pragma unroll
            for (int j = 0; j < RUN; ++j) {
                const int t = tb + j;
                if (t >= t_hi) break;
                const T* h0 = reinterpret_cast<const T*>(&hist[0]);
                const T* h1 = reinterpret_cast<const T*>(&hist[1]);
                const T* h2 = reinterpret_cast<const T*>(&hist[2]);
                const T* cu = reinterpret_cast<const T*>(&run[j]);
                T out[VEC];
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    const float cur = Elem<T>::to_f32(cu[u]);
                    float acc = bias_v[u];
                    acc += wt(0, u) * Elem<T>::to_f32(h0[u]);
                    acc += wt(1, u) * Elem<T>::to_f32(h1[u]);
                    acc += wt(2, u) * Elem<T>::to_f32(h2[u]);
                    acc += wt(3, u) * cur;
                    out[u] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc, cur));
                }
                hist[0] = hist[1];
                hist[1] = hist[2];
                hist[2] = run[j];
                *reinterpret_cast<Run*>(y_r + static_cast<long long>(t) * C) =
                    *reinterpret_cast<const Run*>(out);
            }
        }
    }
}


template <class T>
__global__ void ssm_causal_conv1d_state_commit(
    const T* __restrict__ x,
    T* __restrict__ state_out_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int C, int K, int dil,
    const u8* __restrict__ write_state_mask,
    const int* commit_len,
    const int* begin_at,
    const u32* __restrict__ win,
    int lanes_stated)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = static_cast<int>(blockIdx.y);
    if (r >= lanes) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    if (write_state_mask != nullptr && write_state_mask[rl] == 0) return;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > Nr) b = Nr;
        if (b > 0) { t0 += b; Nr -= b; }
    }
    if (commit_len != nullptr) {
        const int cl = commit_len[rl];
        if (cl < Nr) Nr = cl;
    }
    if (Nr <= 0) return;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int span = (K - 1) * dil + 1;
    const T* x_r = x + static_cast<long long>(t0 + row0) * C;
    T* state = state_out_base + static_cast<long long>(slot) * slot_stride_elems;
    for (int s = 0; s < span; ++s) {
        const int src_t = Nr - span + s;
        const float v = (src_t < 0)
            ? Elem<T>::to_f32(state[(span + src_t) * C + c])
            : Elem<T>::to_f32(x_r[static_cast<long long>(src_t) * C + c]);
        state[s * C + c] = Elem<T>::from_f32(v);
    }
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_causal_conv1d_update_batched(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    T* __restrict__ y,
    int R, int C, int K, int dil,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int r = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R || c >= C) return;

    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    T* state = state_base + (long long)slot * slot_stride_elems;
    const T* x_r = x + (long long)r_row * C;
    T* y_r = y + (long long)r_row * C;

    const float bias_v = bias ? Elem<T>::to_f32(bias[c]) : 0.f;
    const float new_x  = Elem<T>::to_f32(x_r[c]);

    float acc = bias_v;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= K) break;
        float xv;
        if (k < K - 1) {
            xv = Elem<T>::to_f32(state[(k * dil + 1) * C + c]);
        } else {
            xv = new_x;
        }
        const float wv = Elem<T>::to_f32(weight[c * K + k]);
        acc += wv * xv;
    }
    y_r[c] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc, new_x));

    if (!folds(write_state, write_state_mask, rl)) return;
    const int span = (K - 1) * dil;
    for (int k = 0; k < span; ++k) {
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[span * C + c] = Elem<T>::from_f32(new_x);
}

template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_causal_conv1d_update_batched_vec8(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    T* __restrict__ y,
    int R, int C, int K,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    constexpr int VEC = 8;
    constexpr int K_MAX = 8;
    union Vec { uint4 raw; T e[VEC]; };

    const int r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (r >= R || c0 >= C) return;

    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    T* state = state_base + (long long)slot * slot_stride_elems;
    const int span = K - 1;

    Vec xv;
    xv.raw = *reinterpret_cast<const uint4*>(x + (long long)r_row * C + c0);
    Vec bv;
    bv.raw = make_uint4(0u, 0u, 0u, 0u);
    if (bias != nullptr) {
        bv.raw = *reinterpret_cast<const uint4*>(bias + c0);
    }

    Vec wv[K_MAX];
    const uint4* w = reinterpret_cast<const uint4*>(weight + (long long)c0 * K);
    #pragma unroll
    for (int t = 0; t < K_MAX; ++t) {
        if (t < K) wv[t].raw = w[t];
    }
    Vec window[K_MAX - 1];
    #pragma unroll
    for (int k = 0; k < K_MAX - 1; ++k) {
        if (k < span) {
            window[k].raw = *reinterpret_cast<const uint4*>(state + (long long)(k + 1) * C + c0);
        }
    }
    const T* wflat = &wv[0].e[0];

    float acc[VEC];
    #pragma unroll
    for (int u = 0; u < VEC; ++u) {
        acc[u] = bias != nullptr ? Elem<T>::to_f32(bv.e[u]) : 0.f;
    }
    #pragma unroll
    for (int k = 0; k < K_MAX - 1; ++k) {
        if (k >= span) break;
        #pragma unroll
        for (int u = 0; u < VEC; ++u) {
            acc[u] += Elem<T>::to_f32(wflat[u * K + k]) * Elem<T>::to_f32(window[k].e[u]);
        }
    }
    #pragma unroll
    for (int u = 0; u < VEC; ++u) {
        acc[u] += Elem<T>::to_f32(wflat[u * K + span]) * Elem<T>::to_f32(xv.e[u]);
    }
    Vec yv;
    #pragma unroll
    for (int u = 0; u < VEC; ++u) {
        yv.e[u] = Elem<T>::from_f32(conv_out<SILU, RESIDUAL>(acc[u], Elem<T>::to_f32(xv.e[u])));
    }
    *reinterpret_cast<uint4*>(y + (long long)r_row * C + c0) = yv.raw;

    if (!folds(write_state, write_state_mask, rl)) return;
    #pragma unroll
    for (int k = 0; k < K_MAX - 1; ++k) {
        if (k < span) {
            *reinterpret_cast<uint4*>(state + (long long)k * C + c0) = window[k].raw;
        }
    }
    *reinterpret_cast<uint4*>(state + (long long)span * C + c0) = xv.raw;
}

template <class T>
__global__ void widen(
    const T* __restrict__ x, float* __restrict__ y, usize n)
{
    const usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) y[i] = Elem<T>::to_f32(x[i]);
}

template <class T>
__global__ void narrow(
    const float* __restrict__ x, T* __restrict__ y, usize n)
{
    const usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) y[i] = Elem<T>::from_f32(x[i]);
}

template <class T>
__global__ void repeat_interleave_heads_fp32(
    const T* __restrict__ in, T* __restrict__ out,
    int K_h, int V_h, int D, int repeat,
    const u32* __restrict__ win)
{
    const int n   = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int h_v = blockIdx.y;
    const int d   = threadIdx.x;
    if (h_v >= V_h || d >= D) return;
    const int h_k = h_v / repeat;
    const long long src = ((long long)n * K_h + h_k) * D + d;
    const long long dst = ((long long)n * V_h + h_v) * D + d;
    if (d < D) out[dst] = in[src];

    for (int dd = d + blockDim.x; dd < D; dd += blockDim.x) {
        out[((long long)n * V_h + h_v) * D + dd] =
            in[((long long)n * K_h + h_k) * D + dd];
    }
}

template <class T, int BLOCK>
__global__ void l2norm_scale(
    const T* __restrict__ x,
    float*               __restrict__ y,
    int hidden, float scale, float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + (long long)row * hidden;
    float*               yr = y + (long long)row * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = Elem<T>::to_f32(xr[i]) * inv * scale;
    }
}

template <class T>
__global__ void ssm_gdn_prep_g_beta(
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ A_log,
    const T* __restrict__ dt_bias,
    float*               __restrict__ g_log_out,
    float*               __restrict__ beta_out,
    int N, int V_h)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= N || h >= V_h) return;

    const float av  = Elem<T>::to_f32(a[(long long)t * V_h + h]);
    const float bv  = Elem<T>::to_f32(b[(long long)t * V_h + h]);
    const float Alh = A_log[h];
    const float dtb = Elem<T>::to_f32(dt_bias[h]);

    const float z = av + dtb;
    const float sp = (z > 20.f) ? z : log1pf(__expf(z));

    g_log_out[(long long)t * V_h + h] = -__expf(Alh) * sp;
    beta_out[(long long)t * V_h + h]  = 1.f / (1.f + __expf(-bv));
}

template <class T, int BLOCK>
__global__ void ssm_gdn_prep_qk_norm(
    const T* __restrict__ qkv_post,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int K_h, int K_d, int conv_dim,
    float q_scale,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const T* q_base =
        qkv_post + (long long)n_row * conv_dim + (long long)h * K_d;
    const T* k_base =
        qkv_post + (long long)n_row * conv_dim + K_dim + (long long)h * K_d;

    float q_sum = 0.f;
    float k_sum = 0.f;
    for (int i = tid; i < K_d; i += BLOCK) {
        const float qv = Elem<T>::to_f32(q_base[i]);
        const float kv = Elem<T>::to_f32(k_base[i]);
        q_sum += qv * qv;
        k_sum += kv * kv;
    }

    __shared__ float q_buf[BLOCK];
    __shared__ float k_buf[BLOCK];
    q_buf[tid] = q_sum;
    k_buf[tid] = k_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            q_buf[tid] += q_buf[tid + off];
            k_buf[tid] += k_buf[tid + off];
        }
        __syncthreads();
    }

    const float q_inv = rsqrtf(q_buf[0] + 1e-6f) * q_scale;
    const float k_inv = rsqrtf(k_buf[0] + 1e-6f);
    float* q_dst = q_out + ((long long)n * K_h + h) * K_d;
    float* k_dst = k_out + ((long long)n * K_h + h) * K_d;
    for (int i = tid; i < K_d; i += BLOCK) {
        q_dst[i] = Elem<T>::to_f32(q_base[i]) * q_inv;
        k_dst[i] = Elem<T>::to_f32(k_base[i]) * k_inv;
    }
}

template <class T, int BLOCK>
__global__ void ssm_gdn_prep_v_g_beta(
    const T* __restrict__ qkv_post,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ A_log,
    const T* __restrict__ dt_bias,
    float* __restrict__ v_out,
    float* __restrict__ g_log_out,
    float* __restrict__ beta_out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const T* v_base =
        qkv_post + (long long)n * conv_dim + 2 * K_dim + (long long)h * V_d;
    float* v_dst = v_out + ((long long)n * V_h + h) * V_d;
    for (int i = tid; i < V_d; i += BLOCK) {
        v_dst[i] = Elem<T>::to_f32(v_base[i]);
    }

    if (tid == 0) {
        const long long gh = (long long)n * V_h + h;
        const float av = Elem<T>::to_f32(a[gh]);
        const float bv = Elem<T>::to_f32(b[gh]);
        const float z = av + Elem<T>::to_f32(dt_bias[h]);
        const float sp = (z > 20.f) ? z : log1pf(__expf(z));
        g_log_out[gh] = -__expf(A_log[h]) * sp;
        beta_out[gh] = 1.f / (1.f + __expf(-bv));
    }
}

template <class T>
__global__ void ssm_gdn_prep_ba_gates(
    const T* __restrict__ ba,
    const float* __restrict__ A_log,
    const T* __restrict__ dt_bias,
    float* __restrict__ gates,
    int N, int V_h,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= N || h >= V_h) return;

    const T* row = ba + (long long)t_row * 2 * V_h;
    const float bv = Elem<T>::to_f32(row[h]);
    const float av = Elem<T>::to_f32(row[V_h + h]);

    const float z = av + Elem<T>::to_f32(dt_bias[h]);
    const float sp = (z > 20.f) ? z : log1pf(__expf(z));

    float* out = gates + (long long)t_row * 2 * V_h;
    out[h] = -__expf(A_log[h]) * sp;
    out[V_h + h] = 1.f / (1.f + __expf(-bv));
}

template <class T, int BLOCK>
__global__ void ssm_gdn_prep_v_gates(
    const T* __restrict__ qkv_post,
    const float* __restrict__ gates,
    float* __restrict__ v_out,
    float* __restrict__ g_log_out,
    float* __restrict__ beta_out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const T* v_base =
        qkv_post + (long long)n_row * conv_dim + 2 * K_dim + (long long)h * V_d;
    float* v_dst = v_out + ((long long)n * V_h + h) * V_d;
    for (int i = tid; i < V_d; i += BLOCK) {
        v_dst[i] = Elem<T>::to_f32(v_base[i]);
    }

    if (tid == 0) {
        const long long gh = (long long)n * V_h + h;
        const float* row = gates + (long long)n_row * 2 * V_h;
        g_log_out[gh] = row[h];
        beta_out[gh] = row[V_h + h];
    }
}

template <typename StateT>
__device__ __forceinline__ float state_load(const StateT* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float state_load<__nv_bfloat16>(
    const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

template <typename StateT>
__device__ __forceinline__ void state_store(StateT* p, float v) {
    *p = static_cast<StateT>(v);
}

template <>
__device__ __forceinline__ void state_store<__nv_bfloat16>(
    __nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}

// What a value reads back as once it has been through the state's cell: the
// unfolded step rounds where the folded one stores, so both answer alike.
template <typename StateT>
__device__ __forceinline__ float state_round(float v) {
    StateT cell;
    state_store(&cell, v);
    return state_load(&cell);
}

template <bool KLast>
__device__ __forceinline__ long long state_offset(
    int k_idx, int v_idx, int K_d, int V_d) {
    if constexpr (KLast) {
        return (long long)v_idx * K_d + k_idx;
    } else {
        return (long long)k_idx * V_d + v_idx;
    }
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    return __shfl_sync(0xffffffffu, x, 0);
}

template <typename StateT, bool KLast>
__global__ void ssm_gated_delta_step(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;

    const long long bh = (long long)b * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    state += bh * (long long)K_d * V_d;
    out   += bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out[v_idx] = out_v;
    }
}

template <typename StateT, bool KLast>
__global__ void ssm_gated_delta_chunked_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;

    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v      + bh * V_d;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];

        float* out_bh = out + ((long long)(t0 + t + row0) * V_h + h) * V_d;

        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h[i];
            sk[i] = k_h[i];
        }
        __syncthreads();

        for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
            float kv_mem = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = state_load(state + off) * g_h;
                state_store(state + off, s);
                kv_mem += s * sk[k_idx];
            }

            const float v_t   = v_h[v_idx];
            const float delta = (v_t - kv_mem) * beta_h;

            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = state_load(state + off) + sk[k_idx] * delta;
                state_store(state + off, s);
                out_v += s * sq[k_idx];
            }
            out_bh[v_idx] = out_v;
        }

        __syncthreads();
    }
}

template <typename StateT, bool KLast>
__global__ void ssm_gated_delta_chunked_batched_cached(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float s_state[];
    const int state_elems = K_d * V_d;
    for (int i = threadIdx.x; i < state_elems; i += blockDim.x) {
        s_state[i] = state_load(state + i);
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v      + bh * V_d;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];
        float* out_bh = out + bh * V_d;

        for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
            float kv_mem = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = s_state[off] * g_h;
                s_state[off] = s;
                kv_mem += s * k_h[k_idx];
            }

            const float delta = (v_h[v_idx] - kv_mem) * beta_h;
            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = s_state[off] + k_h[k_idx] * delta;
                s_state[off] = s;
                out_v += s * q_h[k_idx];
            }
            out_bh[v_idx] = out_v;
        }
    }

    if (write_state && row_persists(write_state_mask, r)) {
        __syncthreads();
        for (int i = threadIdx.x; i < state_elems; i += blockDim.x) {
            state_store(state + i, s_state[i]);
        }
    }
}

template <typename StateT, bool KLast>
__global__ void ssm_gated_delta_chunked_batched_warp_tiled_gqa(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    constexpr int WARPS = 4;
    constexpr int MAX_K_PER_LANE = 8;
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int v_tile = blockIdx.z * WARPS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v_idx = v_tile + warp;
    if (warp >= WARPS || v_idx >= V_d) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    const int repeat = V_h / K_h;
    const int qk_h = h / repeat;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s_vals[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s_vals[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v_idx, K_d, V_d));
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long qk_bh = ((long long)(t0 + t) * K_h + qk_h);
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm_kh + qk_bh * K_d;
        const float* k_h = k_norm_kh + qk_bh * K_d;
        const float* v_h = v + vh * V_d;
        const float g_h = __expf(g_log[vh]);
        const float beta_h = beta[vh];

        float kv_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] * g_h;
                s_vals[i] = s;
                kv_part += s * k_h[k_idx];
            }
        }
        const float kv_mem = warp_sum(kv_part);
        const float delta = (v_h[v_idx] - kv_mem) * beta_h;

        float out_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] + k_h[k_idx] * delta;
                s_vals[i] = s;
                out_part += s * q_h[k_idx];
            }
        }
        const float out_v = warp_sum(out_part);
        if (lane == 0) {

            const long long out_vh = (long long)(t0 + t + row0) * V_h + h;
            out[out_vh * (long long)V_d + v_idx] = out_v;
        }
    }

    if (write_state && row_persists(write_state_mask, rl)) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(
                        k_vals[i], v_idx, K_d, V_d),
                    s_vals[i]);
            }
        }
    }
}

template <typename StateT, bool KLast>
__global__ void ssm_gated_delta_chunked_batched_warp_tiled_gqa_ilp2(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask)
{
    constexpr int WARPS = 4;
    constexpr int ILP_V = 2;
    constexpr int TILE_V = WARPS * ILP_V;
    constexpr int MAX_K_PER_LANE = 8;
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v0 = blockIdx.z * TILE_V + warp * ILP_V;
    const int v1 = v0 + 1;
    if (warp >= WARPS || v0 >= V_d) return;
    const bool has_v1 = v1 < V_d;

    const int repeat = V_h / K_h;
    const int qk_h = h / repeat;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s0[MAX_K_PER_LANE];
    float s1[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s0[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v0, K_d, V_d));
        s1[n_k] = has_v1
            ? state_load(state + state_offset<KLast>(k_idx, v1, K_d, V_d))
            : 0.f;
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long qk_bh = ((long long)(t0 + t) * K_h + qk_h);
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm_kh + qk_bh * K_d;
        const float* k_h = k_norm_kh + qk_bh * K_d;
        const float* v_h = v + vh * V_d;
        const float g_h = __expf(g_log[vh]);
        const float beta_h = beta[vh];

        float kv_part0 = 0.f;
        float kv_part1 = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float k_val = k_h[k_idx];
                const float s_v0 = s0[i] * g_h;
                s0[i] = s_v0;
                kv_part0 += s_v0 * k_val;
                if (has_v1) {
                    const float s_v1 = s1[i] * g_h;
                    s1[i] = s_v1;
                    kv_part1 += s_v1 * k_val;
                }
            }
        }
        const float kv_mem0 = warp_sum(kv_part0);
        const float kv_mem1 = has_v1 ? warp_sum(kv_part1) : 0.f;
        const float delta0 = (v_h[v0] - kv_mem0) * beta_h;
        const float delta1 = has_v1 ? (v_h[v1] - kv_mem1) * beta_h : 0.f;

        float out_part0 = 0.f;
        float out_part1 = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float k_val = k_h[k_idx];
                const float q_val = q_h[k_idx];
                const float new_s0 = s0[i] + k_val * delta0;
                s0[i] = new_s0;
                out_part0 += new_s0 * q_val;
                if (has_v1) {
                    const float new_s1 = s1[i] + k_val * delta1;
                    s1[i] = new_s1;
                    out_part1 += new_s1 * q_val;
                }
            }
        }
        const float out_v0 = warp_sum(out_part0);
        const float out_v1 = has_v1 ? warp_sum(out_part1) : 0.f;
        if (lane == 0) {
            out[vh * (long long)V_d + v0] = out_v0;
            if (has_v1) out[vh * (long long)V_d + v1] = out_v1;
        }
    }

    if (write_state && row_persists(write_state_mask, r)) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(k_vals[i], v0, K_d, V_d),
                    s0[i]);
                if (has_v1) {
                    state_store(
                        state + state_offset<KLast>(k_vals[i], v1, K_d, V_d),
                        s1[i]);
                }
            }
        }
    }
}

template <typename StateT, bool KLast>
__global__ void ssm_gated_delta_step_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t   = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) + sk[k_idx] * delta;
            state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

template <typename StateT, bool KLast>
__global__ void ssm_gated_delta_step_batched_gqa(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + ((long long)r_row * V_h + h) * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            if (fold) state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float decayed = fold
                ? state_load(state + off)
                : state_round<StateT>(state_load(state + off) * g_h);
            const float s = decayed + sk[k_idx] * delta;
            if (fold) state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

template <typename StateT, bool KLast, int K_D_MAX>
__global__ void ssm_gated_delta_step_batched_fused(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    float* sm_scalars = smem + 2 * K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float partial = 0.f;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        partial += sk[i] * sq[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_xor_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        float w = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
        for (int offset = 16; offset > 0; offset /= 2) {
            w += __shfl_xor_sync(0xffffffffu, w, offset);
        }
        if (threadIdx.x == 0) sm_scalars[0] = w;
    }
    __syncthreads();
    const float sum_sk_sq = sm_scalars[0];

    float s_cache[K_D_MAX];

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {

        float sum_s_sk = 0.f;
        float sum_s_sq = 0.f;
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off);
            s_cache[k_idx] = s;
            sum_s_sk += s * sk[k_idx];
            sum_s_sq += s * sq[k_idx];
        }

        const float kv_mem = g_h * sum_s_sk;
        const float v_t    = v_h[v_idx];
        const float delta  = (v_t - kv_mem) * beta_h;

        const float out_v  = g_h * sum_s_sq + delta * sum_sk_sq;
        out_bh[v_idx] = out_v;

        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s_new = s_cache[k_idx] * g_h + sk[k_idx] * delta;
            state_store(state + off, s_new);
        }
    }
}

template <typename StateT, bool KLast, int K_D_MAX>
__global__ void ssm_gated_delta_step_batched_gqa_fused(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;
    float* sm_scalars = smem + 2 * K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float partial = 0.f;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        partial += sk[i] * sq[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_xor_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        float w = (threadIdx.x < num_warps) ? warp_sums[lane] : 0.f;
        for (int offset = 16; offset > 0; offset /= 2) {
            w += __shfl_xor_sync(0xffffffffu, w, offset);
        }
        if (threadIdx.x == 0) sm_scalars[0] = w;
    }
    __syncthreads();
    const float sum_sk_sq = sm_scalars[0];

    float s_cache[K_D_MAX];

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float sum_s_sk = 0.f;
        float sum_s_sq = 0.f;
        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off);
            s_cache[k_idx] = s;
            sum_s_sk += s * sk[k_idx];
            sum_s_sq += s * sq[k_idx];
        }

        const float kv_mem = g_h * sum_s_sk;
        const float v_t    = v_h[v_idx];
        const float delta  = (v_t - kv_mem) * beta_h;
        const float out_v  = g_h * sum_s_sq + delta * sum_sk_sq;
        out_bh[v_idx] = out_v;

        #pragma unroll 4
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s_new = s_cache[k_idx] * g_h + sk[k_idx] * delta;
            state_store(state + off, s_new);
        }
    }
}

template <typename StateT, int BV, int BK_MAX>
__global__ void ssm_gated_delta_step_batched_fla(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const long long bh = (long long)r * V_h + h;
    const float* q_h = q_norm + bh * K_d;
    const float* k_h = k_norm + bh * K_d;
    const float* v_h = v      + bh * V_d;
    const float  g_h = __expf(g_log[bh]);
    const float  beta_h = beta[bh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + bh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] *= g_h;
    }
    float kv_mem = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        kv_mem += bh_state[k_idx] * sk[k_idx];
    }
    const float v_t   = v_h[v_idx];
    const float delta = (v_t - kv_mem) * beta_h;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] += sk[k_idx] * delta;
    }
    float out_v = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        out_v += bh_state[k_idx] * sq[k_idx];
    }
    out_bh[v_idx] = out_v;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

template <typename StateT, int BV, int BK_MAX>
__global__ void ssm_gated_delta_step_batched_gqa_fla(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*   __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + vh * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] *= g_h;
    }
    float kv_mem = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        kv_mem += bh_state[k_idx] * sk[k_idx];
    }
    const float v_t   = v_h[v_idx];
    const float delta = (v_t - kv_mem) * beta_h;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] += sk[k_idx] * delta;
    }
    float out_v = 0.f;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        out_v += bh_state[k_idx] * sq[k_idx];
    }
    out_bh[v_idx] = out_v;
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

template <int BV>
__global__ void ssm_gated_delta_step_batched_gqa_smem(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* v_h = v + vh * V_d;
    const float g_h = __expf(g_log[vh]);
    const float beta_h = beta[vh];

    __nv_bfloat16* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
    float* out_bh = out + ((long long)r_row * V_h + h) * V_d;

    extern __shared__ __nv_bfloat16 smem_smem_step[];
    __nv_bfloat16* s_state = smem_smem_step;
    float* sq = (float*)(smem_smem_step + K_d * BV);
    float* sk = sq + K_d;

    const bool vec_tile =
        (BV == V_d) && ((V_d & 7) == 0) &&
        ((reinterpret_cast<usize>(state) & 15) == 0);
    const int n_vec = (K_d * V_d) >> 3;
    if (vec_tile) {
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(s_state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    } else {
        constexpr int kStageTile = 16;
        __nv_bfloat16 staged[kStageTile];
        int k = 0;
        for (; k + kStageTile <= K_d; k += kStageTile) {
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                staged[u] = state[(long long)(k + u) * V_d + v_idx];
            }
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                s_state[(k + u) * BV + threadIdx.x] = staged[u];
            }
        }
        for (; k < K_d; ++k) {
            s_state[k * BV + threadIdx.x] =
                state[(long long)k * V_d + v_idx];
        }
    }
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float kv_mem = 0.f;
    for (int k = 0; k < K_d; ++k) {
        float s = __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h;
        kv_mem += s * sk[k];
    }
    const float delta = (v_h[v_idx] - kv_mem) * beta_h;

    float out_v = 0.f;
    for (int k = 0; k < K_d; ++k) {
        const float sg = __bfloat162float(__float2bfloat16(
            __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h));
        float s = sg + sk[k] * delta;
        out_v += s * sq[k];

        if (vec_tile) {
            s_state[k * BV + threadIdx.x] = __float2bfloat16(s);
        } else if (fold) {
            state[(long long)k * V_d + v_idx] = __float2bfloat16(s);
        }
    }
    out_bh[v_idx] = out_v;
    if (vec_tile && fold) {
        __syncthreads();
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(s_state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    }
}

__device__ __forceinline__ void gdn_load8(
    const __nv_bfloat16* __restrict__ p, float (&s)[8]) {
    if ((reinterpret_cast<usize>(p) & 15) == 0) {
        const uint4 raw = *reinterpret_cast<const uint4*>(p);
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 f = __bfloat1622float2(pairs[u]);
            s[2 * u] = f.x;
            s[2 * u + 1] = f.y;
        }
    } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) s[u] = __bfloat162float(p[u]);
    }
}

__device__ __forceinline__ void gdn_store8(
    __nv_bfloat16* __restrict__ p, const float (&s)[8]) {
    if ((reinterpret_cast<usize>(p) & 15) == 0) {
        uint4 raw;
        __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            pairs[u] = __floats2bfloat162_rn(s[2 * u], s[2 * u + 1]);
        }
        *reinterpret_cast<uint4*>(p) = raw;
    } else {
        #pragma unroll
        for (int u = 0; u < 8; ++u) p[u] = __float2bfloat16(s[u]);
    }
}

// A bf16 pair packed from two floats (round to nearest even) and unpacked.
__device__ __forceinline__ unsigned gdn_pack(float lo, float hi) {
    unsigned bits;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(bits) : "f"(lo), "f"(hi));
    return bits;
}

__device__ __forceinline__ float2 gdn_unpack(unsigned bits) {
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

// 16- and 4-byte `cp.async` copies that zero-fill what they do not read, and
// their group fences.
__device__ __forceinline__ void gdn_cp16(void* dst, const void* src, bool live) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(d), "l"(src), "r"(live ? 16 : 0)
                 : "memory");
}

__device__ __forceinline__ void gdn_cp4(void* dst, const void* src, bool live) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :
                 : "r"(d), "l"(src), "r"(live ? 4 : 0)
                 : "memory");
}

__device__ __forceinline__ void gdn_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void gdn_cp_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// A warp's column sum: the lanes that share a column half hold disjoint key
// rows of it, so a butterfly over the other four lane bits folds the sum
// into every one of them.
__device__ __forceinline__ void gdn_rows_sum(float (&x)[8]) {
    #pragma unroll
    for (int off = 2; off < 32; off <<= 1) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) x[u] += __shfl_xor_sync(0xffffffffu, x[u], off);
    }
}

// One decode step of the gated delta rule, streamed. A warp owns a
// sixteen-wide value tile of one head at a time and keeps the whole
// `K_d x 16` state tile in registers: lane `l` holds the eight columns
// `(l & 1) * 8` of key rows `l / 2 + 16 m`, so each state row is one 32-byte
// sector two lanes read whole, and the key and query rows a lane needs are
// the rows it holds — the norms, both column sums and the update stay inside
// the warp. The warps are persistent: each walks every `gridDim.x * W`-th
// tile, and the next tile's state, q and k rows, value slice and gates
// stream into the warp's other shared-memory stage while this one is
// computed, so a warp keeps a tile's bytes in flight all the time. The host
// keeps the state, the qkv row and the output 16-byte aligned and the key
// width a whole number of sixteen-row steps no wider than 128.
constexpr int GDN_STREAM_STAGE = 4096 + 512 + 32 + 32;

template <int W>
__global__ void __launch_bounds__(W * 32) ssm_gdn_decode_step(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    float q_scale,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int tiles,
    const u32* __restrict__ win)
{
    constexpr int MR = 8;
    constexpr int VEC = 8;
    extern __shared__ __align__(16) unsigned char gdn_ring[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned char* ring = gdn_ring + warp * 2 * GDN_STREAM_STAGE;
    const int i0 = lane >> 1;
    const int vts = V_d / 16;
    const int group = V_h / K_h;
    const int K_dim = K_h * K_d;
    const int live_rows = win != nullptr ? static_cast<int>(win[0]) : 0x7fffffff;

    struct Tile {
        bool live;
        bool fold;
        int r_row;
        int h;
        int j0;
        __nv_bfloat16* cell;
    };
    auto plan = [&](int tile) {
        Tile tl;
        tl.live = false;
        if (tile >= tiles) return tl;
        const int vt = tile % vts;
        const int rest = tile / vts;
        tl.h = rest % V_h;
        const int r = rest / V_h;
        if (r >= live_rows) return tl;
        tl.r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const int slot = slot_ids[rl];
        if (slot < 0) return tl;
        tl.live = true;
        tl.fold = folds(write_state, write_state_mask, rl);
        tl.j0 = vt * 16 + (lane & 1) * VEC;
        tl.cell = state_base + (long long)slot * slot_stride_elems
            + (long long)tl.h * K_d * V_d + tl.j0;
        return tl;
    };
    auto issue = [&](const Tile& tl, int stage) {
        if (!tl.live) return;
        unsigned char* st = ring + stage * GDN_STREAM_STAGE;
        #pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (16 * m < K_d) {
                gdn_cp16(st + (m * 32 + lane) * 16, tl.cell + (long long)(i0 + 16 * m) * V_d, true);
            }
        }
        const bf16* row = qkv_post + (long long)tl.r_row * conv_dim;
        const int h_k = tl.h / group;
        const int c = lane & 15;
        if (c * 8 < K_d) {
            const bf16* src = row + (lane < 16 ? 0 : K_dim) + (long long)h_k * K_d + c * 8;
            gdn_cp16(st + 4096 + (lane >> 4) * 256 + c * 16, src, true);
        }
        if (lane < 2) {
            gdn_cp16(st + 4608 + lane * 16, row + 2 * K_dim + (long long)tl.h * V_d + tl.j0, true);
            gdn_cp4(st + 4640 + lane * 4, gates + (long long)tl.r_row * 2 * V_h + lane * V_h + tl.h, true);
        }
    };

    const int stride = gridDim.x * W;
    int t = blockIdx.x * W + warp;
    Tile cur = plan(t);
    issue(cur, 0);
    gdn_cp_commit();
    for (int it = 0; t < tiles; t += stride, ++it) {
        const Tile nxt = plan(t + stride);
        issue(nxt, (it + 1) & 1);
        gdn_cp_commit();
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncwarp();
        if (cur.live) {
            const unsigned char* st = ring + (it & 1) * GDN_STREAM_STAGE;
            float s[MR][VEC];
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                const uint4 raw = 16 * m < K_d
                    ? *reinterpret_cast<const uint4*>(st + (m * 32 + lane) * 16)
                    : make_uint4(0u, 0u, 0u, 0u);
                const unsigned* w32 = reinterpret_cast<const unsigned*>(&raw);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 f = gdn_unpack(w32[u]);
                    s[m][2 * u] = f.x;
                    s[m][2 * u + 1] = f.y;
                }
            }
            const bf16* sq = reinterpret_cast<const bf16*>(st + 4096);
            const bf16* sk = reinterpret_cast<const bf16*>(st + 4096 + 256);
            float qi[MR];
            float ki[MR];
            float qs = 0.f;
            float ks = 0.f;
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                const int i = i0 + 16 * m;
                qi[m] = 16 * m < K_d ? Elem<bf16>::to_f32(sq[i]) : 0.f;
                ki[m] = 16 * m < K_d ? Elem<bf16>::to_f32(sk[i]) : 0.f;
                qs += qi[m] * qi[m];
                ks += ki[m] * ki[m];
            }
            float vv[VEC];
            {
                const uint4 vraw = *reinterpret_cast<const uint4*>(st + 4608 + (lane & 1) * 16);
                const unsigned* w32 = reinterpret_cast<const unsigned*>(&vraw);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 f = gdn_unpack(w32[u]);
                    vv[2 * u] = f.x;
                    vv[2 * u + 1] = f.y;
                }
            }
            const float* sgate = reinterpret_cast<const float*>(st + 4640);
            const float g = __expf(sgate[0]);
            const float beta = sgate[1];

            // Both lanes of a row pair hold the row, so the warp sums it twice.
            const float nq = rsqrtf(0.5f * warp_sum(qs) + 1e-6f) * q_scale;
            const float nk = rsqrtf(0.5f * warp_sum(ks) + 1e-6f);
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                qi[m] *= nq;
                ki[m] *= nk;
            }

            float x[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) x[u] = 0.f;
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                #pragma unroll
                for (int u = 0; u < VEC; ++u) x[u] += s[m][u] * g * ki[m];
            }
            gdn_rows_sum(x);

            float o[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                x[u] = (vv[u] - x[u]) * beta;
                o[u] = 0.f;
            }
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    const float sg = __bfloat162float(__float2bfloat16(s[m][u] * g));
                    const float sn = sg + ki[m] * x[u];
                    s[m][u] = sn;
                    o[u] += sn * qi[m];
                }
            }
            if (cur.fold) {
                #pragma unroll
                for (int m = 0; m < MR; ++m) {
                    if (16 * m >= K_d) break;
                    uint4 packed;
                    unsigned* w32 = reinterpret_cast<unsigned*>(&packed);
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) w32[u] = gdn_pack(s[m][2 * u], s[m][2 * u + 1]);
                    *reinterpret_cast<uint4*>(cur.cell + (long long)(i0 + 16 * m) * V_d) = packed;
                }
            }
            gdn_rows_sum(o);
            if (lane < 2) {
                float4* dst = reinterpret_cast<float4*>(
                    out + ((long long)cur.r_row * V_h + cur.h) * V_d + cur.j0);
                dst[0] = make_float4(o[0], o[1], o[2], o[3]);
                dst[1] = make_float4(o[4], o[5], o[6], o[7]);
            }
        }
        __syncwarp();
        cur = nxt;
    }
    gdn_cp_wait_all();
}

template <typename StateT, int BV, int BK_MAX>
__global__ void ssm_gated_delta_chunked_batched_fla(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const int* __restrict__ commit_len,
    const u8* __restrict__ write_state_mask,
    const int* __restrict__ begin_at,
    bool fused_decay,
    const u32* __restrict__ win)
{

    const int gqa_repeat = V_h / K_h;

    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int h_k = h / gqa_repeat;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int T  = static_cast<int>(qo_indptr[r + 1]) - t0;

    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > T) b = T;
        if (b > 0) { t0 += b; T -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[rl];
        if (c < T) T = c;
    }
    if (T <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;

    __nv_bfloat162 bh_state[BK_MAX / 2];
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float s0 = state_load(state + (long long)k0 * V_d + v_idx);
        const float s1 = (k1 < K_d)
            ? state_load(state + (long long)k1 * V_d + v_idx)
            : 0.f;
        bh_state[j] = __floats2bfloat162_rn(s0, s1);
    }

    const bool single_round = fused_decay;

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];

        const long long bh_qk = (long long)(t0 + t) * K_h + h_k;
        const float* q_h_t = q_norm + bh_qk * K_d;
        const float* k_h_t = k_norm + bh_qk * K_d;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h_t[i];
            sk[i] = k_h_t[i];
        }
        __syncthreads();

        float kv_mem = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            s.x *= g_h;
            if (k1 < K_d) s.y *= g_h;
            if (!single_round) bh_state[j] = __floats2bfloat162_rn(s.x, s.y);
            kv_mem += s.x * sk[k0];
            if (k1 < K_d) kv_mem += s.y * sk[k1];
        }
        const float v_t   = v[bh * V_d + v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            float sx, sy;
            if (single_round) {
                sx = s.x * g_h + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y * g_h + sk[k1] * delta) : s.y;
            } else {

                sx = s.x + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y + sk[k1] * delta) : s.y;
            }
            bh_state[j] = __floats2bfloat162_rn(sx, sy);
            out_v += sx * sq[k0];
            if (k1 < K_d) out_v += sy * sq[k1];
        }

        out[((long long)(t0 + t + row0) * V_h + h) * V_d + v_idx] = out_v;
        __syncthreads();
    }

    if (!write_state || !row_persists(write_state_mask, rl)) return;
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float2 s = __bfloat1622float2(bh_state[j]);
        state_store(state + (long long)k0 * V_d + v_idx, s.x);
        if (k1 < K_d) state_store(state + (long long)k1 * V_d + v_idx, s.y);
    }
}

template <typename StateT, int BV, int BK_MAX>
__global__ void ssm_gated_delta_chunked_batched_gqa_fla(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;

    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[r];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;

    float bh_state[BK_MAX];
    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        bh_state[k_idx] = state_load(state + (long long)k_idx * V_d + v_idx);
    }

    for (int t = 0; t < T; ++t) {
        const long long qh = ((long long)(t0 + t) * K_h + h_k) * K_d;
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float  g_h = __expf(g_log[vh]);
        const float  beta_h = beta[vh];

        const float* q_h_t = q_norm_kh + qh;
        const float* k_h_t = k_norm_kh + qh;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h_t[i];
            sk[i] = k_h_t[i];
        }
        __syncthreads();

        float kv_mem = 0.f;
        #pragma unroll
        for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
            if (k_idx >= K_d) break;
            bh_state[k_idx] *= g_h;
            kv_mem += bh_state[k_idx] * sk[k_idx];
        }
        const float v_t   = v[vh * V_d + v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        #pragma unroll
        for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
            if (k_idx >= K_d) break;
            bh_state[k_idx] += sk[k_idx] * delta;
            out_v += bh_state[k_idx] * sq[k_idx];
        }
        out[vh * V_d + v_idx] = out_v;
        __syncthreads();
    }

    #pragma unroll
    for (int k_idx = 0; k_idx < BK_MAX; ++k_idx) {
        if (k_idx >= K_d) break;
        state_store(state + (long long)k_idx * V_d + v_idx, bh_state[k_idx]);
    }
}

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.f / (1.f + __expf(-x));
}

template <class ElemT>
__global__ void ssm_kda_gate_beta(
    const ElemT* __restrict__ raw_g,
    const ElemT* __restrict__ raw_beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    int T, int H, int D,
    float lower_bound,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    const int h = blockIdx.y;
    if (t >= T || h >= H) return;

    const float a = __expf(A_log[h]);
    const long long base = ((long long)t * H + h) * D;
    const long long src = ((long long)t_row * H + h) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float g = Elem<ElemT>::to_f32(raw_g[src + d]) + dt_bias[(long long)h * D + d];
        float gate;
        if (lower_bound < 0.f) {
            gate = lower_bound * sigmoidf(a * g);
        } else {

            const float sp = (g > 20.f) ? g : __logf(1.f + __expf(g));
            gate = -a * sp;
        }
        gate_out[base + d] = gate;
    }

    if (threadIdx.x == 0) {
        beta_out[(long long)t * H + h] =
            sigmoidf(Elem<ElemT>::to_f32(raw_beta[(long long)t_row * H + h]));
    }
}

template <class ElemT, int BLOCK>
__global__ void ssm_kda_qkv_prep(
    const ElemT* __restrict__ mixed,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int width, int head_dim, float eps,
    const u32* __restrict__ win)
{

    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int plane = blockIdx.y;
    const int tid = threadIdx.x;
    const ElemT* src =
        mixed + (long long)n_row * 3 * width + (long long)plane * width;
    float* dst =
        (plane == 0 ? q_out : (plane == 1 ? k_out : v_out)) +
        (long long)n * width;
    if (plane == 2) {
        for (int i = tid; i < width; i += BLOCK) {
            dst[i] = Elem<ElemT>::to_f32(src[i]);
        }
        return;
    }
    constexpr int kMaxHeads = 256;
    __shared__ float sums[kMaxHeads];
    const int heads = head_dim > 0 ? width / head_dim : 1;

    {
        constexpr int kWarps = BLOCK / 32;
        const int warp = tid / 32;
        const int lane = tid % 32;
        for (int h = warp; h < heads && h < kMaxHeads; h += kWarps) {
            float acc = 0.f;
            for (int d = lane; d < head_dim; d += 32) {
                const float x = Elem<ElemT>::to_f32(src[h * head_dim + d]);
                acc += x * x;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) sums[h] = acc;
        }
    }
    __syncthreads();
    const float q_scale = (plane == 0) ? rsqrtf(static_cast<float>(head_dim)) : 1.f;
    for (int i = tid; i < width; i += BLOCK) {
        const float inv = rsqrtf(sums[(i / head_dim) % kMaxHeads] + eps);
        dst[i] = Elem<ElemT>::to_f32(src[i]) * inv * q_scale;
    }
}

// The KDA scan walks its `D x D` head state in place, so a row that does not
// fold this fire's boundary (an unfolded buffer, a masked lane, the replayed
// tail past a commit) walks a copy of it in `work` and the slot stays what
// the fold before it left.
__device__ __forceinline__ float* kda_state_or_copy(
    float* __restrict__ st,
    float* __restrict__ work,
    long long at,
    int D,
    bool fold)
{
    if (fold) return st;
    float* copy = work + at * D * D;
    for (int i = threadIdx.x; i < D * D; i += blockDim.x) copy[i] = st[i];
    return copy;
}

__global__ void ssm_kda_step_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    float* __restrict__ work,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h = blockIdx.y;

    const long long rh = (long long)r * H + h;
    const float* q_h = q_norm + rh * D;
    const float* k_h = k_norm + rh * D;
    const float* v_h = v      + rh * D;
    const float* g_h = gate   + rh * D;
    const float beta_h = beta[rh];

    const int slot = slot_ids[r];
    if (slot < 0) return;
    float* st = kda_state_or_copy(
        state_base + (long long)slot * slot_stride_elems + (long long)h * D * D,
        work, rh, D, folds(write_state, write_state_mask, r));
    float* out_h = out + ((long long)r_row * H + h) * D;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
        sg[i] = __expf(g_h[i]);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (int vi = warp; vi < D; vi += warps) {
        float* row = st + (long long)vi * D;
        float mem = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] * sg[ki];
            row[ki] = sv;
            mem += sv * sk[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            mem += __shfl_down_sync(0xffffffffu, mem, off);
        }
        const float delta = __shfl_sync(0xffffffffu, (v_h[vi] - mem) * beta_h, 0);

        float acc = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] + sk[ki] * delta;
            row[ki] = sv;
            acc += sv * sq[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out_h[vi] = acc;
    }
}

__global__ void ssm_kda_chunked_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D,
    bool write_state,
    const int* __restrict__ commit_len,
    const u8* __restrict__ write_state_mask,
    const int* __restrict__ begin_at,
    float* __restrict__ work,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;

    long long begin = qo_indptr[r];
    long long end = qo_indptr[r + 1];
    if (begin_at != nullptr) {
        const long long b = begin + begin_at[rl];
        if (b > begin) begin = b < end ? b : end;
    }
    if (commit_len != nullptr) {
        const long long c = begin + commit_len[rl];
        if (c < end) end = c;
    }
    if (end <= begin) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    float* st = kda_state_or_copy(
        state_base + (long long)slot * slot_stride_elems + (long long)h * D * D,
        work, (long long)r * H + h, D, folds(write_state, write_state_mask, rl));

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (long long t = begin; t < end; ++t) {
        const long long th = t * H + h;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            sq[i] = q_norm[th * D + i];
            sk[i] = k_norm[th * D + i];
            sg[i] = __expf(gate[th * D + i]);
        }
        __syncthreads();

        const float beta_h = beta[th];
        for (int vi = warp; vi < D; vi += warps) {
            float* row = st + (long long)vi * D;
            float mem = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] * sg[ki];
                row[ki] = sv;
                mem += sv * sk[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                mem += __shfl_down_sync(0xffffffffu, mem, off);
            }
            const float delta =
                __shfl_sync(0xffffffffu, (v[th * D + vi] - mem) * beta_h, 0);

            float acc = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] + sk[ki] * delta;
                row[ki] = sv;
                acc += sv * sq[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }

            if (lane == 0) out[((t + row0) * H + h) * D + vi] = acc;
        }

        __syncthreads();
    }
}

constexpr int GDN_CH = 64;

// A gated delta walk's f32 rows narrowed to bf16 where its trace asked for
// a bf16 output and the walk could not land one itself: one block a row,
// the staged window's rows only.
__global__ void ssm_gdn_narrow_rows(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int width,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        dst[row * width + c] = __float2bfloat16(src[row * width + c]);
    }
}

struct GdnLane {
    int t0;
    int T;
    int rl;
    int slot;
};

__device__ __forceinline__ GdnLane gdn_lane(
    int r,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ slot_ids,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    const u32* __restrict__ win)
{
    GdnLane lane;
    lane.rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    lane.t0 = static_cast<int>(qo_indptr[r]);
    lane.T = static_cast<int>(qo_indptr[r + 1]) - lane.t0;
    if (begin_at != nullptr) {
        int b = begin_at[lane.rl];
        if (b > lane.T) b = lane.T;
        if (b > 0) { lane.t0 += b; lane.T -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[lane.rl];
        if (c < lane.T) lane.T = c;
    }
    lane.slot = lane.T > 0 ? slot_ids[lane.rl] : -1;
    return lane;
}

__device__ __forceinline__ int gdn_chunk_base(
    int r,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ begin_at,
    const u32* __restrict__ win)
{
    int t0 = static_cast<int>(qo_indptr[r]);
    if (begin_at != nullptr) {
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const int T = static_cast<int>(qo_indptr[r + 1]) - t0;
        int b = begin_at[rl];
        if (b > T) b = T;
        if (b > 0) t0 += b;
    }
    return t0 / GDN_CH + r;
}

__device__ __forceinline__ int gdn_slot_lane(
    int s, int lanes,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ begin_at,
    const u32* __restrict__ win)
{
    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (gdn_chunk_base(mid, qo_indptr, begin_at, win) <= s) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// The chunked scan's tensor-core pieces: a bf16 `m16n8k16` with f32
// accumulators and `ldmatrix` fragment loads, plain and transposed.
__device__ __forceinline__ void gdn_mma(
    float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void gdn_ldsm(unsigned (&r)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void gdn_ldsm_t(unsigned (&r)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// Lane `l`'s row address for each fragment an x4 `ldmatrix` lands, over a
// row-major tile with leading dimension `ld`:
// the A operand of rows `m0..m0+16`, columns `k0..k0+16`;
__device__ __forceinline__ const bf16* gdn_at_a(const bf16* s, int ld, int m0, int k0, int l) {
    return s + (m0 + (l & 15)) * ld + k0 + ((l >> 4) << 3);
}

// the B operand of two eight-wide column tiles `n0`, `n0 + 8` stored as
// `[n][k]` rows (no transpose);
__device__ __forceinline__ const bf16* gdn_at_bt(const bf16* s, int ld, int n0, int k0, int l) {
    return s + (n0 + (l & 7) + ((l >> 4) << 3)) * ld + k0 + (((l >> 3) & 1) << 3);
}

// the B operand of the same two tiles stored as `[k][n]` rows (transposed);
__device__ __forceinline__ const bf16* gdn_at_b(const bf16* s, int ld, int k0, int n0, int l) {
    return s + (k0 + (l & 7) + (((l >> 3) & 1) << 3)) * ld + n0 + ((l >> 4) << 3);
}

// the A operand of rows `m0..m0+16` stored as `[k][m]` rows (transposed).
__device__ __forceinline__ const bf16* gdn_at_at(const bf16* s, int ld, int k0, int m0, int l) {
    return s + (k0 + (l & 7) + ((l >> 4) << 3)) * ld + m0 + (((l >> 3) & 1) << 3);
}

// A 128-wide bf16 row padded so the eight rows one `ldmatrix` reads land in
// distinct banks, and a 64-wide row likewise.
constexpr int GDN_LD = 136;
constexpr int GDN_LDC = 72;
constexpr int GDN_LDA = 68;

// The chunk-parallel half of the chunked gated delta rule: one block per
// (chunk, value head) normalizes its chunk's key rows, scans the decay
// within the chunk, inverts `I + A` with `A = tril(beta_i k_i . k_j
// exp(G_i - G_j), -1)` and lands `w = T (beta exp(G) k)` and
// `u = T (beta v)`, the chunk's update directions, and the chunk's decayed
// causal scores `P = tril(q_i . k_j exp(G_i - G_j))`, beside the normalized
// queries and keys (written once per key head) and the decay's running sum.
template <int BK, int BV>
__global__ void __launch_bounds__(256) ssm_gdn_chunk_prep(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    float* __restrict__ gcum,
    bf16* __restrict__ qn_out,
    bf16* __restrict__ kn_out,
    bf16* __restrict__ w_out,
    bf16* __restrict__ u_out,
    bf16* __restrict__ p_out,
    int lanes, int K_h, int V_h, int conv_dim,
    float q_scale,
    const u32* __restrict__ win)
{
    static_assert(BK == 128 && BV == 128, "a lane carries four of a 128-wide row");
    constexpr int THREADS = 256;
    extern __shared__ __align__(16) unsigned char gdn_smem[];
    bf16* ks = reinterpret_cast<bf16*>(gdn_smem);
    bf16* vs = ks + GDN_CH * GDN_LD;
    bf16* t1 = vs + GDN_CH * GDN_LD;
    bf16* t2 = t1 + GDN_CH * GDN_LDC;
    float* A = reinterpret_cast<float*>(t2 + GDN_CH * GDN_LDC);
    float* G = A + GDN_CH * GDN_LDA;
    float* B = G + GDN_CH;

    const int h = blockIdx.x;
    const int s = blockIdx.y;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, qo_indptr, begin_at, win);
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int j = s - (lane.t0 / GDN_CH + r);
    if (j < 0 || j * GDN_CH >= lane.T) return;
    const int ta = lane.t0 + j * GDN_CH;
    const int L = min(GDN_CH, lane.t0 + lane.T - ta);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int group = V_h / K_h;
    const int hk = h / group;
    const bool writer = h % group == 0;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int l = tid & 31;
    const int K_dim = K_h * BK;

    // The chunk's raw key, value and query rows land whole before any is
    // read; the queries borrow the decay tile's space until the scores are
    // formed.
    bf16* qs = reinterpret_cast<bf16*>(A);
    for (int e = tid; e < GDN_CH * (BK / 8); e += THREADS) {
        const int i = e / (BK / 8);
        const int c = (e % (BK / 8)) * 8;
        const bool in = i < L;
        const bf16* row = qkv_post + static_cast<long long>(ta + (in ? i : 0) + row0) * conv_dim;
        gdn_cp16(ks + i * GDN_LD + c, row + K_dim + hk * BK + c, in);
        gdn_cp16(vs + i * GDN_LD + c, row + 2 * K_dim + h * BV + c, in);
        gdn_cp16(qs + i * GDN_LD + c, row + hk * BK + c, in);
    }
    gdn_cp_commit();
    if (tid < GDN_CH) {
        const bool in = tid < L;
        const float* grow = gates + static_cast<long long>(ta + (in ? tid : 0) + row0) * 2 * V_h;
        G[tid] = in ? grow[h] : 0.f;
        B[tid] = in ? grow[V_h + h] : 0.f;
    }
    gdn_cp_wait_all();
    __syncthreads();
    #pragma unroll 4
    for (int rr = 0; rr < 8; ++rr) {
        const int i = warp * 8 + rr;
        const bool in = i < L;
        uint2* kp = reinterpret_cast<uint2*>(ks + i * GDN_LD + l * 4);
        const uint2 kr = *kp;
        const float2 k01 = gdn_unpack(kr.x);
        const float2 k23 = gdn_unpack(kr.y);
        const float kinv = rsqrtf(
            warp_sum(k01.x * k01.x + k01.y * k01.y + k23.x * k23.x + k23.y * k23.y) + 1e-6f);
        const uint2 kw = make_uint2(gdn_pack(k01.x * kinv, k01.y * kinv),
                                    gdn_pack(k23.x * kinv, k23.y * kinv));
        *kp = kw;
        uint2* vp = reinterpret_cast<uint2*>(vs + i * GDN_LD + l * 4);
        const uint2 vr = *vp;
        const float beta = B[i];
        const float2 v01 = gdn_unpack(vr.x);
        const float2 v23 = gdn_unpack(vr.y);
        *vp = make_uint2(gdn_pack(v01.x * beta, v01.y * beta), gdn_pack(v23.x * beta, v23.y * beta));
        uint2* qp = reinterpret_cast<uint2*>(qs + i * GDN_LD + l * 4);
        const uint2 qr = *qp;
        const float2 q01 = gdn_unpack(qr.x);
        const float2 q23 = gdn_unpack(qr.y);
        const float qinv = rsqrtf(
            warp_sum(q01.x * q01.x + q01.y * q01.y + q23.x * q23.x + q23.y * q23.y) + 1e-6f)
            * q_scale;
        const uint2 qw = make_uint2(gdn_pack(q01.x * qinv, q01.y * qinv),
                                    gdn_pack(q23.x * qinv, q23.y * qinv));
        *qp = qw;
        if (writer && in) {
            const long long at = (static_cast<long long>(ta + i) * K_h + hk) * BK + l * 4;
            *reinterpret_cast<uint2*>(kn_out + at) = kw;
            *reinterpret_cast<uint2*>(qn_out + at) = qw;
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float a = G[2 * l];
        const float b = a + G[2 * l + 1];
        float run = b;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float up = __shfl_up_sync(0xffffffffu, run, off);
            if (l >= off) run += up;
        }
        const float before = run - b;
        G[2 * l] = before + a;
        G[2 * l + 1] = run;
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = 2 * l + e;
            if (i < L) gcum[static_cast<long long>(ta + i) * V_h + h] = G[i];
        }
    }
    __syncthreads();

    // The warps split the chunk's rows four ways and the columns two ways.
    const int g = l >> 2;
    const int t = l & 3;
    const int m0 = (warp & 3) * 16;
    const int half_n = warp >> 2;
    {
        // `k k^T` and `q k^T` share their key operand.
        const int j0 = half_n * 32;
        float acc[4][4];
        float qkc[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[nt][e] = 0.f;
                qkc[nt][e] = 0.f;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned a[4];
            unsigned aq[4];
            gdn_ldsm(a, gdn_at_a(ks, GDN_LD, m0, kk * 16, l));
            gdn_ldsm(aq, gdn_at_a(qs, GDN_LD, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned b[4];
                gdn_ldsm(b, gdn_at_bt(ks, GDN_LD, j0 + np * 16, kk * 16, l));
                gdn_mma(acc[2 * np], a, b[0], b[1]);
                gdn_mma(acc[2 * np + 1], a, b[2], b[3]);
                gdn_mma(qkc[2 * np], aq, b[0], b[1]);
                gdn_mma(qkc[2 * np + 1], aq, b[2], b[3]);
            }
        }
        bf16* pd = p_out + (static_cast<long long>(s) * V_h + h) * GDN_CH * GDN_CH;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int i = m0 + g + 8 * half;
                const int jj = j0 + nt * 8 + 2 * t;
                const float p0 = (jj <= i && i < L)
                    ? qkc[nt][2 * half] * __expf(G[i] - G[jj]) : 0.f;
                const float p1 = (jj + 1 <= i && i < L)
                    ? qkc[nt][2 * half + 1] * __expf(G[i] - G[jj + 1]) : 0.f;
                *reinterpret_cast<unsigned*>(pd + i * GDN_CH + jj) = gdn_pack(p0, p1);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = m0 + g + 8 * (e >> 1);
                const int jj = j0 + nt * 8 + 2 * t + (e & 1);
                A[i * GDN_LDA + jj] = (jj < i && i < L)
                    ? acc[nt][e] * B[i] * __expf(G[i] - G[jj])
                    : 0.f;
            }
        }
    }
    __syncthreads();

    // `T = (I + A)^-1` by sixteen-row blocks, in f32 throughout: each warp
    // inverts one diagonal block `D_b` by forward substitution, then each
    // later block row solves its off-diagonal blocks from the rows above it,
    // `T_ib = -D_i sum_{b<=k<i} A_ik T_kb`, one block a warp. `T` lands in
    // the space its two bf16 scalings take after it.
    float* Tm = reinterpret_cast<float*>(t1);
    float* E = B + GDN_CH;
    float* X = E + GDN_CH + warp * 256;
    if (tid < GDN_CH) E[tid] = B[tid] * __expf(G[tid]);
    if (warp < 4 && l < 16) {
        const int d0 = warp * 16;
        float tcol[16];
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            const float* ar = A + (d0 + rr) * GDN_LDA + d0;
            float sum = 0.f;
            #pragma unroll
            for (int m = 0; m < rr; ++m) sum = __fmaf_rn(ar[m], tcol[m], sum);
            tcol[rr] = rr == l ? 1.f : (rr < l ? 0.f : -sum);
        }
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) Tm[(d0 + rr) * GDN_LDA + d0 + l] = tcol[rr];
    }
    {
        const int rr = l >> 1;
        const int cc = (l & 1) * 8;
        for (int bi = 1; bi < 4; ++bi) {
            __syncthreads();
            if (warp >= bi) continue;
            const int bb = warp;
            float x[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) x[u] = 0.f;
            for (int kb = bb; kb < bi; ++kb) {
                const float* ar = A + (bi * 16 + rr) * GDN_LDA + kb * 16;
                #pragma unroll
                for (int m = 0; m < 16; ++m) {
                    const float av = ar[m];
                    const float* tr = Tm + (kb * 16 + m) * GDN_LDA + bb * 16 + cc;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) x[u] = __fmaf_rn(av, tr[u], x[u]);
                }
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) X[rr * 16 + cc + u] = x[u];
            __syncwarp();
            float y[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) y[u] = 0.f;
            const float* dr = Tm + (bi * 16 + rr) * GDN_LDA + bi * 16;
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                const float dv = dr[m];
                #pragma unroll
                for (int u = 0; u < 8; ++u) y[u] = __fmaf_rn(dv, X[m * 16 + cc + u], y[u]);
            }
            float* tw = Tm + (bi * 16 + rr) * GDN_LDA + bb * 16 + cc;
            #pragma unroll
            for (int u = 0; u < 8; ++u) tw[u] = -y[u];
            __syncwarp();
        }
    }
    __syncthreads();
    {
        constexpr int PER = GDN_CH * GDN_CH / THREADS;
        float tv[PER];
        #pragma unroll
        for (int q = 0; q < PER; ++q) {
            const int e = q * THREADS + tid;
            const int i = e / GDN_CH;
            const int c = e % GDN_CH;
            tv[q] = (c >> 4) > (i >> 4) ? 0.f : Tm[i * GDN_LDA + c];
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < PER; ++q) {
            const int e = q * THREADS + tid;
            const int i = e / GDN_CH;
            const int c = e % GDN_CH;
            t1[i * GDN_LDC + c] = bf16(tv[q] * E[c]);
            t2[i * GDN_LDC + c] = bf16(tv[q]);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const bf16* tm = pass == 0 ? t1 : t2;
        const bf16* rhs = pass == 0 ? ks : vs;
        bf16* dst = pass == 0 ? w_out : u_out;
        const int n0 = half_n * 64;
        float acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
        }
        #pragma unroll
        for (int kk = 0; kk < GDN_CH / 16; ++kk) {
            unsigned a[4];
            gdn_ldsm(a, gdn_at_a(tm, GDN_LDC, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned b[4];
                gdn_ldsm_t(b, gdn_at_b(rhs, GDN_LD, kk * 16, n0 + np * 16, l));
                gdn_mma(acc[2 * np], a, b[0], b[1]);
                gdn_mma(acc[2 * np + 1], a, b[2], b[3]);
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            if (i >= L) continue;
            bf16* o = dst + (static_cast<long long>(ta + i) * V_h + h) * BV + n0 + 2 * t;
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                *reinterpret_cast<unsigned*>(o + nt * 8) =
                    gdn_pack(acc[nt][2 * half], acc[nt][2 * half + 1]);
            }
        }
    }
}

// The sequential half: one block per (request, value head, `BV`-wide value
// tile) walks the request's chunks in order with the `BK x BV` state in f32
// accumulators. Each chunk lands its output — the decayed state's reading
// `exp(G_i) q_i S` plus the chunk's own causal part `P v_new`, with `P` the
// prep's decayed causal scores and `v_new = u - w S` — and then folds the
// chunk into the state, `S = exp(G_L) S + k^T (exp(G_L - G_j) v_new)`. The
// chunk's rows split four ways and its value columns two ways over the
// eight warps. Each operand of the next chunk streams in as soon as this
// chunk is done with its buffer: q, w and u once `v_new` is formed, the
// scores once the output is, and the decay into a second buffer. The keys,
// read by the fold alone, land in one buffer during the chunk they fold:
// so two blocks fit an SM, and one's syncs hide behind the other's work.
template <int BV>
__global__ void __launch_bounds__(256) ssm_gdn_chunk_scan(
    const float* __restrict__ gcum,
    const bf16* __restrict__ qn,
    const bf16* __restrict__ kn,
    const bf16* __restrict__ w,
    const bf16* __restrict__ u,
    const bf16* __restrict__ pm,
    __nv_bfloat16* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    float* __restrict__ out,
    int lanes, int K_h, int V_h,
    const u32* __restrict__ win)
{
    constexpr int BK = 128;
    constexpr int V = 128;
    constexpr int THREADS = 256;
    constexpr int NT = BV / 8;
    constexpr int CN = BV / 2;
    constexpr int CT = CN / 8;
    constexpr int LDV = BV + 8;
    static_assert(CN % 16 == 0 && V % BV == 0, "a warp's value columns are whole sixteen-wide pairs");
    extern __shared__ __align__(16) unsigned char gdn_smem[];
    bf16* sq = reinterpret_cast<bf16*>(gdn_smem);
    bf16* sw = sq + GDN_CH * GDN_LD;
    bf16* sk = sw + GDN_CH * GDN_LD;
    bf16* su = sk + GDN_CH * GDN_LD;
    bf16* sp = su + GDN_CH * LDV;
    bf16* hs = sp + GDN_CH * GDN_LDC;
    bf16* vn = hs + BK * LDV;
    bf16* vsc = vn + GDN_CH * LDV;
    float* sg0 = reinterpret_cast<float*>(vsc + GDN_CH * LDV);

    const int nvt = V / BV;
    const int vt = static_cast<int>(blockIdx.x) % nvt;
    const int h = (static_cast<int>(blockIdx.x) / nvt) % V_h;
    const int r = static_cast<int>(blockIdx.x) / (nvt * V_h);
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (r >= lanes) return;
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int hk = h / (V_h / K_h);
    const int v0 = vt * BV;
    const int base = lane.t0 / GDN_CH + r;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int l = tid & 31;
    const int g = l >> 2;
    const int t = l & 3;
    const int m0 = (warp & 3) * 16;
    const int c0 = (warp >> 2) * CN;
    // The state splits over the warps as the output does: 32 key rows by
    // the warp's value half, two row tiles of `CT` column tiles each.
    const int kr0 = (warp & 3) * 32;
    auto h_row = [&](int q, int half) { return kr0 + (q / CT) * 16 + g + 8 * half; };
    auto h_col = [&](int q) { return c0 + (q % CT) * 8 + 2 * t; };
    __nv_bfloat16* state = state_base
        + static_cast<long long>(lane.slot) * slot_stride_elems
        + static_cast<long long>(h) * BK * V;

    float hacc[NT][4];
    #pragma unroll
    for (int q = 0; q < NT; ++q) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const float2 f = gdn_unpack(*reinterpret_cast<const unsigned*>(
                state + static_cast<long long>(h_row(q, half)) * V + v0 + h_col(q)));
            hacc[q][2 * half] = f.x;
            hacc[q][2 * half + 1] = f.y;
        }
    }

    const int nchunks = (lane.T + GDN_CH - 1) / GDN_CH;
    auto span = [&](int jc, int& ta, int& L) {
        ta = lane.t0 + jc * GDN_CH;
        L = min(GDN_CH, lane.t0 + lane.T - ta);
    };
    auto load_k = [&](int jc) {
        int ta, L;
        span(jc, ta, L);
        #pragma unroll
        for (int e = tid; e < GDN_CH * (BK / 8); e += THREADS) {
            const int i = e / (BK / 8);
            const int c = (e % (BK / 8)) * 8;
            const bool in = i < L;
            gdn_cp16(sk + i * GDN_LD + c,
                     kn + (static_cast<long long>(ta + (in ? i : 0)) * K_h + hk) * BK + c, in);
        }
    };
    auto load_g = [&](int jc) {
        int ta, L;
        span(jc, ta, L);
        float* sg = sg0 + (jc & 1) * GDN_CH;
        if (tid < GDN_CH) {
            const bool in = tid < L;
            gdn_cp4(sg + tid, gcum + static_cast<long long>(ta + (in ? tid : 0)) * V_h + h, in);
        }
    };
    auto load_qwu = [&](int jc) {
        int ta, L;
        span(jc, ta, L);
        #pragma unroll
        for (int e = tid; e < GDN_CH * (BK / 8); e += THREADS) {
            const int i = e / (BK / 8);
            const int c = (e % (BK / 8)) * 8;
            const bool in = i < L;
            const long long row = ta + (in ? i : 0);
            gdn_cp16(sq + i * GDN_LD + c, qn + (row * K_h + hk) * BK + c, in);
            gdn_cp16(sw + i * GDN_LD + c, w + (row * V_h + h) * BK + c, in);
        }
        #pragma unroll
        for (int e = tid; e < GDN_CH * (BV / 8); e += THREADS) {
            const int i = e / (BV / 8);
            const int c = (e % (BV / 8)) * 8;
            const bool in = i < L;
            gdn_cp16(su + i * LDV + c,
                     u + (static_cast<long long>(ta + (in ? i : 0)) * V_h + h) * V + v0 + c, in);
        }
    };
    auto load_p = [&](int jc) {
        const bf16* src = pm + (static_cast<long long>(base + jc) * V_h + h) * GDN_CH * GDN_CH;
        #pragma unroll
        for (int e = tid; e < GDN_CH * (GDN_CH / 8); e += THREADS) {
            const int i = e / (GDN_CH / 8);
            const int c = (e % (GDN_CH / 8)) * 8;
            gdn_cp16(sp + i * GDN_LDC + c, src + i * GDN_CH + c, true);
        }
    };

    load_g(0);
    load_qwu(0);
    load_p(0);
    gdn_cp_commit();
    #pragma unroll
    for (int q = 0; q < NT; ++q) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            *reinterpret_cast<unsigned*>(hs + h_row(q, half) * LDV + h_col(q)) =
                gdn_pack(hacc[q][2 * half], hacc[q][2 * half + 1]);
        }
    }

    for (int jc = 0; jc < nchunks; ++jc) {
        gdn_cp_wait_all();
        __syncthreads();
        const bool more = jc + 1 < nchunks;
        // This chunk's keys, read only by its state fold, into the one key
        // buffer the last fold is done with; the next chunk's decay ahead.
        load_k(jc);
        if (more) load_g(jc + 1);
        gdn_cp_commit();

        int ta, L;
        span(jc, ta, L);
        const float* sg = sg0 + (jc & 1) * GDN_CH;
        const float g_last = sg[L - 1];

        float qh[CT][4];
        float wh[CT][4];
        #pragma unroll
        for (int nt = 0; nt < CT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                qh[nt][e] = 0.f;
                wh[nt][e] = 0.f;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned aq[4];
            unsigned aw[4];
            gdn_ldsm(aq, gdn_at_a(sq, GDN_LD, m0, kk * 16, l));
            gdn_ldsm(aw, gdn_at_a(sw, GDN_LD, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < CT / 2; ++np) {
                unsigned b[4];
                gdn_ldsm_t(b, gdn_at_b(hs, LDV, kk * 16, c0 + np * 16, l));
                gdn_mma(qh[2 * np], aq, b[0], b[1]);
                gdn_mma(qh[2 * np + 1], aq, b[2], b[3]);
                gdn_mma(wh[2 * np], aw, b[0], b[1]);
                gdn_mma(wh[2 * np + 1], aw, b[2], b[3]);
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            const bool in = i < L;
            const float decay = in ? __expf(g_last - sg[i]) : 0.f;
            #pragma unroll
            for (int nt = 0; nt < CT; ++nt) {
                const int c = c0 + nt * 8 + 2 * t;
                const float2 uu = gdn_unpack(*reinterpret_cast<const unsigned*>(su + i * LDV + c));
                const float n0 = in ? uu.x - wh[nt][2 * half] : 0.f;
                const float n1 = in ? uu.y - wh[nt][2 * half + 1] : 0.f;
                *reinterpret_cast<unsigned*>(vn + i * LDV + c) = gdn_pack(n0, n1);
                *reinterpret_cast<unsigned*>(vsc + i * LDV + c) = gdn_pack(n0 * decay, n1 * decay);
            }
        }
        __syncthreads();
        if (more) load_qwu(jc + 1);
        gdn_cp_commit();

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            const float eg = __expf(sg[i]);
            #pragma unroll
            for (int nt = 0; nt < CT; ++nt) {
                qh[nt][2 * half] *= eg;
                qh[nt][2 * half + 1] *= eg;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GDN_CH / 16; ++kk) {
            unsigned pa[4];
            gdn_ldsm(pa, gdn_at_a(sp, GDN_LDC, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < CT / 2; ++np) {
                unsigned b[4];
                gdn_ldsm_t(b, gdn_at_b(vn, LDV, kk * 16, c0 + np * 16, l));
                gdn_mma(qh[2 * np], pa, b[0], b[1]);
                gdn_mma(qh[2 * np + 1], pa, b[2], b[3]);
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            if (i >= L) continue;
            float* o = out + (static_cast<long long>(ta + i + row0) * V_h + h) * V + v0 + c0 + 2 * t;
            #pragma unroll
            for (int nt = 0; nt < CT; ++nt) {
                *reinterpret_cast<float2*>(o + nt * 8) =
                    make_float2(qh[nt][2 * half], qh[nt][2 * half + 1]);
            }
        }
        __syncthreads();
        if (more) load_p(jc + 1);
        gdn_cp_commit();

        // The keys' group is in once the two after it alone may be pending.
        asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads();
        const float el = __expf(g_last);
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) hacc[nt][e] *= el;
        }
        #pragma unroll
        for (int kk = 0; kk < GDN_CH / 16; ++kk) {
            unsigned a[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                gdn_ldsm_t(a[mt], gdn_at_at(sk, GDN_LD, kk * 16, kr0 + mt * 16, l));
            }
            #pragma unroll
            for (int np = 0; np < CT / 2; ++np) {
                unsigned b[4];
                gdn_ldsm_t(b, gdn_at_b(vsc, LDV, kk * 16, c0 + np * 16, l));
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    gdn_mma(hacc[mt * CT + 2 * np], a[mt], b[0], b[1]);
                    gdn_mma(hacc[mt * CT + 2 * np + 1], a[mt], b[2], b[3]);
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < NT; ++q) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                *reinterpret_cast<unsigned*>(hs + h_row(q, half) * LDV + h_col(q)) =
                    gdn_pack(hacc[q][2 * half], hacc[q][2 * half + 1]);
            }
        }
    }
    gdn_cp_wait_all();

    if (write_state && row_persists(write_state_mask, lane.rl)) {
        #pragma unroll
        for (int q = 0; q < NT; ++q) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                *reinterpret_cast<unsigned*>(
                    state + static_cast<long long>(h_row(q, half)) * V + v0 + h_col(q)) =
                    gdn_pack(hacc[q][2 * half], hacc[q][2 * half + 1]);
            }
        }
    }
}


// The chunked rule in three wmma launches, for a card whose blocks cannot
// hold the scan's shared memory: per (chunk, value head) the update
// directions `w` and `u`; per (request, value head, sixteen value columns)
// the chunk-by-chunk state walk; per (chunk, value head) the output.
constexpr int GDN_WARPS = 8;

template <int BK, int BV>
__global__ void __launch_bounds__(256) ssm_gdn_chunk_wu(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    float* __restrict__ gcum,
    __nv_bfloat16* __restrict__ qb,
    __nv_bfloat16* __restrict__ kb_out,
    __nv_bfloat16* __restrict__ kT,
    __nv_bfloat16* __restrict__ w_out,
    __nv_bfloat16* __restrict__ u_out,
    int lanes, int K_h, int V_h,
    const u32* __restrict__ win)
{
    static_assert(BV == BK, "the value rows reuse the key buffer");
    namespace wmma = ::nvcuda::wmma;
    extern __shared__ float smem[];
    float* A = smem;
    __nv_bfloat16* kb = reinterpret_cast<__nv_bfloat16*>(smem + GDN_CH * GDN_CH);
    __nv_bfloat16* Tb = kb + GDN_CH * BK;
    float* G = reinterpret_cast<float*>(Tb + GDN_CH * GDN_CH);
    float* B = G + GDN_CH;
    float* BE = B + GDN_CH;
    float* arow = BE + GDN_CH;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, qo_indptr, begin_at, win);
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int j = s - (lane.t0 / GDN_CH + r);
    if (j < 0 || j * GDN_CH >= lane.T) return;
    const int ta = lane.t0 + j * GDN_CH;
    const int L = min(GDN_CH, lane.t0 + lane.T - ta);
    const int hk = h / (V_h / K_h);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < L; ++i) {
            acc += g_log[static_cast<long long>(ta + i) * V_h + h];
            G[i] = acc;
        }
    }
    for (int i = tid; i < L; i += blockDim.x) {
        B[i] = beta[static_cast<long long>(ta + i) * V_h + h];
    }
    __nv_bfloat16* kT_s = kT + (static_cast<long long>(s) * K_h + hk) * BK * GDN_CH;
    for (int e = tid; e < GDN_CH * BK; e += blockDim.x) {
        const int i = e / BK, kk = e % BK;
        const long long row = (static_cast<long long>(ta + i) * K_h + hk) * BK + kk;
        const __nv_bfloat16 kv = i < L ? __float2bfloat16(k_norm[row]) : __float2bfloat16(0.f);
        kb[e] = kv;
        kT_s[static_cast<long long>(kk) * GDN_CH + i] = kv;
        if (i < L) {
            kb_out[row] = kv;
            qb[row] = __float2bfloat16(q_norm[row]);
        }
    }
    __syncthreads();
    for (int i = tid; i < L; i += blockDim.x) {
        gcum[static_cast<long long>(ta + i) * V_h + h] = G[i];
        BE[i] = B[i] * __expf(G[i]);
    }
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        for (int tile = warp; tile < 16; tile += GDN_WARPS) {
            const int m0 = (tile >> 2) * 16, n0 = (tile & 3) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                wmma::load_matrix_sync(a, kb + m0 * BK + k0, BK);
                wmma::load_matrix_sync(b, kb + n0 * BK + k0, BK);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(A + m0 * GDN_CH + n0, c, GDN_CH, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        A[e] = (i < L && jj < i) ? A[e] * B[i] * __expf(G[i] - G[jj]) : 0.f;
    }
    __syncthreads();
    for (int i = 0; i < L; ++i) {
        if (tid < i) arow[tid] = A[i * GDN_CH + tid];
        __syncthreads();
        if (tid < GDN_CH) {
            const int c = tid;
            float t = (c == i) ? 1.f : 0.f;
            if (c < i) {
                for (int jj = c; jj < i; ++jj) t -= arow[jj] * A[jj * GDN_CH + c];
            }
            A[i * GDN_CH + c] = t;
        }
        __syncthreads();
    }
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        Tb[e] = __float2bfloat16((i < L && jj < L) ? A[e] * BE[jj] : 0.f);
    }
    __syncthreads();
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        float* mine = reinterpret_cast<float*>(kb) + warp * 256;
        const int lane_id = tid & 31;
        for (int tile = warp; tile < 4 * (BK / 16); tile += GDN_WARPS) {
            const int m0 = (tile / (BK / 16)) * 16, n0 = (tile % (BK / 16)) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, Tb + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, kT_s + n0 * GDN_CH + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(mine, c, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane_id; e < 256; e += 32) {
                const int i = m0 + e / 16, kk = n0 + e % 16;
                if (i < L) {
                    w_out[(static_cast<long long>(ta + i) * V_h + h) * BK + kk] = __float2bfloat16(mine[e]);
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        Tb[e] = __float2bfloat16((i < L && jj < L) ? A[e] : 0.f);
    }
    __nv_bfloat16* vbT = kb;
    for (int e = tid; e < GDN_CH * BV; e += blockDim.x) {
        const int i = e / BV, vv = e % BV;
        const float val = i < L ? v[(static_cast<long long>(ta + i) * V_h + h) * BV + vv] * B[i] : 0.f;
        vbT[vv * GDN_CH + i] = __float2bfloat16(val);
    }
    __syncthreads();
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        float* mine = A + warp * 256;
        const int lane_id = tid & 31;
        for (int tile = warp; tile < 4 * (BV / 16); tile += GDN_WARPS) {
            const int m0 = (tile / (BV / 16)) * 16, n0 = (tile % (BV / 16)) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, Tb + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, vbT + n0 * GDN_CH + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(mine, c, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane_id; e < 256; e += 32) {
                const int i = m0 + e / 16, vv = n0 + e % 16;
                if (i < L) {
                    u_out[(static_cast<long long>(ta + i) * V_h + h) * BV + vv] = __float2bfloat16(mine[e]);
                }
            }
            __syncwarp();
        }
    }
}

template <typename StateT, int BK, int BV, int BVT>
__global__ void __launch_bounds__(256) ssm_gdn_chunk_h(
    const float* __restrict__ gcum,
    const __nv_bfloat16* __restrict__ kT,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ u,
    __nv_bfloat16* __restrict__ vnT,
    __nv_bfloat16* __restrict__ hcT,
    StateT* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int lanes, int K_h, int V_h,
    const u32* __restrict__ win)
{
    static_assert(BVT == 16, "one value tile per mma column");
    namespace wmma = ::nvcuda::wmma;
    __shared__ float hf[BK * BVT];
    __shared__ float dl[BK * BVT];
    __shared__ float vn[GDN_CH * BVT];
    __shared__ __nv_bfloat16 hT[BVT * BK];
    __shared__ __nv_bfloat16 vnTs[BVT * GDN_CH];
    __shared__ float sE[GDN_CH];
    const int vt = static_cast<int>(blockIdx.x) % (BV / BVT);
    const int h = (static_cast<int>(blockIdx.x) / (BV / BVT)) % V_h;
    const int r = static_cast<int>(blockIdx.x) / ((BV / BVT) * V_h);
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (r >= lanes) return;
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int hk = h / (V_h / K_h);
    const int v0 = vt * BVT;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    StateT* state = state_base
        + static_cast<long long>(lane.slot) * slot_stride_elems
        + static_cast<long long>(h) * BK * BV;
    for (int e = tid; e < BK * BVT; e += blockDim.x) {
        const int kk = e / BVT, vv = e % BVT;
        hf[e] = state_load(state + static_cast<long long>(kk) * BV + v0 + vv);
    }
    __syncthreads();
    const int base = lane.t0 / GDN_CH + r;
    const int nchunks = (lane.T + GDN_CH - 1) / GDN_CH;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    for (int j = 0; j < nchunks; ++j) {
        const int sj = base + j;
        const int ta = lane.t0 + j * GDN_CH;
        const int L = min(GDN_CH, lane.t0 + lane.T - ta);
        __nv_bfloat16* hcT_s = hcT + (static_cast<long long>(sj) * V_h + h) * BV * BK;
        for (int e = tid; e < BVT * BK; e += blockDim.x) {
            const int vv = e / BK, kk = e % BK;
            const __nv_bfloat16 hv = __float2bfloat16(hf[kk * BVT + vv]);
            hT[e] = hv;
            hcT_s[static_cast<long long>(v0 + vv) * BK + kk] = hv;
        }
        if (tid < GDN_CH) {
            const float gl = gcum[static_cast<long long>(ta + L - 1) * V_h + h];
            sE[tid] = tid < L ? __expf(gl - gcum[static_cast<long long>(ta + tid) * V_h + h]) : 0.f;
        }
        __syncthreads();
        if (warp < 4) {
            const int m0 = warp * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                wmma::load_matrix_sync(a, w + (static_cast<long long>(ta + m0) * V_h + h) * BK + k0, V_h * BK);
                wmma::load_matrix_sync(b, hT + k0, BK);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(vn + m0 * BVT, c, BVT, wmma::mem_row_major);
        }
        __syncthreads();
        __nv_bfloat16* vnT_s = vnT + (static_cast<long long>(sj) * V_h + h) * BV * GDN_CH;
        for (int e = tid; e < GDN_CH * BVT; e += blockDim.x) {
            const int i = e / BVT, vv = e % BVT;
            float vnew = 0.f;
            if (i < L) {
                const long long ui = (static_cast<long long>(ta + i) * V_h + h) * BV + v0 + vv;
                vnew = __bfloat162float(u[ui]) - vn[e];
                u[ui] = __float2bfloat16(vnew);
            }
            const __nv_bfloat16 vb = __float2bfloat16(vnew);
            vnT_s[static_cast<long long>(v0 + vv) * GDN_CH + i] = vb;
            vnTs[vv * GDN_CH + i] = __float2bfloat16(vnew * sE[i]);
        }
        __syncthreads();
        {
            const int m0 = warp * 16;
            wmma::fill_fragment(c, 0.f);
            const __nv_bfloat16* kT_s = kT + (static_cast<long long>(sj) * K_h + hk) * BK * GDN_CH;
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, kT_s + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, vnTs + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(dl + m0 * BVT, c, BVT, wmma::mem_row_major);
        }
        __syncthreads();
        const float gl = __expf(gcum[static_cast<long long>(ta + L - 1) * V_h + h]);
        for (int e = tid; e < BK * BVT; e += blockDim.x) {
            hf[e] = hf[e] * gl + dl[e];
        }
        __syncthreads();
    }
    if (write_state && row_persists(write_state_mask, lane.rl)) {
        for (int e = tid; e < BK * BVT; e += blockDim.x) {
            const int kk = e / BVT, vv = e % BVT;
            state_store(state + static_cast<long long>(kk) * BV + v0 + vv, hf[e]);
        }
    }
}

template <int BK, int BV, int BVT>
__global__ void __launch_bounds__(256) ssm_gdn_chunk_o(
    const float* __restrict__ gcum,
    const __nv_bfloat16* __restrict__ qb,
    const __nv_bfloat16* __restrict__ kb,
    const __nv_bfloat16* __restrict__ vnT,
    const __nv_bfloat16* __restrict__ hcT,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    float* __restrict__ out,
    int lanes, int K_h, int V_h,
    const u32* __restrict__ win)
{
    static_assert(BVT == 16, "one value tile per mma column");
    namespace wmma = ::nvcuda::wmma;
    __shared__ float S[GDN_CH * GDN_CH];
    __shared__ __nv_bfloat16 Sb[GDN_CH * GDN_CH];
    __shared__ float o1[GDN_CH * BVT];
    __shared__ float o2[GDN_CH * BVT];
    __shared__ float sG[GDN_CH];
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, qo_indptr, begin_at, win);
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int j = s - (lane.t0 / GDN_CH + r);
    if (j < 0 || j * GDN_CH >= lane.T) return;
    const int ta = lane.t0 + j * GDN_CH;
    const int L = min(GDN_CH, lane.t0 + lane.T - ta);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int hk = h / (V_h / K_h);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    for (int i = tid; i < GDN_CH; i += blockDim.x) {
        sG[i] = i < L ? gcum[static_cast<long long>(ta + i) * V_h + h] : 0.f;
    }
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    const __nv_bfloat16* q_s = qb + (static_cast<long long>(ta) * K_h + hk) * BK;
    const __nv_bfloat16* k_s = kb + (static_cast<long long>(ta) * K_h + hk) * BK;
    for (int tile = warp; tile < 16; tile += GDN_WARPS) {
        const int m0 = (tile >> 2) * 16, n0 = (tile & 3) * 16;
        wmma::fill_fragment(c, 0.f);
        for (int k0 = 0; k0 < BK; k0 += 16) {
            wmma::load_matrix_sync(a, q_s + static_cast<long long>(m0) * K_h * BK + k0, K_h * BK);
            wmma::load_matrix_sync(b, k_s + static_cast<long long>(n0) * K_h * BK + k0, K_h * BK);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(S + m0 * GDN_CH + n0, c, GDN_CH, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        Sb[e] = __float2bfloat16((i < L && jj <= i) ? S[e] * __expf(sG[i] - sG[jj]) : 0.f);
    }
    __syncthreads();
    const __nv_bfloat16* hcT_s = hcT + (static_cast<long long>(s) * V_h + h) * BV * BK;
    const __nv_bfloat16* vnT_s = vnT + (static_cast<long long>(s) * V_h + h) * BV * GDN_CH;
    for (int vt = 0; vt < BV / BVT; ++vt) {
        const int v0 = vt * BVT;
        if (warp < 4) {
            const int m0 = warp * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                wmma::load_matrix_sync(a, q_s + static_cast<long long>(m0) * K_h * BK + k0, K_h * BK);
                wmma::load_matrix_sync(b, hcT_s + static_cast<long long>(v0) * BK + k0, BK);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(o1 + m0 * BVT, c, BVT, wmma::mem_row_major);
        } else {
            const int m0 = (warp - 4) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, Sb + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, vnT_s + static_cast<long long>(v0) * GDN_CH + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(o2 + m0 * BVT, c, BVT, wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < GDN_CH * BVT; e += blockDim.x) {
            const int i = e / BVT, vv = e % BVT;
            if (i < L) {
                out[(static_cast<long long>(ta + i + row0) * V_h + h) * BV + v0 + vv]
                    = o1[e] * __expf(sG[i]) + o2[e];
            }
        }
        __syncthreads();
    }
}

}

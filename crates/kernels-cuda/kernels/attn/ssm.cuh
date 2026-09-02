#pragma once

#include <cuda_bf16.h>

#include "prelude/device.cuh"

namespace pie::attn {

using f32 = float;
using state_bf16 = __nv_bfloat16;

constexpr int gqa_smem_bv = 128;

__device__ __forceinline__ float silu_f(float z) {
    return z / (1.f + __expf(-z));
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

template <class T>
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
    const u32* __restrict__ win)
{
    const int c = blockIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    // **THE STAGED-GEOMETRY SEAT, ON THE LANE AXIS** (the chunked-arm wave).
    // This grid counts REQUESTS, not rows, so the word that retires a ceiling
    // grid's padding is `win[2]` — the window's live LANE count — and not
    // `win[0]`, which is its row count and belongs to the row-gridded entries.
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    // **AND THE LANE SPLIT, WHICH IS THIS WAVE'S CRUX.** Two kinds of per-lane
    // vector arrive here and they are indexed DIFFERENTLY:
    //
    //   * `qo_indptr` is the WINDOW'S OWN CSR — rebased, staged into the
    //     fixed-stride window blob at an address a body may bake — so it is
    //     read at the window-local ordinal `r` and at nothing else.
    //   * `slot_ids`, `write_state_mask`, `commit_len` and `begin_at` are the
    //     FIRE'S tables, handed over whole under a plane base
    //     (`Run::recurrent_absolute`) because `lane_offset` is not a function
    //     of a body key and a sliced pointer would be stale on every replay
    //     but its recording one. Those are read at `r + win[3]`.
    //
    // Unarmed, `rl == r` and the tables arrive sliced, which is the launch
    // this kernel has always made.
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    // And the ROW axis, for the two planes: `x` and `y` are the fire's
    // activations, handed as PLANE bases under an armed seat, while the CSR
    // above counts from the window's zero. `win[1]` is the bridge, added once,
    // at the pointer.
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;

    // THE SEGMENT THIS LAUNCH OWNS (the 2R split): `begin_at` cuts the front
    // and `commit_len` cuts the back, and the two are never bound together —
    // the head runs `[0, n)` and folds, the tail runs `[n, rows)` and does
    // not. The front cut moves the row origin, so every index below is the
    // segment's own.
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

    // The state slab is addressed by the slot's VALUE — a bank id, not a
    // position in this fire — so it is never shifted by anything here.
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
        y_r[t * C + c] = Elem<T>::from_f32(silu_f(acc));
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

template <class T>
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
    const u32* __restrict__ win)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= C) return;

    // The seat, the lane split and the row bridge — the per-channel form
    // above carries the whole argument, and this arm differs only in how it
    // tiles the channels. `win[2]` retires the padded lanes, `win[3]` turns
    // this window's request number into a fire lane for the fire-wide tables,
    // the window's own CSR stays on `r`, and `win[1]` shifts the two planes.
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int Nr = static_cast<int>(qo_indptr[r + 1]) - t0;

    // The segment this launch owns — see the per-channel form above.
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

    // Addressed by the slot's VALUE, so never shifted.
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
        y_r[static_cast<long long>(t) * C + c] = Elem<T>::from_f32(silu_f(acc));
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

template <class T>
__global__ void ssm_causal_conv1d_update_batched(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    T* __restrict__ y,
    int R, int C, int K, int dil,
    const u32* __restrict__ win)
{
    const int r = blockIdx.y;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns — the
    // channel row in and the channel row out. The slot table is the LANES',
    // and a lane ordinal is not a row.
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R || c >= C) return;

    const int slot = slot_ids[r];
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
    y_r[c] = Elem<T>::from_f32(silu_f(acc));

    const int span = (K - 1) * dil;
    for (int k = 0; k < span; ++k) {
        state[k * C + c] = state[(k + 1) * C + c];
    }
    state[span * C + c] = Elem<T>::from_f32(new_x);
}

// Eight channels per thread, dilation 1: the row, the window and the
// weights move as 16-byte vectors, every read issued before any arithmetic,
// and the shift is written back from the window already in registers.
template <class T>
__global__ void ssm_causal_conv1d_update_batched_vec8(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    long long slot_stride_elems,
    T* __restrict__ y,
    int R, int C, int K,
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

    const int slot = slot_ids[r];
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
    // `weight` is `[C][K]`: this thread's eight channels are `K` vectors.
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
    for (int u = 0; u < VEC; ++u) yv.e[u] = Elem<T>::from_f32(silu_f(acc[u]));
    *reinterpret_cast<uint4*>(y + (long long)r_row * C + c0) = yv.raw;

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
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
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
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. The
    // projection is read there; the two normed planes it lands in are the
    // fire's own scratch, which starts at its own zero.
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
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns — the
    // projection it folds and the gate row it lands share that row axis.
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
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. The
    // projection and the fused gate row are read there; the three planes this
    // lands in are the fire's own scratch, which starts at its own zero.
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

__device__ __forceinline__ bool row_persists(
    const u8* __restrict__ mask, int r) {
    return mask == nullptr || mask[r] != 0;
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
    // **THE SEAT, AND THE LANE SPLIT** (the chunked-arm wave). This grid
    // counts REQUESTS, so `win[2]` is the live lane count that retires a
    // ceiling grid's padding. And the two kinds of per-lane vector are read at
    // two different indices: `qo_indptr` is the WINDOW's own rebased CSR and
    // stays on `r`; `slot_ids` is the FIRE's table, handed whole under a plane
    // base so a body cannot bake a stale slice of it, and is read at
    // `r + win[3]`.
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    // **AND THE ROW AXIS SPLITS TOO, WHICH IS WHY IT IS TWO INDICES BELOW.**
    // The five staged planes (`q_norm` .. `beta`) are this fire's own SCRATCH,
    // laid by the preps at the launch-local row (`ssm_gdn_prep_qk_norm` writes
    // `n` and reads `n + win[1]`), so they are addressed off the rebased CSR
    // with nothing added. `out` is the fire's activation plane, handed as a
    // BASE under an armed seat, so it takes `win[1]`. The state slab is
    // addressed by the slot's VALUE and is shifted by neither.
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
        // The scratch row is `bh`; the PLANE row is `bh` shifted — see the
        // seat's note above.
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

    // The seat, the lane split and the row bridge — the plain chunked scan
    // above carries the argument in full. `win[2]` retires padded lanes;
    // `slot_ids` and the fold predicate are the FIRE's tables at `r + win[3]`;
    // the window's own CSR stays on `r`; the five staged planes are scratch at
    // the launch-local row and `out` is a plane base at `win[1]`.
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
            // `vh` is the SCRATCH row; the plane row is the shifted one.
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
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. Only the
    // accumulator is one — the five prep planes are this fire's own scratch,
    // written at the same launch-local row this reads them at, and the slot
    // table is the lanes'.
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
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
            state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t = v_h[v_idx];
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
    const u32* __restrict__ win)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. Only the
    // accumulator is one — the five prep planes are this fire's own scratch,
    // written at the same launch-local row this reads them at, and the slot
    // table is the lanes'.
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int slot = slot_ids[r];
    if (slot < 0) return;

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
        } else {
            state[(long long)k * V_d + v_idx] = __float2bfloat16(s);
        }
    }
    out_bh[v_idx] = out_v;
    if (vec_tile) {
        __syncthreads();
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(s_state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    }
}

// ---- fused decode step ------------------------------------------------------
//
// One launch per step: q/k are L2-normed in the block, v and the gates are
// read where the projection landed them, and each block owns a `K_d x BV`
// tile of the head's state, so a one-token fire still spreads the state over
// the device. Same arithmetic as the smem step above: the decayed state is
// rounded to bf16 before the update, `kv_mem` reads it unrounded.

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

// Sums `x` over the rows of the tile: lanes `TPR` apart in a warp hold the
// same columns, the warps meet in `red`, and the first `BV` threads fold the
// warps into `tot`. Every thread leaves with the totals of its own eight
// columns.
template <int TPR, int WARPS, int BV>
__device__ __forceinline__ void gdn_column_sum(
    float (&x)[8], float (*red)[BV], float* tot, int tid, int lane, int warp) {
    #pragma unroll
    for (int off = TPR; off < 32; off <<= 1) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) x[u] += __shfl_xor_sync(0xffffffffu, x[u], off);
    }
    __syncthreads();
    if (lane < TPR) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) red[warp][lane * 8 + u] = x[u];
    }
    __syncthreads();
    if (tid < BV) {
        float acc = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) acc += red[w][tid];
        tot[tid] = acc;
    }
    __syncthreads();
    const int c = (tid % TPR) * 8;
    #pragma unroll
    for (int u = 0; u < 8; ++u) x[u] = tot[c + u];
}

template <int BV>
__global__ void __launch_bounds__(256) ssm_gdn_decode_step(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    float q_scale,
    const u32* __restrict__ win)
{
    constexpr int BLOCK = 256;
    constexpr int VEC = 8;
    constexpr int TPR = BV / VEC;
    constexpr int RPP = BLOCK / TPR;
    constexpr int WARPS = BLOCK / 32;
    static_assert(BV % VEC == 0 && BLOCK % TPR == 0 && TPR <= 32 && 32 % TPR == 0,
                  "a tile row is a whole number of eight-wide vectors");

    const int vt = blockIdx.x;
    const int r = blockIdx.y;
    const int h = blockIdx.z;
    // The staged-geometry seat (qkv_fused.cuh's idiom): the live-rows word
    // retires a bucket's padded rows, `win[1]` is the plane row the launch's
    // first row stands at; the slot table is the lanes'.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int slot = slot_ids[r];
    if (slot < 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int h_k = h / (V_h / K_h);
    const int K_dim = K_h * K_d;
    const bf16* row = qkv_post + (long long)r_row * conv_dim;
    const bf16* q = row + (long long)h_k * K_d;
    const bf16* k = row + K_dim + (long long)h_k * K_d;

    __shared__ float sq[RPP];
    __shared__ float sk[RPP];
    __shared__ float red[WARPS][BV];
    __shared__ float tot[BV];
    __shared__ float norms[2];

    // Every global read is issued here, before the first barrier, so the
    // block pays one memory latency rather than one per phase.
    const int i = tid / TPR;
    const int j0 = vt * BV + (tid % TPR) * VEC;
    const bool live = i < K_d;
    __nv_bfloat16* cell = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d
        + (long long)i * V_d + j0;
    float qv = 0.f;
    float kv = 0.f;
    if (tid < K_d) {
        qv = Elem<bf16>::to_f32(q[tid]);
        kv = Elem<bf16>::to_f32(k[tid]);
    }
    float s[VEC];
    if (live) {
        gdn_load8(cell, s);
    } else {
        #pragma unroll
        for (int u = 0; u < VEC; ++u) s[u] = 0.f;
    }
    float vv[VEC];
    gdn_load8(reinterpret_cast<const __nv_bfloat16*>(row + 2 * K_dim + (long long)h * V_d + j0), vv);
    const float* gate_row = gates + (long long)r_row * 2 * V_h;
    const float g = __expf(gate_row[h]);
    const float beta = gate_row[V_h + h];

    const float qs = warp_sum(qv * qv);
    const float ks = warp_sum(kv * kv);
    if (lane == 0) {
        red[warp][0] = qs;
        red[warp][1] = ks;
    }
    __syncthreads();
    if (tid == 0) {
        float a = 0.f;
        float b = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            a += red[w][0];
            b += red[w][1];
        }
        norms[0] = rsqrtf(a + 1e-6f) * q_scale;
        norms[1] = rsqrtf(b + 1e-6f);
    }
    __syncthreads();
    if (tid < K_d) {
        sq[tid] = qv * norms[0];
        sk[tid] = kv * norms[1];
    }
    __syncthreads();
    const float ki = live ? sk[i] : 0.f;
    const float qi = live ? sq[i] : 0.f;

    float x[VEC];
    #pragma unroll
    for (int u = 0; u < VEC; ++u) x[u] = s[u] * g * ki;
    gdn_column_sum<TPR, WARPS, BV>(x, red, tot, tid, lane, warp);

    #pragma unroll
    for (int u = 0; u < VEC; ++u) {
        const float delta = (vv[u] - x[u]) * beta;
        const float sg = __bfloat162float(__float2bfloat16(s[u] * g));
        const float sn = sg + ki * delta;
        s[u] = sn;
        x[u] = sn * qi;
    }
    if (live) gdn_store8(cell, s);
    gdn_column_sum<TPR, WARPS, BV>(x, red, tot, tid, lane, warp);
    if (tid < TPR) {
        float* o = out + ((long long)r_row * V_h + h) * V_d + j0;
        #pragma unroll
        for (int u = 0; u < VEC; ++u) o[u] = x[u];
    }
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

    // The seat, the lane split and the row bridge — the plain chunked scan
    // carries the argument in full, and this arm's request axis is `blockIdx.y`
    // rather than `x`. `win[2]` retires the lanes a ceiling grid padded in;
    // the four RS tables and `slot_ids` are the FIRE's, read at `r + win[3]`;
    // `qo_indptr` is the window's own and stays on `r`; the five staged planes
    // are scratch at the launch-local row and `out` is a plane base at
    // `win[1]`. The state slab is a slot VALUE and moves for nothing.
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int T  = static_cast<int>(qo_indptr[r + 1]) - t0;

    // THE SEGMENT THIS LAUNCH OWNS (the 2R split), exactly as the chunked
    // conv reads it: `begin_at` cuts the front, `commit_len` cuts the back.
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

    // **THE ROUNDING POLICY IS ITS OWN ARGUMENT** (alto F3b).
    //
    // This was `commit_len != nullptr`, which read a LENGTH as a rounding:
    // `single_round` folds the decay into the update instead of rounding the
    // decayed state to bf16 first, so binding the length seat changed the
    // NUMBERS as well as the count and a replay that accepted its whole
    // window stopped being the fold it replaced. The two are now two
    // arguments, and the shell binds the fold's own policy on every path —
    // which is what makes a bound-but-non-truncating seat exact and a
    // truncated fold exact against a shorter buffer.
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
        // `bh` is the SCRATCH row; the plane row is the shifted one.
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
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. The two
    // projections are read there; the gates are this fire's own scratch, laid
    // at the launch-local row the step will read them back at.
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
    // q and k are L2-normed PER HEAD, and q is scaled by head_dim^-1/2 (the
    // reference recurrence's `scale`); v is widened as stored.
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
    for (int h = tid; h < heads && h < kMaxHeads; h += BLOCK) sums[h] = 0.f;
    __syncthreads();
    for (int i = tid; i < width; i += BLOCK) {
        const float x = Elem<ElemT>::to_f32(src[i]);
        atomicAdd(&sums[(i / head_dim) % kMaxHeads], x * x);
    }
    __syncthreads();
    const float q_scale = (plane == 0) ? rsqrtf(static_cast<float>(head_dim)) : 1.f;
    for (int i = tid; i < width; i += BLOCK) {
        const float inv = rsqrtf(sums[(i / head_dim) % kMaxHeads] + eps);
        dst[i] = Elem<ElemT>::to_f32(src[i]) * inv * q_scale;
    }
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
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. Only the
    // accumulator is one — the five prep planes are this fire's own scratch,
    // written at the same launch-local row this reads them at, and the slot
    // table is the lanes'.
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
    float* st = state_base + (long long)slot * slot_stride_elems +
                (long long)h * D * D;
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
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    // The seat, the lane split and the row bridge (the chunked-arm wave; the
    // chunked delta scan carries the argument in full). This grid counts
    // REQUESTS, so `win[2]` is what retires a ceiling grid's padded lanes.
    // `qo_indptr` is the WINDOW's own rebased CSR and stays on `r`; `slot_ids`
    // is the FIRE's table, handed whole so a body cannot bake a stale slice,
    // and is read at `r + win[3]`. The five staged planes are this fire's
    // SCRATCH at the launch-local row — `ssm_kda_qkv_prep` writes them at `n`
    // and reads the projection at `n + win[1]` — so they take no shift, while
    // `out` is the activation PLANE and takes `win[1]`. The state slab is
    // addressed by the slot's VALUE and is shifted by neither.
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;

    const long long begin = qo_indptr[r];
    const long long end = qo_indptr[r + 1];
    if (end <= begin) return;

    float* st = state_base + (long long)slot_ids[rl] * slot_stride_elems +
                (long long)h * D * D;

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
            // `th` is the SCRATCH row; the plane row is the shifted one.
            if (lane == 0) out[((t + row0) * H + h) * D + vi] = acc;
        }

        __syncthreads();
    }
}

}

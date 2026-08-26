#pragma once

#include "prelude/device.cuh"

#include "prelude/kv_paged_addr.cuh"

#include "prelude/rope.cuh"

namespace pie::elemwise {

template <class T>
using Elem = ::pie::Elem<T>;

template <class P>
__global__ void standard_table(
    const P* __restrict__ positions,
    float* __restrict__ table,
    int head_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int half = head_dim / 2;
    const int pos = static_cast<int>(positions[n]);
    float* row = table + static_cast<long long>(n) * head_dim;
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += blockDim.x) {
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = cos_v;
        row[dim_pair + half] = sin_v;
    }
}

template <bool kWriteKv, bool kHnd>
__global__ void rope_full(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    bool interleaved,
    int cache_pairs,
    int heads_per_block,
    const bf16* __restrict__ v,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int R,
    int page_size)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;

    const int half = head_dim / 2;
    const int pos = positions[n];

    KvSlot slot{};
    bool write_this_row = false;
    if constexpr (kWriteKv) {
        write_this_row = (row_valid == nullptr) || (row_valid[n] != 0);
        if (write_this_row) {
            slot = kv_slot_for_token(qo_indptr, kv_page_indices, kv_page_indptr,
                                     kv_last_page_lens, n, R, page_size);
        }
    }

    extern __shared__ float rope_cs[];
    const int cached = cache_pairs;
    for (int dp = threadIdx.x; dp < cached; dp += blockDim.x) {
        const float freq = powf(theta, -2.f * static_cast<float>(dp) /
                                       static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        rope_cs[dp] = c;
        rope_cs[cached + dp] = s;
    }
    if (cached > 0) __syncthreads();

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;

        float cos_v, sin_v;
        if (dim_pair < cached) {
            cos_v = rope_cs[dim_pair];
            sin_v = rope_cs[cached + dim_pair];
        } else {
            const float freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                           static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }

        if (head_idx < num_q_heads) {
            bf16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cos_v, sin_v);
            else rotate_pair(qp, half, dim_pair, cos_v, sin_v);
            continue;
        }
        {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cos_v, sin_v);
            else rotate_pair(kp, half, dim_pair, cos_v, sin_v);
            if constexpr (kWriteKv) {
                if (write_this_row) {

                    const int j0 = interleaved ? dim_pair * 2 : dim_pair;
                    const int j1 = interleaved ? dim_pair * 2 + 1
                                               : dim_pair + half;
                    const bf16* vp =
                        v + (static_cast<long long>(n) * num_kv_heads + kv_h) *
                                head_dim;
                    const int base = kv_h * head_dim;
                    const long long d0 = kv_dst_index<kHnd>(
                        slot, base + j0, page_size, num_kv_heads, head_dim);
                    const long long d1 = kv_dst_index<kHnd>(
                        slot, base + j1, page_size, num_kv_heads, head_dim);
                    k_pages[d0] = kp[j0];
                    k_pages[d1] = kp[j1];
                    v_pages[d0] = vp[j0];
                    v_pages[d1] = vp[j1];
                }
            }
        }
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rotate(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rotate_rounded(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const bf16 norm_a = f32_to_bf16(
            bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]));
        const bf16 norm_b = f32_to_bf16(
            bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]));
        const float a = bf16_to_f32(norm_a);
        const float b = bf16_to_f32(norm_b);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rotate_mrope(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    int s0, int s1, int s2)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos_t = positions[3 * n + 0];
    const int pos_h = positions[3 * n + 1];
    const int pos_w = positions[3 * n + 2];
    (void)s0;
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const bf16 norm_a = f32_to_bf16(
            bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]));
        const bf16 norm_b = f32_to_bf16(
            bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]));
        const float a = bf16_to_f32(norm_a);
        const float b = bf16_to_f32(norm_b);

        int axis_pos;
        const int m = dim_pair % 3;
        if (m == 1 && dim_pair < 3 * s1)      axis_pos = pos_h;
        else if (m == 2 && dim_pair < 3 * s2) axis_pos = pos_w;
        else                                  axis_pos = pos_t;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(axis_pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rotate_devwin(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const u32* __restrict__ win,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    {
        const int w0 = static_cast<int>(win[0]);
        const int w1 = static_cast<int>(win[1]);
        if (n < w0 || n >= w0 + w1) return;
    }
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

__device__ __forceinline__ float yarn_freq(
    float base_freq, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos)
{
    constexpr float TWO_PI = 6.2831853071795864769f;
    const float wavelen   = TWO_PI / base_freq;
    const float low_wave  = orig_max_pos / low_freq_factor;
    const float high_wave = orig_max_pos / high_freq_factor;
    if (wavelen < high_wave) return base_freq;
    if (wavelen > low_wave)  return base_freq / factor;
    const float smooth = (orig_max_pos / wavelen - low_freq_factor) /
                         (high_freq_factor - low_freq_factor);
    return (1.f - smooth) * (base_freq / factor) + smooth * base_freq;
}

__global__ void rotate_yarn(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos,
    int heads_per_block)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;

        const float base_freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float freq = yarn_freq(base_freq, factor,
                                     low_freq_factor, high_freq_factor,
                                     orig_max_pos);
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            bf16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            rotate_pair(qp, half, dim_pair, cos_v, sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            rotate_pair(kp, half, dim_pair, cos_v, sin_v);
        }
    }
}

__global__ void rope_yarn(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_dim, float high_dim,
    float mscale,
    bool interleaved,
    int heads_per_block,
    int cache_pairs)
{
    extern __shared__ float2 yarn_cs[];
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    auto angle = [&](int d) -> float2 {
        const float base_freq = powf(theta,
            -2.f * static_cast<float>(d) / static_cast<float>(head_dim));
        const float freq = yarn_original_freq(base_freq, factor,
                                              low_dim, high_dim, d);
        float cos_v, sin_v;
        __sincosf(static_cast<float>(pos) * freq, &sin_v, &cos_v);
        return make_float2(cos_v * mscale, sin_v * mscale);
    };
    for (int d = threadIdx.x; d < cache_pairs; d += blockDim.x) {
        yarn_cs[d] = angle(d);
    }
    if (cache_pairs > 0) __syncthreads();

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;
        const float2 cs = dim_pair < cache_pairs ? yarn_cs[dim_pair]
                                                 : angle(dim_pair);

        if (head_idx < num_q_heads) {
            bf16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cs.x, cs.y);
            else             rotate_pair(qp, half, dim_pair, cs.x, cs.y);
        } else {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cs.x, cs.y);
            else             rotate_pair(kp, half, dim_pair, cs.x, cs.y);
        }
    }
}

template <class T>
__global__ void rope_partial(
    T* __restrict__ q,
    T* __restrict__ k,
    const i32* __restrict__ positions,
    int position_delta,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;
    const int pos = positions[n] + position_delta;

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        float cos_v = 1.f, sin_v = 0.f;
        if (dim_pair < rope_angles) {
            const float freq = powf(theta,
                -2.f * static_cast<float>(dim_pair) /
                       static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }

        if (dim_pair >= rope_angles) continue;

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}

__global__ void rope_partial_last(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    bool inverse,
    bool interleaved,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int rope_half = rotary_dim / 2;
    const int offset = head_dim - rotary_dim;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * rope_half; t += blockDim.x) {
        const int head_idx = t / rope_half;
        const int dim_pair = t % rope_half;

        float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor,
                                      yarn_low_dim, yarn_high_dim, dim_pair);
        }
        const float ang = (inverse ? -1.f : 1.f) * static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        const bool is_q = (head_idx < num_q_heads);
        bf16* base = is_q
            ? q + static_cast<long long>(n * num_q_heads + head_idx) * head_dim
            : k + static_cast<long long>(n * num_kv_heads + (head_idx - num_q_heads)) * head_dim;

        const int i = interleaved ? offset + 2 * dim_pair : offset + dim_pair;
        const int j = interleaved ? offset + 2 * dim_pair + 1
                                  : offset + dim_pair + rope_half;
        const float a = bf16_to_f32(base[i]);
        const float b = bf16_to_f32(base[j]);
        base[i] = f32_to_bf16(a * cos_v - b * sin_v);
        base[j] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}
}

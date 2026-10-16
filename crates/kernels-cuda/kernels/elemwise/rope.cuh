#pragma once

#include "prelude/device.cuh"

#include "prelude/kv_paged_addr.cuh"

#include "prelude/rope.cuh"

#include "attn/dsv4_kv.cuh"

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
    int page_size,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total_heads = num_q_heads + num_kv_heads;

    const int half = head_dim / 2;
    const int pos = positions[row];

    KvSlot slot{};
    bool write_this_row = false;
    if constexpr (kWriteKv) {
        write_this_row = (row_valid == nullptr) || (row_valid[row] != 0);
        if (write_this_row) {
            slot = kv_slot_for_token(qo_indptr, kv_page_indices, kv_page_indptr,
                                     kv_last_page_lens, row, R, page_size);
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
            bf16* qp = q + (static_cast<long long>(row) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cos_v, sin_v);
            else rotate_pair(qp, half, dim_pair, cos_v, sin_v);
            continue;
        }
        {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(row) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cos_v, sin_v);
            else rotate_pair(kp, half, dim_pair, cos_v, sin_v);
            if constexpr (kWriteKv) {
                if (write_this_row) {

                    const int j0 = interleaved ? dim_pair * 2 : dim_pair;
                    const int j1 = interleaved ? dim_pair * 2 + 1
                                               : dim_pair + half;
                    const bf16* vp =
                        v + (static_cast<long long>(row) * num_kv_heads + kv_h) *
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
    const u32* __restrict__ devwin,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;

    {
        const int w0 = static_cast<int>(devwin[0]);
        const int w1 = static_cast<int>(devwin[1]);
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

// YaRN over q and k in place; with kWriteKv, each rotated k pair and its v
// pair land in the paged (NHD) cache at the row's explicit page and offset,
// as `kv_append_explicit` lands them.
template <bool kWriteKv>
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
    int cache_pairs,
    const bf16* __restrict__ v,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    const u32* __restrict__ win)
{
    extern __shared__ float2 yarn_cs[];
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    KvSlot slot{};
    bool write_this_row = false;
    if constexpr (kWriteKv) {
        slot.page = static_cast<int>(w_page[row]);
        slot.offset_in_page = static_cast<int>(w_off[row]);
        write_this_row = (row_valid == nullptr || row_valid[row] != 0) &&
                         slot.offset_in_page >= 0 && slot.offset_in_page < page_size;
    }

    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[row];

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
            bf16* qp = q + (static_cast<long long>(row) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cs.x, cs.y);
            else             rotate_pair(qp, half, dim_pair, cs.x, cs.y);
        } else {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(row) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cs.x, cs.y);
            else             rotate_pair(kp, half, dim_pair, cs.x, cs.y);
            if constexpr (kWriteKv) {
                if (write_this_row) {
                    const int j0 = interleaved ? dim_pair * 2 : dim_pair;
                    const int j1 = interleaved ? dim_pair * 2 + 1 : dim_pair + half;
                    const bf16* vp =
                        v + (static_cast<long long>(row) * num_kv_heads + kv_h) * head_dim;
                    const int base = kv_h * head_dim;
                    const long long d0 =
                        kv_dst_index<false>(slot, base + j0, page_size, num_kv_heads, head_dim);
                    const long long d1 =
                        kv_dst_index<false>(slot, base + j1, page_size, num_kv_heads, head_dim);
                    k_pages[d0] = kp[j0];
                    k_pages[d1] = kp[j1];
                    v_pages[d0] = vp[j0];
                    v_pages[d1] = vp[j1];
                }
            }
        }
    }
}

// `rope_yarn<true>` reading q, k and v straight out of a packed
// `[rows, q | k | v]` row: each rotated q pair lands in `q_out`, each rotated
// k pair and its v pair in the paged (NHD) cache. The arithmetic is
// `rope_yarn`'s pair for pair, so the same bits land as split, rotate, write.
__global__ void rope_yarn_split_append(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    const i32* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_dim, float high_dim,
    float mscale,
    bool interleaved,
    int heads_per_block,
    int cache_pairs,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    const u32* __restrict__ win)
{
    extern __shared__ float2 yarn_cs[];
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    KvSlot slot{};
    slot.page = static_cast<int>(w_page[row]);
    slot.offset_in_page = static_cast<int>(w_off[row]);
    const bool write_this_row = (row_valid == nullptr || row_valid[row] != 0) &&
                                slot.offset_in_page >= 0 && slot.offset_in_page < page_size;

    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[row];
    const long long stride = static_cast<long long>(num_q_heads + 2 * num_kv_heads) * head_dim;
    const bf16* src = packed + static_cast<long long>(row) * stride;

    auto angle = [&](int d) -> float2 {
        const float base_freq = powf(theta,
            -2.f * static_cast<float>(d) / static_cast<float>(head_dim));
        const float freq = yarn_original_freq(base_freq, factor,
                                              low_dim, high_dim, d);
        float cos_v, sin_v;
        __sincosf(static_cast<float>(pos) * freq, &sin_v, &cos_v);
        return make_float2(cos_v * mscale, sin_v * mscale);
    };
    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    if (heads_here * half <= static_cast<int>(blockDim.x)) {
        // A pair a thread: its operands load before the angles are cached,
        // so the row's reads overlap the angle math and the barrier. The
        // arithmetic is the loop's below, pair for pair.
        const int t = threadIdx.x;
        const bool live = t < heads_here * half;
        const int head_idx = head_base + (live ? t / half : 0);
        const int dim_pair = live ? t % half : 0;
        const int j0 = interleaved ? dim_pair * 2 : dim_pair;
        const int j1 = interleaved ? dim_pair * 2 + 1 : dim_pair + half;
        const bool is_q = head_idx < num_q_heads;
        // A k head sits where its index says, past the q heads; its v head
        // the kv heads further on.
        const bf16* x = src + static_cast<long long>(head_idx) * head_dim;
        float a = 0.f, b = 0.f;
        bf16 v0{}, v1{};
        if (live && (is_q || write_this_row)) {
            a = bf16_to_f32(x[j0]);
            b = bf16_to_f32(x[j1]);
            if (!is_q) {
                const bf16* vs = x + static_cast<long long>(num_kv_heads) * head_dim;
                v0 = vs[j0];
                v1 = vs[j1];
            }
        }
        for (int d = threadIdx.x; d < cache_pairs; d += blockDim.x) {
            yarn_cs[d] = angle(d);
        }
        if (cache_pairs > 0) __syncthreads();
        if (!live) return;
        const float2 cs = dim_pair < cache_pairs ? yarn_cs[dim_pair] : angle(dim_pair);
        const bf16 r0 = f32_to_bf16(a * cs.x - b * cs.y);
        const bf16 r1 = f32_to_bf16(b * cs.x + a * cs.y);
        if (is_q) {
            bf16* qd = q_out + (static_cast<long long>(row) * num_q_heads + head_idx) * head_dim;
            qd[j0] = r0;
            qd[j1] = r1;
        } else if (write_this_row) {
            const int base = (head_idx - num_q_heads) * head_dim;
            const long long d0 =
                kv_dst_index<false>(slot, base + j0, page_size, num_kv_heads, head_dim);
            const long long d1 =
                kv_dst_index<false>(slot, base + j1, page_size, num_kv_heads, head_dim);
            k_pages[d0] = r0;
            k_pages[d1] = r1;
            v_pages[d0] = v0;
            v_pages[d1] = v1;
        }
        return;
    }
    for (int d = threadIdx.x; d < cache_pairs; d += blockDim.x) {
        yarn_cs[d] = angle(d);
    }
    if (cache_pairs > 0) __syncthreads();

    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;
        const float2 cs = dim_pair < cache_pairs ? yarn_cs[dim_pair]
                                                 : angle(dim_pair);
        const int j0 = interleaved ? dim_pair * 2 : dim_pair;
        const int j1 = interleaved ? dim_pair * 2 + 1 : dim_pair + half;

        if (head_idx < num_q_heads) {
            const bf16* qs = src + static_cast<long long>(head_idx) * head_dim;
            bf16* qd = q_out + (static_cast<long long>(row) * num_q_heads +
                                head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved_to(qs, qd, dim_pair, cs.x, cs.y);
            else             rotate_pair_to(qs, qd, half, dim_pair, cs.x, cs.y);
        } else if (write_this_row) {
            const int kv_h = head_idx - num_q_heads;
            const bf16* ks = src + static_cast<long long>(num_q_heads + kv_h) * head_dim;
            const bf16* vs =
                src + static_cast<long long>(num_q_heads + num_kv_heads + kv_h) * head_dim;
            const float a = bf16_to_f32(ks[j0]);
            const float b = bf16_to_f32(ks[j1]);
            const int base = kv_h * head_dim;
            const long long d0 =
                kv_dst_index<false>(slot, base + j0, page_size, num_kv_heads, head_dim);
            const long long d1 =
                kv_dst_index<false>(slot, base + j1, page_size, num_kv_heads, head_dim);
            k_pages[d0] = f32_to_bf16(a * cs.x - b * cs.y);
            k_pages[d1] = f32_to_bf16(b * cs.x + a * cs.y);
            v_pages[d0] = vs[j0];
            v_pages[d1] = vs[j1];
        }
    }
}

// Qwen3.5's attention prologue in one launch, a warp a (row, head), PER
// (head_dim / 32) consecutive values a lane in one 8- or 16-byte load: a q
// head cut from its [q | gate] row (the gate copied out beside it) and a k
// head, each normed per head with its plus-one weight and its rotary prefix
// turned (`rope_partial`'s pairs, `span` apart, from the rounded normed
// values, as the norm then the rotation land them); the k head and its v
// head land in the paged (NHD) cache at the row's page and offset.
template <int PER> struct QgRun { using type = uint4; };
template <> struct QgRun<4> { using type = uint2; };

template <int PER>
__global__ __launch_bounds__(256) void qg_qknorm_rope_append(
    const bf16* __restrict__ qg,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ q_norm,
    const bf16* __restrict__ k_norm,
    const i32* __restrict__ positions,
    bf16* __restrict__ q_out,
    bf16* __restrict__ gate_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    int Hq,
    int Hk,
    int units,
    int rotary_dim,
    int span,
    float theta,
    float q_eps,
    float k_eps,
    const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8, "a lane's run is one 8- or 16-byte load");
    using Run = typename QgRun<PER>::type;
    constexpr int D = 32 * PER;
    __shared__ float freqs[D / 2];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // The pairs' frequencies once a block, as rope_partial takes them.
    for (int pr = threadIdx.x; pr < rotary_dim / 2; pr += blockDim.x) {
        freqs[pr] = powf(theta, -static_cast<float>(pr) / static_cast<float>(span));
    }
    __syncthreads();
    const int unit = static_cast<int>(blockIdx.x) * 8 + warp;
    if (unit >= units) return;
    const int heads = Hq + Hk;
    const int n = unit / heads;
    const int head = unit - n * heads;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const bool is_q = head < Hq;
    const int kvh = head - Hq;

    KvSlot slot{};
    if (!is_q) {
        slot.page = static_cast<int>(w_page[row]);
        slot.offset_in_page = static_cast<int>(w_off[row]);
        const bool write_this_row = (row_valid == nullptr || row_valid[row] != 0) &&
                                    slot.offset_in_page >= 0 && slot.offset_in_page < page_size;
        if (!write_this_row) return;
    }
    const bf16* x = is_q ? qg + (static_cast<long long>(row) * Hq + head) * 2 * D
                         : k + (static_cast<long long>(row) * Hk + kvh) * D;
    const bf16* w = is_q ? q_norm : k_norm;
    const float eps = is_q ? q_eps : k_eps;
    const int e0 = lane * PER;

    bf16 xv[PER], wv[PER];
    *reinterpret_cast<Run*>(xv) = *reinterpret_cast<const Run*>(x + e0);
    *reinterpret_cast<Run*>(wv) = *reinterpret_cast<const Run*>(w + e0);
    if (is_q) {
        *reinterpret_cast<Run*>(gate_out + (static_cast<long long>(row) * Hq + head) * D + e0) =
            *reinterpret_cast<const Run*>(x + D + e0);
    }
    float local = 0.f;
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        const float t = bf16_to_f32(xv[u]);
        local += t * t;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    const float inv_rms = rsqrtf(local / static_cast<float>(D) + eps);
    float y[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        y[u] = bf16_to_f32(f32_to_bf16(bf16_to_f32(xv[u]) * inv_rms * (bf16_to_f32(wv[u]) + 1.f)));
    }
    // A pair's mate is `span` on, `span / PER` lanes away at the same slot
    // (the caller holds span to a whole number of lanes' runs).
    const int hop = span / PER;
    float up[PER], down[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        up[u] = __shfl_sync(0xffffffffu, y[u], (lane + hop) & 31);
        down[u] = __shfl_sync(0xffffffffu, y[u], (lane - hop) & 31);
    }

    const int pos = positions[row];
    const int pairs = rotary_dim / 2;
    bf16 out[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        const int i = e0 + u;
        float o = y[u];
        int pair = -1;
        if (i < pairs) pair = i;
        else if (i >= span && i < span + pairs) pair = i - span;
        if (pair >= 0) {
            const float ang = static_cast<float>(pos) * freqs[pair];
            float cos_v, sin_v;
            __sincosf(ang, &sin_v, &cos_v);
            const float a = i == pair ? y[u] : down[u];
            const float b = i == pair ? up[u] : y[u];
            o = i == pair ? a * cos_v - b * sin_v : b * cos_v + a * sin_v;
        }
        out[u] = f32_to_bf16(o);
    }
    if (is_q) {
        *reinterpret_cast<Run*>(q_out + (static_cast<long long>(row) * Hq + head) * D + e0) =
            *reinterpret_cast<const Run*>(out);
    } else {
        const long long d = kv_dst_index<false>(slot, kvh * D + e0, page_size, Hk, D);
        *reinterpret_cast<Run*>(k_pages + d) = *reinterpret_cast<const Run*>(out);
        *reinterpret_cast<Run*>(v_pages + d) =
            *reinterpret_cast<const Run*>(v + (static_cast<long long>(row) * Hk + kvh) * D + e0);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void q_rmsnorm_rope_partial(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    const i32* __restrict__ positions,
    bf16* __restrict__ y,
    int head_dim,
    int heads,
    int rotary_dim,
    int span,
    float theta,
    float eps,
    const u32* __restrict__ win)
{
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "whole warps");
    constexpr int kWarps = BLOCK / 32;
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int head = blockIdx.y;
    const long long at = (static_cast<long long>(row) * heads + head) * head_dim;
    const bf16* xr = x + at;
    bf16* yr = y + at;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(xr[i]);
        local += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_xor_sync(0xffffffffu, local, off);
    }
    __shared__ float partial[kWarps + 1];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) partial[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float v = threadIdx.x < kWarps ? partial[threadIdx.x] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
        }
        if (lane == 0) partial[kWarps] = v;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(partial[kWarps] / static_cast<float>(head_dim) + eps);

    // the norm covers the whole head; the first rotary_dim/2 pairs turn, each
    // pair `span` apart at theta^(-i/span): span = rotary_dim/2 for a plain
    // partial rotary (HF, mlx-lm, llama.cpp), head_dim/2 for a proportional
    // one (Gemma 4's global layers).
    const int pos = positions[row];
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * inv_rms * bf16_to_f32(weight[i]));
    }
    __syncthreads();
    for (int dim_pair = threadIdx.x; dim_pair < rotary_dim / 2; dim_pair += BLOCK) {
        const int mate = dim_pair + span;
        const float a = bf16_to_f32(xr[dim_pair]) * inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(xr[mate]) * inv_rms * bf16_to_f32(weight[mate]);
        const float freq = powf(theta,
            -static_cast<float>(dim_pair) / static_cast<float>(span));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        yr[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        yr[mate] = f32_to_bf16(b * cos_v + a * sin_v);
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
    int span,
    float theta,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total_heads = num_q_heads + num_kv_heads;
    // the first rotary_dim/2 pairs of each head turn, each pair `span` apart
    // at theta^(-i/span): span = rotary_dim/2 for a plain partial rotary (HF,
    // mlx-lm, llama.cpp), head_dim/2 for a proportional one (Gemma 4's global
    // layers). Every other entry is left alone.
    const int pairs = rotary_dim / 2;
    const int pos = positions[row] + position_delta;

    for (int t = threadIdx.x; t < total_heads * pairs; t += blockDim.x) {
        const int head_idx = t / pairs;
        const int dim_pair = t % pairs;

        const float freq = powf(theta,
            -static_cast<float>(dim_pair) / static_cast<float>(span));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(row) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + span]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + span] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(row) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + span]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + span] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}

// `rope_partial_last` with a row's heads spread over `gridDim.y` blocks of
// `heads_per_block`, each block working the row's `rotary_dim / 2` angles
// out once (the same expression the per-element walk evaluates) into shared
// memory before its heads rotate. `rotary_dim / 2 <= blockDim.x`.
__global__ void rope_partial_last_spread(
    bf16* __restrict__ q,
    const i32* __restrict__ positions,
    int num_q_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    bool inverse,
    bool interleaved,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    int heads_per_block,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int rope_half = rotary_dim / 2;
    const int offset = head_dim - rotary_dim;
    const int pos = positions[row];
    extern __shared__ float2 cs_part[];
    for (int d = threadIdx.x; d < rope_half; d += blockDim.x) {
        float freq = powf(theta, -2.f * static_cast<float>(d) / static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor, yarn_low_dim, yarn_high_dim, d);
        }
        const float ang = (inverse ? -1.f : 1.f) * static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        cs_part[d] = make_float2(cos_v, sin_v);
    }
    __syncthreads();
    const int h0 = static_cast<int>(blockIdx.y) * heads_per_block;
    const int h1 = min(h0 + heads_per_block, num_q_heads);
    for (int t = threadIdx.x; t < (h1 - h0) * rope_half; t += blockDim.x) {
        const int head_idx = h0 + t / rope_half;
        const int dim_pair = t % rope_half;
        const float2 c = cs_part[dim_pair];
        bf16* base = q + static_cast<long long>(row * num_q_heads + head_idx) * head_dim;
        const int i = interleaved ? offset + 2 * dim_pair : offset + dim_pair;
        const int j = interleaved ? offset + 2 * dim_pair + 1 : offset + dim_pair + rope_half;
        const float a = bf16_to_f32(base[i]);
        const float b = bf16_to_f32(base[j]);
        base[i] = f32_to_bf16(a * c.x - b * c.y);
        base[j] = f32_to_bf16(b * c.x + a * c.y);
    }
}

// `rope_partial_last_spread` four pairs a thread: interleaved, a thread's
// pairs are one 16-byte run of the tail; split halves, an 8-byte run of
// each. The angles are the spread form's, worked out once a block into
// shared memory, so the same values land. `rotary_dim % 8 == 0`,
// `head_dim % 8 == 0`, `rotary_dim / 2 <= blockDim.x`.
__global__ void rope_partial_last_vec(
    bf16* __restrict__ q,
    const i32* __restrict__ positions,
    int num_q_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    bool inverse,
    bool interleaved,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    int heads_per_block,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int rope_half = rotary_dim / 2;
    const int offset = head_dim - rotary_dim;
    const int pos = positions[row];
    extern __shared__ float2 cs_part[];
    for (int d = threadIdx.x; d < rope_half; d += blockDim.x) {
        float freq = powf(theta, -2.f * static_cast<float>(d) / static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor, yarn_low_dim, yarn_high_dim, d);
        }
        const float ang = (inverse ? -1.f : 1.f) * static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        cs_part[d] = make_float2(cos_v, sin_v);
    }
    __syncthreads();
    const int per_head = rope_half / 4;
    const int h0 = static_cast<int>(blockIdx.y) * heads_per_block;
    const int h1 = min(h0 + heads_per_block, num_q_heads);
    for (int t = threadIdx.x; t < (h1 - h0) * per_head; t += blockDim.x) {
        const int head_idx = h0 + t / per_head;
        const int c = t % per_head;
        bf16* base = q + static_cast<long long>(row * num_q_heads + head_idx) * head_dim + offset;
        if (interleaved) {
            bf16 v[8];
            *reinterpret_cast<uint4*>(v) = *reinterpret_cast<const uint4*>(base + 8 * c);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 cs = cs_part[4 * c + u];
                const float a = bf16_to_f32(v[2 * u]);
                const float b = bf16_to_f32(v[2 * u + 1]);
                v[2 * u] = f32_to_bf16(a * cs.x - b * cs.y);
                v[2 * u + 1] = f32_to_bf16(b * cs.x + a * cs.y);
            }
            *reinterpret_cast<uint4*>(base + 8 * c) = *reinterpret_cast<const uint4*>(v);
        } else {
            bf16 lo[4], hi[4];
            *reinterpret_cast<uint2*>(lo) = *reinterpret_cast<const uint2*>(base + 4 * c);
            *reinterpret_cast<uint2*>(hi) = *reinterpret_cast<const uint2*>(base + rope_half + 4 * c);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 cs = cs_part[4 * c + u];
                const float a = bf16_to_f32(lo[u]);
                const float b = bf16_to_f32(hi[u]);
                lo[u] = f32_to_bf16(a * cs.x - b * cs.y);
                hi[u] = f32_to_bf16(b * cs.x + a * cs.y);
            }
            *reinterpret_cast<uint2*>(base + 4 * c) = *reinterpret_cast<const uint2*>(lo);
            *reinterpret_cast<uint2*>(base + rope_half + 4 * c) = *reinterpret_cast<const uint2*>(hi);
        }
    }
}

// A head's RMS norm (times `weight` where given) and the rotation of its
// last `rotary_dim` dims, a warp a (row, head) unit: lane `l` holds the
// head's runs of RUN elements `l`, `l + 32`, ... (RUN 4 for a 128-wide head,
// else 8), rounds its normed values to bf16 as the norm's own plane would,
// and turns the interleaved pairs of the tail it holds by the angles
// `rope_partial_last_spread` works out, a lane working one pair's and
// handing it on. `head_dim = 32 * PER`; the tail starts on a run and is at
// most 128 wide. APPEND (one 512-wide head a row, DeepSeek-V4's kv plane)
// also writes the row as DeepSeek's fp8 row at its page and offset, as
// `ds_append` writes it: a 64-wide tile's exponent over its eight lanes'
// runs, the rotated tile kept in bf16.
// A unit's head as its lane holds it: which plane row (-1 past the stage's
// live rows) and its runs.
template <int PER>
struct NormRuns {
    static constexpr int RUN = PER == 4 ? 4 : 8;
    static constexpr int RUNS = PER / RUN;
    int plane_row;
    bf16 v[RUNS][RUN];
};

// Loads unit `unit`'s head (its plane row's, where the stage's window keeps it).
template <int PER>
__device__ __forceinline__ void norm_rope_fetch(NormRuns<PER>& in, int unit, const bf16* __restrict__ x,
                                                int heads, const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8 || PER == 16, "a lane holds one to two runs");
    constexpr int RUN = NormRuns<PER>::RUN;
    in.plane_row = -1;
    if (win != nullptr && unit / heads >= static_cast<int>(win[0])) return;
    in.plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x & 31;
    const long long at = static_cast<long long>(in.plane_row) * (32 * PER);
#pragma unroll
    for (int r = 0; r < NormRuns<PER>::RUNS; ++r) {
        const int e = (lane + 32 * r) * RUN;
        if constexpr (RUN == 4) {
            *reinterpret_cast<uint2*>(in.v[r]) = *reinterpret_cast<const uint2*>(x + at + e);
        } else {
            *reinterpret_cast<uint4*>(in.v[r]) = *reinterpret_cast<const uint4*>(x + at + e);
        }
    }
}

// One (row, head) unit of `norm_rope_last` from its fetched head, its
// block's frequencies in `freq_s`.
template <int PER, bool WEIGHTED, bool APPEND>
__device__ __forceinline__ void norm_rope_unit(
    const float* freq_s,
    const NormRuns<PER>& in,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    const i32* __restrict__ positions,
    int heads,
    float eps,
    int rotary_dim,
    bool inverse,
    std::uint8_t* __restrict__ pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size)
{
    static_assert(!APPEND || PER == 16, "an fp8 row is 512 wide");
    constexpr int head_dim = 32 * PER;
    constexpr int RUN = NormRuns<PER>::RUN;
    constexpr int RUNS = NormRuns<PER>::RUNS;
    if (in.plane_row < 0) return;
    const int plane_row = in.plane_row;
    const int row = plane_row / heads;
    const int lane = threadIdx.x & 31;
    const long long at = static_cast<long long>(plane_row) * head_dim;

    bf16 v[RUNS][RUN];
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < RUN; ++u) v[r][u] = in.v[r][u];
    // The tail's angles a lane each (pairs `lane` and `lane + 32`), handed
    // to the lanes that hold their pairs; worked out while the head loads.
    const int offset = head_dim - rotary_dim;
    const int rope_half = rotary_dim / 2;
    const int pos = positions[row];
    float2 cs[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int d = lane + 32 * k;
        cs[k] = make_float2(1.f, 0.f);
        if (d < rope_half) {
            const float ang = (inverse ? -1.f : 1.f) * static_cast<float>(pos) * freq_s[d];
            __sincosf(ang, &cs[k].y, &cs[k].x);
        }
    }
    float local = 0.f;
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < RUN; ++u) {
            const float f = bf16_to_f32(v[r][u]);
            local = fmaf(f, f, local);
        }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) local += __shfl_xor_sync(0xffffffffu, local, off);
    const float inv_rms = rsqrtf(local / static_cast<float>(head_dim) + eps);
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < RUN; ++u) {
            float n = bf16_to_f32(v[r][u]) * inv_rms;
            if constexpr (WEIGHTED) n *= bf16_to_f32(weight[(lane + 32 * r) * RUN + u]);
            v[r][u] = f32_to_bf16(n);
        }

#pragma unroll
    for (int r = 0; r < RUNS; ++r) {
        const int e = (lane + 32 * r) * RUN;
        const bool tail = e >= offset;
        const int first = tail ? (e - offset) / 2 : 0;
        // Whether any of the warp's runs here is the tail's (alike across it).
        const bool any_tail = (32 * r + 31) * RUN >= offset;
#pragma unroll
        for (int u = 0; u < RUN / 2; ++u) {
            if (!any_tail) break;
            const int d = first + u;
            const float c0 = __shfl_sync(0xffffffffu, cs[0].x, d & 31);
            const float s0 = __shfl_sync(0xffffffffu, cs[0].y, d & 31);
            const float c1 = __shfl_sync(0xffffffffu, cs[1].x, d & 31);
            const float s1 = __shfl_sync(0xffffffffu, cs[1].y, d & 31);
            if (tail) {
                const float cos_v = d < 32 ? c0 : c1;
                const float sin_v = d < 32 ? s0 : s1;
                const float a = bf16_to_f32(v[r][2 * u]);
                const float b = bf16_to_f32(v[r][2 * u + 1]);
                v[r][2 * u] = f32_to_bf16(a * cos_v - b * sin_v);
                v[r][2 * u + 1] = f32_to_bf16(b * cos_v + a * sin_v);
            }
        }
        if constexpr (RUN == 4) {
            *reinterpret_cast<uint2*>(y + at + e) = *reinterpret_cast<const uint2*>(v[r]);
        } else {
            *reinterpret_cast<uint4*>(y + at + e) = *reinterpret_cast<const uint4*>(v[r]);
        }
    }
    if constexpr (APPEND) {
        namespace ds = ::pie::attn::dsv4;
        if (row_valid != nullptr && row_valid[plane_row] == 0) return;
        const int page = static_cast<int>(w_page[plane_row]);
        const int off = static_cast<int>(w_off[plane_row]);
        if (off < 0 || off >= page_size) return;
        const ds::DsRow dst = ds::ds_row(pages, page, off, page_size);
#pragma unroll
        for (int r = 0; r < RUNS; ++r) {
            const int e = (lane + 32 * r) * RUN;
            float f[RUN];
            float amax = 0.f;
#pragma unroll
            for (int u = 0; u < RUN; ++u) {
                f[u] = bf16_to_f32(v[r][u]);
                amax = fmaxf(amax, fabsf(f[u]));
            }
            // A tile is eight lanes' runs (alike across the warp: r says
            // whether this run is the rotated tile's).
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            if (e >= ds::kDsNope) {
                *reinterpret_cast<uint4*>(dst.data + ds::kDsNope + (e - ds::kDsNope) * 2) =
                    *reinterpret_cast<const uint4*>(v[r]);
                continue;
            }
            amax = fmaxf(amax, 1e-4f);
            const float ex = ceilf(log2f(amax / ds::kFp8Max));
            const float inv = exp2f(-ex);
            std::uint8_t codes[RUN];
#pragma unroll
            for (int u = 0; u < RUN; ++u) {
                const float c = fminf(fmaxf(f[u] * inv, -ds::kFp8Max), ds::kFp8Max);
                codes[u] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3));
            }
            *reinterpret_cast<uint2*>(dst.data + e) = *reinterpret_cast<const uint2*>(codes);
            if ((lane & 7) == 0) {
                dst.scales[e / ds::kDsTile] = static_cast<std::uint8_t>(fmaxf(fminf(ex + 127.f, 255.f), 0.f));
            }
        }
        if (lane == 0) dst.scales[ds::kDsScales] = 0;
    }
}


template <int PER, bool WEIGHTED, bool APPEND = false>
__global__ void __launch_bounds__(256) norm_rope_last(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    const i32* __restrict__ positions,
    int units,
    int heads,
    float eps,
    int rotary_dim,
    float theta,
    bool inverse,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    std::uint8_t* __restrict__ pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    const u32* __restrict__ win)
{
    // The tail's frequencies once a block, as the spread form works them.
    __shared__ float freq_s[64];
    if (threadIdx.x < rotary_dim / 2) {
        const int d = static_cast<int>(threadIdx.x);
        float freq = powf(theta, -2.f * static_cast<float>(d) / static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor, yarn_low_dim, yarn_high_dim, d);
        }
        freq_s[d] = freq;
    }
    __syncthreads();
    // A warp a unit, the grid striding over them where it is smaller; the
    // next unit's head loads while this one's is worked.
    const int stride = static_cast<int>(gridDim.x) * 8;
    int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    if (unit >= units) return;
    NormRuns<PER> cur;
    norm_rope_fetch<PER>(cur, unit, x, heads, win);
    for (; unit < units; unit += stride) {
        NormRuns<PER> next;
        next.plane_row = -1;
        if (unit + stride < units) norm_rope_fetch<PER>(next, unit + stride, x, heads, win);
        norm_rope_unit<PER, WEIGHTED, APPEND>(freq_s, cur, weight, y, positions, heads, eps, rotary_dim,
                                              inverse, pages, w_page, w_off, row_valid, page_size);
        cur = next;
    }
}

// DeepSeek-V4's q (scale-free, QPER a lane) and its kv plane (scaled, one
// 512-wide head, written to the cache where APPEND) normed and rotated in
// one launch: warps [0, q_units) take q's (row, head) units, the rest the
// plane's rows, the frequencies shared.
template <int QPER, bool APPEND>
__global__ void __launch_bounds__(256) norm_rope_pair(
    const bf16* __restrict__ qx,
    bf16* __restrict__ qy,
    int q_units,
    int q_heads,
    float q_eps,
    const bf16* __restrict__ kx,
    const bf16* __restrict__ kw,
    bf16* __restrict__ ky,
    int kv_units,
    float kv_eps,
    const i32* __restrict__ positions,
    int rotary_dim,
    float theta,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    std::uint8_t* __restrict__ pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    const u32* __restrict__ win)
{
    __shared__ float freq_s[64];
    if (threadIdx.x < rotary_dim / 2) {
        const int d = static_cast<int>(threadIdx.x);
        float freq = powf(theta, -2.f * static_cast<float>(d) / static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor, yarn_low_dim, yarn_high_dim, d);
        }
        freq_s[d] = freq;
    }
    __syncthreads();
    // A warp a unit, the grid striding over them where it is smaller; the
    // next q unit's head loads while this one's is worked.
    const int stride = static_cast<int>(gridDim.x) * 8;
    int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    NormRuns<QPER> cur;
    if (unit < q_units) norm_rope_fetch<QPER>(cur, unit, qx, q_heads, win);
    for (; unit < q_units; unit += stride) {
        NormRuns<QPER> next;
        next.plane_row = -1;
        if (unit + stride < q_units) norm_rope_fetch<QPER>(next, unit + stride, qx, q_heads, win);
        norm_rope_unit<QPER, false, false>(freq_s, cur, nullptr, qy, positions, q_heads, q_eps, rotary_dim,
                                           false, nullptr, nullptr, nullptr, nullptr, 0);
        cur = next;
    }
    for (; unit < q_units + kv_units; unit += stride) {
        NormRuns<16> kv;
        norm_rope_fetch<16>(kv, unit - q_units, kx, 1, win);
        norm_rope_unit<16, true, APPEND>(freq_s, kv, kw, ky, positions, 1, kv_eps, rotary_dim, false, pages,
                                         w_page, w_off, row_valid, page_size);
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
    float yarn_high_dim,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total_heads = num_q_heads + num_kv_heads;
    const int rope_half = rotary_dim / 2;
    const int offset = head_dim - rotary_dim;
    const int pos = positions[row];

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
            ? q + static_cast<long long>(row * num_q_heads + head_idx) * head_dim
            : k + static_cast<long long>(row * num_kv_heads + (head_idx - num_q_heads)) * head_dim;

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

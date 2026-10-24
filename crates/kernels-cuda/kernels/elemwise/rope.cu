#include "prim/device.cuh"
#include "prim/paged_kv.cuh"
#include "prim/rope.cuh"
#include "prim/kv_rows.cuh"

namespace pie::elemwise {

template <class T>
using Elem = ::pie::Elem<T>;

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
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
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

constexpr int kRopeInterleaved = 0;
constexpr int kRopeNeox = 1;
constexpr int kRopeSplit = 2;
constexpr int kRopeSplitLadder = 3;

template <class T, int FORM>
__global__ void rope_axes(
    const T* __restrict__ x,
    const float* __restrict__ positions,
    T* __restrict__ o,
    int axes,
    int d0, int d1, int d2, int d3,
    float t0, float t1, float t2, float t3,
    int rotary_dim,
    int head_dim,
    int heads,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int dims[4] = {d0, d1, d2, d3};
    const float thetas[4] = {t0, t1, t2, t3};
    const int angles = rotary_dim / 2;
    const int width = heads * head_dim;

    const float* pos = positions + static_cast<long long>(row) * axes;
    const T* xr = x + static_cast<long long>(row) * width;
    T* orow = o + static_cast<long long>(row) * width;

    if constexpr (FORM == kRopeSplitLadder) {

        int span = 0;
        for (int a = 0; a < axes; ++a) span += dims[a];
        const int pad = (heads * rotary_dim - span) / 2;
        for (int idx = threadIdx.x; idx < heads * angles; idx += blockDim.x) {
            const int head = idx / angles;
            const int angle = idx % angles;
            float cos_v = 1.f;
            float sin_v = 0.f;
            if (idx >= pad) {
                const int slot = idx - pad;
                const int axis = slot % axes;
                const int f = slot / axes;
                const int ladder = dims[axis] / 2;
                const float exponent =
                    ladder > 1 ? static_cast<float>(f) / static_cast<float>(ladder - 1) : 0.f;
                sincosf(pos[axis] * powf(thetas[axis], exponent), &sin_v, &cos_v);
            }
            const int lo = angle;
            const int hi = angle + angles;
            const T* xh = xr + static_cast<long long>(head) * head_dim;
            T* oh = orow + static_cast<long long>(head) * head_dim;
            const float a = Elem<T>::to_f32(xh[lo]);
            const float b = Elem<T>::to_f32(xh[hi]);
            oh[lo] = Elem<T>::from_f32(a * cos_v - b * sin_v);
            oh[hi] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
        return;
    }

    for (int idx = threadIdx.x; idx < heads * angles; idx += blockDim.x) {
        const int head = idx / angles;
        const int angle = idx % angles;

        int axis = 0;
        int first_angle = 0;
        int first_channel = 0;
        while (axis < axes && angle >= first_angle + dims[axis] / 2) {
            first_angle += dims[axis] / 2;
            first_channel += dims[axis];
            ++axis;
        }

        if (axis >= axes) continue;
        const int within = angle - first_angle;

        const float freq = powf(thetas[axis],
            -2.f * static_cast<float>(within) / static_cast<float>(dims[axis]));
        float cos_v, sin_v;
        __sincosf(pos[axis] * freq, &sin_v, &cos_v);

        int lo;
        int hi;
        if constexpr (FORM == kRopeInterleaved) {
            lo = first_channel + 2 * within;
            hi = lo + 1;
        } else if constexpr (FORM == kRopeNeox) {
            lo = angle;
            hi = angle + angles;
        } else {
            lo = first_channel + within;
            hi = first_channel + dims[axis] / 2 + within;
        }

        const T* xh = xr + static_cast<long long>(head) * head_dim;
        T* oh = orow + static_cast<long long>(head) * head_dim;
        const float a = Elem<T>::to_f32(xh[lo]);
        const float b = Elem<T>::to_f32(xh[hi]);
        oh[lo] = Elem<T>::from_f32(a * cos_v - b * sin_v);
        oh[hi] = Elem<T>::from_f32(b * cos_v + a * sin_v);
    }

    if (o != x) {
        const int tail = head_dim - rotary_dim;
        for (int idx = threadIdx.x; idx < heads * tail; idx += blockDim.x) {
            const int head = idx / tail;
            const int i = rotary_dim + idx % tail;
            orow[head * head_dim + i] = xr[head * head_dim + i];
        }
    }
}

template <class T>
using Elem = ::pie::Elem<T>;

template <class T>
__global__ void rope_mrope(
    T* __restrict__ q,
    T* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    int s0, int s1, int s2,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total_heads = num_q_heads + num_kv_heads;
    // The rotated prefix is its own neox head: pairs (i, i + rotary_dim/2)
    // at theta^(-2i/rotary_dim), as `rope_partial` and upstream turn it.
    const int half = rotary_dim / 2;

    const int pos_t = positions[3 * row + 0];
    const int pos_h = positions[3 * row + 1];
    const int pos_w = positions[3 * row + 2];
    (void)s0;

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        int axis_pos;
        const int m = dim_pair % 3;
        if (m == 1 && dim_pair < 3 * s1)      axis_pos = pos_h;
        else if (m == 2 && dim_pair < 3 * s2) axis_pos = pos_w;
        else                                  axis_pos = pos_t;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(rotary_dim));
        const float ang = static_cast<float>(axis_pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(row) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(row) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}


template <class T>
__global__ void rope_mrope_blocked(
    T* __restrict__ q,
    T* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    int s0, int s1, int s2,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;
    const int total = s0 + s1 + s2;

    const int pos[3] = { positions[3 * row + 0],
                         positions[3 * row + 1],
                         positions[3 * row + 2] };

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        if (dim_pair >= rope_angles) continue;
        if (dim_pair >= total) continue;

        int axis;
        int within;
        if (dim_pair < s0)           { axis = 0; within = dim_pair; }
        else if (dim_pair < s0 + s1) { axis = 1; within = dim_pair - s0; }
        else                         { axis = 2; within = dim_pair - s0 - s1; }

        const float freq = powf(theta,
            -2.f * static_cast<float>(within) / static_cast<float>(total));
        const float ang = static_cast<float>(pos[axis]) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(row) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(row) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}

// The most pairs a stated table turns: a 512-wide head turned whole.
constexpr int kRopeTablePairs = 256;

// A rotation's frequencies stated outright (a checkpoint's rope scaling):
// pair i turns at position * inv_freq[i], its cos and sin scaled by mscale.
struct RopeTable {
    float inv_freq[kRopeTablePairs];
    float mscale;
};

// q's and k's heads turned at a stated table: `pairs` pairs a head, pair i
// being (i, i + span), or (2i, 2i + 1) where interleaved. With `s1` or `s2`
// set the positions are (t, h, w) triples and pair i reads its axis as
// Qwen's interleaved M-RoPE does (`rope_mrope`); otherwise one position a
// row. A row's angles are taken once, in full precision (`sincosf`), as the
// reference's cos/sin cache holds them at any length.
template <class T>
__global__ void rope_table(
    T* __restrict__ q,
    T* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int pairs,
    int span,
    bool interleaved,
    int s1,
    int s2,
    const RopeTable table,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const bool mrope = s1 > 0 || s2 > 0;

    extern __shared__ float rope_table_cs[];
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        int pos;
        if (mrope) {
            const int m = i % 3;
            if (m == 1 && i < 3 * s1)      pos = positions[3 * row + 1];
            else if (m == 2 && i < 3 * s2) pos = positions[3 * row + 2];
            else                           pos = positions[3 * row + 0];
        } else {
            pos = positions[row];
        }
        float s, c;
        sincosf(static_cast<float>(pos) * table.inv_freq[i], &s, &c);
        rope_table_cs[i] = c * table.mscale;
        rope_table_cs[pairs + i] = s * table.mscale;
    }
    __syncthreads();

    const int total_heads = num_q_heads + num_kv_heads;
    for (int t = threadIdx.x; t < total_heads * pairs; t += blockDim.x) {
        const int head_idx = t / pairs;
        const int i = t % pairs;
        T* h = head_idx < num_q_heads
            ? q + (static_cast<long long>(row) * num_q_heads + head_idx) * head_dim
            : k + (static_cast<long long>(row) * num_kv_heads + head_idx - num_q_heads) * head_dim;
        const int j0 = interleaved ? 2 * i : i;
        const int j1 = interleaved ? 2 * i + 1 : i + span;
        const float c = rope_table_cs[i];
        const float s = rope_table_cs[pairs + i];
        const float a = ::pie::Elem<T>::to_f32(h[j0]);
        const float b = ::pie::Elem<T>::to_f32(h[j1]);
        h[j0] = ::pie::Elem<T>::from_f32(a * c - b * s);
        h[j1] = ::pie::Elem<T>::from_f32(b * c + a * s);
    }
}

}

#pragma once

// Rotary position embeddings, applied in place on q and k (or, fused behind
// a per-head rmsnorm, from x into y).
//
// Angles are f32 with a precise sincos: a fast one drifts at long positions,
// where the angle is large. A neox rotation turns pair i with i + half; an
// interleaved one turns 2i with 2i + 1.

#include "prelude/device.hip"

namespace pie::elemwise {

// theta^(-i/span) radians per position: pair i of a rotation `2·span` wide.
__device__ __forceinline__ float frequency(float theta, int i, int span) {
    return powf(theta, -static_cast<float>(i) / static_cast<float>(span));
}

__device__ __forceinline__ void turn(int pos, float freq, float& c, float& s) {
    sincosf(static_cast<float>(pos) * freq, &s, &c);
}

__device__ __forceinline__ void rotation(int pos, int i, int span, float theta, float& c, float& s) {
    turn(pos, frequency(theta, i, span), c, s);
}

// YaRN's ramp from the extrapolated pairs (below low_dim, kept) to the
// interpolated ones (above high_dim, slowed by `factor`).
__device__ __forceinline__ float yarn_frequency(float freq, float factor, float low_dim,
                                                float high_dim, int i) {
    const float denom = high_dim == low_dim ? high_dim + 1e-3f - low_dim : high_dim - low_dim;
    const float ramp = fminf(fmaxf((static_cast<float>(i) - low_dim) / denom, 0.f), 1.f);
    return freq * ((1.f - ramp) + ramp / factor);
}

// Entries a and b of a head turn by the angle whose cosine and sine are c, s.
template <class T>
__device__ __forceinline__ void turn_pair(T* head, int a, int b, float c, float s) {
    const float x = to_f32(head[a]);
    const float y = to_f32(head[b]);
    head[a] = from_f32<T>(x * c - y * s);
    head[b] = from_f32<T>(y * c + x * s);
}

template <class T>
__device__ __forceinline__ void rotate_pair(T* head, int i, int span, float c, float s) {
    turn_pair(head, i, i + span, c, s);
}

template <class T>
__device__ __forceinline__ void rotate_any(T* head, int i, int half, bool interleaved, float c,
                                           float s) {
    if (interleaved) turn_pair(head, 2 * i, 2 * i + 1, c, s);
    else turn_pair(head, i, i + half, c, s);
}

// Head `head` of a row: q's heads first, then k's.
template <class T>
__device__ __forceinline__ T* head_at(T* q, T* k, long long row, int head, int num_q_heads,
                                      int num_kv_heads, int head_dim) {
    return head < num_q_heads ? q + (row * num_q_heads + head) * head_dim
                              : k + (row * num_kv_heads + (head - num_q_heads)) * head_dim;
}

// Full rotary over every pair of every head. Grid (rows, head groups): a
// block turns `heads_per_block` heads of one row, and the row's first
// `cache_pairs` angles are taken once into LDS (cos then sin) and shared by
// its heads. The fused page append of the CUDA original is not ported: its
// arguments stay in the signature, unread, as the launcher passes them.
template <bool kWriteKv, bool kHnd>
__global__ void rope_full(bf16* __restrict__ q, bf16* __restrict__ k,
                          const i32* __restrict__ positions, int num_q_heads, int num_kv_heads,
                          int head_dim, float theta, bool interleaved, int cache_pairs,
                          int heads_per_block, const bf16* __restrict__ v, bf16* __restrict__ k_pages,
                          bf16* __restrict__ v_pages, const u32* __restrict__ qo_indptr,
                          const u32* __restrict__ kv_page_indices,
                          const u32* __restrict__ kv_page_indptr,
                          const u32* __restrict__ kv_last_page_lens, const u8* __restrict__ row_valid,
                          int R, int page_size, const u32* __restrict__ win) {
    static_assert(!kWriteKv, "the page append is attn/kv.hip's, not fused here");
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int half = head_dim / 2;
    const int pos = positions[row];

    extern __shared__ float rope_cs[];
    for (int i = threadIdx.x; i < cache_pairs; i += blockDim.x) {
        rotation(pos, i, half, theta, rope_cs[i], rope_cs[cache_pairs + i]);
    }
    if (cache_pairs > 0) lds_barrier();

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, num_q_heads + num_kv_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head = head_base + t / half;
        const int i = t % half;
        float c, s;
        if (i < cache_pairs) {
            c = rope_cs[i];
            s = rope_cs[cache_pairs + i];
        } else {
            rotation(pos, i, half, theta, c, s);
        }
        rotate_any(head_at(q, k, row, head, num_q_heads, num_kv_heads, head_dim), i, half,
                   interleaved, c, s);
    }
}

// Full rotary with YaRN's frequency ramp and its attention factor folded
// into cos and sin. Laid out as `rope_full`, with (cos, sin) pairs cached.
__global__ void rope_yarn(bf16* __restrict__ q, bf16* __restrict__ k,
                          const i32* __restrict__ positions, int num_q_heads, int num_kv_heads,
                          int head_dim, float theta, float factor, float low_dim, float high_dim,
                          float mscale, bool interleaved, int heads_per_block, int cache_pairs,
                          const u32* __restrict__ win) {
    extern __shared__ float2 yarn_cs[];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int half = head_dim / 2;
    const int pos = positions[row];

    auto angle = [&](int i) {
        float c, s;
        turn(pos, yarn_frequency(frequency(theta, i, half), factor, low_dim, high_dim, i), c, s);
        return make_float2(c * mscale, s * mscale);
    };
    for (int i = threadIdx.x; i < cache_pairs; i += blockDim.x) yarn_cs[i] = angle(i);
    if (cache_pairs > 0) lds_barrier();

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, num_q_heads + num_kv_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head = head_base + t / half;
        const int i = t % half;
        const float2 cs = i < cache_pairs ? yarn_cs[i] : angle(i);
        rotate_any(head_at(q, k, row, head, num_q_heads, num_kv_heads, head_dim), i, half,
                   interleaved, cs.x, cs.y);
    }
}

// Partial rotary: the first rotary_dim/2 pairs of every head turn, pair i
// with i + span (span = rotary_dim/2 for a plain partial rotary, head_dim/2
// for a proportional one); the rest of the head is left alone. One block
// per token row, q heads then k heads; the row's first `cache_pairs` angles
// are taken once into LDS (cos then sin) and shared by its heads.
template <class T>
__global__ void rope_partial(T* __restrict__ q, T* __restrict__ k,
                             const i32* __restrict__ positions, int position_delta,
                             int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim,
                             int span, float theta, int cache_pairs, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int pairs = rotary_dim / 2;
    const int pos = positions[row] + position_delta;
    extern __shared__ float partial_cs[];
    for (int i = threadIdx.x; i < cache_pairs; i += blockDim.x) {
        rotation(pos, i, span, theta, partial_cs[i], partial_cs[cache_pairs + i]);
    }
    if (cache_pairs > 0) lds_barrier();
    for (int t = threadIdx.x; t < (num_q_heads + num_kv_heads) * pairs; t += blockDim.x) {
        const int head = t / pairs;
        const int i = t % pairs;
        float c, s;
        if (i < cache_pairs) {
            c = partial_cs[i];
            s = partial_cs[cache_pairs + i];
        } else {
            rotation(pos, i, span, theta, c, s);
        }
        rotate_pair(head_at(q, k, row, head, num_q_heads, num_kv_heads, head_dim), i, span, c, s);
    }
}

// The rotated part is the last rotary_dim entries of every head, its own
// rotation at theta^(-2i/rotary_dim) (with YaRN's ramp when factor > 1);
// `inverse` turns it back by the negated angle. One block per token row.
__global__ void rope_partial_last(bf16* __restrict__ q, bf16* __restrict__ k,
                                  const i32* __restrict__ positions, int num_q_heads,
                                  int num_kv_heads, int head_dim, int rotary_dim, float theta,
                                  bool inverse, bool interleaved, float yarn_factor,
                                  float yarn_low_dim, float yarn_high_dim,
                                  const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int half = rotary_dim / 2;
    const int offset = head_dim - rotary_dim;
    const int pos = inverse ? -positions[row] : positions[row];
    for (int t = threadIdx.x; t < (num_q_heads + num_kv_heads) * half; t += blockDim.x) {
        const int head = t / half;
        const int i = t % half;
        float freq = frequency(theta, i, half);
        if (yarn_factor > 1.f) freq = yarn_frequency(freq, yarn_factor, yarn_low_dim, yarn_high_dim, i);
        float c, s;
        turn(pos, freq, c, s);
        rotate_any(head_at(q, k, row, head, num_q_heads, num_kv_heads, head_dim) + offset, i, half,
                   interleaved, c, s);
    }
}

// y = rmsnorm(x)·w per head, then the partial rotary of `rope_partial` on
// the normed values (unrounded between the two). Grid (rows, heads); a
// thread writes each entry it reads, so x and y may be one plane.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void q_rmsnorm_rope_partial(
    const bf16* __restrict__ x, const bf16* __restrict__ weight, const i32* __restrict__ positions,
    bf16* __restrict__ y, int head_dim, int heads, int rotary_dim, int span, float theta, float eps,
    const u32* __restrict__ win) {
    static_assert(BLOCK % kWave == 0, "whole wavefronts");
    __shared__ float smem[BLOCK / kWave];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const long long at = (row * heads + blockIdx.y) * head_dim;
    const bf16* xr = x + at;
    bf16* yr = y + at;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = to_f32(xr[i]);
        local += v * v;
    }
    const float inv = rsqrtf(block_sum(local, smem) / static_cast<float>(head_dim) + eps);
    auto normed = [&](int i) { return to_f32(xr[i]) * inv * to_f32(weight[i]); };

    const int pairs = rotary_dim / 2;
    const int pos = positions[row];
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        if (i < pairs) {
            const int mate = i + span;
            const float a = normed(i);
            const float b = normed(mate);
            float c, s;
            rotation(pos, i, span, theta, c, s);
            yr[i] = from_f32<bf16>(a * c - b * s);
            yr[mate] = from_f32<bf16>(b * c + a * s);
        } else if (i < span || i >= span + pairs) {
            yr[i] = from_f32<bf16>(normed(i));
        }
    }
}

}  // namespace pie::elemwise

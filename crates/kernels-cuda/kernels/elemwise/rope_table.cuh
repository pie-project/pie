#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

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

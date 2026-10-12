#pragma once

// Rotary position embeddings, applied in place on q and k.

#include "prelude/device.hip"

namespace pie::elemwise {

// The rotation of pair `i` at position `pos`: theta^(-i/span) radians per
// position, in f32 with a precise sincos (a fast one drifts at long
// positions, where the angle is large).
__device__ __forceinline__ void rotation(int pos, int i, int span, float theta, float& c, float& s) {
    const float freq = powf(theta, -static_cast<float>(i) / static_cast<float>(span));
    sincosf(static_cast<float>(pos) * freq, &s, &c);
}

template <class T>
__device__ __forceinline__ void rotate_pair(T* head, int i, int span, float c, float s) {
    const float a = to_f32(head[i]);
    const float b = to_f32(head[i + span]);
    head[i] = from_f32<T>(a * c - b * s);
    head[i + span] = from_f32<T>(b * c + a * s);
}

// Partial rotary: the first rotary_dim/2 pairs of every head turn, pair i
// with i + span (span = rotary_dim/2 for a plain partial rotary, head_dim/2
// for a proportional one); the rest of the head is left alone. One block
// per token row, q heads then k heads.
template <class T>
__global__ void rope_partial(T* __restrict__ q, T* __restrict__ k,
                             const i32* __restrict__ positions, int position_delta,
                             int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim,
                             int span, float theta, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int pairs = rotary_dim / 2;
    const int pos = positions[row] + position_delta;
    for (int t = threadIdx.x; t < (num_q_heads + num_kv_heads) * pairs; t += blockDim.x) {
        const int head = t / pairs;
        const int i = t % pairs;
        float c, s;
        rotation(pos, i, span, theta, c, s);
        T* at = head < num_q_heads
                    ? q + (row * num_q_heads + head) * head_dim
                    : k + (row * num_kv_heads + (head - num_q_heads)) * head_dim;
        rotate_pair(at, i, span, c, s);
    }
}

}  // namespace pie::elemwise

#pragma once

// Multimodal rotary (M-RoPE): a (t, h, w) position a row, sections s0..s2
// naming which pairs turn by which axis: interleaved (pair i by i % 3 within
// the sections, Qwen3-VL), blocked (by the section holding i, Qwen2-VL) or
// split (axis a's own 2·s_a block, a rotate_half, Gemma 4 vision). A block a
// row, in place.

#include "elemwise/rope.hip"

namespace pie::elemwise {

struct Triple {
    int t, h, w;

    __device__ __forceinline__ int operator[](int axis) const {
        return axis == 0 ? t : (axis == 1 ? h : w);
    }
};

__device__ __forceinline__ Triple triple_at(const i32* positions, long long row) {
    return {positions[3 * row + 0], positions[3 * row + 1], positions[3 * row + 2]};
}

__global__ void rope_mrope(bf16* __restrict__ q, bf16* __restrict__ k, const i32* __restrict__ positions,
                           int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim,
                           float theta, int s0, int s1, int s2, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int half = rotary_dim / 2;
    const Triple pos = triple_at(positions, row);
    (void)s0;
    for (int t = threadIdx.x; t < (num_q_heads + num_kv_heads) * half; t += blockDim.x) {
        const int head = t / half;
        const int i = t % half;
        const int m = i % 3;
        const int axis = m == 1 && i < 3 * s1 ? 1 : (m == 2 && i < 3 * s2 ? 2 : 0);
        float c, s;
        rotation(pos[axis], i, half, theta, c, s);
        rotate_pair(head_at(q, k, row, head, num_q_heads, num_kv_heads, head_dim), i, half, c, s);
    }
}

__global__ void rope_mrope_blocked(bf16* __restrict__ q, bf16* __restrict__ k,
                                   const i32* __restrict__ positions, int num_q_heads,
                                   int num_kv_heads, int head_dim, int rotary_dim, float theta,
                                   int s0, int s1, int s2, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int half = head_dim / 2;
    const int total = s0 + s1 + s2;
    // Pairs past the rotated prefix or past the sections stay as they are.
    const int pairs = min(rotary_dim / 2, total);
    const Triple pos = triple_at(positions, row);
    for (int t = threadIdx.x; t < (num_q_heads + num_kv_heads) * pairs; t += blockDim.x) {
        const int head = t / pairs;
        const int i = t % pairs;
        const int axis = i < s0 ? 0 : (i < s0 + s1 ? 1 : 2);
        const int within = i - (axis == 0 ? 0 : (axis == 1 ? s0 : s0 + s1));
        float c, s;
        turn(pos[axis], powf(theta, -2.f * static_cast<float>(within) / static_cast<float>(total)),
             c, s);
        rotate_pair(head_at(q, k, row, head, num_q_heads, num_kv_heads, head_dim), i, half, c, s);
    }
}

__global__ void rope_mrope_split(bf16* __restrict__ q, bf16* __restrict__ k,
                                 const i32* __restrict__ positions, int num_q_heads,
                                 int num_kv_heads, int head_dim, int rotary_dim, float theta,
                                 int s0, int s1, int s2, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int pairs = min(rotary_dim / 2, s0 + s1 + s2);
    const Triple pos = triple_at(positions, row);
    for (int t = threadIdx.x; t < (num_q_heads + num_kv_heads) * pairs; t += blockDim.x) {
        const int head = t / pairs;
        const int i = t % pairs;
        const int axis = i < s0 ? 0 : (i < s0 + s1 ? 1 : 2);
        const int before = axis == 0 ? 0 : (axis == 1 ? s0 : s0 + s1);
        const int width = axis == 0 ? s0 : (axis == 1 ? s1 : s2);
        const int within = i - before;
        float c, s;
        rotation(pos[axis], within, width, theta, c, s);
        bf16* at = head_at(q, k, row, head, num_q_heads, num_kv_heads, head_dim) + 2 * before;
        rotate_pair(at, within, width, c, s);
    }
}

}  // namespace pie::elemwise

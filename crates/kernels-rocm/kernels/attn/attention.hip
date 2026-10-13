#pragma once

// What rides beside the attention kernels: the final-logit soft cap, the
// attention-sink rescale, and the merge of two partial attentions by their
// log-sum-exp.
//
// An lse plane is f32 [row][head] in the log2 domain (paged.hip's
// convention): lse = m + log2(Σ 2^(s - m)), -inf for a row that saw no key.

#include "prelude/device.hip"

namespace pie::attn {

constexpr float kLn2 = 0.69314718055994530942f;

// x = cap · tanh(x / cap), in place: Gemma's logit soft cap.
template <class T>
__global__ void logit_softcap(T* __restrict__ x, float cap, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv_cap = 1.f / cap;
    x[i] = from_f32<T>(cap * tanhf(to_f32(x[i]) * inv_cap));
}

// An attention sink is one more key per head whose logit is `sinks[h]` and
// whose value is zero: it takes softmax mass and gives back nothing, so the
// output a row already has is scaled by Σ / (Σ + e^sink) = σ(ln Σ − sink).
// One block per (row, head); a row that saw no key keeps its output.
template <class T>
__global__ void sink_rescale(T* __restrict__ o, const float* __restrict__ lse,
                             const T* __restrict__ sinks, i32 N, i32 num_q_heads, i32 head_dim,
                             const u32* __restrict__ win) {
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    if (t >= N || h >= num_q_heads || !window.live(t)) return;
    const long long row = window.plane(t);

    const float lse_val = lse[row * num_q_heads + h];
    const float r = isfinite(lse_val) ? 1.f / (1.f + __expf(-(lse_val * kLn2 - to_f32(sinks[h])))) : 1.f;

    T* out = o + (row * num_q_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) out[d] = from_f32<T>(to_f32(out[d]) * r);
}

// o = (o1·2^lse1 + o2·2^lse2) / (2^lse1 + 2^lse2), lse = log2 of the
// denominator. A side whose lse is not finite saw no key and drops out; the
// copy is skipped where the output already is that side. One block per
// (row, head).
template <class T>
__global__ void merge_lse_combine(const T* __restrict__ o1, const float* __restrict__ lse1,
                                  const T* __restrict__ o2, const float* __restrict__ lse2,
                                  T* __restrict__ o_out, float* __restrict__ lse_out,
                                  int num_heads, int head_dim, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    const long long cell = n * num_heads + h;
    const long long off = cell * head_dim;

    const float l1 = lse1[cell];
    const float l2 = lse2[cell];

    if (!isfinite(l2)) {
        if (o1 != o_out)
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) o_out[off + d] = o1[off + d];
        if (lse_out != nullptr && lse_out != lse1 && threadIdx.x == 0) lse_out[cell] = l1;
        return;
    }
    if (!isfinite(l1)) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) o_out[off + d] = o2[off + d];
        if (lse_out != nullptr && threadIdx.x == 0) lse_out[cell] = l2;
        return;
    }

    const float big = fmaxf(l1, l2);
    const float w1 = exp2f(l1 - big);
    const float w2 = exp2f(l2 - big);
    const float inv_total = 1.f / (w1 + w2);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v1 = to_f32(o1[off + d]);
        const float v2 = to_f32(o2[off + d]);
        o_out[off + d] = from_f32<T>((v1 * w1 + v2 * w2) * inv_total);
    }
    if (lse_out != nullptr && threadIdx.x == 0) lse_out[cell] = big + log2f(w1 + w2);
}

}  // namespace pie::attn

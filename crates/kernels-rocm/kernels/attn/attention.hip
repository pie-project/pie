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

// V elements a thread (V divides the head width), a row's heads·head_dim / V
// threads over blockIdx.y blocks of blockDim.x: thread (t, i) owns elements
// V·i .. of row t, head V·i / head_dim.
template <class T, int V>
struct alignas(sizeof(T) * V) HeadRun {
    T v[V];
};

// An attention sink is one more key per head whose logit is `sinks[h]` and
// whose value is zero: it takes softmax mass and gives back nothing, so the
// output a row already has is scaled by Σ / (Σ + e^sink) = σ(ln Σ − sink).
// A row that saw no key keeps its output.
template <class T, class S, int V>
__global__ void sink_rescale(T* __restrict__ o, const float* __restrict__ lse,
                             const S* __restrict__ sinks, i32 N, i32 num_q_heads, i32 head_dim,
                             const u32* __restrict__ win) {
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    const int i = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    if (t >= N || i >= num_q_heads * head_dim / V || !window.live(t)) return;
    const long long row = window.plane(t);
    const int h = V * i / head_dim;

    const float lse_val = lse[row * num_q_heads + h];
    const float r = isfinite(lse_val) ? 1.f / (1.f + __expf(-(lse_val * kLn2 - static_cast<float>(sinks[h])))) : 1.f;

    auto* out = reinterpret_cast<HeadRun<T, V>*>(o + row * num_q_heads * head_dim) + i;
    HeadRun<T, V> x = *out;
#pragma unroll
    for (int j = 0; j < V; ++j) x.v[j] = from_f32<T>(to_f32(x.v[j]) * r);
    *out = x;
}

// o = (o1·2^lse1 + o2·2^lse2) / (2^lse1 + 2^lse2), lse = log2 of the
// denominator. A side whose lse is not finite saw no key and drops out; the
// copy is skipped where the output already is that side.
template <class T, int V>
__global__ void merge_lse_combine(const T* __restrict__ o1, const float* __restrict__ lse1,
                                  const T* __restrict__ o2, const float* __restrict__ lse2,
                                  T* __restrict__ o_out, float* __restrict__ lse_out,
                                  int num_heads, int head_dim, const u32* __restrict__ win) {
    using R = HeadRun<T, V>;
    const Window window{win};
    const int i = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    if (i >= num_heads * head_dim / V || !window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int h = V * i / head_dim;
    const long long cell = n * num_heads + h;
    const long long at = n * num_heads * head_dim / V + i;
    const bool lead = (V * i) % head_dim == 0;  // the head's first thread lands its lse

    const float l1 = lse1[cell];
    const float l2 = lse2[cell];

    if (!isfinite(l2)) {
        if (o1 != o_out) reinterpret_cast<R*>(o_out)[at] = reinterpret_cast<const R*>(o1)[at];
        if (lse_out != nullptr && lse_out != lse1 && lead) lse_out[cell] = l1;
        return;
    }
    if (!isfinite(l1)) {
        reinterpret_cast<R*>(o_out)[at] = reinterpret_cast<const R*>(o2)[at];
        if (lse_out != nullptr && lead) lse_out[cell] = l2;
        return;
    }

    const float big = fmaxf(l1, l2);
    const float w1 = exp2f(l1 - big);
    const float w2 = exp2f(l2 - big);
    const float inv_total = 1.f / (w1 + w2);
    const R x1 = reinterpret_cast<const R*>(o1)[at];
    const R x2 = reinterpret_cast<const R*>(o2)[at];
    R y;
#pragma unroll
    for (int j = 0; j < V; ++j) y.v[j] = from_f32<T>((to_f32(x1.v[j]) * w1 + to_f32(x2.v[j]) * w2) * inv_total);
    reinterpret_cast<R*>(o_out)[at] = y;
    if (lse_out != nullptr && lead) lse_out[cell] = big + log2f(w1 + w2);
}

}  // namespace pie::attn

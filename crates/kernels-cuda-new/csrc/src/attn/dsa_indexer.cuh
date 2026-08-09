//===-- dsa_indexer.cuh - glm5's sparse-attention index network ----------===//
//
// Three `__global__` templates and the RoPE helper they share. No host code.
//
// glm5 attends SPARSELY: a small side network scores every (query, key) pair
// and only the top-k keys per query reach the attention kernel. The three
// kernels here are that network -- normalise and rotate the index keys,
// rotate the index queries, then rank and threshold. They exist as a separate
// file because the scoring runs at a DIFFERENT width from attention itself
// (one index head-dim, a handful of index heads), and fusing them into the
// attention path would tie the sparse budget to the dense geometry.
//
// # Which of them got a row, and why the other two did not
//
// `index_knorm_rope` was `<<<tokens, 256>>>` -- one block per token, a
// 256-wide block reduction inside. That is `LaunchRule::Rms` exactly, and it
// has a row. `Rms` also hands the kernel 32 bytes of dynamic shared memory
// for `block_sum`'s warp scratch, which this kernel does not use: its
// reduction is a static `__shared__ float red[256]`. Thirty-two unused bytes
// per block is not worth a second rule, and the alternative -- rewriting the
// reduction to the prelude's -- changes the fold ORDER and therefore the last
// bit of a LayerNorm that feeds a top-k ranking.
//
// `index_q_rope` was `<<<tokens, roundup32(n_heads)>>>`: one block per token,
// one THREAD PER HEAD. `LaunchRule::RouteRows` is the rule with that shape --
// one block per row, as wide as the row rounded to a warp -- but it reads the
// row's WIDTH, and `idx_q`'s row is `n_heads * head_dim` wide. The two differ
// by a factor of `head_dim`, which is 64 or 128, so `RouteRows` would open
// 128x the block and every thread past `n_heads` would return immediately
// after the grid had already been sized wrong. No rule states "as wide as the
// row divided by its trailing extent", so this kernel has no row. It is here
// anyway, once, NVRTC-clean, waiting for one.
//
// `index_topk_mask` was `<<<tokens, 256, tokens * sizeof(float)>>>`. The
// dynamic shared memory is sized on the ROW COUNT -- one float per key, and
// every key is a token -- and no ported rule produces smem from `rows`;
// `Rms`'s smem is a function of the block width. That is the whole obstacle:
// the geometry is otherwise `Rms`'s. Reported rather than approximated,
// because a launch that under-sizes shared memory does not fail, it reads
// another block's floats.
//
// # `buf[256]` is a local array, and that is a decision
//
// Both RoPE kernels stage `rope_dim` floats in a per-thread array before
// rotating them. 256 floats is 1 KB of local memory per thread, which on this
// path is one thread per block (`index_knorm_rope`) or one per head
// (`index_q_rope`) and therefore cheap. Sizing it on `rope_dim` is not
// possible -- it is a run-time value -- and spilling to shared would need a
// launch bound the rule does not carry. A model with `rope_dim > 256`
// overruns it; the bound is stated here so the next reader can find it.
//
// # `powf` and `__sincosf` are NVRTC's, and they are the originals'
//
// Measured: NVRTC 13.0 accepts `powf`, `__sincosf`, `rsqrtf`, `fmaxf`,
// `fminf`, `atomicAdd` on shared `int`, and `device::pos_inf()` in place of
// the `INFINITY` macro -- which it does NOT define, because `<math.h>` is one
// of the 31 standard headers it answers with nothing. That substitution is
// the only edit the top-k kernel needed.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::pos_inf;
using ::pie_cuda_driver::kernels::device::u8;

/// The block width `LaunchRule::Rms` opens, and the width `red` is sized on.
constexpr int kBlock = 256;
/// The per-thread RoPE staging bound. See the header.
constexpr int kMaxRopeDim = 256;

/// Interleaved RoPE, in place on a staged fp32 buffer.
///
/// INTERLEAVED and not split-half: pairs are `(2i, 2i+1)`, which is what
/// glm5's index network trains against. A split-half rotation on the same
/// buffer is a different function, and the two agree only when `rope_dim` is
/// 2 -- so getting it wrong is invisible in a unit test with tiny dims.
__device__ __forceinline__ void rope_interleave_inplace(
    float* v, int rope_dim, int pos, float theta)
{
    const int half = rope_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = powf(theta, -2.f * static_cast<float>(i) /
                                       static_cast<float>(rope_dim));
        const float ang = static_cast<float>(pos) * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        const float a = v[2 * i];
        const float b = v[2 * i + 1];
        v[2 * i]     = a * c - b * s;
        v[2 * i + 1] = b * c + a * s;
    }
}

/// LayerNorm over `head_dim` then interleaved RoPE, in place on `idx_k`.
///
/// LayerNorm and not RMSNorm -- the mean is subtracted, and there is a bias.
/// One block per token; the row bound the launcher used to check is
/// `LaunchRule::Rms`'s grid.
template <class T>
__global__ void index_knorm_rope(
    T* __restrict__ idx_k,
    const T* __restrict__ w,
    const T* __restrict__ b,
    const i32* __restrict__ positions,
    i32 head_dim, i32 rope_dim, float theta, float eps)
{
    const int n = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    T* row = idx_k + static_cast<long long>(n) * head_dim;

    __shared__ float red[kBlock];
    float s = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) s += Elem<T>::to_f32(row[d]);
    red[tid] = s; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float mean = red[0] / head_dim;
    __syncthreads();
    float vv = 0.f;
    for (int d = tid; d < head_dim; d += kBlock) { float x = Elem<T>::to_f32(row[d]) - mean; vv += x * x; }
    red[tid] = vv; __syncthreads();
    for (int o = kBlock / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / head_dim + eps);
    __syncthreads();
    for (int d = tid; d < head_dim; d += kBlock) {
        const float x = (Elem<T>::to_f32(row[d]) - mean) * inv;
        row[d] = Elem<T>::from_f32(x * Elem<T>::to_f32(w[d]) + Elem<T>::to_f32(b[d]));
    }
    __syncthreads();
    if (tid == 0) {
        float buf[kMaxRopeDim];
        for (int d = 0; d < rope_dim; ++d) buf[d] = Elem<T>::to_f32(row[d]);
        rope_interleave_inplace(buf, rope_dim, positions[n], theta);
        for (int d = 0; d < rope_dim; ++d) row[d] = Elem<T>::from_f32(buf[d]);
    }
}

/// Interleaved RoPE on the first `rope_dim` of each index head of `idx_q`.
///
/// One block per token, one thread per head. NO ROW STATES THIS KERNEL: the
/// block width is `roundup32(n_heads)` and every rule that sizes a block on a
/// row sizes it on the row's WIDTH, which here is `n_heads * head_dim`. See
/// the header.
template <class T>
__global__ void index_q_rope(
    T* __restrict__ idx_q,
    const i32* __restrict__ positions,
    i32 n_heads, i32 head_dim, i32 rope_dim, float theta)
{
    const int n = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(threadIdx.x);
    if (h >= n_heads) return;
    T* row = idx_q + (static_cast<long long>(n) * n_heads + h) * head_dim;
    float buf[kMaxRopeDim];
    for (int d = 0; d < rope_dim; ++d) buf[d] = Elem<T>::to_f32(row[d]);
    rope_interleave_inplace(buf, rope_dim, positions[n], theta);
    for (int d = 0; d < rope_dim; ++d) row[d] = Elem<T>::from_f32(buf[d]);
}

/// Causal top-k mask over the index scores. One block per query token.
///
///     logit[i, j] = sum_h relu(q[i, h] . k[j]) * w[i, h]
///
/// The softmax scale is monotonic and therefore irrelevant to a RANKING, so
/// it is omitted rather than computed and divided out.
///
/// The threshold is found by forty rounds of bisection on the logit range
/// rather than by a sort: a sort of `nkeys` floats per block costs shared
/// memory proportional to the sequence and a partial sort still has to be
/// exact at the boundary. Forty halvings of an fp32 interval reach the
/// representable neighbourhood of the true k-th value, and the tie behaviour
/// (`>= thr` admits every equal logit) is the original's -- so a row of equal
/// scores admits more than `topk` keys, exactly as it did.
///
/// Its dynamic shared memory is `tokens * 4` bytes, sized on the ROW COUNT --
/// which is exactly what `LaunchRule::RowScores` states, and this launcher is
/// the one `runtime::launch::row_scores` was ported from. The sentence that
/// stood here, *"no ported rule derives smem from rows"*, was a report on the
/// vocabulary and not on the kernel; `attn::dsa_index_topk_mask` is a row.
template <class T>
__global__ void index_topk_mask(
    const T* __restrict__ idx_q,
    const T* __restrict__ idx_k,
    const T* __restrict__ idx_w,
    u8* __restrict__ mask,
    i32 N, i32 H, i32 D, i32 topk)
{
    const int i = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    u8* mrow = mask + static_cast<long long>(i) * N;
    const int nkeys = i + 1;  // causal

    for (int j = nkeys + tid; j < N; j += kBlock) mrow[j] = 0;

    if (nkeys <= topk) {
        for (int j = tid; j < nkeys; j += kBlock) mrow[j] = 1;
        return;
    }

    extern __shared__ float logit[];  // [nkeys]
    const T* qi = idx_q + static_cast<long long>(i) * H * D;
    const T* wi = idx_w + static_cast<long long>(i) * H;
    for (int j = tid; j < nkeys; j += kBlock) {
        const T* kj = idx_k + static_cast<long long>(j) * D;
        float acc = 0.f;
        for (int h = 0; h < H; ++h) {
            const T* qh = qi + static_cast<long long>(h) * D;
            float dot = 0.f;
            for (int d = 0; d < D; ++d) dot += Elem<T>::to_f32(qh[d]) * Elem<T>::to_f32(kj[d]);
            acc += fmaxf(dot, 0.f) * Elem<T>::to_f32(wi[h]);
        }
        logit[j] = acc;
    }
    __syncthreads();

    __shared__ float lo_s, hi_s;
    if (tid == 0) {
        float lo = pos_inf(), hi = -pos_inf();
        for (int j = 0; j < nkeys; ++j) { lo = fminf(lo, logit[j]); hi = fmaxf(hi, logit[j]); }
        lo_s = lo; hi_s = hi;
    }
    __syncthreads();
    float lo = lo_s, hi = hi_s;
    __shared__ int cnt_s;
    float thr = hi;
    for (int it = 0; it < 40; ++it) {
        const float mid = 0.5f * (lo + hi);
        if (tid == 0) cnt_s = 0;
        __syncthreads();
        int c = 0;
        for (int j = tid; j < nkeys; j += kBlock) if (logit[j] >= mid) c++;
        atomicAdd(&cnt_s, c);
        __syncthreads();
        const int cnt = cnt_s;
        if (cnt > topk) lo = mid; else hi = mid;
        __syncthreads();
        thr = hi;
    }
    for (int j = tid; j < nkeys; j += kBlock) mrow[j] = (logit[j] >= thr) ? 1 : 0;
}

}  // namespace pie_cuda_driver::kernels::attn::device

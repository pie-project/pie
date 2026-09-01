#pragma once

#include "prelude/device.cuh"
#include "elemwise/norm.cuh"

namespace pie::elemwise {

/// **THE CENTRING, WHICH IS THE HALF OF A `LayerNorm` NO IMPORT CAN FOLD**
/// (`.wiki/alto/multimodal.md` §6.1, §9.1), shared by both entries below.
///
/// `c = (x - mean(x)) * rsqrt(var(x) + eps)` over a whole row, and then an
/// epilogue: nothing at all, or `c * w + b`. Every qwen vision block is
/// `nn.LayerNorm` — the checkpoints publish `blocks.{l}.norm1.bias` beside
/// `.weight`, and an RMSNorm has no bias.
///
/// **TWO REDUCTIONS AND NOT ONE.** `var = E[x^2] - E[x]^2` would halve the
/// syncs and is the reason this kernel would be subtly wrong: a tower row
/// whose mean is large against its spread cancels catastrophically in f32,
/// and the failure shows up as a slightly wrong norm rather than as a NaN.
/// The mean is reduced, then the centred squares are reduced against it, the
/// way `torch.nn.LayerNorm` computes it.
///
/// `eps` sits INSIDE the root beside the variance, which is where LayerNorm
/// puts it and where the rms family next door puts it too.
///
/// `AFFINE` picks the EPILOGUE and nothing else — it does not touch what is
/// reduced, which is the line `norm.rs` draws when it says a family's members
/// may share a row helper and its reductions may not take a flag. `weight`
/// and `bias` are read only when `AFFINE`, so the scale-less entry passes
/// null for both and the compiler drops the loads.
template <class T, int BLOCK, bool AFFINE>
__device__ __forceinline__ void layernorm_row(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    const int row = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && row >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows START: with a stage armed `x` and
    // `y` arrive at the plane's base, so this block's row is its block index
    // shifted by that start; with none they arrive pre-shifted. `weight` and
    // `bias` are indexed by column and never move.
    const int plane_row = win != nullptr ? row + static_cast<int>(win[1]) : row;

    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(plane_row) * hidden;
    T* yr = y + static_cast<long long>(plane_row) * hidden;

    __shared__ float buf[BLOCK];

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        local += Elem<T>::to_f32(xr[i]);
    }
    const float mean = block_reduce_sum_exact<BLOCK>(local, buf) /
                       static_cast<float>(hidden);

    __syncthreads();

    float spread = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float c = Elem<T>::to_f32(xr[i]) - mean;
        spread += c * c;
    }
    const float inv = rsqrtf(block_reduce_sum_exact<BLOCK>(spread, buf) /
                                 static_cast<float>(hidden) +
                             eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float c = (Elem<T>::to_f32(xr[i]) - mean) * inv;
        if constexpr (AFFINE) {
            yr[i] = Elem<T>::from_f32(
                fmaf(c, Elem<T>::to_f32(weight[i]), Elem<T>::to_f32(bias[i])));
        } else {
            yr[i] = Elem<T>::from_f32(c);
        }
    }
}

/// **THE CENTRED NORM, WITHOUT THE TWO VECTORS §6.1 EXPECTED TO BAKE.** What
/// a text writes when the scale really does fold into the GEMM behind it:
/// `LN(x)*M^T = (c/rms(c))*diag(w)*M^T + b*M^T`. §9.1 found that fold half
/// expressible for the qwen towers, which is why `layernorm` below exists
/// too; this entry is unchanged and still the one a folding text wants.
template <class T, int BLOCK = 256>
__global__ void layernorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    layernorm_row<T, BLOCK, false>(x, nullptr, nullptr, y, hidden, eps, win);
}

/// **THE WHOLE `nn.LayerNorm`, IN ONE LAUNCH** (`.wiki/alto/next.md` B5).
///
/// `y = (x - mean(x)) * rsqrt(var(x) + eps) * w + b`, which is what every
/// qwen vision block computes and what its checkpoint publishes the two
/// planes for. Until this entry a text spelled it
/// `add_bias(b, rmsnorm(layernorm_no_scale(x, eps), w, eps))` — three
/// launches, two intermediate rectangles — because multimodal §9.1 found the
/// import fold half-expressible.
///
/// **AND IT IS THE IDEAL ARITHMETIC, NOT THE COMPOSITION'S.** The three-op
/// form rounds the centred row to `T` before the `rmsnorm` reads it, then
/// reduces THOSE rounded values and multiplies by their reciprocal rms — a
/// uniform per-row factor of `1 +/- 1.4e-4`, which is the composition's own
/// artifact and not LayerNorm. Reproducing it here would mean reducing over
/// an intermediate this kernel does not have: a fictional round-trip through
/// a storage type the fused op no longer names, and one whose value would
/// change if the destination were ever f16 instead of bf16. So the centred
/// row stays in f32 all the way to the single rounding at the store, which is
/// strictly nearer `torch.nn.LayerNorm` than the form it replaces.
///
/// `fmaf` and not a multiply-then-add: the scale and the bias are one
/// operation here, which is the one place the fusion buys accuracy as well as
/// launches.
template <class T, int BLOCK = 256>
__global__ void layernorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int hidden,
    float eps,
    const u32* __restrict__ win)
{
    layernorm_row<T, BLOCK, true>(x, weight, bias, y, hidden, eps, win);
}

}

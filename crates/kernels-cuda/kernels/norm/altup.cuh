//===-- altup.cuh - Gemma-3n's altup predict and correct -----------------===//
//
// Two `__global__` templates and nothing else: no host function, no `<<<>>>`,
// no entry point. The two launchers that remain in `altup.cu` are ahead-of-time
// entry points, not part of this header.
//
// # These kernels have rows now, and the refusal that preceded them was right
//
// This section opened *"No rows name these kernels, and that is a
// measurement, not an oversight"*, and it closed by saying restoring them
// *"wants a rule for one block per (row, group) pair, tiled over the row,
// which nobody has written yet."* Someone wrote it. It is
// `LaunchRule::AltUpStreams`, ported from the launchers below, and both
// kernels are rowed.
//
// Both launchers compute a THREE-AXIS grid:
//
//     const dim3 grid(T, K, (H + 128 - 1) / 128);
//     kernel<<<grid, 128, 0, stream>>>(...);
//
// -- token on x, altup stream on y, hidden tile on z, and a 128-wide block.
// When this was written, no rule produced a `gridDim.y` or a `gridDim.z` at
// all. Two do now, and the note that stating a near-miss *"would launch a
// grid these kernels do not index and quietly compute a slice of the answer,
// which is worse than not firing at all"* turned out to be exactly right, and
// was measured rather than argued:
//
//   * `WarpTiledScan` reaches three axes at this same 128-wide block, and its
//     `z` is `ceil(V_d/4)` where this is `ceil(H/128)`. At the shapes tested
//     that is 1/32 of the blocks and leaves 31/32 of hidden untouched.
//   * Its `grid.y` is filled from `Dims::kv_heads`, an ATTENTION head count,
//     where `K` is a STREAM count. Fired at `kv_heads == 8`, that near miss
//     produced `[5,8,64]` against the launcher's `[5,4,4]` -- and **0 of
//     20,480 bytes differed**, because the kernel's own `t >= T || k >= K`
//     guard absorbs the overrun. Only firing the OTHER shape
//     (`kv_heads == 2`, `[5,2,256]`) exposes it, at 10,221 of 20,480 bytes.
//
// So `AltUpStreams` reads `Dims::altup_streams`, a field added for it
// precisely because `kv_heads` is a different quantity that happens to be a
// number. `new-horizon.md` §22.5 records the same decision for
// `stated_head_dim` and the seven rules that read `head_dim`.
//
// The guard survives for the reason given below, and the measurement above is
// why it matters: it is what made a wrong grid silent.
//
// The `t >= T || k >= K` half of the guard is dead under those exact extents
// and the `h >= H` half is not -- `H` is tiled by 128 and the last tile is
// ragged. The whole guard is kept: a guard that a future rule's grid would
// need is a guard worth carrying, and it costs one predicate.
//
// # Why they are templates when the originals were not
//
// The originals were `_bf16` and only `_bf16` because an AOT build has to
// choose its instantiations. The bodies are written over `T` through
// `Elem<T>` so that the day a rule fits, a second numeric format
// costs a row instead of a translation unit. The coefficients stay fp32 in
// both cases -- they are a small dense matrix the host computes, and rounding
// them to `T` would change the sum this kernel exists to make exact.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::norm {


/// `predictions[k] = Σ_j coefs[t, j, k] · streams[j] + streams[k]`.
///
/// The residual term is added AFTER the sum, in fp32, exactly as the
/// reference does: folding it into the loop as a `+1` on the diagonal
/// coefficient reassociates the accumulation and moves the last bit.
template <class T>
__global__ void altup_predict(
    const T* __restrict__ streams,
    const float* __restrict__ coefs,
    T* __restrict__ predictions,
    int K, int T_len, int H)
{
    const int t = blockIdx.x;
    const int k = blockIdx.y;
    const int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T_len || k >= K || h >= H) return;

    const long long stream_stride = (long long)T_len * H;

    float sum = 0.f;
    for (int j = 0; j < K; ++j) {
        const float c = coefs[(long long)t * K * K + (long long)j * K + k];
        const float s = Elem<T>::to_f32(
            streams[(long long)j * stream_stride + (long long)t * H + h]);
        sum += c * s;
    }
    sum += Elem<T>::to_f32(streams[(long long)k * stream_stride + (long long)t * H + h]);
    predictions[(long long)k * stream_stride + (long long)t * H + h] = Elem<T>::from_f32(sum);
}

/// `corrected[k] = (activated - predictions[active]) · coef[t, k] + predictions[k]`.
///
/// `active_idx` selects the stream the block actually ran, so the correction
/// is the residual of that one stream broadcast across all `K`.
template <class T>
__global__ void altup_correct(
    const T* __restrict__ predictions,
    const T* __restrict__ activated,
    const float* __restrict__ correction_coefs_plus_one,
    T* __restrict__ corrected,
    int K, int T_len, int H, int active_idx)
{
    const int t = blockIdx.x;
    const int k = blockIdx.y;
    const int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T_len || k >= K || h >= H) return;

    const long long stream_stride = (long long)T_len * H;
    const float a     = Elem<T>::to_f32(activated[(long long)t * H + h]);
    const float p_act = Elem<T>::to_f32(
        predictions[(long long)active_idx * stream_stride + (long long)t * H + h]);
    const float p_k = Elem<T>::to_f32(
        predictions[(long long)k * stream_stride + (long long)t * H + h]);
    const float coef   = correction_coefs_plus_one[(long long)t * K + k];
    const float result = (a - p_act) * coef + p_k;
    corrected[(long long)k * stream_stride + (long long)t * H + h] = Elem<T>::from_f32(result);
}

}  // namespace pie::norm

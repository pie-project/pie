//===-- attn_res.cuh - the residual-block blend ---------------------------===//
//
// One `__global__` template and the block reduction it folds through. No host
// code: the `<<<T, 256>>>` that used to sit below it is `LaunchRule::Rms`,
// stated in the row and evaluated by `runtime::launch`.
//
// # What this computes, and why it is one kernel
//
// K3's residual stream is not a single vector: a prefix and `B` candidate
// blocks compete, and the layer picks a convex combination of them. The score
// of each candidate is an RMS-normalised projection -- normalise the row,
// dot it against `norm_weight * proj_weight`, softmax the `B + 1` scores,
// blend. Fusing it is not an optimisation but a memory decision: `B + 1`
// passes over `H` from L2 beat `B + 1` round trips to a scratch buffer, and
// the scores never leave shared memory.
//
// `kMaxBlocks = 32` bounds the softmax scratch. K3 opens `ceil(93 / 12) = 8`;
// 32 is slack, and a model that wants more gets a compile error rather than a
// silent out-of-bounds shared write -- which is the property a fixed array
// buys over a dynamic one here.
//
// # The launcher's guard, split in two
//
//     if (T <= 0 || H <= 0) return;
//     attn_res_blend_kernel<<<T, 256, 0, stream>>>(..., block_rows > 0 ? block_rows : T, ...);
//
// `T <= 0` is `Ungeometric::Empty`, which `eval` already returns. `H <= 0`
// is the same statement about the row width and the row states `H` as
// `Source::OutWidth(0)` -- a zero-width output is a lowering bug the binder
// reports rather than a shape this kernel absorbs. `block_rows > 0 ? … : T`
// was the launcher choosing a default; the row states `Source::Rows` for it,
// which is the value the ternary produced on every call site that existed.
//
// `T` itself is gone from the signature. It did two jobs and lost both:
// `if (t >= T) return;` was a bound check, and `Rule::Rms` opens exactly
// `rows` blocks, so `blockIdx.x` is in range by the rule's promise rather
// than this kernel's assumption -- the trade `norm/altup_aux.cuh` documents.
// The other job was the block stride, and that survives as `block_rows`,
// which is a STRIDE that happens to equal an extent -- the same pair
// `mean_streams`' `t_stride` names, and the one an operand list has to keep
// because the kernel cannot address `blocks` without it.
//
// # The reduction is the file's, not the prelude's
//
// `pie_device.cuh` has `block_sum`, and this file keeps its own
// `block_reduce_sum` anyway. They fold in DIFFERENT orders -- the prelude's
// unrolls a fixed 16-down shuffle, this one derives the offsets from
// `warpSize` and then re-sums the warp partials on thread 0 with a serial
// loop, broadcasting through shared memory. Two orders sum the same values
// to different last bits, and this kernel's output feeds a softmax whose
// argmax the tolerance contract holds to zero. So the port carries the
// original's order, not the tidier one.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::i32;

/// K3 opens `ceil(93 / 12) = 8` blocks; 32 is slack.
constexpr int kMaxBlocks = 32;
/// The block width `LaunchRule::Rms` opens. Named here because `scratch` is
/// sized on it and a mismatch is a silent race, not a launch failure.
constexpr int kThreads = 256;

/// Block-wide sum of `x`, in every thread, through `scratch[blockDim.x / warpSize]`.
///
/// Deliberately not `device::block_sum`: see the header. The final
/// `__syncthreads()` after the broadcast read is what lets a caller reuse
/// `scratch` on the next iteration without a race.
__device__ __forceinline__ float block_reduce_sum(float x, float* scratch) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    const int warps = blockDim.x / warpSize;
    float total = 0.f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < warps; ++w) total += scratch[w];
        scratch[0] = total;
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

/// One block per token. `B + 1` passes over `H` for the scores, one more for
/// the blend.
template <class T>
__global__ void attn_res_blend(
    const T* __restrict__ prefix,
    const T* __restrict__ blocks,
    const T* __restrict__ norm_weight,
    const T* __restrict__ proj_weight,
    T* __restrict__ out,
    i32 B, i32 H, i32 block_rows, float eps)
{
    const i32 t = static_cast<i32>(blockIdx.x);

    __shared__ float scratch[kThreads / 32];
    __shared__ float prob_s[kMaxBlocks + 1];

    const long long token_off = static_cast<long long>(t) * H;
    const i32 rows = B + 1;

    // `row(j)` is block j for j < B, and the running prefix for j == B.
    auto row_ptr = [&](i32 j) -> const T* {
        return (j < B) ? blocks + (static_cast<long long>(j) * block_rows + t) * H
                       : prefix + token_off;
    };

    for (i32 j = 0; j < rows; ++j) {
        const T* v = row_ptr(j);
        float ss = 0.f;
        for (i32 h = static_cast<i32>(threadIdx.x); h < H;
             h += static_cast<i32>(blockDim.x)) {
            const float x = Elem<T>::to_f32(v[h]);
            ss += x * x;
        }
        ss = block_reduce_sum(ss, scratch);
        const float scale = rsqrtf(ss / static_cast<float>(H) + eps);

        float dot = 0.f;
        for (i32 h = static_cast<i32>(threadIdx.x); h < H;
             h += static_cast<i32>(blockDim.x)) {
            dot += Elem<T>::to_f32(v[h]) * scale *
                   Elem<T>::to_f32(norm_weight[h]) *
                   Elem<T>::to_f32(proj_weight[h]);
        }
        dot = block_reduce_sum(dot, scratch);
        if (threadIdx.x == 0) {
            prob_s[j] = dot;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float m = prob_s[0];
        for (i32 j = 1; j < rows; ++j) m = fmaxf(m, prob_s[j]);
        float sum = 0.f;
        for (i32 j = 0; j < rows; ++j) {
            prob_s[j] = __expf(prob_s[j] - m);
            sum += prob_s[j];
        }
        const float inv = 1.f / sum;
        for (i32 j = 0; j < rows; ++j) prob_s[j] *= inv;
    }
    __syncthreads();

    for (i32 h = static_cast<i32>(threadIdx.x); h < H;
         h += static_cast<i32>(blockDim.x)) {
        float acc = 0.f;
        for (i32 j = 0; j < rows; ++j) {
            acc += prob_s[j] * Elem<T>::to_f32(row_ptr(j)[h]);
        }
        out[token_off + h] = Elem<T>::from_f32(acc);
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device

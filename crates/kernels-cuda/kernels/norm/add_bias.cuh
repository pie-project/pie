//===-- add_bias.cuh - a bias added onto rows already in place -----------===//
//
// Two `__global__` templates over one `__device__` body, and nothing else: no
// host function, no `<<<>>>`, no entry point. Everything else about them is a
// row in `kernels_cuda::families::norm`.
//
// # What the launchers were doing, and where it went
//
// Both were the same three lines:
//
//     if (num_rows <= 0 || dim <= 0) return;
//     kernel<<<num_rows, 256, 0, stream>>>(..., num_rows, dim);
//
// -- one block per row, 256 threads striding the row. That is
// `LaunchRule::RouteRows`: *"one block per row, the block sized to the row"*.
// The rule recovers `num_rows` from the fire's rectangle, so `num_rows` left
// the kernel signature with it; the empty guard is `Ungeometric::Empty`,
// which `bind::launch` already refuses on before it reaches a kernel. The
// grid was exactly `num_rows` blocks, so `if (n >= num_rows) return;` could
// never fire and is gone too -- a guard against a grid the rule computes is a
// guard against the rule.
//
// # Why two kernels and not one with a pitch
//
// `add_bias` and `add_bias_strided` differ in one term: the row pitch. They
// were two `__global__`s with two copies of a four-line loop, which is the
// shape that drifts -- fix a rounding mode in one and the other keeps the old
// answer for whichever caller happens to reach it. So the loop is written
// once, as `add_bias_row`, and the two kernels are the two ways to find a
// row.
//
// They stay two kernels rather than one kernel with `stride == dim`, because
// `model-compiler` lowers two statements to two symbols and a row names one
// instantiation. One symbol per row, one instantiation per symbol: the map
// stays a bijection, and neither statement can be fired through the other's
// contract by accident.
//
// # Why they are templates when the originals were not
//
// The originals were `_bf16` and only `_bf16` because an AOT build has to
// choose its instantiations. Under a JIT the element type is the row's, so
// the body is written over `T` through `Elem<T>` and a second numeric
// format costs a row instead of a translation unit -- the trick
// `elementwise.cuh` documents.
//
// The arithmetic is unchanged: widen both addends to fp32, add, narrow back.
// That is what the originals did and what the bf16 tolerance contract was
// measured against.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::norm {


/// One row of `row[d] += bias[d]`, striding by the block width.
///
/// `__device__`, not a free function: NVRTC does not forgive an unannotated
/// helper the way nvcc does inside a `.cu`.
template <class T>
__device__ __forceinline__ void add_bias_row(
    T* __restrict__ row,
    const T* __restrict__ bias,
    int dim)
{
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = Elem<T>::to_f32(row[d]) + Elem<T>::to_f32(bias[d]);
        row[d] = Elem<T>::from_f32(v);
    }
}

/// `out[n][d] += bias[d]` over a contiguous `[rows, dim]` rectangle.
///
/// Accumulates into its FIRST operand, which the row states as
/// `in_place = &[(0, 0)]`.
template <class T>
__global__ void add_bias(
    T* __restrict__ out,
    const T* __restrict__ bias,
    int dim)
{
    add_bias_row<T>(out + static_cast<long long>(blockIdx.x) * dim, bias, dim);
}

/// The same sum into a strided view: `dim` columns of a `stride`-wide row.
///
/// This is how a fused bias is split -- one `[rows, stride]` buffer, a bias
/// per slice, each slice added over its own `dim` columns. The launcher, not
/// the kernel, refuses `stride < dim`: the rule states a rectangle, and a
/// pitch narrower than the row it carries is a caller error no geometry can
/// see.
template <class T>
__global__ void add_bias_strided(
    T* __restrict__ out,
    const T* __restrict__ bias,
    int dim,
    int stride)
{
    add_bias_row<T>(out + static_cast<long long>(blockIdx.x) * stride, bias, dim);
}

}  // namespace pie::norm

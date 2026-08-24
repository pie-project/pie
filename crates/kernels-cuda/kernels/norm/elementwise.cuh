//===-- elementwise.cuh - the norm family's pointwise kernels ------------===//
//
// Three `__global__` templates and nothing else: no host function, no
// `<<<>>>`, no entry point. Everything else about them is Rust, in
// `src/norm.rs`'s `elementwise`.
//
// The third arrived last and belongs here rather than in a file of its own:
// `scale` is `scalar_mul` with the factor read from a device `[1]` bank
// instead of the params run, and a pointwise norm kernel is what this file
// collects. Splitting it out would put two spellings of one loop in two
// places, which is the drift `deinterleave.cuh`'s header records the cost
// of.
//
// # What the launchers were doing, and where it went
//
// Both `.cu` launchers were the same four lines:
//
//     if (n == 0) return;
//     const auto blocks = (n + BLOCK - 1) / BLOCK;
//     kernel<<<blocks, BLOCK, 0, stream>>>(...);
//
// which is `LaunchRule::Elementwise` -- *"pointwise over one row, 256-wide;
// rows stack flat, `width * rows` threads on one axis"* -- stated in the row
// and evaluated by `bind::launch`. The `n == 0` guard is
// `Ungeometric::Empty`, which the rule already returns and the binder already
// refuses on. So the launchers are not ported here; they are DELETED, and the
// row says what they said.
//
// # Why they are templates when the originals were not
//
// The originals were `_bf16` and only `_bf16`, because an AOT build has to
// choose its instantiations and nobody was going to spend a translation unit
// on a second one. Under a JIT the element type is the row's, so the kernel
// is written over `T` and the fp16 version costs a line in a table --
// `norm_device`'s `tanh_inplace` already demonstrates that, and this is the
// same trick applied where it was never worth the build time before.
//
// The arithmetic is unchanged: widen to fp32, compute, narrow back. That is
// what the originals did and what the bf16 tolerance contract was measured
// against.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::norm {


/// `y += x`, elementwise.
///
/// Accumulates into its FIRST operand, which the row states as
/// `in_place = &[(0, 0)]` -- that is what lets a text add into a window and
/// have the window keep the result.
///
/// There is no bound check against a row count and no `rows` argument: the
/// grid covers `n` and the guard below is the only one there has ever been.
template <class T>
__global__ void residual_add(T* __restrict__ y, const T* __restrict__ x, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = Elem<T>::to_f32(y[i]);
    const float b = Elem<T>::to_f32(x[i]);
    y[i] = Elem<T>::from_f32(a + b);
}

/// `x *= s`, elementwise, with the scalar rounded to `T` first.
///
/// **The rounding is the kernel, not a detail.** `s` arrives as fp32 and is
/// narrowed to `T` and widened back before the multiply, so the product is
/// `T(x) * T(s)` evaluated in fp32 -- which is how PyTorch evaluates
/// `tensor * bf16_scalar`. Gemma-4's `embed_normalizer` is stored as bf16 in
/// the reference adapter for exactly this reason: a raw fp32 scalar produces
/// a 1-ULP-per-element drift that RMSNorm amplifies into multi-unit
/// divergence by layer ~5.
template <class T>
__global__ void scalar_mul(T* __restrict__ x, float s, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s_rounded = Elem<T>::to_f32(Elem<T>::from_f32(s));
    x[i] = Elem<T>::from_f32(Elem<T>::to_f32(x[i]) * s_rounded);
}

/// `x *= *s`, elementwise, with the factor read from DEVICE MEMORY.
///
/// `scalar_mul`'s sibling, and one indirection is the whole difference. There
/// the factor rides the fire's params run as an fp32 the HOST chose -- a
/// `sqrt(hidden)` the text computed. Here it is a learned `[1]` bank the
/// load-time parameter table bound: gemma's per-layer `layer_scalar`, which
/// the legacy driver read to the host once at load
/// (`read_bf16_scalar_once`) precisely because no kernel would take it where
/// it lives. This is that kernel, and it is why `norm.scale` and
/// `norm.mul_scalar` are two points at the same shape -- the factor's
/// PROVENANCE differs, and provenance is what a slot's mark says.
///
/// NO ROUNDING TRIP, where `scalar_mul` has one. `scalar_mul` narrows its
/// fp32 to `T` and widens it back because that is how PyTorch evaluates
/// `tensor * bf16_scalar`, and a raw fp32 factor drifts a ULP per element.
/// A `[1]` bank is ALREADY `T`: the checkpoint did that rounding when it
/// stored the number, and doing it again here would be a second rounding of
/// a value that has none left to lose.
///
/// Every thread loads `s[0]`. One broadcast off L1 per warp against a launch
/// already bound by the traffic on `x`; staging it through shared memory
/// would buy a load and cost a `__syncthreads()`.
template <class T>
__global__ void scale(T* __restrict__ x, const T* __restrict__ s, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float f = Elem<T>::to_f32(s[0]);
    x[i] = Elem<T>::from_f32(Elem<T>::to_f32(x[i]) * f);
}

}  // namespace pie::norm

//===-- softcap.cuh - the logit cap, as a template -----------------------===//
//
// One `__global__` template and nothing else: no host function, no `<<<>>>`,
// no entry point. Everything else about it is a row in
// `kernels_cuda::families::attn`.
//
// # What the launcher was doing, and where it went
//
//     if (n == 0 || !(cap > 0.f)) return;
//     const auto blocks = (n + 255) / 256;
//     logit_softcap_bf16_kernel<<<blocks, 256, 0, stream>>>(x, 1.f / cap, cap, n);
//
// which is `LaunchRule::Elementwise` -- *"pointwise over one row, 256-wide;
// rows stack flat"* -- stated in the row and evaluated by `runtime::launch`.
// The `n == 0` half of the guard is `Ungeometric::Empty`, which the rule
// already returns and the binder already refuses on; the `cap > 0` half is
// the row's `Source::CtxNonZero("final_logit_softcap")`, which is where it
// always belonged -- a fire whose model states no cap does not bind this
// kernel at all.
//
// # The reciprocal moved, and it is the same bits
//
// The launcher passed BOTH `1.f / cap` and `cap`, because a `<<<>>>` is the
// only place a host can do arithmetic on the way to a kernel. A row is not a
// place to do arithmetic -- `Source` composes EXTENTS, not floats -- so the
// division happens on the device now. It is the same number: this crate
// compiles every unit with `--prec-div=true`, so `1.f / cap` is the
// correctly-rounded fp32 quotient on the device exactly as it was on the
// host, and `--fmad=false` keeps the multiply from being contracted into
// something else. Two IEEE-754 operations either way, same rounding, same
// result.
//
// # Why it is a template when the original was not
//
// The original was `_bf16` and only `_bf16`, because an ahead-of-time build
// has to choose its instantiations and nobody spends a translation unit on a
// second one. Under a JIT the element type is the row's, so a capped fp16
// logit row costs a line in a table rather than a `cicc` invocation --
// `norm/elementwise.cuh`'s fp16 `residual_add` is the same trick, and this is
// it applied to the family that most often meets a head in a second format.
//
// The arithmetic is unchanged: widen to fp32, `cap * tanh(x / cap)`, narrow
// once. That is what the original did and what the bf16 tolerance contract
// was measured against.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {


/// `x = cap * tanh(x / cap)`, elementwise and in place.
///
/// The cap is a SATURATION, not a clamp: the tail is squashed smoothly, which
/// is what gemma-2/3 and grok train against, and a hard `min`/`max` here
/// changes the distribution the sampler then reads.
///
/// There is no bound check against a row count and no `rows` argument: the
/// grid covers `n` and the guard below is the only one there has ever been.
template <class T>
__global__ void logit_softcap(T* __restrict__ x, float cap, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float inv_cap = 1.f / cap;
    const float v = Elem<T>::to_f32(x[i]);
    x[i] = Elem<T>::from_f32(cap * tanhf(v * inv_cap));
}

}  // namespace pie::attn

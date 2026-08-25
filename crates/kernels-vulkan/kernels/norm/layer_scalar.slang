// Per-layer scalar multiply: `out = x * s`.
//
// The Metal port's important finding is preserved: the grid is the extent, and
// the one field of `LayerScalarParams` was the hidden width -- a per-ROW bound
// on a `width * rows` dispatch, which would skip every prefill row after the
// first. It was never read here.
//
// So the block is gone rather than bound-and-ignored. slangc already deleted
// the binding for being unread, which meant this module declared a descriptor
// its routine forwarded and its SPIR-V did not have -- the shape
// `Holds::params_block`'s sentinel exists to paper over. `layer_scalar_mul`
// states no mark in its place, so now the routine and the module agree.
//
// ## `norm.scale` READS its factor; `norm.mul_scalar` is TOLD one, and the
// ## difference is a rounding trip
//
// `PIE_STATED` is the second point, and it is NOT the first with a buffer
// swapped for a push word. `norm.scale` takes a `[1]` bank that is ALREADY
// this tensor's element -- a bf16 in a bf16 model -- so widening it and
// multiplying is the whole kernel. The checkpoint did the rounding when it
// stored the number and there is none left to lose.
//
// `norm.mul_scalar` takes an `f32` off the STATEMENT, and the reference
// narrows it to bf16 and widens it back before the multiply:
//
//     s_rounded = f32(bf16(s));   out[i] = bf16(f32(x[i]) * s_rounded)
//
// which is `pie::norm::scalar_mul` in `kernels-cuda/kernels/norm/elementwise.cuh`
// and `layer_scalar_mul_stated` in `kernels-metal`. It is how PyTorch evaluates
// `tensor * bf16_scalar`, and gemma-4's `embed_normalizer` is stored as bf16 in
// the reference adapter for exactly this reason: a raw f32 factor drifts one
// ULP per element, which the RMSNorm downstream amplifies into multi-unit
// divergence by about layer five.
//
// `f32_to_bf16` in `common/bf16.slang` is round-to-nearest-even, so the trip is
// the same rounding `__float2bfloat16` and MSL's `static_cast<bfloat>` perform,
// and `PIE_LOAD(PIE_STORE(s))` is the whole of it. THE MEASUREMENT, on this
// plane, at gemma-4's own 4096-wide row and its own normalizer: 87 of 4096
// elements land on a different bf16 without the trip -- the same 87 metal
// measured, on the same input, which is what a port getting the rounding right
// looks like.
//
// ## The bindings renumber, because the factor stops being one
//
// A descriptor set is written from a DENSE list, so the stated arm cannot leave
// a hole where the `scalar` buffer was: `out_` is binding 1 there and 2 here.
// The same conditional-binding shape `norm/rms.slang` uses for its residual
// arms, and for the same reason.

#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);

#if defined(PIE_STATED)
PIE_BUFFER_RW(1, out_);
struct Push { float scalar; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
#else
PIE_BUFFER_RO(1, scalar);
PIE_BUFFER_RW(2, out_);
#endif

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
#if defined(PIE_STATED)
    float s = PIE_LOAD(PIE_STORE(pc.scalar));
#else
    float s = PIE_LOAD(scalar[0]);
#endif
    out_[gid.x] = PIE_STORE(PIE_LOAD(x[gid.x]) * s);
}

// pie:instantiate layer_scalar_mul_bfloat16
// pie:instantiate layer_scalar_mul_stated_bfloat16 PIE_STATED=1

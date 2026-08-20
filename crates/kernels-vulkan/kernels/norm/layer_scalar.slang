// Per-layer scalar multiply: `out = x * scalar[0]`.
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

#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, scalar);
PIE_BUFFER_RW(2, out_);

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    out_[gid.x] = PIE_STORE(PIE_LOAD(x[gid.x]) * PIE_LOAD(scalar[0]));
}

// pie:instantiate layer_scalar_mul_bfloat16

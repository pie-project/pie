// The `InOut` operand's bytes, moved into the rectangle a kernel is about to
// write through.
//
// # Why this is a dispatch and not a host copy
//
// It was a host copy — `fire::run::stage_blits`, every operand moved before a
// single dispatch of the fire had run. That is correct only if an operand's
// bytes are already final when the fire starts, and they are not:
// `norm.residual_add` at op14 of a qwen3.5 decode takes the output of
// `layout.embed`, which is dispatch 0 of the same fire. The copy happened
// first and the residual stream was added to the zeros the arena was cleared
// to, from layer 0 onward. Nothing refused; a whole tower fired and answered a
// real forward of something else.
//
// A dispatch is ordered by the same hazard tracker every other statement is,
// and it is recorded into the same indirect command buffer — so the sentence
// "a recording cannot carry a blit" stops being true by there being no blits,
// only dispatches.
//
// # Why the element is `ushort`
//
// Because every rectangle this moves is a whole number of them. A bf16 plane
// is two bytes an element and an f32 plane is four, so a two-byte copy covers
// both without a tail — where a `uint` copy would need one whenever a bf16
// rectangle held an odd element count. `model_compiler::program::carve` aligns
// every rectangle to 256 bytes, so the addresses are aligned for either.
//
// The copy moves BITS and does not widen: a reinterpretation through `float`
// would be lossless at best and a canonicalisation of a NaN at worst.
// `layout/row_gather.metal` makes the same argument for the same reason.

#include <metal_stdlib>
using namespace metal;

kernel void blit_bfloat16(
    const device ushort* src [[buffer(0)]],
    device ushort* dst [[buffer(1)]],
    const constant uint& elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= elements) return;
  dst[tid] = src[tid];
}

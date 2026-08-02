#include <metal_stdlib>
using namespace metal;

struct PtirLogitsCopyParams {
  uint source_row;
  uint destination_row;
  uint vocab;
  uint reserved;
};

// One dispatch stages every row the fire needs, `tid.y` selecting the row.
//
// This used to copy a single row and be submitted as its OWN command buffer,
// once per row -- so a sixteen-request fire paid sixteen command-buffer round
// trips per token to move sixteen vocabulary rows. That was ~3ms of a 23.5ms
// step, and it scaled linearly with the batch, which is what made the sampler
// look linear in lanes when the sampler itself is 0.5ms of GPU.
kernel void ptir_copy_logits_bf16(
    const device bfloat* source [[buffer(0)]],
    device bfloat* destination [[buffer(1)]],
    const device PtirLogitsCopyParams* params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  const device PtirLogitsCopyParams& p = params[tid.y];
  if (tid.x >= p.vocab) return;
  destination[ulong(p.destination_row) * p.vocab + tid.x] =
      source[ulong(p.source_row) * p.vocab + tid.x];
}

#include <metal_stdlib>
using namespace metal;

struct PtirLogitsCopyParams {
  uint source_row;
  uint destination_row;
  uint vocab;
  uint reserved;
};

kernel void copy_logits_bf16(
    const device bfloat* source [[buffer(0)]],
    device bfloat* destination [[buffer(1)]],
    const device PtirLogitsCopyParams* params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  const device PtirLogitsCopyParams& p = params[tid.y];
  if (tid.x >= p.vocab) return;
  destination[ulong(p.destination_row) * p.vocab + tid.x] =
      source[ulong(p.source_row) * p.vocab + tid.x];
}

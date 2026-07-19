#include <metal_stdlib>
using namespace metal;

struct PtirLogitsCopyParams {
  uint source_row;
  uint destination_row;
  uint vocab;
  uint reserved;
};

kernel void ptir_copy_logits_bf16(
    const device bfloat* source [[buffer(0)]],
    device bfloat* destination [[buffer(1)]],
    const device PtirLogitsCopyParams* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params->vocab) return;
  destination[ulong(params->destination_row) * params->vocab + gid] =
      source[ulong(params->source_row) * params->vocab + gid];
}

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

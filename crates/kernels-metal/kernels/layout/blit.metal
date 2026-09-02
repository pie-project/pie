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

// A run of 32-bit words, one thread each. The recurrent buffer's scatter and
// gather (`engine_metal::rs`) are memcpys between an activation rectangle and
// a page slab, both minted at 4-byte-aligned offsets; a kernel rather than a
// blit encoder so the move sits in the compute encoder's own order, between
// the dispatch that wrote the rows and the one that reads them.
kernel void rs_copy_words(
    const device uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    const constant uint& words [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= words) return;
  dst[tid] = src[tid];
}

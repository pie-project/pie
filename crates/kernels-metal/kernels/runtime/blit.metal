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

kernel void rs_copy_words(
    const device uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    const constant uint& words [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= words) return;
  dst[tid] = src[tid];
}

// Up to four word copies in one launch: row `y` of the grid is copy `y`.
// The copies one recurrent op lands are disjoint, so they run side by side.
kernel void rs_copy_words4(
    const device uint* src0 [[buffer(0)]],
    device uint* dst0 [[buffer(1)]],
    const device uint* src1 [[buffer(2)]],
    device uint* dst1 [[buffer(3)]],
    const device uint* src2 [[buffer(4)]],
    device uint* dst2 [[buffer(5)]],
    const device uint* src3 [[buffer(6)]],
    device uint* dst3 [[buffer(7)]],
    const constant uint& words0 [[buffer(8)]],
    const constant uint& words1 [[buffer(9)]],
    const constant uint& words2 [[buffer(10)]],
    const constant uint& words3 [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint w = tid.x;
  switch (tid.y) {
    case 0: if (w < words0) dst0[w] = src0[w]; break;
    case 1: if (w < words1) dst1[w] = src1[w]; break;
    case 2: if (w < words2) dst2[w] = src2[w]; break;
    default: if (w < words3) dst3[w] = src3[w]; break;
  }
}

// Read-only streaming probe: the machine's achievable bandwidth roof, with no
// arithmetic worth counting.  Each thread pulls 8 bf16 as one 16-byte vector and
// folds them so the loads cannot be eliminated; nothing is stored unless the sum
// is non-zero, which zero-filled input guarantees it is not.
#include <metal_stdlib>
using namespace metal;

[[kernel]] void stream_read_bf16(
    const device uint4* src [[buffer(0)]],
    device float* sink      [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  uint4 v = src[tid];
  uint acc = v.x ^ v.y ^ v.z ^ v.w;
  if (acc == 0xFFFFFFFFu) sink[0] = float(acc);
}

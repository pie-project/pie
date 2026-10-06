#include <metal_stdlib>

using namespace metal;

// The MLP's input for the Neural Engine, which reads fp16. `stamp` is the
// shared-event value the host signals behind this dispatch; the kernel
// itself never reads it.
[[kernel]] void ane_stage_bfloat16(
    const device bfloat* x  [[buffer(0)]],
    device half* staged     [[buffer(1)]],
    const constant uint& n  [[buffer(2)]],
    const constant uint& stamp [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) {
    return;
  }
  staged[i] = half(clamp(float(x[i]), -65504.0f, 65504.0f));
}

// Adds the Neural Engine's half of the MLP into the GPU's half. The host
// waits on `stamp` before this dispatch runs.
[[kernel]] void ane_join_bfloat16(
    device bfloat* y          [[buffer(0)]],
    const device half* other  [[buffer(1)]],
    const constant uint& n    [[buffer(2)]],
    const constant uint& stamp [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= n) {
    return;
  }
  y[i] = bfloat(float(y[i]) + float(other[i]));
}

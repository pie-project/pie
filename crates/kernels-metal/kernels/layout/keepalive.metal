#include <metal_stdlib>
using namespace metal;

// A few hundred microseconds of dependent arithmetic on one simdgroup: what
// the shell's keep-alive queue runs while a fire's CPU-side copies leave the
// device idle, so its clocks do not fall between segments. Writes only if the
// impossible happens, so the loop is not folded away.
kernel void keepalive_spin(
    device float* out             [[buffer(0)]],
    const constant uint& iters    [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]]) {
  float v = float(lid) * 1e-3f;
  for (uint i = 0; i < iters; ++i) {
    v = fma(v, 0.9999f, 1e-7f);
  }
  if (v == 12345.678f) out[lid] = v;
}

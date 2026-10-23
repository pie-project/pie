#include <metal_stdlib>
using namespace metal;

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

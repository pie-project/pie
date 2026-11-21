#include <metal_stdlib>
using namespace metal;

// How many full workgroups the GPU keeps resident at once: each group takes
// the most threads a group may and the most threadgroup memory a group may
// (so a core holds as many as its memory and threads allow), counts itself
// in on arrival and the most in at once, works a while, and counts itself
// out. counters[0] is the groups in now, counters[1] the most at once.
kernel void resident_groups(
    device atomic_uint* counters  [[buffer(0)]],
    const constant uint& iters    [[buffer(1)]],
    device float* sink            [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  threadgroup float held[8192];
  if (lid == 0) {
    const uint now = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed) + 1u;
    atomic_fetch_max_explicit(&counters[1], now, memory_order_relaxed);
  }
  float v = float(lid) * 1e-3f;
  for (uint i = 0; i < iters; ++i) {
    v = fma(v, 0.9999f, 1e-7f);
  }
  held[lid % 8192u] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < threads && i < 8192u; i += 64u) total += held[i];
    if (total == 12345.678f) sink[0] = total;
    atomic_fetch_sub_explicit(&counters[0], 1u, memory_order_relaxed);
  }
}

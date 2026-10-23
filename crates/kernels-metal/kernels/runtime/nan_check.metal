#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void nan_check(
    const device T* x                  [[buffer(0)]],
    device atomic_uint* flags          [[buffer(1)]],
    const constant uint& slot          [[buffer(2)]],
    const constant uint& n             [[buffer(3)]],
    const constant float& limit        [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]) {
  if (gid >= n) return;
  const float v = float(x[gid]);

  if (v != v || fabs(v) > limit) {
    atomic_fetch_max_explicit(&flags[slot], gid + 1u, memory_order_relaxed);
  }
}

#define instantiate_nan_check(name, type)                                \
  template [[host_name("nan_check_" #name)]]                             \
  [[kernel]] void nan_check<type>(                                       \
      const device type*, device atomic_uint*, const constant uint&,     \
      const constant uint&, const constant float&, uint);

instantiate_nan_check(float32, float)
instantiate_nan_check(bfloat16, bfloat)
instantiate_nan_check(float16, half)

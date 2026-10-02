#include <metal_stdlib>
using namespace metal;
kernel void kv_append_paged_q8(
    const device bfloat *k [[buffer(0)]], const device bfloat *v [[buffer(1)]],
    device char *kp [[buffer(2)]], device char *vp [[buffer(3)]], constant int &d [[buffer(5)]],
    constant int &page_size [[buffer(10)]], constant int &heads [[buffer(12)]],
    const device uint *pages [[buffer(13)]], const device uint *offsets [[buffer(14)]],
    constant int &src_stride [[buffer(15)]], uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint h = group.y, row = group.z;
  const ulong src = ulong(row) * (src_stride > 0 ? src_stride : heads * d) + h * d;
  const ulong slot = ulong(pages[row]) * page_size + offsets[row];
  const ulong dst = (slot * heads + h) * ulong(2 * d);
  float km = 0, vm = 0;
  for (uint i = lane; i < uint(d); i += 32) {
    km = max(km, abs(float(k[src + i])));
    vm = max(vm, abs(float(v[src + i])));
  }
  km = simd_max(km);
  vm = simd_max(vm);
  const float ki = km > 0 ? 127.0f / km : 0, vi = vm > 0 ? 127.0f / vm : 0;
  for (uint i = lane; i < uint(d); i += 32) {
    kp[dst + i] = char(clamp(rint(float(k[src + i]) * ki), -127.0f, 127.0f));
    if (vp != kp)
      vp[dst + i] = char(clamp(rint(float(v[src + i]) * vi), -127.0f, 127.0f));
  }
  if (lane == 0) {
    *reinterpret_cast<device float *>(kp + dst + d) = km / 127.0f;
    if (vp != kp)
      *reinterpret_cast<device float *>(vp + dst + d) = vm / 127.0f;
  }
}

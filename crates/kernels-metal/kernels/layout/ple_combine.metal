#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void ple_combine(
    const device T* proj            [[buffer(0)]],
    const device T* token           [[buffer(1)]],
    device T* out                   [[buffer(2)]],
    const constant float& inv_sqrt2 [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]) {
  const float v = (static_cast<float>(proj[gid]) + static_cast<float>(token[gid])) *
                  inv_sqrt2;
  out[gid] = static_cast<T>(v);
}

#define instantiate_ple_combine(name, itype)                           \
  template [[host_name("ple_combine_" #name)]]                         \
  [[kernel]] void ple_combine<itype>(                                  \
      const device itype*, const device itype*, device itype*,         \
      const constant float&, uint);

instantiate_ple_combine(bfloat16, bfloat)

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void add_bias(
    device T* out            [[buffer(0)]],
    const device T* bias     [[buffer(1)]],
    const constant int& width [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(width) + size_t(tid.x);
  out[i] = T(float(out[i]) + float(bias[tid.x]));
}

#define instantiate_add_bias(name, itype)                          \
  template [[host_name("add_bias_" #name)]]                        \
  [[kernel]] void add_bias<itype>(                                 \
      device itype*, const device itype*, const constant int&, uint2);

instantiate_add_bias(bfloat16, bfloat)

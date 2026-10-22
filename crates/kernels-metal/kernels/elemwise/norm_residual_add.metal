#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void residual_add(
    const device T* x        [[buffer(0)]],
    const device T* residual [[buffer(1)]],
    device T* out            [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = T(float(x[tid]) + float(residual[tid]));
}

#define instantiate_residual_add(name, itype)                     \
  template [[host_name("residual_add_" #name)]]                   \
  [[kernel]] void residual_add<itype>(                            \
      const device itype*, const device itype*, device itype*, uint);

instantiate_residual_add(bfloat16, bfloat)
instantiate_residual_add(float32, float)

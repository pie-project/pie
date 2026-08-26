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

template <typename T>
[[kernel]] void residual_add_strided(
    const device T* x        [[buffer(0)]],
    const device T* residual [[buffer(1)]],
    device T* out            [[buffer(2)]],
    const constant int& row_pitch [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(row_pitch) + size_t(tid.x);
  out[i] = T(float(x[i]) + float(residual[i]));
}

#define instantiate_residual_add_strided(name, itype)             \
  template [[host_name("residual_add_strided_" #name)]]           \
  [[kernel]] void residual_add_strided<itype>(                    \
      const device itype*, const device itype*, device itype*,    \
      const constant int&, uint2);

instantiate_residual_add_strided(bfloat16, bfloat)

#define instantiate_residual_add(name, itype)                     \
  template [[host_name("residual_add_" #name)]]                   \
  [[kernel]] void residual_add<itype>(                            \
      const device itype*, const device itype*, device itype*, uint);

instantiate_residual_add(bfloat16, bfloat)

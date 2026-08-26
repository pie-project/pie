#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void layer_scalar_mul(
    const device T* x                [[buffer(0)]],
    const device T* scalar           [[buffer(1)]],
    device T* out                    [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]) {
  const float s = static_cast<float>(scalar[0]);
  out[gid] = static_cast<T>(static_cast<float>(x[gid]) * s);
}

#define instantiate_layer_scalar(name, itype)                          \
  template [[host_name("layer_scalar_mul_" #name)]]                    \
  [[kernel]] void layer_scalar_mul<itype>(                             \
      const device itype*, const device itype*, device itype*, uint);

instantiate_layer_scalar(bfloat16, bfloat)

template <typename T>
[[kernel]] void layer_scalar_mul_stated(
    const device T* x                [[buffer(0)]],
    const constant float& scalar     [[buffer(1)]],
    device T* out                    [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]) {
  const float s = static_cast<float>(static_cast<T>(scalar));
  out[gid] = static_cast<T>(static_cast<float>(x[gid]) * s);
}

#define instantiate_layer_scalar_stated(name, itype)                   \
  template [[host_name("layer_scalar_mul_stated_" #name)]]             \
  [[kernel]] void layer_scalar_mul_stated<itype>(                      \
      const device itype*, const constant float&, device itype*, uint);

instantiate_layer_scalar_stated(bfloat16, bfloat)

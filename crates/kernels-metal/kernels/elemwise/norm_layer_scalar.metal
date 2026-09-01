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

// `silu(s * x)`, in place — qwen4's shared-expert gate, whose scale is a plan
// constant and not a plane. The scalar is a launch argument and moves with
// nothing; the grid is exact, so no lane is out of the row.
template <typename T>
[[kernel]] void silu_scaled(
    device T* x                      [[buffer(0)]],
    const constant float& scalar     [[buffer(1)]],
    uint gid                         [[thread_position_in_grid]]) {
  const float v = static_cast<float>(x[gid]) * scalar;
  x[gid] = static_cast<T>(v / (1.0f + precise::exp(-v)));
}

#define instantiate_silu_scaled(name, itype)                           \
  template [[host_name("silu_scaled_" #name)]]                         \
  [[kernel]] void silu_scaled<itype>(                                  \
      device itype*, const constant float&, uint);

instantiate_silu_scaled(bfloat16, bfloat)

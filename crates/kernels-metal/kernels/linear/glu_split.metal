#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void swiglu_clamp(
    const device T* gate        [[buffer(0)]],
    const device T* up          [[buffer(1)]],
    device T* out               [[buffer(2)]],
    const constant float& limit [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  float g = float(gate[gid]);
  float u = float(up[gid]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  out[gid] = static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
}

#define instantiate_swiglu_clamp(name, itype)                      \
  template [[host_name("swiglu_clamp_" #name)]]                    \
  [[kernel]] void swiglu_clamp<itype>(                             \
      const device itype*, const device itype*, device itype*,     \
      const constant float&, uint);

instantiate_swiglu_clamp(bfloat16, bfloat)

inline float gelu_tanh(float x) {
  constexpr float k = 0.7978845608028654f;
  const float inner = k * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + precise::tanh(inner));
}

template <typename T>
[[kernel]] void geglu_tanh(
    const device T* gate      [[buffer(0)]],
    const device T* up        [[buffer(1)]],
    device T* out             [[buffer(2)]],
    uint gid                  [[thread_position_in_grid]]) {
  const float g = gelu_tanh(static_cast<float>(gate[gid]));
  out[gid] = static_cast<T>(g * static_cast<float>(up[gid]));
}

#define instantiate_geglu_tanh(name, itype)                            \
  template [[host_name("geglu_tanh_" #name)]]                          \
  [[kernel]] void geglu_tanh<itype>(                                   \
      const device itype*, const device itype*, device itype*, uint);

instantiate_geglu_tanh(bfloat16, bfloat)

template <typename T>
[[kernel]] void mlp_gelu_tanh(
    const device T* x         [[buffer(0)]],
    device T* out             [[buffer(1)]],
    uint gid                  [[thread_position_in_grid]]) {
  out[gid] = static_cast<T>(gelu_tanh(static_cast<float>(x[gid])));
}

#define instantiate_mlp_gelu_tanh(name, itype)                         \
  template [[host_name("mlp_gelu_tanh_" #name)]]                       \
  [[kernel]] void mlp_gelu_tanh<itype>(                                \
      const device itype*, device itype*, uint);

instantiate_mlp_gelu_tanh(bfloat16, bfloat)


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

// packed_swiglu_clamp_alpha over a gate and an up plane held apart.
template <typename T>
[[kernel]] void swiglu_clamp_alpha(
    const device T* gate        [[buffer(0)]],
    const device T* up          [[buffer(1)]],
    device T* out               [[buffer(2)]],
    const constant float& limit [[buffer(3)]],
    const constant float& alpha [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  float g = float(gate[gid]);
  float u = float(up[gid]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  const float sig = 1.0f / (1.0f + fast::exp(-alpha * g));
  out[gid] = static_cast<T>((g * sig) * (u + 1.0f));
}

#define instantiate_swiglu_clamp_alpha(name, itype)                \
  template [[host_name("swiglu_clamp_alpha_" #name)]]              \
  [[kernel]] void swiglu_clamp_alpha<itype>(                       \
      const device itype*, const device itype*, device itype*,     \
      const constant float&, const constant float&, uint);

instantiate_swiglu_clamp_alpha(bfloat16, bfloat)

// packed_situ over a gate and an up plane held apart.
template <typename T>
[[kernel]] void situ(
    const device T* gate         [[buffer(0)]],
    const device T* up           [[buffer(1)]],
    device T* out                [[buffer(2)]],
    const constant float& beta   [[buffer(3)]],
    const constant float& up_cap [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const float g = float(gate[gid]);
  float u = float(up[gid]);
  const float s = beta * precise::tanh(g / beta) / (1.0f + metal::exp(-g));
  if (up_cap > 0.0f) {
    u = up_cap * precise::tanh(u / up_cap);
  }
  out[gid] = static_cast<T>(s * u);
}

#define instantiate_situ(name, itype)                              \
  template [[host_name("situ_" #name)]]                            \
  [[kernel]] void situ<itype>(                                     \
      const device itype*, const device itype*, device itype*,     \
      const constant float&, const constant float&, uint);

instantiate_situ(bfloat16, bfloat)

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

// `relu(x)^2`, element for element (Nemotron-H's experts').
template <typename T>
[[kernel]] void mlp_relu2(
    const device T* x         [[buffer(0)]],
    device T* out             [[buffer(1)]],
    uint gid                  [[thread_position_in_grid]]) {
  const float r = metal::max(static_cast<float>(x[gid]), 0.0f);
  out[gid] = static_cast<T>(r * r);
}

template [[host_name("mlp_relu2_bfloat16")]] [[kernel]] void mlp_relu2<bfloat>(
    const device bfloat*, device bfloat*, uint);


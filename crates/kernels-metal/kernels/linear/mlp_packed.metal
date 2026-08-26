#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void packed_swiglu(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  const float u = float(packed[packed_row + intermediate + i]);
  out[row + i] = static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
}

#define instantiate_packed_swiglu(name, itype)                     \
  template [[host_name("packed_swiglu_" #name)]]                   \
  [[kernel]] void packed_swiglu<itype>(                            \
      const device itype*, device itype*,                          \
      const constant uint&, uint2);

instantiate_packed_swiglu(bfloat16, bfloat)

template <typename T>
[[kernel]] void packed_swiglu_clamp(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& limit       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  float g = float(packed[packed_row + i]);
  float u = float(packed[packed_row + intermediate + i]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  out[row + i] = static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
}

#define instantiate_packed_swiglu_clamp(name, itype)               \
  template [[host_name("packed_swiglu_clamp_" #name)]]             \
  [[kernel]] void packed_swiglu_clamp<itype>(                      \
      const device itype*, device itype*,                          \
      const constant uint&, const constant float&, uint2);

instantiate_packed_swiglu_clamp(bfloat16, bfloat)

template <typename T>
[[kernel]] void packed_gptoss_swiglu(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& limit       [[buffer(3)]],
    const constant float& alpha       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  float g = float(packed[packed_row + i]);
  float u = float(packed[packed_row + intermediate + i]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  const float sig = 1.0f / (1.0f + fast::exp(-alpha * g));
  out[row + i] = static_cast<T>((g * sig) * (u + 1.0f));
}

#define instantiate_packed_gptoss_swiglu(name, itype)              \
  template [[host_name("packed_gptoss_swiglu_" #name)]]            \
  [[kernel]] void packed_gptoss_swiglu<itype>(                     \
      const device itype*, device itype*,                          \
      const constant uint&, const constant float&,                 \
      const constant float&, uint2);

instantiate_packed_gptoss_swiglu(bfloat16, bfloat)

template <typename T>
[[kernel]] void packed_geglu_tanh(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  constexpr float k = 0.7978845608028654f;
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  const float u = float(packed[packed_row + intermediate + i]);
  const float inner = k * (g + 0.044715f * g * g * g);
  const float gelu = 0.5f * g * (1.0f + precise::tanh(inner));
  out[row + i] = static_cast<T>(gelu * u);
}

#define instantiate_packed_geglu_tanh(name, itype)                 \
  template [[host_name("packed_geglu_tanh_" #name)]]               \
  [[kernel]] void packed_geglu_tanh<itype>(                        \
      const device itype*, device itype*,                          \
      const constant uint&, uint2);

instantiate_packed_geglu_tanh(bfloat16, bfloat)

template <typename T>
[[kernel]] void packed_situ(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& beta        [[buffer(3)]],
    const constant float& up_cap      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  float u = float(packed[packed_row + intermediate + i]);
  const float s = beta * precise::tanh(g / beta) / (1.0f + metal::exp(-g));
  if (up_cap > 0.0f) {
    u = up_cap * precise::tanh(u / up_cap);
  }
  out[row + i] = static_cast<T>(s * u);
}

#define instantiate_packed_situ(name, itype)                       \
  template [[host_name("packed_situ_" #name)]]                     \
  [[kernel]] void packed_situ<itype>(                              \
      const device itype*, device itype*,                          \
      const constant uint&, const constant float&,                 \
      const constant float&, uint2);

instantiate_packed_situ(bfloat16, bfloat)

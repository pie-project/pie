#include <metal_stdlib>

using namespace metal;

template <typename T>
inline T sigmoid_mlx(T x) {
  float xf = float(x);
  float y = 1.0f / (1.0f + metal::exp(-metal::fabs(xf)));
  float s = (xf < 0.0f) ? (1.0f - y) : y;
  return T(s);
}

template <typename T>
[[kernel]] void silu_mul(
    const device T* gate [[buffer(0)]],
    const device T* up   [[buffer(1)]],
    device T* out        [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  T g   = gate[tid];
  T sg  = sigmoid_mlx(g);
  T sil = T(float(g) * float(sg));
  out[tid] = T(float(sil) * float(up[tid]));
}

template <typename T>
[[kernel]] void silu_mul_strided(
    const device T* gate [[buffer(0)]],
    const device T* up   [[buffer(1)]],
    device T* out        [[buffer(2)]],
    const constant int& row_pitch [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(row_pitch) + size_t(tid.x);
  T g   = gate[i];
  T sg  = sigmoid_mlx(g);
  T sil = T(float(g) * float(sg));
  out[i] = T(float(sil) * float(up[i]));
}

#define instantiate_silu_mul_strided(name, itype)                 \
  template [[host_name("silu_mul_strided_" #name)]]               \
  [[kernel]] void silu_mul_strided<itype>(                        \
      const device itype*, const device itype*, device itype*,    \
      const constant int&, uint2);

instantiate_silu_mul_strided(bfloat16, bfloat)

#define instantiate_silu_mul(name, itype)                         \
  template [[host_name("silu_mul_" #name)]]                       \
  [[kernel]] void silu_mul<itype>(                                \
      const device itype*, const device itype*, device itype*, uint);

instantiate_silu_mul(bfloat16, bfloat)

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
[[kernel]] void geglu_tanh_strided(
    const device T* gate            [[buffer(0)]],
    const device T* up              [[buffer(1)]],
    device T* out                   [[buffer(2)]],

    const constant uint& width      [[buffer(3)]],
    const constant uint& rows       [[buffer(4)]],
    const constant uint& gate_pitch [[buffer(5)]],
    const constant uint& up_pitch   [[buffer(6)]],
    const constant uint& out_pitch  [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  (void)rows;
  const uint m = gid / width;
  const uint k = gid - m * width;
  const float g = float(gate[size_t(m) * size_t(gate_pitch) + k]);
  const float u = float(up[size_t(m) * size_t(up_pitch) + k]);
  out[size_t(m) * size_t(out_pitch) + k] = static_cast<T>(gelu_tanh(g) * u);
}

#define instantiate_geglu_strided(name, itype)                     \
  template [[host_name("geglu_tanh_strided_" #name)]]              \
  [[kernel]] void geglu_tanh_strided<itype>(                       \
      const device itype*, const device itype*, device itype*,     \
      const constant uint&, const constant uint&, const constant uint&, \
      const constant uint&, const constant uint&, uint);

instantiate_geglu_strided(bfloat16, bfloat)

template <typename T>
[[kernel]] void gptoss_swiglu(
    const device T* gate            [[buffer(0)]],
    const device T* up              [[buffer(1)]],
    device T* out                   [[buffer(2)]],
    const constant float& limit     [[buffer(3)]],
    const constant float& alpha     [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  float g = float(gate[gid]);
  float u = float(up[gid]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  const float sig = 1.0f / (1.0f + fast::exp(-alpha * g));
  out[gid] = static_cast<T>((g * sig) * (u + 1.0f));
}

#define instantiate_gptoss_swiglu(name, itype)                     \
  template [[host_name("gptoss_swiglu_" #name)]]                   \
  [[kernel]] void gptoss_swiglu<itype>(                            \
      const device itype*, const device itype*, device itype*,     \
      const constant float&, const constant float&, uint);

instantiate_gptoss_swiglu(bfloat16, bfloat)

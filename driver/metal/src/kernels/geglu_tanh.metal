// Raw-Metal gemma4 GeGLU-tanh activation (decode, M=1).
//
//   out[i] = gelu_tanh(gate[i]) * up[i]
//   gelu_tanh(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
//
// gemma4's FFN + PLE blocks use the tanh-approx GeGLU (cf. qwen3.6's silu_mul).
// Elementwise over the intermediate width N; one thread per element. Compute in
// float for parity; bfloat is native on Metal 4. Mirrors MLX's `geglu` /
// `gelu_approx` so the math is bit-exact by construction.
// bind::Geglu = { Gate=0, Up=1, Out=2 }; N is static geometry (GegluParams).

#include <metal_stdlib>
using namespace metal;

struct GegluParams {
  uint n;  // element count (intermediate width)
};

inline float gelu_tanh(float x) {
  constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
  const float inner = k * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + precise::tanh(inner));
}

template <typename T>
[[kernel]] void geglu_tanh(
    const device T* gate      [[buffer(0)]],
    const device T* up        [[buffer(1)]],
    device T* out             [[buffer(2)]],
    constant GegluParams& p   [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const float g = gelu_tanh(static_cast<float>(gate[gid]));
  out[gid] = static_cast<T>(g * static_cast<float>(up[gid]));
}

#define instantiate_geglu_tanh(name, itype)                            \
  template [[host_name("geglu_tanh_" #name)]]                          \
  [[kernel]] void geglu_tanh<itype>(                                   \
      const device itype*, const device itype*, device itype*,         \
      constant GegluParams&, uint);

instantiate_geglu_tanh(float32, float)
instantiate_geglu_tanh(float16, half)
instantiate_geglu_tanh(bfloat16, bfloat)

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

instantiate_geglu_tanh(bfloat16, bfloat)

// ── Strided variant: the operands are rows of DIFFERENT pitches ──────────────
//
// gemma4's per-layer-embedding GeGLU gates the residual stream by one layer's
// slice of the PLE table. At M=1 that slice is a byte offset and the flat kernel
// above serves. At M>1 it is not: the table is `[rows, n_layers*ple_dim]` row
// major, so layer L's slice is `ple_dim` wide with a stride of `n_layers*ple_dim`
// between rows, while the gate and the output are densely `[rows, ple_dim]`.
//
// A byte offset cannot express that, and the flat kernel reading it walks into
// the NEXT layers' slices after the first row -- which is not a crash and not
// even implausible numbers, since those slices are the same table.
//
// So the pitches are stated. At rows==1 every pitch is unused and this is the
// flat kernel with an offset, which is what the M=1 path keeps doing.
struct GegluStridedParams {
  uint width;       // elements per row (ple_dim)
  uint rows;        // token rows
  uint gate_pitch;  // elements between rows of `gate`
  uint up_pitch;    // ... of `up` -- the wide one
  uint out_pitch;   // ... of `out`
};

template <typename T>
[[kernel]] void geglu_tanh_strided(
    const device T* gate            [[buffer(0)]],
    const device T* up              [[buffer(1)]],
    device T* out                   [[buffer(2)]],
    constant GegluStridedParams& p  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint k = gid.x;
  const uint m = gid.y;
  if (k >= p.width || m >= p.rows) return;
  const float g = float(gate[size_t(m) * size_t(p.gate_pitch) + k]);
  const float u = float(up[size_t(m) * size_t(p.up_pitch) + k]);
  out[size_t(m) * size_t(p.out_pitch) + k] = static_cast<T>(gelu_tanh(g) * u);
}

#define instantiate_geglu_strided(name, itype)                     \
  template [[host_name("geglu_tanh_strided_" #name)]]              \
  [[kernel]] void geglu_tanh_strided<itype>(                       \
      const device itype*, const device itype*, device itype*,     \
      constant GegluStridedParams&, uint2);

instantiate_geglu_strided(bfloat16, bfloat)

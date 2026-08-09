// The gated activations: one file because they are one BINDING CONTRACT.
//
// All three take `(gate, up, out)` at buffers 0, 1, 2 and differ only in the
// arithmetic between them -- and, for two of the three, in a params buffer at
// 3. They were three files of forty to ninety lines, which is a directory
// listing that tells a reader nothing about which one their tensors fit.
//
//   silu_mul       out = silu(gate) * up                    no params
//   geglu_tanh     out = gelu_tanh(gate) * up               GegluParams
//   gptoss_swiglu  gpt-oss's clamped, alpha-scaled SwiGLU   GptOssSwiGluParams
//
// The third earns its model name and keeps it: it bakes gpt-oss's asymmetric
// clamp, its `alpha` and its `(up + 1)` term, which is nobody else's SwiGLU.

#include <metal_stdlib>

using namespace metal;

// MLX numerically-stable sigmoid (unary_ops.h Sigmoid); compute in float, round to T.
template <typename T>
inline T sigmoid_mlx(T x) {
  float xf = float(x);
  float y = 1.0f / (1.0f + metal::exp(-metal::fabs(xf)));
  float s = (xf < 0.0f) ? (1.0f - y) : y;
  return T(s);
}

template <typename T>
[[kernel]] void silu_mul(
    const device T* gate [[buffer(0)]],   // [intermediate]
    const device T* up   [[buffer(1)]],   // [intermediate]
    device T* out        [[buffer(2)]],   // [intermediate]
    uint tid [[thread_position_in_grid]]) {
  T g   = gate[tid];
  T sg  = sigmoid_mlx(g);                  // sigmoid(gate), rounded to T
  T sil = T(float(g) * float(sg));         // silu(gate) = gate*sigmoid(gate), round
  out[tid] = T(float(sil) * float(up[tid]));
}

// Prefill variant: rows are a uniform `row_pitch` elements apart, so the whole
// prompt runs as one dispatch.  tid.y selects the row; the arithmetic is identical.
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

// THE GRID IS THE EXTENT, so this carries no count.
//
// `n` used to be here and used to be read as `if (gid >= p.n) return;`. It
// was stated by the text as the INTERMEDIATE WIDTH -- one row -- and the
// dispatch covers `width * rows`, so every row after the first returned
// immediately. A prefill's second token came back as zeros; a decode is one
// row and never noticed.
//
// A per-row number cannot bound a whole-tensor dispatch, and the text cannot
// state the whole: its shape says `[Tokens, intermediate]` and `Tokens` is
// not known until a fire lowers. The driver is what knows, and it already
// spends that knowledge on the grid -- `Rule::Elementwise` sizes on the
// output operand, so the dispatch is exactly the element count and a second
// bound is a second answer to a question already answered.
//
// `silu_mul` above takes no params for exactly this reason and has always
// been right. This is now the same shape.
struct GegluParams {
  uint unused;
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
  (void)p;
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

struct GptOssSwiGluParams {
  uint unused;   // was a per-row element count -- see `GegluParams`
  float limit;   // 7.0
  float alpha;   // 1.702
};

// gpt-oss's SwiGLU, which is not anyone else's.
//
//   gate = min(gate, limit)              -- clamped ABOVE only
//   up   = clamp(up, -limit, limit)      -- clamped both ways
//   out  = gate * sigmoid(alpha*gate) * (up + 1)
//
// The `+1` on the linear branch and the asymmetric clamp are why `silu_mul`
// cannot serve: dropping either produces a model that runs and is wrong.
template <typename T>
[[kernel]] void gptoss_swiglu(
    const device T* gate            [[buffer(0)]],
    const device T* up              [[buffer(1)]],
    device T* out                   [[buffer(2)]],
    constant GptOssSwiGluParams& p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  float g = float(gate[gid]);
  float u = float(up[gid]);
  g = min(g, p.limit);
  u = clamp(u, -p.limit, p.limit);
  const float sig = 1.0f / (1.0f + fast::exp(-p.alpha * g));
  out[gid] = static_cast<T>((g * sig) * (u + 1.0f));
}

#define instantiate_gptoss_swiglu(name, itype)                     \
  template [[host_name("gptoss_swiglu_" #name)]]                   \
  [[kernel]] void gptoss_swiglu<itype>(                            \
      const device itype*, const device itype*, device itype*,     \
      constant GptOssSwiGluParams&, uint);

instantiate_gptoss_swiglu(bfloat16, bfloat)

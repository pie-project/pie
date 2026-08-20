// The gated activations: one file because they are one BINDING CONTRACT.
//
// All three take `(gate, up, out)` at buffers 0, 1, 2 and differ only in the
// arithmetic between them -- and, for one of the three, in a params buffer at
// 3. They were three files of forty to ninety lines, which is a directory
// listing that tells a reader nothing about which one their tensors fit.
//
//   silu_mul       out = silu(gate) * up                    no params
//   geglu_tanh     out = gelu_tanh(gate) * up               no params
//   gptoss_swiglu  gpt-oss's clamped, alpha-scaled SwiGLU   limit, alpha
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

// THE GRID IS THE EXTENT, so this carries no count -- and, now, no params
// buffer either.
//
// `struct GegluParams { uint unused; }` stood here, taken by `geglu_tanh` as
// `constant GegluParams& p [[buffer(3)]]` and immediately `(void)p`'d. Its one
// field was `n`, read as `if (gid >= p.n) return;`. `n` was stated by the text
// as the INTERMEDIATE WIDTH -- one row -- and the dispatch covers
// `width * rows`, so every row after the first returned immediately. A
// prefill's second token came back as zeros; a decode is one row and never
// noticed.
//
// A per-row number cannot bound a whole-tensor dispatch, and the text cannot
// state the whole: its shape says `[Tokens, intermediate]` and `Tokens` is
// not known until a fire lowers. The driver is what knows, and it already
// spends that knowledge on the grid -- `Rule::Elementwise` sizes on the
// output operand, so the dispatch is exactly the element count and a second
// bound is a second answer to a question already answered.
//
// So the bound went and the field stayed, renamed `unused`, holding the
// struct's size open. What kept the ARGUMENT after that was the row: it stated
// `params: Buf`, an argument slot the encoder had to fill because an argument
// table with a hole in it is not something it can be asked for, and all three
// backends shaped their bind layouts from the same row. With the row retired
// and `mlp::geglu_tanh` no longer calling `ctx.params()`, one field nothing
// reads is not worth a buffer, a staged word and a slot: buffer 3 is simply
// absent from this entrypoint and from its `instantiate` list.
//
// `silu_mul` above takes no params for exactly this reason and has always
// been right. This is now literally the same shape, and not merely the same
// shape with a dead argument on the end.
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
    uint gid                  [[thread_position_in_grid]]) {
  const float g = gelu_tanh(static_cast<float>(gate[gid]));
  out[gid] = static_cast<T>(g * static_cast<float>(up[gid]));
}

#define instantiate_geglu_tanh(name, itype)                            \
  template [[host_name("geglu_tanh_" #name)]]                          \
  [[kernel]] void geglu_tanh<itype>(                                   \
      const device itype*, const device itype*, device itype*, uint);

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

// THE GRID IS THE EXTENT, and the second word is dead for the same reason
// `GegluParams`' one word was, before that struct was deleted outright.
//
// This body read `gid.y` as the row and bounded itself with
// `if (k >= p.width || m >= p.rows) return;`. Its row states
// `LaunchRule::Elementwise`, which is `[width * rows, 1, 1]` -- FLAT -- so
// `gid.y` was always zero and every row but the first returned without
// writing. Rows 1..N of gemma's PLE tail kept whatever the arena was born
// with, on every layer of every prefill longer than one token, and the
// dispatch succeeded. Measured at 8 rows through `model_dispatch`: a grid of
// [2048, 1, 1] against a stated `(256 wide, 1 rows)` reached 256 of 2048
// elements.
//
// The second failure is the reason the guard cannot simply be kept and the
// grid made 2D: `dsl::metal::geglu_strided` states the row count as the
// literal `1`, with a comment that the count "is the fire's and rides the
// shape". Nothing fills it. `p.rows` is not a row count and never was, so any
// bound computed from it clamps to one row -- which is the very defect,
// wearing the other mask.
//
// A count a text cannot state has one honest answer here, and this file
// already gives it four times: `silu_mul` takes no params at all and indexes
// `tid` raw. Metal dispatches EXACTLY the threads asked for
// (`dispatchThreads:threadsPerThreadgroup:`, non-uniform threadgroups), so
// the grid is the bound, no guard is needed, and the row and the body agree
// without a table, text or driver edit. `kernels-vulkan` and `kernels-wgpu`
// flattened their bodies for the same reason; their guard still reads
// `p.rows` and so is still capped at one row.
template <typename T>
[[kernel]] void geglu_tanh_strided(
    const device T* gate            [[buffer(0)]],
    const device T* up              [[buffer(1)]],
    device T* out                   [[buffer(2)]],
    // THE FIVE THAT WERE `GegluStridedParams`, one `setBytes` apiece. `rows`
    // is bound and not read here: the grid is the extent on this plane, and
    // the field is kept in the argument list because the ROW states five words
    // and the three planes share that run field for field.
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

// `GptOssSwiGluParams` opened with a per-row element count, dead for the reason
// stated at length above `gelu_tanh` -- the same number `GegluParams` held, and
// it outlived that struct only because `limit` and `alpha` beside it are read.
// A struct has to carry a dead field to keep the live ones at their offsets;
// two `setBytes` arguments do not, so the dead word is gone from the ABI here
// and `gptoss_swiglu` declares a slot-holder mark for it host-side instead.

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
    const constant float& limit     [[buffer(3)]],   // 7.0
    const constant float& alpha     [[buffer(4)]],   // 1.702
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

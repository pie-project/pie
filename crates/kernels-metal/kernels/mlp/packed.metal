// The packed activations: one file because they are one BINDING CONTRACT.
//
// `mlp/gated.metal` beside this one takes `(gate, up, out)` at buffers 0, 1
// and 2 -- two planes some earlier dispatch already cut apart. These five take
// the ONE row a fused gate/up projection wrote and cut it themselves:
//
//   packed[n, 0 .. I)    the gate half
//   packed[n, I .. 2I)   the up half
//
// so the contract is `(packed, out, I, ...)` at 0, 1, 2 and up. That is a
// DIFFERENT contract, not a fourth arithmetic under the same one, and the two
// files are two for exactly that reason: a kernel from either bound against
// the other's operands does not fault -- it reads the gate half as a whole
// activation and returns a plausible wrong number.
//
//   packed_swiglu         y = silu(g) * u                       no params
//   packed_swiglu_clamp   both halves clamped, then SwiGLU      limit
//   packed_gptoss_swiglu  gpt-oss's asymmetric clamp and alpha  limit, alpha
//   packed_geglu_tanh     y = gelu_tanh(g) * u                  no params
//   packed_situ           SiTU's tanh-saturated gate            beta, up_cap
//
// THE GATE HALF IS FIRST IN ALL FIVE. `mlp/swiglu.cuh` carries a
// `GateSecond` template parameter and a `_gate_second` twin for three of
// these, because some checkpoints export `[up | gate]`; no point on the
// declaration floor states which order it holds, so a second entrypoint here
// would be a name nothing can ask for, reading like a choice.
//
// THE GRID IS THE EXTENT, so no kernel here takes a row count and none of
// them guards. The dispatch is `[I, rows]` and Metal runs exactly the threads
// asked for (`dispatchThreads:threadsPerThreadgroup:`, non-uniform
// threadgroups), which is the reasoning `gated.metal` sets out at length
// above its `geglu_tanh`.
//
// `intermediate` is bound all the same, and it is not a second bound: it is
// the stride from a row's gate half to its up half. `swiglu.cuh` draws the
// same line -- an extent the grid computes is geometry and belongs to the
// fire, while an address the kernel computes is layout and belongs to the
// kernel -- and `I` is on the layout side of it.

#include <metal_stdlib>

using namespace metal;

/// `y = silu(gate) * up`, over the packed row.
///
/// `silu(g) = g * sigmoid(g)`, spelled as the division `g / (1 + exp(-g))`
/// that `pie::mlp::chunked_swiglu` spells it as. The `sigmoid_mlx` in
/// `gated.metal` rounds its sigmoid to `T` before the multiply because the
/// split kernel it serves was transcribed from MLX; this one widens once,
/// computes in float and rounds once, which is what every packed form in the
/// tree does.
template <typename T>
[[kernel]] void packed_swiglu(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t half = row * 2;
  const float g = float(packed[half + i]);
  const float u = float(packed[half + intermediate + i]);
  out[row + i] = static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
}

#define instantiate_packed_swiglu(name, itype)                     \
  template [[host_name("packed_swiglu_" #name)]]                   \
  [[kernel]] void packed_swiglu<itype>(                            \
      const device itype*, device itype*,                          \
      const constant uint&, uint2);

instantiate_packed_swiglu(bfloat16, bfloat)

/// SwiGLU with both halves clamped to `limit`.
///
/// The gate is clamped ABOVE ONLY and the up half BOTH WAYS, which is not a
/// symmetry anyone should restore: a gate clamped from below saturates the
/// branch the activation exists to switch off, and the model still runs.
template <typename T>
[[kernel]] void packed_swiglu_clamp(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& limit       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t half = row * 2;
  float g = float(packed[half + i]);
  float u = float(packed[half + intermediate + i]);
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

/// gpt-oss's GLU over the packed row.
///
/// THE SAME ARITHMETIC AS `gated.metal`'s `gptoss_swiglu`, SPELLED THE SAME
/// WAY -- `fast::exp` and not `metal::exp`, `(g * sig) * (u + 1.0f)` and not
/// `(u + 1) * g * sig` -- because the only difference between the two entry
/// points is where the two halves came from, and a second entry into one
/// activation is worth having only if the two agree bit for bit.
///
/// The transcription is the discipline and not what a test can reach: at bf16
/// the fast and the precise exponential round to the same eight mantissa
/// bits, so swapping them is invisible. What a comparison does catch is a
/// symmetric clamp on the gate, a dropped `alpha` or a swapped half, and
/// keeping the spelling identical is what leaves those as the only ways the
/// two can drift.
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
  const size_t half = row * 2;
  float g = float(packed[half + i]);
  float u = float(packed[half + intermediate + i]);
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

/// The GELU-tanh gate over the packed row.
///
/// `k = sqrt(2/pi)` and the cubic coefficient is the canonical 0.044715 that
/// `torch.nn.functional.gelu(approximate="tanh")` uses, which is HF's
/// `gelu_pytorch_tanh`. `precise::tanh` for `gated.metal`'s reason: the fast
/// one is a rational approximation whose error lands in the middle of the
/// gate's range rather than at its tails.
template <typename T>
[[kernel]] void packed_geglu_tanh(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  constexpr float k = 0.7978845608028654f;
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t half = row * 2;
  const float g = float(packed[half + i]);
  const float u = float(packed[half + intermediate + i]);
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

/// SiTU: `beta * tanh(g / beta) * sigmoid(g)`, with an optional tanh soft-cap
/// on the up half.
///
/// Not a SwiGLU variant. The tanh saturates far enough out that the gate is
/// bounded by `beta` rather than by the logit, which is the point of it, and
/// it is why the whole computation stays in float: rounding the inner
/// `g / beta` to bf16 first loses exactly the distinction the saturation
/// exists to make.
///
/// `up_cap <= 0` means NO CAP, which is how a statement that has no soft-cap
/// asks for the plain product without a second entrypoint.
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
  const size_t half = row * 2;
  const float g = float(packed[half + i]);
  float u = float(packed[half + intermediate + i]);
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

// Raw-Metal SwiGLU activation for Phase-0 decode (golden tag `swiglu`).
//
// out = silu(gate) * up = (gate * sigmoid(gate)) * up, elementwise over the MLP
// intermediate dim (3584). Mirrors MLX `swiglu(gate, up) = multiply(silu(gate), up)`
// where `silu(x) = multiply(x, sigmoid(x))` — three op-by-op bf16 roundings, so we
// round at each step (sigmoid -> silu -> *up) to match MLX's eval-per-op semantics.
// sigmoid uses MLX's numerically-stable Sigmoid<T> (unary_ops.h). Used every layer.
//
// Launch: dispatchThreads grid=(intermediate, 1, 1), tg=(256, 1, 1).

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
    const constant int& width [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  (void)width;
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
    const constant int& width [[buffer(3)]],
    const constant int& row_pitch [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  (void)width;
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
      const constant int&, const constant int&, uint2);

instantiate_silu_mul_strided(bfloat16, bfloat)

#define instantiate_silu_mul(name, itype)                         \
  template [[host_name("silu_mul_" #name)]]                       \
  [[kernel]] void silu_mul<itype>(                                \
      const device itype*, const device itype*, device itype*, const constant int&, uint);

instantiate_silu_mul(bfloat16, bfloat)

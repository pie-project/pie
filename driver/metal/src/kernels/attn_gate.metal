// Raw-Metal gated-attention kernels for Phase-0 decode (qwen3.5 full-attn layers).
//
// qwen3.5 full-attention uses GATED attention: q_proj is 2x wide, per head the
// layout is [query(head_dim) | gate(head_dim)] (config n_q*head_dim but stored
// n_q*2*head_dim). The query feeds qk-norm/rope/sdpa; the gate multiplies the
// attention output (sigmoid) before o_proj. See model/qwen36.cpp:132-187.
//
// Two M=1 kernels (charlie's golden tags: q_proj=the 2x tensor, attn_gated=post-gate):
//   * q_gate_split — deinterleave qg[n_q,2,head_dim] -> Q[n_q,head_dim] +
//     gate[n_q,head_dim]. Internal (no golden tag); lets the existing contiguous
//     qk-norm/rope/sdpa ports run unchanged on Q.
//   * attn_gate — attn[i] *= sigmoid(gate[i]), in-place, head-major contiguous
//     [n_q*head_dim]. Emits `attn_gated`. sigmoid matches MLX's numerically-stable
//     Sigmoid<T> (backend/metal/kernels/unary_ops.h) for bit-exact parity.

#include <metal_stdlib>
using namespace metal;

// Deinterleave the 2x-wide q_proj output into contiguous query + gate.
// Launch: dispatchThreads grid=(head_dim, n_q, 1), tg=(head_dim,1,1).
template <typename T>
[[kernel]] void q_gate_split(
    const device T* qg       [[buffer(0)]],  // [n_q, 2, head_dim] interleaved
    device T* q_out          [[buffer(1)]],  // [n_q, head_dim]
    device T* gate_out       [[buffer(2)]],  // [n_q, head_dim]
    const constant int& head_dim [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(tid.x);   // channel within head_dim
  const int h = int(tid.y);   // query head
  const int hd = head_dim;
  const int n_q = int(grid.y);
  const size_t row = size_t(tid.z) * n_q * hd;
  q_out[row + h * hd + i]    = qg[row * 2 + h * 2 * hd + i];
  gate_out[row + h * hd + i] = qg[row * 2 + h * 2 * hd + hd + i];
}

// MLX's numerically-stable sigmoid (unary_ops.h Sigmoid). bf16/f16 have no native
// exp, so MLX upcasts to float internally — compute in float, round to T on store.
template <typename T>
inline T sigmoid_mlx(T x) {
  float xf = float(x);
  float y = 1.0f / (1.0f + metal::exp(-metal::fabs(xf)));
  float s = (xf < 0.0f) ? (1.0f - y) : y;
  return T(s);
}

// Attention output gate: attn *= sigmoid(gate), in-place. head-major contiguous.
// Launch: dispatchThreads grid=(n_q*head_dim, 1, 1), tg=(256,1,1).
template <typename T>
[[kernel]] void attn_gate(
    device T* attn         [[buffer(0)]],  // [n_q*head_dim] in-place
    const device T* gate   [[buffer(1)]],  // [n_q*head_dim]
    const constant int& width [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  (void)width;
  attn[tid] = attn[tid] * sigmoid_mlx(gate[tid]);
}

#define instantiate_q_gate_split(name, itype)                     \
  template [[host_name("q_gate_split_" #name)]]                   \
  [[kernel]] void q_gate_split<itype>(                            \
      const device itype*, device itype*, device itype*,          \
      const constant int&, uint3, uint3);

#define instantiate_attn_gate(name, itype)                        \
  template [[host_name("attn_gate_" #name)]]                      \
  [[kernel]] void attn_gate<itype>(                               \
      device itype*, const device itype*, const constant int&, uint);

instantiate_q_gate_split(float32, float)
instantiate_q_gate_split(float16, half)
instantiate_q_gate_split(bfloat16, bfloat)

instantiate_attn_gate(float32, float)
instantiate_attn_gate(float16, half)
instantiate_attn_gate(bfloat16, bfloat)

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
// Launch: dispatchThreads grid=(head_dim, n_q, rows), tg=(head_dim,1,1).
//
// z is the token row. It always was -- a decode simply passes one -- and the
// only thing that stopped a prefill from putting its whole prompt there was the
// pitch: `qg_row_stride` and `out_row_stride` are 0 for the packed layout a
// decode hands over, and the scratch arena's common width for a prefill, whose
// rows are as far apart as the widest tensor in the step regardless of how
// narrow this one is.
template <typename T>
[[kernel]] void q_gate_split(
    const device T* qg       [[buffer(0)]],  // [rows, n_q, 2, head_dim] interleaved
    device T* q_out          [[buffer(1)]],  // [rows, n_q, head_dim]
    device T* gate_out       [[buffer(2)]],  // [rows, n_q, head_dim]
    const constant int& head_dim [[buffer(3)]],
    const constant int& qg_row_stride  [[buffer(4)]],
    const constant int& out_row_stride [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(tid.x);   // channel within head_dim
  const int h = int(tid.y);   // query head
  const int hd = head_dim;
  const int n_q = int(grid.y);
  const size_t out_row = out_row_stride > 0 ? size_t(tid.z) * size_t(out_row_stride)
                                            : size_t(tid.z) * n_q * hd;
  const size_t qg_row  = qg_row_stride > 0 ? size_t(tid.z) * size_t(qg_row_stride)
                                           : size_t(tid.z) * n_q * hd * 2;
  q_out[out_row + h * hd + i]    = qg[qg_row + h * 2 * hd + i];
  gate_out[out_row + h * hd + i] = qg[qg_row + h * 2 * hd + hd + i];
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
// Launch: dispatchThreads grid=(n_q*head_dim, rows, 1), tg=(256,1,1).
//
// y is the token row, and `row_stride` is 0 for the packed single row a decode
// hands over. Both tensors live in the same arena at the same pitch, so one
// number covers the read and the in-place write.
template <typename T>
[[kernel]] void attn_gate(
    device T* attn         [[buffer(0)]],  // [rows, n_q*head_dim] in-place
    const device T* gate   [[buffer(1)]],  // [rows, n_q*head_dim]
    const constant int& row_stride [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const size_t row = row_stride > 0 ? size_t(tid.y) * size_t(row_stride)
                                    : size_t(tid.y) * size_t(grid.x);
  const size_t i = row + size_t(tid.x);
  attn[i] = attn[i] * sigmoid_mlx(gate[i]);
}

#define instantiate_q_gate_split(name, itype)                     \
  template [[host_name("q_gate_split_" #name)]]                   \
  [[kernel]] void q_gate_split<itype>(                            \
      const device itype*, device itype*, device itype*,          \
      const constant int&, const constant int&, const constant int&, \
      uint3, uint3);

#define instantiate_attn_gate(name, itype)                        \
  template [[host_name("gate_" #name)]]                      \
  [[kernel]] void attn_gate<itype>(                               \
      device itype*, const device itype*, const constant int&, uint2, uint2);

instantiate_q_gate_split(bfloat16, bfloat)

instantiate_attn_gate(bfloat16, bfloat)

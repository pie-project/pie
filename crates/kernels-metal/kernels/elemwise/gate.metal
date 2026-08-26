#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void q_gate_split(
    const device T* qg       [[buffer(0)]],
    device T* q_out          [[buffer(1)]],
    device T* gate_out       [[buffer(2)]],
    const constant int& head_dim [[buffer(3)]],
    const constant int& qg_row_stride  [[buffer(4)]],
    const constant int& out_row_stride [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(tid.x);
  const int h = int(tid.y);
  const int hd = head_dim;
  const int n_q = int(grid.y);
  const size_t out_row = out_row_stride > 0 ? size_t(tid.z) * size_t(out_row_stride)
                                            : size_t(tid.z) * n_q * hd;
  const size_t qg_row  = qg_row_stride > 0 ? size_t(tid.z) * size_t(qg_row_stride)
                                           : size_t(tid.z) * n_q * hd * 2;
  q_out[out_row + h * hd + i]    = qg[qg_row + h * 2 * hd + i];
  gate_out[out_row + h * hd + i] = qg[qg_row + h * 2 * hd + hd + i];
}

template <typename T>
inline T sigmoid_mlx(T x) {
  float xf = float(x);
  float y = 1.0f / (1.0f + metal::exp(-metal::fabs(xf)));
  float s = (xf < 0.0f) ? (1.0f - y) : y;
  return T(s);
}

template <typename T>
[[kernel]] void attn_gate(
    device T* attn         [[buffer(0)]],
    const device T* gate   [[buffer(1)]],
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

#include <metal_stdlib>
#include "../prim/register_table.h"
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
inline T sigmoid_stable(T x) {
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
  attn[i] = attn[i] * sigmoid_stable(gate[i]);
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

template <typename T>
[[kernel]] void gate_sigmoid_mul_heads(
    device T* x                   [[buffer(0)]],
    const device T* gate          [[buffer(1)]],
    const constant uint& heads    [[buffer(2)]],
    const constant uint& head_dim [[buffer(3)]],
    const constant float& scale   [[buffer(4)]],
    uint2 tid                     [[thread_position_in_grid]]) {
  const uint width = heads * head_dim;
  const uint col = tid.x;
  const uint row = tid.y;
  if (col >= width) return;
  const float g = float(gate[size_t(row) * size_t(heads) + size_t(col / head_dim)]);
  const float s = scale / (1.0f + precise::exp(-g));
  const size_t at = size_t(row) * size_t(width) + size_t(col);
  x[at] = static_cast<T>(float(x[at]) * s);
}

#define instantiate_gate_sigmoid_mul_heads(name, itype)              \
  template [[host_name("gate_sigmoid_mul_heads_" #name)]]            \
  [[kernel]] void gate_sigmoid_mul_heads<itype>(                     \
      device itype*, const device itype*, const constant uint&,      \
      const constant uint&, const constant float&, uint2);

instantiate_gate_sigmoid_mul_heads(bfloat16, bfloat)


// attn_gate over at most eight rows that also lays the register tile's input
// table of the gated rows, so the out projection runs no table pass: a
// threadgroup per row of the table's eight, the rows past `rows` laid as
// zeros, the first zeroing the split tile's `counters` arrival counters.
[[kernel]] void gate_table_bfloat16(
    device bfloat* attn         [[buffer(0)]],
    const device bfloat* gate   [[buffer(1)]],
    const constant int& width   [[buffer(2)]],
    device bfloat* table        [[buffer(3)]],
    const constant uint& rows   [[buffer(4)]],
    const constant uint& counters [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 threads2 [[threads_per_threadgroup]]) {
  threadgroup bfloat values[8192];
  const uint lid = lid2.x, threads = threads2.x;
  const uint n = uint(width), row = group.y;
  const bool live = row < rows;
  device bfloat* x = attn + size_t(row) * size_t(n);
  const device bfloat* g = gate + size_t(row) * size_t(n);
  for (uint k = lid; k < n; k += threads) {
    bfloat v = bfloat(0.0f);
    if (live) {
      v = x[k] * sigmoid_stable(g[k]);
      x[k] = v;
    }
    table[register_table::slot(k, row)] = v;
    values[k] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup bfloat* laid = values;
  register_table::lay_sums(laid, table, n, row, lid, threads);
  if (row == 0)
    register_table::lay_rest(table, n, 8, counters, lid, threads);
}

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

// One 32-input half (`q` of 64-input group `h`) of row `row` of the register
// tile's input table (`qmm_sg_table` in linear/quant_qmm_sg.metal), from that
// row's inputs at `src` (indexed by input): each input at its lane slot, the
// 32 summed in the order `qmm_sg_table` adds them. A row past the rows the
// matmul reads is laid as zeros (`live` false).
METAL_FUNC void lay_register_half(
    const device bfloat* src, bool live, device bfloat* table, uint width, uint row,
    uint h, uint q) {
  // The sums add in `qmm_sg_table`'s order, not one the compiler picks.
#pragma clang fp reassociate(off)
  const uint c = row >> 1, r = row & 1;
  float part[2][2];
  for (uint cl = 0; cl < 2; ++cl) {
    const uint cp = 2 * q + cl;
    for (uint i = 0; i < 2; ++i) {
      const uint fm = 2 * cp + i;
      float sum = 0.0f;
      for (uint j = 0; j < 2; ++j)
        for (uint p = 0; p < 4; ++p) {
          const uint k = 64 * h + 8 * (2 * cp + j) + p + 4 * i;
          const bfloat v = live ? src[k] : bfloat(0.0f);
          table[(ulong(h) * 32 + fm * 4 + c) * 16 + 8 * j + 2 * p + r] = v;
          sum += float(v);
        }
      part[cl][i] = sum;
    }
  }
  device float* sums = reinterpret_cast<device float*>(table + 8 * ulong(width));
  sums[(2 * ulong(h) + q) * 8 + row] = (part[0][0] + part[0][1]) + (part[1][0] + part[1][1]);
}

// attn_gate over at most eight rows that also lays the register tile's input
// table of the gated rows, so the out projection runs no table pass: a thread
// per 32 inputs of a row (x: 64-input group x / 2, half x % 2; y: the row, of
// the table's eight), rows past `rows` laid as zeros, the split tile's
// `counters` arrival counters zeroed. Bit for bit `qmm_sg_table`'s table.
[[kernel]] void gate_table_bfloat16(
    device bfloat* attn         [[buffer(0)]],
    const device bfloat* gate   [[buffer(1)]],
    const constant int& width   [[buffer(2)]],
    device bfloat* table        [[buffer(3)]],
    const constant uint& rows   [[buffer(4)]],
    const constant uint& counters [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const uint row = tid.y, h = tid.x >> 1, q = tid.x & 1;
  const bool live = row < rows;
  device bfloat* x = attn + size_t(row) * size_t(width);
  const device bfloat* g = gate + size_t(row) * size_t(width);
  if (live)
    for (uint k = 32 * tid.x; k < 32 * tid.x + 32; ++k)
      x[k] = x[k] * sigmoid_mlx(g[k]);
  lay_register_half(x, live, table, uint(width), row, h, q);
  device uint* count = reinterpret_cast<device uint*>(table + 8 * ulong(width)) + width / 4;
  for (uint n = tid.y * grid.x + tid.x; n < counters; n += grid.x * grid.y)
    count[n] = 0;
}

#include <metal_stdlib>

using namespace metal;

// Where an expert's plane of a bank lies: the GPU address of its first
// row (the weight cache's pool slot, or the bank's own copy), written by the
// host into a buffer of these, one an expert.
struct ExpertAt {
  const device char* at;
};

// `pieces > 0` reads output row `r` of expert `e` through
// `expert_at[e * pieces + r / piece_rows]`, row `r % piece_rows` of that
// piece (`bank` is then not read); `pieces == 0` reads the contiguous
// `bank`. `counting != 0` adds one to `hits[e]` for each route that reads
// expert `e`: the weight cache's use record.
[[kernel]] void select_gemv(
    const device bfloat* x    [[buffer(0)]],
    const device bfloat* bank [[buffer(1)]],
    const device int* routes  [[buffer(2)]],
    device bfloat* y          [[buffer(3)]],
    const constant uint& in_width      [[buffer(4)]],
    const constant uint& out_width     [[buffer(5)]],
    const constant uint& slots_per_row [[buffer(6)]],
    const constant uint& x_row_stride  [[buffer(7)]],
    const constant uint& x_slot_stride [[buffer(8)]],
    const device ExpertAt* expert_at   [[buffer(9)]],
    const constant int& pieces         [[buffer(10)]],
    const constant int& piece_rows     [[buffer(11)]],
    device atomic_uint* hits           [[buffer(12)]],
    const constant int& counting       [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint out_row = gid.x >> 5;
  const uint route = gid.y;
  const size_t at = size_t(route) * size_t(out_width) + size_t(out_row);
  const int e = routes[route];
  if (e < 0) {
    if (lane == 0) y[at] = bfloat(0);
    return;
  }

  const device bfloat* w =
      pieces > 0
          ? reinterpret_cast<const device bfloat*>(
                expert_at[uint(e) * uint(pieces) + out_row / uint(piece_rows)].at) +
                size_t(out_row % uint(piece_rows)) * size_t(in_width)
          : bank + (size_t(uint(e)) * size_t(out_width) + size_t(out_row)) * size_t(in_width);
  if (pieces > 0 && counting != 0 && out_row == 0 && lane == 0) {
    atomic_fetch_add_explicit(&hits[e], 1u, memory_order_relaxed);
  }
  const uint k = slots_per_row < 1u ? 1u : slots_per_row;
  const device bfloat* a =
      x + size_t(route / k) * size_t(x_row_stride) + size_t(route % k) * size_t(x_slot_stride);

  float acc = 0.0f;
  for (uint i = lane; i < in_width; i += 32u) {
    acc += float(w[i]) * float(a[i]);
  }
  acc = simd_sum(acc);
  if (lane == 0) y[at] = static_cast<bfloat>(acc);
}

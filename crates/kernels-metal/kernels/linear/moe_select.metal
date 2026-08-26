#include <metal_stdlib>

using namespace metal;

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
      bank + (size_t(uint(e)) * size_t(out_width) + size_t(out_row)) * size_t(in_width);
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

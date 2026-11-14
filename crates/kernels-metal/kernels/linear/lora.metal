#include <metal_stdlib>

using namespace metal;


constant constexpr uint kLoraMaxRank = 128;

// One GPU address an adapter id: where that adapter's plane of a bank lies
// (an `Adapter` slot, or an operator seat of the bank), read through.
struct PlaneAt {
  const device bfloat* at;
};

[[kernel]] void lora_correct(
    const device bfloat* x      [[buffer(0)]],
    const device PlaneAt* table_a [[buffer(1)]],
    const device PlaneAt* table_b [[buffer(2)]],
    const device int* routes    [[buffer(3)]],
    device bfloat* y            [[buffer(4)]],
    const constant uint& in_width  [[buffer(5)]],
    const constant uint& out_width [[buffer(6)]],
    const constant uint& rank      [[buffer(7)]],
    uint3 lid3    [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint3 tgsize  [[threads_per_threadgroup]],
    uint3 tgid    [[threadgroup_position_in_grid]]) {
  const uint row = tgid.y;
  const int adapter = routes[row];
  if (adapter < 0) return;

  const uint r = min(rank, kLoraMaxRank);
  const device bfloat* down = table_a[adapter].at;
  const device bfloat* up = table_b[adapter].at;
  const device bfloat* a = x + size_t(row) * size_t(in_width);
  device bfloat* out = y + size_t(row) * size_t(out_width);

  threadgroup float waist[kLoraMaxRank];

  const uint n_simd = max((tgsize.x + 31u) / 32u, 1u);
  for (uint i = simd_gid; i < r; i += n_simd) {
    const device bfloat* w = down + size_t(i) * size_t(in_width);
    float acc = 0.0f;
    for (uint c = simd_lid; c < in_width; c += 32u) {
      acc += float(w[c]) * float(a[c]);
    }
    acc = simd_sum(acc);
    if (simd_lid == 0) waist[i] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint n = lid3.x; n < out_width; n += tgsize.x) {
    const device bfloat* brow = up + size_t(n) * size_t(r);
    float acc = 0.0f;
    for (uint i = 0; i < r; ++i) {
      acc += float(brow[i]) * waist[i];
    }
    out[n] = static_cast<bfloat>(float(out[n]) + acc);
  }
}

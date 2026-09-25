#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#include "../common/mpp.metal"
template <int D, bool DIRECT>
kernel void
q8_mpp(const device bfloat *q [[buffer(0)]], const device int8_t *keys [[buffer(1)]],
       const device int8_t *values [[buffer(2)]], device bfloat *out [[buffer(3)]],
       constant int &gqa [[buffer(4)]], const device int *positions [[buffer(5)]],
       const device int *owners [[buffer(6)]], const device uint *pages [[buffer(7)]],
       const device uint *ptr [[buffer(8)]], constant int &page_size [[buffer(9)]],
       constant int &kv_heads [[buffer(10)]], constant float &scale [[buffer(11)]],
       const device uchar *mask [[buffer(12)]], constant uint &mask_stride [[buffer(13)]],
       const device uchar *enabled [[buffer(14)]], constant int &window [[buffer(15)]],
       const device bfloat *sinks [[buffer(16)]], constant int &rows [[buffer(17)]],
       uint2 tile [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
       uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
       uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int M = 32, N = 32, SG = 8;
  const int head = tile.x, heads = grid.x, kv = head / gqa, row0 = tile.y * M;
  threadgroup bfloat qs[DIRECT ? 1 : M * D];
  threadgroup float scores[M * N], maxima[M], totals[M], factors[M];
  threadgroup bfloat probs[M * N];
  if constexpr (!DIRECT)
    for (uint i = tid; i < M * D; i += 256) {
      const uint r = i / D, d = i % D;
      qs[i] = row0 + r < rows ? q[(ulong(row0 + r) * heads + head) * D + d] : bfloat(0);
    }
  for (uint r = tid; r < M; r += 256) {
    maxima[r] = -1e30f;
    totals[r] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto a = [&]() {
    if constexpr (DIRECT) {
      auto view = tensor(const_cast<device bfloat *>(q) + head * D, dextents<int, 2>{D, rows},
                         array<int, 2>{1, heads * D});
      return view.template slice<D, M>(0, row0);
    } else {
      return tensor(qs, extents<int, D, M>{}, array<int, 2>{1, D});
    }
  }();
  auto score_view = tensor(scores, extents<int, N, M>{}, array<int, 2>{1, N});
  auto p = tensor(probs, extents<int, N, M>{}, array<int, 2>{1, N});
  auto v0 = tensor(const_cast<device int8_t *>(values), dextents<int, 2>{D, N},
                   array<int, 2>{1, kv_heads * 2 * D});
  constexpr auto qkd = matmul2d_descriptor(M, N, D, false, true, false);
  constexpr auto pvd = matmul2d_descriptor(M, D, N, false, false, false);
  matmul2d<qkd, execution_simdgroups<SG>> qkop;
  matmul2d<pvd, execution_simdgroups<SG>> pvop;
  auto acc = pvop.template get_destination_cooperative_tensor<decltype(p), decltype(v0), float>();
  const ulong valid = mpp_valid_mask(acc);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    acc[i] = 0;
  int sub = 0;
  while (sub < M && row0 + sub < rows) {
    const int req = owners[row0 + sub];
    int end = sub + 1;
    while (end < M && row0 + end < rows && owners[row0 + end] == req)
      ++end;
    int last = 0, first = 0x7fffffff;
    for (int r = sub; r < end; ++r) {
      const int pos = positions[row0 + r];
      last = max(last, enabled[row0 + r] == 2 ? int(ptr[req + 1] - ptr[req]) * N - 1 : pos);
      first = min(first, window > 0 ? max(0, pos - window + 1) : 0);
    }
    for (int base = (first / N) * N; base <= last; base += N) {
      const uint physical = pages[ptr[req] + base / N];
      const ulong offset = (ulong(physical) * N * kv_heads + kv) * 2 * D;
      auto b = tensor(const_cast<device int8_t *>(keys) + offset, dextents<int, 2>{D, N},
                      array<int, 2>{1, kv_heads * 2 * D});
      auto score =
          qkop.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
      qkop.run(a, b, score);
      score.store(score_view);
      auto v = tensor(const_cast<device int8_t *>(values) + offset, dextents<int, 2>{D, N},
                      array<int, 2>{1, kv_heads * 2 * D});
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = sg; r < M; r += SG) {
        const int row = row0 + r, pos = row < rows ? positions[row] : 0, kpos = base + lane;
        bool keep = int(r) >= sub && int(r) < end && row < rows && owners[row] == req &&
                    (enabled[row] == 2 || kpos <= pos) && (window <= 0 || kpos >= pos - window + 1);
        if (keep && enabled[row])
          keep = uint(kpos) < mask_stride && mask[ulong(row) * mask_stride + kpos] != 0;
        const ulong cell = offset + ulong(lane) * kv_heads * 2 * D;
        const float ks = *reinterpret_cast<const device float *>(keys + cell + D);
        const float vs = *reinterpret_cast<const device float *>(values + cell + D);
        const float s = keep ? scores[r * N + lane] * scale * ks : -1e30f;
        const float newmax = max(maxima[r], simd_max(s));
        const float factor = totals[r] > 0 ? fast::exp(maxima[r] - newmax) : 0;
        const float e = keep ? fast::exp(s - newmax) : 0;
        const float total = simd_sum(e);
        probs[r * N + lane] = bfloat(e * vs);
        if (lane == 0) {
          maxima[r] = newmax;
          totals[r] = totals[r] * factor + total;
          factors[r] = factor;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial =
          pvop.template get_destination_cooperative_tensor<decltype(p), decltype(v), float>();
      pvop.run(p, v, partial);
      mpp_for_each(acc, valid, [&](ushort i) __attribute__((always_inline)) {
        auto ix = acc.get_multidimensional_index(i);
        acc[i] = acc[i] * factors[ix[1]] + partial[i];
      });
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sub = end;
  }
  mpp_for_each(acc, valid, [&](ushort i) __attribute__((always_inline)) {
    auto ix = acc.get_multidimensional_index(i);
    if (row0 + ix[1] < rows)
      out[(ulong(row0 + ix[1]) * heads + head) * D + ix[0]] =
          bfloat(totals[ix[1]] > 0 ? acc[i] / totals[ix[1]] : 0);
  });
}

#define Q8_MPP(DIM, DIRECT, NAME)                                                                  \
  template [[host_name(NAME)]] kernel void q8_mpp<DIM, DIRECT>(                                    \
      const device bfloat *, const device int8_t *, const device int8_t *, device bfloat *,        \
      constant int &, const device int *, const device int *, const device uint *,                 \
      const device uint *, constant int &, constant int &, constant float &, const device uchar *, \
      constant uint &, const device uchar *, constant int &, const device bfloat *,                \
      constant int &, uint2, uint2, uint, uint, uint);
Q8_MPP(128, false, "q8_mpp_d128")
Q8_MPP(256, false, "q8_mpp_d256")
Q8_MPP(256, true, "q8_visit_direct")

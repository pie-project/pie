#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#include "../prim/mpp.h"
#include "../prim/kv_page.h"
// KV is int8_t over a q8 pool, whose cells carry their own scale, and bfloat over a pool
// stored as it was written. P is the pool's tokens a page: a 32-key tile is one page of 32,
// or two pages of 16 gathered by their own addresses, each half scored and summed into the
// tile's output by its own matmuls (the second accumulating onto the first), never copied.
template <int D, bool DIRECT, typename KV, bool LSE = false, int P = 32>
kernel void
q8_mpp(const device bfloat *q [[buffer(0)]], const device KV *keys [[buffer(1)]],
       const device KV *values [[buffer(2)]], device bfloat *out [[buffer(3)]],
       constant int &gqa [[buffer(4)]], const device int *positions [[buffer(5)]],
       const device int *owners [[buffer(6)]], const device uint *pages [[buffer(7)]],
       const device uint *ptr [[buffer(8)]], constant int &page_size [[buffer(9)]],
       constant int &kv_heads [[buffer(10)]], constant float &scale [[buffer(11)]],
       const device uchar *mask [[buffer(12)]], constant uint &mask_stride [[buffer(13)]],
       const device uint *enabled [[buffer(14)]], constant int &window [[buffer(15)]],
       const device bfloat *sinks [[buffer(16)]], constant int &rows [[buffer(17)]],
       device float *lse [[buffer(18)]], const constant ulong &k_page_stride [[buffer(19)]],
       const constant ulong &v_page_stride [[buffer(20)]],
       const constant ulong &k_offset [[buffer(21)]], const constant ulong &v_offset [[buffer(22)]],
       uint2 tile [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
       uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
       uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int M = 32, N = 32, SG = 8;
  constexpr int T = N / P;
  static_assert(P * T == N && (T == 1 || T == 2), "a tile is one page of 32 or two of 16");
  constexpr bool Q8 = sizeof(KV) == 1;
  constexpr int W = Q8 ? 2 * D : D;
  const device KV *kplane = pie_kv_plane(keys, k_offset);
  const device KV *vplane = pie_kv_plane(values, v_offset);
  const int head = tile.x, heads = grid.x, kv = head / gqa, row0 = tile.y * M;
  threadgroup bfloat qs[DIRECT ? 1 : M * D];
  threadgroup float scores[M * N], maxima[M], totals[M], factors[M];
  // A 64-wide head keeps its probabilities in half, as the simdgroup kernel it replaces does.
  using Prob = metal::conditional_t<D == 64, half, bfloat>;
  threadgroup Prob probs[M * N];
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
  auto v0 = tensor(const_cast<device KV *>(values), dextents<int, 2>{D, N},
                   array<int, 2>{1, kv_heads * W});
  constexpr auto qkd = matmul2d_descriptor(M, N, D, false, true, false);
  constexpr auto pvd = matmul2d_descriptor(M, D, N, false, false, false);
  matmul2d<qkd, execution_simdgroups<SG>> qkop;
  matmul2d<pvd, execution_simdgroups<SG>> pvop;
  // A page's half of the tile: its scores, and its share of the output, the second half
  // accumulating onto the first.
  constexpr auto qkh = matmul2d_descriptor(M, P, D, false, true, false);
  constexpr auto pvh = matmul2d_descriptor(M, D, P, false, false, false);
  constexpr auto pvh_acc = matmul2d_descriptor(M, D, P, false, false, false,
                                               matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qkh, execution_simdgroups<SG>> qkhop;
  matmul2d<pvh, execution_simdgroups<SG>> pvhop;
  matmul2d<pvh_acc, execution_simdgroups<SG>> pvhaccop;
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
      last = max(last, (enabled[row0 + r] & 3u) == 2 ? int(ptr[req + 1] - ptr[req]) * P - 1 : pos);
      first = min(first, window > 0 ? max(0, pos - window + 1) : 0);
    }
    for (int base = (first / N) * N; base <= last; base += N) {
      // The tile's pages: a half wholly past the last key read takes the first page's
      // address, so no lane reads a page the request does not hold (its keys are masked).
      ulong k_at[T], v_at[T];
#pragma unroll
      for (int t = 0; t < T; ++t) {
        const bool held = t == 0 || base + t * P <= last;
        const uint physical = pages[ptr[req] + base / P + (held ? t : 0)];
        k_at[t] = pie_page_at(physical, k_page_stride / sizeof(KV), sizeof(KV)) + ulong(kv) * W;
        v_at[t] = pie_page_at(physical, v_page_stride / sizeof(KV), sizeof(KV)) + ulong(kv) * W;
      }
      if constexpr (T == 1) {
        auto b = tensor(const_cast<device KV *>(kplane) + k_at[0], dextents<int, 2>{D, N},
                        array<int, 2>{1, kv_heads * W});
        auto score =
            qkop.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
        qkop.run(a, b, score);
        score.store(score_view);
      } else {
#pragma unroll
        for (int t = 0; t < T; ++t) {
          auto b = tensor(const_cast<device KV *>(kplane) + k_at[t], dextents<int, 2>{D, P},
                          array<int, 2>{1, kv_heads * W});
          auto half_view = tensor(scores + t * P, extents<int, P, M>{}, array<int, 2>{1, N});
          auto score =
              qkhop.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
          qkhop.run(a, b, score);
          score.store(half_view);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint r = sg; r < M; r += SG) {
        const int row = row0 + r, pos = row < rows ? positions[row] : 0, kpos = base + lane;
        bool keep = int(r) >= sub && int(r) < end && row < rows && owners[row] == req &&
                    ((enabled[row] & 3u) == 2 || kpos <= pos) && (window <= 0 || kpos >= pos - window + 1);
        if (keep && (enabled[row] & 3u))
          keep = pie_mask_admits(mask, mask_stride, enabled[row] >> 2, uint(row), uint(kpos));
        const ulong cell = ulong(lane % P) * kv_heads * W;
        float ks = 1, vs = 1;
        if constexpr (Q8) {
          ks = *reinterpret_cast<const device float *>(kplane + k_at[lane / P] + cell + D);
          vs = *reinterpret_cast<const device float *>(vplane + v_at[lane / P] + cell + D);
        }
        const float s = keep ? scores[r * N + lane] * scale * ks : -1e30f;
        const float newmax = max(maxima[r], simd_max(s));
        const float factor = totals[r] > 0 ? fast::exp(maxima[r] - newmax) : 0;
        const float e = keep ? fast::exp(s - newmax) : 0;
        const float total = simd_sum(e);
        probs[r * N + lane] = Prob(keep ? e * vs : 0.0f);
        if (lane == 0) {
          maxima[r] = newmax;
          totals[r] = totals[r] * factor + total;
          factors[r] = factor;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // The tile's keys a row may read end at `last`: the V cells past it were never
      // written, and a zero probability times what they hold is not zero when that is a
      // NaN, so the matmul's key extent stops there (cells past an extent read as zero).
      // A wide row (`enabled & 3 == 2`) sets `last` to its pages' end and keeps its
      // unwritten tail out by its mask bytes alone, which this extent does not cover. This
      // kernel runs only causal (`attn::q8::attention` and `mpp_fits` both require it), and
      // a served model stages wide rows only for a class that attends both ways
      // (`serve::load`), so none reaches it there; a caller that passes a wide row to a
      // causal op over pages with unwritten tails would read them.
      const int held = min(N, last + 1 - base);
      auto v = tensor(const_cast<device KV *>(vplane) + v_at[0], dextents<int, 2>{D, held},
                      array<int, 2>{1, kv_heads * W});
      auto partial =
          pvop.template get_destination_cooperative_tensor<decltype(p), decltype(v), float>();
      if constexpr (T == 1) {
        pvop.run(p, v, partial);
      } else {
        auto p0 = tensor(probs, extents<int, P, M>{}, array<int, 2>{1, N});
        auto p1 = tensor(probs + P, extents<int, P, M>{}, array<int, 2>{1, N});
        auto v0h = tensor(const_cast<device KV *>(vplane) + v_at[0],
                          dextents<int, 2>{D, min(P, held)}, array<int, 2>{1, kv_heads * W});
        auto v1h = tensor(const_cast<device KV *>(vplane) + v_at[1],
                          dextents<int, 2>{D, clamp(held - P, 1, P)}, array<int, 2>{1, kv_heads * W});
        pvhop.run(p0, v0h, partial);
        pvhaccop.run(p1, v1h, partial);
      }
      mpp_for_each(acc, valid, [&](ushort i) __attribute__((always_inline)) {
        auto ix = acc.get_multidimensional_index(i);
        acc[i] = acc[i] * factors[ix[1]] + partial[i];
      });
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sub = end;
  }
  // The log-sum-exp in base 2 of each row's scaled scores, as the simdgroup kernel publishes it.
  if constexpr (LSE)
    for (uint r = tid; r < M; r += 256)
      if (row0 + int(r) < rows)
        lse[ulong(row0 + r) * heads + head] =
            totals[r] > 0 ? maxima[r] * 1.44269504088896340736f + log2(totals[r]) : -INFINITY;
  mpp_for_each(acc, valid, [&](ushort i) __attribute__((always_inline)) {
    auto ix = acc.get_multidimensional_index(i);
    if (row0 + ix[1] < rows)
      out[(ulong(row0 + ix[1]) * heads + head) * D + ix[0]] =
          bfloat(totals[ix[1]] > 0 ? acc[i] / totals[ix[1]] : 0);
  });
}

#define Q8_MPP_PAGE(DIM, DIRECT, KV, LSE, PAGE, NAME)                                              \
  template [[host_name(NAME)]] kernel void q8_mpp<DIM, DIRECT, KV, LSE, PAGE>(                     \
      const device bfloat *, const device KV *, const device KV *, device bfloat *,                \
      constant int &, const device int *, const device int *, const device uint *,                 \
      const device uint *, constant int &, constant int &, constant float &, const device uchar *, \
      constant uint &, const device uint *, constant int &, const device bfloat *,                 \
      constant int &, device float *, PIE_KV_LAYOUT_TYPES uint2, uint2, uint, uint, uint);
#define Q8_MPP_LSE(DIM, DIRECT, KV, LSE, NAME) Q8_MPP_PAGE(DIM, DIRECT, KV, LSE, 32, NAME)
#define Q8_MPP(DIM, DIRECT, KV, NAME) Q8_MPP_LSE(DIM, DIRECT, KV, false, NAME)
Q8_MPP(128, false, int8_t, "q8_mpp_d128")
Q8_MPP(256, false, int8_t, "q8_mpp_d256")
Q8_MPP(256, true, int8_t, "q8_visit_direct")
Q8_MPP(128, false, bfloat, "sdpa_mpp_d128")
Q8_MPP(256, false, bfloat, "sdpa_mpp_d256")
Q8_MPP(256, true, bfloat, "sdpa_visit_direct")
Q8_MPP(64, false, bfloat, "sdpa_mpp_d64")
Q8_MPP_LSE(64, false, bfloat, true, "sdpa_mpp_lse_d64")
// Over pools of 16-token pages, as a served model's chunk plan builds them.
Q8_MPP_PAGE(128, false, bfloat, false, 16, "sdpa_mpp_d128_p16")
Q8_MPP_PAGE(256, false, bfloat, false, 16, "sdpa_mpp_d256_p16")
Q8_MPP_PAGE(256, true, bfloat, false, 16, "sdpa_visit_direct_p16")
Q8_MPP_PAGE(64, false, bfloat, false, 16, "sdpa_mpp_d64_p16")
Q8_MPP_PAGE(64, false, bfloat, true, 16, "sdpa_mpp_lse_d64_p16")

#include <metal_stdlib>
using namespace metal;
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
template <int GH>
[[kernel]] void pie_q8_batch_split(
    const device bfloat *queries [[buffer(0)]], const device int8_t *k_pages [[buffer(1)]],
    const device int8_t *v_pages [[buffer(2)]], device bfloat *out [[buffer(3)]],
    const constant int &gqa_factor [[buffer(4)]], const device int *position_ids [[buffer(5)]],
    const device int *req_of_token [[buffer(6)]], const device uint *kv_page_indices [[buffer(7)]],
    const device uint *kv_page_indptr [[buffer(8)]], const constant int &page_size [[buffer(9)]],
    const constant int &n_kv_heads [[buffer(10)]], const constant float &scale [[buffer(11)]],
    const device uchar *attention_mask [[buffer(12)]],
    const device uint &attention_mask_stride [[buffer(13)]],
    const device uchar *attention_mask_enabled [[buffer(14)]],
    const constant int &window [[buffer(15)]], const device bfloat *sinks [[buffer(16)]],
    device float *workspace [[buffer(17)]], constant uint &splits [[buffer(18)]],
    constant uint &qrows [[buffer(19)]], constant uint &tile_base [[buffer(20)]],
    uint3 tile [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int SG = 8;
  constexpr int M = 8 * GH, N = 32, D = 256;
  const uint heads = GH * n_kv_heads, packed_rows = 8 * heads;
  const uint query_words = packed_rows * D / 2;
  workspace += tile.z * (query_words + splits * packed_rows * (D + 2));
  const uint tid = simd_gid * 32 + simd_lid, nt = SG * 32;
  const int row0 = (tile_base + tile.z) * 8, kv = tile.x;
  bool simple = row0 + 8 <= qrows && window == 0;
  if (simple) {
    const int req0 = req_of_token[row0], pos0 = position_ids[row0];
    for (int r = 0; r < 8; ++r)
      simple &= req_of_token[row0 + r] == req0 && position_ids[row0 + r] == pos0 + r &&
                attention_mask_enabled[row0 + r] == 0;
  }
  const device bfloat *packed_q =
      reinterpret_cast<const device bfloat *>(workspace) + tile.x * M * D;
  alignas(16) threadgroup float scores[M * N], maxima[M], totals[M], factors[M];
  alignas(16) threadgroup bfloat probs[M * N];
  threadgroup float kscale[N], vscale[N];
  using Prob = bfloat;
  for (uint r = tid; r < M; r += nt) {
    maxima[r] = -1e30f;
    totals[r] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto a = tensor(const_cast<device bfloat *>(packed_q), extents<int, D, M>{}, array<int, 2>{1, D});
  auto sv = tensor(scores, extents<int, N, M>{}, array<int, 2>{1, N});
  auto p = tensor(probs, extents<int, N, M>{}, array<int, 2>{1, N});
  auto v0 = tensor(const_cast<device int8_t *>(v_pages), dextents<int, 2>{D, N},
                   array<int, 2>{1, n_kv_heads * 2 * D});
  constexpr auto qkd = matmul2d_descriptor(M, N, D, false, true, false);
  constexpr auto pvd = matmul2d_descriptor(M, D, N, false, false, false,
                                           matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qkd, execution_simdgroups<SG>> qkop;
  matmul2d<pvd, execution_simdgroups<SG>> pvop;
  auto acc = pvop.template get_destination_cooperative_tensor<decltype(p), decltype(v0), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0;
  for (uint query = 0; query < (simple ? 1 : 8) && row0 + query < qrows; ++query) {
    const int req = req_of_token[row0 + query];
    bool seen = false;
    for (uint prev = 0; prev < query; ++prev)
      seen |= req_of_token[row0 + prev] == req;
    if (seen)
      continue;
    int last = 0, first = 0x7fffffff;
    for (int r = 0; r < 8 && row0 + r < int(qrows); ++r) {
      if (req_of_token[row0 + r] != req)
        continue;
      int pos = position_ids[row0 + r];
      const bool wide = window < 0 || attention_mask_enabled[row0 + r] == 2;
      const int extent = window < 0 ? -window - 1 : window;
      last = max(last, wide ? int(kv_page_indptr[req + 1] - kv_page_indptr[req]) * N - 1 : pos);
      first = min(first, extent > 0 ? max(0, pos - extent + 1) : 0);
    }
    const int firstpage = first / N, npages = last / N - firstpage + 1;
    for (int step = 0; step < (npages + int(splits) - 1) / int(splits); ++step) {
      const int page = firstpage + step * int(splits) + int(tile.y);
      const bool active = page <= last / N;

      if (!active)
        break;

      const uint physical = kv_page_indices[kv_page_indptr[req] + page];
      const ulong off = (ulong(physical) * N * n_kv_heads + kv) * 2 * D;
      auto k = tensor(const_cast<device int8_t *>(k_pages) + off, dextents<int, 2>{D, N},
                      array<int, 2>{1, n_kv_heads * 2 * D});
      auto score =
          qkop.template get_destination_cooperative_tensor<decltype(a), decltype(k), float>();
      qkop.run(a, k, score);
      score.store(sv);
      if (tid < N) {
        kscale[tid] = *reinterpret_cast<const device float *>(k_pages + off +
                                                              ulong(tid) * n_kv_heads * 2 * D + D);
        vscale[tid] = *reinterpret_cast<const device float *>(v_pages + off +
                                                              ulong(tid) * n_kv_heads * 2 * D + D);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint r = tid / 4; r < M; r += SG * 8) {
        const uint col = (tid % 4) * 8;
        const int row = row0 + r / GH, pos = row < int(qrows) ? position_ids[row] : -1, ri = r;
        float svv[8], ex[8];
        bool keep[8];
        float mx = -1e30f;
        for (uint j = 0; j < 8; j++) {
          const int kp = page * N + col + j;
          if (simple) {
            keep[j] = active && kp <= pos;
          } else {
            const bool wide = window < 0 || (row < int(qrows) && attention_mask_enabled[row] == 2);
            const int extent = window < 0 ? -window - 1 : window;
            keep[j] = active && row < int(qrows) && req_of_token[row] == req &&
                      (wide || kp <= pos) && (extent <= 0 || kp >= pos - extent + 1);
            if (keep[j] && attention_mask_enabled[row])
              keep[j] = uint(kp) < attention_mask_stride &&
                        attention_mask[ulong(row) * attention_mask_stride + kp] != 0;
          }
          const float ks = kscale[col + j];
          svv[j] = keep[j] ? scores[r * N + col + j] * scale * ks : -1e30f;
          mx = max(mx, svv[j]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        mx = max(mx, maxima[ri]);
        float f = totals[ri] > 0 ? fast::exp(maxima[ri] - mx) : 0, total = 0;
        for (uint j = 0; j < 8; j++) {
          ex[j] = keep[j] ? fast::exp(svv[j] - mx) : 0;
          total += ex[j];
        }
        total += simd_shuffle_xor(total, 1);
        total += simd_shuffle_xor(total, 2);
        for (uint j = 0; j < 8; j++) {
          const float vs = vscale[col + j];
          probs[r * N + col + j] = Prob(ex[j] * vs);
        }
        if (col == 0) {
          maxima[ri] = mx;
          totals[ri] = totals[ri] * f + total;
          factors[ri] = f;
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto v = tensor(const_cast<device int8_t *>(v_pages) + off, dextents<int, 2>{D, N},
                      array<int, 2>{1, n_kv_heads * 2 * D});

      for (ushort i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) {
          auto ix = acc.get_multidimensional_index(i);
          acc[i] *= factors[ix[1]];
        }
      pvop.run(p, v, acc);

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  device float *results = workspace + query_words + (tile.y * packed_rows + tile.x * M) * D;
  auto rv = tensor(results, extents<int, D, M>{}, array<int, 2>{1, D});
  acc.store(rv);
  device float *stats = workspace + query_words + splits * packed_rows * D + (tile.y * packed_rows + tile.x * M) * 2;
  for (uint r = tid; r < M; r += nt) {
    stats[r * 2] = maxima[r];
    stats[r * 2 + 1] = totals[r];
  }
}

template <int GH>
kernel void
pie_q8_batch_pack(const device bfloat *q [[buffer(0)]], device float *workspace [[buffer(1)]],
                  constant uint &qrows [[buffer(2)]], constant uint &splits [[buffer(3)]],
                  constant uint &tile_base [[buffer(4)]], constant uint &heads [[buffer(5)]],
                  uint i [[thread_position_in_grid]]) {
  const uint packed_rows = 8 * heads, elements = packed_rows * 256;
  uint tile = i / elements, local = i % elements, d = local % 256, h = (local / 256) % heads,
       r = local / (heads * 256);
  uint row = (tile_base + tile) * 8 + r;
  device bfloat *dst = reinterpret_cast<device bfloat *>(
      workspace + tile * (elements / 2 + splits * packed_rows * 258));
  dst[((h / GH) * (8 * GH) + r * GH + h % GH) * 256 + d] =
      row < qrows ? q[(row * heads + h) * 256 + d] : bfloat(0);
}
template <int GH>
kernel void pie_q8_batch_reduce(const device float *workspace [[buffer(0)]],
                                device bfloat *out [[buffer(1)]],
                                constant uint &splits [[buffer(2)]],
                                constant uint &qrows [[buffer(3)]],
                                constant uint &tile_base [[buffer(4)]],
                                constant uint &heads [[buffer(5)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint d [[thread_index_in_threadgroup]]) {
  const uint packed_rows = 8 * heads, query_words = packed_rows * 128;
  uint tile = group / packed_rows, r = group % packed_rows, head = (r / (8 * GH)) * GH + r % GH,
       row = (tile_base + tile) * 8 + (r % (8 * GH)) / GH;
  if (row >= qrows)
    return;
  workspace += tile * (query_words + splits * packed_rows * 258);
  threadgroup float weights[32];
  const device float *results = workspace + query_words;
  const device float *stats = results + splits * packed_rows * 256;
  uint lane = d % 32;
  float mx = simd_max(lane < splits ? stats[(lane * packed_rows + r) * 2] : -1e30f);
  if (d < splits)
    weights[d] = fast::exp(stats[(d * packed_rows + r) * 2] - mx);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0, den = 0;
  for (uint s = 0; s < splits; ++s) {
    float f = weights[s];
    sum += results[(s * packed_rows + r) * 256 + d] * f;
    den += stats[(s * packed_rows + r) * 2 + 1] * f;
  }
  out[(row * heads + head) * 256 + d] = bfloat(den > 0 ? sum / den : 0);
}
#define PIE_Q8_BATCH(gh) \
  template [[host_name("pie_q8_batch_split_g" #gh)]] [[kernel]] void pie_q8_batch_split<gh>( \
      const device bfloat *, const device int8_t *, const device int8_t *, device bfloat *, \
      const constant int &, const device int *, const device int *, const device uint *, \
      const device uint *, const constant int &, const constant int &, const constant float &, \
      const device uchar *, const device uint &, const device uchar *, const constant int &, \
      const device bfloat *, device float *, constant uint &, constant uint &, constant uint &, \
      uint3, uint3, uint, uint); \
  template [[host_name("pie_q8_batch_pack_g" #gh)]] [[kernel]] void pie_q8_batch_pack<gh>( \
      const device bfloat *, device float *, constant uint &, constant uint &, constant uint &, \
      constant uint &, uint); \
  template [[host_name("pie_q8_batch_reduce_g" #gh)]] [[kernel]] void pie_q8_batch_reduce<gh>( \
      const device float *, device bfloat *, constant uint &, constant uint &, constant uint &, \
      constant uint &, uint, uint);

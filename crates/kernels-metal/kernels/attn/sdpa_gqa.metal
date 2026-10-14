#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

// The paged decode read of one row for one kv head's whole group of query heads at once, as
// Splash's verify tiles read a page for all the query heads it serves: the group's (up to eight)
// heads are the rows of 8x8 simdgroup MMAs, so each key and value is read once per row and kv
// head instead of once per query head. Four simdgroups split the row's keys eight at a time and
// fold their online softmaxes in threadgroup memory. Same attention as `sdpa_paged_decode`:
// the row's request, causal end and window, its mask where it states one; bf16 pages.

namespace sdpa_gqa {

template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

constant constexpr int SG = 4;

} // namespace sdpa_gqa

template <int D>
[[kernel]] [[max_total_threads_per_threadgroup(128)]] void sdpa_paged_decode_gqa(
    const device bfloat *queries [[buffer(0)]], const device bfloat *k_pages [[buffer(1)]],
    const device bfloat *v_pages [[buffer(2)]], device bfloat *out [[buffer(3)]],
    const constant int &gqa [[buffer(4)]], const device int *position_ids [[buffer(5)]],
    const device int *req_of_token [[buffer(6)]], const device uint *kv_page_indices [[buffer(7)]],
    const device uint *kv_page_indptr [[buffer(8)]], const constant int &page_size [[buffer(9)]],
    const constant int &n_kv_heads [[buffer(10)]], const constant float &scale [[buffer(11)]],
    const device uchar *attention_mask [[buffer(12)]], const device uint &attention_mask_stride [[buffer(13)]],
    const device uchar *attention_mask_enabled [[buffer(14)]], const constant int &window [[buffer(15)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using namespace sdpa_gqa;
  constexpr int C = D / 8; // 8-wide chunks of a head
  threadgroup float folded[2 * 8 * D];
  threadgroup float maxes[SG][8], sums[SG][8];

  const int head = int(group.x), row = int(group.y);
  const int n_q = n_kv_heads * gqa;
  const int r = req_of_token[row];
  const int q_pos = position_ids[row];
  const int mask_word = int(attention_mask_enabled[row]);
  const bool wide = window < 0 || mask_word == 2;
  const int extent = window < 0 ? (-window - 1) : window;
  const int kv_start = (extent > 0 && q_pos >= extent) ? (q_pos - extent + 1) : 0;
  const int page_base = int(kv_page_indptr[r]);
  const int last_kp = wide ? int(kv_page_indptr[r + 1] - uint(page_base)) * page_size - 1 : q_pos;
  const bool masked = mask_word != 0;

  // Lane (fm, fn): element (fm, fn) and (fm, fn + 1) of each 8x8 fragment.
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const bool live_head = int(fm) < gqa;

  // The group's queries, scaled: rows are heads, columns the head's dimensions.
  simdgroup_matrix<bfloat, 8, 8> q[C];
  {
    const device bfloat *qrow = queries + (ulong(row) * n_q + ulong(head) * gqa + fm) * D;
#pragma unroll
    for (int i = 0; i < C; ++i)
      te(q[i]) = live_head ? vec<bfloat, 2>(bfloat(scale * float(qrow[8 * i + fn])),
                                            bfloat(scale * float(qrow[8 * i + fn + 1])))
                           : vec<bfloat, 2>(bfloat(0.0f), bfloat(0.0f));
  }
  simdgroup_matrix<float, 8, 8> o[C];
#pragma unroll
  for (int i = 0; i < C; ++i)
    o[i] = simdgroup_matrix<float, 8, 8>(0.0f);
  float m = -INFINITY, l = 0.0f;
  const ulong key_stride = ulong(n_kv_heads) * D;

  for (int k0 = (kv_start & ~7) + 8 * int(simd); k0 <= last_kp; k0 += 8 * SG) {
    // Eight keys of one page (page_size is a multiple of 8).
    const ulong slot0 = ulong(kv_page_indices[page_base + k0 / page_size]) * page_size + k0 % page_size;
    const device bfloat *kb = k_pages + slot0 * key_stride + ulong(head) * D;
    const device bfloat *vb = v_pages + slot0 * key_stride + ulong(head) * D;
    simdgroup_matrix<float, 8, 8> s(0.0f);
#pragma unroll
    for (int i = 0; i < C; ++i) {
      simdgroup_matrix<bfloat, 8, 8> kt;
      simdgroup_load(kt, kb + 8 * i, key_stride, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s, q[i], kt, s);
    }
    float2 sc = float2(te(s));
    float row_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int kp = k0 + int(fn) + j;
      bool ok = kp >= kv_start && kp <= last_kp;
      if (ok && masked)
        ok = uint(kp) < attention_mask_stride &&
             attention_mask[ulong(row) * attention_mask_stride + uint(kp)] != 0;
      sc[j] = ok ? sc[j] : -INFINITY;
      row_max = max(row_max, sc[j]);
    }
    row_max = max(row_max, simd_shuffle_xor(row_max, 1u));
    row_max = max(row_max, simd_shuffle_xor(row_max, 8u));
    const float m_new = max(m, row_max);
    const float alpha = m_new == -INFINITY ? 1.0f : fast::exp(m - m_new);
    const float2 p = m_new == -INFINITY ? float2(0.0f) : float2(fast::exp(sc[0] - m_new), fast::exp(sc[1] - m_new));
    float row_sum = p.x + p.y;
    row_sum += simd_shuffle_xor(row_sum, 1u);
    row_sum += simd_shuffle_xor(row_sum, 8u);
    l = l * alpha + row_sum;
    m = m_new;
    simdgroup_matrix<bfloat, 8, 8> pm;
    te(pm) = vec<bfloat, 2>(bfloat(p.x), bfloat(p.y));
    // A block past the row's keys (or under its mask) reads its value rows as zero, so a slot
    // the page never held cannot reach the output through a zero probability.
    const bool whole = !masked && k0 >= kv_start && k0 + 7 <= last_kp;
    const int vk = k0 + int(fm);
    bool vlive = vk >= kv_start && vk <= last_kp;
    if (vlive && masked)
      vlive = uint(vk) < attention_mask_stride &&
              attention_mask[ulong(row) * attention_mask_stride + uint(vk)] != 0;
#pragma unroll
    for (int i = 0; i < C; ++i) {
      te(o[i]) *= alpha;
      simdgroup_matrix<bfloat, 8, 8> v;
      if (whole) {
        simdgroup_load(v, vb + 8 * i, key_stride);
      } else {
        const device bfloat *vr = vb + ulong(fm) * key_stride + 8 * i + fn;
        te(v) = vlive ? vec<bfloat, 2>(vr[0], vr[1]) : vec<bfloat, 2>(bfloat(0.0f), bfloat(0.0f));
      }
      simdgroup_multiply_accumulate(o[i], pm, v, o[i]);
    }
  }

  // Fold the simdgroups: each scales its rows to the group's maximum, then pairs add.
  if ((lane & 9) == 0) {
    maxes[simd][fm] = m;
    sums[simd][fm] = l;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float top = -INFINITY;
#pragma unroll
  for (int s = 0; s < SG; ++s)
    top = max(top, maxes[s][fm]);
  float total = 0.0f;
#pragma unroll
  for (int s = 0; s < SG; ++s)
    total += maxes[s][fm] == -INFINITY ? 0.0f : sums[s][fm] * fast::exp(maxes[s][fm] - top);
  const float mine = m == -INFINITY ? 0.0f : fast::exp(m - top);
#pragma unroll
  for (int i = 0; i < C; ++i)
    te(o[i]) *= mine;
  if (simd >= 2) {
#pragma unroll
    for (int i = 0; i < C; ++i)
      simdgroup_store(o[i], folded + (simd - 2) * 8 * D + 8 * i, D);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd < 2) {
#pragma unroll
    for (int i = 0; i < C; ++i) {
      simdgroup_matrix<float, 8, 8> other;
      simdgroup_load(other, folded + simd * 8 * D + 8 * i, D);
      te(o[i]) += te(other);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 1) {
#pragma unroll
    for (int i = 0; i < C; ++i)
      simdgroup_store(o[i], folded + 8 * i, D);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0 && live_head) {
    device bfloat *orow = out + (ulong(row) * n_q + ulong(head) * gqa + fm) * D;
    const float inv = total == 0.0f ? 0.0f : 1.0f / total;
#pragma unroll
    for (int i = 0; i < C; ++i) {
      const float2 a = float2(te(o[i])) + float2(folded[fm * D + 8 * i + fn], folded[fm * D + 8 * i + fn + 1]);
      orow[8 * i + fn] = bfloat(a.x * inv);
      orow[8 * i + fn + 1] = bfloat(a.y * inv);
    }
  }
}

template [[host_name("sdpa_paged_decode_gqa_bfloat16_d_256")]] [[kernel]] void sdpa_paged_decode_gqa<256>(
    const device bfloat *, const device bfloat *, const device bfloat *, device bfloat *, const constant int &,
    const device int *, const device int *, const device uint *, const device uint *, const constant int &,
    const constant int &, const constant float &, const device uchar *, const device uint &, const device uchar *,
    const constant int &, uint2, uint, uint);
template [[host_name("sdpa_paged_decode_gqa_bfloat16_d_128")]] [[kernel]] void sdpa_paged_decode_gqa<128>(
    const device bfloat *, const device bfloat *, const device bfloat *, device bfloat *, const constant int &,
    const device int *, const device int *, const device uint *, const device uint *, const constant int &,
    const constant int &, const constant float &, const device uchar *, const device uint &, const device uchar *,
    const constant int &, uint2, uint, uint);

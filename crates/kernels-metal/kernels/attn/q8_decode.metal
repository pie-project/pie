#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

METAL_FUNC void sdpa_online_update(float score, thread float &max_score,
                                   thread float &sum_exp_score, thread float &history_scale,
                                   thread float &score_scale) {
  const float new_max = max(max_score, score);
  history_scale = fast::exp(max_score - new_max);
  score_scale = fast::exp(score - new_max);
  max_score = new_max;
  sum_exp_score = sum_exp_score * history_scale + score_scale;
}

METAL_FUNC float sdpa_lse_base2(float max_score, float sum_exp_score) {
  constexpr float kLog2E = 1.44269504088896340736f;
  return sum_exp_score > 0.0f ? (max_score * kLog2E + log2(sum_exp_score)) : -INFINITY;
}

template <typename T, int D, int V, bool WITH_LSE, int MERGE = 1>
inline void q8_decode_body(const device T *queries, const device T *k_pages,
                           const device T *v_pages, device T *out, const int gqa_factor,
                           const device int *position_ids, const device int *req_of_token,
                           const device uint *kv_page_indices, const device uint *kv_page_indptr,
                           const int page_size, const int n_kv_heads, const float scale,
                           const device uchar *attention_mask, const uint attention_mask_stride,
                           const device uchar *attention_mask_enabled, const int window,
                           const device T *sinks, device float *lse, threadgroup float *outputs,
                           threadgroup float *max_scores, threadgroup float *sum_exp_scores,
                           uint3 tid, uint3 tpg, uint simd_gid, uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr float NEG_INF = -3.0e38f;

  typedef float U;
  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U v[v_per_thread];
  thread U o[v_per_thread];

  const int q_batch_head_idx = tid.x;
  const int row = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int n_q_heads = int(tpg.x);

  const int r = req_of_token[row];
  const int q_pos = position_ids[row];

  const int mask_word = int(attention_mask_enabled[row]);
  const bool wide = window < 0 || mask_word == 2;
  const int extent = window < 0 ? (-window - 1) : window;
  const int kv_start = (extent > 0 && q_pos >= extent) ? (q_pos - extent + 1) : 0;
  const int page_base = int(kv_page_indptr[r]);

  queries += (size_t(row) * n_q_heads + q_batch_head_idx) * D + simd_lid * qk_per_thread;
  out += (size_t(row) * n_q_heads + q_batch_head_idx) * V;
  out += simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++)
    q[i] = static_cast<U>(scale) * queries[i];
  for (int i = 0; i < v_per_thread; i++)
    o[i] = 0;

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  const bool masked = mask_word != 0;

  auto absorb = [&](size_t slot) {
    const ulong at = (slot * n_kv_heads + kv_head_idx) * ulong(2 * D);
    const device char *kb = reinterpret_cast<const device char *>(k_pages) + at;
    const device char *vb = reinterpret_cast<const device char *>(v_pages) + at;
    const float ks = *reinterpret_cast<const device float *>(kb + D);
    const float vs = *reinterpret_cast<const device float *>(vb + D);
    for (int j = 0; j < qk_per_thread; ++j)
      k[j] = float(kb[simd_lid * qk_per_thread + j]) * ks;
    for (int j = 0; j < v_per_thread; ++j)
      v[j] = float(vb[simd_lid * v_per_thread + j]) * vs;
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++)
      score += q[j] * k[j];
    score = simd_sum(score);
    U factor, exp_score;
    sdpa_online_update(score, max_score, sum_exp_score, factor, exp_score);
    for (int j = 0; j < v_per_thread; j++)
      o[j] = o[j] * factor + exp_score * v[j];
  };

  auto attends = [&](int kp) {
    return !masked || (uint(kp) < attention_mask_stride &&
                       attention_mask[size_t(row) * attention_mask_stride + uint(kp)] != 0);
  };

  const int stride = page_size;

  const int last_kp = wide ? int(kv_page_indptr[r + 1] - uint(page_base)) * stride - 1 : q_pos;
  const int first_page = kv_start / stride;
  const int last_page = last_kp / stride;

  if (last_page - first_page + 1 >= BN) {
    for (int pix = first_page + simd_gid; pix <= last_page; pix += BN) {
      const size_t base = size_t(kv_page_indices[page_base + pix]) * stride;
      const int lo = max(kv_start, pix * stride);
      const int hi = min(last_kp, pix * stride + stride - 1);
      for (int kp = lo; kp <= hi; ++kp) {
        if (attends(kp))
          absorb(base + size_t(kp - pix * stride));
      }
    }
  } else {
    for (int kp = kv_start + simd_gid; kp <= last_kp; kp += BN) {

      size_t slot;

      const int page_ix = kp / page_size;
      const int page_off = kp % page_size;
      slot = size_t(kv_page_indices[page_base + page_ix]) * stride + page_off;

      if (attends(kp))
        absorb(slot);
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  if constexpr (WITH_LSE) {
    if (simd_gid == 0 && simd_lid == 0) {
      lse[size_t(row) * size_t(n_q_heads) + size_t(q_batch_head_idx)] =
          sdpa_lse_base2(new_max, sum_exp_score);
    }
  }

  for (int base = 0; base < v_per_thread; base += MERGE) {
#pragma unroll
    for (int j = 0; j < MERGE; ++j)
      outputs[(simd_lid * BD + simd_gid) * MERGE + j] = o[base + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int j = 0; j < MERGE; ++j) {
      o[base + j] = simd_sum(outputs[(simd_gid * BD + simd_lid) * MERGE + j] * factor);
      o[base + j] = sum_exp_score == 0 ? o[base + j] : o[base + j] / sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0)
    for (int i = 0; i < v_per_thread; i++)
      out[i] = static_cast<T>(o[i]);
}

template <int BN> constexpr int sdpa_decode_outputs() { return BN * 32; }

template <typename T, int D, int V = D, int MERGE = (V == 256 ? 4 : 1)>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void q8_decode(
    const device T *queries [[buffer(0)]], const device T *k_pages [[buffer(1)]],
    const device T *v_pages [[buffer(2)]], device T *out [[buffer(3)]],
    const constant int &gqa_factor [[buffer(4)]], const device int *position_ids [[buffer(5)]],
    const device int *req_of_token [[buffer(6)]], const device uint *kv_page_indices [[buffer(7)]],
    const device uint *kv_page_indptr [[buffer(8)]], const constant int &page_size [[buffer(9)]],
    const constant int &n_kv_heads [[buffer(10)]], const constant float &scale [[buffer(11)]],
    const device uchar *attention_mask [[buffer(12)]],
    const device uint &attention_mask_stride [[buffer(13)]],
    const device uchar *attention_mask_enabled [[buffer(14)]],
    const constant int &window [[buffer(15)]], const device T *sinks [[buffer(16)]],
    uint3 tid [[threadgroup_position_in_grid]], uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  threadgroup float outputs[sdpa_decode_outputs<BN>() * MERGE];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];
  q8_decode_body<T, D, V, false, MERGE>(queries, k_pages, v_pages, out, gqa_factor, position_ids,
                                        req_of_token, kv_page_indices, kv_page_indptr, page_size,
                                        n_kv_heads, scale, attention_mask, attention_mask_stride,
                                        attention_mask_enabled, window, sinks, nullptr, outputs,
                                        max_scores, sum_exp_scores, tid, tpg, simd_gid, simd_lid);
}

template <typename T, int D, int V = D>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void q8_decode_lse(
    const device T *queries [[buffer(0)]], const device T *k_pages [[buffer(1)]],
    const device T *v_pages [[buffer(2)]], device T *out [[buffer(3)]],
    const constant int &gqa_factor [[buffer(4)]], const device int *position_ids [[buffer(5)]],
    const device int *req_of_token [[buffer(6)]], const device uint *kv_page_indices [[buffer(7)]],
    const device uint *kv_page_indptr [[buffer(8)]], const constant int &page_size [[buffer(9)]],
    const constant int &n_kv_heads [[buffer(10)]], const constant float &scale [[buffer(11)]],
    const device uchar *attention_mask [[buffer(12)]],
    const device uint &attention_mask_stride [[buffer(13)]],
    const device uchar *attention_mask_enabled [[buffer(14)]],
    const constant int &window [[buffer(15)]], const device T *sinks [[buffer(16)]],
    device float *lse [[buffer(17)]], uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  threadgroup float outputs[sdpa_decode_outputs<BN>()];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];
  q8_decode_body<T, D, V, true>(queries, k_pages, v_pages, out, gqa_factor, position_ids,
                                req_of_token, kv_page_indices, kv_page_indptr, page_size,
                                n_kv_heads, scale, attention_mask, attention_mask_stride,
                                attention_mask_enabled, window, sinks, lse, outputs, max_scores,
                                sum_exp_scores, tid, tpg, simd_gid, simd_lid);
}

#define instantiate_sdpa_paged(name, itype, d, v)                                                  \
  template [[host_name("q8_decode_" #name "_d_" #d)]] [[kernel]] void q8_decode<itype, d, v>(      \
      const device itype *, const device itype *, const device itype *, device itype *,            \
      const constant int &, const device int *, const device int *, const device uint *,           \
      const device uint *, const constant int &, const constant int &, const constant float &,     \
      const device uchar *, const device uint &, const device uchar *, const constant int &,       \
      const device itype *, uint3, uint3, uint, uint);

instantiate_sdpa_paged(bfloat16, bfloat, 256, 256)
instantiate_sdpa_paged(bfloat16, bfloat, 512, 512)
instantiate_sdpa_paged(bfloat16, bfloat, 128, 128)
instantiate_sdpa_paged(bfloat16, bfloat, 64, 64)

#define instantiate_sdpa_paged_lse(name, itype, d, v)                                              \
  template [[host_name("q8_decode_lse_" #name "_d_" #d)]] [[kernel]] void                          \
  q8_decode_lse<itype, d, v>(                                                                      \
      const device itype *, const device itype *, const device itype *, device itype *,            \
      const constant int &, const device int *, const device int *, const device uint *,           \
      const device uint *, const constant int &, const constant int &, const constant float &,     \
      const device uchar *, const device uint &, const device uchar *, const constant int &,       \
      const device itype *, device float *, uint3, uint3, uint, uint);

instantiate_sdpa_paged_lse(bfloat16, bfloat, 64, 64)
instantiate_sdpa_paged_lse(bfloat16, bfloat, 128, 128)
instantiate_sdpa_paged_lse(bfloat16, bfloat, 256, 256)
instantiate_sdpa_paged_lse(bfloat16, bfloat, 512, 512)

template <int GH>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void
q8_vector_gqa(const device bfloat *queries [[buffer(0)]], const device bfloat *keys [[buffer(1)]],
              const device bfloat *values [[buffer(2)]], device bfloat *out [[buffer(3)]],
              constant int &gqa [[buffer(4)]], const device int *positions [[buffer(5)]],
              const device int *owners [[buffer(6)]], const device uint *pages [[buffer(7)]],
              const device uint *ptr [[buffer(8)]], constant int &page_size [[buffer(9)]],
              constant int &kv_heads [[buffer(10)]], constant float &scale [[buffer(11)]],
              const device uchar *mask [[buffer(12)]], constant uint &mask_stride [[buffer(13)]],
              const device uchar *enabled [[buffer(14)]], constant int &window [[buffer(15)]],
              const device bfloat *sinks [[buffer(16)]],
              uint3 tile [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]],
              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int D = 256, PER = 8, BN = 32, MERGE = 4;
  const int head = tile.x * GH, heads = grid.x * GH, row = tile.y, kv = head / gqa;
  const int req = owners[row], pos = positions[row], flag = enabled[row];
  const bool wide = window < 0 || flag == 2;
  const int extent = window < 0 ? -window - 1 : window;
  const int start = extent > 0 && pos >= extent ? pos - extent + 1 : 0;
  const int pagebase = ptr[req];
  const int last = wide ? int(ptr[req + 1] - uint(pagebase)) * page_size - 1 : pos;
  const int firstpage = start / page_size, lastpage = last / page_size;
  float q[GH][PER], o[GH][PER], maxima[GH], totals[GH];
#pragma unroll
  for (int h = 0; h < GH; ++h) {
    maxima[h] = -3.0e38f;
    totals[h] = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      q[h][j] = scale * float(queries[(size_t(row) * heads + head + h) * D + lane * PER + j]);
      o[h][j] = 0;
    }
  }
  auto absorb = [&](size_t slot) {
    const size_t at = (slot * kv_heads + kv) * 2 * D;
    const device char *kb = reinterpret_cast<const device char *>(keys) + at;
    const device char *vb = reinterpret_cast<const device char *>(values) + at;
    const float ks = *reinterpret_cast<const device float *>(kb + D);
    const float vs = *reinterpret_cast<const device float *>(vb + D);
    float k[PER], v[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      k[j] = float(kb[lane * PER + j]) * ks;
      v[j] = float(vb[lane * PER + j]) * vs;
    }
#pragma unroll
    for (int h = 0; h < GH; ++h) {
      float score = 0;
#pragma unroll
      for (int j = 0; j < PER; ++j)
        score += q[h][j] * k[j];
      score = simd_sum(score);
      float factor, exp_score;
      sdpa_online_update(score, maxima[h], totals[h], factor, exp_score);
#pragma unroll
      for (int j = 0; j < PER; ++j)
        o[h][j] = o[h][j] * factor + exp_score * v[j];
    }
  };
  auto attends = [&](int kp) {
    return flag == 0 || (uint(kp) < mask_stride && mask[size_t(row) * mask_stride + uint(kp)] != 0);
  };
  if (lastpage - firstpage + 1 >= BN) {
    for (int pix = firstpage + sg; pix <= lastpage; pix += BN) {
      const size_t base = size_t(pages[pagebase + pix]) * page_size;
      const int lo = max(start, pix * page_size), hi = min(last, pix * page_size + page_size - 1);
      for (int kp = lo; kp <= hi; ++kp)
        if (attends(kp))
          absorb(base + size_t(kp - pix * page_size));
    }
  } else {
    for (int kp = start + sg; kp <= last; kp += BN) {
      const size_t slot = size_t(pages[pagebase + kp / page_size]) * page_size + kp % page_size;
      if (attends(kp))
        absorb(slot);
    }
  }
  threadgroup float ms[GH * BN], ss[GH * BN], outputs[BN * 32 * MERGE];
  if (lane == 0) {
    for (int h = 0; h < GH; ++h) {
      ms[h * BN + sg] = maxima[h];
      ss[h * BN + sg] = totals[h];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (int h = 0; h < GH; ++h) {
    const float mx = ms[h * BN + lane], newmax = simd_max(mx), factor = fast::exp(mx - newmax);
    const float denominator = simd_sum(ss[h * BN + lane] * factor);
    for (int base = 0; base < PER; base += MERGE) {
#pragma unroll
      for (int j = 0; j < MERGE; ++j)
        outputs[(lane * 32 + sg) * MERGE + j] = o[h][base + j];
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (int j = 0; j < MERGE; ++j) {
        o[h][base + j] = simd_sum(outputs[(sg * 32 + lane) * MERGE + j] * factor);
        o[h][base + j] = denominator == 0 ? o[h][base + j] : o[h][base + j] / denominator;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0)
      for (int j = 0; j < PER; ++j)
        out[(size_t(row) * heads + head + h) * D + sg * PER + j] = bfloat(o[h][j]);
  }
}
#define PIE_VECTOR_GQA(name, heads)                                                                \
  template [[host_name(name)]] [[kernel]] void q8_vector_gqa<heads>(                               \
      const device bfloat *, const device bfloat *, const device bfloat *, device bfloat *,        \
      constant int &, const device int *, const device int *, const device uint *,                 \
      const device uint *, constant int &, constant int &, constant float &, const device uchar *, \
      constant uint &, const device uchar *, constant int &, const device bfloat *, uint3, uint3,  \
      uint, uint);
PIE_VECTOR_GQA("q8_vector_gqa_2", 2)

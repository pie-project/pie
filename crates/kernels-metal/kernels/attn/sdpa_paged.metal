#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

METAL_FUNC void sdpa_online_update(
    float score, thread float& max_score, thread float& sum_exp_score,
    thread float& history_scale, thread float& score_scale) {
  const float new_max = max(max_score, score);
  history_scale = fast::exp(max_score - new_max);
  score_scale = fast::exp(score - new_max);
  max_score = new_max;
  sum_exp_score = sum_exp_score * history_scale + score_scale;
}

METAL_FUNC float sdpa_merge_sink(
    float sink, float reference_max, thread float& sum_exp_score) {
  const float merged_max = max(reference_max, sink);
  const float output_scale = fast::exp(reference_max - merged_max);
  sum_exp_score =
      sum_exp_score * output_scale + fast::exp(sink - merged_max);
  return output_scale;
}

METAL_FUNC float sdpa_lse_base2(float max_score, float sum_exp_score) {
  constexpr float kLog2E = 1.44269504088896340736f;
  return sum_exp_score > 0.0f ? (max_score * kLog2E + log2(sum_exp_score))
                              : -INFINITY;
}

template <
    typename T,
    int D,
    int V,
    bool WITH_SINK,
    bool WITH_LSE,
    int PAGE_SIZE,
    bool FAST_FULL,
    int BN>
inline void sdpa_paged_decode_body(
    const device T* queries,
    const device T* k_pages,
    const device T* v_pages,
    device T* out,
    const int gqa_factor,
    const device int* position_ids,
    const device int* req_of_token,
    const device uint* kv_page_indices,
    const device uint* kv_page_indptr,
    const int page_size,
    const int n_kv_heads,
    const float scale,
    const device uchar* attention_mask,
    const uint attention_mask_stride,
    const device uchar* attention_mask_enabled,
    const int window,
    const device T* sinks,
    device float* lse,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    threadgroup float* factors,
    threadgroup float* final_sum,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid) {
  static_assert(
      !(WITH_SINK && WITH_LSE),
      "a folded sink and a published lse are two readings of one denominator");
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
  const int row              = tid.y;
  const int kv_head_idx      = q_batch_head_idx / gqa_factor;
  const int n_q_heads        = int(tpg.x);

  const int r          = req_of_token[row];
  const int q_pos      = position_ids[row];

  const int kv_start   =
      FAST_FULL ? 0 : ((window > 0 && q_pos >= window) ? (q_pos - window + 1) : 0);
  const int page_base  = int(kv_page_indptr[r]);

  queries += (size_t(row) * n_q_heads + q_batch_head_idx) * D + simd_lid * qk_per_thread;
  out     += (size_t(row) * n_q_heads + q_batch_head_idx) * V;
  if constexpr (BN == 32) out += simd_gid * v_per_thread;

  for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<U>(scale) * queries[i];
  for (int i = 0; i < v_per_thread; i++) o[i] = 0;

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  const bool masked = !FAST_FULL && attention_mask_enabled[row] != 0;

  auto absorb = [&](size_t slot) {
    const device T* kptr =
        k_pages + (slot * n_kv_heads + kv_head_idx) * D + simd_lid * qk_per_thread;
    const device T* vptr =
        v_pages + (slot * n_kv_heads + kv_head_idx) * D + simd_lid * v_per_thread;
    for (int j = 0; j < qk_per_thread; j++) k[j] = kptr[j];
    for (int j = 0; j < v_per_thread; j++) v[j] = static_cast<U>(vptr[j]);
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
    score = simd_sum(score);
    U factor, exp_score;
    sdpa_online_update(score, max_score, sum_exp_score, factor, exp_score);
    for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * v[j];
  };

  auto attends = [&](int kp) {
    if constexpr (FAST_FULL) {
      return true;
    } else {
      return !masked ||
          (uint(kp) < attention_mask_stride &&
           attention_mask[size_t(row) * attention_mask_stride + uint(kp)] != 0);
    }
  };

  const int stride = PAGE_SIZE == 0 ? page_size : PAGE_SIZE;
  const int first_page = kv_start / stride;
  const int last_page = q_pos / stride;

  if (last_page - first_page + 1 >= BN) {
    for (int pix = first_page + simd_gid; pix <= last_page; pix += BN) {
      const size_t base = size_t(kv_page_indices[page_base + pix]) * stride;
      const int lo = max(kv_start, pix * stride);
      const int hi = min(q_pos, pix * stride + stride - 1);
      for (int kp = lo; kp <= hi; ++kp) {
        if (attends(kp)) absorb(base + size_t(kp - pix * stride));
      }
    }
  } else {
    int fast_page_ix = 0;
    int fast_page_off = simd_gid;
    for (int kp = kv_start + simd_gid; kp <= q_pos; kp += BN) {

      size_t slot;
      if constexpr (PAGE_SIZE == 32 && FAST_FULL) {
        slot = size_t(kv_page_indices[page_base + fast_page_ix]) * 32 + fast_page_off;
        fast_page_off += BN;
        if (fast_page_off >= 32) {
          fast_page_off -= 32;
          ++fast_page_ix;
        }
      } else {
        const int page_ix = PAGE_SIZE == 32 ? (kp >> 5) : (kp / page_size);
        const int page_off = PAGE_SIZE == 32 ? (kp & 31) : (kp % page_size);
        slot = size_t(kv_page_indices[page_base + page_ix]) * stride + page_off;
      }
      if (attends(kp)) absorb(slot);
    }
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (BN == 32) {
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

    U orescale = 1;
    if (WITH_SINK) {
      const U sink = static_cast<U>(sinks[tid.x]);
      orescale = sdpa_merge_sink(sink, new_max, sum_exp_score);
    }

    for (int i = 0; i < v_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) * orescale;
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0)
      for (int i = 0; i < v_per_thread; i++) out[i] = static_cast<T>(o[i]);
  } else {
    if (simd_gid == 0) {
      const U lane_max = simd_lid < BN ? max_scores[simd_lid] : NEG_INF;
      U new_max = simd_max(lane_max);
      U factor = simd_lid < BN ? fast::exp(lane_max - new_max) : 0;
      U total = simd_sum(
          simd_lid < BN ? sum_exp_scores[simd_lid] * factor : U(0));
      if (simd_lid < BN) factors[simd_lid] = factor;
      if (simd_lid == 0) {

        if constexpr (WITH_LSE) {
          lse[size_t(row) * size_t(n_q_heads) + size_t(q_batch_head_idx)] =
              sdpa_lse_base2(new_max, total);
        }
        if (WITH_SINK) {
          const U sink = static_cast<U>(sinks[tid.x]);
          const U rescale = sdpa_merge_sink(sink, new_max, total);
          for (int s = 0; s < BN; ++s) factors[s] *= rescale;
        }
        final_sum[0] = total;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < v_per_thread; ++i) {
      outputs[simd_gid * BD + simd_lid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0) {
        U acc = 0;
        for (int s = 0; s < BN; ++s)
          acc += outputs[s * BD + simd_lid] * factors[s];
        const U denom = final_sum[0];
        out[simd_lid * v_per_thread + i] =
            static_cast<T>(denom == 0 ? acc : acc / denom);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

template <int BN> constexpr int sdpa_decode_outputs() { return BN * 32; }

template <
    typename T,
    int D,
    int V = D,
    bool WITH_SINK = false,
    int PAGE_SIZE = 0,
    bool FAST_FULL = false,
    int BN = 32>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_decode(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup float outputs[sdpa_decode_outputs<BN>()];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];
  threadgroup float factors[BN];
  threadgroup float final_sum[1];
  sdpa_paged_decode_body<T, D, V, WITH_SINK, false, PAGE_SIZE, FAST_FULL, BN>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, nullptr, outputs, max_scores, sum_exp_scores, factors,
      final_sum, tid, tpg, simd_gid, simd_lid);
}

template <typename T, int D, int V = D>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_decode_lse(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],
    device float* lse                          [[buffer(17)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  threadgroup float outputs[sdpa_decode_outputs<BN>()];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];
  threadgroup float factors[BN];
  threadgroup float final_sum[1];
  sdpa_paged_decode_body<T, D, V, false, true, 0, false, BN>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, lse, outputs, max_scores, sum_exp_scores, factors, final_sum, tid,
      tpg, simd_gid, simd_lid);
}

constant constexpr int kSdpaQT = 32;
template <int D> constexpr int sdpa_kt() { return 4096 / D; }
template <int D, bool KPL> constexpr int sdpa_kq() { return KPL ? kSdpaQT * D : 1; }
template <int D, bool KPL> constexpr int sdpa_kp_tile() {
  return KPL ? kSdpaQT * sdpa_kt<D>() : 1;
}

template <int D, bool KPL> constexpr int sdpa_kstride() { return KPL ? D + 2 : D; }

template <typename T, int D, int V, bool WITH_SINK, bool KEY_PER_LANE, bool WITH_LSE>
inline void sdpa_paged_tiled_body(
    const device T* queries,
    const device T* k_pages,
    const device T* v_pages,
    device T* out,
    const int gqa_factor,
    const device int* position_ids,
    const device int* req_of_token,
    const device uint* kv_page_indices,
    const device uint* kv_page_indptr,
    const int page_size,
    const int n_kv_heads,
    const float scale,
    const device uchar* attention_mask,
    const uint attention_mask_stride,
    const device uchar* attention_mask_enabled,
    const int window,
    const device T* sinks,
    device float* lse,
    const int n_rows,
    const int q_row_pitch,
    const int o_row_pitch,
    threadgroup T* ktile,
    threadgroup T* vtile,
    threadgroup T* qtile,
    threadgroup float* ptile,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid) {

  static_assert(V == D, "the tiled path stages one shape for K and V");
  static_assert(
      !(WITH_SINK && WITH_LSE),
      "a folded sink and a published lse are two readings of one denominator");

  constexpr int QT = 32;
  constexpr int KT = 4096 / D;
  constexpr int per_lane = D / 32;
  constexpr float NEG_INF = -3.0e38f;
  typedef float U;

  constexpr int kstride = KEY_PER_LANE ? D + 2 : D;
  constexpr int kcols = KEY_PER_LANE ? (KT + 31) / 32 : 1;

  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;
  const int row_lo    = int(tid.y) * QT;
  const int row       = row_lo + int(simd_gid);
  const bool live     = row < n_rows;
  const uint lid      = simd_gid * 32u + simd_lid;

  thread U q[per_lane];
  thread U o[per_lane];
  for (int i = 0; i < per_lane; i++) o[i] = 0;
  if (live) {
    const device T* qp =
        queries + (q_row_pitch > 0 ? size_t(row) * size_t(q_row_pitch)
                                  : size_t(row) * size_t(n_q_heads) * size_t(D)) +
        size_t(q_head) * size_t(D) + simd_lid * per_lane;
    for (int i = 0; i < per_lane; i++) q[i] = static_cast<U>(scale) * static_cast<U>(qp[i]);

    if (KEY_PER_LANE) {
      for (int i = 0; i < per_lane; i++)
        qtile[simd_gid * D + simd_lid * per_lane + i] = qp[i];
    }
  } else {
    for (int i = 0; i < per_lane; i++) q[i] = 0;
    if (KEY_PER_LANE) {
      for (int i = 0; i < per_lane; i++)
        qtile[simd_gid * D + simd_lid * per_lane + i] = T(0);
    }
  }

  const int q_pos     = live ? position_ids[row] : 0;
  const int my_start  = (window > 0 && q_pos >= window) ? (q_pos - window + 1) : 0;
  const bool masked   = live && attention_mask_enabled[row] != 0;

  U max_score = NEG_INF;
  U sum_exp_score = 0;

  int sub = 0;
  while (sub < QT && row_lo + sub < n_rows) {
    const int r = req_of_token[row_lo + sub];
    int sub_hi = sub + 1;
    while (sub_hi < QT && row_lo + sub_hi < n_rows && req_of_token[row_lo + sub_hi] == r) sub_hi++;

    int kp_hi = 0;
    int kp_lo = 0x7fffffff;
    for (int i = sub; i < sub_hi; i++) {
      const int p = position_ids[row_lo + i];
      kp_hi = max(kp_hi, p);
      kp_lo = min(kp_lo, (window > 0 && p >= window) ? (p - window + 1) : 0);
    }
    const int page_base = int(kv_page_indptr[r]);
    const bool mine = live && int(simd_gid) >= sub && int(simd_gid) < sub_hi;

    for (int base = kp_lo; base <= kp_hi; base += KT) {
      const int cnt = min(KT, kp_hi + 1 - base);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = int(lid); e < cnt * D; e += 1024) {
        const int kk = e / D;
        const int d  = e - kk * D;
        const int kp = base + kk;
        const int page = int(kv_page_indices[page_base + kp / page_size]);
        const size_t slot = size_t(page) * page_size + (kp % page_size);
        const size_t off = (slot * n_kv_heads + kv_head) * D + d;
        ktile[kk * kstride + d] = k_pages[off];
        vtile[e] = v_pages[off];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (!mine) continue;

      const threadgroup T* vbase = vtile + simd_lid * per_lane;

      const auto keeps = [&](int kp) {
        if (kp > q_pos || kp < my_start) return false;
        return !(masked && (uint(kp) >= attention_mask_stride ||
                            attention_mask[size_t(row) * attention_mask_stride +
                                           uint(kp)] == 0));
      };

      if (KEY_PER_LANE) {

        const threadgroup T* qrow = qtile + simd_gid * D;
        U s[kcols];
        U block_max = NEG_INF;
        for (int c = 0; c < kcols; c++) {
          const int kk = c * 32 + int(simd_lid);
          U sc = NEG_INF;
          if (kk < cnt && keeps(base + kk)) {
            const threadgroup T* kptr = ktile + kk * kstride;
            U acc = 0;
            for (int d = 0; d < D; d++)
              acc += static_cast<U>(qrow[d]) * static_cast<U>(kptr[d]);
            sc = static_cast<U>(scale) * acc;
          }
          s[c] = sc;
          block_max = sc > block_max ? sc : block_max;
        }

        block_max = simd_max(block_max);
        const U new_max = block_max > max_score ? block_max : max_score;

        if (new_max > NEG_INF) {
          const U factor = max_score == NEG_INF ? U(0) : fast::exp(max_score - new_max);
          U block_sum = 0;
          for (int c = 0; c < kcols; c++) {
            const int kk = c * 32 + int(simd_lid);
            const U p = s[c] == NEG_INF ? U(0) : fast::exp(s[c] - new_max);

            if (kk < KT) ptile[simd_gid * KT + kk] = p;
            block_sum += p;
          }
          block_sum = simd_sum(block_sum);
          simdgroup_barrier(mem_flags::mem_threadgroup);
          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + block_sum;

          for (int j = 0; j < per_lane; j++) o[j] *= factor;
          for (int kk = 0; kk < cnt; kk++) {

            const U p = ptile[simd_gid * KT + kk];
            if (p == U(0)) continue;
            const threadgroup T* vptr = vbase + kk * D;
            for (int j = 0; j < per_lane; j++)
              o[j] += p * static_cast<U>(vptr[j]);
          }
        }
      } else {
        const threadgroup T* kbase = ktile + simd_lid * per_lane;
        for (int kk = 0; kk < cnt; kk++) {
          const int kp = base + kk;

          if (!keeps(kp)) continue;
          const threadgroup T* kptr = kbase + kk * kstride;
          U score = 0;
          for (int j = 0; j < per_lane; j++) score += q[j] * static_cast<U>(kptr[j]);
          score = simd_sum(score);

          U factor, exp_score;
          sdpa_online_update(
              score, max_score, sum_exp_score, factor, exp_score);
          const threadgroup T* vptr = vbase + kk * D;
          for (int j = 0; j < per_lane; j++)
            o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
        }
      }
    }
    sub = sub_hi;
  }

  if (!live) return;

  if constexpr (WITH_LSE) {
    if (simd_lid == 0) {
      lse[size_t(row) * size_t(n_q_heads) + size_t(q_head)] =
          sdpa_lse_base2(max_score, sum_exp_score);
    }
  }

  U orescale = 1;
  if (WITH_SINK) {
    const U sink = static_cast<U>(sinks[q_head]);
    orescale = sdpa_merge_sink(sink, max_score, sum_exp_score);
  }

  device T* op = out +
                 (o_row_pitch > 0 ? size_t(row) * size_t(o_row_pitch)
                                  : size_t(row) * size_t(n_q_heads) * size_t(V)) +
                 size_t(q_head) * size_t(V) + simd_lid * per_lane;
  for (int j = 0; j < per_lane; j++) {
    const U x = o[j] * orescale;
    op[j] = static_cast<T>(sum_exp_score == 0 ? x : x / sum_exp_score);
  }
}

template <typename T, int D, int V = D, bool WITH_SINK = false, bool KEY_PER_LANE = false>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_tiled(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],
    const constant int& n_rows                 [[buffer(17)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup T ktile[sdpa_kt<D>() * sdpa_kstride<D, KEY_PER_LANE>()];
  threadgroup T vtile[sdpa_kt<D>() * D];
  threadgroup T qtile[sdpa_kq<D, KEY_PER_LANE>()];
  threadgroup float ptile[sdpa_kp_tile<D, KEY_PER_LANE>()];
  sdpa_paged_tiled_body<T, D, V, WITH_SINK, KEY_PER_LANE, false>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, nullptr, n_rows,
      0, 0,
      ktile, vtile, qtile, ptile, tid, tpg, simd_gid, simd_lid);
}

template <typename T, int D, int V = D, bool KEY_PER_LANE = false>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_tiled_lse(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],
    const constant int& n_rows                 [[buffer(17)]],
    device float* lse                          [[buffer(18)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup T ktile[sdpa_kt<D>() * sdpa_kstride<D, KEY_PER_LANE>()];
  threadgroup T vtile[sdpa_kt<D>() * D];
  threadgroup T qtile[sdpa_kq<D, KEY_PER_LANE>()];
  threadgroup float ptile[sdpa_kp_tile<D, KEY_PER_LANE>()];
  sdpa_paged_tiled_body<T, D, V, false, KEY_PER_LANE, true>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, lse, n_rows,
      0, 0,
      ktile, vtile, qtile, ptile, tid, tpg, simd_gid, simd_lid);
}

template <typename T, int D, int V = D, bool WITH_SINK = false, bool KEY_PER_LANE = false>
[[kernel]] [[max_total_threads_per_threadgroup(1024)]] void sdpa_paged_tiled_strided(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor          [[buffer(4)]],
    const device int* position_ids          [[buffer(5)]],
    const device int* req_of_token          [[buffer(6)]],
    const device uint* kv_page_indices      [[buffer(7)]],
    const device uint* kv_page_indptr       [[buffer(8)]],
    const constant int& page_size           [[buffer(9)]],
    const constant int& n_kv_heads          [[buffer(10)]],
    const constant float& scale             [[buffer(11)]],
    const device uchar* attention_mask      [[buffer(12)]],
    const device uint& attention_mask_stride[[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],
    const constant int& n_rows                 [[buffer(17)]],
    const constant int& q_row_pitch            [[buffer(18)]],
    const constant int& o_row_pitch            [[buffer(19)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup T ktile[sdpa_kt<D>() * sdpa_kstride<D, KEY_PER_LANE>()];
  threadgroup T vtile[sdpa_kt<D>() * D];
  threadgroup T qtile[sdpa_kq<D, KEY_PER_LANE>()];
  threadgroup float ptile[sdpa_kp_tile<D, KEY_PER_LANE>()];
  sdpa_paged_tiled_body<T, D, V, WITH_SINK, KEY_PER_LANE, false>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, nullptr, n_rows,
      q_row_pitch, o_row_pitch,
      ktile, vtile, qtile, ptile, tid, tpg, simd_gid, simd_lid);
}

#define instantiate_sdpa_tiled_strided(fn, name, itype, d, v, sink, kpl)   \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_tiled_strided<itype, d, v, sink, kpl>(        \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, const constant int&,       \
      const constant int&, const constant int&,                            \
      uint3, uint3, uint, uint);

instantiate_sdpa_tiled_strided("sdpa_paged_tiled_strided", bfloat16, bfloat, 256, 256, false, false)

#define instantiate_sdpa_tiled_impl(fn, name, itype, d, v, sink, kpl)      \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_tiled<itype, d, v, sink, kpl>(                \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, const constant int&,       \
      uint3, uint3, uint, uint);

instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 128, 128, false, true)
instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 64, 64, false, true)

instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 256, 256, false, false)
instantiate_sdpa_tiled_impl("sdpa_paged_tiled", bfloat16, bfloat, 512, 512, false, false)

instantiate_sdpa_tiled_impl("sdpa_paged_tiled_sink", bfloat16, bfloat, 64, 64, true, true)

#define instantiate_sdpa_tiled_lse(fn, name, itype, d, v, kpl)             \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_tiled_lse<itype, d, v, kpl>(                  \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, const constant int&,       \
      device float*, uint3, uint3, uint, uint);

instantiate_sdpa_tiled_lse("sdpa_paged_tiled_lse", bfloat16, bfloat, 64, 64, true)

#define instantiate_sdpa_paged_impl(fn, name, itype, d, v, sink)           \
  template [[host_name(fn "_" #name "_d_" #d)]]                            \
  [[kernel]] void sdpa_paged_decode<itype, d, v, sink, 0, false, 32>(      \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*,                            \
      uint3, uint3, uint, uint);

#define instantiate_sdpa_paged(name, itype, d, v)                          \
  instantiate_sdpa_paged_impl("sdpa_paged_decode", name, itype, d, v, false)

#define instantiate_sdpa_paged_sink(name, itype, d, v)                     \
  instantiate_sdpa_paged_impl("sdpa_paged_decode_sink", name, itype, d, v, true)

instantiate_sdpa_paged(bfloat16, bfloat, 256, 256)
instantiate_sdpa_paged(bfloat16, bfloat, 512, 512)
instantiate_sdpa_paged(bfloat16, bfloat, 128, 128)
instantiate_sdpa_paged(bfloat16, bfloat, 64, 64)
instantiate_sdpa_paged_sink(bfloat16, bfloat, 64, 64)

#define instantiate_sdpa_paged_lse(name, itype, d, v)                      \
  template [[host_name("sdpa_paged_decode_lse_" #name "_d_" #d)]]          \
  [[kernel]] void sdpa_paged_decode_lse<itype, d, v>(                      \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const constant int&, const device int*,               \
      const device int*, const device uint*, const device uint*,           \
      const constant int&, const constant int&, const constant float&,     \
      const device uchar*, const device uint&, const device uchar*,        \
      const constant int&, const device itype*, device float*,             \
      uint3, uint3, uint, uint);

instantiate_sdpa_paged_lse(bfloat16, bfloat, 64, 64)

#define instantiate_sdpa_paged_p32(name, itype, d, v)                       \
  template [[host_name("sdpa_paged_decode_" #name "_d_" #d "_p32")]]        \
  [[kernel]] void sdpa_paged_decode<itype, d, v, false, 32, true, 32>(      \
      const device itype*, const device itype*, const device itype*,        \
      device itype*, const constant int&, const device int*,                \
      const device int*, const device uint*, const device uint*,            \
      const constant int&, const constant int&, const constant float&,      \
      const device uchar*, const device uint&, const device uchar*,         \
      const constant int&, const device itype*,                             \
      uint3, uint3, uint, uint);

instantiate_sdpa_paged_p32(bfloat16, bfloat, 128, 128)
instantiate_sdpa_paged_p32(bfloat16, bfloat, 64, 64)

#define instantiate_sdpa_paged_short(name, itype, d, v, sg)                  \
  template [[host_name("sdpa_paged_decode_" #name "_d_" #d "_p32_sg" #sg)]] \
  [[kernel]] void sdpa_paged_decode<itype, d, v, false, 32, true, sg>(       \
      const device itype*, const device itype*, const device itype*,         \
      device itype*, const constant int&, const device int*,                 \
      const device int*, const device uint*, const device uint*,             \
      const constant int&, const constant int&, const constant float&,       \
      const device uchar*, const device uint&, const device uchar*,          \
      const constant int&, const device itype*,                              \
      uint3, uint3, uint, uint);

instantiate_sdpa_paged_short(bfloat16, bfloat, 64, 64, 8)

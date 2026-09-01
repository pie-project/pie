#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

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

/// **THE LOG-SUM-EXP PLANE THIS KERNEL PUBLISHES IS THE SCALAR KERNEL'S, TO
/// THE DEFINITION AND NOT MERELY TO THE TOLERANCE.** `WITH_LSE` writes
/// `sdpa_lse_base2(max_score, sum_exp)` — base 2, one f32 per query row per
/// query head at `lse[row * n_q_heads + q_head]`, and `-INFINITY` exactly
/// when the row kept no key — which is byte-for-byte the arm
/// `sdpa_paged_tiled_lse` in `attn/sdpa_paged.metal` takes, out of the same
/// helper. The two consumers (`attn/merge_lse.metal`'s fold and
/// `attn/attn_sink.metal`'s rescale) both branch on `isfinite` and both read
/// base 2, so the empty row has to be a true `-inf` and not a large
/// negative: `NEG_INF` is the running max's SEED and never its published
/// value.
///
/// **The granularity is per row, and the fold that gets it there is already
/// done.** A simdgroup owns eight whole query rows (`RPS == 8`), so no
/// cross-simdgroup reduction exists or is owed — the only fold is across the
/// four lanes that hold one fragment row's eight columns, and the online
/// softmax already runs it every tile (`simd_shuffle_xor` by 1 and by 8, the
/// two bits that leave `fm` fixed and move `fn`). `max_score` and `sum_exp`
/// are therefore the whole row's, replicated on those four lanes, and the
/// epilogue publishes from the one lane holding column zero (`fn == 0`),
/// which is exactly one lane per row of the fragment.
///
/// **A folded sink and a published lse are two readings of one denominator**,
/// so they are mutually exclusive here as they are next door: gpt-oss takes
/// the `_lse` arm and `attention.sink` folds that mass in afterwards off the
/// plane this writes.
template <typename T, int D, int KT, bool WITH_SINK, bool WITH_LSE>
inline void sdpa_paged_mma_body(
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
    const int n_rows,
    device float* lse,
    threadgroup half* qtile,
    threadgroup half* ktile,
    threadgroup half* vtile,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid) {
  constexpr int QT  = 32;
  constexpr int SGS = 4;
  constexpr int RPS = QT / SGS;
  constexpr int DF  = D / 8;
  constexpr int KF  = KT / 8;
  constexpr float NEG_INF = -3.0e38f;

  static_assert(D % 8 == 0 && KT % 8 == 0, "the matrix unit tiles in eights");
  static_assert(RPS == 8, "a simdgroup owns exactly one fragment row of queries");
  static_assert(
      !(WITH_SINK && WITH_LSE),
      "a folded sink and a published lse are two readings of one denominator");

  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;
  const int row_lo    = int(tid.y) * QT;
  const uint lid      = simd_gid * 32u + simd_lid;

  const short qid = short(simd_lid) / 4;
  const short fm  = (qid & 4) + ((short(simd_lid) / 2) % 4);
  const short fn  = (qid & 2) * 2 + (short(simd_lid) % 2) * 2;

  const int my_row = row_lo + int(simd_gid) * RPS + int(fm);
  const bool live  = my_row < n_rows;

  for (uint e = lid; e < uint(QT * D); e += 128u) {
    const int r  = int(e) / D;
    const int d  = int(e) - r * D;
    const int gr = row_lo + r;
    qtile[e] = gr < n_rows
                   ? half(float(queries[(size_t(gr) * n_q_heads + q_head) * D + d]) * scale)
                   : half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_matrix<half, 8, 8> Qf[DF];
  for (int i = 0; i < DF; i++) {
    simdgroup_load(Qf[i], qtile, D, ulong2(uint(i * 8), uint(simd_gid) * RPS), false);
  }

  const int q_pos    = live ? position_ids[my_row] : 0;
  const int my_start = (window > 0 && q_pos >= window) ? (q_pos - window + 1) : 0;
  const bool masked  = live && attention_mask_enabled[my_row] != 0;
  const int my_req   = live ? req_of_token[my_row] : -1;

  float Ov[DF * 2];
  for (int i = 0; i < DF * 2; i++) Ov[i] = 0.0f;
  float max_score = NEG_INF;
  float sum_exp = 0.0f;

  int sub = 0;
  while (sub < QT && row_lo + sub < n_rows) {
    const int r = req_of_token[row_lo + sub];
    int sub_hi = sub + 1;
    while (sub_hi < QT && row_lo + sub_hi < n_rows && req_of_token[row_lo + sub_hi] == r)
      sub_hi++;

    int kp_hi = 0;
    int kp_lo = 0x7fffffff;
    for (int i = sub; i < sub_hi; i++) {
      const int p = position_ids[row_lo + i];
      kp_hi = max(kp_hi, p);
      kp_lo = min(kp_lo, (window > 0 && p >= window) ? (p - window + 1) : 0);
    }
    const int page_base = int(kv_page_indptr[r]);

    const bool mine = live && my_req == r;

    for (int base = kp_lo; base <= kp_hi; base += KT) {
      const int cnt = min(KT, kp_hi + 1 - base);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lid; e < uint(KT * D); e += 128u) {
        const int kk = int(e) / D;
        const int d  = int(e) - kk * D;
        if (kk < cnt) {
          const int kp = base + kk;
          const int page = int(kv_page_indices[page_base + kp / page_size]);
          const size_t slot = size_t(page) * page_size + size_t(kp % page_size);
          const size_t off = (slot * n_kv_heads + kv_head) * D + d;
          ktile[d * KT + kk] = half(float(k_pages[off]));
          vtile[e] = half(float(v_pages[off]));
        } else {

          ktile[d * KT + kk] = half(0);
          vtile[e] = half(0);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      simdgroup_matrix<half, 8, 8> S[KF];
      for (int c = 0; c < KF; c++) S[c] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
      for (int i = 0; i < DF; i++) {
        for (int c = 0; c < KF; c++) {

          simdgroup_matrix<half, 8, 8> Bk;
          simdgroup_load(Bk, ktile, KT, ulong2(uint(c * 8), uint(i * 8)), false);
          simdgroup_multiply_accumulate(S[c], Qf[i], Bk, S[c]);
        }
      }

      float sv[KF * 2];
      float lmax = NEG_INF;
      for (int c = 0; c < KF; c++) {
        thread auto& e = S[c].thread_elements();
        for (int j = 0; j < 2; j++) {
          const int kk = c * 8 + int(fn) + j;
          bool keep = mine && kk < cnt;
          if (keep) {
            const int kp = base + kk;
            keep = kp <= q_pos && kp >= my_start;
            if (keep && masked) {
              keep = !(uint(kp) >= attention_mask_stride ||
                       attention_mask[size_t(my_row) * attention_mask_stride + uint(kp)] == 0);
            }
          }
          const float s = keep ? float(e[j]) : NEG_INF;
          sv[c * 2 + j] = s;
          lmax = s > lmax ? s : lmax;
        }
      }

      lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
      lmax = max(lmax, simd_shuffle_xor(lmax, 8u));

      const float new_max = max(max_score, lmax);
      const float factor = max_score == NEG_INF ? 0.0f : fast::exp(max_score - new_max);
      float lsum = 0.0f;
      for (int c = 0; c < KF; c++) {
        thread auto& e = S[c].thread_elements();
        for (int j = 0; j < 2; j++) {
          const float p = sv[c * 2 + j] == NEG_INF ? 0.0f : fast::exp(sv[c * 2 + j] - new_max);
          e[j] = half(p);
          lsum += p;
        }
      }
      lsum += simd_shuffle_xor(lsum, 1u);
      lsum += simd_shuffle_xor(lsum, 8u);

      max_score = new_max;
      sum_exp = sum_exp * factor + lsum;
      for (int i = 0; i < DF * 2; i++) Ov[i] *= factor;

      simdgroup_matrix<half, 8, 8> PV[DF];
      for (int n = 0; n < DF; n++) PV[n] = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
      for (int c = 0; c < KF; c++) {
        for (int n = 0; n < DF; n++) {
          simdgroup_matrix<half, 8, 8> Bv;
          simdgroup_load(Bv, vtile, D, ulong2(uint(n * 8), uint(c * 8)), false);
          simdgroup_multiply_accumulate(PV[n], S[c], Bv, PV[n]);
        }
      }

      for (int n = 0; n < DF; n++) {
        thread auto& e = PV[n].thread_elements();
        Ov[n * 2 + 0] += float(e[0]);
        Ov[n * 2 + 1] += float(e[1]);
      }
    }
    sub = sub_hi;
  }

  if (!live) return;

  if constexpr (WITH_LSE) {

    if (fn == 0) {
      lse[size_t(my_row) * size_t(n_q_heads) + size_t(q_head)] =
          sdpa_lse_base2(max_score, sum_exp);
    }
  }

  float orescale = 1.0f;
  if (WITH_SINK) {
    orescale = sdpa_merge_sink(float(sinks[q_head]), max_score, sum_exp);
  }
  device T* op = out + (size_t(my_row) * n_q_heads + q_head) * D;
  for (int n = 0; n < DF; n++) {
    for (int j = 0; j < 2; j++) {
      const float x = Ov[n * 2 + j] * orescale;
      op[n * 8 + int(fn) + j] = static_cast<T>(sum_exp == 0.0f ? x : x / sum_exp);
    }
  }
}

template <typename T, int D, int KT, bool WITH_SINK>
[[kernel]] [[max_total_threads_per_threadgroup(128)]] void sdpa_paged_mma(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor             [[buffer(4)]],
    const device int* position_ids             [[buffer(5)]],
    const device int* req_of_token             [[buffer(6)]],
    const device uint* kv_page_indices         [[buffer(7)]],
    const device uint* kv_page_indptr          [[buffer(8)]],
    const constant int& page_size              [[buffer(9)]],
    const constant int& n_kv_heads             [[buffer(10)]],
    const constant float& scale                [[buffer(11)]],
    const device uchar* attention_mask         [[buffer(12)]],
    const device uint& attention_mask_stride   [[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],
    const constant int& n_rows                 [[buffer(17)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup half qtile[32 * D];
  threadgroup half ktile[D * KT];
  threadgroup half vtile[KT * D];
  sdpa_paged_mma_body<T, D, KT, WITH_SINK, false>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, n_rows, nullptr,
      qtile, ktile, vtile, tid, tpg, simd_gid, simd_lid);
}

template <typename T, int D, int KT>
[[kernel]] [[max_total_threads_per_threadgroup(128)]] void sdpa_paged_mma_lse(
    const device T* queries     [[buffer(0)]],
    const device T* k_pages     [[buffer(1)]],
    const device T* v_pages     [[buffer(2)]],
    device T* out               [[buffer(3)]],
    const constant int& gqa_factor             [[buffer(4)]],
    const device int* position_ids             [[buffer(5)]],
    const device int* req_of_token             [[buffer(6)]],
    const device uint* kv_page_indices         [[buffer(7)]],
    const device uint* kv_page_indptr          [[buffer(8)]],
    const constant int& page_size              [[buffer(9)]],
    const constant int& n_kv_heads             [[buffer(10)]],
    const constant float& scale                [[buffer(11)]],
    const device uchar* attention_mask         [[buffer(12)]],
    const device uint& attention_mask_stride   [[buffer(13)]],
    const device uchar* attention_mask_enabled [[buffer(14)]],
    const constant int& window                 [[buffer(15)]],
    const device T* sinks                      [[buffer(16)]],
    const constant int& n_rows                 [[buffer(17)]],
    device float* lse                          [[buffer(18)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup half qtile[32 * D];
  threadgroup half ktile[D * KT];
  threadgroup half vtile[KT * D];
  sdpa_paged_mma_body<T, D, KT, false, true>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      sinks, n_rows, lse,
      qtile, ktile, vtile, tid, tpg, simd_gid, simd_lid);
}

#define instantiate_sdpa_paged_mma(sfx, name, itype, d, kt, sink)            \
  template [[host_name("sdpa_paged_mma" sfx "_" #name "_d_" #d)]]            \
  [[kernel]] void sdpa_paged_mma<itype, d, kt, sink>(                        \
      const device itype*, const device itype*, const device itype*,         \
      device itype*, const constant int&, const device int*,                 \
      const device int*, const device uint*, const device uint*,             \
      const constant int&, const constant int&, const constant float&,       \
      const device uchar*, const device uint&, const device uchar*,          \
      const constant int&, const device itype*, const constant int&,         \
      uint3, uint3, uint, uint);

instantiate_sdpa_paged_mma("", bfloat16, bfloat, 64, 16, false)
instantiate_sdpa_paged_mma("_sink", bfloat16, bfloat, 64, 16, true)

#define instantiate_sdpa_paged_mma_lse(name, itype, d, kt)                   \
  template [[host_name("sdpa_paged_mma_lse_" #name "_d_" #d)]]               \
  [[kernel]] void sdpa_paged_mma_lse<itype, d, kt>(                          \
      const device itype*, const device itype*, const device itype*,         \
      device itype*, const constant int&, const device int*,                 \
      const device int*, const device uint*, const device uint*,             \
      const constant int&, const constant int&, const constant float&,       \
      const device uchar*, const device uint&, const device uchar*,          \
      const constant int&, const device itype*, const constant int&,         \
      device float*, uint3, uint3, uint, uint);

instantiate_sdpa_paged_mma_lse(bfloat16, bfloat, 64, 16)

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

  const int mask_word = live ? int(attention_mask_enabled[my_row]) : 0;
  const bool masked  = mask_word != 0;
  const bool wide    = mask_word == 2;
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
    bool wide_here = false;
    for (int i = sub; i < sub_hi; i++) {
      const int p = position_ids[row_lo + i];
      kp_hi = max(kp_hi, p);
      kp_lo = min(kp_lo, (window > 0 && p >= window) ? (p - window + 1) : 0);
      wide_here = wide_here || attention_mask_enabled[row_lo + i] == 2;
    }
    const int page_base = int(kv_page_indptr[r]);

    if (wide_here) {
      kp_hi = int(kv_page_indptr[r + 1] - uint(page_base)) * page_size - 1;
    }

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
            keep = (wide || kp <= q_pos) && kp >= my_start;
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

// A wide head stages a 16-key block as bf16 rows padded by eight, the
// values taking the keys' place once the scores are in. A 256-wide head's
// queries stage beside them; a 512-wide head's stay in registers.
template <int D> constexpr int sdpa_wide_ld() { return D + 8; }
template <int D> constexpr bool sdpa_wide_qreg() { return D > 256; }
template <int D> constexpr int sdpa_wide_stage() {
  return (sdpa_wide_qreg<D>() ? 0 : 32 * sdpa_wide_ld<D>()) + 16 * sdpa_wide_ld<D>();
}

// A lane's share of a 16-key block of one kv head, sixteen bytes (eight bf16)
// a load; rows past `cnt` are zero.
template <int D, int KT, int NT> constexpr int sdpa_wide_loads() { return KT * (D / 8) / NT; }

template <int D, int KT, int NT>
METAL_FUNC void sdpa_wide_fetch(
    thread uint4* w,
    const device bfloat* pages,
    const device uint* kv_page_indices,
    const int page_base,
    const int page_size,
    const int n_kv_heads,
    const int kv_head,
    const int base,
    const int cnt,
    const uint lid) {
  constexpr int CH = D / 8;
  static_assert(KT * CH % NT == 0, "every lane loads the same share");
  _Pragma("clang loop unroll(full)")
  for (int l = 0; l < sdpa_wide_loads<D, KT, NT>(); l++) {
    const int e = int(lid) + l * NT;
    const int kk = e / CH;
    const int d0 = (e - kk * CH) * 8;
    w[l] = uint4(0);
    if (kk < cnt) {
      const int kp = base + kk;
      const int page = int(kv_page_indices[page_base + kp / page_size]);
      const size_t slot = size_t(page) * page_size + size_t(kp % page_size);
      w[l] = *reinterpret_cast<const device uint4*>(pages + (slot * n_kv_heads + kv_head) * D + d0);
    }
  }
}

template <int D, int KT, int LD, int NT>
METAL_FUNC void sdpa_wide_land(threadgroup bfloat* dst, const thread uint4* w, const uint lid) {
  constexpr int CH = D / 8;
  _Pragma("clang loop unroll(full)")
  for (int l = 0; l < sdpa_wide_loads<D, KT, NT>(); l++) {
    const int e = int(lid) + l * NT;
    const int kk = e / CH;
    const int d0 = (e - kk * CH) * 8;
    *reinterpret_cast<threadgroup uint4*>(dst + kk * LD + d0) = w[l];
  }
}

// The simdgroup-matrix prompt read for wide heads (256 and 512). A threadgroup
// owns 32 query rows of one head; its 4 x WN simdgroups split the rows in
// eights and the head in 128-wide slices. Each simdgroup scores its slice of
// the queries against a 16-key block staged in threadgroup memory, the
// slices' partial scores are summed through the same memory before the online
// softmax, and each simdgroup accumulates its own slice of the output, in
// f32. MLX's steel attention splits a 256-wide head the same way.
template <typename T, int D, bool WITH_LSE>
inline void sdpa_paged_mma_wide_body(
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
    const int n_rows,
    device float* lse,
    threadgroup bfloat* kv,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid) {
  constexpr int QT  = 32;
  constexpr int KT  = 16;
  constexpr int WM  = 4;
  constexpr int DS  = 128;
  constexpr int WN  = D / DS;
  constexpr int NT  = WM * WN * 32;
  constexpr int SF  = DS / 8;
  constexpr int KF  = KT / 8;
  constexpr int LD  = sdpa_wide_ld<D>();
  constexpr bool QREG = sdpa_wide_qreg<D>();
  constexpr float NEG_INF = -3.0e38f;

  static_assert(D % DS == 0 && WN >= 1 && WN <= 4, "a head splits in 128-wide slices");
  static_assert(
      WM * WN * 32 * KF * 2 * sizeof(float) <= KT * LD * sizeof(bfloat),
      "the score exchange fits in the key stage");

  const int q_head    = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int kv_head   = q_head / gqa_factor;
  // The last tiles read the most keys under the causal mask: they go first.
  const int row_lo    = int(tpg.y - 1u - tid.y) * QT;
  const uint lid      = simd_gid * 32u + simd_lid;
  const int rg        = int(simd_gid) / WN;
  const int part      = int(simd_gid) % WN;

  const short qid = short(simd_lid) / 4;
  const short fm  = (qid & 4) + ((short(simd_lid) / 2) % 4);
  const short fn  = (qid & 2) * 2 + (short(simd_lid) % 2) * 2;

  const int my_row = row_lo + rg * 8 + int(fm);
  const bool live  = my_row < n_rows;

  threadgroup bfloat* qs = kv + KT * LD;
  simdgroup_matrix<bfloat, 8, 8> Qf[QREG ? SF : 1];
  if (QREG) {
    const device T* qp = queries + (size_t(live ? my_row : 0) * n_q_heads + q_head) * D +
                         part * DS + fn;
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < SF; i++) {
      thread auto& e = Qf[QREG ? i : 0].thread_elements();
      e[0] = live ? qp[i * 8 + 0] : bfloat(0);
      e[1] = live ? qp[i * 8 + 1] : bfloat(0);
    }
  } else {
    for (int e = int(lid); e < QT * D; e += NT) {
      const int r = e / D;
      const int d = e - r * D;
      const int gr = row_lo + r;
      qs[r * LD + d] = gr < n_rows ? queries[(size_t(gr) * n_q_heads + q_head) * D + d] : bfloat(0);
    }
  }

  const int q_pos    = live ? position_ids[my_row] : 0;
  const int my_start = (window > 0 && q_pos >= window) ? (q_pos - window + 1) : 0;

  const int mask_word = live ? int(attention_mask_enabled[my_row]) : 0;
  const bool masked  = mask_word != 0;
  const bool wide    = mask_word == 2;

  simdgroup_matrix<float, 8, 8> Of[SF];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < SF; i++) Of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float max_score = NEG_INF;
  float sum_exp = 0.0f;

  threadgroup float* sx = reinterpret_cast<threadgroup float*>(kv);

  int sub = 0;
  while (sub < QT && row_lo + sub < n_rows) {
    const int r = req_of_token[row_lo + sub];
    int sub_hi = sub + 1;
    while (sub_hi < QT && row_lo + sub_hi < n_rows && req_of_token[row_lo + sub_hi] == r)
      sub_hi++;

    int kp_hi = 0;
    int kp_lo = 0x7fffffff;
    bool wide_here = false;
    for (int i = sub; i < sub_hi; i++) {
      const int p = position_ids[row_lo + i];
      kp_hi = max(kp_hi, p);
      kp_lo = min(kp_lo, (window > 0 && p >= window) ? (p - window + 1) : 0);
      wide_here = wide_here || attention_mask_enabled[row_lo + i] == 2;
    }
    const int page_base = int(kv_page_indptr[r]);
    if (wide_here) {
      kp_hi = int(kv_page_indptr[r + 1] - uint(page_base)) * page_size - 1;
    }
    // Only this run's rows: a request's rows may come back in a later run of
    // the tile, which reads its keys again.
    const int local = my_row - row_lo;
    const bool mine = live && local >= sub && local < sub_hi;

    constexpr int NL = sdpa_wide_loads<D, KT, NT>();

    for (int base = kp_lo; base <= kp_hi; base += KT) {
      const int cnt = min(KT, kp_hi + 1 - base);

      // Stage the block's keys as rows: the score multiply reads them
      // transposed.
      {
        uint4 kw[NL];
        sdpa_wide_fetch<D, KT, NT>(
            kw, k_pages, kv_page_indices, page_base, page_size, n_kv_heads, kv_head, base, cnt, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sdpa_wide_land<D, KT, LD, NT>(kv, kw, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      simdgroup_matrix<float, 8, 8> S[KF];
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < KF; c++) S[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < SF; i++) {
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KF; c++) {
          simdgroup_matrix<bfloat, 8, 8> Bk;
          simdgroup_load(Bk, kv, LD, ulong2(uint(part * DS + i * 8), uint(c * 8)), true);
          if (QREG) {
            simdgroup_multiply_accumulate(S[c], Qf[QREG ? i : 0], Bk, S[c]);
          } else {
            simdgroup_matrix<bfloat, 8, 8> Qm;
            simdgroup_load(Qm, qs, LD, ulong2(uint(part * DS + i * 8), uint(rg * 8)), false);
            simdgroup_multiply_accumulate(S[c], Qm, Bk, S[c]);
          }
        }
      }

      float sv[KF * 2];
      if (WN > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* slot = sx + simd_gid * (32 * KF * 2) + simd_lid * (KF * 2);
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KF; c++) {
          thread auto& e = S[c].thread_elements();
          slot[c * 2 + 0] = e[0];
          slot[c * 2 + 1] = e[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KF; c++) {
          thread auto& e = S[c].thread_elements();
          sv[c * 2 + 0] = 0.0f;
          sv[c * 2 + 1] = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (int p = 0; p < WN; p++) {
            const threadgroup float* peer =
                sx + (rg * WN + p) * (32 * KF * 2) + simd_lid * (KF * 2);
            sv[c * 2 + 0] += p == part ? float(e[0]) : peer[c * 2 + 0];
            sv[c * 2 + 1] += p == part ? float(e[1]) : peer[c * 2 + 1];
          }
        }
      } else {
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KF; c++) {
          thread auto& e = S[c].thread_elements();
          sv[c * 2 + 0] = e[0];
          sv[c * 2 + 1] = e[1];
        }
      }

      float lmax = NEG_INF;
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < KF; c++) {
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 2; j++) {
          sv[c * 2 + j] *= scale;
          const int kk = c * 8 + int(fn) + j;
          bool keep = mine && kk < cnt;
          if (keep) {
            const int kp = base + kk;
            keep = (wide || kp <= q_pos) && kp >= my_start;
            if (keep && masked) {
              keep = !(uint(kp) >= attention_mask_stride ||
                       attention_mask[size_t(my_row) * attention_mask_stride + uint(kp)] == 0);
            }
          }
          if (!keep) sv[c * 2 + j] = NEG_INF;
          lmax = max(lmax, sv[c * 2 + j]);
        }
      }
      lmax = max(lmax, simd_shuffle_xor(lmax, 1u));
      lmax = max(lmax, simd_shuffle_xor(lmax, 8u));

      const float new_max = max(max_score, lmax);
      const float factor = max_score == NEG_INF ? 0.0f : fast::exp(max_score - new_max);
      simdgroup_matrix<float, 8, 8> P[KF];
      float lsum = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int c = 0; c < KF; c++) {
        thread auto& e = P[c].thread_elements();
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 2; j++) {
          const float p = sv[c * 2 + j] == NEG_INF ? 0.0f : fast::exp(sv[c * 2 + j] - new_max);
          e[j] = p;
          lsum += p;
        }
      }
      lsum += simd_shuffle_xor(lsum, 1u);
      lsum += simd_shuffle_xor(lsum, 8u);
      max_score = new_max;
      sum_exp = sum_exp * factor + lsum;
      _Pragma("clang loop unroll(full)")
      for (int n = 0; n < SF; n++) {
        thread auto& e = Of[n].thread_elements();
        e[0] *= factor;
        e[1] *= factor;
      }

      // The values take the keys' place.
      {
        uint4 vw[NL];
        sdpa_wide_fetch<D, KT, NT>(
            vw, v_pages, kv_page_indices, page_base, page_size, n_kv_heads, kv_head, base, cnt, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sdpa_wide_land<D, KT, LD, NT>(kv, vw, lid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      _Pragma("clang loop unroll(full)")
      for (int n = 0; n < SF; n++) {
        _Pragma("clang loop unroll(full)")
        for (int c = 0; c < KF; c++) {
          simdgroup_matrix<bfloat, 8, 8> Bv;
          simdgroup_load(Bv, kv, LD, ulong2(uint(part * DS + n * 8), uint(c * 8)), false);
          simdgroup_multiply_accumulate(Of[n], P[c], Bv, Of[n]);
        }
      }
    }
    sub = sub_hi;
  }

  if (!live) return;

  if constexpr (WITH_LSE) {
    if (part == 0 && fn == 0) {
      lse[size_t(my_row) * size_t(n_q_heads) + size_t(q_head)] =
          sdpa_lse_base2(max_score, sum_exp);
    }
  }

  device T* op = out + (size_t(my_row) * n_q_heads + q_head) * D + part * DS;
  _Pragma("clang loop unroll(full)")
  for (int n = 0; n < SF; n++) {
    thread auto& e = Of[n].thread_elements();
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 2; j++) {
      const float x = e[j];
      op[n * 8 + int(fn) + j] = static_cast<T>(sum_exp == 0.0f ? x : x / sum_exp);
    }
  }
}

#define sdpa_paged_mma_wide_args(itype)                                      \
    const device itype* queries     [[buffer(0)]],                           \
    const device itype* k_pages     [[buffer(1)]],                           \
    const device itype* v_pages     [[buffer(2)]],                           \
    device itype* out               [[buffer(3)]],                           \
    const constant int& gqa_factor             [[buffer(4)]],                \
    const device int* position_ids             [[buffer(5)]],                \
    const device int* req_of_token             [[buffer(6)]],                \
    const device uint* kv_page_indices         [[buffer(7)]],                \
    const device uint* kv_page_indptr          [[buffer(8)]],                \
    const constant int& page_size              [[buffer(9)]],                \
    const constant int& n_kv_heads             [[buffer(10)]],               \
    const constant float& scale                [[buffer(11)]],               \
    const device uchar* attention_mask         [[buffer(12)]],               \
    const device uint& attention_mask_stride   [[buffer(13)]],               \
    const device uchar* attention_mask_enabled [[buffer(14)]],               \
    const constant int& window                 [[buffer(15)]],               \
    const device itype* sinks                  [[buffer(16)]],               \
    const constant int& n_rows                 [[buffer(17)]]

template <typename T, int D>
[[kernel]] [[max_total_threads_per_threadgroup(D / 128 * 128)]] void sdpa_paged_mma_wide(
    sdpa_paged_mma_wide_args(T),
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup bfloat kv[sdpa_wide_stage<D>()];
  (void)sinks;
  sdpa_paged_mma_wide_body<T, D, false>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      n_rows, nullptr, kv, tid, tpg, simd_gid, simd_lid);
}

template <typename T, int D>
[[kernel]] [[max_total_threads_per_threadgroup(D / 128 * 128)]] void sdpa_paged_mma_wide_lse(
    sdpa_paged_mma_wide_args(T),
    device float* lse                          [[buffer(18)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint3 tpg       [[threadgroups_per_grid]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]],
    uint simd_lid   [[thread_index_in_simdgroup]]) {
  threadgroup bfloat kv[sdpa_wide_stage<D>()];
  (void)sinks;
  sdpa_paged_mma_wide_body<T, D, true>(
      queries, k_pages, v_pages, out, gqa_factor, position_ids, req_of_token,
      kv_page_indices, kv_page_indptr, page_size, n_kv_heads, scale,
      attention_mask, attention_mask_stride, attention_mask_enabled, window,
      n_rows, lse, kv, tid, tpg, simd_gid, simd_lid);
}

#define sdpa_paged_mma_wide_types(itype)                                     \
      const device itype*, const device itype*, const device itype*,         \
      device itype*, const constant int&, const device int*,                 \
      const device int*, const device uint*, const device uint*,             \
      const constant int&, const constant int&, const constant float&,       \
      const device uchar*, const device uint&, const device uchar*,          \
      const constant int&, const device itype*, const constant int&

#define instantiate_sdpa_paged_mma_wide(name, itype, d)                       \
  template [[host_name("sdpa_paged_mma_wide_" #name "_d_" #d)]]              \
  [[kernel]] void sdpa_paged_mma_wide<itype, d>(                             \
      sdpa_paged_mma_wide_types(itype), uint3, uint3, uint, uint);           \
  template [[host_name("sdpa_paged_mma_wide_lse_" #name "_d_" #d)]]          \
  [[kernel]] void sdpa_paged_mma_wide_lse<itype, d>(                         \
      sdpa_paged_mma_wide_types(itype), device float*, uint3, uint3, uint, uint);

instantiate_sdpa_paged_mma_wide(bfloat16, bfloat, 256)
instantiate_sdpa_paged_mma_wide(bfloat16, bfloat, 512)

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

constexpr constant int kMaskNone = 0;
constexpr constant int kMaskBias = 1;
constexpr constant int kMaskCustom = 2;

// The custom mask keeps segment s's pair (qi, ki), each indexed within the
// segment, where bit qi * kv_len + ki of its bits, from byte mask_indptr[s],
// is set (kernels-cuda's fa2 custom-mask arm).
inline bool custom_keeps(const device uchar* bits, const device int* mask_indptr, int s,
                         int qi, int ki, int kv_len) {
  const ulong cell = ulong(qi) * ulong(kv_len) + ulong(ki);
  return ((bits[size_t(mask_indptr[s]) + size_t(cell >> 3)] >> uint(cell & 7ul)) & 1u) != 0u;
}

inline int segment_of(const device int* indptr, int segments, int row) {
  const int first = indptr[0];
  const int total = indptr[segments];
  if (row < first || row >= total) return -1;
  int lo = 0;
  int hi = segments - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (indptr[mid] <= row) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

template <typename T, int HEAD_DIM_MAX, int SIMDS, int MASK>
[[kernel]] void ragged_scalar(
    const device T* q                 [[buffer(0)]],
    const device T* k                 [[buffer(1)]],
    const device T* v                 [[buffer(2)]],
    device T* o                       [[buffer(3)]],
    const device int* q_indptr        [[buffer(4)]],
    const device int* kv_indptr       [[buffer(5)]],
    const constant int& segments      [[buffer(6)]],
    const constant int& num_q_heads   [[buffer(7)]],
    const constant int& num_kv_heads  [[buffer(8)]],
    const constant int& head_dim      [[buffer(9)]],
    const constant float& sm_scale    [[buffer(10)]],
    const device float* bias          [[buffer(11)]],
    const constant int& max_len       [[buffer(12)]],
    const device uchar* bits          [[buffer(13)]],
    const device int* mask_indptr     [[buffer(14)]],
    uint3 tgid     [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int VPT = HEAD_DIM_MAX / 32;
  constexpr int THREADS = SIMDS * 32;
  constexpr float NEG_INF = -3.0e38f;

  const int head = int(tgid.x);
  const int row = int(tgid.y);
  const int lane = int(simd_lid);
  const int warp = int(simd_gid);
  const int tid = warp * 32 + lane;

  threadgroup float q_s[HEAD_DIM_MAX];
  threadgroup float wacc[SIMDS * HEAD_DIM_MAX];
  threadgroup float wm[SIMDS];
  threadgroup float wl[SIMDS];
  threadgroup int span[4];

  if (tid == 0) {
    const int s = segment_of(q_indptr, segments, row);
    span[0] = s < 0 ? 0 : kv_indptr[s];
    span[1] = s < 0 ? 0 : kv_indptr[s + 1];
    span[2] = s < 0 ? 0 : q_indptr[s];
    span[3] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int begin = span[0];
  const int end = span[1];
  const int q_begin = span[2];
  const int segment = span[3];

  device T* out =
      o + (size_t(row) * size_t(num_q_heads) + size_t(head)) * size_t(head_dim);
  if (end <= begin) {

    for (int d = tid; d < head_dim; d += THREADS) {
      out[d] = static_cast<T>(0.0f);
    }
    return;
  }

  const device T* q_row =
      q + (size_t(row) * size_t(num_q_heads) + size_t(head)) * size_t(head_dim);
  for (int d = tid; d < head_dim; d += THREADS) {
    q_s[d] = float(q_row[d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int group = num_q_heads / num_kv_heads;
  const int kv_head = head / group;
  const int span_len = MASK == kMaskBias ? 2 * max_len - 1 : 0;
  const int qi = row - q_begin;

  thread float acc[VPT];
  for (int u = 0; u < VPT; ++u) {
    acc[u] = 0.0f;
  }
  float running_max = NEG_INF;
  float running_sum = 0.0f;

  for (int j = begin + warp; j < end; j += SIMDS) {
    // A whole simdgroup walks one key, so skipping it stays uniform.
    if (MASK == kMaskCustom &&
        !custom_keeps(bits, mask_indptr, segment, qi, j - begin, end - begin)) {
      continue;
    }
    const device T* k_row =
        k + (size_t(j) * size_t(num_kv_heads) + size_t(kv_head)) * size_t(head_dim);
    float dot = 0.0f;
    for (int u = 0; u < VPT; ++u) {
      const int d = lane + u * 32;
      if (d < head_dim) {
        dot += q_s[d] * float(k_row[d]);
      }
    }
    dot = simd_sum(dot);

    float score = dot * sm_scale;
    if (MASK == kMaskBias) {

      const int at = clamp(j - begin - qi + max_len - 1, 0, span_len - 1);
      score += bias[size_t(head) * size_t(span_len) + size_t(at)];
    }

    const float widened = max(running_max, score);
    const float rescale = fast::exp(running_max - widened);
    const float weight = fast::exp(score - widened);

    const device T* v_row =
        v + (size_t(j) * size_t(num_kv_heads) + size_t(kv_head)) * size_t(head_dim);
    for (int u = 0; u < VPT; ++u) {
      const int d = lane + u * 32;
      if (d < head_dim) {
        acc[u] = acc[u] * rescale + weight * float(v_row[d]);
      }
    }
    running_sum = running_sum * rescale + weight;
    running_max = widened;
  }

  if (lane == 0) {
    wm[warp] = running_max;
    wl[warp] = running_sum;
  }
  for (int u = 0; u < VPT; ++u) {
    const int d = lane + u * 32;
    if (d < head_dim) {
      wacc[warp * HEAD_DIM_MAX + d] = acc[u];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float folded_max = NEG_INF;
  for (int w = 0; w < SIMDS; ++w) {
    folded_max = max(folded_max, wm[w]);
  }
  float denominator = 0.0f;
  for (int w = 0; w < SIMDS; ++w) {
    denominator += wl[w] * fast::exp(wm[w] - folded_max);
  }
  const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;

  for (int d = tid; d < head_dim; d += THREADS) {
    float sum = 0.0f;
    for (int w = 0; w < SIMDS; ++w) {
      sum += wacc[w * HEAD_DIM_MAX + d] * fast::exp(wm[w] - folded_max);
    }
    out[d] = static_cast<T>(sum * inv);
  }
}

#define instantiate_ragged_scalar(name, itype, d, mask, msfx)              \
  template [[host_name("ragged_scalar_" #name "_d_" #d "_" msfx)]]         \
  [[kernel]] void ragged_scalar<itype, d, 4, mask>(                        \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const device int*, const device int*,                 \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, const constant float&, const device float*,     \
      const constant int&, const device uchar*, const device int*,         \
      uint3, uint, uint);

#define instantiate_ragged_scalar_masks(name, itype, d)                    \
  instantiate_ragged_scalar(name, itype, d, kMaskNone, "none")             \
  instantiate_ragged_scalar(name, itype, d, kMaskBias, "bias")             \
  instantiate_ragged_scalar(name, itype, d, kMaskCustom, "custom")

instantiate_ragged_scalar_masks(bfloat16, bfloat, 32)
instantiate_ragged_scalar_masks(bfloat16, bfloat, 64)
instantiate_ragged_scalar_masks(bfloat16, bfloat, 128)
instantiate_ragged_scalar_masks(bfloat16, bfloat, 256)

template <typename T, int D, int KT, int MASK>
[[kernel]] [[max_total_threads_per_threadgroup(128)]] void ragged_mma(
    const device T* q                 [[buffer(0)]],
    const device T* k                 [[buffer(1)]],
    const device T* v                 [[buffer(2)]],
    device T* o                       [[buffer(3)]],
    const device int* q_indptr        [[buffer(4)]],
    const device int* kv_indptr       [[buffer(5)]],
    const constant int& segments      [[buffer(6)]],
    const constant int& num_kv_heads  [[buffer(7)]],
    const constant float& sm_scale    [[buffer(8)]],
    const device float* bias          [[buffer(9)]],
    const constant int& max_len       [[buffer(10)]],
    const constant int& n_rows        [[buffer(11)]],
    const device uchar* bits          [[buffer(12)]],
    const device int* mask_indptr     [[buffer(13)]],
    uint3 tid      [[threadgroup_position_in_grid]],
    uint3 tpg      [[threadgroups_per_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int QT = 32;
  constexpr int SGS = 4;
  constexpr int RPS = QT / SGS;
  constexpr int DF = D / 8;
  constexpr int KF = KT / 8;
  constexpr float NEG_INF = -3.0e38f;

  static_assert(D % 8 == 0 && KT % 8 == 0, "the matrix unit tiles in eights");
  static_assert(RPS == 8, "a simdgroup owns exactly one fragment row of queries");

  threadgroup half qtile[QT * D];
  threadgroup half ktile[D * KT];
  threadgroup half vtile[KT * D];

  const int q_head = int(tid.x);
  const int n_q_heads = int(tpg.x);
  const int group = n_q_heads / num_kv_heads;
  const int kv_head = q_head / group;
  const int row_lo = int(tid.y) * QT;
  const uint lid = simd_gid * 32u + simd_lid;

  const short qid = short(simd_lid) / 4;
  const short fm = (qid & 4) + ((short(simd_lid) / 2) % 4);
  const short fn = (qid & 2) * 2 + (short(simd_lid) % 2) * 2;

  const int my_row = row_lo + int(simd_gid) * RPS + int(fm);
  const bool live = my_row < n_rows;

  for (uint e = lid; e < uint(QT * D); e += 128u) {
    const int r = int(e) / D;
    const int d = int(e) - r * D;
    const int gr = row_lo + r;
    qtile[e] = gr < n_rows
                   ? half(float(q[(size_t(gr) * n_q_heads + q_head) * D + d]) * sm_scale)
                   : half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_matrix<half, 8, 8> Qf[DF];
  for (int i = 0; i < DF; i++) {
    simdgroup_load(Qf[i], qtile, D, ulong2(uint(i * 8), uint(simd_gid) * RPS), false);
  }

  const int my_segment = live ? segment_of(q_indptr, segments, my_row) : -1;
  const int span_len = MASK == kMaskBias ? 2 * max_len - 1 : 0;

  float Ov[DF * 2];
  for (int i = 0; i < DF * 2; i++) Ov[i] = 0.0f;
  float max_score = NEG_INF;
  float sum_exp = 0.0f;

  int sub = 0;
  while (sub < QT && row_lo + sub < n_rows) {
    const int s = segment_of(q_indptr, segments, row_lo + sub);
    int sub_hi = sub + 1;
    while (sub_hi < QT && row_lo + sub_hi < n_rows &&
           segment_of(q_indptr, segments, row_lo + sub_hi) == s) {
      sub_hi++;
    }
    if (s < 0) {
      sub = sub_hi;
      continue;
    }
    const int kv_begin = kv_indptr[s];
    const int kv_end = kv_indptr[s + 1];
    const int q_begin = q_indptr[s];
    const bool mine = live && my_segment == s;
    const int qi = live ? my_row - q_begin : 0;

    for (int base = kv_begin; base < kv_end; base += KT) {
      const int cnt = min(KT, kv_end - base);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lid; e < uint(KT * D); e += 128u) {
        const int kk = int(e) / D;
        const int d = int(e) - kk * D;
        if (kk < cnt) {
          const size_t off =
              (size_t(base + kk) * size_t(num_kv_heads) + size_t(kv_head)) * size_t(D) +
              size_t(d);
          ktile[d * KT + kk] = half(float(k[off]));
          vtile[e] = half(float(v[off]));
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
          const bool keep =
              mine && kk < cnt &&
              (MASK != kMaskCustom ||
               custom_keeps(bits, mask_indptr, s, qi, base + kk - kv_begin, kv_end - kv_begin));
          float s_val = 0.0f;
          if (keep) {
            s_val = float(e[j]);
            if (MASK == kMaskBias) {
              const int at =
                  clamp(base + kk - kv_begin - qi + max_len - 1, 0, span_len - 1);
              s_val += bias[size_t(q_head) * size_t(span_len) + size_t(at)];
            }
          }
          const float s = keep ? s_val : NEG_INF;
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
          const float p =
              sv[c * 2 + j] == NEG_INF ? 0.0f : fast::exp(sv[c * 2 + j] - new_max);
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

  device T* op = o + (size_t(my_row) * size_t(n_q_heads) + size_t(q_head)) * size_t(D);
  for (int n = 0; n < DF; n++) {
    for (int j = 0; j < 2; j++) {
      const float x = Ov[n * 2 + j];
      op[n * 8 + int(fn) + j] = static_cast<T>(sum_exp == 0.0f ? 0.0f : x / sum_exp);
    }
  }
}

#define instantiate_ragged_mma(name, itype, d, kt, mask, msfx)             \
  template [[host_name("ragged_mma_" #name "_d_" #d "_" msfx)]]            \
  [[kernel]] void ragged_mma<itype, d, kt, mask>(                          \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const device int*, const device int*,                 \
      const constant int&, const constant int&, const constant float&,     \
      const device float*, const constant int&, const constant int&,       \
      const device uchar*, const device int*, uint3, uint3, uint, uint);

#define instantiate_ragged_mma_masks(name, itype, d, kt)                   \
  instantiate_ragged_mma(name, itype, d, kt, kMaskNone, "none")            \
  instantiate_ragged_mma(name, itype, d, kt, kMaskBias, "bias")            \
  instantiate_ragged_mma(name, itype, d, kt, kMaskCustom, "custom")

instantiate_ragged_mma_masks(bfloat16, bfloat, 64, 16)
instantiate_ragged_mma_masks(bfloat16, bfloat, 128, 16)

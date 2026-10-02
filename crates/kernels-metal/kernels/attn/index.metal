#include <metal_stdlib>
using namespace metal;


constant constexpr int kIndexBlock = 256;

constant constexpr int kIndexSimds = kIndexBlock / 32;

constant constexpr int kMaxRopeDim = 256;

// `index_topk`'s scorer for a handful of heads: lanes that share one key, and
// the widest key it holds a register slice of.
constant constexpr int kSmallHeads = 4;
constant constexpr int kSmallLanes = 8;
constant constexpr int kSmallMaxDim = 128;


inline float index_tg_sum(float v, threadgroup float* partials,
                          threadgroup float* bcast, uint simd_lane,
                          uint simd_group) {
  const float t = simd_sum(v);
  if (simd_lane == 0) partials[simd_group] = t;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float x = (simd_lane < uint(kIndexSimds)) ? partials[simd_lane] : 0.0f;
    x = simd_sum(x);
    if (simd_lane == 0) bcast[0] = x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = bcast[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

inline float index_tg_min(float v, threadgroup float* partials,
                          threadgroup float* bcast, uint simd_lane,
                          uint simd_group) {
  const float t = simd_min(v);
  if (simd_lane == 0) partials[simd_group] = t;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float x = (simd_lane < uint(kIndexSimds)) ? partials[simd_lane] : INFINITY;
    x = simd_min(x);
    if (simd_lane == 0) bcast[0] = x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = bcast[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

inline float index_tg_max(float v, threadgroup float* partials,
                          threadgroup float* bcast, uint simd_lane,
                          uint simd_group) {
  const float t = simd_max(v);
  if (simd_lane == 0) partials[simd_group] = t;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float x = (simd_lane < uint(kIndexSimds)) ? partials[simd_lane] : -INFINITY;
    x = simd_max(x);
    if (simd_lane == 0) bcast[0] = x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = bcast[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

inline void index_rope_pair(device bfloat* row, int i, int rope_dim, int pos,
                            float theta) {
  const float freq =
      precise::pow(theta, -2.0f * float(i) / float(rope_dim));
  const float ang = float(pos) * freq;
  const float c = fast::cos(ang);
  const float s = fast::sin(ang);
  const float a = float(row[2 * i]);
  const float b = float(row[2 * i + 1]);
  row[2 * i] = bfloat(a * c - b * s);
  row[2 * i + 1] = bfloat(b * c + a * s);
}

[[kernel]] void index_knorm_rope_bfloat16(
    device bfloat* idx_k          [[buffer(0)]],
    const device bfloat* w        [[buffer(1)]],
    const device bfloat* b        [[buffer(2)]],
    const device int* positions   [[buffer(3)]],
    const constant int& head_dim  [[buffer(4)]],
    const constant int& rope_dim  [[buffer(5)]],
    const constant float& theta   [[buffer(6)]],
    const constant float& eps     [[buffer(7)]],
    uint3 tgpos     [[threadgroup_position_in_grid]],
    uint3 lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const int n = int(tgpos.y);
  const int tid = int(lid.x);
  device bfloat* row = idx_k + size_t(n) * size_t(head_dim);

  threadgroup float partials[kIndexSimds];
  threadgroup float bcast[1];

  float s = 0.0f;
  for (int d = tid; d < head_dim; d += kIndexBlock) {
    s += float(row[d]);
  }
  const float mean =
      index_tg_sum(s, partials, bcast, simd_lane, simd_group) / float(head_dim);

  float vv = 0.0f;
  for (int d = tid; d < head_dim; d += kIndexBlock) {
    const float x = float(row[d]) - mean;
    vv += x * x;
  }
  const float inv = precise::rsqrt(
      index_tg_sum(vv, partials, bcast, simd_lane, simd_group) / float(head_dim) +
      eps);

  for (int d = tid; d < head_dim; d += kIndexBlock) {
    const float x = (float(row[d]) - mean) * inv;
    row[d] = bfloat(x * float(w[d]) + float(b[d]));
  }

  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  const int pos = positions[n];
  const int pairs = rope_dim / 2;
  for (int i = tid; i < pairs; i += kIndexBlock) {
    index_rope_pair(row, i, rope_dim, pos, theta);
  }
}

[[kernel]] void index_q_rope_bfloat16(
    device bfloat* idx_q          [[buffer(0)]],
    const device int* positions   [[buffer(1)]],
    const constant int& n_heads   [[buffer(2)]],
    const constant int& head_dim  [[buffer(3)]],
    const constant int& rope_dim  [[buffer(4)]],
    const constant float& theta   [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int h = int(tid.x);
  if (h >= n_heads) return;
  const int n = int(tid.y);
  device bfloat* row =
      idx_q + (size_t(n) * size_t(n_heads) + size_t(h)) * size_t(head_dim);
  const int pos = positions[n];
  const int pairs = rope_dim / 2;
  for (int i = 0; i < pairs; ++i) {
    index_rope_pair(row, i, rope_dim, pos, theta);
  }
}

[[kernel]] void index_topk_paged_bfloat16(
    const device bfloat* idx_q         [[buffer(0)]],
    const device bfloat* idx_w         [[buffer(1)]],
    const device bfloat* key_pages     [[buffer(2)]],
    const device int* positions        [[buffer(3)]],
    const device int* req_of_token     [[buffer(4)]],
    const device uint* kv_page_indices [[buffer(5)]],
    const device uint* kv_page_indptr  [[buffer(6)]],
    device float* scores               [[buffer(7)]],
    device int* selection              [[buffer(8)]],
    const constant int& H              [[buffer(9)]],
    const constant int& D              [[buffer(10)]],
    const constant int& page_size      [[buffer(11)]],
    const constant int& score_stride   [[buffer(12)]],
    const constant int& topk           [[buffer(13)]],

    const constant int& ratio          [[buffer(14)]],
    const constant int& has_weights    [[buffer(15)]],
    uint3 tgpos     [[threadgroup_position_in_grid]],
    uint3 lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const int t = int(tgpos.y);
  const int tid = int(lid.x);
  device int* srow = selection + size_t(t) * size_t(topk);

  threadgroup float partials[kIndexSimds];
  threadgroup float bcast[1];
  threadgroup int sg_count[kIndexSimds];
  threadgroup float key_partials[kIndexSimds];

  const int r = req_of_token[t];
  const int pages_first = int(kv_page_indptr[r]);
  const int stride = (ratio > 0) ? ratio : 1;
  int nkeys = (positions[t] + 1) / stride;
  if (nkeys > score_stride) nkeys = score_stride;
  if (nkeys < 0) nkeys = 0;

  // Every complete block is named, so there is nothing to rank and nothing to
  // score: below the budget this kernel is only the identity selection.
  if (nkeys <= topk) {
    for (int n = tid; n < topk; n += kIndexBlock) {
      srow[n] = (n < nkeys) ? n : -1;
    }
    return;
  }

  device float* frow = scores + size_t(t) * size_t(score_stride);
  const device bfloat* qi = idx_q + size_t(t) * size_t(H) * size_t(D);
  const device bfloat* wi = idx_w + size_t(t) * size_t(H);

  if (H <= kSmallHeads && D % (2 * kSmallLanes) == 0 && D <= kSmallMaxDim) {
    // A few heads cannot fill the lane-per-head layout below: at 4 heads, 60
    // of its 64 lanes idle and each of the others walks a whole dot product
    // alone. Here `kSmallLanes` lanes share one key, each holding a slice of
    // the width for every head, and fold the slices with shuffles.
    constexpr int kSlice = kSmallMaxDim / kSmallLanes;
    constexpr int kKeysPerPass = kIndexBlock / kSmallLanes;
    const int dpl = D / kSmallLanes;
    const int sub = int(simd_lane) % kSmallLanes;
    const int slot = tid / kSmallLanes;
    float qr[kSmallHeads][kSlice];
    float w[kSmallHeads];
    for (int h = 0; h < kSmallHeads; ++h) {
      for (int i = 0; i < kSlice; ++i) {
        qr[h][i] = (h < H && i < dpl)
                       ? float(qi[size_t(h) * size_t(D) + size_t(sub * dpl + i)])
                       : 0.0f;
      }
      w[h] = (h < H) ? ((has_weights != 0) ? float(wi[h]) : 1.0f) : 0.0f;
    }
    for (int j0 = 0; j0 < nkeys; j0 += kKeysPerPass) {
      const int j = j0 + slot;
      float part[kSmallHeads];
      for (int h = 0; h < kSmallHeads; ++h) part[h] = 0.0f;
      if (j < nkeys) {
        const int cell = (j + 1) * stride - 1;
        const int page = int(kv_page_indices[pages_first + cell / page_size]);
        const int off = cell % page_size;
        // Two bf16 a word: `dpl` is even and the row is D-aligned, so the
        // slice starts on a 4-byte boundary.
        const device uint* kw = (const device uint*)(
            key_pages + (size_t(page) * size_t(page_size) + size_t(off)) * size_t(D) +
            size_t(sub * dpl));
        for (int i = 0; i < kSlice; i += 2) {
          if (i < dpl) {
            const uint bits = kw[i / 2];
            const float k0 = as_type<float>(bits << 16);
            const float k1 = as_type<float>(bits & 0xffff0000u);
            for (int h = 0; h < kSmallHeads; ++h) {
              part[h] = fma(qr[h][i], k0, fma(qr[h][i + 1], k1, part[h]));
            }
          }
        }
      }
      // Every lane folds, the tail's idle ones with zeros: a shuffle wants
      // the whole simdgroup.
      for (int h = 0; h < kSmallHeads; ++h) {
        for (int m = 1; m < kSmallLanes; m <<= 1) {
          part[h] += simd_shuffle_xor(part[h], ushort(m));
        }
      }
      if (j < nkeys && sub == 0) {
        float acc = 0.0f;
        for (int h = 0; h < kSmallHeads; ++h) {
          acc += max(part[h], 0.0f) * w[h];
        }
        frow[j] = acc;
      }
    }
  } else {
    constexpr int kLanesPerKey = 64;
    constexpr int kKeysPerPass = kIndexBlock / kLanesPerKey;
    constexpr int kSimdsPerKey = kLanesPerKey / 32;
    const int slot = tid / kLanesPerKey;
    const int hh = tid % kLanesPerKey;
    for (int j0 = 0; j0 < nkeys; j0 += kKeysPerPass) {
      const int j = j0 + slot;
      float acc = 0.0f;
      if (j < nkeys) {
        const int cell = (j + 1) * stride - 1;
        const int page = int(kv_page_indices[pages_first + cell / page_size]);
        const int off = cell % page_size;
        const device bfloat* kj =
            key_pages + (size_t(page) * size_t(page_size) + size_t(off)) * size_t(D);
        for (int h = hh; h < H; h += kLanesPerKey) {
          const device bfloat* qh = qi + size_t(h) * size_t(D);
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            dot += float(qh[d]) * float(kj[d]);
          }
          acc += max(dot, 0.0f) * ((has_weights != 0) ? float(wi[h]) : 1.0f);
        }
      }
      const float folded = simd_sum(acc);
      if (simd_lane == 0) key_partials[simd_group] = folded;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < kKeysPerPass && j0 + tid < nkeys) {
        float total = 0.0f;
        for (int g = 0; g < kSimdsPerKey; ++g) {
          total += key_partials[tid * kSimdsPerKey + g];
        }
        frow[j0 + tid] = total;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  float lo_l = INFINITY;
  float hi_l = -INFINITY;
  for (int j = tid; j < nkeys; j += kIndexBlock) {
    lo_l = min(lo_l, frow[j]);
    hi_l = max(hi_l, frow[j]);
  }
  float lo = index_tg_min(lo_l, partials, bcast, simd_lane, simd_group);
  float hi = index_tg_max(hi_l, partials, bcast, simd_lane, simd_group);

  float thr = hi;
  for (int it = 0; it < 40; ++it) {
    const float mid = 0.5f * (lo + hi);
    // Adjacent floats: `mid` is `lo` or `hi`, and no pass can move either.
    if (mid <= lo || mid >= hi) break;
    float c = 0.0f;
    for (int j = tid; j < nkeys; j += kIndexBlock) {
      if (frow[j] >= mid) c += 1.0f;
    }

    const int cnt = int(index_tg_sum(c, partials, bcast, simd_lane, simd_group));
    if (cnt > topk) {
      lo = mid;
    } else {
      hi = mid;
    }
    thr = hi;
    // Exactly the budget at or above `hi`: any lower threshold that still
    // keeps the count within it names the same set.
    if (cnt == topk) break;
  }

  // Each thread takes a contiguous run of blocks in order, so an exclusive
  // prefix of the counts puts every id where one thread walking them all
  // would have: ascending, and the budget's first when a tie overfills it.
  const int order = int(simd_group) * 32 + int(simd_lane);
  const int chunk = (nkeys + kIndexBlock - 1) / kIndexBlock;
  const int b0 = min(order * chunk, nkeys);
  const int b1 = min(b0 + chunk, nkeys);
  int mine = 0;
  for (int j = b0; j < b1; ++j) {
    if (frow[j] >= thr) ++mine;
  }
  const int before = simd_prefix_exclusive_sum(mine);
  const int simd_total = simd_sum(mine);
  if (simd_lane == 0) sg_count[simd_group] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int n = before;
  int total = 0;
  for (int g = 0; g < kIndexSimds; ++g) {
    if (g < int(simd_group)) n += sg_count[g];
    total += sg_count[g];
  }
  for (int j = b0; j < b1 && n < topk; ++j) {
    if (frow[j] >= thr) srow[n++] = j;
  }
  for (int m = min(total, topk) + tid; m < topk; m += kIndexBlock) {
    srow[m] = -1;
  }
}

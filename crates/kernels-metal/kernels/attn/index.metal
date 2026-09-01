#include <metal_stdlib>
using namespace metal;

// The NSA lightning indexer — the metal mirror of `kernels-cuda`'s
// `attn/index.cuh`. This is the top-k SELECTION in front of the sparse
// attention: a small key cache (one contiguous `head_dim`-wide row per cached
// token) is scored against a per-token query of `H` index heads, and the
// `topk` best cached positions are published as an `i32` selection row that
// the selected mla readers then walk.
//
// The scoring statement, `index.cuh` line for line:
//
//     acc(j) = sum_h max(q_h . k_j, 0) * w_h
//
// over `H` index heads of `D` lanes each — `relu` on the per-head dot, then
// the learned per-head weight. `H = 64`, `D = 128` on dsv4-flash; the 512 the
// campaign feared is the SELECTED attention's latent rank and is no part of
// this file.
//
// # Three kernels here, and the fourth is deliberately absent
//
// `index.cuh` carries four. Three are mirrored:
//
//   * `index_knorm_rope`   -> `index_knorm_rope_bfloat16`
//   * `index_q_rope`       -> `index_q_rope_bfloat16`
//   * `index_topk_paged`   -> `index_topk_paged_bfloat16`
//
// **`index_topk_mask` IS NOT PORTED, AND THAT IS A STATEMENT ABOUT THE IR.**
// It is the dense (unpaged) variant: it takes a full `[N, D]` key matrix, a
// causal `nkeys = i + 1`, and publishes a `u8` mask row instead of an index
// row. Nothing dispatches it. The IR has ONE selection op —
// `attention.index_topk` — and its host entry on both planes fires the paged
// point, because the served cache is paged and the consumer
// (`attention.mla_*_selected`) reads an `i32` selection and not a mask plane.
// A port of it here would be a shader with no caller, compiled by the device
// census every run to prove that a thing nothing reaches still builds. If a
// mask-shaped IR op ever lands, the shape it wants is the paged kernel below
// with `mrow[j] = (frow[j] >= thr)` in place of the serial collect, and its
// `logit[nkeys]` dynamic shared array has to become a global slab here
// anyway: metal threadgroup memory is 32 KiB, which is 8192 floats, and the
// dense variant's `nkeys` is a whole sequence.
//
// # The mean-subtracting norm is not the rms norm next door
//
// `index_knorm_rope` is a TRUE LayerNorm — subtract the row mean, divide by
// the row's standard deviation, scale by `w`, shift by `b`. Every other norm
// in this crate's attention family is an rms norm with no mean and no bias
// (`mla_latents_bfloat16` is the near neighbour). Two reductions, not one.
//
// # Where this diverges from the CUDA twin, and why
//
// **The rope is spread across the threadgroup instead of run on thread 0.**
// `index.cuh` ropes from a `float buf[kMaxRopeDim]` held by thread 0 alone,
// because it needs the whole rotated prefix in registers to write it back.
// The rotation is pairwise independent — pair `i` reads and writes lanes
// `2i` and `2i+1` and nothing else — so spreading the pairs over the
// threadgroup computes the identical values from the identical inputs, with
// no staging buffer and no serial tail. The `kMaxRopeDim` ceiling is kept
// anyway (the host refuses past it) because past 256 the CUDA authority is
// undefined behaviour rather than a different answer, and a metal plane that
// quietly served a geometry its twin cannot is a parity claim nobody can
// check.
//
// **The paged selection reads the fire tables for its bound.** `index.cuh`
// re-derives each row's absolute query position from `qo_indptr` and
// `kv_last_page_lens`:
//
//     kv_len = (num_pages - 1) * page_size + last_page_lens[r]
//     abs_q  = kv_len - new_tokens + (t - qo_indptr[r])
//
// The metal pool carries no last-page table (`store::SpaceSeat` keeps one,
// but it reaches the shaders as a `RuntimeInput::Geometry` an op must name,
// and `attention.index_topk` names none), so this kernel takes the same two
// fire tables `mla_naive_paged_bfloat16` and `pool_lse_paged` already take:
// `positions[t]` IS `abs_q`, and `req_of_token[t]` IS the `r` the CUDA
// kernel finds by scanning `qo_indptr`. Same two numbers, read instead of
// rebuilt. This is the divergence `attn/mla.metal`'s header states for the
// same reason, and it is the third op to take it.
//
// **THE KEY STRIDE `ratio` IS THIS PLANE'S AND THE CUDA TWIN REFUSES IT.**
// `index.cuh` scans `j = 0 .. pos` and reads cell `j`, which is one key per
// TOKEN — glm_5's indexer, whose `index_kv_append` wrote exactly that row.
// dsv4-flash's indexer keys one row per COMPRESSED BLOCK: the reference's
// `compressor_prefill(..., rotate=True)` returns `[b, s / ratio, d]`, and
// `pool_store_entries` lands each pooled key at the boundary cell
// `(c+1)*ratio - 1` — the same cell the attention compressor's entry lands
// in its own pool. So this kernel takes the stride as a number: `nkeys =
// (pos+1)/ratio`, key `c` read at cell `(c+1)*ratio - 1`, and the published
// ids are COMPRESSED ROWS. `ratio == 1` is `index.cuh` unchanged, arithmetic
// included, which is why the CUDA arm serves that value and mechanically
// refuses the others rather than claiming a kernel it does not have.

// `pie::attn::kBlock` — the threadgroup every kernel here launches.
constant constexpr int kIndexBlock = 256;

// Simdgroups in one `kIndexBlock` threadgroup: the partials array's width.
constant constexpr int kIndexSimds = kIndexBlock / 32;

// `pie::attn::kMaxRopeDim`. Not a threadgroup bound here (see the header) —
// the ceiling the CUDA authority states, kept so the two planes refuse the
// same geometries.
constant constexpr int kMaxRopeDim = 256;

// ── the threadgroup folds ───────────────────────────────────────────────────
//
// `simd_sum` + one partial per simdgroup + a broadcast slot, the idiom
// `attn/pool.metal` and `elemwise/norm_rms.metal` use. The trailing barrier
// is what lets a caller fold twice against one `bcast` slot without the
// second fold's writer racing the first fold's readers.

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

// ── the interleaved rotation, one pair ──────────────────────────────────────
//
// `rope_interleave_inplace`'s body for a single `i`: pair `(2i, 2i+1)`,
// frequency `theta^(-2i/rope_dim)`, and the rotation
// `(a*c - b*s, b*c + a*s)`. INTERLEAVED and not neox-split — the pair is
// adjacent lanes, which is the half of `elemwise/rope_neox.metal`'s family
// this indexer shares nothing else with.
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

// ── layernorm the index key row, then rope its head ─────────────────────────
//
// One threadgroup per cached key row, in place. Mirrors
// `pie::attn::index_knorm_rope<T>`: mean, variance, affine, barrier, rope.
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
  // THE NORM MUST BE COMPLETE BEFORE THE ROTATION READS IT. The rope reads
  // lanes `[0, rope_dim)` of the SAME row a different thread just wrote —
  // `head_dim / kIndexBlock` strided passes mean lane `2i` and lane `2i+1`
  // need not have been written by the thread about to rotate them.
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  const int pos = positions[n];
  const int pairs = rope_dim / 2;
  for (int i = tid; i < pairs; i += kIndexBlock) {
    index_rope_pair(row, i, rope_dim, pos, theta);
  }
}

// ── rope the index query, per (row, head) ───────────────────────────────────
//
// One thread per `(row, head)`; each rotates its own head's `rope_dim`
// prefix. Mirrors `pie::attn::index_q_rope<T>` — which spends a thread per
// head for the same reason, and stages through a private `float buf` this one
// does not need.
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

// ── score every visible cached key, then select the top k ───────────────────
//
// One threadgroup per QUERY ROW. Mirrors `pie::attn::index_topk_paged<T>`.
//
// The selection is a THRESHOLD BISECTION and not a sort: 40 halvings of
// `[lo, hi]` counting `frow[j] >= mid`, moving `lo` up when the count still
// exceeds the budget and `hi` down when it does not, and taking `thr = hi` —
// so the published threshold is always one that admits AT MOST `topk`. The
// iteration count is part of the contract, not a tolerance: 40 halvings of a
// float range is what fixes which side of a near-tie a key lands on, and a
// plane that ran 39 or 41 would select a different set on the same input.
//
// Ties AT the threshold are broken by POSITION: thread 0 walks `j` ascending
// and takes the first `topk` keys that clear `thr`, so of two keys with the
// same score the earlier cached one wins and the tail is padded with `-1`.
// That serial collect is `index.cuh`'s, kept serial for exactly this reason —
// a parallel compaction would have to reproduce the ordering anyway.
//
// `scores` is a GLOBAL slab of `score_stride` floats per row, not threadgroup
// memory: `nkeys` is a sequence length and metal gives a threadgroup 32 KiB.
// `nkeys` is clamped to `score_stride`, which is `index.cuh`'s own clamp
// against the same slab's width.
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
    // WHICH CACHED ROWS ARE KEYS. `1` is the per-token cache glm_5 writes
    // with `index_kv_append`: key `j` at position `j`, `nkeys = pos + 1`, and
    // the published ids are positions. dsv4-flash keys one row per COMPRESSED
    // BLOCK — its indexer's own compressor pools an entry per `ratio` tokens
    // and `pool_store_entries` puts it at the boundary cell `(c+1)*ratio - 1`
    // — so key `c` is read at that cell, `nkeys = (pos + 1) / ratio`, and the
    // published ids are compressed-row indices `pool_lse_selected_paged`
    // turns back into cells by the same arithmetic.
    const constant int& ratio          [[buffer(14)]],
    uint3 tgpos     [[threadgroup_position_in_grid]],
    uint3 lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const int t = int(tgpos.y);
  const int tid = int(lid.x);
  device int* srow = selection + size_t(t) * size_t(topk);

  threadgroup float partials[kIndexSimds];
  threadgroup float bcast[1];

  // `abs_q` and `r`, read off the fire tables instead of rebuilt from
  // `qo_indptr` and `kv_last_page_lens` — the header states why.
  const int r = req_of_token[t];
  const int pages_first = int(kv_page_indptr[r]);
  const int stride = (ratio > 0) ? ratio : 1;
  int nkeys = (positions[t] + 1) / stride;
  if (nkeys > score_stride) nkeys = score_stride;
  if (nkeys < 0) nkeys = 0;

  device float* frow = scores + size_t(t) * size_t(score_stride);
  const device bfloat* qi = idx_q + size_t(t) * size_t(H) * size_t(D);
  const device bfloat* wi = idx_w + size_t(t) * size_t(H);
  for (int j = tid; j < nkeys; j += kIndexBlock) {
    const int cell = (j + 1) * stride - 1;
    const int page = int(kv_page_indices[pages_first + cell / page_size]);
    const int off = cell % page_size;
    const device bfloat* kj =
        key_pages + (size_t(page) * size_t(page_size) + size_t(off)) * size_t(D);
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      const device bfloat* qh = qi + size_t(h) * size_t(D);
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        dot += float(qh[d]) * float(kj[d]);
      }
      acc += max(dot, 0.0f) * float(wi[h]);
    }
    frow[j] = acc;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  // Everything visible fits the budget: the selection is the identity, and
  // the tail is padded rather than left holding another sequence's ids.
  if (nkeys <= topk) {
    for (int n = tid; n < topk; n += kIndexBlock) {
      srow[n] = (n < nkeys) ? n : -1;
    }
    return;
  }

  float lo_l = INFINITY;
  float hi_l = -INFINITY;
  for (int j = tid; j < nkeys; j += kIndexBlock) {
    lo_l = min(lo_l, frow[j]);
    hi_l = max(hi_l, frow[j]);
  }
  float lo = index_tg_min(lo_l, partials, bcast, simd_lane, simd_group);
  float hi = index_tg_max(hi_l, partials, bcast, simd_lane, simd_group);

  // FORTY, EXACTLY. See the header.
  float thr = hi;
  for (int it = 0; it < 40; ++it) {
    const float mid = 0.5f * (lo + hi);
    float c = 0.0f;
    for (int j = tid; j < nkeys; j += kIndexBlock) {
      if (frow[j] >= mid) c += 1.0f;
    }
    // The count is exact in `float` for every `nkeys` a cache can hold: it is
    // a sum of ones bounded by `score_stride`, and `2^24` is millions of
    // tokens past any page budget. This is the fold `atomicAdd(&cnt_s, c)`
    // is on the CUDA plane, in the idiom the neighbouring shaders use.
    const int cnt = int(index_tg_sum(c, partials, bcast, simd_lane, simd_group));
    if (cnt > topk) {
      lo = mid;
    } else {
      hi = mid;
    }
    thr = hi;
  }

  if (tid == 0) {
    int n = 0;
    for (int j = 0; j < nkeys && n < topk; ++j) {
      if (frow[j] >= thr) srow[n++] = j;
    }
    for (; n < topk; ++n) {
      srow[n] = -1;
    }
  }
}

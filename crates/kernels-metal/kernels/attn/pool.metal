#include <metal_stdlib>

using namespace metal;

// `Pool`: pooled (compressed) attention — the dsv4 compressor's KV-time-axis
// pooling, ported organ-for-organ from `kernels-cuda/kernels/attn/pool.cuh`.
// Every `ratio` tokens close a boundary whose pooled entry lands in its own
// compressed cache; a later attention reads the compressed entries and
// publishes a log-sum-exp plane the cascade merge folds against the dense
// pass.
//
// The boundary kernels emit three columns: `out_pos` (the CELL the entry is
// cached at — the block's LAST token, which is what the reader addresses at
// `(c+1)*ratio - 1`), `out_req` (the lane), and `out_rope` (the COMPRESSED
// ROW'S POSITION the entry is roped at — `(p / ratio) * ratio`, the block's
// FIRST token, the reference's `rows = arange(0, cutoff, ratio)` in
// `v4mlx/compressor.py`). The last column had no IR seat for as long as the
// gated compressor was deferred and was dropped here; the compressor fires
// now, so it is an output like the two beside it, and a text that roped at
// `out_pos` instead is off by `ratio - 1` on every compressed key.

// The paged-cell address: page-table indirection then the in-page slot, the
// twin of `pool.cuh`'s `paged_slot`.
inline size_t pool_paged_slot(
    const device uint* page_indices,
    const device uint* page_indptr,
    int req, int pos, int page_size) {
  const uint page = page_indices[page_indptr[req] + uint(pos / page_size)];
  return size_t(page) * size_t(page_size) + size_t(pos % page_size);
}

// ---- boundary kernels -----------------------------------------------------

// Marks which decode rows close a pooling boundary. `row_valid` is the
// CUDA-graph padding mask, an op-named input.
[[kernel]] void pool_boundary_decode(
    const device int* positions   [[buffer(0)]],
    device int* out_pos           [[buffer(1)]],
    device int* out_req           [[buffer(2)]],
    device int* out_rope          [[buffer(3)]],
    const constant int& n         [[buffer(4)]],
    const constant int& ratio     [[buffer(5)]],
    const device uchar* row_valid [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const int t = int(gid);
  if (t >= n) return;
  const int p = positions[t];
  const bool valid = row_valid[t] != 0;
  const bool is_boundary = valid && (((p + 1) % ratio) == 0);
  out_pos[t] = is_boundary ? p : -1;
  out_req[t] = t;
  out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;
}

// The prefill twin: boundaries within each request's ragged span. The owning
// request is a binary search over the fire's `qo_indptr`.
[[kernel]] void pool_boundary_prefill(
    const device int* positions    [[buffer(0)]],
    const device uint* qo_indptr   [[buffer(1)]],
    device int* out_pos            [[buffer(2)]],
    device int* out_req            [[buffer(3)]],
    device int* out_rope           [[buffer(4)]],
    const constant int& n          [[buffer(5)]],
    const constant int& num_requests [[buffer(6)]],
    const constant int& ratio      [[buffer(7)]],
    const device uchar* row_valid  [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  const int t = int(gid);
  if (t >= n) return;
  const int p = positions[t];
  const bool valid = row_valid[t] != 0;
  const bool is_boundary = valid && (((p + 1) % ratio) == 0);
  out_pos[t] = is_boundary ? p : -1;
  out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;

  int lo = 0;
  int hi = num_requests;
  while (lo + 1 < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (int(qo_indptr[mid]) <= t) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  out_req[t] = lo;
}

// ---- the rolling state's writer -------------------------------------------

// Scatters this fire's compressor projections into the rolling state the
// gather below pools out of: `state_kv[slot] = wkv·x` and
// `state_score[slot] = wgate·x`, at `slot = w_page[i] * page_size + w_off[i]`
// — the SOURCE cache's own cell for token row `i`, which is the cell
// `kv_append_paged` writes the latent into in the same fire.
//
// **THE STATE IS ADDRESSED BY THE CACHE AND NOT BY THE FIRE**, which is the
// whole reason this is a scatter and not a rectangle: a pooling window
// closing at this fire's boundary reaches back `coff * ratio` positions, and
// most of those tokens were written by earlier fires. `pool_paged_slot` in
// the gather and this `slot` are the same arithmetic said two ways — the
// gather has a `(req, pos)` and walks the page table, this has the write
// descriptors the fire already carries for that row.
//
// One thread per `(column, row)`; `state_pitch` is the plane's row, which is
// not always this layer's `width` (see the gather's note on two ratios in one
// artifact).
template <typename T>
[[kernel]] void pool_state_write(
    const device T* kv             [[buffer(0)]],
    const device T* score          [[buffer(1)]],
    device T* state_kv             [[buffer(2)]],
    device T* state_score          [[buffer(3)]],
    const device uint* w_page      [[buffer(4)]],
    const device uint* w_off       [[buffer(5)]],
    const constant int& width      [[buffer(6)]],
    const constant int& page_size  [[buffer(7)]],
    const constant int& state_pitch[[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int d = int(gid.x);
  const int i = int(gid.y);
  if (d >= width) return;
  const size_t slot =
      size_t(w_page[i]) * size_t(page_size) + size_t(w_off[i]);
  const size_t dst = slot * size_t(state_pitch) + size_t(d);
  const size_t src = size_t(i) * size_t(width) + size_t(d);
  state_kv[dst] = kv[src];
  state_score[dst] = score[src];
}

#define instantiate_pool_state_write(name, itype)                        \
  template [[host_name("pool_state_write_" #name)]]                      \
  [[kernel]] void pool_state_write<itype>(                               \
      const device itype*, const device itype*, device itype*,           \
      device itype*, const device uint*, const device uint*,             \
      const constant int&, const constant int&, const constant int&,     \
      uint2);

instantiate_pool_state_write(bfloat16, bfloat)

// ---- gather (the gated softmax pool) --------------------------------------

// Pools the closing `2*ratio` window out of the rolling compressor state into
// one per-boundary entry, with the learned gate: a softmax over the window's
// score plane (`state_score` + the intra-block absolute-position embedding
// `ape`) weighting the value plane (`state_kv`). One thread per (entry,
// head-dim lane); the window is walked serially per thread.
//
// `has_ape` selects whether the absolute-position plane is folded into the
// gate logits (the CUDA twin keyed on `ape != nullptr`).
template <typename T>
[[kernel]] void pool_gather_paged(
    const device T* state_kv       [[buffer(0)]],
    const device T* state_score    [[buffer(1)]],
    const device float* ape        [[buffer(2)]],
    const device int* boundary_pos [[buffer(3)]],
    const device int* boundary_req [[buffer(4)]],
    const device uint* page_indices[[buffer(5)]],
    const device uint* page_indptr [[buffer(6)]],
    device T* out                  [[buffer(7)]],
    const constant int& head_dim   [[buffer(8)]],
    const constant int& ratio      [[buffer(9)]],
    const constant int& coff       [[buffer(10)]],
    const constant int& page_size  [[buffer(11)]],
    const constant int& has_ape    [[buffer(12)]],
    // The ROW PITCH the two state slabs are laid out at, which is not always
    // `coff * head_dim`: one artifact can hold pooled layers at two ratios
    // (dsv4-flash carries ratio 4 and ratio 128), the reservation lays ONE
    // plane at the widest of them, and a narrower gather must still stride by
    // the plane's row and read its own `coff * head_dim` columns inside it.
    const constant int& state_pitch[[buffer(13)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int d = int(gid.x);
  const int c = int(gid.y);
  if (d >= head_dim) return;

  const int window = coff * ratio;
  const int width = coff * head_dim;
  const size_t pitch = size_t(state_pitch);
  const int bpos = boundary_pos[c];
  const int req = boundary_req[c];

  if (bpos < 0) {
    out[size_t(c) * size_t(head_dim) + size_t(d)] = T(0);
    return;
  }

  float max_s = -INFINITY;
  for (int i = 0; i < window; ++i) {
    const int pos = bpos + i - (window - 1);
    if (pos < 0) continue;
    const int col = ((i >= ratio) ? head_dim : 0) + d;
    const size_t slot =
        pool_paged_slot(page_indices, page_indptr, req, pos, page_size);
    float sc = float(state_score[slot * pitch + size_t(col)]);
    if (has_ape != 0) {
      sc += ape[size_t(pos % ratio) * size_t(width) + size_t(col)];
    }
    max_s = max(max_s, sc);
  }
  if (!isfinite(max_s)) {
    out[size_t(c) * size_t(head_dim) + size_t(d)] = T(0);
    return;
  }

  float sum_e = 0.0f;
  float acc = 0.0f;
  for (int i = 0; i < window; ++i) {
    const int pos = bpos + i - (window - 1);
    if (pos < 0) continue;
    const int col = ((i >= ratio) ? head_dim : 0) + d;
    const size_t slot =
        pool_paged_slot(page_indices, page_indptr, req, pos, page_size);
    float sc = float(state_score[slot * pitch + size_t(col)]);
    if (has_ape != 0) {
      sc += ape[size_t(pos % ratio) * size_t(width) + size_t(col)];
    }
    const float e = precise::exp(sc - max_s);
    sum_e += e;
    acc += e * float(state_kv[slot * pitch + size_t(col)]);
  }
  out[size_t(c) * size_t(head_dim) + size_t(d)] =
      T(sum_e > 0.0f ? acc / sum_e : 0.0f);
}

#define instantiate_pool_gather_paged(name, itype)                       \
  template [[host_name("pool_gather_paged_" #name)]]                     \
  [[kernel]] void pool_gather_paged<itype>(                              \
      const device itype*, const device itype*, const device float*,     \
      const device int*, const device int*, const device uint*,          \
      const device uint*, device itype*, const constant int&,            \
      const constant int&, const constant int&, const constant int&,     \
      const constant int&, const constant int&, uint2);

instantiate_pool_gather_paged(bfloat16, bfloat)

// ---- store ----------------------------------------------------------------

// Stores each pooled entry into its cell of the compressed cache. One thread
// per (entry, head-dim lane); a masked-out boundary (`bpos < 0`) writes
// nothing.
template <typename T>
[[kernel]] void pool_store_entries(
    const device T* entries        [[buffer(0)]],
    device T* comp_kv_pages        [[buffer(1)]],
    const device int* boundary_pos [[buffer(2)]],
    const device int* boundary_req [[buffer(3)]],
    const device uint* page_indices[[buffer(4)]],
    const device uint* page_indptr [[buffer(5)]],
    const constant int& head_dim   [[buffer(6)]],
    const constant int& page_size  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int d = int(gid.x);
  const int c = int(gid.y);
  if (d >= head_dim) return;
  if (boundary_pos[c] < 0) return;
  const size_t slot = pool_paged_slot(
      page_indices, page_indptr, boundary_req[c], boundary_pos[c], page_size);
  comp_kv_pages[slot * size_t(head_dim) + size_t(d)] =
      entries[size_t(c) * size_t(head_dim) + size_t(d)];
}

#define instantiate_pool_store_entries(name, itype)                      \
  template [[host_name("pool_store_entries_" #name)]]                    \
  [[kernel]] void pool_store_entries<itype>(                             \
      const device itype*, device itype*, const device int*,             \
      const device int*, const device uint*, const device uint*,         \
      const constant int&, const constant int&, uint2);

instantiate_pool_store_entries(bfloat16, bfloat)

// ---- attention over the compressed entries --------------------------------

constant int POOL_ATTN_BLOCK = 128;
// The widest head this threadgroup-resident flash reader holds in its q tile.
constant int POOL_HEAD_MAX = 512;

// One threadgroup per (query row, query head). The threadgroup streams the
// `num_visible = (qpos+1)/ratio` compressed keys (the entry that closes each
// window at position `(c+1)*ratio - 1`), runs an online-max / weighted-sum
// flash softmax, and publishes `o` plus the base-2 log-sum-exp column the
// cascade merge reads. Twin of `pool.cuh`'s `pool_lse_paged`.
[[kernel]] void pool_lse_paged(
    const device bfloat* q            [[buffer(0)]],
    const device bfloat* comp_kv_pages[[buffer(1)]],
    device bfloat* o                  [[buffer(2)]],
    device float* lse_out             [[buffer(3)]],
    const device int* positions       [[buffer(4)]],
    const device uint* page_indices   [[buffer(5)]],
    const device uint* page_indptr    [[buffer(6)]],
    const device int* req_of_token    [[buffer(7)]],
    const constant int& num_q_heads   [[buffer(8)]],
    const constant int& head_dim      [[buffer(9)]],
    const constant int& ratio         [[buffer(10)]],
    const constant int& page_size     [[buffer(11)]],
    const constant float& scale       [[buffer(12)]],
    uint3 tgpos    [[threadgroup_position_in_grid]],
    uint3 lid      [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group[[simdgroup_index_in_threadgroup]]) {
  const int qi = int(tgpos.y);
  const int q_head = int(tgpos.z);
  const int tid = int(lid.x);

  threadgroup float q_smem[POOL_HEAD_MAX];
  threadgroup float partials[4];  // POOL_ATTN_BLOCK / 32 simdgroups
  threadgroup float bcast[1];

  const int req = req_of_token[qi];
  const int qpos = positions[qi];
  const int num_visible = (qpos + 1) / ratio;

  const device bfloat* q_row =
      q + (size_t(qi) * size_t(num_q_heads) + size_t(q_head)) * size_t(head_dim);
  for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
    q_smem[d] = float(q_row[d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device bfloat* o_row =
      o + (size_t(qi) * size_t(num_q_heads) + size_t(q_head)) * size_t(head_dim);

  if (num_visible <= 0) {
    for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
      o_row[d] = bfloat(0.0f);
    }
    if (tid == 0) {
      lse_out[qi * num_q_heads + q_head] = -INFINITY;
    }
    return;
  }

  // Pass 1: the row max over every visible compressed key.
  float local_max = -INFINITY;
  for (int c = tid; c < num_visible; c += POOL_ATTN_BLOCK) {
    const size_t slot = pool_paged_slot(
        page_indices, page_indptr, req, (c + 1) * ratio - 1, page_size);
    const device bfloat* k_row = comp_kv_pages + slot * size_t(head_dim);
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += q_smem[d] * float(k_row[d]);
    }
    local_max = max(local_max, dot * scale);
  }
  {
    float m = simd_max(local_max);
    if (simd_lane == 0) partials[simd_group] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      float v = (simd_lane < 4) ? partials[simd_lane] : -INFINITY;
      v = simd_max(v);
      if (simd_lane == 0) bcast[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float row_max = bcast[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 2: the flash weighted sum. Each key's QK dot is reduced across the
  // whole threadgroup; the running denominator and the per-thread output
  // accumulators (one dim-tile per thread) build up over the keys.
  const int dims_per_thread = (head_dim + POOL_ATTN_BLOCK - 1) / POOL_ATTN_BLOCK;
  float acc[8] = {0};
  float local_z = 0.0f;

  for (int c = 0; c < num_visible; ++c) {
    const size_t slot = pool_paged_slot(
        page_indices, page_indptr, req, (c + 1) * ratio - 1, page_size);
    const device bfloat* k_row = comp_kv_pages + slot * size_t(head_dim);
    float dot = 0.0f;
    for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
      dot += q_smem[d] * float(k_row[d]);
    }
    float s = simd_sum(dot);
    if (simd_lane == 0) partials[simd_group] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      float v = (simd_lane < 4) ? partials[simd_lane] : 0.0f;
      v = simd_sum(v);
      if (simd_lane == 0) bcast[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float w = precise::exp(bcast[0] * scale - row_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_z += w;
    for (int i = 0; i < dims_per_thread; ++i) {
      const int d = tid + i * POOL_ATTN_BLOCK;
      if (d < head_dim) {
        acc[i] += w * float(k_row[d]);
      }
    }
  }

  const float inv_z = local_z > 0.0f ? 1.0f / local_z : 0.0f;
  if (tid == 0) {
    const float kLog2e = 1.44269504088896340736f;
    lse_out[qi * num_q_heads + q_head] =
        local_z > 0.0f ? ((precise::log(local_z) + row_max) * kLog2e) : -INFINITY;
  }
  for (int i = 0; i < dims_per_thread; ++i) {
    const int d = tid + i * POOL_ATTN_BLOCK;
    if (d < head_dim) {
      o_row[d] = bfloat(acc[i] * inv_z);
    }
  }
}

// ---- the selected reader: the same attention over a chosen subset ---------
//
// **THE NSA FINE BRANCH.** `pool_lse_paged` above attends every compressed
// row its query can see (`num_visible = (qpos+1)/ratio`); this one attends
// only the rows `attention.index_topk` chose, `selection[qi*top_k + n]`,
// ascending with `-1` padding the tail of a row that saw fewer keys than its
// budget. Everything else is the same kernel: the same `(c+1)*ratio - 1`
// cell arithmetic, the same two-pass online softmax, the same base-2
// log-sum-exp `merge_lse` folds against the sliding-window branch, and the
// same per-head `attn_sink` closing the merged pair downstream.
//
// **IT REDUCES TO THE DENSE READER EXACTLY.** `index_topk_paged`'s
// `nkeys <= topk` arm publishes the identity `0..nkeys-1`, so a row inside
// its budget walks the same keys in the same order and accumulates the same
// sum. That is what makes the selected branch safe to fire on short
// sequences, and it is the deviceless equality the tests pin.
//
// A selected id is bounded by the query's own visible count as well as by
// the plane: an id at or past `num_visible` is SKIPPED rather than clamped,
// which is `mla_naive_paged`'s `j < 0 || j >= j_end` guard in this geometry.
[[kernel]] void pool_lse_selected_paged(
    const device bfloat* q            [[buffer(0)]],
    const device bfloat* comp_kv_pages[[buffer(1)]],
    const device int* selection       [[buffer(2)]],
    device bfloat* o                  [[buffer(3)]],
    device float* lse_out             [[buffer(4)]],
    const device int* positions       [[buffer(5)]],
    const device uint* page_indices   [[buffer(6)]],
    const device uint* page_indptr    [[buffer(7)]],
    const device int* req_of_token    [[buffer(8)]],
    const constant int& num_q_heads   [[buffer(9)]],
    const constant int& head_dim      [[buffer(10)]],
    const constant int& ratio         [[buffer(11)]],
    const constant int& top_k         [[buffer(12)]],
    const constant int& page_size     [[buffer(13)]],
    const constant float& scale       [[buffer(14)]],
    uint3 tgpos    [[threadgroup_position_in_grid]],
    uint3 lid      [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group[[simdgroup_index_in_threadgroup]]) {
  const int qi = int(tgpos.y);
  const int q_head = int(tgpos.z);
  const int tid = int(lid.x);

  threadgroup float q_smem[POOL_HEAD_MAX];
  threadgroup float partials[4];  // POOL_ATTN_BLOCK / 32 simdgroups
  threadgroup float bcast[1];

  const int req = req_of_token[qi];
  const int qpos = positions[qi];
  const int num_visible = (qpos + 1) / ratio;
  const device int* srow = selection + size_t(qi) * size_t(top_k);

  const device bfloat* q_row =
      q + (size_t(qi) * size_t(num_q_heads) + size_t(q_head)) * size_t(head_dim);
  for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
    q_smem[d] = float(q_row[d]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device bfloat* o_row =
      o + (size_t(qi) * size_t(num_q_heads) + size_t(q_head)) * size_t(head_dim);

  if (num_visible <= 0) {
    for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
      o_row[d] = bfloat(0.0f);
    }
    if (tid == 0) {
      lse_out[qi * num_q_heads + q_head] = -INFINITY;
    }
    return;
  }

  // Pass 1: the row max over the SELECTED keys only.
  float local_max = -INFINITY;
  for (int n = tid; n < top_k; n += POOL_ATTN_BLOCK) {
    const int c = srow[n];
    if (c < 0 || c >= num_visible) continue;
    const size_t slot = pool_paged_slot(
        page_indices, page_indptr, req, (c + 1) * ratio - 1, page_size);
    const device bfloat* k_row = comp_kv_pages + slot * size_t(head_dim);
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += q_smem[d] * float(k_row[d]);
    }
    local_max = max(local_max, dot * scale);
  }
  {
    float m = simd_max(local_max);
    if (simd_lane == 0) partials[simd_group] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      float v = (simd_lane < 4) ? partials[simd_lane] : -INFINITY;
      v = simd_max(v);
      if (simd_lane == 0) bcast[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float row_max = bcast[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // A row whose whole selection was padding or out of range attends nothing;
  // its branch contributes no mass and `-inf` is what the merge folds as
  // "this branch saw no key", exactly as `num_visible <= 0` above.
  if (!isfinite(row_max)) {
    for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
      o_row[d] = bfloat(0.0f);
    }
    if (tid == 0) {
      lse_out[qi * num_q_heads + q_head] = -INFINITY;
    }
    return;
  }

  // Pass 2: the flash weighted sum, over the same selected keys in the same
  // ascending order the selection carries.
  const int dims_per_thread = (head_dim + POOL_ATTN_BLOCK - 1) / POOL_ATTN_BLOCK;
  float acc[8] = {0};
  float local_z = 0.0f;

  for (int n = 0; n < top_k; ++n) {
    const int c = srow[n];
    if (c < 0 || c >= num_visible) continue;
    const size_t slot = pool_paged_slot(
        page_indices, page_indptr, req, (c + 1) * ratio - 1, page_size);
    const device bfloat* k_row = comp_kv_pages + slot * size_t(head_dim);
    float dot = 0.0f;
    for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
      dot += q_smem[d] * float(k_row[d]);
    }
    float s = simd_sum(dot);
    if (simd_lane == 0) partials[simd_group] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      float v = (simd_lane < 4) ? partials[simd_lane] : 0.0f;
      v = simd_sum(v);
      if (simd_lane == 0) bcast[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float w = precise::exp(bcast[0] * scale - row_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    local_z += w;
    for (int i = 0; i < dims_per_thread; ++i) {
      const int d = tid + i * POOL_ATTN_BLOCK;
      if (d < head_dim) {
        acc[i] += w * float(k_row[d]);
      }
    }
  }

  const float inv_z = local_z > 0.0f ? 1.0f / local_z : 0.0f;
  if (tid == 0) {
    const float kLog2e = 1.44269504088896340736f;
    lse_out[qi * num_q_heads + q_head] =
        local_z > 0.0f ? ((precise::log(local_z) + row_max) * kLog2e) : -INFINITY;
  }
  for (int i = 0; i < dims_per_thread; ++i) {
    const int d = tid + i * POOL_ATTN_BLOCK;
    if (d < head_dim) {
      o_row[d] = bfloat(acc[i] * inv_z);
    }
  }
}

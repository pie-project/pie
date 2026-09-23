#include <metal_stdlib>

using namespace metal;


inline size_t pool_paged_slot(
    const device uint* page_indices,
    const device uint* page_indptr,
    int req, int pos, int page_size) {
  const uint page = page_indices[page_indptr[req] + uint(pos / page_size)];
  return size_t(page) * size_t(page_size) + size_t(pos % page_size);
}


[[kernel]] void pool_boundary_decode(
    const device int* positions   [[buffer(0)]],
    device int* out_pos           [[buffer(1)]],
    device int* out_req           [[buffer(2)]],
    device int* out_rope          [[buffer(3)]],
    const constant int& n         [[buffer(4)]],
    const constant int& ratio     [[buffer(5)]],
    const device uchar* row_valid [[buffer(6)]],
    const device int* req_of_token[[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  const int t = int(gid);
  if (t >= n) return;
  const int p = positions[t];
  const bool valid = row_valid[t] != 0;
  const bool is_boundary = valid && (((p + 1) % ratio) == 0);
  out_pos[t] = is_boundary ? p : -1;
  // The request this row belongs to in the FIRE, because the pages the block is
  // filed into are the fire's. A row index inside this class's own window would
  // name whichever request sat at that offset.
  out_req[t] = req_of_token[t];
  out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;
}

[[kernel]] void pool_boundary_prefill(
    const device int* positions    [[buffer(0)]],
    device int* out_pos            [[buffer(1)]],
    device int* out_req            [[buffer(2)]],
    device int* out_rope           [[buffer(3)]],
    const constant int& n          [[buffer(4)]],
    const constant int& ratio      [[buffer(5)]],
    const device uchar* row_valid  [[buffer(6)]],
    const device int* req_of_token [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  const int t = int(gid);
  if (t >= n) return;
  const int p = positions[t];
  const bool valid = row_valid[t] != 0;
  const bool is_boundary = valid && (((p + 1) % ratio) == 0);
  out_pos[t] = is_boundary ? p : -1;
  out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;

  // As above: the fire's request, not this window's row group.
  out_req[t] = req_of_token[t];
}


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


template <typename T>
[[kernel]] void index_block_mean_paged(
    const device T* key_pages      [[buffer(0)]],
    device T* out                  [[buffer(1)]],
    const device int* boundary_pos [[buffer(2)]],
    const device int* boundary_req [[buffer(3)]],
    const device uint* page_indices[[buffer(4)]],
    const device uint* page_indptr [[buffer(5)]],
    const constant int& head_dim   [[buffer(6)]],
    const constant int& ratio      [[buffer(7)]],
    const constant int& page_size  [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int d = int(gid.x);
  const int c = int(gid.y);
  if (d >= head_dim) return;
  const int bpos = boundary_pos[c];
  const int req = boundary_req[c];
  if (bpos < 0) {
    out[size_t(c) * size_t(head_dim) + size_t(d)] = T(0);
    return;
  }
  float acc = 0.0f;
  for (int i = 0; i < ratio; ++i) {
    const int pos = bpos + i - (ratio - 1);
    if (pos < 0) continue;
    const size_t slot =
        pool_paged_slot(page_indices, page_indptr, req, pos, page_size);
    acc += float(key_pages[slot * size_t(head_dim) + size_t(d)]);
  }
  out[size_t(c) * size_t(head_dim) + size_t(d)] = T(acc / float(ratio));
}

#define instantiate_index_block_mean_paged(name, itype)                  \
  template [[host_name("index_block_mean_paged_" #name)]]                \
  [[kernel]] void index_block_mean_paged<itype>(                         \
      const device itype*, device itype*, const device int*,             \
      const device int*, const device uint*, const device uint*,         \
      const constant int&, const constant int&, const constant int&,     \
      uint2);

instantiate_index_block_mean_paged(bfloat16, bfloat)


constant int POOL_ATTN_BLOCK = 128;

constant int POOL_HEAD_MAX = 512;

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
  threadgroup float partials[4];
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
  threadgroup float partials[4];
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

  float local_max = -INFINITY;

  for (int n = tid; n < top_k; n += POOL_ATTN_BLOCK) {
    const int c = srow[n];
    if (c < 0) break;
    if (c >= num_visible) continue;
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

  if (!isfinite(row_max)) {
    for (int d = tid; d < head_dim; d += POOL_ATTN_BLOCK) {
      o_row[d] = bfloat(0.0f);
    }
    if (tid == 0) {
      lse_out[qi * num_q_heads + q_head] = -INFINITY;
    }
    return;
  }

  const int dims_per_thread = (head_dim + POOL_ATTN_BLOCK - 1) / POOL_ATTN_BLOCK;
  float acc[8] = {0};
  float local_z = 0.0f;

  for (int n = 0; n < top_k; ++n) {
    const int c = srow[n];
    if (c < 0) break;
    if (c >= num_visible) continue;
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

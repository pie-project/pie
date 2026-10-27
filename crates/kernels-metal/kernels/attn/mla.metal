#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;


constant constexpr int kMaxCkvPer = 16;
constant constexpr int kMaxKpePer = 4;

constant constexpr int kMlaSplit = 8;

[[kernel]] void mla_latents_bfloat16(
    const device bfloat* kv_a          [[buffer(0)]],
    const device bfloat* norm_weight   [[buffer(1)]],
    device bfloat* kv_c                [[buffer(2)]],
    device bfloat* k_pe                [[buffer(3)]],
    const constant int& kv_lora        [[buffer(4)]],
    const constant int& rope           [[buffer(5)]],
    const constant int& src_row_stride [[buffer(6)]],
    const constant float& eps          [[buffer(7)]],
    uint gid            [[threadgroup_position_in_grid]],
    uint lid            [[thread_position_in_threadgroup]],
    uint simd_lane      [[thread_index_in_simdgroup]],
    uint simd_group     [[simdgroup_index_in_threadgroup]],
    uint tg_size        [[threads_per_threadgroup]]) {
  const device bfloat* row = kv_a + size_t(gid) * size_t(src_row_stride);

  for (int d = int(lid); d < rope; d += int(tg_size)) {
    k_pe[size_t(gid) * size_t(rope) + d] = row[kv_lora + d];
  }

  float local = 0.0f;
  for (int d = int(lid); d < kv_lora; d += int(tg_size)) {
    const float v = float(row[d]);
    local += v * v;
  }
  threadgroup float partials[32];
  threadgroup float inv_rms[1];
  local = simd_sum(local);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    float acc = simd_sum(partials[simd_lane]);
    if (simd_lane == 0) inv_rms[0] = precise::rsqrt(acc / float(kv_lora) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = inv_rms[0];

  for (int d = int(lid); d < kv_lora; d += int(tg_size)) {
    const float v = float(row[d]);
    const float w = float(norm_weight[d]);
    kv_c[size_t(gid) * size_t(kv_lora) + d] = bfloat(v * inv * w);
  }
}

[[kernel]] void mla_split_q_b_bfloat16(
    const device bfloat* q_b   [[buffer(0)]],
    device bfloat* q_nope      [[buffer(1)]],
    device bfloat* q_pe        [[buffer(2)]],
    const constant int& total  [[buffer(3)]],
    const constant int& heads  [[buffer(4)]],
    const constant int& nope   [[buffer(5)]],
    const constant int& rope   [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const int i = int(gid);
  if (i >= total) return;
  const int per = nope + rope;
  const int d = i % per;
  const int h = (i / per) % heads;
  const int n = i / (heads * per);
  const bfloat v = q_b[i];
  if (d < nope) {
    q_nope[(size_t(n) * heads + h) * nope + d] = v;
  } else {
    q_pe[(size_t(n) * heads + h) * rope + (d - nope)] = v;
  }
}

[[kernel]] void mla_kv_append_bfloat16(
    const device bfloat* kv_c    [[buffer(0)]],
    const device bfloat* k_pe    [[buffer(1)]],
    device bfloat* ckv_pages     [[buffer(2)]],
    device bfloat* kpe_pages     [[buffer(3)]],
    const device uint* w_page    [[buffer(4)]],
    const device uint* w_off     [[buffer(5)]],
    const constant int& page_size [[buffer(6)]],
    const constant int& kv_lora   [[buffer(7)]],
    const constant int& rope      [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);
  const int row = int(tid.y);
  const size_t slot = size_t(w_page[row]) * size_t(page_size) + size_t(w_off[row]);
  if (d < kv_lora) {
    ckv_pages[slot * size_t(kv_lora) + d] = kv_c[size_t(row) * size_t(kv_lora) + d];
  }
  if (d < rope) {
    kpe_pages[slot * size_t(rope) + d] = k_pe[size_t(row) * size_t(rope) + d];
  }
}

[[kernel]] void mla_absorb_q_bfloat16(
    const device bfloat* q_nope [[buffer(0)]],
    const device bfloat* kv_b   [[buffer(1)]],
    device bfloat* q_latent     [[buffer(2)]],
    const constant int& heads   [[buffer(3)]],
    const constant int& rank    [[buffer(4)]],
    const constant int& nope    [[buffer(5)]],
    const constant int& v_dim   [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int i = int(tid.x);
  const int h = int(tid.y);
  const int t = int(tid.z);
  if (i >= rank) return;
  const size_t qn_base = (size_t(t) * heads + h) * nope;
  const size_t kb_base = size_t(h) * size_t(nope + v_dim) * size_t(rank);
  float acc = 0.0f;
  for (int j = 0; j < nope; ++j) {
    acc += float(q_nope[qn_base + j]) * float(kv_b[kb_base + size_t(j) * rank + i]);
  }
  q_latent[(size_t(t) * heads + h) * rank + i] = bfloat(acc);
}

[[kernel]] void mla_absorb_out_bfloat16(
    const device bfloat* latent [[buffer(0)]],
    const device bfloat* kv_b   [[buffer(1)]],
    device bfloat* o            [[buffer(2)]],
    const constant int& heads   [[buffer(3)]],
    const constant int& rank    [[buffer(4)]],
    const constant int& v_dim   [[buffer(5)]],
    const constant int& nope    [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int j = int(tid.x);
  const int h = int(tid.y);
  const int t = int(tid.z);
  if (j >= v_dim) return;
  const size_t lat_base = (size_t(t) * heads + h) * size_t(rank);
  const size_t wv_base =
      size_t(h) * size_t(nope + v_dim) * size_t(rank) + size_t(nope) * size_t(rank);
  float acc = 0.0f;
  for (int i = 0; i < rank; ++i) {
    acc += float(latent[lat_base + i]) * float(kv_b[wv_base + size_t(j) * rank + i]);
  }
  o[(size_t(t) * heads + h) * size_t(v_dim) + j] = bfloat(acc);
}

inline void mla_naive_paged_body(
    const device bfloat* q_nope,
    const device bfloat* q_pe,
    const device bfloat* ckv_pages,
    const device bfloat* kpe_pages,
    device bfloat* o,
    const device int* position_ids,
    const device int* req_of_token,
    const device uint* kv_page_indices,
    const device uint* kv_page_indptr,
    const device int* selection,
    int top_k,
    int page_size,
    int heads,
    int ckv,
    int kpe,
    float sm_scale,
    uint2 gid,
    uint lane,
    uint sg,
    threadgroup float* part_m,
    threadgroup float* part_l,
    threadgroup float* part_acc,
    device float* lse = nullptr) {
  const int h   = int(gid.x);
  const int row = int(gid.y);
  const int per  = ckv / 32;
  const int pper = kpe / 32;

  const int r      = req_of_token[row];
  const int q_pos  = position_ids[row];
  const int j_end  = q_pos + 1;
  const int page_base = int(kv_page_indptr[r]);

  const device int* srow =
      (selection != nullptr) ? selection + size_t(row) * size_t(top_k) : nullptr;

  const device bfloat* qn = q_nope + (size_t(row) * heads + h) * size_t(ckv);
  const device bfloat* qp = q_pe   + (size_t(row) * heads + h) * size_t(kpe);
  float qn_r[kMaxCkvPer];
  float qp_r[kMaxKpePer];
  for (int i = 0; i < per; ++i)  qn_r[i] = float(qn[lane + i * 32]);
  for (int i = 0; i < pper; ++i) qp_r[i] = float(qp[lane + i * 32]);

  float acc[kMaxCkvPer];
  for (int i = 0; i < per; ++i) acc[i] = 0.0f;
  float m = -3.0e38f, lsum = 0.0f;

  const int steps = (srow != nullptr) ? top_k : j_end;
  for (int n = int(sg); n < steps; n += kMlaSplit) {
    int j = n;
    if (srow != nullptr) {
      j = srow[n];

      if (j < 0) break;
      if (j >= j_end) continue;
    }
    const uint page = kv_page_indices[page_base + j / page_size];
    const size_t slot = size_t(page) * size_t(page_size) + size_t(j % page_size);
    const device bfloat* ckv_j = ckv_pages + slot * size_t(ckv);
    const device bfloat* kpe_j = kpe_pages + slot * size_t(kpe);

    float kv[kMaxCkvPer];
    float pd = 0.0f;
    for (int i = 0; i < per; ++i) {
      kv[i] = float(ckv_j[lane + i * 32]);
      pd += qn_r[i] * kv[i];
    }
    for (int i = 0; i < pper; ++i) {
      pd += qp_r[i] * float(kpe_j[lane + i * 32]);
    }
    pd = simd_sum(pd);
    const float score = pd * sm_scale;
    const float m_new = max(m, score);
    const float corr = fast::exp(m - m_new);
    const float p = fast::exp(score - m_new);
    lsum = lsum * corr + p;
    for (int i = 0; i < per; ++i) acc[i] = acc[i] * corr + p * kv[i];
    m = m_new;
  }

  if (lane == 0) {
    part_m[sg] = m;
    part_l[sg] = lsum;
  }
  for (int i = 0; i < per; ++i) part_acc[sg * (kMaxCkvPer * 32) + lane + i * 32] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;

  float M = part_m[0];
  for (int s = 1; s < kMlaSplit; ++s) M = max(M, part_m[s]);
  float L = 0.0f;
  float out[kMaxCkvPer];
  for (int i = 0; i < per; ++i) out[i] = 0.0f;
  for (int s = 0; s < kMlaSplit; ++s) {
    const float w = fast::exp(part_m[s] - M);
    L += part_l[s] * w;
    for (int i = 0; i < per; ++i) out[i] += part_acc[s * (kMaxCkvPer * 32) + lane + i * 32] * w;
  }
  const float inv = (L > 0.0f) ? (1.0f / L) : 0.0f;
  device bfloat* orow = o + (size_t(row) * heads + h) * size_t(ckv);
  for (int i = 0; i < per; ++i) orow[lane + i * 32] = bfloat(out[i] * inv);
  // The log-sum-exp a sink rescales by, base 2 as the sink reads it.
  if (lse != nullptr && lane == 0) {
    lse[size_t(row) * size_t(heads) + size_t(h)] =
        (L > 0.0f) ? (M + precise::log(L)) * 1.4426950408889634f : -INFINITY;
  }
}

[[kernel]] void mla_naive_paged_bfloat16(
    const device bfloat* q_nope     [[buffer(0)]],
    const device bfloat* q_pe       [[buffer(1)]],
    const device bfloat* ckv_pages  [[buffer(2)]],
    const device bfloat* kpe_pages  [[buffer(3)]],
    device bfloat* o                [[buffer(4)]],
    const device int* position_ids  [[buffer(5)]],
    const device int* req_of_token  [[buffer(6)]],
    const device uint* kv_page_indices [[buffer(7)]],
    const device uint* kv_page_indptr  [[buffer(8)]],
    const constant int& page_size   [[buffer(9)]],
    const constant int& heads       [[buffer(10)]],
    const constant int& ckv         [[buffer(11)]],
    const constant int& kpe         [[buffer(12)]],
    const constant float& sm_scale  [[buffer(13)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part_m[kMlaSplit];
  threadgroup float part_l[kMlaSplit];
  threadgroup float part_acc[kMlaSplit * kMaxCkvPer * 32];
  mla_naive_paged_body(q_nope, q_pe, ckv_pages, kpe_pages, o, position_ids,
                       req_of_token, kv_page_indices, kv_page_indptr,
                       (const device int*)nullptr, 0, page_size, heads, ckv,
                       kpe, sm_scale, gid, lane, sg,
                       part_m, part_l, part_acc);
}

[[kernel]] void mla_naive_paged_selected_bfloat16(
    const device bfloat* q_nope     [[buffer(0)]],
    const device bfloat* q_pe       [[buffer(1)]],
    const device bfloat* ckv_pages  [[buffer(2)]],
    const device bfloat* kpe_pages  [[buffer(3)]],
    device bfloat* o                [[buffer(4)]],
    const device int* position_ids  [[buffer(5)]],
    const device int* req_of_token  [[buffer(6)]],
    const device uint* kv_page_indices [[buffer(7)]],
    const device uint* kv_page_indptr  [[buffer(8)]],
    const constant int& page_size   [[buffer(9)]],
    const constant int& heads       [[buffer(10)]],
    const constant int& ckv         [[buffer(11)]],
    const constant int& kpe         [[buffer(12)]],
    const constant float& sm_scale  [[buffer(13)]],
    const device int* selection     [[buffer(14)]],
    const constant int& top_k       [[buffer(15)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part_m[kMlaSplit];
  threadgroup float part_l[kMlaSplit];
  threadgroup float part_acc[kMlaSplit * kMaxCkvPer * 32];
  mla_naive_paged_body(q_nope, q_pe, ckv_pages, kpe_pages, o, position_ids,
                       req_of_token, kv_page_indices, kv_page_indptr, selection,
                       top_k, page_size, heads, ckv, kpe, sm_scale, gid, lane, sg,
                       part_m, part_l, part_acc);
}

[[kernel]] void mla_naive_paged_selected_lse_bfloat16(
    const device bfloat* q_nope     [[buffer(0)]],
    const device bfloat* q_pe       [[buffer(1)]],
    const device bfloat* ckv_pages  [[buffer(2)]],
    const device bfloat* kpe_pages  [[buffer(3)]],
    device bfloat* o                [[buffer(4)]],
    const device int* position_ids  [[buffer(5)]],
    const device int* req_of_token  [[buffer(6)]],
    const device uint* kv_page_indices [[buffer(7)]],
    const device uint* kv_page_indptr  [[buffer(8)]],
    const constant int& page_size   [[buffer(9)]],
    const constant int& heads       [[buffer(10)]],
    const constant int& ckv         [[buffer(11)]],
    const constant int& kpe         [[buffer(12)]],
    const constant float& sm_scale  [[buffer(13)]],
    const device int* selection     [[buffer(14)]],
    const constant int& top_k       [[buffer(15)]],
    device float* lse               [[buffer(16)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part_m[kMlaSplit];
  threadgroup float part_l[kMlaSplit];
  threadgroup float part_acc[kMlaSplit * kMaxCkvPer * 32];
  mla_naive_paged_body(q_nope, q_pe, ckv_pages, kpe_pages, o, position_ids,
                       req_of_token, kv_page_indices, kv_page_indptr, selection,
                       top_k, page_size, heads, ckv, kpe, sm_scale, gid, lane, sg,
                       part_m, part_l, part_acc, lse);
}

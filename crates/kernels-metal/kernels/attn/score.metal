#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <typename T, int HEAD_DIM_MAX, int SIMDS>
[[kernel]] void attn_score_capture(
    const device T* q                   [[buffer(0)]],
    const device int* qo_indptr         [[buffer(1)]],
    const device T* k_pages             [[buffer(2)]],
    const device uint* kv_page_indices  [[buffer(3)]],
    const device uint* kv_page_indptr   [[buffer(4)]],
    const device int* position_ids      [[buffer(5)]],
    device float* scores                [[buffer(6)]],
    const constant int& page_size       [[buffer(7)]],
    const constant int& num_q_heads     [[buffer(8)]],
    const constant int& num_kv_heads    [[buffer(9)]],
    const constant int& head_dim        [[buffer(10)]],
    const constant float& sm_scale      [[buffer(11)]],
    const constant int& observe         [[buffer(12)]],
    const constant int& kv_max          [[buffer(13)]],
    uint3 tgid     [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int VPT = HEAD_DIM_MAX / 32;
  constexpr int THREADS = SIMDS * 32;
  constexpr float NEG_INF = -3.0e38f;

  const int request = int(tgid.x);
  const int head = int(tgid.y);
  const int lane = int(simd_lid);
  const int warp = int(simd_gid);
  const int tid = warp * 32 + lane;

  threadgroup float q_s[HEAD_DIM_MAX];
  threadgroup float wm[SIMDS];
  threadgroup float wl[SIMDS];

  device float* out =
      scores + (size_t(request) * size_t(num_q_heads) + size_t(head)) *
                   size_t(kv_max);
  for (int i = tid; i < kv_max; i += THREADS) {
    out[i] = 0.0f;
  }

  const int page_first = int(kv_page_indptr[request]);
  const int pages = int(kv_page_indptr[request + 1]) - page_first;

  const int capacity = pages * page_size;
  const int qo_hi = qo_indptr[request + 1];
  const int qo_len = qo_hi - qo_indptr[request];
  const int rows = observe < qo_len ? observe : qo_len;

  if (pages <= 0 || capacity <= 0 || rows <= 0) {
    return;
  }

  threadgroup_barrier(mem_flags::mem_device);

  const int group = num_q_heads / num_kv_heads;
  const int kv_head = head / group;
  const int row_stride = num_kv_heads * head_dim;
  const float inv_rows = 1.0f / float(rows);

  for (int w = 0; w < rows; ++w) {

    const int q_index = qo_hi - rows + w;
    const int causal = position_ids[q_index] + 1;
    const int limit = causal < capacity ? causal : capacity;

    if (limit <= 0) {
      continue;
    }

    const device T* q_row =
        q + (size_t(q_index) * size_t(num_q_heads) + size_t(head)) * size_t(head_dim);
    for (int d = tid; d < head_dim; d += THREADS) {
      q_s[d] = float(q_row[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = NEG_INF;
    float running_sum = 0.0f;
    for (int j = warp; j < limit; j += SIMDS) {
      const size_t slot =
          size_t(kv_page_indices[page_first + j / page_size]) * size_t(page_size) +
          size_t(j % page_size);
      const device T* k_row =
          k_pages + slot * size_t(row_stride) + size_t(kv_head) * size_t(head_dim);
      float dot = 0.0f;
      for (int u = 0; u < VPT; ++u) {
        const int d = lane + u * 32;
        if (d < head_dim) {
          dot += q_s[d] * float(k_row[d]);
        }
      }

      dot = simd_sum(dot);

      const float score = dot * sm_scale;
      const float widened = max(running_max, score);
      running_sum =
          running_sum * fast::exp(running_max - widened) + fast::exp(score - widened);
      running_max = widened;
    }
    if (lane == 0) {
      wm[warp] = running_max;
      wl[warp] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float folded_max = NEG_INF;
    for (int u = 0; u < SIMDS; ++u) {
      folded_max = max(folded_max, wm[u]);
    }
    float denominator = 0.0f;
    for (int u = 0; u < SIMDS; ++u) {

      denominator += wl[u] * fast::exp(wm[u] - folded_max);
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;

    for (int j = warp; j < limit; j += SIMDS) {
      const size_t slot =
          size_t(kv_page_indices[page_first + j / page_size]) * size_t(page_size) +
          size_t(j % page_size);
      const device T* k_row =
          k_pages + slot * size_t(row_stride) + size_t(kv_head) * size_t(head_dim);
      float dot = 0.0f;
      for (int u = 0; u < VPT; ++u) {
        const int d = lane + u * 32;
        if (d < head_dim) {
          dot += q_s[d] * float(k_row[d]);
        }
      }
      dot = simd_sum(dot);

      if (lane == 0 && j < kv_max) {
        out[j] += fast::exp(dot * sm_scale - folded_max) * inv * inv_rows;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define instantiate_attn_score_capture(name, itype, d)                        \
  template [[host_name("attn_score_capture_" #name "_d_" #d)]]                \
  [[kernel]] void attn_score_capture<itype, d, 8>(                            \
      const device itype*, const device int*, const device itype*,            \
      const device uint*, const device uint*, const device int*,              \
      device float*, const constant int&, const constant int&,                \
      const constant int&, const constant int&, const constant float&,        \
      const constant int&, const constant int&, uint3, uint, uint);

instantiate_attn_score_capture(bfloat16, bfloat, 64)
instantiate_attn_score_capture(bfloat16, bfloat, 128)
instantiate_attn_score_capture(bfloat16, bfloat, 256)

instantiate_attn_score_capture(bfloat16, bfloat, 512)

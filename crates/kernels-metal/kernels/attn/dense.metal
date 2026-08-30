#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

/// **BIDIRECTIONAL DENSE ATTENTION OVER A PATCH WINDOW** — the vision
/// towers' one real kernel, the Metal mirror of `kernels-cuda`'s
/// `attn/dense.cuh` (`.wiki/alto/multimodal.md` §2).
///
/// It is the simplest attention this plane owns, and every simplification is
/// a fact about the second row axis rather than a shortcut: patch rows are
/// not a cache, so there are no pages, no append and no page tables; a patch
/// attends to every patch of its own image and to no other, so there is no
/// causal ladder, no sliding window and no mask plane — the block-diagonal
/// IS the segment list; nothing merges against a second pass, so there is no
/// log-sum-exp plane to carry out. What is left is q, k, v, one indptr, and
/// the softmax.
///
/// **The segment list is the mask.** `segments` is the patch axis's own
/// indptr — `int`, `[images + 1]`, image `i` owning rows
/// `[segments[i], segments[i + 1])` — and a threadgroup finds its image by
/// binary search rather than by carrying a per-row image id, so the only
/// geometry the launch reads is the one the fold already assembled. A row at
/// or past `segments[num_segments]` belongs to NO image: that is a
/// patch-axis rung's padding, and it lands zeros rather than reading a
/// neighbour's keys, which is what keeps a bucketed patch window as harmless
/// as a bucketed token one.
///
/// **The reduction is one simdgroup per key chunk, merged in threadgroup
/// memory.** Simdgroup `w` walks keys `begin + w, begin + w + SIMDS, ...`
/// keeping its own running (max, sum, accumulator) — the online softmax, so
/// nothing of size `rows x rows` is ever materialised and the kernel needs no
/// workspace at all. That is the capture argument: no scratch means no slab
/// to warm and no allocation on the fire path. The per-simdgroup states are
/// then folded by one rescaled sum, and a simdgroup that drew no keys folds
/// in weighing zero.
///
/// **`HEAD_DIM_MAX` IS A STAMP, NOT A SHAPE, AND ON THIS PLANE IT IS ALSO
/// THE THREADGROUP ALLOCATION.** It fixes how many accumulator registers
/// each lane holds (`HEAD_DIM_MAX / 32`) and how wide the threadgroup planes
/// below are; the live `head_dim` may be anything at or below it — 64 for
/// qwen35's tower, 72 or 80 for a SigLIP-shaped one, none of which divide by
/// 32. The entry picks the tightest stamp that holds the head. The CUDA twin
/// sizes its shared plane dynamically from the live `head_dim`; a `Fire` here
/// carries no threadgroup-memory length (`encode.rs` dropped the field with
/// the rest of the CUDA-plane geometry), so the plane is a static array at
/// the stamp and `wacc` strides by `HEAD_DIM_MAX` where the twin strides by
/// `head_dim`. Same values, wider stride, and the widest stamp costs
/// `(256 + 4 * 256 + 8) * 4` = 5 KiB of a 32 KiB budget.
///
/// Grouped heads are read, never expanded: `num_q_heads / num_kv_heads`
/// query heads share one kv head, so a tower that ships plain MHA states the
/// two counts equal and pays nothing for the divide.
///
/// **`NEG_INF` IS FINITE HERE AND INFINITE THERE, AND NOTHING MOVES.** The
/// CUDA twin opens its running max at `-inf`; this plane uses the sentinel
/// its own neighbour `sdpa_paged.metal` opens with (`-3e38`), because
/// `wm[w] - folded_max` would be `-inf - -inf` for an all-empty fold. Every
/// live expression agrees to the bit: `exp(-3e38 - score)` and
/// `exp(-inf - score)` are both zero, and `exp(score - score)` is one.
template <typename T, int HEAD_DIM_MAX, int SIMDS>
[[kernel]] void dense_bidirectional(
    const device T* q                 [[buffer(0)]],
    const device T* k                 [[buffer(1)]],
    const device T* v                 [[buffer(2)]],
    device T* o                       [[buffer(3)]],
    const device int* segments        [[buffer(4)]],
    const constant int& num_segments  [[buffer(5)]],
    const constant int& num_q_heads   [[buffer(6)]],
    const constant int& num_kv_heads  [[buffer(7)]],
    const constant int& head_dim      [[buffer(8)]],
    const constant float& sm_scale    [[buffer(9)]],
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
  threadgroup int span[2];

  // Which image owns this row. The list is short (images in the fire), so
  // one thread walks it and the threadgroup reads the answer.
  if (tid == 0) {
    int begin = -1;
    int end = -1;
    const int first = segments[0];
    const int total = segments[num_segments];
    if (row >= first && row < total) {
      int lo = 0;
      int hi = num_segments - 1;
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (segments[mid] <= row) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      begin = segments[lo];
      end = segments[lo + 1];
    }
    span[0] = begin;
    span[1] = end;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int begin = span[0];
  const int end = span[1];

  device T* out =
      o + (size_t(row) * size_t(num_q_heads) + size_t(head)) * size_t(head_dim);
  if (end <= begin) {
    // A rung's padding row: no image claims it, and it reads nobody's keys.
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

  thread float acc[VPT];
  for (int u = 0; u < VPT; ++u) {
    acc[u] = 0.0f;
  }
  float running_max = NEG_INF;
  float running_sum = 0.0f;

  for (int j = begin + warp; j < end; j += SIMDS) {
    const device T* k_row =
        k + (size_t(j) * size_t(num_kv_heads) + size_t(kv_head)) * size_t(head_dim);
    float dot = 0.0f;
    for (int u = 0; u < VPT; ++u) {
      const int d = lane + u * 32;
      if (d < head_dim) {
        dot += q_s[d] * float(k_row[d]);
      }
    }
    // Every lane leaves with the whole score, so the rescale below needs no
    // broadcast — `simd_sum` is the butterfly the twin spells `__shfl_xor`.
    dot = simd_sum(dot);

    const float score = dot * sm_scale;
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

#define instantiate_dense_bidirectional(name, itype, d)                    \
  template [[host_name("dense_bidirectional_" #name "_d_" #d)]]            \
  [[kernel]] void dense_bidirectional<itype, d, 4>(                        \
      const device itype*, const device itype*, const device itype*,       \
      device itype*, const device int*, const constant int&,               \
      const constant int&, const constant int&, const constant int&,       \
      const constant float&, uint3, uint, uint);

instantiate_dense_bidirectional(bfloat16, bfloat, 64)
instantiate_dense_bidirectional(bfloat16, bfloat, 128)
instantiate_dense_bidirectional(bfloat16, bfloat, 256)

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

/// **PER-KEY ATTENTION MASS OVER AN OBSERVATION WINDOW** — the alto
/// observability door's one device kernel on this plane
/// (`.wiki/alto/attn-score.md` §4), and the mirror of `kernels-cuda`'s
/// `attn/score.cuh`.
///
/// **A NEW FILE ON PURPOSE, AND THE SAME REASON ON BOTH PLANES.** What this
/// computes is not what the attention next door computes. A flash-family
/// kernel never materialises the softmax weights — it carries a running
/// (max, sum) and folds them into `o` — so the per-key row the eviction and
/// interpretability papers define does not exist anywhere in
/// `attn/sdpa_paged.metal`'s output and cannot be read back out of it. So the
/// weights are RECOMPUTED straight out of the pages, and that recompute
/// shares nothing with the sdpa family but the pages it reads: no plan
/// arbitration, no tile, no mask plane, no log-sum-exp, no `o` at all. It is
/// a file of its own beside them rather than an arm inside them, and the two
/// can be read, changed and broken independently.
///
/// **The quantity, exactly.** For one request `r`, one query head `h`, and
/// the observation window's rows `w in [0, rows)` — the LAST
/// `rows = min(observe, qo_len)` query rows of the request —
///
/// ```text
///   s_j     = sm_scale * <q[row(w), h], k[j, kv_head(h)]>
///   p_j(w)  = exp(s_j - max_j s_j) / sum_j exp(s_j - max_j s_j)   over j < limit(w)
///   out[j]  = sum_w p_j(w) / rows
/// ```
///
/// Each output row is a probability distribution over the request's live KV
/// summing to one — the mean over the window of that row's own softmax, which
/// is TOVA's number at `observe = 1` and SnapKV's at `observe = 32`. The head
/// fold the papers apply on top is deliberately NOT taken: §4 rules the
/// contract per-head and lets the guest fold.
///
/// **THE CAUSAL BOUND IS THE POSITION TABLE, WHERE THE CUDA TWIN COMPUTES
/// IT.** That side reads `kv_last_page_lens`, reconstructs
/// `kv_len = (pages - 1) * page_size + last`, and spells the bound as
/// `min(kv_len - rows + w + 1, kv_len)`. This pool carries no last-page
/// vector at all — the metal `KvPool` is `{keys, values, page_indices,
/// page_indptr, page_size, strides}` — and it does not need one, because the
/// number that arithmetic reconstructs is already on the plan: a query row's
/// ABSOLUTE position is its causal bound less one, which is the same fact
/// `attn/sdpa_paged.metal` walks `kp <= q_pos` on. Row `w` of the window
/// therefore takes `limit = position_ids[q_index] + 1`, and the two planes
/// agree term for term — `position(qo_hi - 1) = kv_len - 1` after the append,
/// so `position(qo_hi - rows + w) + 1` IS `kv_len - rows + w + 1`. One table
/// read where the twin needs two.
///
/// **THE WHOLE ROW IS WRITTEN, ALWAYS — and that is a safety property, not
/// tidiness.** The slab is a caller-owned rectangle reused across fires, so a
/// tail left as it was is not "unset", it is the PREVIOUS fire's mass at a
/// longer kv extent, sitting on keys that no longer exist. An eviction policy
/// ranking on that garbage would drop live tokens and never fault. So every
/// slot in `[0, kv_max)` is stored to: the live prefix gets the mass and the
/// rest gets exactly `0.0f`, on every path — including the degenerate ones
/// (no pages, empty cache, empty window), which zero the row and return
/// rather than leaving it alone.
///
/// A kv extent past `kv_max` is a caller error the engine refuses upstream,
/// but the extent is a DEVICE-SIDE number and no host refusal can see it — so
/// the kernel stays safe on its own: the softmax is still taken over the true
/// `[0, limit)`, and only the STORE is clamped to `kv_max`. The walk is
/// clamped a second time, to the pages the request actually holds, so a
/// position table and a page table that disagree read nobody else's keys.
///
/// **Two passes over the pages per window row, and the keys are re-read.**
/// Pass one walks `[0, limit)` carrying the online (max, sum) and folds the
/// per-simdgroup states; pass two walks it again and stores
/// `exp(s - M) / L` scaled by `1 / rows`. The alternative — materialise the
/// scores and normalise them afterwards, as the C++ lineage did across three
/// kernels — needs a `heads x window x kv_len` F32 slab, which is the
/// allocation that made the old path refuse above 1 GiB. Reading the pages
/// twice buys that whole allocation away, and this path only runs for the
/// lanes that asked to be observed.
///
/// **THERE IS EXACTLY ONE CROSS-THREAD DEVICE DEPENDENCY AND IT IS THE
/// ZEROING.** Every thread zeroes a stride of the row; in pass two, key `j`
/// is owned by simdgroup `j % SIMDS` for the whole window, so an accumulation
/// needs no atomic and no thread ever touches another's `j` — but it does
/// touch bytes some OTHER thread zeroed a moment ago. That one edge is what
/// the `mem_device` barrier below publishes, and it is why there is one
/// rather than one per row.
///
/// `HEAD_DIM_MAX` is a stamp and not a shape, exactly as in
/// `attn/dense.metal`: it fixes the unrolled length of the per-lane dot
/// (`HEAD_DIM_MAX / 32` elements per lane) and the threadgroup plane, and the
/// live `head_dim` may be anything at or below it — 64, 72 and 80 all ride
/// the 128-wide stamp unpadded. `SIMDS` is the key-side parallelism knob and
/// the width of the fold, as there, and it is the CUDA twin's `WARPS`.
///
/// **`NEG_INF` IS FINITE HERE AND INFINITE THERE, AND NOTHING MOVES**, for
/// `attn/dense.metal`'s reason: `wm[u] - folded_max` would be `-inf - -inf`
/// for an all-empty fold. Every live expression agrees to the bit.
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
    const constant int& lane_offset     [[buffer(13)]],
    const constant int& plane_stride    [[buffer(14)]],
    const constant int& plane           [[buffer(15)]],
    const constant int& kv_max          [[buffer(16)]],
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

  // The output row. Its address is a pure function of the fire lane, the
  // slab's plane geometry and the head — never of anything the device reads —
  // so a lane that returns early still zeroes exactly its own row and touches
  // no neighbour's.
  device float* out =
      scores + (size_t(lane_offset + request) * size_t(plane_stride) +
                size_t(plane + head)) *
                   size_t(kv_max);
  for (int i = tid; i < kv_max; i += THREADS) {
    out[i] = 0.0f;
  }

  const int page_first = int(kv_page_indptr[request]);
  const int pages = int(kv_page_indptr[request + 1]) - page_first;
  // What the request's own pages can hold. The causal bound below is clamped
  // to it so a position table that outran the page table reads no key the
  // request does not own.
  const int capacity = pages * page_size;
  const int qo_hi = qo_indptr[request + 1];
  const int qo_len = qo_hi - qo_indptr[request];
  const int rows = observe < qo_len ? observe : qo_len;

  // A request with no pages, no live cache or no observation window has a
  // ZEROED row, not an untouched one: the caller reads the whole rectangle and
  // an untouched row would be the last fire's numbers. Every value here is
  // uniform across the threadgroup, so this returns all of it or none of it
  // and the barrier below is never divergently reached.
  if (pages <= 0 || capacity <= 0 || rows <= 0) {
    return;
  }

  // The one edge in this kernel where a thread reads bytes another wrote: the
  // zeroing above against the accumulation in pass two. Published once, here,
  // rather than per row — nothing after this point crosses threads in device
  // memory, because a key belongs to one simdgroup for the whole window.
  threadgroup_barrier(mem_flags::mem_device);

  const int group = num_q_heads / num_kv_heads;
  const int kv_head = head / group;
  const int row_stride = num_kv_heads * head_dim;
  const float inv_rows = 1.0f / float(rows);

  for (int w = 0; w < rows; ++w) {
    // Window-relative: the observation window is the request's LAST `rows`
    // query rows, and `qo_indptr` is already rebased onto `q` — as is
    // `position_ids`, which the plan carries cut to the same window.
    const int q_index = qo_hi - rows + w;
    const int causal = position_ids[q_index] + 1;
    const int limit = causal < capacity ? causal : capacity;
    // Uniform across the threadgroup: every thread takes the same branch, so
    // the barriers below are never divergently reached.
    if (limit <= 0) {
      continue;
    }

    const device T* q_row =
        q + (size_t(q_index) * size_t(num_q_heads) + size_t(head)) * size_t(head_dim);
    for (int d = tid; d < head_dim; d += THREADS) {
      q_s[d] = float(q_row[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── pass one: the online (max, sum) over [0, limit) ──────────────────
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
      // Every lane leaves with the whole score, so the running state below
      // needs no broadcast — `simd_sum` is the butterfly the twin spells
      // `__shfl_xor_sync`.
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
      // A simdgroup that drew no keys folds in as `0 * exp(-3e38 - M) = 0`.
      denominator += wl[u] * fast::exp(wm[u] - folded_max);
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;

    // ── pass two: the same walk, storing the normalised mass ─────────────
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
      // One simdgroup owns key `j` for the whole window row, and only its
      // first lane stores — the accumulation needs no atomic.
      if (lane == 0 && j < kv_max) {
        out[j] += fast::exp(dot * sm_scale - folded_max) * inv * inv_rows;
      }
    }
    // Before the next window row overwrites `q_s`, `wm` and `wl`.
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
      const constant int&, const constant int&, const constant int&,          \
      const constant int&, const constant int&, uint3, uint, uint);

instantiate_attn_score_capture(bfloat16, bfloat, 64)
instantiate_attn_score_capture(bfloat16, bfloat, 128)
instantiate_attn_score_capture(bfloat16, bfloat, 256)

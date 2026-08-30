#pragma once

#include "prelude/device.cuh"
#include "prelude/kv_paged_addr.cuh"

namespace pie::attn {

/// **PER-KEY ATTENTION MASS OVER AN OBSERVATION WINDOW** — the alto
/// observability door's one device kernel (`.wiki/alto/attn-score.md` §4).
///
/// **A NEW FILE ON PURPOSE.** What this computes is not what the attention
/// next door computes. A flash-family kernel never materialises the softmax
/// weights — it carries a running (max, sum) and folds them into `o` — so
/// the per-key row the eviction and interpretability papers define does not
/// exist anywhere in `attn/attention.cuh`'s output and cannot be read back
/// out of it. The C++ lineage therefore RECOMPUTED the weights in a second
/// pass over the pages (`attn_score.cu`, reproduced here in one kernel), and
/// that recompute shares nothing with the fa2 lattice but the pages it
/// reads: no plan schedule, no split-kv partials, no merge, no lse plane, no
/// `o` at all. So it is a file of its own, next to the family rather than
/// inside it, and the two can be read, changed and broken independently.
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
/// with `limit(w) = min(kv_len - rows + w + 1, kv_len)`: the causal bound the
/// C++ lineage's `attn_prefill_score_normalize` spells, taken here in one
/// pass rather than as a post-processing sweep over a materialised
/// `heads x window x kv_len` slab. Each output row is a probability
/// distribution over `[0, kv_len)` summing to one — the mean over the window
/// of that row's own softmax, which is TOVA's number at `observe = 1` and
/// SnapKV's at `observe = 32`. The head fold the papers apply on top
/// (`attn_prefill_score_fold`'s extra `1 / num_q_heads`) is deliberately NOT
/// taken: §4 rules the contract per-head and lets the guest fold.
///
/// **THE WHOLE ROW IS WRITTEN, ALWAYS — and that is a safety property, not
/// tidiness.** The slab is a caller-owned rectangle reused across fires, so
/// a tail left as it was is not "unset", it is the PREVIOUS fire's mass at a
/// longer `kv_len`, sitting on keys that no longer exist. An eviction policy
/// ranking on that garbage would drop live tokens and never fault. So every
/// slot in `[0, kv_max)` is stored to: `[0, kv_len)` gets the mass and
/// `[kv_len, kv_max)` gets exactly `0.f`, on every path — including the
/// degenerate ones (no pages, empty cache, empty window), which zero the row
/// and return rather than leaving it alone.
///
/// `kv_len > kv_max` is a caller error the engine refuses upstream, but
/// `kv_len` is a DEVICE-SIDE number and no host refusal can see it — so the
/// kernel stays safe on its own: the softmax is still taken over the true
/// `[0, limit)`, and only the STORE is clamped to `kv_max`. Nothing is
/// written past the row.
///
/// **Two passes over the pages per window row, and the keys are re-read.**
/// Pass one walks `[0, limit)` carrying the online (max, sum) and folds the
/// per-warp states; pass two walks it again and stores `exp(s - M) / L`
/// scaled by `1 / rows`. The alternative — materialise the scores and
/// normalise them afterwards, as the C++ lineage did across three kernels —
/// needs the `heads x window x kv_len` F32 slab that made the old path
/// refuse above 1 GiB. Reading the pages twice buys that whole allocation
/// away, and this path only runs for the lanes that asked to be observed.
///
/// `HEAD_DIM_MAX` is a stamp and not a shape, exactly as in
/// `attn/dense.cuh`: it fixes the unrolled length of the per-lane dot
/// (`HEAD_DIM_MAX / 32` elements per lane), and the live `head_dim` may be
/// anything at or below it — 64, 72 and 80 all ride the 128-wide stamp
/// unpadded. `WARPS` is the key-side parallelism knob and the width of the
/// fold, as there. `HND_LAYOUT` is the pool's own page enumerator, read
/// through `kv_dst_index` so this kernel spells the paged addressing
/// exactly once, in the same place `attn/kv.cuh`'s appenders spell it.
template <int HEAD_DIM_MAX, int WARPS, bool HND_LAYOUT>
__global__ void score_capture(
    const bf16* __restrict__ q,
    const i32* __restrict__ qo_indptr,
    const bf16* __restrict__ k_pages,
    const i32* __restrict__ kv_page_indices,
    const i32* __restrict__ kv_page_indptr,
    const i32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    int page_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    int observe,
    int lane_offset,
    int plane_stride,
    int plane,
    int kv_max)
{
    constexpr int VPT = HEAD_DIM_MAX / 32;

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int threads = static_cast<int>(blockDim.x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    extern __shared__ float smem[];
    float* q_s = smem;                    // head_dim
    float* wm = q_s + head_dim;           // WARPS
    float* wl = wm + WARPS;               // WARPS

    // The output row. Its address is a pure function of the fire lane, the
    // slab's plane geometry and the head — never of anything the device
    // reads — so a lane that returns early still zeroes exactly its own row
    // and touches no neighbour's.
    const long long out_row =
        (static_cast<long long>(lane_offset + request) * plane_stride + plane + head) *
        static_cast<long long>(kv_max);
    float* out = scores + out_row;

    for (int i = static_cast<int>(threadIdx.x); i < kv_max; i += threads) {
        out[i] = 0.f;
    }

    const int page_first = static_cast<int>(kv_page_indptr[request]);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) - page_first;
    const int kv_len =
        pages > 0 ? (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request])
                  : 0;
    const int qo_lo = static_cast<int>(qo_indptr[request]);
    const int qo_hi = static_cast<int>(qo_indptr[request + 1]);
    const int qo_len = qo_hi - qo_lo;
    const int rows = observe < qo_len ? observe : qo_len;

    // A request with no pages, no live cache or no observation window has a
    // ZEROED row, not an untouched one: the caller reads the whole rectangle
    // and an untouched row would be the last fire's numbers.
    if (pages <= 0 || kv_len <= 0 || rows <= 0) {
        return;
    }

    const int group = num_q_heads / num_kv_heads;
    const int kv_head = head / group;
    const float inv_rows = 1.f / static_cast<float>(rows);

    for (int w = 0; w < rows; ++w) {
        // Window-relative: the observation window is the request's LAST
        // `rows` query rows, and `qo_indptr` is already rebased onto `q`.
        const int q_index = qo_hi - rows + w;
        const int causal = kv_len - rows + w + 1;
        const int limit = causal < kv_len ? causal : kv_len;
        // Uniform across the block: every thread takes the same branch, so
        // the `__syncthreads` below are never divergently reached.
        if (limit <= 0) {
            continue;
        }

        const bf16* q_row =
            q + (static_cast<long long>(q_index) * num_q_heads + head) * head_dim;
        for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
            q_s[d] = bf16_to_f32(q_row[d]);
        }
        __syncthreads();

        // ── pass one: the online (max, sum) over [0, limit) ──────────────
        float running_max = neg_inf();
        float running_sum = 0.f;
        for (int j = warp; j < limit; j += WARPS) {
            const int page_in_req = j / page_size;
            KvSlot slot;
            slot.page = static_cast<int>(kv_page_indices[page_first + page_in_req]);
            slot.offset_in_page = j - page_in_req * page_size;
            const bf16* k_row =
                k_pages + kv_dst_index<HND_LAYOUT>(slot, kv_head * head_dim, page_size,
                                                   num_kv_heads, head_dim);
            float dot = 0.f;
#pragma unroll
            for (int u = 0; u < VPT; ++u) {
                const int d = lane + u * 32;
                if (d < head_dim) {
                    dot += q_s[d] * bf16_to_f32(k_row[d]);
                }
            }
            // XOR rather than DOWN: every lane leaves with the whole score,
            // so the running state below needs no broadcast.
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot += __shfl_xor_sync(0xffffffffu, dot, off);
            }
            const float score = dot * sm_scale;
            const float widened = fmaxf(running_max, score);
            running_sum = running_sum * __expf(running_max - widened) +
                          __expf(score - widened);
            running_max = widened;
        }
        if (lane == 0) {
            wm[warp] = running_max;
            wl[warp] = running_sum;
        }
        __syncthreads();

        float folded_max = neg_inf();
#pragma unroll
        for (int u = 0; u < WARPS; ++u) {
            folded_max = fmaxf(folded_max, wm[u]);
        }
        float denominator = 0.f;
#pragma unroll
        for (int u = 0; u < WARPS; ++u) {
            // A warp that drew no keys folds in as `0 * exp(-inf - M) = 0`.
            denominator += wl[u] * __expf(wm[u] - folded_max);
        }
        const float inv = denominator > 0.f ? 1.f / denominator : 0.f;

        // ── pass two: the same walk, storing the normalised mass ─────────
        for (int j = warp; j < limit; j += WARPS) {
            const int page_in_req = j / page_size;
            KvSlot slot;
            slot.page = static_cast<int>(kv_page_indices[page_first + page_in_req]);
            slot.offset_in_page = j - page_in_req * page_size;
            const bf16* k_row =
                k_pages + kv_dst_index<HND_LAYOUT>(slot, kv_head * head_dim, page_size,
                                                   num_kv_heads, head_dim);
            float dot = 0.f;
#pragma unroll
            for (int u = 0; u < VPT; ++u) {
                const int d = lane + u * 32;
                if (d < head_dim) {
                    dot += q_s[d] * bf16_to_f32(k_row[d]);
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot += __shfl_xor_sync(0xffffffffu, dot, off);
            }
            // One warp owns key `j` for the whole window row, and only its
            // first lane stores — the accumulation needs no atomic.
            if (lane == 0 && j < kv_max) {
                out[j] += __expf(dot * sm_scale - folded_max) * inv * inv_rows;
            }
        }
        // Before the next window row overwrites `q_s`, `wm` and `wl`.
        __syncthreads();
    }
}

}

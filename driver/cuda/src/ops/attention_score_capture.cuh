//===-- attention_score_capture.cuh ------------------------------*- CUDA -*-===//
//
// Attention-score observation for the decode path (design doc
// `inference-time-algorithms/12-attention-observability-design.md` §1.3, §3).
//
// H2O (arXiv:2306.14048), TOVA (arXiv:2305.19370) and SnapKV
// (arXiv:2404.14469) all evict KV positions by accumulated attention mass, so
// they need `p[head, kv_idx]` -- a quantity FlashAttention never materialises,
// because not materialising `S = QK^T` is the entire point of the algorithm.
//
// FlashInfer's supported answer is a custom *attention variant*: a policy
// struct threaded through the kernel as a template parameter, with a
// `LogitsTransform` hook called on every score the kernel computes. That is an
// endorsed extension point, not a fork -- see FlashInfer issue #838, where the
// maintainer confirms score dumping "is feasible by defining your own
// attention variant", and the maintained example
// `tests/utils/test_jit_example.py::DumpLogits`.
//
// The maintainer's one stated objection is the O(n^2) global-memory write. It
// does not bind here: H2O and TOVA observe at *decode*, where `qo_len == 1`, so
// the write is O(kv_len) per step -- 512 KB per layer at a 128K context.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstdint>
#include <cstddef>

#include <flashinfer/attention/variants.cuh>

namespace pie_cuda_driver::ops::fa2 {

/// Ragged score sink, bolted onto a FlashInfer decode params struct.
///
/// FlashInfer duck-types params via `DEFINE_HAS_MEMBER` SFINAE and templates
/// every decode kernel on `typename Params`, so deriving and adding fields is
/// supported by construction rather than by luck.
///
/// Layout is ragged because requests in a decode batch have unrelated
/// `kv_len`s and a dense `[R, H, max_kv_len]` buffer would be mostly padding:
///
///     score_out[score_indptr[b] + h * kv_len(b) + kv_idx]
///
/// `score_indptr` has `R + 1` entries; the last is the total element count.
template <typename Base, typename IdT>
struct PieScoreParams : Base {
    float* score_out = nullptr;
    const IdT* score_indptr = nullptr;
};

/// Wraps a stock `DefaultAttention` instantiation and records every score it
/// transforms. Composition rather than replacement: the base hook still runs,
/// so the *attention output* is bit-identical to the uncaptured path. Capture
/// is a pure side effect.
///
/// Three properties of the decode kernel this relies on, all verified against
/// the pinned FlashInfer v0.6.15 sources:
///
///  1. `kv_idx` is an absolute KV index. `compute_qk`'s `kv_idx_base` is
///     `rope_pos_offset[batch] + chunk_start + ...` (`decode.cuh:539-544`), and
///     pie never sets `rope_pos_offset` -- `paged_kv_t` leaves it null
///     (`page.cuh:77`). If that ever changes, the recorded index shifts
///     silently, so `attention_flashinfer.cu` asserts it stays null.
///
///  2. Every `(qo_head, kv_idx)` pair is written exactly once, so no atomics
///     are needed. One CTA owns one `(request chunk, kv_head)`
///     (`decode.cuh:417-419`), `threadIdx.y` picks the query head within the
///     GQA group, and split-KV chunks are disjoint by construction
///     (`chunk_start = kv_tile_idx * max_chunk_size`, `decode.cuh:427-429`).
///
///  3. Within a CTA the `bdx` lanes of `threadIdx.x` hold *identical* `s[j]`
///     after the butterfly reduction (`decode.cuh:85-88`), so they would all
///     write the same value to the same address. Benign, but `bdx`-fold write
///     amplification -- hence the `threadIdx.x == 0` guard.
///
/// The hook fires *before* the kernel's bounds check (`decode.cuh:91` vs
/// `:98`), so `kv_idx` can run past the sequence. `kv_len` is inherited from
/// `DefaultAttention`, which computed it in its constructor, so the guard is a
/// register compare rather than a reload.
template <class Base>
struct PieScoreCapture : Base {
    // Resolved once per CTA. `score_indptr[batch_idx]` is a *global* load, and
    // leaving it in the hook put one on the kernel's innermost loop, on the
    // dependency path of the store address. `kv_len` is inherited and already
    // cached by `DefaultAttention`; this gives the base the same treatment.
    float* score_row = nullptr;

    template <typename Params>
    __device__ __host__ PieScoreCapture(
        const Params& params, uint32_t batch_idx, uint8_t* smem_ptr)
        : Base(params, batch_idx, smem_ptr) {
        if (params.score_out != nullptr && params.score_indptr != nullptr) {
            score_row = params.score_out +
                        static_cast<std::size_t>(params.score_indptr[batch_idx]);
        }
    }

    REGISTER_LOGITS_TRANSFORM(
        params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
            const T out = this->Base::template LogitsTransform<Params, T>(
                params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx,
                kv_head_idx);
            const uint32_t len = this->kv_len;
            if (threadIdx.x == 0 && kv_idx < len && this->score_row != nullptr) {
                // The scale the kernel is about to apply is
                // `sm_scale * log2e` (it runs softmax in base 2). Record the
                // natural-log-space scaled logit -- what the paper calls the
                // pre-softmax score -- and leave the base change to the
                // normalisation kernel.
                this->score_row[qo_head_idx * len + kv_idx] =
                    static_cast<float>(out) * params.sm_scale;
            }
            return out;
        })
};

}  // namespace pie_cuda_driver::ops::fa2

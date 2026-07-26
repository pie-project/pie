#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver {

struct KvCacheLayerView;

namespace model {

// One layer's attention scores, published by the model body for the duration
// of that layer's `OnAttn` hook.
//
// The counterpart to `AttentionObservation`: that one tells a PTIR program what
// the fire's KV geometry is, this one tells it how much attention each live KV
// position actually received. It is the read side of `AttnScore`.
//
// `values` is ragged and head-folded -- request `r` occupies
// `[offsets[r], offsets[r + 1])`, one float per live KV position, averaged over
// query heads, summing to 1. It is NOT padded to any program's declared ceiling;
// the PTIR side pads per lane because only it knows the declared extent.
//
// `layer` is carried so a consumer can refuse a payload from a different layer
// rather than silently scoring the wrong one. A stale score row is exactly the
// class of failure `12-attention-observability-design.md` is written around.
struct AttentionScores {
    const float* values = nullptr;

    // Host side, `num_requests + 1` entries, in ELEMENTS of `values`.
    const std::uint32_t* offsets_h = nullptr;

    std::uint32_t num_requests = 0;
    std::uint32_t layer = 0;

    bool usable() const noexcept {
        return values != nullptr && offsets_h != nullptr && num_requests > 0;
    }
};

// Null unless a model body is inside an `OnAttn` hook for a layer whose scores
// were captured.
const AttentionScores* active_attention_scores() noexcept;

class ScopedAttentionScores {
  public:
    explicit ScopedAttentionScores(const AttentionScores* scores) noexcept;
    ~ScopedAttentionScores() noexcept;

    ScopedAttentionScores(const ScopedAttentionScores&) = delete;
    ScopedAttentionScores& operator=(const ScopedAttentionScores&) = delete;

  private:
    const AttentionScores* previous_ = nullptr;
};

// RAII capture of one layer's scores, for use inside a model body.
//
// Constructing it is a no-op unless the fire's stage hooks asked for scores
// (`StageHooks::wants_attn_score`) and the fire's geometry is capturable. A
// model family therefore pays nothing when no program observes scores, and the
// call site reads as a plain substitution of one attention dispatch for another:
//
//     model::LayerScoreCapture capture(L, num_q_heads, stream);
//     if (capture.active()) {
//         ops::dispatch_attention_flashinfer_decode_capture(
//             ..., capture.raw(), capture.indptr_d(), ...);
//         capture.publish(kv_page_indptr_d, kv_last_page_lens_d, page_size);
//     } else {
//         ops::dispatch_attention_flashinfer_decode(...);
//     }
//
// The publication lives until the object is destroyed, which must be after the
// layer's `OnAttn` hook has run.
class LayerScoreCapture {
  public:
    // `capturable` is the caller's statement that THIS layer's attention is
    // the plain full-context decode the capture variant implements. A sliding
    // window would make the row describe a truncated context while claiming to
    // describe all of it, so a windowed layer passes false and the PTIR side
    // then fails loudly instead of ranking positions the kernel never scored.
    LayerScoreCapture(
        std::uint32_t layer,
        std::uint32_t num_q_heads,
        bool capturable,
        cudaStream_t stream) noexcept;
    ~LayerScoreCapture();

    LayerScoreCapture(const LayerScoreCapture&) = delete;
    LayerScoreCapture& operator=(const LayerScoreCapture&) = delete;

    bool active() const noexcept { return active_; }
    float* raw() const noexcept { return raw_; }
    const std::int32_t* indptr_d() const noexcept { return indptr_d_; }

    // Fold heads and publish. `page_size` and the two CSR arrays must be the
    // ones the capture dispatch itself was given.
    void publish(
        const std::uint32_t* kv_page_indptr_d,
        const std::uint32_t* kv_last_page_lens_d,
        int page_size);

  private:
    void release() noexcept;

    bool active_ = false;
    bool published_ = false;
    cudaStream_t stream_ = nullptr;
    std::uint32_t layer_ = 0;
    std::uint32_t num_q_heads_ = 0;

    float* raw_ = nullptr;
    float* folded_ = nullptr;
    std::int32_t* indptr_d_ = nullptr;

    // `raw` element offsets (host), `num_requests + 1`; the folded offsets are
    // these divided by `num_q_heads`.
    const std::uint32_t* folded_offsets_h_ = nullptr;

    AttentionScores payload_{};
    ScopedAttentionScores* binding_ = nullptr;
};

}  // namespace model
}  // namespace pie_cuda_driver

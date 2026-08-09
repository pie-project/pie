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

// The prefill observation window this build uses: SnapKV's default of 32 query
// rows, overridable with `PIE_ATTN_SCORE_WINDOW` for experiments. SnapKV's own
// ablation (arXiv:2404.14469 §4) finds the choice broadly insensitive across
// 8-64, so a constant is honest rather than a placeholder -- and a wider window
// is strictly more evidence, averaged.
std::uint32_t default_attn_score_window() noexcept;

struct StageHooks;
class HookSidebandArena;
struct AttentionObservation;

// Stage 6 increment 4 — the hook-graph prepare pass's fire-level view of the
// DECODE score capture. A captured hook body replays `LayerScoreCapture`'s
// stream ops (the CSR upload and the folded memset) with capture-time
// parameters, so before EVERY replay the prepare pass must (a) refresh the
// host CSR the captured upload reads — same thread-local storage, same
// contents the constructor would compute — and (b) pre-grow the arena score
// slot so the capture-time constructor (and nothing at replay) ever hits the
// arena's stream-synced growth path. This helper does both and hands back
// the arena-stable addresses the pass folds into its fingerprint. It
// enqueues NO stream work and takes NO slot: the arena acquire is released
// before returning (growth is its only lasting effect).
struct DecodeScoreCapturePlan {
    bool ok = false;
    // Arena-stable device addresses (stable until the score region grows).
    const float* folded = nullptr;
    const std::int32_t* indptr_d = nullptr;
    // The host raw-offset CSR the captured `cudaMemcpyAsync` reads at replay
    // time. Thread-local storage: the ADDRESS must be fingerprinted, because
    // a different fire shape may reallocate the vector under the captured
    // node's recorded source pointer.
    const void* indptr_h_data = nullptr;
    // Folded (per-position) offsets, host, `num_requests + 1` entries.
    const std::uint32_t* folded_offsets_h = nullptr;
    std::uint32_t num_requests = 0;
};

DecodeScoreCapturePlan prepare_decode_score_capture(
    HookSidebandArena* arena,
    const AttentionObservation& observation,
    std::uint32_t num_q_heads,
    cudaStream_t stream);

namespace detail {

// The three device buffers every score capture needs: the raw per-head rows the
// kernel writes, the head-folded row a PTIR program reads, and the ragged CSR
// both are addressed by. Shared by the decode and prefill captures so the two
// cannot drift in how they acquire, zero, or release -- the failure mode of a
// drift here is a program reading a row that was never zeroed, which looks like
// a plausible attention distribution rather than like a bug.
//
// The bytes live in the fire's `HookSidebandArena` score slot rather than in a
// fresh `cudaMallocAsync` per layer: every layer of a fire has identical
// geometry and at most one capture is live at a time, so one slot sized on the
// first layer serves the whole fire, and the addresses stay stable across
// layers AND across same-geometry fires (the increment-4 graph-capture
// precondition — see hook_sideband_arena.hpp).
struct ScoreBuffers {
    float* raw = nullptr;
    float* folded = nullptr;
    std::int32_t* indptr_d = nullptr;

    // Carve raw/folded/indptr out of the arena's score slot and stage the
    // host CSR into `indptr_d`.
    //
    // `folded` is zeroed but `raw` is not — and because the slot is REUSED,
    // that asymmetry is now load-bearing across layers too: every element of
    // `folded` that the fold kernel skips must still read as "this position
    // received no attention" (so it is re-zeroed on every acquire), whereas
    // `raw` is only ever read back by the fold kernel over exactly the region
    // the capture kernel wrote this layer, so the previous layer's leftovers
    // in it are dead bytes by construction.
    bool acquire(
        HookSidebandArena* arena,
        std::uint64_t raw_elems,
        std::uint64_t folded_elems,
        const std::int32_t* indptr_h,
        std::uint32_t num_requests,
        cudaStream_t stream) noexcept;
    void release() noexcept;

  private:
    // The arena the slot was acquired from; null while nothing is held.
    HookSidebandArena* arena_ = nullptr;
};

}  // namespace detail

// RAII capture of one layer's scores, for use inside a model body.
//
// Constructing it is a no-op unless the fire's stage hooks asked for scores
// (`StageHooks::wants_attn_score`) and the fire's geometry is capturable. A
// model family therefore pays nothing when no program observes scores, and the
// call site reads as a plain substitution of one attention dispatch for another:
//
//     model::LayerScoreCapture capture(hooks, L, num_q_heads, stream);
//     if (capture.active()) {
//         kernels::attn::dispatch_attention_flashinfer_decode_capture(
//             ..., capture.raw(), capture.indptr_d(), ...);
//         capture.publish(kv_page_indptr_d, kv_last_page_lens_d, page_size);
//     } else {
//         kernels::attn::dispatch_attention_flashinfer_decode(...);
//     }
//     invoke_stage_hook(hooks, OnAttn, ..., {.scores = capture.scores()});
//
// The payload lives until the object is destroyed, which must be after the
// layer's `OnAttn` hook has run.
class LayerScoreCapture {
  public:
    // `capturable` is the caller's statement that THIS layer's attention is
    // the plain full-context decode the capture variant implements. A sliding
    // window would make the row describe a truncated context while claiming to
    // describe all of it, so a windowed layer passes false and the PTIR side
    // then fails loudly instead of ranking positions the kernel never scored.
    LayerScoreCapture(
        const StageHooks* hooks,
        std::uint32_t layer,
        std::uint32_t num_q_heads,
        bool capturable,
        cudaStream_t stream) noexcept;
    ~LayerScoreCapture();

    LayerScoreCapture(const LayerScoreCapture&) = delete;
    LayerScoreCapture& operator=(const LayerScoreCapture&) = delete;

    bool active() const noexcept { return active_; }
    float* raw() const noexcept { return buf_.raw; }
    const std::int32_t* indptr_d() const noexcept { return buf_.indptr_d; }

    // Fold heads and finalise the payload. `page_size` and the two CSR arrays
    // must be the ones the capture dispatch itself was given.
    void publish(
        const std::uint32_t* kv_page_indptr_d,
        const std::uint32_t* kv_last_page_lens_d,
        int page_size);

    // The published payload, for the layer's `OnAttn` sideband; null until
    // `publish()` ran.
    const AttentionScores* scores() const noexcept {
        return published_ ? &payload_ : nullptr;
    }

  private:
    void release() noexcept;

    bool active_ = false;
    bool published_ = false;
    cudaStream_t stream_ = nullptr;
    std::uint32_t layer_ = 0;
    std::uint32_t num_q_heads_ = 0;

    // Borrowed from the constructor; outlives the capture (the fire's hooks
    // bracket the whole body).
    const StageHooks* hooks_ = nullptr;

    detail::ScoreBuffers buf_{};

    // `raw` element offsets (host), `num_requests + 1`; the folded offsets are
    // these divided by `num_q_heads`.
    const std::uint32_t* folded_offsets_h_ = nullptr;

    AttentionScores payload_{};
};

// RAII capture of one layer's PREFILL scores -- SnapKV's observation window.
//
// The decode capture records one query row per request, because that is all a
// decode step has. A prefill step has `qo_len` of them, and SnapKV's selection
// is defined over only the last few: it asks which prefix positions the TAIL of
// the prompt attended to, on the theory that the tail is the best available
// proxy for what the not-yet-generated continuation will attend to.
//
// So this records the last `window` query rows and folds over both heads and
// window rows, publishing the same `AttentionScores` shape the decode capture
// does. A program cannot tell which one produced its row, which is the point:
// `AttnScore` means "how much attention did each live KV position receive",
// and that is well-defined at both stages.
//
// The observation window is the last `window` rows OF THIS FORWARD CHUNK. For a
// one-shot prefill that is the last `window` tokens of the prompt, which is
// SnapKV's definition. For a chunked prefill the hook fires once per chunk and
// the final firing is the one whose window matches SnapKV; earlier firings are
// well-defined observations of earlier windows, so a policy that simply acts on
// the most recent firing gets SnapKV's semantics for free.
class LayerPrefillScoreCapture {
  public:
    LayerPrefillScoreCapture(
        const StageHooks* hooks,
        std::uint32_t layer,
        std::uint32_t num_q_heads,
        std::uint32_t window,
        bool capturable,
        cudaStream_t stream) noexcept;
    ~LayerPrefillScoreCapture();

    LayerPrefillScoreCapture(const LayerPrefillScoreCapture&) = delete;
    LayerPrefillScoreCapture& operator=(const LayerPrefillScoreCapture&) =
        delete;

    bool active() const noexcept { return active_; }
    float* raw() const noexcept { return buf_.raw; }
    float* folded() const noexcept { return buf_.folded; }
    const std::int32_t* indptr_d() const noexcept { return buf_.indptr_d; }
    int window() const noexcept { return static_cast<int>(window_); }

    // Finalise the folded row for the layer's `OnAttn` sideband. Unlike the
    // decode capture this launches nothing: normalisation and folding are part
    // of the capture dispatch, because the causal limit of each window row is
    // only derivable there.
    void publish();

    const AttentionScores* scores() const noexcept {
        return published_ ? &payload_ : nullptr;
    }

  private:
    void release() noexcept;

    bool active_ = false;
    bool published_ = false;
    cudaStream_t stream_ = nullptr;
    std::uint32_t layer_ = 0;
    std::uint32_t num_q_heads_ = 0;
    std::uint32_t window_ = 0;
    std::uint32_t num_requests_ = 0;

    detail::ScoreBuffers buf_{};
    const std::uint32_t* folded_offsets_h_ = nullptr;

    AttentionScores payload_{};
};

}  // namespace model
}  // namespace pie_cuda_driver

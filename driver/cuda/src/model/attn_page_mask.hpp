#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::model {

/// The write side of the attention hook.
///
/// `attn_page_mask` is a *configuration* sink (PTIR overview 6.1): the program
/// hands the backend a per-page keep mask and the layer's attention is expected
/// to honour it. Everything else that crosses this boundary flows model ->
/// dispatch (`AttentionObservation`), and this deliberately flows the same way:
/// the **model body owns the buffer**. The dispatch only fills it.
///
/// That ownership is not a style choice. The mask has to outlive the PTIR stage
/// that produced it -- it is consumed by the decode call that runs *after* the
/// hook returns -- and `execute_declared_phase` frees its own temporaries at
/// scope exit. A dispatch-owned mask would therefore either leak or dangle. The
/// model, which already knows the fire's page geometry and already brackets the
/// layer, has exactly the right lifetime.
struct AttentionMaskSink {
    /// `[num_requests, stride]` u8, row-major; 1 keeps the page. Entry
    /// `[r, p]` governs slot `p` of request `r`'s page list. Pre-filled with 1
    /// by the owner before every hook, so a program that emits no sink for this
    /// layer leaves attention unrestricted rather than evicting everything.
    ///
    /// The **fixed stride is the point.** The obvious layout -- one entry per
    /// page, sliced by the fire's page CSR -- forces the writer (host) and the
    /// reader (a device kernel walking the real page table) to agree on that
    /// CSR, and they do not: on the decode-envelope path the host holds a
    /// conservative BOUND while the device holds geometry it resolved itself
    /// (`frame.cpp`, `FixedDecodeDeviceBuffers`). A per-request stride removes
    /// the page CSR from the mask's addressing entirely, so there is nothing
    /// left to disagree about -- the only shared fact is "slot p of request r",
    /// which is exactly what the program means when it writes `mask[p]`.
    std::uint8_t* keep = nullptr;

    std::uint32_t num_requests = 0;

    /// Entries per request. An upper bound on any request's page count, so a
    /// row is always at least as long as the page list it governs.
    std::uint32_t stride = 0;

    /// Layer whose sink last wrote `keep`, or -1 for "nothing written". The tag
    /// is what stops a mask computed for layer L from silently governing layer
    /// L+1 when the program stops emitting the sink -- the same class of stale-
    /// view bug the layer tag on `AttentionScores` prevents.
    int written_layer = -1;

    bool usable() const noexcept {
        return keep != nullptr && num_requests > 0 && stride > 0;
    }
};

struct StageHooks;
class HookSidebandArena;
struct AttentionObservation;

/// Stage 6 increment 4 — the hook-graph prepare pass's fire-level view of the
/// page-mask sideband. A captured hook body bakes the mask buffers' addresses
/// (the PTIR sink kernel's destination rows, the seeding memset, the
/// compaction kernel's outputs the attention then reads), so the prepare pass
/// must know — before the body constructs its `FirePageMask` — exactly where
/// the arena's mask slot will carve them, and must pre-grow the slot so no
/// growth (a stream-synced free+realloc) can happen inside a captured region.
/// Mirrors `FirePageMask`'s constructor byte-for-byte via a shared layout
/// function; enqueues no stream work and holds no slot.
struct PageMaskCapturePlan {
    bool ok = false;
    std::uint8_t* keep = nullptr;
    std::uint32_t num_requests = 0;
    std::uint32_t stride = 0;
    const std::uint32_t* out_indices = nullptr;
    const std::uint32_t* out_indptr = nullptr;
    const std::uint32_t* out_last_lens = nullptr;
};

PageMaskCapturePlan prepare_page_mask_capture(
    HookSidebandArena* arena,
    const AttentionObservation& observation,
    cudaStream_t stream);

/// Fire-scoped owner of the page mask and of the compacted CSR it produces.
///
/// One instance brackets the whole layer loop: the keep buffer and the
/// compaction outputs are acquired once and reused by every layer, because the
/// page geometry is a property of the fire, not of the layer. The bytes come
/// from the fire's `HookSidebandArena` mask slot (via `hooks`), so across
/// fires there is no allocation at all once the arena has grown to the
/// workload's max — and the addresses stay stable while it suffices, the
/// increment-4 graph-capture precondition (hook_sideband_arena.hpp).
///
/// Usage per layer:
///
///     mask.begin_layer(stream);          // re-seed to "keep everything"
///     invoke_stage_hook(hooks, OnAttnProj, ...,
///                       {.mask_sink = mask.sink()});  // the sink may write
///     if (mask.written_for(L)) mask.compact(...);
///     ... attention, with mask.page_indices() when compacted ...
class FirePageMask {
  public:
    /// Reads `wants_page_mask` — the launch's own answer to "does any program
    /// write the sink" — and the fire geometry off `hooks`. The fire's host
    /// page CSR is used only to SIZE the rows (as an upper bound); it never
    /// addresses them, so a conservative host CSR costs a little memory and
    /// nothing else.
    FirePageMask(const StageHooks* hooks, cudaStream_t stream);
    ~FirePageMask();

    FirePageMask(const FirePageMask&) = delete;
    FirePageMask& operator=(const FirePageMask&) = delete;

    bool active() const noexcept { return active_; }

    void begin_layer(cudaStream_t stream) noexcept;
    bool written_for(std::uint32_t layer) const noexcept {
        return active_ && sink_.written_layer >= 0 &&
               static_cast<std::uint32_t>(sink_.written_layer) == layer;
    }

    /// Gather the fire's page table down to the kept pages. Inputs are the
    /// device CSR the layer would otherwise attend over; they are not modified.
    void compact(
        const std::uint32_t* page_indices_d,
        const std::uint32_t* page_indptr_d,
        const std::uint32_t* last_page_lens_d,
        std::uint32_t num_requests,
        cudaStream_t stream);

    const std::uint32_t* page_indices() const noexcept { return out_indices_; }
    const std::uint32_t* page_indptr() const noexcept { return out_indptr_; }
    const std::uint32_t* last_page_lens() const noexcept {
        return out_last_lens_;
    }

    /// The write destination, for the layer's `OnAttnProj` sideband; null when
    /// the fire wants no mask.
    AttentionMaskSink* sink() noexcept { return active_ ? &sink_ : nullptr; }

  private:
    AttentionMaskSink sink_{};
    std::uint32_t* out_indices_ = nullptr;
    std::uint32_t* out_indptr_ = nullptr;
    std::uint32_t* out_last_lens_ = nullptr;
    // Scratch for the compaction's per-request survivor counts. Acquired once
    // per fire and reused by all 28-odd layers, rather than once per layer.
    std::uint32_t* counts_ = nullptr;
    // The arena the mask slot was acquired from; null when the fire wants no
    // mask. Released (not freed) in the destructor.
    HookSidebandArena* arena_ = nullptr;
    bool active_ = false;
};

}  // namespace pie_cuda_driver::model

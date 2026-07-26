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

/// Null outside a model body, or inside one whose fire runs no page-mask sink.
AttentionMaskSink* active_attention_mask_sink() noexcept;

class ScopedAttentionMaskSink {
  public:
    explicit ScopedAttentionMaskSink(AttentionMaskSink* sink) noexcept;
    ~ScopedAttentionMaskSink() noexcept;

    ScopedAttentionMaskSink(const ScopedAttentionMaskSink&) = delete;
    ScopedAttentionMaskSink& operator=(const ScopedAttentionMaskSink&) = delete;

  private:
    AttentionMaskSink* previous_ = nullptr;
};

/// Fire-scoped owner of the page mask and of the compacted CSR it produces.
///
/// One instance brackets the whole layer loop: the keep buffer and the
/// compaction outputs are allocated once and reused by every layer, because the
/// page geometry is a property of the fire, not of the layer.
///
/// Usage per layer:
///
///     mask.begin_layer(stream);          // re-seed to "keep everything"
///     invoke_stage_hook(OnAttnProj, ...); // the sink may write into it
///     if (mask.written_for(L)) mask.compact(...);
///     ... attention, with mask.page_indices() when compacted ...
class FirePageMask {
  public:
    /// `wanted` is the launch's own answer to "does any program write the
    /// sink". The fire's host page CSR is used only to SIZE the rows (as an
    /// upper bound); it never addresses them, so a conservative host CSR costs
    /// a little memory and nothing else.
    FirePageMask(bool wanted, cudaStream_t stream);
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

  private:
    AttentionMaskSink sink_{};
    ScopedAttentionMaskSink* binding_ = nullptr;
    std::uint32_t* out_indices_ = nullptr;
    std::uint32_t* out_indptr_ = nullptr;
    std::uint32_t* out_last_lens_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool active_ = false;
};

}  // namespace pie_cuda_driver::model

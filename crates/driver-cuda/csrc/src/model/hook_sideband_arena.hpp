#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::model {

// Grow-only device arena for the hook sidebands: the per-layer score-capture
// buffers (`ScoreBuffers` in attn_score.hpp) and the per-fire page-mask
// buffers (`FirePageMask`). Before this arena existed those were
// `cudaMallocAsync`/`cudaFreeAsync` churn on the hot path — 3 allocations per
// layer per fire for the scores plus 5 per fire for the mask, ~90 alloc/free
// pairs on a 28-layer hook fire.
//
// Owned beside `model::Workspace` (context.cpp) — engine lifetime, one per
// context, never static — and handed to the model body on the fire's
// `StageHooks`. Like `Workspace`, it is confined to the single lane thread
// that runs fires; nothing here is thread-safe.
//
// Shape: two independent single-slot regions, one per sideband.
//
//  * `Region::Score` — ONE slot, re-acquired every layer. The decode and
//    prefill score captures are mutually exclusive within a layer and layers
//    run in sequence, so at most one capture is live at a time and every
//    layer of a fire has identical geometry; the slot therefore reaches the
//    fire's max on the first layer and every later acquire is a pointer
//    return. The busy flag turns any future overlap into a loud refusal
//    instead of two captures silently sharing bytes.
//  * `Region::Mask` — ONE slot, acquired once per fire by `FirePageMask` and
//    held for the whole layer loop.
//  * `Region::ScoreRows` — ONE slot, acquired transiently per fire by the
//    hook-graph prepare pass (stage 6 increment 4): the folded-offset device
//    CSR plus one padded `[kv_max]` f32 row per score-reading (layer, lane).
//    Acquired and released within the prepare pass — the busy flag only
//    guards overlapping acquisition; the contents are produced and consumed
//    in stream order inside one fire's body.
//
// "Allocation" is thus a capacity check; the caller carves its sub-buffers as
// offsets into the returned block. A fire that needs more than the region has
// grows it: stream-synced free + realloc, logged once per growth (the ring /
// channel-registry growth idiom), rare after warmup.
//
// GRAPH-CAPTURE PRECONDITION (stage 6 increment 4): while a region's capacity
// suffices, the addresses it hands out are STABLE across fires of the same
// geometry — that is what lets a captured hook fire replay against the same
// sideband pointers. A growth moves the region and bumps `generation()`;
// anything that baked arena addresses (a captured graph) must treat a
// generation change as invalidation.
class HookSidebandArena {
  public:
    enum class Region : int { Score = 0, Mask = 1, ScoreRows = 2 };

    HookSidebandArena() = default;
    ~HookSidebandArena();

    HookSidebandArena(const HookSidebandArena&) = delete;
    HookSidebandArena& operator=(const HookSidebandArena&) = delete;

    // The region's base pointer with at least `bytes` of capacity, growing
    // the backing allocation when this fire needs more. Returns nullptr when
    // the region is already held (overlapping captures — a bug upstream) or
    // when device allocation fails; the callers' existing refusal paths
    // handle both. The block's contents are NOT zeroed: each caller owns its
    // zeroing discipline (attn_score.hpp explains why `folded` must be
    // zeroed per use while `raw` must not be).
    void* acquire(Region region, std::size_t bytes, cudaStream_t stream) noexcept;

    // Release the region's slot (per layer for Score, per fire for Mask).
    // Frees nothing — the backing allocation is reused by the next acquire.
    void release(Region region) noexcept;

    // Bumped on every growth. See the graph-capture precondition above.
    std::uint64_t generation() const noexcept { return generation_; }

    // Fire boundary for the `PIE_SIDEBAND_TRACE=1` evidence counters: logs
    // the finished fire's acquire/growth counts (each acquire is one
    // cudaMallocAsync the pre-arena code would have issued; growths are the
    // only real device allocations left) and resets the per-fire counts.
    void begin_fire() noexcept;

  private:
    struct Slot {
        std::uint8_t* base = nullptr;
        std::size_t capacity = 0;
        bool busy = false;
    };

    static const char* region_name(Region region) noexcept;

    Slot slots_[3];
    std::uint64_t generation_ = 0;

    // PIE_SIDEBAND_TRACE=1 evidence counters.
    std::uint64_t fire_index_ = 0;
    std::uint32_t fire_acquires_ = 0;
    std::uint32_t fire_grows_ = 0;
    std::uint64_t total_acquires_ = 0;
    std::uint64_t total_grows_ = 0;
};

}  // namespace pie_cuda_driver::model

#include "model/attn_page_mask.hpp"

#include <algorithm>
#include <new>
#include <stdexcept>

#include "kernels/page_compact.hpp"
#include "model/attn_observation.hpp"
#include "model/hook_sideband_arena.hpp"
#include "model/stage_hooks.hpp"

namespace pie_cuda_driver::model {

namespace {

// Sub-buffer alignment inside the arena's mask slot (mirrors attn_score.cu).
constexpr std::size_t kSidebandAlign = 256;

constexpr std::size_t align_up(std::size_t n) noexcept {
    return (n + kSidebandAlign - 1) & ~(kSidebandAlign - 1);
}

}  // namespace

FirePageMask::FirePageMask(const StageHooks* hooks, cudaStream_t stream) {
    if (hooks == nullptr || !hooks->wants_page_mask) return;

    const AttentionObservation* obs = hooks->observation;
    if (obs == nullptr || !obs->usable()) {
        throw std::runtime_error(
            "attn_page_mask needs a fire with kv geometry");
    }
    const std::uint32_t requests =
        static_cast<std::uint32_t>(obs->num_requests);
    const std::uint32_t total_pages = obs->kv_page_indptr_h[requests];
    if (total_pages == 0) {
        throw std::runtime_error("attn_page_mask fire has no kv pages");
    }

    // Rows are sized from the host CSR and addressed by request index, so a
    // conservative host CSR only over-allocates. The widest request sets the
    // stride: every row is then at least as long as the page list it governs,
    // whatever the device later resolves that list to be.
    std::uint32_t stride = 0;
    for (std::uint32_t r = 0; r < requests; ++r) {
        const std::uint32_t begin = obs->kv_page_indptr_h[r];
        const std::uint32_t end = obs->kv_page_indptr_h[r + 1];
        if (end < begin || end > total_pages) {
            throw std::runtime_error("attn_page_mask saw a malformed page CSR");
        }
        stride = std::max(stride, end - begin);
    }
    if (stride == 0) {
        throw std::runtime_error("attn_page_mask fire has no kv pages");
    }
    const std::size_t keep_bytes =
        static_cast<std::size_t>(requests) * static_cast<std::size_t>(stride);

    const std::size_t idx_bytes =
        static_cast<std::size_t>(total_pages) * sizeof(std::uint32_t);
    const std::size_t indptr_bytes =
        (static_cast<std::size_t>(requests) + 1) * sizeof(std::uint32_t);
    const std::size_t lens_bytes =
        static_cast<std::size_t>(requests) * sizeof(std::uint32_t);

    // One arena slot for all five buffers, acquired once per fire and carved
    // by offset: the u32 outputs first, the u8 keep rows last, each aligned.
    // Reuse across fires is safe because every buffer is (re)written before
    // it is read — `begin_layer` re-seeds `keep` every layer, and the
    // compaction outputs are written by `compact` before attention reads
    // them; nothing here needs a fresh-allocation guarantee.
    if (hooks->sideband_arena == nullptr) {
        throw std::runtime_error(
            "attn_page_mask fire carries no hook sideband arena");
    }
    auto* base = static_cast<std::uint8_t*>(hooks->sideband_arena->acquire(
        HookSidebandArena::Region::Mask,
        align_up(idx_bytes) + align_up(indptr_bytes) + 2 * align_up(lens_bytes) +
            keep_bytes,
        stream));
    if (base == nullptr) {
        throw std::runtime_error(
            "attn_page_mask could not acquire its page buffers");
    }
    arena_ = hooks->sideband_arena;
    out_indices_ = reinterpret_cast<std::uint32_t*>(base);
    base += align_up(idx_bytes);
    out_indptr_ = reinterpret_cast<std::uint32_t*>(base);
    base += align_up(indptr_bytes);
    counts_ = reinterpret_cast<std::uint32_t*>(base);
    base += align_up(lens_bytes);
    out_last_lens_ = reinterpret_cast<std::uint32_t*>(base);
    base += align_up(lens_bytes);
    std::uint8_t* keep = base;

    sink_.keep = keep;
    sink_.num_requests = requests;
    sink_.stride = stride;
    sink_.written_layer = -1;
    active_ = true;
}

void FirePageMask::begin_layer(cudaStream_t stream) noexcept {
    if (!active_) return;
    // Seed to "keep everything". A layer whose program emits no sink then
    // attends over its full page list, which is the only safe default -- an
    // all-zero seed would evict the whole cache for any layer the policy chose
    // not to score.
    cudaMemsetAsync(
        sink_.keep, 1,
        static_cast<std::size_t>(sink_.num_requests) *
            static_cast<std::size_t>(sink_.stride),
        stream);
    sink_.written_layer = -1;
}

void FirePageMask::compact(
    const std::uint32_t* page_indices_d,
    const std::uint32_t* page_indptr_d,
    const std::uint32_t* last_page_lens_d,
    std::uint32_t num_requests,
    cudaStream_t stream) {
    if (!active_) return;
    if (num_requests != sink_.num_requests) {
        throw std::runtime_error(
            "attn_page_mask compaction and fire disagree on request count");
    }
    kernels::launch_compact_page_csr(
        page_indices_d, page_indptr_d, last_page_lens_d, sink_.keep, counts_,
        sink_.stride, static_cast<int>(sink_.num_requests), out_indices_,
        out_indptr_, out_last_lens_, stream);
}

FirePageMask::~FirePageMask() {
    // Nothing to free: the bytes belong to the arena and are handed back for
    // the next fire's mask to reuse.
    if (arena_ != nullptr) {
        arena_->release(HookSidebandArena::Region::Mask);
    }
    arena_ = nullptr;
    sink_ = AttentionMaskSink{};
    out_indices_ = nullptr;
    out_indptr_ = nullptr;
    out_last_lens_ = nullptr;
    counts_ = nullptr;
    active_ = false;
}

}  // namespace pie_cuda_driver::model

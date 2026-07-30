#include "model/attn_page_mask.hpp"

#include <algorithm>
#include <new>
#include <stdexcept>

#include "kernels/page_compact.hpp"
#include "model/attn_observation.hpp"
#include "model/stage_hooks.hpp"

namespace pie_cuda_driver::model {

FirePageMask::FirePageMask(const StageHooks* hooks, cudaStream_t stream)
    : stream_(stream) {
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

    std::uint8_t* keep = nullptr;
    const bool allocated =
        cudaMallocAsync(
            reinterpret_cast<void**>(&keep), keep_bytes, stream) ==
            cudaSuccess &&
        cudaMallocAsync(
            reinterpret_cast<void**>(&out_indices_), idx_bytes, stream) ==
            cudaSuccess &&
        cudaMallocAsync(
            reinterpret_cast<void**>(&out_indptr_), indptr_bytes, stream) ==
            cudaSuccess &&
        cudaMallocAsync(
            reinterpret_cast<void**>(&counts_), lens_bytes, stream) ==
            cudaSuccess &&
        cudaMallocAsync(
            reinterpret_cast<void**>(&out_last_lens_), lens_bytes, stream) ==
            cudaSuccess;
    if (!allocated) {
        if (keep != nullptr) cudaFreeAsync(keep, stream);
        if (out_indices_ != nullptr) cudaFreeAsync(out_indices_, stream);
        if (out_indptr_ != nullptr) cudaFreeAsync(out_indptr_, stream);
        if (out_last_lens_ != nullptr) cudaFreeAsync(out_last_lens_, stream);
        if (counts_ != nullptr) cudaFreeAsync(counts_, stream);
        out_indices_ = nullptr;
        out_indptr_ = nullptr;
        out_last_lens_ = nullptr;
        counts_ = nullptr;
        throw std::runtime_error(
            "attn_page_mask could not allocate its page buffers");
    }

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
    if (sink_.keep != nullptr) cudaFreeAsync(sink_.keep, stream_);
    if (out_indices_ != nullptr) cudaFreeAsync(out_indices_, stream_);
    if (out_indptr_ != nullptr) cudaFreeAsync(out_indptr_, stream_);
    if (out_last_lens_ != nullptr) cudaFreeAsync(out_last_lens_, stream_);
    if (counts_ != nullptr) cudaFreeAsync(counts_, stream_);
    sink_ = AttentionMaskSink{};
    out_indices_ = nullptr;
    out_indptr_ = nullptr;
    out_last_lens_ = nullptr;
    active_ = false;
}

}  // namespace pie_cuda_driver::model

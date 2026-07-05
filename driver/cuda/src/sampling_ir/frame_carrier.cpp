// X2 — CUDA frames/mirrors carrier implementation. See frame_carrier.hpp.
//
// The real-device backing of X1's `FrameAddresses` triple + the direct
// driver<->inferlet frame transport. Built + device-verified in isolation
// (test_frame_carrier_device) before it composes into the executor, exactly like
// the tensor_io.cpp substrate it mirrors.

#include "sampling_ir/frame_carrier.hpp"

#include "sampling_ir/frame_carrier_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace pie_cuda_driver::sampling_ir {

namespace {
// Fail-loud local check (the module is built into the light isolation target with
// no driver-lib include; the executor-grade CUDA_CHECK is picked up at the cut) —
// the tensor_io.cpp `ck` idiom.
inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[frame_carrier] CUDA error: %s (%s)\n",
                     cudaGetErrorString(e), what);
        std::abort();
    }
}

// Provisional trace-derived sizing — identical in shape to X1's `MockProgram`
// (frame sized from the trace, tiny mirror/word rings). guru swaps this for the
// real channel-list layout at the reconcile.
constexpr std::size_t kMinFrameBytes = 64;
constexpr std::size_t kMirrorBytes = 64;
constexpr std::size_t kWordBytes = 64;  // 8 u64 ring-index words
}  // namespace

#define FC_CK(x) ck((x), #x)

FrameCarrierEngine& FrameCarrierEngine::instance() {
    static FrameCarrierEngine engine;
    return engine;
}

FrameCarrierEngine::FrameCarrierEngine() {
    // Dedicated non-blocking copy stream (the tensor_io.cpp / weight_copy_engine
    // pattern) — the carrier's copies never serialize against the forward stream.
    //
    // LOAD-BEARING INVARIANT: this is the ONE copy stream every carry (all instances
    // + fires) runs on. Its FIFO ordering is what makes an instance's carries commit
    // in enqueue order → the monotonic head lands monotonically (no regression) AND
    // the D2H mirror stays fresh, even under X4 depth-2 run-ahead; and it's what
    // `close_instance`'s single `cudaStreamSynchronize(stream_)` drains. If this ever
    // becomes multi-stream-per-instance (for D2H parallelism), the head publish must
    // switch to a monotonic-max store and close must drain per-instance events.
    FC_CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

FrameCarrierEngine::~FrameCarrierEngine() {
    for (auto& kv : instances_) {
        FrameInstance* fi = kv.second;
        if (fi == nullptr) continue;
        if (fi->device_frame) cudaFree(fi->device_frame);
        if (fi->host_mirror) cudaFreeHost(fi->host_mirror);
        if (fi->host_words) cudaFreeHost(fi->host_words);
        delete fi;
    }
    if (stream_) cudaStreamDestroy(stream_);
}

std::uint64_t FrameCarrierEngine::register_program(const std::uint8_t* /*trace*/,
                                                   std::size_t trace_len) {
    std::lock_guard<std::mutex> guard(mu_);
    programs_.push_back(FrameLayout{
        /*frame_bytes=*/std::max(trace_len, kMinFrameBytes),
        /*mirror_bytes=*/kMirrorBytes,
        /*word_bytes=*/kWordBytes,
    });
    return next_program_++;  // 1-based; the just-pushed layout is at id-1
}

std::uint64_t FrameCarrierEngine::bind_instance(std::uint64_t program,
                                                std::uint64_t* out_frame_base,
                                                std::uint64_t* out_mirror_base,
                                                std::uint64_t* out_word_base) {
    std::lock_guard<std::mutex> guard(mu_);
    const std::size_t pidx = static_cast<std::size_t>(program - 1);
    if (program == 0 || pidx >= programs_.size()) {
        return 0;  // unknown program — the runtime turns 0 into a loud Err
    }
    const FrameLayout layout = programs_[pidx];

    auto* fi = new FrameInstance();
    fi->program = program;
    fi->frame_bytes = layout.frame_bytes;
    fi->mirror_bytes = layout.mirror_bytes;
    fi->word_bytes = layout.word_bytes;
    // Dedicated backing allocations — the bases are FIXED for the instance's
    // lifetime (B6), so they are never slab-recycled; freed only at close.
    FC_CK(cudaMalloc(&fi->device_frame, layout.frame_bytes));
    FC_CK(cudaHostAlloc(&fi->host_mirror, layout.mirror_bytes, cudaHostAllocDefault));
    FC_CK(cudaHostAlloc(reinterpret_cast<void**>(&fi->host_words), layout.word_bytes,
                        cudaHostAllocDefault));
    // Zero the pinned ring words so a waiter never observes a stale index (B9).
    std::fill(reinterpret_cast<char*>(fi->host_words),
              reinterpret_cast<char*>(fi->host_words) + layout.word_bytes, 0);

    const std::uint64_t id = next_instance_++;
    instances_[id] = fi;

    *out_frame_base = reinterpret_cast<std::uint64_t>(fi->device_frame);
    *out_mirror_base = reinterpret_cast<std::uint64_t>(fi->host_mirror);
    *out_word_base = reinterpret_cast<std::uint64_t>(fi->host_words);
    return id;
}

FrameInstance* FrameCarrierEngine::lookup(std::uint64_t instance) {
    auto it = instances_.find(instance);
    return it == instances_.end() ? nullptr : it->second;
}

std::uint64_t FrameCarrierEngine::bind_channels(std::uint32_t n_channels,
                                                const std::uint32_t* cell_bytes,
                                                const std::uint32_t* cap1,
                                                std::uint64_t* out_frame_base,
                                                std::uint64_t* out_mirror_base,
                                                std::uint64_t* out_word_base) {
    std::lock_guard<std::mutex> guard(mu_);
    const std::uint64_t id = next_instance_++;
    return bind_channels_locked(id, n_channels, cell_bytes, cap1, out_frame_base,
                                out_mirror_base, out_word_base);
}

std::uint64_t FrameCarrierEngine::bind_channels_keyed(std::uint64_t key,
                                                      std::uint32_t n_channels,
                                                      const std::uint32_t* cell_bytes,
                                                      const std::uint32_t* cap1,
                                                      std::uint64_t* out_frame_base,
                                                      std::uint64_t* out_mirror_base,
                                                      std::uint64_t* out_word_base) {
    std::lock_guard<std::mutex> guard(mu_);
    return bind_channels_locked(key, n_channels, cell_bytes, cap1, out_frame_base,
                                out_mirror_base, out_word_base);
}

// Shared bind body — caller holds mu_ and supplies the instance key (minted or the
// caller-supplied wire id). Allocates the pinned mirror (per-channel rings) + words.
std::uint64_t FrameCarrierEngine::bind_channels_locked(std::uint64_t id,
                                                       std::uint32_t n_channels,
                                                       const std::uint32_t* cell_bytes,
                                                       const std::uint32_t* cap1,
                                                       std::uint64_t* out_frame_base,
                                                       std::uint64_t* out_mirror_base,
                                                       std::uint64_t* out_word_base) {
    auto* fi = new FrameInstance();
    fi->program = 0;  // registry-owned cells; no provisional program layout
    fi->host_vis.reserve(n_channels);
    std::size_t mirror = 0;
    for (std::uint32_t c = 0; c < n_channels; ++c) {
        const std::uint32_t cb = cell_bytes[c] == 0 ? 1u : cell_bytes[c];
        const std::uint32_t k1 = cap1[c] == 0 ? 1u : cap1[c];
        fi->host_vis.push_back(FrameChannel{cb, k1, mirror});
        mirror += static_cast<std::size_t>(cb) * k1;
    }
    // pacing[0] + head/tail per channel (WordLayout::words(n) = 1 + 2n).
    const std::size_t word_bytes = (1u + 2u * n_channels) * sizeof(std::uint64_t);
    fi->frame_bytes = 0;  // vestigial — device cells live in the registry
    fi->mirror_bytes = mirror == 0 ? 1 : mirror;
    fi->word_bytes = word_bytes;
    fi->device_frame = nullptr;  // Q1 reconcile: no per-instance contiguous frame
    FC_CK(cudaHostAlloc(&fi->host_mirror, fi->mirror_bytes, cudaHostAllocDefault));
    std::memset(fi->host_mirror, 0, fi->mirror_bytes);
    // P1: words are MAPPED so a device kernel can publish head/tail/pacing into
    // them on the copy stream (P2). `words_dev` is the device-space alias.
    FC_CK(cudaHostAlloc(reinterpret_cast<void**>(&fi->host_words), word_bytes,
                        cudaHostAllocMapped));
    std::fill(reinterpret_cast<char*>(fi->host_words),
              reinterpret_cast<char*>(fi->host_words) + word_bytes, 0);
    FC_CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&fi->words_dev),
                                   fi->host_words, 0));

    instances_[id] = fi;

    if (out_frame_base) *out_frame_base = 0;  // vestigial
    if (out_mirror_base) *out_mirror_base = reinterpret_cast<std::uint64_t>(fi->host_mirror);
    if (out_word_base) *out_word_base = reinterpret_cast<std::uint64_t>(fi->host_words);
    return id;
}

void FrameCarrierEngine::publish(std::uint64_t instance, std::uint32_t n_channels,
                                 const void* const* src, const std::uint32_t* ring_index,
                                 const std::uint32_t* head, const std::uint32_t* tail,
                                 std::uint64_t pacing) {
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) {
        std::fprintf(stderr, "[frame_carrier] publish on unknown instance %llu\n",
                     static_cast<unsigned long long>(instance));
        std::abort();
    }
    // Copy each committed cell into its mirror ring slot + publish head/tail. The
    // value lands before pacing[0] (publish-before-wake): the runtime loads word[0]
    // with acquire, so the release fence below orders the mirror + head/tail writes
    // ahead of the pacing store.
    for (std::uint32_t c = 0; c < n_channels && c < fi->host_vis.size(); ++c) {
        const FrameChannel& fc = fi->host_vis[c];
        if (src[c] != nullptr && fc.cell_bytes > 0) {
            const std::size_t slot = static_cast<std::size_t>(ring_index[c]) * fc.cell_bytes;
            std::memcpy(static_cast<char*>(fi->host_mirror) + fc.mirror_off + slot,
                        src[c], fc.cell_bytes);
        }
        fi->host_words[1 + 2 * c] = head[c];  // WordLayout::head(c)
        fi->host_words[2 + 2 * c] = tail[c];  // WordLayout::tail(c)
    }
    std::atomic_thread_fence(std::memory_order_release);
    reinterpret_cast<std::atomic<std::uint64_t>*>(&fi->host_words[0])
        ->store(pacing, std::memory_order_relaxed);
}

void FrameCarrierEngine::publish_device(std::uint64_t instance, std::uint32_t n_channels,
                                        const void* const* src, const std::uint32_t* ring_index,
                                        const std::uint32_t* head, const std::uint32_t* tail,
                                        std::uint64_t pacing) {
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) {
        std::fprintf(stderr, "[frame_carrier] publish_device on unknown instance %llu\n",
                     static_cast<unsigned long long>(instance));
        std::abort();
    }
    // Value moves by DMA (C5): each committed DEVICE cell → its pinned mirror ring
    // slot on the copy stream — no host bounce buffer. The words (head/tail/pacing)
    // are then DEVICE-published into the MAPPED word region on the SAME stream (P2)
    // — stream-ordered AFTER the cell DMAs (publish-before-wake) — and one sync
    // drains the whole publish. The runtime loads word[0] with acquire; the
    // kernel's threadfence orders head/tail ahead of the pacing store.
    for (std::uint32_t c = 0; c < n_channels && c < fi->host_vis.size(); ++c) {
        const FrameChannel& fc = fi->host_vis[c];
        if (src[c] != nullptr && fc.cell_bytes > 0) {
            const std::size_t slot = static_cast<std::size_t>(ring_index[c]) * fc.cell_bytes;
            FC_CK(cudaMemcpyAsync(static_cast<char*>(fi->host_mirror) + fc.mirror_off + slot,
                                  src[c], fc.cell_bytes, cudaMemcpyDeviceToHost, stream_));
        }
    }
    launch_publish_words(fi->words_dev, n_channels, head, tail, pacing, stream_);
    FC_CK(cudaStreamSynchronize(stream_));
}

std::uint32_t FrameCarrierEngine::layout(std::uint64_t instance, std::uint32_t n_channels,
                                         std::uint32_t* out_cell_bytes,
                                         std::uint32_t* out_cap1,
                                         std::uint64_t* out_mirror_off,
                                         std::uint64_t* out_mirror_base,
                                         std::uint64_t* out_word_base) {
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) return 0;
    if (out_mirror_base != nullptr)
        *out_mirror_base = reinterpret_cast<std::uint64_t>(fi->host_mirror);
    if (out_word_base != nullptr)
        *out_word_base = reinterpret_cast<std::uint64_t>(fi->host_words);
    for (std::uint32_t c = 0; c < n_channels && c < fi->host_vis.size(); ++c) {
        const FrameChannel& fc = fi->host_vis[c];
        if (out_cell_bytes != nullptr) out_cell_bytes[c] = fc.cell_bytes;
        if (out_cap1 != nullptr) out_cap1[c] = fc.cap1;
        if (out_mirror_off != nullptr) out_mirror_off[c] = fc.mirror_off;
    }
    return static_cast<std::uint32_t>(fi->host_vis.size());
}

void FrameCarrierEngine::close_instance(std::uint64_t instance) {
    // B6/§5.2 grace — DRAIN before free (the free-before-drain UAF class): every
    // carry's D2H mirror + word publish + host_cb runs on `stream_`, so freeing this
    // instance's device frame / pinned mirror / pinned words while a carry is still
    // in flight lets that pending carry write into FREED memory (and a reader's
    // PinnedRingWord load reads freed pinned words). Sync the copy stream first. It
    // is engine-global + stable, so we sync OUTSIDE the registry lock (B1: never hold
    // a lock across a GPU wait). The runtime's InFlightTracker close-gate is the
    // higher-level B6 grace; this makes the carrier UAF-safe standalone too.
    FC_CK(cudaStreamSynchronize(stream_));
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) {
        std::fprintf(stderr, "[frame_carrier] close of unknown/closed instance %llu\n",
                     static_cast<unsigned long long>(instance));
        std::abort();  // fail-loud (never a silent trap) — the tensor_io house style
    }
    if (fi->device_frame) FC_CK(cudaFree(fi->device_frame));
    if (fi->host_mirror) FC_CK(cudaFreeHost(fi->host_mirror));
    if (fi->host_words) FC_CK(cudaFreeHost(fi->host_words));
    delete fi;
    instances_.erase(instance);
}

void FrameCarrierEngine::carry_in(std::uint64_t instance, const void* host_src,
                                  std::size_t n_bytes, std::size_t frame_offset) {
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) {
        std::fprintf(stderr, "[frame_carrier] carry_in on unknown instance %llu\n",
                     static_cast<unsigned long long>(instance));
        std::abort();
    }
    if (frame_offset + n_bytes > fi->frame_bytes) {
        std::fprintf(stderr,
                     "[frame_carrier] carry_in OOB: off=%zu n=%zu > frame=%zu\n",
                     frame_offset, n_bytes, fi->frame_bytes);
        std::abort();
    }
    FC_CK(cudaMemcpyAsync(static_cast<char*>(fi->device_frame) + frame_offset, host_src,
                          n_bytes, cudaMemcpyHostToDevice, stream_));
}

std::size_t FrameCarrierEngine::live_instances() const {
    std::lock_guard<std::mutex> guard(mu_);
    std::size_t n = 0;
    for (const auto& kv : instances_) {
        if (kv.second != nullptr) ++n;
    }
    return n;
}

}  // namespace pie_cuda_driver::sampling_ir

// ── extern "C" FFI surface ───────────────────────────────────────────────────

using pie_cuda_driver::sampling_ir::FrameCarrierEngine;

extern "C" {

std::uint64_t pie_frame_register(const std::uint8_t* trace, std::size_t trace_len) {
    return FrameCarrierEngine::instance().register_program(trace, trace_len);
}

std::uint64_t pie_frame_bind(std::uint64_t program, std::uint64_t* out_frame_base,
                             std::uint64_t* out_mirror_base, std::uint64_t* out_word_base) {
    return FrameCarrierEngine::instance().bind_instance(program, out_frame_base,
                                                        out_mirror_base, out_word_base);
}

std::uint64_t pie_frame_bind_channels(std::uint32_t n_channels,
                                      const std::uint32_t* cell_bytes,
                                      const std::uint32_t* cap1,
                                      std::uint64_t* out_frame_base,
                                      std::uint64_t* out_mirror_base,
                                      std::uint64_t* out_word_base) {
    return FrameCarrierEngine::instance().bind_channels(
        n_channels, cell_bytes, cap1, out_frame_base, out_mirror_base, out_word_base);
}

std::uint64_t pie_frame_bind_channels_keyed(std::uint64_t key, std::uint32_t n_channels,
                                            const std::uint32_t* cell_bytes,
                                            const std::uint32_t* cap1,
                                            std::uint64_t* out_frame_base,
                                            std::uint64_t* out_mirror_base,
                                            std::uint64_t* out_word_base) {
    return FrameCarrierEngine::instance().bind_channels_keyed(
        key, n_channels, cell_bytes, cap1, out_frame_base, out_mirror_base, out_word_base);
}

void pie_frame_publish(std::uint64_t instance, std::uint32_t n_channels,
                       const void* const* src, const std::uint32_t* ring_index,
                       const std::uint32_t* head, const std::uint32_t* tail,
                       std::uint64_t pacing) {
    FrameCarrierEngine::instance().publish(instance, n_channels, src, ring_index, head,
                                           tail, pacing);
}

void pie_frame_close(std::uint64_t instance) {
    FrameCarrierEngine::instance().close_instance(instance);
}

std::uint32_t pie_frame_layout(std::uint64_t instance, std::uint32_t n_channels,
                               std::uint32_t* out_cell_bytes, std::uint32_t* out_cap1,
                               std::uint64_t* out_mirror_off,
                               std::uint64_t* out_mirror_base,
                               std::uint64_t* out_word_base) {
    return FrameCarrierEngine::instance().layout(instance, n_channels, out_cell_bytes,
                                                 out_cap1, out_mirror_off, out_mirror_base,
                                                 out_word_base);
}

void pie_frame_write(std::uint64_t instance, const void* host_src, std::size_t n_bytes,
                     std::size_t frame_offset) {
    FrameCarrierEngine::instance().carry_in(instance, host_src, n_bytes, frame_offset);
}

}  // extern "C"

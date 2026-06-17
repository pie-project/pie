#pragma once

// WeightCopyEngine: the storage executor's host->device copy path, factored out
// so the executor body stays materialize/layout logic. It owns the copy streams,
// the pinned staging slots, the parallel reader-lane pool, and the pending-copy
// queue. Callers enqueue copies (a checkpoint file span -> a raw device dst) and
// flush(); the engine batches / pins / pipelines the H2D. Its only dependencies
// are the checkpoint source (for host bytes) and an optional LoadExecutionStats
// sink for counters — it does not touch the buffer map or the storage program.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "loader/safetensors.hpp"
#include "loader/tensor_spec.hpp"
#include "loader/loader_config.hpp"
#include "loader/loader_helpers.hpp"
#include "loader/phase_timer.hpp"

#if __has_include(<cuda_runtime.h>)
#define PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA 1
#include <cuda_runtime.h>
#include "cuda_check.hpp"
#include "loader/staged_h2d.hpp"
#else
#define PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA 0
#endif

namespace pie_cuda_driver {

class WeightCopyEngine {
public:
    explicit WeightCopyEngine(SafetensorsCheckpointSource& loader)
        : loader_(loader) {}

    ~WeightCopyEngine() { destroy_noexcept(); }

    WeightCopyEngine(const WeightCopyEngine&) = delete;
    WeightCopyEngine& operator=(const WeightCopyEngine&) = delete;

    // Counter sink for the current load (set to nullptr between loads).
    void set_stats(LoadExecutionStats* stats) noexcept { stats_ = stats; }

    // Queue one copy: checkpoint file span -> device dst. Batched/pinned and
    // pipelined at flush(); may flush internally when the pending queue is full.
    void queue(std::uint32_t shard_id, std::uint64_t file_offset,
               std::uint64_t span_bytes, void* dst)
    {
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
        if (copy_streams_enabled()) {
            ensure_copy_streams();
            if (pinned_staging_enabled()) {
                if (enqueue_pinned_staged_copy(shard_id, file_offset, span_bytes, dst)) {
                    return;
                }
            }
            cudaStream_t stream = copy_streams_[next_copy_stream_];
            next_copy_stream_ = (next_copy_stream_ + 1) % copy_streams_.size();
            if (batched_copies_enabled()) {
                enqueue_batched_copy(shard_id, file_offset, span_bytes, dst, stream);
            } else {
                loader_.copy_storage_bytes_to_device_async(
                    shard_id, file_offset, span_bytes, dst, stream);
            }
            ++pending_copy_count_;
            if (stats_ != nullptr) {
                stats_->max_pending_copies_seen =
                    std::max(stats_->max_pending_copies_seen, pending_copy_count_);
            }
            if (pending_copy_count_ >= max_pending_copies_) {
                flush();
            }
            return;
        }
#endif
        loader_.copy_storage_bytes_to_device(shard_id, file_offset, span_bytes, dst);
    }

#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
    // Queue a copy that must land on a specific stream so a follow-up kernel on
    // that stream sees the data without an explicit sync. Bypasses the
    // batched/pinned ring to keep ordering trivially correct.
    void queue_on_stream(std::uint32_t shard_id, std::uint64_t file_offset,
                         std::uint64_t span_bytes, void* dst, cudaStream_t stream)
    {
        loader_.copy_storage_bytes_to_device_async(
            shard_id, file_offset, span_bytes, dst, stream);
    }

    // A round-robin copy stream for a caller that runs its own async ops on it
    // (e.g. slab-scatter staging). Ensures the stream pool exists first.
    cudaStream_t acquire_stream()
    {
        ensure_copy_streams();
        cudaStream_t stream = copy_streams_[next_copy_stream_];
        next_copy_stream_ = (next_copy_stream_ + 1) % copy_streams_.size();
        return stream;
    }
#endif

    // Drain all queued copies and wait for their DMAs to complete.
    void flush()
    {
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
        if (pending_copy_count_ == 0) {
            return;
        }
        PhaseTimer _pt(stats_ != nullptr ? &stats_->phase_transfer_ms
                                         : &transfer_ms_sink_);
        flush_batched_copies();
        for (auto stream : copy_streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        release_inflight_pinned_slots();
        if (stats_ != nullptr) {
            ++stats_->copy_stream_flushes;
        }
        pending_copy_count_ = 0;
#endif
    }

private:
    bool copy_streams_enabled() const
    {
        return !loader_config::env_truthy("PIE_CUDA_DISABLE_PARALLEL_WEIGHT_COPIES");
    }

    bool pinned_staging_enabled() const
    {
        // Opt-in (default OFF). Measured no-op-to-negative on the real load path:
        // the bulk of bytes go through BulkExtentWrite -> the staged reader lanes,
        // which bypass this pinned ring, so it only covers the minority single-
        // ExtentWrite copies and its slot-busy flushes shrink pipelining depth.
        // Kept as a knob; see WEIGHT_LOADER_TODO.md A1.1 for the measurement.
        return loader_config::env_truthy("PIE_CUDA_ENABLE_PINNED_WEIGHT_STAGING");
    }

    std::uint64_t pinned_staging_min_bytes() const
    {
        return loader_config::env_u64("PIE_CUDA_PINNED_WEIGHT_MIN_BYTES",
                                      loader_config::kPinnedMinBytesDefault);
    }

    std::uint64_t pinned_staging_pool_bytes() const
    {
        const std::uint64_t mb = loader_config::env_u64("PIE_CUDA_PINNED_WEIGHT_POOL_MB", 0);
        return mb != 0 ? mb * loader_config::kMiB : loader_config::kPinnedPoolBytesDefault;
    }

    bool batched_copies_enabled() const
    {
#if CUDART_VERSION >= 12080
        return !loader_config::env_truthy("PIE_CUDA_DISABLE_BATCHED_WEIGHT_COPIES");
#else
        return false;
#endif
    }

#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
    // One queued host->device copy: device dst, host (mmap) src, size, stream.
    struct PendingCopy {
        void* dst = nullptr;
        void* src = nullptr;
        std::size_t size = 0;
        cudaStream_t stream = nullptr;
    };

    struct PinnedSlot {
        void* ptr = nullptr;
        std::uint64_t capacity = 0;
        cudaStream_t stream = nullptr;
        bool busy = false;
    };

    std::size_t pinned_staging_slot_count() const
    {
        std::size_t count = std::max<std::size_t>(copy_streams_.size(), 1);
        const std::uint64_t slots = loader_config::env_u64("PIE_CUDA_PINNED_WEIGHT_SLOTS", 0);
        if (slots != 0) {
            count = std::min<std::size_t>(slots, loader_config::kPinnedSlotsMax);
        }
        return count;
    }

    void ensure_copy_streams()
    {
        if (!copy_streams_.empty()) {
            return;
        }
        std::size_t count = loader_config::kCopyStreamsDefault;
        const std::uint64_t streams = loader_config::env_u64("PIE_CUDA_WEIGHT_COPY_STREAMS", 0);
        if (streams != 0) {
            count = std::min<std::size_t>(streams, loader_config::kCopyStreamsMax);
        }
        copy_streams_.resize(count);
        for (auto& stream : copy_streams_) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
    }

    void ensure_pinned_slots()
    {
        if (!pinned_slots_.empty()) {
            return;
        }
        pinned_slots_.resize(pinned_staging_slot_count());
    }

    void enqueue_batched_copy(std::uint32_t shard_id, std::uint64_t file_offset,
                              std::uint64_t span_bytes, void* dst, cudaStream_t stream)
    {
        pending_copies_.push_back(PendingCopy{
            dst,
            const_cast<std::uint8_t*>(
                loader_.storage_host_ptr(shard_id, file_offset, span_bytes)),
            static_cast<std::size_t>(span_bytes),
            stream,
        });
    }

    bool enqueue_pinned_staged_copy(std::uint32_t shard_id, std::uint64_t file_offset,
                                    std::uint64_t span_bytes, void* dst)
    {
        const std::uint64_t min_bytes = pinned_staging_min_bytes();
        if (span_bytes < min_bytes) {
            return false;
        }
        ensure_pinned_slots();
        const std::uint64_t pool_bytes = pinned_staging_pool_bytes();
        const std::uint64_t max_slot_bytes =
            pool_bytes / std::max<std::uint64_t>(pinned_slots_.size(), 1);
        if (span_bytes > max_slot_bytes) {
            return false;
        }

        PinnedSlot& slot = pinned_slots_[next_pinned_slot_];
        next_pinned_slot_ = (next_pinned_slot_ + 1) % pinned_slots_.size();
        if (slot.busy) {
            flush();
        }
        if (slot.capacity < span_bytes) {
            const std::uint64_t next_capacity = next_power_of_two(span_bytes);
            if (pinned_pool_capacity_bytes_ - slot.capacity + next_capacity > pool_bytes) {
                return false;
            }
            if (slot.ptr != nullptr) {
                CUDA_CHECK(cudaFreeHost(slot.ptr));
                pinned_pool_capacity_bytes_ -= slot.capacity;
                slot.ptr = nullptr;
                slot.capacity = 0;
            }
            CUDA_CHECK(cudaMallocHost(&slot.ptr, static_cast<std::size_t>(next_capacity)));
            slot.capacity = next_capacity;
            pinned_pool_capacity_bytes_ += next_capacity;
        }

        cudaStream_t stream = copy_streams_[next_copy_stream_];
        next_copy_stream_ = (next_copy_stream_ + 1) % copy_streams_.size();
        loader_.read_storage_bytes_to_host(shard_id, file_offset, span_bytes, slot.ptr);
        CUDA_CHECK(cudaMemcpyAsync(
            dst, slot.ptr, span_bytes, cudaMemcpyHostToDevice, stream));
        slot.stream = stream;
        slot.busy = true;
        ++pending_copy_count_;
        if (stats_ != nullptr) {
            ++stats_->h2d_pinned_copy_count;
            stats_->h2d_pinned_copy_bytes += span_bytes;
            stats_->max_pending_copies_seen =
                std::max(stats_->max_pending_copies_seen, pending_copy_count_);
        }
        if (pending_copy_count_ >= max_pending_copies_) {
            flush();
        }
        return true;
    }

    // Stage all pending copies (mmap host src -> device) through the shared
    // pinned-pipelined engine, round-robin across reader lanes.
    void parallel_staged_flush()
    {
        const std::size_t lanes = std::max<std::size_t>(loader_config::reader_lane_count(), 1);
        if (reader_pool_ == nullptr || reader_pool_->lanes() < lanes) {
            reader_pool_ = std::make_unique<PinnedLanePool>(
                lanes, loader_config::reader_buf_bytes());
        }
        {
            // One-time pinned/stream allocation; timed separately so the profiler
            // can distinguish staging-buffer setup from the actual transfer.
            PhaseTimer _pt(stats_ != nullptr ? &stats_->phase_pinned_alloc_ms
                                             : &transfer_ms_sink_);
            reader_pool_->prepare();
        }

        std::vector<StagedCopy> staged;
        staged.reserve(pending_copies_.size());
        std::uint64_t staged_bytes = 0;
        for (const auto& c : pending_copies_) {
            staged.push_back(StagedCopy{c.dst, c.src, c.size});
            staged_bytes += c.size;
        }
        staged_pinned_h2d(*reader_pool_, staged);

        if (stats_ != nullptr) {
            stats_->h2d_pinned_copy_count += pending_copies_.size();
            stats_->h2d_pinned_copy_bytes += staged_bytes;
            ++stats_->copy_stream_flushes;
        }
    }

    void flush_batched_copies()
    {
        if (pending_copies_.empty()) {
            return;
        }
        if (loader_config::reader_lane_count() > 0) {
            parallel_staged_flush();
            pending_copies_.clear();
            return;
        }
#if CUDART_VERSION >= 12080
        // Single cudaMemcpyAttributes applied to every copy in the batch. For
        // host->device, srcAccessOrder=Any lets the runtime reorder reads from
        // pinned host pages for max throughput.
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
        attr.flags = cudaMemcpyFlagDefault;
        std::size_t attrs_idx = 0;
        for (auto stream : copy_streams_) {
            batched_dsts_.clear();
            batched_srcs_.clear();
            batched_sizes_.clear();
            for (const auto& copy : pending_copies_) {
                if (copy.stream != stream) {
                    continue;
                }
                batched_dsts_.push_back(copy.dst);
                batched_srcs_.push_back(copy.src);
                batched_sizes_.push_back(copy.size);
            }
            if (batched_dsts_.empty()) {
                continue;
            }
            // The CUDA 12.8 batched H2D path takes one API call for the whole
            // batch — far cheaper than N cudaMemcpyAsync launches. Chunk at 1024
            // copies/call to stay under internal sizing limits.
            constexpr std::size_t kChunk = loader_config::kBatchChunk;
            const std::size_t total = batched_dsts_.size();
            for (std::size_t off = 0; off < total; off += kChunk) {
                const std::size_t n = std::min(kChunk, total - off);
                // cudaMemcpyBatchAsync's signature changed between CUDA 12.8
                // (preview: non-const ptrs + a trailing `size_t* failIdx`
                // out-param, 9 args) and CUDA 13.0 (final: const-qualified
                // ptrs, no failIdx, 8 args). Pick the call shape per toolkit.
#if CUDART_VERSION >= 13000
                const cudaError_t err = ::cudaMemcpyBatchAsync(
                    batched_dsts_.data() + off,
                    const_cast<const void**>(batched_srcs_.data() + off),
                    batched_sizes_.data() + off,
                    n, &attr, &attrs_idx, /*numAttrs=*/1, stream);
#else
                std::size_t fail_idx = 0;
                const cudaError_t err = ::cudaMemcpyBatchAsync(
                    batched_dsts_.data() + off,
                    batched_srcs_.data() + off,
                    batched_sizes_.data() + off,
                    n, &attr, &attrs_idx, /*numAttrs=*/1, &fail_idx, stream);
#endif
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaMemcpyBatchAsync failed: ") +
                        cudaGetErrorString(err));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (stats_ != nullptr) {
                ++stats_->h2d_batch_calls;
            }
        }
        pending_copies_.clear();
#else
        pending_copies_.clear();
#endif
    }

    void release_inflight_pinned_slots() noexcept
    {
        for (auto& slot : pinned_slots_) {
            if (slot.busy) {
                slot.busy = false;
                slot.stream = nullptr;
            }
        }
    }

    void free_pinned_slots_noexcept() noexcept
    {
        for (auto& slot : pinned_slots_) {
            if (slot.ptr != nullptr) {
                (void)cudaFreeHost(slot.ptr);
                slot.ptr = nullptr;
            }
            slot.capacity = 0;
            slot.busy = false;
            slot.stream = nullptr;
        }
        pinned_slots_.clear();
        pinned_pool_capacity_bytes_ = 0;
        next_pinned_slot_ = 0;
    }
#endif  // PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA

    void destroy_noexcept() noexcept
    {
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
        if (copy_streams_.empty()) {
            free_pinned_slots_noexcept();
            reader_pool_.reset();
            return;
        }
        for (auto stream : copy_streams_) {
            if (stream != nullptr) {
                if (pending_copy_count_ != 0) {
                    (void)cudaStreamSynchronize(stream);
                }
                (void)cudaStreamDestroy(stream);
            }
        }
        free_pinned_slots_noexcept();
        reader_pool_.reset();
        copy_streams_.clear();
        pending_copy_count_ = 0;
        next_copy_stream_ = 0;
#endif
    }

    SafetensorsCheckpointSource& loader_;
    LoadExecutionStats* stats_ = nullptr;
    double transfer_ms_sink_ = 0.0;
    std::size_t pending_copy_count_ = 0;
    std::size_t max_pending_copies_ = loader_config::kMaxPendingCopies;
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
    std::vector<cudaStream_t> copy_streams_;
    std::size_t next_copy_stream_ = 0;
    std::vector<PendingCopy> pending_copies_;
    std::vector<PinnedSlot> pinned_slots_;
    std::size_t next_pinned_slot_ = 0;
    std::uint64_t pinned_pool_capacity_bytes_ = 0;
    std::unique_ptr<PinnedLanePool> reader_pool_;
    std::vector<void*> batched_dsts_;
    std::vector<void*> batched_srcs_;
    std::vector<std::size_t> batched_sizes_;
#endif
};

}  // namespace pie_cuda_driver

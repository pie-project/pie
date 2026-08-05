#pragma once

// WeightCopyEngine: the storage executor's host->device copy path, factored out
// so the executor body stays materialize/layout logic. It owns the copy streams,
// the pinned staging slots, the parallel reader-lane pool, and the pending-copy
// queue. Callers enqueue copies (a checkpoint file span -> a raw device dst) and
// flush(); the engine batches / pins / pipelines the H2D. Its only dependencies
// are the checkpoint source (for host bytes) and an optional LoadExecutionStats
// sink for counters — it does not touch the buffer map or the LoadPlan.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pie_loader/checkpoint_source.hpp"
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
    explicit WeightCopyEngine(pie_loader::CheckpointSource& loader)
        : loader_(loader) {}

    ~WeightCopyEngine() { destroy_noexcept(); }

    WeightCopyEngine(const WeightCopyEngine&) = delete;
    WeightCopyEngine& operator=(const WeightCopyEngine&) = delete;

    /// Tune for many small transfers rather than one large load.
    ///
    /// Everything this engine does by default trades setup for throughput --
    /// a pool of copy streams, host reader lanes staging through pinned
    /// buffers -- and every one of those is amortised over a whole model. A
    /// page-in moves a few hundred kilobytes, inside a forward pass that is
    /// blocked on it, so there is nothing to overlap and the setup is the
    /// cost: a lane per stream to synchronise at each flush, and a thread
    /// pool dispatch and join to copy from pages that are already mapped.
    ///
    /// One stream, straight from the mapping. Must be set before the first
    /// copy, which is what creates the pool.
    void prefer_small_transfers() noexcept
    {
        stream_limit_ = 1;
        reader_lanes_ = false;
    }

    // Counter sink for the current load (set to nullptr between loads).
    void set_stats(LoadExecutionStats* stats) noexcept { stats_ = stats; }

#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
    // One checkpoint span, straight to the device, on the caller's stream.
    //
    // The mapping and the copy used to sit together inside the checkpoint
    // source.
    // They are separated because they answer to different owners: the span is
    // valid because the plan named it, the copy is correct because this engine
    // owns the stream it runs on. Keeping them apart also leaves the reader
    // CUDA-free, which is the shape the Metal reader already had.
    void copy_span_to_device(std::uint32_t shard_id, std::uint64_t file_offset,
                             std::uint64_t span_bytes, void* dst,
                             cudaStream_t stream)
    {
        if (dst == nullptr && span_bytes != 0) {
            throw std::runtime_error("checkpoint destination is null");
        }
        const std::uint8_t* src =
            loader_.storage_host_ptr(shard_id, file_offset, span_bytes);
        CUDA_CHECK(cudaMemcpyAsync(
            dst, src, span_bytes, cudaMemcpyHostToDevice, stream));
    }
#endif

    // Queue one copy: checkpoint file span -> device dst. Batched/pinned and
    // pipelined at flush(); may flush internally when the pending queue is full.
    void queue(std::uint32_t shard_id, std::uint64_t file_offset,
               std::uint64_t span_bytes, void* dst)
    {
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
        if (copy_streams_enabled()) {
            ensure_copy_streams();
            cudaStream_t stream = next_stream();
            if (batched_copies_enabled()) {
                enqueue_batched_copy(shard_id, file_offset, span_bytes, dst, stream);
            } else {
                copy_span_to_device(
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
        copy_span_blocking(shard_id, file_offset, span_bytes, dst);
    }

    // The no-copy-stream path: the default stream, synchronised before returning.
    void copy_span_blocking(std::uint32_t shard_id, std::uint64_t file_offset,
                            std::uint64_t span_bytes, void* dst)
    {
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
        if (dst == nullptr && span_bytes != 0) {
            throw std::runtime_error("checkpoint destination is null");
        }
        const std::uint8_t* src =
            loader_.storage_host_ptr(shard_id, file_offset, span_bytes);
        CUDA_CHECK(cudaMemcpy(dst, src, span_bytes, cudaMemcpyHostToDevice));
#else
        (void)shard_id; (void)file_offset; (void)span_bytes; (void)dst;
        throw std::runtime_error("weight copy engine: built without CUDA");
#endif
    }

#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
    // Queue a copy that must land on a specific stream so a follow-up kernel on
    // that stream sees the data without an explicit sync. Bypasses the
    // batched/pinned ring to keep ordering trivially correct.
    void queue_on_stream(std::uint32_t shard_id, std::uint64_t file_offset,
                         std::uint64_t span_bytes, void* dst, cudaStream_t stream)
    {
        copy_span_to_device(shard_id, file_offset, span_bytes, dst, stream);
    }

    // A round-robin copy stream for a caller that runs its own async ops on it
    // (e.g. slab-scatter staging). Ensures the stream pool exists first.
    cudaStream_t acquire_stream()
    {
        ensure_copy_streams();
        return next_stream();
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
        // Only the streams this flush's copies actually landed on. Syncing an
        // idle stream is not free -- it is a driver round trip of the same
        // order as a small copy -- and a plan flushes several times, so a
        // whole-pool sweep costs stream-count times flush-count round trips
        // whatever the plan moved. Immaterial when a plan moves a model;
        // most of a page-in when it moves one expert.
        for (std::size_t i = 0; i < copy_streams_.size(); ++i) {
            if (!stream_used_[i]) continue;
            CUDA_CHECK(cudaStreamSynchronize(copy_streams_[i]));
            stream_used_[i] = false;
        }
        if (stats_ != nullptr) {
            ++stats_->copy_stream_flushes;
        }
        pending_copy_count_ = 0;
#endif
    }

private:
    bool copy_streams_enabled() const
    {
        return true;
    }

    bool batched_copies_enabled() const
    {
#if CUDART_VERSION >= 12080
        return true;
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

    void ensure_copy_streams()
    {
        if (!copy_streams_.empty()) {
            return;
        }
        std::size_t count = loader_config::kCopyStreamsDefault;
        if (stream_limit_ != 0) {
            count = std::min(count, stream_limit_);
        }
        copy_streams_.resize(count);
        stream_used_.assign(count, false);
        for (auto& stream : copy_streams_) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
    }

    /// The next stream in the rotation, marked as owing a sync at flush.
    cudaStream_t next_stream()
    {
        const std::size_t i = next_copy_stream_;
        next_copy_stream_ = (next_copy_stream_ + 1) % copy_streams_.size();
        stream_used_[i] = true;
        return copy_streams_[i];
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
        if (reader_lanes_ && loader_config::reader_lane_count() > 0) {
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


#endif  // PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA

    void destroy_noexcept() noexcept
    {
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
        if (copy_streams_.empty()) {
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
        reader_pool_.reset();
        copy_streams_.clear();
        pending_copy_count_ = 0;
        next_copy_stream_ = 0;
#endif
    }

    pie_loader::CheckpointSource& loader_;
    LoadExecutionStats* stats_ = nullptr;
    double transfer_ms_sink_ = 0.0;
    std::size_t pending_copy_count_ = 0;
    std::size_t max_pending_copies_ = loader_config::kMaxPendingCopies;
#if PIE_CUDA_WEIGHT_COPY_ENGINE_HAS_CUDA
    std::vector<cudaStream_t> copy_streams_;
    std::size_t next_copy_stream_ = 0;
    /// Which streams have had work queued since the last flush.
    std::vector<bool> stream_used_;
    /// 0 means the default pool.
    std::size_t stream_limit_ = 0;
    bool reader_lanes_ = true;
    std::vector<PendingCopy> pending_copies_;
    std::unique_ptr<PinnedLanePool> reader_pool_;
    std::vector<void*> batched_dsts_;
    std::vector<void*> batched_srcs_;
    std::vector<std::size_t> batched_sizes_;
#endif
};

}  // namespace pie_cuda_driver

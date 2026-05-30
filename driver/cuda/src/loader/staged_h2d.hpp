#pragma once

// Shared pinned-pipelined host->device copy engine.
//
// Both the storage executor's bulk weight flush and the artifact-cache restore
// move bytes through this one path so they behave identically: a set of
// host->device copies is partitioned round-robin across `lanes` worker threads,
// and each lane double-buffers a small pinned staging buffer so the next host
// memcpy overlaps the DMA that is already in flight.
//
// The per-lane loop uses raw CUDA calls (no throwing across the worker-thread
// boundary, and no device pre-warm assumptions); any hard failure surfaces at
// the caller's next device synchronize. Lane setup (streams/events/pinned
// buffers) runs on the calling thread and does throw via CUDA_CHECK.

#include <cstdint>

#if __has_include(<cuda_runtime.h>)

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include "cuda_check.hpp"

namespace pie_cuda_driver {

// One host->device copy: device dst, host src (mmap or any host memory), bytes.
struct StagedCopy {
    void* dst = nullptr;
    const void* src = nullptr;
    std::uint64_t size = 0;
};

// Caller-owned pool of pinned double-buffer lanes (each with its own stream +
// completion events), reused across staged_pinned_h2d() calls. The executor
// holds one for an entire load; restore makes a local one. Buffers are
// allocated lazily (prepare(), idempotent) and freed in the destructor.
class PinnedLanePool {
public:
    PinnedLanePool(std::size_t lanes, std::uint64_t buf_bytes)
        : buf_bytes_(buf_bytes)
    {
        lanes_.resize(std::max<std::size_t>(lanes, 1));
    }

    ~PinnedLanePool()
    {
        for (auto& l : lanes_) {
            for (int b = 0; b < 2; ++b) {
                if (l.pinned[b] != nullptr) {
                    cudaFreeHost(l.pinned[b]);
                }
                if (l.done[b] != nullptr) {
                    cudaEventDestroy(l.done[b]);
                }
            }
            if (l.stream != nullptr) {
                cudaStreamDestroy(l.stream);
            }
        }
    }

    PinnedLanePool(const PinnedLanePool&) = delete;
    PinnedLanePool& operator=(const PinnedLanePool&) = delete;

    std::size_t lanes() const noexcept { return lanes_.size(); }

    // Allocate the pinned buffers / streams / events. Idempotent; safe to call
    // on its own (e.g. wrapped in a profiling timer) before staged_pinned_h2d().
    void prepare()
    {
        if (device_ < 0) {
            cudaGetDevice(&device_);
        }
        for (auto& l : lanes_) {
            if (l.stream == nullptr) {
                CUDA_CHECK(cudaStreamCreate(&l.stream));
            }
            for (int b = 0; b < 2; ++b) {
                if (l.pinned[b] == nullptr) {
                    CUDA_CHECK(cudaMallocHost(
                        &l.pinned[b], static_cast<std::size_t>(buf_bytes_)));
                }
                if (l.done[b] == nullptr) {
                    CUDA_CHECK(cudaEventCreateWithFlags(
                        &l.done[b], cudaEventDisableTiming));
                }
            }
        }
    }

private:
    friend void staged_pinned_h2d(PinnedLanePool&, const std::vector<StagedCopy>&);

    struct Lane {
        void* pinned[2] = {nullptr, nullptr};
        cudaEvent_t done[2] = {nullptr, nullptr};
        cudaStream_t stream = nullptr;
    };

    std::vector<Lane> lanes_;
    std::uint64_t buf_bytes_ = 0;
    int device_ = -1;
};

// Copy every [src, src+size) -> dst across the pool's lanes, blocking until all
// DMAs land. Copies are sliced into <= buf_bytes sub-chunks and split into one
// contiguous run per lane, so a single large buffer (e.g. a dense model's whole
// arena) still saturates every lane rather than stalling one. Each lane streams
// its run through its pinned double-buffer, reusing a buffer only once its prior
// H2D has landed (event-gated). Contiguous (not interleaved) lane runs keep each
// lane's host reads sequential for the kernel's read-ahead.
inline void staged_pinned_h2d(PinnedLanePool& pool,
                              const std::vector<StagedCopy>& copies)
{
    if (copies.empty()) {
        return;
    }
    pool.prepare();
    const std::size_t lanes = pool.lanes();
    const std::uint64_t cap = pool.buf_bytes_;
    const int device = pool.device_;

    // Flatten into <= cap sub-chunks so the work can be balanced by bytes.
    std::vector<StagedCopy> chunks;
    for (const auto& c : copies) {
        for (std::uint64_t off = 0; off < c.size; off += cap) {
            const std::uint64_t n = std::min<std::uint64_t>(cap, c.size - off);
            chunks.push_back(StagedCopy{
                static_cast<std::uint8_t*>(c.dst) + off,
                static_cast<const std::uint8_t*>(c.src) + off, n});
        }
    }
    const std::size_t per_lane = (chunks.size() + lanes - 1) / lanes;

    auto run_lane = [&pool, &chunks, per_lane, device](std::size_t li) {
        if (device >= 0) {
            cudaSetDevice(device);
        }
        PinnedLanePool::Lane& lane = pool.lanes_[li];
        const std::size_t begin = li * per_lane;
        const std::size_t end = std::min(begin + per_lane, chunks.size());
        int buf = 0;
        for (std::size_t i = begin; i < end; ++i) {
            cudaEventSynchronize(lane.done[buf]);
            std::memcpy(lane.pinned[buf], chunks[i].src,
                        static_cast<std::size_t>(chunks[i].size));
            cudaMemcpyAsync(chunks[i].dst, lane.pinned[buf],
                            static_cast<std::size_t>(chunks[i].size),
                            cudaMemcpyHostToDevice, lane.stream);
            cudaEventRecord(lane.done[buf], lane.stream);
            buf ^= 1;
        }
        cudaStreamSynchronize(lane.stream);
    };

    std::vector<std::thread> workers;
    workers.reserve(lanes);
    for (std::size_t li = 0; li < lanes; ++li) {
        workers.emplace_back(run_lane, li);
    }
    for (auto& w : workers) {
        w.join();
    }
}

}  // namespace pie_cuda_driver

#endif  // __has_include(<cuda_runtime.h>)

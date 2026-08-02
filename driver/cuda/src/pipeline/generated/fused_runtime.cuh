#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "batch/fire_timing.hpp"
#include "kernels/envelope_device.cuh"
#include "cuda_check.hpp"
#include "pipeline/region_support.hpp"
#include "pipeline/generated/module_cache.hpp"
#include "pipeline/grouped_runtime.cuh"
#include "runahead.hpp"

namespace pie_cuda_driver::pipeline::generated {
namespace detail {

struct AsyncAllocations {
    cudaStream_t stream = nullptr;
    std::vector<void*> values;

    explicit AsyncAllocations(cudaStream_t value) : stream(value) {}

    ~AsyncAllocations() {
        for (void* value : values) {
            if (value != nullptr) cudaFreeAsync(value, stream);
        }
    }

    template <class T>
    T* allocate(std::size_t count = 1) {
        T* result = nullptr;
        CUDA_CHECK(cudaMallocAsync(
            reinterpret_cast<void**>(&result),
            std::max<std::size_t>(count * sizeof(T), 1),
            stream));
        values.push_back(result);
        return result;
    }
};

struct PinnedUploadLease {
    std::uint8_t* data = nullptr;
    std::size_t slot = 0;
};

class PinnedUploadStaging {
  public:
    PinnedUploadStaging() = default;
    PinnedUploadStaging(const PinnedUploadStaging&) = delete;
    PinnedUploadStaging& operator=(const PinnedUploadStaging&) = delete;

    ~PinnedUploadStaging() {
        for (std::size_t slot = 0; slot < data_.size(); ++slot) {
            if (pending_[slot] && done_[slot] != nullptr) {
                cudaEventSynchronize(done_[slot]);
            }
            if (done_[slot] != nullptr) cudaEventDestroy(done_[slot]);
            if (data_[slot] != nullptr) cudaFreeHost(data_[slot]);
        }
        // Superseded blocks parked by hot-path growth. Their uploads were
        // enqueued before the slot events synchronized above on the same
        // stream, so they have retired by now.
        for (std::uint8_t* block : retired_) {
            if (block != nullptr) cudaFreeHost(block);
        }
    }

    PinnedUploadLease acquire(std::size_t bytes) {
        const std::size_t slot = next_slot_;
        next_slot_ = (next_slot_ + 1) % data_.size();
        if (capacity_[slot] < bytes) {
            // Growing NEVER frees on the hot path: cudaFreeHost
            // synchronizes the entire device, so a capacity ratchet on a
            // prefill-mixed wave stalled the submitting thread for a full
            // GPU wave (measured 3–5 ms per regrow, ~10 waves per cohort
            // turnover on the c64 workload — V6 iteration 1). The
            // superseded block is parked until destruction — an in-flight
            // upload keeps reading valid memory, so no event sync is
            // needed either — and capacity grows geometrically so each
            // slot ratchets O(log) times per process.
            if (data_[slot] != nullptr) {
                retired_.push_back(data_[slot]);
                data_[slot] = nullptr;
            }
            const std::size_t grown = std::max(
                {bytes, capacity_[slot] * 2, std::size_t{4096}});
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&data_[slot]), grown));
            capacity_[slot] = grown;
        } else if (pending_[slot]) {
            // Reusing the buffer in place: the previous upload from this
            // slot must have left the pinned block before we overwrite it.
            CUDA_CHECK(cudaEventSynchronize(done_[slot]));
            pending_[slot] = false;
        }
        if (done_[slot] == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &done_[slot], cudaEventDisableTiming));
        }
        return {data_[slot], slot};
    }

    void record(std::size_t slot, cudaStream_t stream) {
        const cudaError_t status = cudaEventRecord(done_[slot], stream);
        if (status != cudaSuccess) {
            cudaStreamSynchronize(stream);
            CUDA_CHECK(status);
        }
        pending_[slot] = true;
    }

  private:
    // Single-sourced from runahead.hpp: every in-flight wave's
    // grouped-stage metadata upload holds one slot until the GPU passes
    // its copy, so the pool must run deeper than the scheduler's
    // run-ahead (the measured stall lives with the constant).
    static constexpr std::size_t kDepth = kUploadStagingDepth;
    std::array<std::uint8_t*, kDepth> data_{};
    std::array<std::size_t, kDepth> capacity_{};
    std::array<cudaEvent_t, kDepth> done_{};
    std::array<bool, kDepth> pending_{};
    // Blocks superseded by growth, parked until destruction (bounded:
    // geometric growth retires O(log) blocks per slot, and a retired
    // block is never larger than the slot's final capacity).
    std::vector<std::uint8_t*> retired_;
    std::size_t next_slot_ = 0;
};

class PersistentDeviceWorkspace;

class PersistentDeviceLease {
  public:
    PersistentDeviceLease() = default;
    PersistentDeviceLease(
        PersistentDeviceWorkspace* owner,
        std::size_t slot,
        std::uint8_t* metadata,
        std::uint8_t* scratch,
        cudaStream_t stream)
        : owner_(owner),
          slot_(slot),
          metadata_(metadata),
          scratch_(scratch),
          stream_(stream) {}
    ~PersistentDeviceLease();
    PersistentDeviceLease(const PersistentDeviceLease&) = delete;
    PersistentDeviceLease& operator=(const PersistentDeviceLease&) = delete;
    PersistentDeviceLease(PersistentDeviceLease&& other) noexcept;
    PersistentDeviceLease& operator=(PersistentDeviceLease&& other) noexcept;

    std::uint8_t* metadata() const noexcept { return metadata_; }
    std::uint8_t* scratch() const noexcept { return scratch_; }
    void record();

  private:
    void reset() noexcept;

    PersistentDeviceWorkspace* owner_ = nullptr;
    std::size_t slot_ = 0;
    std::uint8_t* metadata_ = nullptr;
    std::uint8_t* scratch_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

class PersistentDeviceWorkspace {
  public:
    PersistentDeviceWorkspace() = default;
    PersistentDeviceWorkspace(const PersistentDeviceWorkspace&) = delete;
    PersistentDeviceWorkspace& operator=(const PersistentDeviceWorkspace&) = delete;

    ~PersistentDeviceWorkspace() {
        for (Slot& slot : slots_) {
            if (slot.quarantined || slot.leased) continue;
            if (slot.pending && slot.done != nullptr) {
                const cudaError_t status =
                    cudaEventSynchronize(slot.done);
                if (status != cudaSuccess) {
                    std::fprintf(
                        stderr,
                        "[pie-driver-cuda] generated device slot teardown fence failed: %s\n",
                        cudaGetErrorString(status));
                    continue;
                }
            }
            if (slot.done != nullptr) cudaEventDestroy(slot.done);
            if (slot.metadata != nullptr) cudaFree(slot.metadata);
            if (slot.scratch != nullptr) cudaFree(slot.scratch);
            for (const Allocation& allocation : slot.retired) {
                if (allocation.pointer != nullptr) {
                    cudaFree(allocation.pointer);
                }
            }
        }
    }

    std::size_t retained_bytes() const noexcept {
        std::size_t total = 0;
        for (const Slot& slot : slots_) {
            total += slot.metadata_capacity + slot.scratch_capacity;
            for (const Allocation& allocation : slot.retired) {
                total += allocation.bytes;
            }
        }
        return total;
    }

    std::optional<std::pair<PersistentDeviceLease, std::size_t>>
    try_acquire(
        std::size_t metadata_bytes,
        std::size_t scratch_bytes,
        cudaStream_t stream,
        std::size_t available_budget) {
        std::size_t slot_index = slots_.size();
        for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
            const std::size_t candidate =
                (next_slot_ + offset) % slots_.size();
            Slot& slot = slots_[candidate];
            if (slot.leased || slot.quarantined) continue;
            if (slot.pending) {
                const cudaError_t status = cudaEventQuery(slot.done);
                if (status == cudaErrorNotReady) continue;
                if (status != cudaSuccess) {
                    slot.quarantined = true;
                    slot.pending = false;
                    std::fprintf(
                        stderr,
                        "[pie-driver-cuda] generated device slot query failed: %s\n",
                        cudaGetErrorString(status));
                    continue;
                }
                slot.pending = false;
            }
            slot_index = candidate;
            break;
        }
        if (slot_index == slots_.size()) return std::nullopt;
        Slot& slot = slots_[slot_index];
        const std::size_t additional =
            (metadata_bytes > slot.metadata_capacity ? metadata_bytes : 0) +
            (scratch_bytes > slot.scratch_capacity ? scratch_bytes : 0);
        if (additional > available_budget) return std::nullopt;
        next_slot_ = (slot_index + 1) % slots_.size();
        if (slot.done == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &slot.done, cudaEventDisableTiming));
        }

        const bool grow_metadata =
            metadata_bytes > slot.metadata_capacity;
        const bool grow_scratch =
            scratch_bytes > slot.scratch_capacity;
        slot.retired.reserve(
            slot.retired.size() +
            (grow_metadata && slot.metadata != nullptr ? 1 : 0) +
            (grow_scratch && slot.scratch != nullptr ? 1 : 0));
        std::uint8_t* next_metadata = nullptr;
        std::uint8_t* next_scratch = nullptr;
        try {
            if (grow_metadata) {
                CUDA_CHECK(cudaMallocAsync(
                    reinterpret_cast<void**>(&next_metadata),
                    std::max<std::size_t>(metadata_bytes, 1),
                    stream));
            }
            if (grow_scratch) {
                CUDA_CHECK(cudaMallocAsync(
                    reinterpret_cast<void**>(&next_scratch),
                    std::max<std::size_t>(scratch_bytes, 1),
                    stream));
            }
        } catch (...) {
            if (next_metadata != nullptr) {
                cudaFreeAsync(next_metadata, stream);
            }
            if (next_scratch != nullptr) {
                cudaFreeAsync(next_scratch, stream);
            }
            throw;
        }
        if (next_metadata != nullptr) {
            if (slot.metadata != nullptr) {
                slot.retired.push_back(
                    {slot.metadata, slot.metadata_capacity});
            }
            slot.metadata = next_metadata;
            slot.metadata_capacity = metadata_bytes;
        }
        if (next_scratch != nullptr) {
            if (slot.scratch != nullptr) {
                slot.retired.push_back(
                    {slot.scratch, slot.scratch_capacity});
            }
            slot.scratch = next_scratch;
            slot.scratch_capacity = scratch_bytes;
        }
        slot.leased = true;
        return std::make_optional(std::make_pair(
            PersistentDeviceLease(
                this,
                slot_index,
                slot.metadata,
                slot.scratch,
                stream),
            additional));
    }

  private:
    friend class PersistentDeviceLease;

    struct Allocation {
        std::uint8_t* pointer = nullptr;
        std::size_t bytes = 0;
    };

    struct Slot {
        std::uint8_t* metadata = nullptr;
        std::uint8_t* scratch = nullptr;
        std::size_t metadata_capacity = 0;
        std::size_t scratch_capacity = 0;
        cudaEvent_t done = nullptr;
        bool pending = false;
        bool leased = false;
        bool quarantined = false;
        std::vector<Allocation> retired;
    };

    void record(std::size_t index, cudaStream_t stream) {
        Slot& slot = slots_[index];
        const cudaError_t status = cudaEventRecord(slot.done, stream);
        if (status != cudaSuccess) {
            const cudaError_t sync_status = cudaStreamSynchronize(stream);
            slot.leased = false;
            slot.pending = false;
            if (sync_status != cudaSuccess) slot.quarantined = true;
            CUDA_CHECK(status);
        }
        slot.pending = true;
        slot.leased = false;
    }

    void cancel(std::size_t index, cudaStream_t stream) noexcept {
        Slot& slot = slots_[index];
        if (slot.quarantined) return;
        const cudaError_t status = cudaStreamSynchronize(stream);
        if (status == cudaSuccess) {
            slot.leased = false;
        } else {
            slot.quarantined = true;
            slot.leased = false;
            std::fprintf(
                stderr,
                "[pie-driver-cuda] generated device slot cleanup failed: %s\n",
                cudaGetErrorString(status));
        }
    }

    static constexpr std::size_t kDepth = 2;
    std::array<Slot, kDepth> slots_{};
    std::size_t next_slot_ = 0;
};

inline PersistentDeviceLease::~PersistentDeviceLease() {
    reset();
}

inline PersistentDeviceLease::PersistentDeviceLease(
    PersistentDeviceLease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      slot_(other.slot_),
      metadata_(other.metadata_),
      scratch_(other.scratch_),
      stream_(other.stream_) {}

inline PersistentDeviceLease& PersistentDeviceLease::operator=(
    PersistentDeviceLease&& other) noexcept {
    if (this != &other) {
        reset();
        owner_ = std::exchange(other.owner_, nullptr);
        slot_ = other.slot_;
        metadata_ = other.metadata_;
        scratch_ = other.scratch_;
        stream_ = other.stream_;
    }
    return *this;
}

inline void PersistentDeviceLease::record() {
    if (owner_ == nullptr) return;
    owner_->record(slot_, stream_);
    owner_ = nullptr;
}

inline void PersistentDeviceLease::reset() noexcept {
    if (owner_ != nullptr) {
        owner_->cancel(slot_, stream_);
        owner_ = nullptr;
    }
}

// Stable per-stage staging for the prepared execution path
// (`prepare_generated_stage` / `launch_generated_stage`): ONE pinned host
// block and ONE device metadata+scratch pair per (stage runtime id, stream),
// grow-only and generation-counted. This is the same shape as
// `model::HookSidebandArena` (driver/cuda/src/model/hook_sideband_arena.hpp),
// the sibling precedent for graph-capture-ready storage: while capacity
// suffices, the addresses handed out are STABLE across fires; a growth moves
// a block and bumps `generation()`, and anything that baked the addresses (a
// future captured graph — stage6-plan.md increment 1) must treat a
// generation change as invalidation. Like the arena, this is confined to the
// single submitting thread of its stream; nothing here is thread-safe.
//
// Why ONE buffer, with no ring and no ring wait, is enough on the
// attention-phase path (ON_ATTN_PROJ / ON_ATTN): a given stage id fires once
// per model layer, and layers run SEQUENTIALLY on one stream. On the device
// side that is the whole argument — occurrence N+1's H2D metadata copy and
// every kernel reading the block are enqueued on the same stream after
// occurrence N's launches, so stream order alone keeps the overwrite
// unobservable. On the host side, the pinned block is repacked no earlier
// than occurrence N+1's prepare, by which point occurrence N's copy —
// enqueued a full model layer of stream work earlier — has normally
// retired. "Normally" is not "always": across a FIRE boundary the submitter
// can be a whole fire ahead of the GPU (measured live: a handful of stalls
// per run, at the first hook occurrence after a prefill), so `pinned()`
// keeps a `copy_done_` event that it QUERIES — no wait on the steady
// per-layer path — and only when the previous copy is genuinely still in
// flight does it synchronize, with a warning, instead of letting the
// in-flight DMA read a half-repacked block. That residual wait lives in
// PREPARE, which stays outside any captured region, so it is capture-legal;
// what this path removes is the rotating rings' unconditional
// `cudaEventSynchronize` on slot reuse (`PinnedUploadStaging::acquire`,
// `PersistentDeviceWorkspace::try_acquire`) that sat where the body — the
// captured half — runs.
class StableStageBuffers {
  public:
    StableStageBuffers() = default;
    StableStageBuffers(const StableStageBuffers&) = delete;
    StableStageBuffers& operator=(const StableStageBuffers&) = delete;

    ~StableStageBuffers() {
        if (copy_pending_ && copy_done_ != nullptr) {
            cudaEventSynchronize(copy_done_);
        }
        if (copy_done_ != nullptr) cudaEventDestroy(copy_done_);
        if (pinned_ != nullptr) cudaFreeHost(pinned_);
        // Blocks superseded by growth, parked until destruction. Their
        // uploads were enqueued before the event synchronized above on the
        // same stream, so they have retired by now.
        for (std::uint8_t* block : retired_pinned_) {
            if (block != nullptr) cudaFreeHost(block);
        }
        if (metadata_ != nullptr) cudaFree(metadata_);
        if (scratch_ != nullptr) cudaFree(scratch_);
        if (side_ != nullptr) cudaFree(side_);
    }

    // The pinned staging block, safe for the host to repack.
    std::uint8_t* pinned(std::size_t bytes) {
        if (bytes > pinned_capacity_) {
            // Same discipline as PinnedUploadStaging::acquire: growth never
            // calls cudaFreeHost on the hot path (it synchronizes the whole
            // device); the superseded block is parked until destruction —
            // an in-flight upload keeps reading valid memory — and capacity
            // grows geometrically so a stage ratchets O(log) times.
            if (pinned_ != nullptr) {
                retired_pinned_.push_back(pinned_);
                pinned_ = nullptr;
            }
            const std::size_t grown = std::max(
                {bytes, pinned_capacity_ * 2, std::size_t{4096}});
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&pinned_), grown));
            pinned_capacity_ = grown;
            copy_pending_ = false;
        } else if (copy_pending_) {
            copy_pending_ = false;
            const cudaError_t status = cudaEventQuery(copy_done_);
            if (status == cudaErrorNotReady) {
                // Fire-boundary run-ahead (see the class comment): rare,
                // loud, and correct — never silent DMA corruption.
                if (pinned_stall_warnings_++ == 0) {
                    std::fprintf(
                        stderr,
                        "[pie-driver-cuda] stable stage staging: previous "
                        "metadata upload still in flight at repack; "
                        "synchronizing (submitter ran ahead of a full "
                        "layer)\n");
                }
                CUDA_CHECK(cudaEventSynchronize(copy_done_));
            } else if (status != cudaSuccess) {
                CUDA_CHECK(status);
            }
        }
        return pinned_;
    }

    void record_upload(cudaStream_t stream) {
        if (copy_done_ == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &copy_done_, cudaEventDisableTiming));
        }
        const cudaError_t status = cudaEventRecord(copy_done_, stream);
        if (status != cudaSuccess) {
            cudaStreamSynchronize(stream);
            CUDA_CHECK(status);
        }
        copy_pending_ = true;
    }

    // The stable device metadata and scratch blocks, grown to fit.
    std::pair<std::uint8_t*, std::uint8_t*> device_blocks(
        std::size_t metadata_bytes,
        std::size_t scratch_bytes,
        cudaStream_t stream) {
        grow_device(&metadata_, &metadata_capacity_, metadata_bytes,
                    stream, "metadata");
        grow_device(&scratch_, &scratch_capacity_, scratch_bytes,
                    stream, "scratch");
        return {metadata_, scratch_};
    }

    // A stable per-region side-table block (increment 4): per-lane device
    // tables the region launches take as KERNEL ARGUMENTS (the page-mask
    // dest table) live here instead of a per-fire cudaMallocAsync, so a
    // captured launch's baked table address is the address every later
    // prepare re-uploads into. Same growth discipline (and the same
    // generation counter) as the metadata/scratch blocks.
    std::uint8_t* side_block(std::size_t bytes, cudaStream_t stream) {
        grow_device(&side_, &side_capacity_, bytes, stream, "side");
        return side_;
    }

    // Bumped on every device-block growth. See the capture precondition in
    // the class comment (the HookSidebandArena::generation() contract).
    std::uint64_t generation() const noexcept { return generation_; }

    std::size_t retained_device_bytes() const noexcept {
        return metadata_capacity_ + scratch_capacity_;
    }

  private:
    void grow_device(
        std::uint8_t** block,
        std::size_t* capacity,
        std::size_t bytes,
        cudaStream_t stream,
        const char* what) {
        if (bytes <= *capacity) return;
        // Growth path, mirroring HookSidebandArena::acquire: retire
        // everything in flight that may still read the old block, then
        // free+realloc. Synchronous and logged because it is meant to be
        // rare — the steady state is the capacity check above (a stage's
        // first fire, then never again until its lattice widens).
        std::size_t grown = std::max<std::size_t>(
            {bytes, *capacity * 2, std::size_t{4096}});
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (*block != nullptr) {
            cudaFree(*block);
            *block = nullptr;
            *capacity = 0;
        }
        void* fresh = nullptr;
        CUDA_CHECK(cudaMalloc(&fresh, grown));
        *block = static_cast<std::uint8_t*>(fresh);
        *capacity = grown;
        ++generation_;
        std::fprintf(
            stderr,
            "[pie-driver-cuda] generated stable stage %s grew to %zu KiB "
            "(need %zu B, generation %llu)\n",
            what, grown >> 10, bytes,
            static_cast<unsigned long long>(generation_));
    }

    std::uint8_t* pinned_ = nullptr;
    std::size_t pinned_capacity_ = 0;
    std::vector<std::uint8_t*> retired_pinned_;
    cudaEvent_t copy_done_ = nullptr;
    bool copy_pending_ = false;
    std::uint64_t pinned_stall_warnings_ = 0;
    std::uint8_t* metadata_ = nullptr;
    std::size_t metadata_capacity_ = 0;
    std::uint8_t* scratch_ = nullptr;
    std::size_t scratch_capacity_ = 0;
    std::uint8_t* side_ = nullptr;
    std::size_t side_capacity_ = 0;
    std::uint64_t generation_ = 0;
};

inline std::size_t align_generated(std::size_t value) {
    return (value + 255) / 256 * 256;
}

inline GeneratedValueDesc describe_generated_value(
    const plan::ValueType& type,
    const GroupedLaneBinding& lane) {
    GeneratedValueDesc descriptor{};
    descriptor.dtype = type.dtype;
    descriptor.rank = static_cast<std::uint32_t>(type.dims.size());
    std::uint64_t length = 1;
    for (std::size_t dimension = 0;
         dimension < type.dims.size();
         ++dimension) {
        const std::uint32_t value =
            grouped_dimension(type, dimension, lane);
        if (value == 0 ||
            length > std::numeric_limits<std::uint32_t>::max() / value) {
            throw std::runtime_error(
                "generated fused value shape exceeds u32");
        }
        descriptor.dims[dimension] = value;
        length *= value;
    }
    descriptor.len = static_cast<std::uint32_t>(length);
    descriptor.rows = 1;
    if (descriptor.rank >= 2) {
        std::uint64_t rows = 1;
        for (std::size_t dimension = 0;
             dimension + 1 < type.dims.size();
             ++dimension) {
            rows *= descriptor.dims[dimension];
        }
        descriptor.rows = static_cast<std::uint32_t>(rows);
    }
    descriptor.last = descriptor.len / descriptor.rows;
    return descriptor;
}

inline std::size_t generated_value_bytes(
    const GeneratedValueDesc& descriptor) {
    return std::max<std::size_t>(
        descriptor.dtype == PTIR_DT_BOOL
            ? descriptor.len
            : static_cast<std::size_t>(descriptor.len) * 4,
        4);
}

template <class T>
inline void upload_generated(
    T* destination,
    const std::vector<T>& source,
    cudaStream_t stream) {
    if (source.empty()) return;
    CUDA_CHECK(cudaMemcpyAsync(
        destination,
        source.data(),
        source.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream));
}

}  // namespace detail

class GeneratedRuntimeContext {
  public:
    GeneratedRuntimeContext() = default;
    GeneratedRuntimeContext(const GeneratedRuntimeContext&) = delete;
    GeneratedRuntimeContext& operator=(const GeneratedRuntimeContext&) = delete;

    detail::PinnedUploadStaging& metadata_staging(
        std::uint64_t runtime_id,
        cudaStream_t stream) {
        return *entry(runtime_id, stream).staging;
    }

    // The prepared path's stable buffers (see detail::StableStageBuffers):
    // one per (stage runtime id, stream, slot), sized on first use. `slot`
    // is 0 on the eager per-layer path (one buffer per stage, repacked
    // between that stage's occurrences). The hook-graph prepare pass
    // (stage 6 increment 4) instead hands every prepared (phase, layer,
    // occurrence, group) its own nonzero slot: all of a fire's prepares run
    // BEFORE the body launches, so occurrence N+1's repack can no longer
    // hide behind stream order — distinct occurrences need distinct stable
    // blocks, and a captured graph bakes each block's address per slot.
    detail::StableStageBuffers& stable_stage_buffers(
        std::uint64_t runtime_id,
        cudaStream_t stream,
        std::uint32_t slot = 0) {
        return entry(runtime_id, stream, slot).stable;
    }

    std::optional<detail::PersistentDeviceLease> acquire_device_workspace(
        std::uint64_t runtime_id,
        cudaStream_t stream,
        std::size_t metadata_bytes,
        std::size_t scratch_bytes) {
        Entry& selected = entry(runtime_id, stream);
        auto acquired = selected.workspace.try_acquire(
            metadata_bytes,
            scratch_bytes,
            stream,
            kMaximumRetainedDeviceBytes -
                std::min(
                    retained_device_bytes_,
                    kMaximumRetainedDeviceBytes));
        if (!acquired.has_value()) return std::nullopt;
        retained_device_bytes_ += acquired->second;
        return std::optional<detail::PersistentDeviceLease>(
            std::move(acquired->first));
    }

    void clear() {
        entries_.clear();
        retained_device_bytes_ = 0;
    }
    std::size_t size() const noexcept { return entries_.size(); }
    std::size_t retained_device_bytes() const noexcept {
        return retained_device_bytes_;
    }

  private:
    // Raised 1024 -> 4096 for increment 4: the hook-graph prepare pass keys
    // stable buffers per (stage, stream, occurrence slot), so one hook-fire
    // structure contributes O(model layers) entries per attention stage.
    static constexpr std::size_t kMaximumEntries = 4096;
    static constexpr std::size_t kMaximumRetainedDeviceBytes =
        512ull * 1024ull * 1024ull;

    struct Key {
        std::uint64_t runtime_id = 0;
        cudaStream_t stream = nullptr;
        std::uint32_t slot = 0;
        bool operator==(const Key&) const = default;
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            const std::size_t runtime =
                std::hash<std::uint64_t>{}(key.runtime_id);
            const std::size_t stream =
                std::hash<const void*>{}(key.stream);
            return runtime ^ (stream + 0x9e3779b9u +
                              (runtime << 6) + (runtime >> 2)) ^
                   (static_cast<std::size_t>(key.slot) * 0x85ebca6bu);
        }
    };

    struct Entry {
        std::unique_ptr<detail::PinnedUploadStaging> staging;
        detail::PersistentDeviceWorkspace workspace;
        detail::StableStageBuffers stable;
        std::uint64_t last_use = 0;
    };

    Entry& entry(
        std::uint64_t runtime_id,
        cudaStream_t stream,
        std::uint32_t slot = 0) {
        if (runtime_id == 0) {
            throw std::runtime_error(
                "generated executable has no runtime identity");
        }
        const Key key{runtime_id, stream, slot};
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
            found->second.last_use = ++clock_;
            return found->second;
        }
        if (entries_.size() >= kMaximumEntries) {
            throw std::runtime_error(
                "generated runtime context is at capacity");
        }
        auto staging =
            std::make_unique<detail::PinnedUploadStaging>();
        auto [inserted, _] = entries_.try_emplace(key);
        inserted->second.staging = std::move(staging);
        inserted->second.last_use = ++clock_;
        return inserted->second;
    }

    std::unordered_map<Key, Entry, KeyHash> entries_;
    std::uint64_t clock_ = 0;
    std::size_t retained_device_bytes_ = 0;
};

static __global__ void k_generated_scan_f32(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    GroupedRowShape shape,
    bool product) {
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / shape.max_rows;
    const std::uint32_t row = grouped_row % shape.max_rows;
    if (lane >= header->lane_count ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const std::uint32_t rows =
        grouped_lane_rows(lanes[lane], shape);
    const std::uint32_t columns =
        grouped_lane_columns(lanes[lane], shape);
    if (row >= rows) return;
    const float* source =
        grouped_value<float>(values, value_count, lane, input) +
        static_cast<std::size_t>(row) * columns;
    float* destination =
        grouped_value<float>(values, value_count, lane, output) +
        static_cast<std::size_t>(row) * columns;
    float accumulated = product ? 1.0f : 0.0f;
    for (std::uint32_t column = 0; column < columns; ++column) {
        accumulated = product
            ? accumulated * source[column]
            : accumulated + source[column];
        destination[column] = accumulated;
    }
}

static __global__ void k_generated_scan_u32(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    GroupedRowShape shape,
    bool product) {
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / shape.max_rows;
    const std::uint32_t row = grouped_row % shape.max_rows;
    if (lane >= header->lane_count ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const std::uint32_t rows =
        grouped_lane_rows(lanes[lane], shape);
    const std::uint32_t columns =
        grouped_lane_columns(lanes[lane], shape);
    if (row >= rows) return;
    const std::uint32_t* source =
        grouped_value<std::uint32_t>(
            values, value_count, lane, input) +
        static_cast<std::size_t>(row) * columns;
    std::uint32_t* destination =
        grouped_value<std::uint32_t>(
            values, value_count, lane, output) +
        static_cast<std::size_t>(row) * columns;
    std::uint32_t accumulated = product ? 1u : 0u;
    for (std::uint32_t column = 0; column < columns; ++column) {
        accumulated = product
            ? accumulated * source[column]
            : accumulated + source[column];
        destination[column] = accumulated;
    }
}

// `attn_page_mask(mask)` — the write side of the attention hook. Threshold the
// program's per-page mask row into the model-owned keep buffer for this lane.
//
// The mask is read as "nonzero keeps", matching the DSL's documented contract.
// Anything the program did not rank cannot reach here: the host arm refuses a
// lane whose page list is longer than the declared mask, rather than letting
// the tail default either way. Defaulting to keep would silently disable the
// policy; defaulting to drop would silently evict real context.
template <typename T>
static __global__ void k_generated_attn_page_mask(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const GroupedLanePageMaskDevice* dests,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t mask_value) {
    const std::uint32_t lane = blockIdx.x;
    if (lane >= header->lane_count) return;
    if (*grouped_commit(lanes, lane) == 0) return;

    const GroupedLanePageMaskDevice d = dests[lane];
    if (d.out == nullptr) return;
    const T* mask = grouped_value<T>(values, value_count, lane, mask_value);
    if (mask == nullptr) return;

    for (std::uint32_t p = threadIdx.x; p < d.pages; p += blockDim.x) {
        d.out[p] = mask[p] != static_cast<T>(0) ? std::uint8_t{1} : std::uint8_t{0};
    }
}

// `envelope_dot` — Quest page criticality (Tang et al., arXiv:2406.10774).
//
// One block per (lane, page slot). The block reduces the per-head Quest bound
// over the GQA group and then takes the MAX over kv heads.
//
// The max-over-heads is a DELIBERATE deviation from the paper, forced by the
// consumer: Quest selects pages per head, but FlashInfer's paged layout carries
// ONE page list shared by every head of a request, so a per-head selection has
// no way to be expressed downstream. Reducing with max is the union of the
// per-head selections, which keeps a page any head considers critical. That is
// strictly more conservative than Quest: quality >= Quest, speedup <= Quest.
//
// The slice of the fire's page list belonging to this lane is resolved HERE,
// from the device CSR, not on the host. `max_pages` is a host-declared bound on
// the slot count, so the grid is wide enough for any lane; the real count comes
// from `page_indptr`. Under decode envelopes the host CSR is only an upper
// bound, so slicing with it would both start at the wrong page and score more
// slots than the request owns. See `GroupedLaneEnvelope`.
//
// Slot semantics, in the order the branches test them:
//   slot >= page_count   -> -inf. Not a page of this request at all; a
//                           top-k consumer must never choose it.
//   slot >= scored_pages -> +inf. A page this fire is still writing, so its
//                           envelope predates the tokens in it. Always keep.
//                           This is both the fail-safe direction and Quest's
//                           own "keep the local window" rule.
//   otherwise            -> the bound.
template <int BLOCK>
static __global__ void k_generated_envelope_dot(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const GroupedLaneEnvelopeDevice* envelopes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t output_value,
    std::uint32_t max_pages) {
    const std::uint32_t lane = blockIdx.x / max_pages;
    const std::uint32_t slot = blockIdx.x - lane * max_pages;
    if (lane >= header->lane_count) return;
    if (*grouped_commit(lanes, lane) == 0) return;

    const GroupedLaneEnvelopeDevice e = envelopes[lane];
    float* out = grouped_value<float>(values, value_count, lane, output_value);
    if (out == nullptr || e.env_min == nullptr) return;

    const std::uint32_t begin = e.page_indptr[e.request];
    const std::uint32_t end = e.page_indptr[e.request + 1];
    const std::uint32_t page_count = end >= begin ? end - begin : 0u;

    if (slot >= page_count) {
        if (threadIdx.x == 0) out[slot] = -CUDART_INF_F;
        return;
    }

    // `kv_last_page_lens` is POST-append (see `write_kv_kernel`), so the KV
    // length this fire will END at is derivable, and subtracting the fire's own
    // tokens gives the length the envelopes currently describe. Only pages
    // entirely below that are final.
    const std::uint32_t kv_after =
        (page_count - 1u) * e.page_size + e.last_page_lens[e.request];
    const std::uint32_t kv_before =
        kv_after >= e.qo_len ? kv_after - e.qo_len : 0u;
    const std::uint32_t scored_pages =
        min(e.page_size == 0u ? 0u : kv_before / e.page_size, page_count);

    if (slot >= scored_pages) {
        if (threadIdx.x == 0) out[slot] = CUDART_INF_F;
        return;
    }

    const int page = static_cast<int>(e.page_ids[begin + slot]);
    const int num_kv_heads = static_cast<int>(e.num_kv_heads);
    const int head_dim = static_cast<int>(e.head_dim);
    const int group = static_cast<int>(e.num_q_heads) / num_kv_heads;

    __shared__ float buf[BLOCK];
    float best = -CUDART_INF_F;
    for (int kh = 0; kh < num_kv_heads; ++kh) {
        const long env_base =
            (static_cast<long>(page) * num_kv_heads + kh) * head_dim;
        buf[threadIdx.x] = kernels::envelope_dot_thread_partial(
            e.query, e.env_min, e.env_max, env_base, kh * group, group,
            head_dim, static_cast<int>(threadIdx.x), BLOCK);
        __syncthreads();
        for (int off = BLOCK / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
            __syncthreads();
        }
        if (buf[0] > best) best = buf[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[slot] = best;
}

static __global__ void k_generated_matmul_f32(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t left_value,
    std::uint32_t right_value,
    std::uint32_t output_value,
    GroupedRowShape left_shape,
    GroupedRowShape right_shape) {
    const std::uint32_t max_rows = left_shape.max_rows;
    const std::uint32_t max_columns = right_shape.max_columns;
    const std::uint64_t per_lane =
        static_cast<std::uint64_t>(max_rows) * max_columns;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * per_lane;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) +
                 threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / per_lane);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t local = flat % per_lane;
        const std::uint32_t row =
            static_cast<std::uint32_t>(local / max_columns);
        const std::uint32_t column =
            static_cast<std::uint32_t>(local % max_columns);
        const std::uint32_t rows =
            grouped_lane_rows(lanes[lane], left_shape);
        const std::uint32_t inner =
            grouped_lane_columns(lanes[lane], left_shape);
        const std::uint32_t right_rows =
            grouped_lane_rows(lanes[lane], right_shape);
        const std::uint32_t columns =
            grouped_lane_columns(lanes[lane], right_shape);
        if (row >= rows || column >= columns || inner != right_rows) {
            continue;
        }
        const float* left = grouped_value<float>(
            values, value_count, lane, left_value);
        const float* right = grouped_value<float>(
            values, value_count, lane, right_value);
        float sum = 0.0f;
        for (std::uint32_t k = 0; k < inner; ++k) {
            const float left_element =
                left[static_cast<std::size_t>(row) * inner + k];
            if (left_element == 0.0f) continue;
            const float product = __fmul_rn(
                left_element,
                right[static_cast<std::size_t>(k) * columns + column]);
            sum = __fadd_rn(sum, product);
        }
        grouped_value<float>(
            values,
            value_count,
            lane,
            output_value)[static_cast<std::size_t>(row) * columns + column] =
            sum;
    }
}

inline bool generated_stage_supported(
    const FusedStageExecutable& executable,
    const plan::StagePlan& stage,
    std::string* reason = nullptr) {
    auto fail = [&](const char* message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    if (executable.signature_hash != stage.signature_hash ||
        executable.regions.size() != stage.fused.regions.size() ||
        executable.region_analysis.size() != stage.fused.regions.size()) {
        return fail("compiled fused stage identity mismatch");
    }
    for (std::size_t index = 0;
         index < stage.fused.regions.size();
         ++index) {
        const auto& region = stage.fused.regions[index];
        if (region.library) {
            if (region.library_op == PTIR_LIBRARY_SECOND_PARTY) {
                if (!executable.region_analysis[index]
                         .second_party_supported()) {
                    return fail(
                        "fused stage still requires a second-party library");
                }
                continue;
            }
            if (region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE) {
                if (!grouped_nucleus_region_supported(stage, region)) {
                    return fail("nucleus library region ABI is invalid");
                }
                continue;
            }
            if (region.nodes.size() != 1) {
                return fail("stock library region has invalid node count");
            }
            continue;
        }
        if (executable.regions[index] == nullptr ||
            executable.regions[index]->function == nullptr) {
            return fail("generated fused region is unavailable");
        }
    }
    return true;
}

inline std::optional<std::uint32_t>
generated_compact_argmax_value(
    const plan::StagePlan& stage,
    const std::vector<std::uint32_t>& bases,
    const StageRegionAnalysis& direct) {
    if (stage.fused.regions.size() != 1 ||
        stage.fused.regions.front().library) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> aliases(stage.value_types.size());
    for (std::uint32_t value = 0; value < aliases.size(); ++value) {
        aliases[value] = value;
    }
    auto resolve_alias = [&](std::uint32_t value) {
        while (aliases[value] != value) value = aliases[value];
        return value;
    };
    std::uint32_t intrinsic_value = UINT32_MAX;
    std::uint32_t argmax_value = UINT32_MAX;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        const auto& op = stage.ops[node].op;
        if (op.tag == PTIR_OP_RESHAPE && op.results == 1) {
            aliases[bases[node]] = resolve_alias(op.args[0]);
            continue;
        }
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.intr == PTIR_INTR_LOGITS && op.results == 1) {
            if (intrinsic_value != UINT32_MAX) return std::nullopt;
            intrinsic_value = bases[node];
            continue;
        }
        if (op.tag == PTIR_OP_REDUCE_ARGMAX && op.results == 1) {
            const StageRegionArgmax* record =
                direct.argmax_for(static_cast<std::uint32_t>(node));
            if (argmax_value != UINT32_MAX ||
                resolve_alias(op.args[0]) != intrinsic_value ||
                record == nullptr ||
                record->intrinsic != PTIR_INTR_LOGITS) {
                return std::nullopt;
            }
            argmax_value = bases[node];
            continue;
        }
        if (op.tag == PTIR_OP_CHAN_PUT && !op.args.empty()) {
            if (argmax_value == UINT32_MAX ||
                resolve_alias(op.args[0]) != argmax_value) {
                return std::nullopt;
            }
            continue;
        }
        return std::nullopt;
    }
    if (intrinsic_value == UINT32_MAX ||
        argmax_value == UINT32_MAX) {
        return std::nullopt;
    }
    return argmax_value;
}

// ---------------------------------------------------------------------------
// The prepare/body split (stage6-plan.md increment 1).
//
// `run_generated_stage` (today the Prologue/Epilogue-only
// `run_generated_stage_ring`) used to be one eager monolith: per fire it rebuilt
// the host metadata block, leased rotating pinned/device ring slots (with
// event waits on slot reuse), uploaded, and launched — the same defect the
// attention planner had before it was hoisted (batch/forward_graph.hpp:14-24:
// no host-side work and no event waits may live where a CUDA graph will be
// captured). The split puts every piece of host work — metadata build,
// elision analysis, channel-cursor reads, staging pack, async upload,
// per-region launch-shape resolution and side-table uploads — into
// `prepare_generated_stage`, and leaves `launch_generated_stage` a body that
// only issues kernel launches against prepared state. Attention phases now
// go through this seam exclusively — hoisted to fire level for prepared
// pure-decode hook fires (eager and graph alike, since the eager-path
// unification) or called back-to-back at hook time for prepare-vetoed
// fires; only Prologue/Epilogue still take the ring-backed combined wrapper
// (`run_generated_stage_ring`).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Lane-grid lattice (stage6-plan.md increment 3 — the capture prerequisite).
//
// A captured CUDA graph freezes every kernel's grid dimensions, so any grid
// derived from the EXACT per-fire lane count would force one graph per lane
// count. Instead, every lane-derived grid dimension in the body
// (`launch_generated_stage`) is baked at the lattice ceiling of the fire's
// lane count, and the LIVE count travels to the device as data: it is
// `PtirLaneTableHeader::lane_count` in the metadata block that every prepare
// already re-packs and re-uploads per fire. Idle blocks exit on the guard
// every kernel on this path already carries as its FIRST action — the
// grouped kernels (`k_grouped_stage_readiness`, `k_grouped_topk`, the
// nucleus family in grouped_runtime.cuh), the driver-side region kernels in
// this file, and the NVRTC-emitted region kernel
// (compiler/codegen/runtime/cuda/fused_block1.cuh:18-19,
// `dispatch_lane >= header->lane_count`) all guard before touching any
// lane-indexed table — so no emitter change and no golden re-bless is
// needed, and an idle block costs one global read of the header.
//
// The ladder is powers of two rather than `forward_graph_request_bucket`'s
// R lattice (batch/forward_graph.hpp:66: 1,2,4, x8 to 256, then x16). That
// lattice earns its shape from decode batches in the hundreds, and its
// "max_requests is a legal bucket" rule needs a planner bound this layer
// does not have. Lane counts here are the number of programs grouped into
// one stage fire — tens, not hundreds — so pow2 pads by at most 2x in idle
// blocks whose whole cost is the guard read, and gives a future capture
// cache at most ~log2(max lanes) distinct grids per stage.
constexpr std::uint32_t generated_lane_grid_bucket(
    std::uint32_t lane_count) noexcept {
    // 2^31 is the largest u32 power of two; nothing real approaches it
    // (prepare rejects empty fires and lanes are per-fire programs), but
    // stay total rather than overflow the shift.
    if (lane_count > (std::uint32_t{1} << 31)) return lane_count;
    std::uint32_t bucket = 1;
    while (bucket < lane_count) bucket <<= 1;
    return bucket;
}
static_assert(generated_lane_grid_bucket(0) == 1);
static_assert(generated_lane_grid_bucket(1) == 1);
static_assert(generated_lane_grid_bucket(2) == 2);
static_assert(generated_lane_grid_bucket(3) == 4);
static_assert(generated_lane_grid_bucket(4) == 4);
static_assert(generated_lane_grid_bucket(5) == 8);
static_assert(generated_lane_grid_bucket(16) == 16);
static_assert(generated_lane_grid_bucket(17) == 32);
static_assert(generated_lane_grid_bucket(1000) == 1024);

enum class PreparedBufferMode : std::uint8_t {
    // Rotating pinned/device rings with event-guarded slot reuse — the
    // pre-split storage. Still used by the phases that have not migrated to
    // the prepared path (Prologue / Epilogue this slice).
    kRotatingRing,
    // ONE stable pinned block + ONE stable device block per
    // (stage runtime id, stream) — no rings, no ring waits
    // (detail::StableStageBuffers). The attention phases run here, and it is
    // the storage discipline a captured body will replay against.
    kStablePerStage,
};

// One region of a prepared stage, resolved to the point where launching it
// requires no host work: kind, grid geometry, device-side argument tables
// (already uploaded by prepare), and value indices.
struct PreparedRegionLaunch {
    enum class Kind : std::uint8_t {
        kGenerated,
        kNucleus,
        kNucleusScalable,
        kPageMask,
        kEnvelopeDot,
        kScan,
        kMatmul,
        kTopK,
    };
    Kind kind = Kind::kGenerated;
    // kGenerated — the NVRTC-compiled region function; every generated
    // region of a stage shares the prepared argument block.
    const FusedRegionExecutable* region = nullptr;
    // kNucleus / kNucleusScalable
    GroupedNucleusLaunch nucleus{};
    std::uint32_t nucleus_segments = 0;      // padded lanes * rows (grid)
    float* partial_maxima = nullptr;         // kNucleusScalable, device
    float* thread_masses = nullptr;          // kNucleusScalable, device
    float* probabilities = nullptr;          // kNucleusScalable, device
    // kPageMask
    GroupedLanePageMaskDevice* device_dests = nullptr;
    std::uint8_t mask_dtype = 0;
    // kEnvelopeDot
    GroupedLaneEnvelopeDevice* device_envelopes = nullptr;
    std::uint32_t max_pages = 0;
    // kScan
    bool scan_f32 = false;
    bool scan_product = false;
    std::uint32_t scan_blocks = 0;           // padded lanes * max_rows (grid)
    GroupedRowShape shape_a{};               // scan input / matmul left / topk rows
    GroupedRowShape shape_b{};               // matmul right
    GroupedDynamicShape dynamic_shape{};     // topk input
    std::uint32_t value_a = 0;               // input value id
    std::uint32_t value_b = 0;               // matmul right value id
    std::uint32_t out_a = 0;                 // result value id
    std::uint32_t out_b = 0;                 // topk indices value id
    // kMatmul
    std::uint32_t matmul_grid = 0;
    // kTopK
    std::uint32_t topk_rows = 0;
    std::uint32_t topk_length = 0;
    std::uint32_t topk_k = 0;
    bool topk_is_sort = false;
    std::uint32_t topk_vocab = 0;
};

// Everything the body half needs and nothing it must compute: the uploaded
// metadata block's typed device pointers, the resolved region launches, and
// the ownership that keeps the fire's transient device tables alive until
// the caller is past the launches. Heap-owned so the kernel-argument block
// below has a stable address for cuLaunchKernel (and, later, capture).
struct GeneratedStagePrepared {
    explicit GeneratedStagePrepared(cudaStream_t launch_stream)
        : stream(launch_stream), allocations(launch_stream) {}
    GeneratedStagePrepared(const GeneratedStagePrepared&) = delete;
    GeneratedStagePrepared& operator=(const GeneratedStagePrepared&) = delete;

    cudaStream_t stream = nullptr;
    // Per-fire transient device tables (bf16 row tables, page-mask dests,
    // envelope tables, nucleus partials). Freed stream-ordered when the
    // prepared state is destroyed — after the body's launches, exactly where
    // the combined path freed them. Stable-address versions of these are
    // stage6-plan.md increment 2 territory.
    detail::AsyncAllocations allocations;
    // Ring path only; empty under PreparedBufferMode::kStablePerStage.
    std::optional<detail::PersistentDeviceLease> device_workspace;

    std::uint32_t lane_count = 0;
    // The lane-grid lattice ceiling of `lane_count`
    // (`generated_lane_grid_bucket`). Every lane-derived grid dimension the
    // body launches is sized by THIS, never by the live count — the live
    // count reaches the kernels as `device_header->lane_count` and blocks
    // beyond it guard-exit. This is what lets a captured body replay across
    // fires whose lane counts share a bucket.
    std::uint32_t padded_lane_count = 0;
    std::uint32_t value_count = 0;

    // Typed views into the uploaded device metadata block.
    PtirLaneTableHeader* device_header = nullptr;
    PtirLaneRecord* device_lanes = nullptr;
    PtirLaneChannelSlot* device_channels = nullptr;
    GroupedReadinessLane* device_readiness = nullptr;
    std::uint32_t* device_readiness_slots = nullptr;
    std::uint64_t* device_value_pointers = nullptr;
    std::uint8_t* device_scratch = nullptr;

    // Channel-ring device state for the readiness kernel.
    const std::uint8_t* ring_full = nullptr;
    const std::uint32_t* ring_head = nullptr;
    const std::uint32_t* ring_tail = nullptr;
    const std::uint32_t* ring_cap1 = nullptr;
    std::uint32_t readiness_diagnose = 0;

    // Kernel-argument block for the generated regions. The values live here
    // so `generated_arguments` points at storage with the prepared state's
    // lifetime: the body reads only this block, and a captured body can hand
    // it to cuLaunchKernel unchanged.
    CUdeviceptr arg_header = 0;
    CUdeviceptr arg_lanes = 0;
    CUdeviceptr arg_channels = 0;
    CUdeviceptr arg_descriptors = 0;
    CUdeviceptr arg_params = 0;
    CUdeviceptr arg_offsets = 0;
    CUdeviceptr arg_scratch = 0;
    std::uint32_t arg_value_count = 0;
    std::uint32_t arg_scratch_stride = 0;
    std::uint32_t arg_temporary_offset = 0;
    CUdeviceptr arg_pending = 0;
    CUdeviceptr arg_intrinsic_bases = 0;
    CUdeviceptr arg_intrinsic_modes = 0;
    CUdeviceptr arg_intrinsic_widths = 0;
    CUdeviceptr arg_intrinsic_strides = 0;
    CUdeviceptr arg_intrinsic_offsets = 0;
    std::array<void*, 16> generated_arguments{};

    std::vector<PreparedRegionLaunch> regions;

    // Section timing (GroupedExecutionOptions::time_sections).
    bool timing = false;
    std::int64_t t_build_us = -1;
    std::int64_t t_workspace_us = -1;
    std::int64_t t_upload_us = -1;
    fire_timing::Clock::time_point section_mark{};
};

// The prepare half: ALL host work for one stage fire. Builds the lane /
// channel / readiness / descriptor / intrinsic tables, runs the elision
// analysis, packs the metadata block into pinned staging and enqueues its
// upload, uploads every per-region side table, and resolves every region
// into a `PreparedRegionLaunch`. After this returns, launching the stage
// touches no host state beyond the returned object.
inline std::unique_ptr<GeneratedStagePrepared> prepare_generated_stage(
    const std::vector<GroupedLaneBinding>& lanes,
    const FusedStageExecutable& executable,
    GeneratedRuntimeContext& runtime,
    cudaStream_t stream,
    GroupedExecutionOptions options = {},
    PreparedBufferMode buffer_mode = PreparedBufferMode::kRotatingRing,
    // kStablePerStage only: which stable-buffer slot backs this prepare
    // (`GeneratedRuntimeContext::stable_stage_buffers`). 0 = the eager
    // per-layer slot; the hook-graph prepare pass numbers each hoisted
    // (phase, layer, occurrence, group) distinctly.
    std::uint32_t stable_slot = 0) {
    if (lanes.empty()) {
        throw std::runtime_error("generated fused launch has no lanes");
    }
    if (options.reset_commits || options.pull_tickets || options.finalize) {
        throw std::runtime_error(
            "generated staged execution requires prevalidated commit state");
    }
    const plan::StagePlan& stage = *lanes.front().plan;
    std::string support_error;
    if (!generated_stage_supported(executable, stage, &support_error)) {
        throw std::runtime_error(support_error);
    }
    const bool timing = options.time_sections;
    const auto t_section_start = timing
        ? fire_timing::Clock::now()
        : fire_timing::Clock::time_point{};
    auto section_mark = t_section_start;
    const std::uint32_t lane_count =
        static_cast<std::uint32_t>(lanes.size());
    const std::uint32_t channel_count =
        static_cast<std::uint32_t>(stage.channel_bindings.size());
    const std::uint32_t value_count =
        static_cast<std::uint32_t>(stage.value_types.size());
    auto prepared = std::make_unique<GeneratedStagePrepared>(stream);
    prepared->timing = timing;
    prepared->lane_count = lane_count;
    prepared->padded_lane_count = generated_lane_grid_bucket(lane_count);
    prepared->value_count = value_count;
    detail::AsyncAllocations& allocations = prepared->allocations;

    std::vector<PtirLaneRecord> host_lanes(lane_count);
    std::vector<PtirLaneChannelSlot> host_channels(
        static_cast<std::size_t>(lane_count) * channel_count);
    std::vector<std::uint8_t> host_pending(
        static_cast<std::size_t>(lane_count) * channel_count, 0);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const auto& binding = lanes[lane];
        auto& record = host_lanes[lane];
        record.logits_base = reinterpret_cast<std::uint64_t>(
            binding.logits_base);
        record.logits_row_count = binding.logits_row_count;
        record.kv_len = binding.kv_len;
        record.page_count = binding.page_count;
        record.row_count = binding.row_count;
        record.token_count = binding.token_count;
        record.sampled_rows = binding.logits_row_count;
        record.query_len = binding.query_len;
        record.key_len = binding.key_len;
        record.channel_slot_offset = lane * channel_count;
        record.rng_state = reinterpret_cast<std::uint64_t>(
            binding.row_seeds);
        record.commit_slot = reinterpret_cast<std::uint64_t>(
            binding.commit_slot != nullptr
                ? binding.commit_slot
                : binding.instance->commit_device_flag());
        record.active_row_mask =
            binding.logits_row_count >= 64
                ? std::numeric_limits<std::uint64_t>::max()
                : ((std::uint64_t{1} << binding.logits_row_count) - 1);
        record.sample_output_channel_mask =
            binding.sample_output_channel_mask;
        record.row_valid = reinterpret_cast<std::uint64_t>(
            binding.row_valid);
        record.row_valid_offset = binding.row_valid_offset;
        record.reserved0 = 0;
        // Channel-cursor bake (stage6-plan.md, spike hazard (a)): the
        // protocol cursors (`expected_head`/`expected_tail`, and the
        // committed/pending CELL ADDRESSES derived from the live registry
        // head/tail below) are read HERE, per prepare, and baked into the
        // prepared metadata. That is already "per replay = per prepare"
        // semantics: a future captured body replays against the metadata
        // block this prepare uploads, so it sees the channel view of the
        // fire being prepared — but ONLY IF prepare runs before every
        // replay. A replay without a fresh prepare would re-execute a
        // previous fire's channel view (issue #24's failure class, living
        // in runtime data where no static_assert can see it).
        for (std::uint32_t local = 0; local < channel_count; ++local) {
            const std::uint32_t dense =
                binding.plan->channel_bindings[local];
            const std::uint32_t slot =
                binding.instance->view().slot(dense);
            const auto ticket = std::find_if(
                binding.tickets->begin(),
                binding.tickets->end(),
                [slot](const DeviceHostChannelTicket& candidate) {
                    return candidate.slot == slot;
                });
            auto& channel =
                host_channels[record.channel_slot_offset + local];
            if (ticket != binding.tickets->end()) {
                channel.expected_head = ticket->expected_head;
                channel.expected_tail = ticket->expected_tail;
                channel.committed_cell = reinterpret_cast<std::uint64_t>(
                    ticket->cells +
                    static_cast<std::size_t>(
                        (ticket->expected_head == kNoChannelTicket
                             ? binding.instance->view().registry()->host_head(slot)
                             : ticket->expected_head) %
                        ticket->cap1) *
                        ticket->native_bytes);
                channel.pending_cell = reinterpret_cast<std::uint64_t>(
                    ticket->cells +
                    static_cast<std::size_t>(
                        (ticket->expected_tail == kNoChannelTicket
                             ? binding.instance->view().registry()->host_tail(slot)
                             : ticket->expected_tail) %
                        ticket->cap1) *
                        ticket->native_bytes);
            } else {
                channel.expected_head = kNoChannelTicket;
                channel.expected_tail = kNoChannelTicket;
                channel.committed_cell = reinterpret_cast<std::uint64_t>(
                    binding.instance->view().committed_cell(dense));
                channel.pending_cell = reinterpret_cast<std::uint64_t>(
                    binding.instance->view().pending_cell(dense));
            }
            if (binding.prior_put_slots != nullptr &&
                binding.prior_put_slots->contains(slot)) {
                channel.committed_cell = channel.pending_cell;
                host_pending[record.channel_slot_offset + local] = 1;
            }
        }
    }
    std::vector<GroupedReadinessLane> host_readiness(lane_count);
    std::vector<std::uint32_t> readiness_slots;
    std::vector<std::uint8_t> seen_full(channel_count, 0);
    std::vector<std::uint8_t> seen_put(channel_count, 0);
    std::vector<std::uint32_t> need_full;
    std::vector<std::uint32_t> put_channels;
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.tag == PTIR_OP_CHAN_TAKE ||
            op.tag == PTIR_OP_CHAN_READ) {
            const auto channel = static_cast<std::uint32_t>(op.chan);
            if (!seen_put[channel] && !seen_full[channel]) {
                seen_full[channel] = 1;
                need_full.push_back(channel);
            }
        } else if (op.tag == PTIR_OP_CHAN_PUT) {
            const auto channel = static_cast<std::uint32_t>(op.chan);
            if (!seen_put[channel]) {
                seen_put[channel] = 1;
                put_channels.push_back(channel);
            }
        }
    }
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        auto& descriptor = host_readiness[lane];
        descriptor.full_offset =
            static_cast<std::uint32_t>(readiness_slots.size());
        for (const std::uint32_t local : need_full) {
            const std::uint32_t slot = lanes[lane].instance->view().slot(
                lanes[lane].plan->channel_bindings[local]);
            if (lanes[lane].prior_put_slots != nullptr &&
                lanes[lane].prior_put_slots->contains(slot)) {
                continue;
            }
            readiness_slots.push_back(slot);
            ++descriptor.full_count;
        }
        descriptor.empty_offset =
            static_cast<std::uint32_t>(readiness_slots.size());
        for (const std::uint32_t local : put_channels) {
            if (seen_full[local]) continue;
            const std::uint32_t slot = lanes[lane].instance->view().slot(
                lanes[lane].plan->channel_bindings[local]);
            const bool prior_put =
                lanes[lane].prior_put_slots != nullptr &&
                lanes[lane].prior_put_slots->contains(slot);
            const bool prior_take =
                lanes[lane].prior_take_slots != nullptr &&
                lanes[lane].prior_take_slots->contains(slot);
            if (prior_put || prior_take) continue;
            readiness_slots.push_back(slot);
            ++descriptor.empty_count;
        }
    }
    std::vector<GeneratedValueDesc> host_descriptors;
    host_descriptors.reserve(
        static_cast<std::size_t>(lane_count) * value_count);
    std::vector<std::size_t> maximum_value_bytes(value_count, 4);
    std::size_t maximum_value_length = 1;
    for (const auto& lane : lanes) {
        for (std::size_t value = 0; value < value_count; ++value) {
            GeneratedValueDesc descriptor =
                detail::describe_generated_value(
                    stage.value_types[value], lane);
            maximum_value_bytes[value] = std::max(
                maximum_value_bytes[value],
                detail::generated_value_bytes(descriptor));
            maximum_value_length = std::max<std::size_t>(
                maximum_value_length, descriptor.len);
            host_descriptors.push_back(descriptor);
        }
    }
    std::vector<std::uint32_t> bases(stage.ops.size());
    std::uint32_t planned_values = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        bases[node] = planned_values;
        planned_values += stage.ops[node].op.results;
    }
    if (planned_values != value_count) {
        throw std::runtime_error(
            "generated fused value layout mismatch");
    }
    std::vector<std::uint8_t> temporary_elided(value_count, 0);
    std::vector<std::uint8_t> absorbed_scale_nodes(
        stage.ops.size(), 0);
    // The nucleus kernels can read the pre-scale logits straight out of the
    // lane's bf16 rows, so that value need not be materialized. That only holds
    // when the value really is the logits intrinsic, optionally reshaped --
    // exactly what `direct_logits_kind` walks at the launch site. Any other
    // producer (a broadcast of one read-out row across B sampling lanes, say)
    // still runs and still writes a full tensor, so shrinking its slot to four
    // bytes lets it overrun the values laid out after it.
    auto produces_logits_intrinsic = [&](std::uint32_t value) {
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const auto& op = stage.ops[node].op;
            if (value < bases[node] ||
                value >= bases[node] + op.results) {
                continue;
            }
            return op.tag == PTIR_OP_INTRINSIC_VAL &&
                (op.intr == PTIR_INTR_LOGITS ||
                 op.intr == PTIR_INTR_MTP_LOGITS);
        }
        return false;
    };
    auto resolves_to_logits_intrinsic = [&](std::uint32_t value) {
        if (produces_logits_intrinsic(value)) return true;
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const auto& op = stage.ops[node].op;
            if (value < bases[node] ||
                value >= bases[node] + op.results) {
                continue;
            }
            return op.tag == PTIR_OP_RESHAPE && !op.args.empty() &&
                produces_logits_intrinsic(op.args[0]);
        }
        return false;
    };
    // The scale divide is only folded into the nucleus kernels when the driver
    // actually skips it, and it skips it only when the divide is a region of its
    // own. Fused into a larger generated region the divide still runs and still
    // writes a full tensor, so its value has to keep a full slot.
    std::vector<std::uint8_t> lone_generated_node(stage.ops.size(), 0);
    for (const auto& region : stage.fused.regions) {
        if (!region.library && region.nodes.size() == 1) {
            lone_generated_node[region.nodes[0]] = 1;
        }
    }
    for (const auto& region : stage.fused.regions) {
        if (!region.library ||
            region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
            region.inputs.size() != 5) {
            continue;
        }
        const bool direct_logits = std::all_of(
                lanes.begin(), lanes.end(),
                [](const auto& lane) {
                    return lane.logits_bf16_rows != nullptr;
                });
        if (direct_logits &&
            resolves_to_logits_intrinsic(region.inputs[0])) {
            maximum_value_bytes[region.inputs[0]] = 4;
            temporary_elided[region.inputs[0]] = 1;
        }
        const std::uint32_t scaled_value = region.inputs[2];
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const auto& op = stage.ops[node].op;
            if (scaled_value < bases[node] ||
                scaled_value >= bases[node] + op.results) {
                continue;
            }
            if (lone_generated_node[node] == 0) break;
            absorbed_scale_nodes[node] = 1;
            maximum_value_bytes[scaled_value] = 4;
            temporary_elided[scaled_value] = 1;
            if (op.tag == PTIR_OP_DIV && !op.args.empty() &&
                resolves_to_logits_intrinsic(op.args[0])) {
                const std::uint32_t reshape_value = op.args[0];
                maximum_value_bytes[reshape_value] = 4;
                temporary_elided[reshape_value] = 1;
                for (std::size_t producer = 0;
                     producer < stage.ops.size();
                     ++producer) {
                    const auto& source_op = stage.ops[producer].op;
                    if (reshape_value < bases[producer] ||
                        reshape_value >=
                            bases[producer] + source_op.results) {
                        continue;
                    }
                    if (source_op.tag == PTIR_OP_RESHAPE) {
                        absorbed_scale_nodes[producer] = 1;
                    }
                    break;
                }
            }
            break;
        }
    }
    for (std::size_t region_index = 0;
         region_index < stage.fused.regions.size();
         ++region_index) {
        const auto& region = stage.fused.regions[region_index];
        if (region.library) {
            if (region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE) {
                for (const std::uint32_t node : region.nodes) {
                    const auto& op = stage.ops[node].op;
                    for (std::uint32_t result = 0;
                         result < op.results;
                         ++result) {
                        const std::uint32_t value = bases[node] + result;
                        if (std::find(
                                region.outputs.begin(),
                                region.outputs.end(),
                                value) == region.outputs.end()) {
                            maximum_value_bytes[value] = 4;
                            temporary_elided[value] = 1;
                        }
                    }
                }
            }
            continue;
        }
        const StageRegionAnalysis& direct =
            executable.region_analysis[region_index];
        for (const std::uint32_t node : region.nodes) {
            const StageRegionArgmax* record = direct.argmax_for(node);
            if (record != nullptr && record->requires_single_row != 0) {
                const std::uint32_t source = record->source_value;
                for (std::uint32_t lane = 0;
                     lane < lane_count;
                     ++lane) {
                    if (host_descriptors[
                            static_cast<std::size_t>(lane) *
                                value_count +
                            source]
                            .rows != 1) {
                        throw std::runtime_error(
                            "direct argmax reshape requires one runtime row");
                    }
                }
            }
            const auto& op = stage.ops[node].op;
            const bool internal_reshape =
                op.tag == PTIR_OP_RESHAPE &&
                std::find(
                    region.outputs.begin(),
                    region.outputs.end(),
                    bases[node]) == region.outputs.end();
            const bool skipped = direct.is_skipped(node);
            if (!skipped && !internal_reshape) continue;
            for (std::uint32_t result = 0; result < op.results; ++result) {
                maximum_value_bytes[bases[node] + result] = 4;
                if (skipped) {
                    temporary_elided[bases[node] + result] = 1;
                }
            }
        }
    }
    std::size_t maximum_nucleus_value_length = 0;
    for (const auto& region : stage.fused.regions) {
        if (!region.library ||
            region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
            region.inputs.empty()) {
            continue;
        }
        const std::uint32_t logits = region.inputs[0];
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            maximum_nucleus_value_length =
                std::max<std::size_t>(
                    maximum_nucleus_value_length,
                    host_descriptors[
                        static_cast<std::size_t>(lane) * value_count +
                        logits]
                        .len);
        }
    }
    maximum_value_length = 1;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::uint32_t value = 0; value < value_count; ++value) {
            if (temporary_elided[value] != 0) continue;
            maximum_value_length = std::max<std::size_t>(
                maximum_value_length,
                host_descriptors[
                    static_cast<std::size_t>(lane) * value_count +
                    value]
                    .len);
        }
    }
    const std::optional<std::uint32_t> direct_argmax_value =
        generated_compact_argmax_value(
            stage, bases, executable.region_analysis.front());
    std::vector<std::uint32_t> host_offsets(value_count);
    std::size_t scratch_stride = 0;
    std::size_t temporary_offset_size = 0;
    if (direct_argmax_value.has_value()) {
        host_offsets[*direct_argmax_value] = 0;
        temporary_offset_size = 256;
        scratch_stride = 512;
    } else {
        scratch_stride = 256;
        for (std::size_t value = 0; value < value_count; ++value) {
            scratch_stride = detail::align_generated(scratch_stride);
            if (scratch_stride >
                std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "generated fused value offset exceeds u32");
            }
            host_offsets[value] =
                static_cast<std::uint32_t>(scratch_stride);
            if (maximum_value_bytes[value] >
                std::numeric_limits<std::size_t>::max() -
                    scratch_stride) {
                throw std::runtime_error(
                    "generated fused scratch layout overflows size_t");
            }
            scratch_stride +=
                detail::align_generated(maximum_value_bytes[value]);
        }
        temporary_offset_size =
            detail::align_generated(scratch_stride);
        if (temporary_offset_size >
            std::numeric_limits<std::uint32_t>::max() ||
            maximum_value_length >
                std::numeric_limits<std::size_t>::max() / 32) {
            throw std::runtime_error(
                "generated fused scratch exceeds u32");
        }
        const std::size_t temporary_bytes =
            detail::align_generated(maximum_value_length * 32);
        if (temporary_bytes >
            std::numeric_limits<std::size_t>::max() -
                temporary_offset_size) {
            throw std::runtime_error(
                "generated fused temporary layout overflows size_t");
        }
        scratch_stride =
            temporary_offset_size + temporary_bytes;
    }
    if (scratch_stride > std::numeric_limits<std::uint32_t>::max() ||
        (lane_count != 0 &&
         scratch_stride >
             std::numeric_limits<std::size_t>::max() / lane_count)) {
        throw std::runtime_error("generated fused scratch exceeds u32");
    }
    const std::uint32_t temporary_offset =
        static_cast<std::uint32_t>(temporary_offset_size);

    std::vector<GeneratedOpParams> host_params(stage.ops.size());
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        const auto& op = stage.ops[node].op;
        auto& param = host_params[node];
        param.tag = op.tag;
        param.a0 = !op.args.empty() ? op.args[0] : 0;
        param.a1 = op.args.size() > 1
            ? op.args[1]
            : (op.tag == PTIR_OP_PIVOT_THRESHOLD
                   ? op.pred_payload
                   : 0);
        param.a2 = op.args.size() > 2 ? op.args[2] : 0;
        param.o0 = op.results > 0 ? bases[node] : param.a0;
        param.o1 = op.results > 1 ? bases[node] + 1 : param.o0;
        param.imm = op.imm;
        param.imm2 = op.imm2;
        param.imm3 = op.imm3;
        param.kind = op.kind;
        param.pred_tag = op.pred_tag;
        param.lit_dtype = op.lit_dtype;
        param.lit_bits = op.lit_bits;
        param.intr = op.intr;
        param.bool_storage = 0;
        if (op.tag == PTIR_OP_CHAN_PUT) {
            param.sink_bytes = static_cast<std::uint32_t>(
                grouped_channel_bytes(
                    lanes.front(),
                    static_cast<std::uint32_t>(op.chan)));
        }
    }
    std::vector<std::uint64_t> host_intrinsic_bases(lane_count * kPtirIntrinsicSlots, 0);
    std::vector<std::uint32_t> host_intrinsic_modes(lane_count * kPtirIntrinsicSlots, 0);
    std::vector<std::uint32_t> host_intrinsic_widths(lane_count * kPtirIntrinsicSlots, 0);
    std::vector<std::uint32_t> host_intrinsic_strides(lane_count * kPtirIntrinsicSlots, 0);
    std::vector<std::uint32_t> host_intrinsic_offsets(lane_count * kPtirIntrinsicSlots, 0);
    std::vector<std::uint64_t> host_bf16_rows;
    std::vector<std::uint32_t> bf16_row_offsets(lane_count, UINT32_MAX);
    std::vector<std::uint64_t> host_mtp_rows;
    std::vector<std::uint32_t> mtp_row_offsets(lane_count, UINT32_MAX);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        if (lanes[lane].logits_bf16_rows != nullptr) {
            bf16_row_offsets[lane] =
                static_cast<std::uint32_t>(host_bf16_rows.size());
            host_bf16_rows.insert(
                host_bf16_rows.end(),
                lanes[lane].logits_bf16_rows->begin(),
                lanes[lane].logits_bf16_rows->end());
        }
        if (lanes[lane].mtp_logits_bf16_rows != nullptr) {
            mtp_row_offsets[lane] =
                static_cast<std::uint32_t>(host_mtp_rows.size());
            host_mtp_rows.insert(
                host_mtp_rows.end(),
                lanes[lane].mtp_logits_bf16_rows->begin(),
                lanes[lane].mtp_logits_bf16_rows->end());
        }
    }
    std::uint64_t* device_bf16_rows = nullptr;
    std::uint64_t* device_mtp_rows = nullptr;
    if (!host_bf16_rows.empty()) {
        device_bf16_rows =
            allocations.allocate<std::uint64_t>(host_bf16_rows.size());
        detail::upload_generated(
            device_bf16_rows, host_bf16_rows, stream);
    }
    if (!host_mtp_rows.empty()) {
        device_mtp_rows =
            allocations.allocate<std::uint64_t>(host_mtp_rows.size());
        detail::upload_generated(
            device_mtp_rows, host_mtp_rows, stream);
    }
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        if (bf16_row_offsets[lane] != UINT32_MAX) {
            host_lanes[lane].logits_base =
                reinterpret_cast<std::uint64_t>(
                    device_bf16_rows + bf16_row_offsets[lane]);
        }
    }
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const auto& op = stage.ops[node].op;
            if (op.tag != PTIR_OP_INTRINSIC_VAL) continue;
            const std::size_t index =
                static_cast<std::size_t>(lane) * kPtirIntrinsicSlots +
                op.intr;
            const GeneratedValueDesc& descriptor =
                host_descriptors[
                    static_cast<std::size_t>(lane) * value_count +
                    bases[node]];
            host_intrinsic_widths[index] =
                descriptor.last;
            if (op.intr == PTIR_INTR_LOGITS) {
                if (bf16_row_offsets[lane] != UINT32_MAX) {
                    host_intrinsic_bases[index] =
                        reinterpret_cast<std::uint64_t>(
                            device_bf16_rows + bf16_row_offsets[lane]);
                    host_intrinsic_modes[index] = 2;
                    host_intrinsic_strides[index] = lanes[lane].vocab;
                } else {
                    const std::uint32_t stride =
                        lanes[lane].logits_stride == 0
                            ? lanes[lane].vocab
                            : lanes[lane].logits_stride;
                    host_intrinsic_bases[index] =
                        reinterpret_cast<std::uint64_t>(
                            lanes[lane].logits_base +
                            static_cast<std::size_t>(
                                lanes[lane].logits_row_offset) *
                                stride);
                    host_intrinsic_strides[index] = stride;
                }
            } else if (op.intr == PTIR_INTR_MTP_LOGITS) {
                if (mtp_row_offsets[lane] == UINT32_MAX) {
                    throw std::runtime_error(
                        "generated fused MTP intrinsic has no row table");
                }
                host_intrinsic_bases[index] =
                    reinterpret_cast<std::uint64_t>(
                        device_mtp_rows + mtp_row_offsets[lane]);
                host_intrinsic_modes[index] = 2;
                host_intrinsic_strides[index] = lanes[lane].vocab;
            } else if (op.intr == PTIR_INTR_MTP_DRAFTS) {
                throw std::runtime_error(
                    "generated fused MtpDrafts token binding is unavailable");
            } else if (op.intr == PTIR_INTR_QUERY) {
                host_intrinsic_bases[index] =
                    reinterpret_cast<std::uint64_t>(
                        lanes[lane].query_base);
                host_intrinsic_strides[index] = descriptor.last;
            } else if (op.intr == PTIR_INTR_LAYER) {
                host_intrinsic_bases[index] =
                    reinterpret_cast<std::uint64_t>(
                        lanes[lane].layer_base);
                host_intrinsic_widths[index] = 1;
                host_intrinsic_strides[index] = 1;
            } else if (op.intr == PTIR_INTR_ATTN_SCORE) {
                if (lanes[lane].attn_score_base == nullptr) {
                    throw std::runtime_error(
                        "generated fused AttnScore intrinsic has no capture "
                        "buffer for this lane");
                }
                host_intrinsic_bases[index] =
                    reinterpret_cast<std::uint64_t>(
                        lanes[lane].attn_score_base);
                host_intrinsic_strides[index] = descriptor.last;
            } else {
                throw std::runtime_error(
                    "generated fused intrinsic is unavailable");
            }
        }
    }

    PtirLaneTableHeader header{
        PTIR_LANE_TABLE_ABI_VERSION,
        lane_count,
        channel_count,
        0,
    };
    std::uint8_t* device_scratch = nullptr;
    std::vector<std::uint64_t> host_value_pointers(
        static_cast<std::size_t>(lane_count) * value_count);
    struct Segment {
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };
    std::size_t metadata_bytes = 0;
    auto reserve_segment = [&](std::size_t count, std::size_t element) {
        Segment segment;
        if (count == 0) return segment;
        metadata_bytes = (metadata_bytes + 7) / 8 * 8;
        segment.offset = metadata_bytes;
        segment.bytes = count * element;
        metadata_bytes += segment.bytes;
        return segment;
    };
    const Segment header_segment =
        reserve_segment(1, sizeof(PtirLaneTableHeader));
    const Segment lanes_segment =
        reserve_segment(host_lanes.size(), sizeof(PtirLaneRecord));
    const Segment channels_segment =
        reserve_segment(
            host_channels.size(), sizeof(PtirLaneChannelSlot));
    const Segment readiness_segment =
        reserve_segment(
            host_readiness.size(), sizeof(GroupedReadinessLane));
    const Segment readiness_slots_segment =
        reserve_segment(
            readiness_slots.size(), sizeof(std::uint32_t));
    const Segment descriptors_segment =
        reserve_segment(
            host_descriptors.size(), sizeof(GeneratedValueDesc));
    const Segment params_segment =
        reserve_segment(host_params.size(), sizeof(GeneratedOpParams));
    const Segment offsets_segment =
        reserve_segment(host_offsets.size(), sizeof(std::uint32_t));
    const Segment value_pointers_segment =
        reserve_segment(
            host_value_pointers.size(), sizeof(std::uint64_t));
    const Segment pending_segment =
        reserve_segment(host_pending.size(), sizeof(std::uint8_t));
    const Segment intrinsic_bases_segment =
        reserve_segment(
            host_intrinsic_bases.size(), sizeof(std::uint64_t));
    const Segment intrinsic_modes_segment =
        reserve_segment(
            host_intrinsic_modes.size(), sizeof(std::uint32_t));
    const Segment intrinsic_widths_segment =
        reserve_segment(
            host_intrinsic_widths.size(), sizeof(std::uint32_t));
    const Segment intrinsic_strides_segment =
        reserve_segment(
            host_intrinsic_strides.size(), sizeof(std::uint32_t));
    const Segment intrinsic_offsets_segment =
        reserve_segment(
            host_intrinsic_offsets.size(), sizeof(std::uint32_t));
    const bool scalable_nucleus_stage = std::any_of(
        stage.fused.regions.begin(),
        stage.fused.regions.end(),
        [&](const plan::Region& region) {
            return region.library &&
                region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE &&
                !grouped_nucleus_library_supported(stage, region);
        });
    const std::size_t base_scratch_bytes =
        scratch_stride * lane_count;
    const std::size_t nucleus_workspace_offset =
        detail::align_generated(base_scratch_bytes);
    const std::size_t nucleus_workspace_bytes =
        scalable_nucleus_stage
            ? maximum_nucleus_value_length * lane_count * sizeof(float)
            : 0;
    if (nucleus_workspace_bytes >
        std::numeric_limits<std::size_t>::max() -
            nucleus_workspace_offset) {
        throw std::runtime_error(
            "generated nucleus workspace overflows size_t");
    }
    const std::size_t total_scratch_bytes =
        nucleus_workspace_offset + nucleus_workspace_bytes;
    if (timing) {
        const auto now = fire_timing::Clock::now();
        prepared->t_build_us =
            fire_timing::duration_us(section_mark, now);
        section_mark = now;
    }
    std::uint8_t* device_metadata = nullptr;
    detail::StableStageBuffers* stable = nullptr;
    if (buffer_mode == PreparedBufferMode::kStablePerStage) {
        // The prepared path: one stable device block per (stage id, stream),
        // sized on first use, grown (rarely) in place of a ring acquire. No
        // slot search, no cudaEventQuery, no lease.
        stable = &runtime.stable_stage_buffers(
            executable.runtime_id, stream, stable_slot);
        const auto blocks = stable->device_blocks(
            metadata_bytes, total_scratch_bytes, stream);
        device_metadata = blocks.first;
        device_scratch = blocks.second;
    } else {
        prepared->device_workspace = runtime.acquire_device_workspace(
            executable.runtime_id,
            stream,
            metadata_bytes,
            total_scratch_bytes);
        if (prepared->device_workspace.has_value()) {
            device_metadata = prepared->device_workspace->metadata();
            device_scratch = prepared->device_workspace->scratch();
        } else {
            device_metadata =
                allocations.allocate<std::uint8_t>(metadata_bytes);
            device_scratch =
                allocations.allocate<std::uint8_t>(
                    total_scratch_bytes);
        }
    }
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        for (std::uint32_t value = 0; value < value_count; ++value) {
            host_value_pointers[
                static_cast<std::size_t>(lane) * value_count + value] =
                reinterpret_cast<std::uint64_t>(
                    device_scratch +
                    static_cast<std::size_t>(lane) * scratch_stride +
                    host_offsets[value]);
        }
    }
    if (timing) {
        const auto now = fire_timing::Clock::now();
        prepared->t_workspace_us =
            fire_timing::duration_us(section_mark, now);
        section_mark = now;
    }
    std::uint8_t* host_metadata = nullptr;
    detail::PinnedUploadLease metadata_lease{};
    detail::PinnedUploadStaging* metadata_staging = nullptr;
    if (stable != nullptr) {
        host_metadata = stable->pinned(metadata_bytes);
    } else {
        metadata_staging =
            &runtime.metadata_staging(executable.runtime_id, stream);
        metadata_lease = metadata_staging->acquire(metadata_bytes);
        host_metadata = metadata_lease.data;
    }
    std::memset(host_metadata, 0, metadata_bytes);
    auto pack = [&](const Segment& segment,
                    const void* source) {
        if (segment.bytes != 0) {
            std::memcpy(
                host_metadata + segment.offset,
                source,
                segment.bytes);
        }
    };
    pack(header_segment, &header);
    pack(lanes_segment, host_lanes.data());
    pack(channels_segment, host_channels.data());
    pack(readiness_segment, host_readiness.data());
    pack(readiness_slots_segment, readiness_slots.data());
    pack(descriptors_segment, host_descriptors.data());
    pack(params_segment, host_params.data());
    pack(offsets_segment, host_offsets.data());
    pack(value_pointers_segment, host_value_pointers.data());
    // pending_flags re-initialization (stage6-plan.md, spike hazard (b)):
    // the region kernels MUTATE the pending-flag bytes inside the device
    // metadata block as puts commit, so a "stable metadata buffer" is not
    // stable across fires for this segment. The fix is structural: every
    // prepare rebuilds `host_pending` from the fire's own prior-put overlay
    // (above) and repacks + re-uploads the WHOLE block, pending flags
    // included. Rebuild-and-repack was chosen over patching from a pristine
    // template because the block is small (a few KiB), the rebuild shares
    // this one code path with the ring path (no second template to drift),
    // and a partial-segment upload would save nothing while splitting one
    // H2D copy into two. A future captured body therefore relies on prepare
    // running before EVERY replay — same contract as the channel cursors.
    pack(pending_segment, host_pending.data());
    pack(intrinsic_bases_segment, host_intrinsic_bases.data());
    pack(intrinsic_modes_segment, host_intrinsic_modes.data());
    pack(intrinsic_widths_segment, host_intrinsic_widths.data());
    pack(intrinsic_strides_segment, host_intrinsic_strides.data());
    pack(intrinsic_offsets_segment, host_intrinsic_offsets.data());
    CUDA_CHECK(cudaMemcpyAsync(
        device_metadata,
        host_metadata,
        metadata_bytes,
        cudaMemcpyHostToDevice,
        stream));
    if (stable != nullptr) {
        stable->record_upload(stream);
    } else {
        metadata_staging->record(metadata_lease.slot, stream);
    }
    if (timing) {
        const auto now = fire_timing::Clock::now();
        prepared->t_upload_us =
            fire_timing::duration_us(section_mark, now);
        section_mark = now;
    }
    // From here the body half's clock runs: everything below (region
    // resolution included) was part of the combined function's launch
    // section, and `launch_generated_stage` measures t_launch_us from this
    // mark so the section accounting is unchanged by the split.
    prepared->section_mark = section_mark;
    auto pointer = [&](const Segment& segment) -> std::uint8_t* {
        return segment.bytes == 0
            ? nullptr
            : device_metadata + segment.offset;
    };
    auto* device_header =
        reinterpret_cast<PtirLaneTableHeader*>(
            pointer(header_segment));
    auto* device_lanes =
        reinterpret_cast<PtirLaneRecord*>(
            pointer(lanes_segment));
    auto* device_channels =
        reinterpret_cast<PtirLaneChannelSlot*>(
            pointer(channels_segment));
    auto* device_value_pointers =
        reinterpret_cast<std::uint64_t*>(
            pointer(value_pointers_segment));
    prepared->device_header = device_header;
    prepared->device_lanes = device_lanes;
    prepared->device_channels = device_channels;
    prepared->device_readiness =
        reinterpret_cast<GroupedReadinessLane*>(
            pointer(readiness_segment));
    prepared->device_readiness_slots =
        reinterpret_cast<std::uint32_t*>(
            pointer(readiness_slots_segment));
    prepared->device_value_pointers = device_value_pointers;
    prepared->device_scratch = device_scratch;
    ChannelView& group_view = lanes.front().instance->view();
    prepared->ring_full = group_view.d_full();
    prepared->ring_head = group_view.d_head();
    prepared->ring_tail = group_view.d_tail();
    prepared->ring_cap1 = group_view.d_cap1();
    prepared->readiness_diagnose =
        std::getenv("PIE_DEBUG_PULL_VALIDATE") != nullptr ? 1u : 0u;

    prepared->arg_header =
        reinterpret_cast<CUdeviceptr>(device_header);
    prepared->arg_lanes =
        reinterpret_cast<CUdeviceptr>(device_lanes);
    prepared->arg_channels =
        reinterpret_cast<CUdeviceptr>(device_channels);
    prepared->arg_descriptors =
        reinterpret_cast<CUdeviceptr>(pointer(descriptors_segment));
    prepared->arg_params =
        reinterpret_cast<CUdeviceptr>(pointer(params_segment));
    prepared->arg_offsets =
        reinterpret_cast<CUdeviceptr>(pointer(offsets_segment));
    prepared->arg_scratch =
        reinterpret_cast<CUdeviceptr>(device_scratch);
    prepared->arg_value_count = value_count;
    prepared->arg_scratch_stride =
        static_cast<std::uint32_t>(scratch_stride);
    prepared->arg_temporary_offset = temporary_offset;
    prepared->arg_pending =
        reinterpret_cast<CUdeviceptr>(pointer(pending_segment));
    prepared->arg_intrinsic_bases =
        reinterpret_cast<CUdeviceptr>(pointer(intrinsic_bases_segment));
    prepared->arg_intrinsic_modes =
        reinterpret_cast<CUdeviceptr>(pointer(intrinsic_modes_segment));
    prepared->arg_intrinsic_widths =
        reinterpret_cast<CUdeviceptr>(pointer(intrinsic_widths_segment));
    prepared->arg_intrinsic_strides =
        reinterpret_cast<CUdeviceptr>(pointer(intrinsic_strides_segment));
    prepared->arg_intrinsic_offsets =
        reinterpret_cast<CUdeviceptr>(pointer(intrinsic_offsets_segment));
    prepared->generated_arguments = {
        &prepared->arg_header,
        &prepared->arg_lanes,
        &prepared->arg_channels,
        &prepared->arg_descriptors,
        &prepared->arg_params,
        &prepared->arg_offsets,
        &prepared->arg_scratch,
        &prepared->arg_value_count,
        &prepared->arg_scratch_stride,
        &prepared->arg_temporary_offset,
        &prepared->arg_pending,
        &prepared->arg_intrinsic_bases,
        &prepared->arg_intrinsic_modes,
        &prepared->arg_intrinsic_widths,
        &prepared->arg_intrinsic_strides,
        &prepared->arg_intrinsic_offsets,
    };

    const auto direct_logits_kind = [&](std::uint32_t value) {
        for (std::size_t depth = 0; depth < 2; ++depth) {
            bool found = false;
            for (std::size_t node = 0; node < stage.ops.size(); ++node) {
                const auto& op = stage.ops[node].op;
                if (value < bases[node] ||
                    value >= bases[node] + op.results) {
                    continue;
                }
                found = true;
                if (op.tag == PTIR_OP_INTRINSIC_VAL) {
                    if (op.intr == PTIR_INTR_LOGITS &&
                        std::all_of(
                            lanes.begin(), lanes.end(),
                            [](const auto& lane) {
                                return lane.logits_bf16_rows != nullptr;
                            })) {
                        return std::uint8_t{1};
                    }
                    if (op.intr == PTIR_INTR_MTP_LOGITS &&
                        std::all_of(
                            lanes.begin(), lanes.end(),
                            [](const auto& lane) {
                                return lane.mtp_logits_bf16_rows != nullptr;
                            })) {
                        return std::uint8_t{2};
                    }
                }
                if (op.tag == PTIR_OP_RESHAPE && !op.args.empty()) {
                    value = op.args[0];
                    break;
                }
                return std::uint8_t{0};
            }
            if (!found) break;
        }
        return std::uint8_t{0};
    };
    // Resolve every region into a host-work-free launch record. This is the
    // host half of what used to be the launch loop: shape maximization,
    // library support decisions, side-table builds and their uploads all
    // happen HERE; the body half only issues the recorded launches.
    prepared->regions.reserve(executable.regions.size());
    for (std::size_t region_index = 0;
         region_index < executable.regions.size();
         ++region_index) {
        const auto& planned_region =
            stage.fused.regions[region_index];
        if (!planned_region.library &&
            planned_region.nodes.size() == 1 &&
            absorbed_scale_nodes[planned_region.nodes[0]] != 0) {
            continue;
        }
        if (planned_region.library) {
            GroupedLaneBinding launch_shape = lanes.front();
            for (const auto& lane : lanes) {
                launch_shape.logits_row_count = std::max(
                    launch_shape.logits_row_count,
                    lane.logits_row_count);
                auto maximize = [](std::uint32_t& target,
                                   std::uint32_t candidate) {
                    if (candidate == kUnavailableGroupedExtent) return;
                    if (target == kUnavailableGroupedExtent ||
                        candidate > target) {
                        target = candidate;
                    }
                };
                maximize(launch_shape.row_count, lane.row_count);
                maximize(launch_shape.token_count, lane.token_count);
                maximize(launch_shape.kv_len, lane.kv_len);
                maximize(launch_shape.page_count, lane.page_count);
                maximize(launch_shape.query_len, lane.query_len);
                maximize(launch_shape.key_len, lane.key_len);
            }
            if (planned_region.library_op ==
                PTIR_LIBRARY_NUCLEUS_SAMPLE) {
                const bool scaled =
                    planned_region.inputs.size() == 5;
                const std::uint32_t logits =
                    planned_region.inputs[0];
                const std::uint32_t top_p =
                    planned_region.inputs[scaled ? 3 : 1];
                const std::uint32_t rng_state =
                    planned_region.inputs[scaled ? 4 : 2];
                const std::uint32_t output =
                    planned_region.outputs[0];
                const auto& logits_type = stage.value_types[logits];
                const std::uint32_t rows =
                    grouped_rows(logits_type, launch_shape);
                const std::uint32_t length =
                    static_cast<std::uint32_t>(
                        grouped_numel(logits_type, launch_shape) /
                        std::max(rows, 1u));
                GroupedNucleusLaunch launch{
                    logits,
                    top_p,
                    rng_state,
                    output,
                    UINT32_MAX,
                    rows,
                    length,
                    static_cast<std::uint32_t>(grouped_numel(
                        stage.value_types[top_p], launch_shape)),
                    direct_logits_kind(logits),
                };
                if (scaled) {
                    launch.logit_scale =
                        planned_region.inputs[1];
                    launch.logit_scale_numel =
                        static_cast<std::uint32_t>(grouped_numel(
                            stage.value_types[launch.logit_scale],
                            launch_shape));
                }
                PreparedRegionLaunch record;
                record.nucleus = launch;
                // Grid: lane factor at the lattice ceiling (idle lanes
                // guard-exit on `header->lane_count`); `rows` is the
                // program's row shape, already maximized over the fire's
                // lanes above — a data-sized dimension kept exact this
                // slice (stage6-plan.md increment 3 scopes the lattice to
                // the lane factor).
                record.nucleus_segments =
                    prepared->padded_lane_count * rows;
                if (grouped_nucleus_library_supported(
                        stage, planned_region)) {
                    record.kind = PreparedRegionLaunch::Kind::kNucleus;
                } else {
                    record.kind =
                        PreparedRegionLaunch::Kind::kNucleusScalable;
                    // Workspace stays sized by the LIVE lane count: the
                    // padded grid's idle blocks return before touching
                    // `partial_maxima`/`thread_masses`/`probabilities`, so
                    // padding the allocations would buy nothing. (These are
                    // per-fire allocations anyway — stable-address
                    // workspace is increment 2 territory.)
                    const std::uint32_t segments =
                        lane_count * rows;
                    const std::size_t items =
                        static_cast<std::size_t>(segments) * length;
                    if (items >
                        nucleus_workspace_bytes / sizeof(float)) {
                        throw std::runtime_error(
                            "generated nucleus workspace is undersized");
                    }
                    record.probabilities =
                        reinterpret_cast<float*>(
                            device_scratch +
                            nucleus_workspace_offset);
                    const std::size_t chunk_count =
                        static_cast<std::size_t>(segments) *
                        kFastNucleusChunks;
                    record.partial_maxima =
                        allocations.allocate<float>(
                            chunk_count *
                            (1 + kFastNucleusThreads));
                    record.thread_masses =
                        record.partial_maxima + chunk_count;
                }
                prepared->regions.push_back(record);
                continue;
            }
            const std::uint32_t node = planned_region.nodes.front();
            const auto& op = stage.ops[node].op;
            if (planned_region.library_op == PTIR_LIBRARY_SECOND_PARTY) {
                if (op.tag == PTIR_OP_SINK_CALL &&
                    op.name_idx < stage.names.size() &&
                    stage.names[op.name_idx] == "lora") {
                    // `lora` is CONFIGURATION, not computation, so its
                    // lowering here is deliberately nothing. A sink's effect
                    // lands where its consumer lives: the mask sink below is
                    // consumed by a device kernel (the layer's attention), so
                    // it lowers to a device write; the lora sink is consumed
                    // by the host-built launch configuration (the A/B channel
                    // addresses and SITES constant the model body's
                    // projection GEMMs read), which `Dispatch` resolved into
                    // the launch-owned lora table when it assembled this
                    // phase's bindings. By the time this region executes the
                    // sink has already taken effect; a device store here
                    // would have nowhere meaningful to land.
                    continue;
                }
                if (op.tag == PTIR_OP_SINK_CALL) {
                    // `second_party_region_supported` already proved this is
                    // `attn_page_mask` at `OnAttnProj` with a rank-1 argument.
                    // What is left is that the model published a destination
                    // for every lane -- a sink with nowhere to write is a
                    // program whose selection silently never happened, so it
                    // throws rather than becoming a no-op.
                    const auto mask_shape = grouped_row_shape(
                        stage.value_types[op.args[0]], lanes);
                    const std::uint32_t declared = mask_shape.max_columns;
                    std::vector<GroupedLanePageMaskDevice> host_dests(
                        lane_count);
                    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                        const auto& m = lanes[lane].page_mask;
                        if (m.out == nullptr) {
                            throw std::runtime_error(
                                "attn_page_mask lane has no resolved page "
                                "mask destination");
                        }
                        // The row is sized by the fire (widest request), the
                        // mask by the program, and neither bounds the other.
                        // Write the overlap: rows are pre-seeded to "keep", so
                        // a slot the program did not speak for keeps its page.
                        // That is the same direction the in-flight page pinning
                        // already errs in, and the only alternative -- evicting
                        // a page no policy examined -- is not recoverable.
                        const std::uint32_t writable =
                            m.pages < declared ? m.pages : declared;
                        if (writable == 0) {
                            throw std::runtime_error(
                                "attn_page_mask has no page to write; the "
                                "program's mask and the fire's page list do "
                                "not overlap");
                        }
                        host_dests[lane] =
                            GroupedLanePageMaskDevice{m.out, writable, 0u};
                    }
                    GroupedLanePageMaskDevice* device_dests = nullptr;
                    const std::size_t dest_bytes =
                        host_dests.size() * sizeof(GroupedLanePageMaskDevice);
                    if (stable != nullptr) {
                        // Stable side table (increment 4): the captured
                        // launch bakes this address, so it must be the one
                        // every later prepare re-uploads into. Upload from
                        // the (stack) host vector is safe: at prepare time
                        // no capture is active on this stream, and a
                        // pageable H2D async copy stages synchronously with
                        // respect to the host source.
                        device_dests =
                            reinterpret_cast<GroupedLanePageMaskDevice*>(
                                stable->side_block(dest_bytes, stream));
                    } else {
                        CUDA_CHECK(cudaMallocAsync(
                            reinterpret_cast<void**>(&device_dests),
                            dest_bytes, stream));
                        allocations.values.push_back(device_dests);
                    }
                    CUDA_CHECK(cudaMemcpyAsync(
                        device_dests, host_dests.data(), dest_bytes,
                        cudaMemcpyHostToDevice, stream));

                    const std::uint8_t mask_dtype =
                        stage.value_types[op.args[0]].dtype;
                    if (mask_dtype != PTIR_DT_F32 &&
                        mask_dtype != PTIR_DT_I32 &&
                        mask_dtype != PTIR_DT_U32 &&
                        mask_dtype != PTIR_DT_BOOL) {
                        throw std::runtime_error(
                            "attn_page_mask has an unsupported mask dtype");
                    }
                    PreparedRegionLaunch record;
                    record.kind = PreparedRegionLaunch::Kind::kPageMask;
                    record.device_dests = device_dests;
                    record.mask_dtype = mask_dtype;
                    record.value_a = op.args[0];
                    prepared->regions.push_back(record);
                    continue;
                }
                // `generated_stage_supported` already proved this is
                // `envelope_dot` at an attention stage with an f32 vector
                // result; what is left is that the caller actually resolved the
                // fire's KV geometry for every lane. A lane without it means
                // the attention observation was missing, which is a wiring bug,
                // not a data-dependent condition — so it throws.
                const auto out_shape = grouped_row_shape(
                    stage.value_types[bases[node]], lanes);
                const std::uint32_t max_pages = out_shape.max_columns;
                if (max_pages == 0) {
                    throw std::runtime_error(
                        "envelope_dot result has no page slots");
                }
                std::vector<GroupedLaneEnvelopeDevice> host_envelopes(
                    lane_count);
                for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                    const auto& e = lanes[lane].envelope;
                    if (e.env_min == nullptr || e.env_max == nullptr ||
                        e.query == nullptr || e.page_ids == nullptr ||
                        e.page_indptr == nullptr ||
                        e.last_page_lens == nullptr || e.page_size == 0 ||
                        e.num_kv_heads == 0 || e.head_dim == 0 ||
                        e.num_q_heads % e.num_kv_heads != 0) {
                        throw std::runtime_error(
                            "envelope_dot lane has no resolved kv envelope");
                    }
                    // The host page CSR is an upper bound on the device one, so
                    // a result row that fits the bound fits the truth. The
                    // kernel writes -inf to the surplus slots.
                    if (e.page_bound > max_pages) {
                        throw std::runtime_error(
                            "envelope_dot result is narrower than the lane's "
                            "page list");
                    }
                    host_envelopes[lane] = GroupedLaneEnvelopeDevice{
                        e.env_min,        e.env_max,      e.query,
                        e.page_ids,       e.page_indptr,  e.last_page_lens,
                        e.request,        e.qo_len,       e.page_size,
                        e.num_q_heads,    e.num_kv_heads, e.head_dim};
                }
                GroupedLaneEnvelopeDevice* device_envelopes = nullptr;
                const std::size_t envelope_bytes =
                    host_envelopes.size() * sizeof(GroupedLaneEnvelopeDevice);
                CUDA_CHECK(cudaMallocAsync(
                    reinterpret_cast<void**>(&device_envelopes),
                    envelope_bytes, stream));
                allocations.values.push_back(device_envelopes);
                CUDA_CHECK(cudaMemcpyAsync(
                    device_envelopes, host_envelopes.data(), envelope_bytes,
                    cudaMemcpyHostToDevice, stream));
                PreparedRegionLaunch record;
                record.kind = PreparedRegionLaunch::Kind::kEnvelopeDot;
                record.device_envelopes = device_envelopes;
                record.max_pages = max_pages;
                record.out_a = bases[node];
                prepared->regions.push_back(record);
                continue;
            }
            if (planned_region.library_op == PTIR_LIBRARY_SCAN) {
                const auto shape = grouped_row_shape(
                    stage.value_types[op.args[0]], lanes);
                const std::uint8_t dtype =
                    stage.value_types[op.args[0]].dtype;
                if (dtype != PTIR_DT_F32 &&
                    dtype != PTIR_DT_I32 &&
                    dtype != PTIR_DT_U32) {
                    throw std::runtime_error(
                        "generated scan library has an invalid dtype");
                }
                PreparedRegionLaunch record;
                record.kind = PreparedRegionLaunch::Kind::kScan;
                record.scan_f32 = dtype == PTIR_DT_F32;
                record.scan_product = op.tag == PTIR_OP_CUMPROD;
                // Lane factor at the lattice ceiling; `max_rows` is the
                // value's data-sized row bound (max over lanes), kept exact
                // this slice. Idle blocks guard-exit on
                // `header->lane_count`.
                record.scan_blocks =
                    prepared->padded_lane_count * shape.max_rows;
                record.shape_a = shape;
                record.value_a = op.args[0];
                record.out_a = bases[node];
                prepared->regions.push_back(record);
                continue;
            }
            if (planned_region.library_op == PTIR_LIBRARY_MATMUL) {
                const auto& left_type =
                    stage.value_types[op.args[0]];
                const auto& right_type =
                    stage.value_types[op.args[1]];
                if (left_type.dtype != PTIR_DT_F32 ||
                    right_type.dtype != PTIR_DT_F32 ||
                    left_type.dims.size() != 2 ||
                    right_type.dims.size() != 2) {
                    throw std::runtime_error(
                        "generated matmul library has an invalid type");
                }
                const auto left_shape =
                    grouped_row_shape(left_type, lanes);
                const auto right_shape =
                    grouped_row_shape(right_type, lanes);
                // The matmul kernel is grid-stride: its true bound is
                // `header->lane_count * per_lane`, computed on the device,
                // so the grid is only a parallelism cap. Padding the lane
                // factor keeps the cap a function of the lattice bucket
                // (surplus blocks find `flat >= total` and fall through);
                // the row/column factors are data-sized and stay exact
                // this slice.
                const std::uint64_t total =
                    static_cast<std::uint64_t>(
                        prepared->padded_lane_count) *
                    left_shape.max_rows * right_shape.max_columns;
                PreparedRegionLaunch record;
                record.kind = PreparedRegionLaunch::Kind::kMatmul;
                record.matmul_grid = grouped_grid(total);
                record.shape_a = left_shape;
                record.shape_b = right_shape;
                record.value_a = op.args[0];
                record.value_b = op.args[1];
                record.out_a = bases[node];
                prepared->regions.push_back(record);
                continue;
            }
            if (planned_region.library_op == PTIR_LIBRARY_TOP_K ||
                planned_region.library_op == PTIR_LIBRARY_SORT) {
                const auto& input_type =
                    stage.value_types[op.args[0]];
                const auto input_row_shape =
                    grouped_row_shape(input_type, lanes);
                const std::uint32_t rows =
                    planned_region.library_op == PTIR_LIBRARY_SORT
                    ? 1
                    : input_row_shape.max_rows;
                const std::uint32_t length =
                    planned_region.library_op == PTIR_LIBRARY_SORT
                    ? static_cast<std::uint32_t>(
                          grouped_dynamic_shape(
                              input_type, lanes)
                              .max_numel)
                    : input_row_shape.max_columns;
                const std::uint32_t k =
                    planned_region.library_op == PTIR_LIBRARY_SORT
                    ? length
                    : op.imm;
                const auto input_shape =
                    grouped_dynamic_shape(input_type, lanes);
                PreparedRegionLaunch record;
                record.kind = PreparedRegionLaunch::Kind::kTopK;
                record.topk_rows = rows;
                record.topk_length = length;
                record.topk_k = k;
                record.topk_is_sort =
                    planned_region.library_op == PTIR_LIBRARY_SORT;
                record.dynamic_shape = input_shape;
                record.shape_a = input_row_shape;
                record.topk_vocab = launch_shape.vocab;
                record.value_a = op.args[0];
                record.out_a = bases[node];
                record.out_b = bases[node] + 1;
                prepared->regions.push_back(record);
                continue;
            }
            throw std::runtime_error(
                "registered generated library has no CUDA launcher");
        }
        PreparedRegionLaunch record;
        record.kind = PreparedRegionLaunch::Kind::kGenerated;
        record.region = executable.regions[region_index].get();
        prepared->regions.push_back(record);
    }
    return prepared;
}

// The body half: kernel launches only, reading nothing but prepared state.
// No host metadata, no allocation, no event wait, no channel-registry read
// lives here — this is the code a later slice captures into a CUDA graph
// (batch/forward_graph.hpp:14-24 is the doctrine being applied).
inline GroupedLaunchResult launch_generated_stage(
    GeneratedStagePrepared& prepared,
    // Optional launch-stream override (hook-graph mode, stage 6 increment
    // 4): the prepare pass runs on the fire's submission stream, but the
    // body's launches must land on the stream the model body hands the hook
    // — under capture that is the capture stream. The prepared state itself
    // is stream-agnostic device data; only the launches move.
    cudaStream_t stream_override = nullptr) {
    cudaStream_t stream =
        stream_override != nullptr ? stream_override : prepared.stream;
    // Every lane-derived grid below is sized by the lattice bucket, never
    // the live count (see `generated_lane_grid_bucket`); the live count is
    // device data (`prepared.device_header->lane_count`) and idle blocks
    // guard-exit first thing.
    const std::uint32_t grid_lanes = prepared.padded_lane_count;
    const std::uint32_t value_count = prepared.value_count;
    // Readiness under padding (stage6-plan.md increment 3, scope 3): the
    // grid is a function of the bucket alone, so a captured graph's
    // readiness node is dimension-stable across fires in the bucket. Its
    // per-thread guard (`lane >= header->lane_count`,
    // grouped_runtime.cuh:339) already made the surplus threads inert —
    // this kernel always over-launched to the 128 ceiling. What padding
    // does NOT fix is the spike caveat that the kernel evaluates live ring
    // head/tail state: that stays correct while prepare re-runs before
    // every launch (this slice is eager), and device-resolving it for
    // replay is deliberately deferred to the capture slice.
    k_grouped_stage_readiness<<<
        (grid_lanes + 127) / 128, 128, 0, stream>>>(
        prepared.device_header,
        prepared.device_lanes,
        prepared.device_readiness,
        prepared.device_readiness_slots,
        prepared.ring_full,
        prepared.ring_head,
        prepared.ring_tail,
        prepared.ring_cap1,
        prepared.readiness_diagnose);
    CUDA_CHECK(cudaGetLastError());

    GroupedLaunchResult result;
    for (const PreparedRegionLaunch& region : prepared.regions) {
        switch (region.kind) {
            case PreparedRegionLaunch::Kind::kNucleus: {
                k_grouped_nucleus_sample<<<
                    region.nucleus_segments,
                    kCanonicalReduceWidth,
                    0,
                    stream>>>(
                    prepared.device_header,
                    prepared.device_lanes,
                    prepared.device_channels,
                    nullptr,
                    nullptr,
                    prepared.device_value_pointers,
                    value_count,
                    region.nucleus);
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
                result.used_nucleus_library = true;
                break;
            }
            case PreparedRegionLaunch::Kind::kNucleusScalable: {
                const std::uint32_t segments = region.nucleus_segments;
                k_grouped_nucleus_max_chunks<<<
                    segments * kFastNucleusChunks,
                    kFastNucleusThreads,
                    0,
                    stream>>>(
                    prepared.device_header,
                    prepared.device_lanes,
                    nullptr,
                    nullptr,
                    prepared.device_value_pointers,
                    value_count,
                    region.partial_maxima,
                    region.nucleus);
                CUDA_CHECK(cudaGetLastError());
                k_grouped_nucleus_weight_chunks<<<
                    segments * kFastNucleusChunks,
                    kFastNucleusThreads,
                    0,
                    stream>>>(
                    prepared.device_header,
                    prepared.device_lanes,
                    nullptr,
                    nullptr,
                    prepared.device_value_pointers,
                    value_count,
                    region.partial_maxima,
                    region.probabilities,
                    region.thread_masses,
                    region.nucleus);
                CUDA_CHECK(cudaGetLastError());
                k_grouped_nucleus_inverse_sample<<<
                    segments,
                    kFastNucleusThreads,
                    0,
                    stream>>>(
                    prepared.device_header,
                    prepared.device_lanes,
                    prepared.device_channels,
                    prepared.device_value_pointers,
                    value_count,
                    region.probabilities,
                    region.thread_masses,
                    region.nucleus);
                CUDA_CHECK(cudaGetLastError());
                result.body_op_launches += 3;
                result.large_nucleus_scalable = true;
                result.used_nucleus_library = true;
                break;
            }
            case PreparedRegionLaunch::Kind::kPageMask: {
                // One block per PADDED lane; the per-lane dest table is
                // live-sized, but the kernel guards `lane >=
                // header->lane_count` before reading `dests[lane]`.
                if (region.mask_dtype == PTIR_DT_F32) {
                    k_generated_attn_page_mask<float><<<
                        grid_lanes, kTier0Block, 0, stream>>>(
                        prepared.device_header, prepared.device_lanes,
                        region.device_dests,
                        prepared.device_value_pointers, value_count,
                        region.value_a);
                } else if (region.mask_dtype == PTIR_DT_I32) {
                    k_generated_attn_page_mask<std::int32_t><<<
                        grid_lanes, kTier0Block, 0, stream>>>(
                        prepared.device_header, prepared.device_lanes,
                        region.device_dests,
                        prepared.device_value_pointers, value_count,
                        region.value_a);
                } else if (region.mask_dtype == PTIR_DT_U32) {
                    k_generated_attn_page_mask<std::uint32_t><<<
                        grid_lanes, kTier0Block, 0, stream>>>(
                        prepared.device_header, prepared.device_lanes,
                        region.device_dests,
                        prepared.device_value_pointers, value_count,
                        region.value_a);
                } else {
                    // A predicate is what a policy naturally produces
                    // (`ge(scores, threshold)`), and bool values are stored
                    // one byte per element here -- so this is the narrow
                    // instantiation, not a reinterpretation of the u32 one.
                    // (Prepare already refused every other dtype.)
                    k_generated_attn_page_mask<std::uint8_t><<<
                        grid_lanes, kTier0Block, 0, stream>>>(
                        prepared.device_header, prepared.device_lanes,
                        region.device_dests,
                        prepared.device_value_pointers, value_count,
                        region.value_a);
                }
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
                break;
            }
            case PreparedRegionLaunch::Kind::kEnvelopeDot: {
                // Lane factor padded; `max_pages` is the program-declared
                // page-slot bound (data-sized), kept exact this slice —
                // padding it would multiply idle blocks by the page count
                // for a dimension the future capture key can carry
                // exactly. Idle lanes guard-exit before reading
                // `envelopes[lane]` (live-sized table).
                k_generated_envelope_dot<kTier0Block><<<
                    grid_lanes * region.max_pages,
                    kTier0Block,
                    0,
                    stream>>>(
                    prepared.device_header,
                    prepared.device_lanes,
                    region.device_envelopes,
                    prepared.device_value_pointers,
                    value_count,
                    region.out_a,
                    region.max_pages);
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
                break;
            }
            case PreparedRegionLaunch::Kind::kScan: {
                if (region.scan_f32) {
                    k_generated_scan_f32<<<
                        region.scan_blocks, 1, 0, stream>>>(
                        prepared.device_header,
                        prepared.device_lanes,
                        prepared.device_value_pointers,
                        value_count,
                        region.value_a,
                        region.out_a,
                        region.shape_a,
                        region.scan_product);
                } else {
                    k_generated_scan_u32<<<
                        region.scan_blocks, 1, 0, stream>>>(
                        prepared.device_header,
                        prepared.device_lanes,
                        prepared.device_value_pointers,
                        value_count,
                        region.value_a,
                        region.out_a,
                        region.shape_a,
                        region.scan_product);
                }
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
                break;
            }
            case PreparedRegionLaunch::Kind::kMatmul: {
                k_generated_matmul_f32<<<
                    region.matmul_grid,
                    kTier0Block,
                    0,
                    stream>>>(
                    prepared.device_header,
                    prepared.device_lanes,
                    prepared.device_value_pointers,
                    value_count,
                    region.value_a,
                    region.value_b,
                    region.out_a,
                    region.shape_a,
                    region.shape_b);
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
                break;
            }
            case PreparedRegionLaunch::Kind::kTopK: {
                // Lane factor padded; `topk_rows` is the program's row
                // bound (data-sized), kept exact this slice. Idle lanes
                // guard-exit (`k_grouped_topk` reads `header->lane_count`
                // before any lane table).
                k_grouped_topk<<<
                    grid_lanes * region.topk_rows,
                    kTier0Block,
                    0,
                    stream>>>(
                    prepared.device_header,
                    prepared.device_lanes,
                    nullptr,
                    nullptr,
                    prepared.device_value_pointers,
                    value_count,
                    region.value_a,
                    region.out_a,
                    region.out_b,
                    region.topk_rows,
                    region.topk_length,
                    region.topk_k,
                    region.topk_is_sort,
                    region.dynamic_shape,
                    region.shape_a,
                    region.topk_vocab,
                    0);
                CUDA_CHECK(cudaGetLastError());
                result.used_selection_library = true;
                ++result.body_op_launches;
                break;
            }
            case PreparedRegionLaunch::Kind::kGenerated: {
                // One block per PADDED lane. The NVRTC-emitted region
                // kernel's first act is the guard
                // (`dispatch_lane >= header->lane_count`,
                // compiler/codegen/runtime/cuda/fused_block1.cuh:18-19),
                // so idle blocks exit before touching any lane table.
                const CUresult launch_status = cuLaunchKernel(
                    region.region->function,
                    grid_lanes, 1, 1,
                    static_cast<unsigned>(
                        region.region->block_threads), 1, 1,
                    0,
                    stream,
                    prepared.generated_arguments.data(),
                    nullptr);
                if (launch_status != CUDA_SUCCESS) {
                    const char* message = nullptr;
                    cuGetErrorString(launch_status, &message);
                    throw std::runtime_error(
                        "generated fused launch failed: " +
                        std::string(
                            message == nullptr
                                ? "unknown CUDA driver error"
                                : message));
                }
                ++result.body_op_launches;
                break;
            }
        }
    }
    if (prepared.device_workspace.has_value()) {
        prepared.device_workspace->record();
    }
    if (prepared.timing) {
        result.t_build_us = prepared.t_build_us;
        result.t_workspace_us = prepared.t_workspace_us;
        result.t_upload_us = prepared.t_upload_us;
        result.t_launch_us = fire_timing::duration_us(
            prepared.section_mark, fire_timing::Clock::now());
    }
    return result;
}

// The ring-backed combined path: prepare + launch back-to-back on the
// rotating rings — byte-for-byte the pre-split behavior. Prologue/Epilogue
// ONLY: since the eager-path unification, every attention-phase execution
// (eager or graph, fire-level or veto-fallback) goes through
// `prepare_generated_stage`(kStablePerStage) + `launch_generated_stage`
// instead; nothing else may take this wrapper.
// TODO(stage6-plan.md increment 1): migrate the remaining phases onto the
// prepared kStablePerStage path and retire the rotating rings entirely.
inline GroupedLaunchResult run_generated_stage_ring(
    const std::vector<GroupedLaneBinding>& lanes,
    const FusedStageExecutable& executable,
    GeneratedRuntimeContext& runtime,
    cudaStream_t stream,
    GroupedExecutionOptions options = {}) {
    const std::unique_ptr<GeneratedStagePrepared> prepared =
        prepare_generated_stage(
            lanes,
            executable,
            runtime,
            stream,
            options,
            PreparedBufferMode::kRotatingRing);
    return launch_generated_stage(*prepared);
}


// ── Merge stub (2026-08-04): upstream's fused LM-head argmax for the
// generated path is NOT taken this era (tart's stage6 machinery owns
// this file; the upstream feature returns with its re-port). The
// eligibility probe upstream call sites compiled against always says
// no, so the epilogue keeps materialising logits.
template <class StageT, class ExecT>
inline bool generated_stage_is_compact_argmax(const StageT&, const ExecT&) {
    return false;
}

}  // namespace pie_cuda_driver::pipeline::generated

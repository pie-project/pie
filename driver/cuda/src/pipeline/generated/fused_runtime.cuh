#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "cuda_check.hpp"
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
    static constexpr std::size_t kMaximumEntries = 1024;
    static constexpr std::size_t kMaximumRetainedDeviceBytes =
        512ull * 1024ull * 1024ull;

    struct Key {
        std::uint64_t runtime_id = 0;
        cudaStream_t stream = nullptr;
        bool operator==(const Key&) const = default;
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            const std::size_t runtime =
                std::hash<std::uint64_t>{}(key.runtime_id);
            const std::size_t stream =
                std::hash<const void*>{}(key.stream);
            return runtime ^ (stream + 0x9e3779b9u +
                              (runtime << 6) + (runtime >> 2));
        }
    };

    struct Entry {
        std::unique_ptr<detail::PinnedUploadStaging> staging;
        detail::PersistentDeviceWorkspace workspace;
        std::uint64_t last_use = 0;
    };

    Entry& entry(
        std::uint64_t runtime_id,
        cudaStream_t stream) {
        if (runtime_id == 0) {
            throw std::runtime_error(
                "generated executable has no runtime identity");
        }
        const Key key{runtime_id, stream};
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
        executable.signature != stage.signature ||
        executable.regions.size() != stage.fused.regions.size()) {
        return fail("compiled fused stage identity mismatch");
    }
    for (std::size_t index = 0;
         index < stage.fused.regions.size();
         ++index) {
        const auto& region = stage.fused.regions[index];
        if (region.library) {
            if (region.library_op == PTIR_LIBRARY_SECOND_PARTY) {
                return fail(
                    "fused stage still requires a second-party library");
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
    const std::vector<std::uint32_t>& bases) {
    if (stage.fused.regions.size() != 1 ||
        stage.fused.regions.front().library) {
        return std::nullopt;
    }
    const DirectArgmaxAnalysis direct = analyze_direct_argmax(
        stage, stage.fused.regions.front(), bases);
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
            if (argmax_value != UINT32_MAX ||
                resolve_alias(op.args[0]) != intrinsic_value ||
                direct.intrinsic[node] != PTIR_INTR_LOGITS) {
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

inline GroupedLaunchResult run_generated_stage(
    const std::vector<GroupedLaneBinding>& lanes,
    const FusedStageExecutable& executable,
    GeneratedRuntimeContext& runtime,
    cudaStream_t stream,
    GroupedExecutionOptions options = {}) {
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
    detail::AsyncAllocations allocations(stream);

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
        if (direct_logits) {
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
            absorbed_scale_nodes[node] = 1;
            maximum_value_bytes[scaled_value] = 4;
            temporary_elided[scaled_value] = 1;
            if (op.tag == PTIR_OP_DIV && !op.args.empty()) {
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
    for (const auto& region : stage.fused.regions) {
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
        const DirectArgmaxAnalysis direct =
            analyze_direct_argmax(stage, region, bases);
        for (const std::uint32_t node : region.nodes) {
            if (direct.requires_single_row[node] != 0) {
                const std::uint32_t source =
                    direct.source_value[node];
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
            if (direct.skipped[node] == 0 && !internal_reshape) {
                continue;
            }
            for (std::uint32_t result = 0; result < op.results; ++result) {
                maximum_value_bytes[bases[node] + result] = 4;
                if (direct.skipped[node] != 0) {
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
        generated_compact_argmax_value(stage, bases);
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
    std::vector<std::uint64_t> host_intrinsic_bases(lane_count * 7, 0);
    std::vector<std::uint32_t> host_intrinsic_modes(lane_count * 7, 0);
    std::vector<std::uint32_t> host_intrinsic_widths(lane_count * 7, 0);
    std::vector<std::uint32_t> host_intrinsic_strides(lane_count * 7, 0);
    std::vector<std::uint32_t> host_intrinsic_offsets(lane_count * 7, 0);
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
                static_cast<std::size_t>(lane) * 7 + op.intr;
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
    std::int64_t t_build_us = -1;
    if (timing) {
        const auto now = fire_timing::Clock::now();
        t_build_us = fire_timing::duration_us(section_mark, now);
        section_mark = now;
    }
    auto device_workspace = runtime.acquire_device_workspace(
        executable.runtime_id,
        stream,
        metadata_bytes,
        total_scratch_bytes);
    std::uint8_t* device_metadata = nullptr;
    if (device_workspace.has_value()) {
        device_metadata = device_workspace->metadata();
        device_scratch = device_workspace->scratch();
    } else {
        device_metadata =
            allocations.allocate<std::uint8_t>(metadata_bytes);
        device_scratch =
            allocations.allocate<std::uint8_t>(
                total_scratch_bytes);
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
    std::int64_t t_workspace_us = -1;
    if (timing) {
        const auto now = fire_timing::Clock::now();
        t_workspace_us = fire_timing::duration_us(section_mark, now);
        section_mark = now;
    }
    auto& metadata_staging =
        runtime.metadata_staging(executable.runtime_id, stream);
    const detail::PinnedUploadLease metadata_lease =
        metadata_staging.acquire(metadata_bytes);
    std::uint8_t* host_metadata = metadata_lease.data;
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
    metadata_staging.record(metadata_lease.slot, stream);
    std::int64_t t_upload_us = -1;
    if (timing) {
        const auto now = fire_timing::Clock::now();
        t_upload_us = fire_timing::duration_us(section_mark, now);
        section_mark = now;
    }
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
    auto* device_readiness =
        reinterpret_cast<GroupedReadinessLane*>(
            pointer(readiness_segment));
    auto* device_readiness_slots =
        reinterpret_cast<std::uint32_t*>(
            pointer(readiness_slots_segment));
    auto* device_descriptors =
        reinterpret_cast<GeneratedValueDesc*>(
            pointer(descriptors_segment));
    auto* device_params =
        reinterpret_cast<GeneratedOpParams*>(
            pointer(params_segment));
    auto* device_offsets =
        reinterpret_cast<std::uint32_t*>(
            pointer(offsets_segment));
    auto* device_value_pointers =
        reinterpret_cast<std::uint64_t*>(
            pointer(value_pointers_segment));
    auto* device_pending =
        pointer(pending_segment);
    auto* device_intrinsic_bases =
        reinterpret_cast<std::uint64_t*>(
            pointer(intrinsic_bases_segment));
    auto* device_intrinsic_modes =
        reinterpret_cast<std::uint32_t*>(
            pointer(intrinsic_modes_segment));
    auto* device_intrinsic_widths =
        reinterpret_cast<std::uint32_t*>(
            pointer(intrinsic_widths_segment));
    auto* device_intrinsic_strides =
        reinterpret_cast<std::uint32_t*>(
            pointer(intrinsic_strides_segment));
    auto* device_intrinsic_offsets =
        reinterpret_cast<std::uint32_t*>(
            pointer(intrinsic_offsets_segment));
    ChannelView& group_view = lanes.front().instance->view();
    k_grouped_stage_readiness<<<
        (lane_count + 127) / 128, 128, 0, stream>>>(
        device_header,
        device_lanes,
        device_readiness,
        device_readiness_slots,
        group_view.d_full(),
        group_view.d_head(),
        group_view.d_tail(),
        group_view.d_cap1());
    CUDA_CHECK(cudaGetLastError());

    GroupedLaunchResult result;
    std::uint32_t mutable_value_count = value_count;
    std::uint32_t mutable_scratch_stride =
        static_cast<std::uint32_t>(scratch_stride);
    std::uint32_t mutable_temporary_offset = temporary_offset;
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
                if (grouped_nucleus_library_supported(
                        stage, planned_region)) {
                    k_grouped_nucleus_sample<<<
                        lane_count * rows,
                        kCanonicalReduceWidth,
                        0,
                        stream>>>(
                        device_header,
                        device_lanes,
                        device_channels,
                        nullptr,
                        nullptr,
                        device_value_pointers,
                        value_count,
                        launch);
                    CUDA_CHECK(cudaGetLastError());
                    ++result.body_op_launches;
                } else {
                    const std::uint32_t segments =
                        lane_count * rows;
                    const std::size_t items =
                        static_cast<std::size_t>(segments) * length;
                    if (items >
                        nucleus_workspace_bytes / sizeof(float)) {
                        throw std::runtime_error(
                            "generated nucleus workspace is undersized");
                    }
                    float* probabilities =
                        reinterpret_cast<float*>(
                            device_scratch +
                            nucleus_workspace_offset);
                    const std::size_t chunk_count =
                        static_cast<std::size_t>(segments) *
                        kFastNucleusChunks;
                    float* partial_maxima =
                        allocations.allocate<float>(
                            chunk_count *
                            (1 + kFastNucleusThreads));
                    float* thread_masses =
                        partial_maxima + chunk_count;
                    k_grouped_nucleus_max_chunks<<<
                        segments * kFastNucleusChunks,
                        kFastNucleusThreads,
                        0,
                        stream>>>(
                        device_header,
                        device_lanes,
                        nullptr,
                        nullptr,
                        device_value_pointers,
                        value_count,
                        partial_maxima,
                        launch);
                    CUDA_CHECK(cudaGetLastError());
                    k_grouped_nucleus_weight_chunks<<<
                        segments * kFastNucleusChunks,
                        kFastNucleusThreads,
                        0,
                        stream>>>(
                        device_header,
                        device_lanes,
                        nullptr,
                        nullptr,
                        device_value_pointers,
                        value_count,
                        partial_maxima,
                        probabilities,
                        thread_masses,
                        launch);
                    CUDA_CHECK(cudaGetLastError());
                    k_grouped_nucleus_inverse_sample<<<
                        segments,
                        kFastNucleusThreads,
                        0,
                        stream>>>(
                        device_header,
                        device_lanes,
                        device_channels,
                        device_value_pointers,
                        value_count,
                        probabilities,
                        thread_masses,
                        launch);
                    CUDA_CHECK(cudaGetLastError());
                    result.body_op_launches += 3;
                    result.large_nucleus_scalable = true;
                }
                result.used_nucleus_library = true;
                continue;
            }
            const std::uint32_t node = planned_region.nodes.front();
            const auto& op = stage.ops[node].op;
            if (planned_region.library_op == PTIR_LIBRARY_SCAN) {
                const auto shape = grouped_row_shape(
                    stage.value_types[op.args[0]], lanes);
                const std::uint32_t blocks =
                    lane_count * shape.max_rows;
                const bool product = op.tag == PTIR_OP_CUMPROD;
                const std::uint8_t dtype =
                    stage.value_types[op.args[0]].dtype;
                if (dtype == PTIR_DT_F32) {
                    k_generated_scan_f32<<<
                        blocks, 1, 0, stream>>>(
                        device_header,
                        device_lanes,
                        device_value_pointers,
                        value_count,
                        op.args[0],
                        bases[node],
                        shape,
                        product);
                } else if (
                    dtype == PTIR_DT_I32 ||
                    dtype == PTIR_DT_U32) {
                    k_generated_scan_u32<<<
                        blocks, 1, 0, stream>>>(
                        device_header,
                        device_lanes,
                        device_value_pointers,
                        value_count,
                        op.args[0],
                        bases[node],
                        shape,
                        product);
                } else {
                    throw std::runtime_error(
                        "generated scan library has an invalid dtype");
                }
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
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
                const std::uint64_t total =
                    static_cast<std::uint64_t>(lane_count) *
                    left_shape.max_rows * right_shape.max_columns;
                k_generated_matmul_f32<<<
                    grouped_grid(total),
                    kTier0Block,
                    0,
                    stream>>>(
                    device_header,
                    device_lanes,
                    device_value_pointers,
                    value_count,
                    op.args[0],
                    op.args[1],
                    bases[node],
                    left_shape,
                    right_shape);
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
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
                k_grouped_topk<<<
                    lane_count * rows,
                    kTier0Block,
                    0,
                    stream>>>(
                    device_header,
                    device_lanes,
                    nullptr,
                    nullptr,
                    device_value_pointers,
                    value_count,
                    op.args[0],
                    bases[node],
                    bases[node] + 1,
                    rows,
                    length,
                    k,
                    planned_region.library_op == PTIR_LIBRARY_SORT,
                    input_shape,
                    input_row_shape,
                    launch_shape.vocab,
                    0);
                CUDA_CHECK(cudaGetLastError());
                result.used_selection_library = true;
                ++result.body_op_launches;
                continue;
            }
            throw std::runtime_error(
                "registered generated library has no CUDA launcher");
        }
        const auto& region = executable.regions[region_index];
        CUdeviceptr header_pointer =
            reinterpret_cast<CUdeviceptr>(device_header);
        CUdeviceptr lanes_pointer =
            reinterpret_cast<CUdeviceptr>(device_lanes);
        CUdeviceptr channels_pointer =
            reinterpret_cast<CUdeviceptr>(device_channels);
        CUdeviceptr descriptors_pointer =
            reinterpret_cast<CUdeviceptr>(device_descriptors);
        CUdeviceptr params_pointer =
            reinterpret_cast<CUdeviceptr>(device_params);
        CUdeviceptr offsets_pointer =
            reinterpret_cast<CUdeviceptr>(device_offsets);
        CUdeviceptr scratch_pointer =
            reinterpret_cast<CUdeviceptr>(device_scratch);
        CUdeviceptr pending_pointer =
            reinterpret_cast<CUdeviceptr>(device_pending);
        CUdeviceptr bases_pointer =
            reinterpret_cast<CUdeviceptr>(device_intrinsic_bases);
        CUdeviceptr modes_pointer =
            reinterpret_cast<CUdeviceptr>(device_intrinsic_modes);
        CUdeviceptr widths_pointer =
            reinterpret_cast<CUdeviceptr>(device_intrinsic_widths);
        CUdeviceptr strides_pointer =
            reinterpret_cast<CUdeviceptr>(device_intrinsic_strides);
        CUdeviceptr intrinsic_offsets_pointer =
            reinterpret_cast<CUdeviceptr>(device_intrinsic_offsets);
        void* arguments[] = {
            &header_pointer,
            &lanes_pointer,
            &channels_pointer,
            &descriptors_pointer,
            &params_pointer,
            &offsets_pointer,
            &scratch_pointer,
            &mutable_value_count,
            &mutable_scratch_stride,
            &mutable_temporary_offset,
            &pending_pointer,
            &bases_pointer,
            &modes_pointer,
            &widths_pointer,
            &strides_pointer,
            &intrinsic_offsets_pointer,
        };
        CUresult launch_status = cuLaunchKernel(
            region->function,
            lane_count, 1, 1,
            static_cast<unsigned>(region->block_threads), 1, 1,
            0,
            stream,
            arguments,
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
    }
    if (device_workspace.has_value()) {
        device_workspace->record();
    }
    if (timing) {
        result.t_build_us = t_build_us;
        result.t_workspace_us = t_workspace_us;
        result.t_upload_us = t_upload_us;
        result.t_launch_us = fire_timing::duration_us(
            section_mark, fire_timing::Clock::now());
    }
    return result;
}

}  // namespace pie_cuda_driver::pipeline::generated

// PTIR stage-program dispatcher — the nvcc-compiled impl behind the
// CUDA-free `dispatch.hpp` façade. Includes the tier-0 runtime (device
// kernels) here, isolated from the host `.cpp` translation units.

#include "pipeline/dispatch.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_bf16.h>

#include "cuda_check.hpp"
#include "runahead.hpp"
#include "batch/fire_timing.hpp"
#include "batch/forward_graph.hpp"
#include "pipeline/program_runtime.hpp"
#include "pipeline/grouped_runtime.cuh"
#include "pipeline/generated/module_cache.hpp"
#include "pipeline/generated/fused_runtime.cuh"

#include "pipeline/descriptor_resolve.hpp"
#include "pipeline/frame_carrier.hpp"
#include "pipeline/page_translation.hpp"
#include "batch/rs_metadata.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

struct CallbackFence {
    std::atomic<std::uint32_t> pending{0};
};

// W6: fork-join pool for the per-lane PURE host work inside a driver call.
// The lane thread remains the single enqueuer — it forks, participates, and
// joins before any CUDA enqueue or registry mutation depends on the results.
// Workers only ever compute per-lane data into disjoint slots (the audit's
// movability finding: ticket vectors, table entries, and pointer builds are
// pure functions of the launch view and bind-time-immutable registry
// arrays). Tasks are short (~2-20 us); indices are claimed in chunks to
// bound atomic traffic.
class LaneWorkPool {
  public:
    ~LaneWorkPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++epoch_;
        }
        cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    static std::size_t worker_count() {
        static const std::size_t count = [] {
            if (const char* raw = std::getenv("PIE_CUDA_LANE_WORKERS")) {
                char* end = nullptr;
                const long parsed = std::strtol(raw, &end, 10);
                if (end != raw && parsed >= 0 && parsed <= 64) {
                    return static_cast<std::size_t>(parsed);
                }
            }
            // Derivation: the per-lane tasks are short and memory-bound, so
            // wake latency (~10 us/worker) must stay well under the serial
            // pool being split (~0.7 ms at 256 lanes). A quarter of the
            // cores, capped at 6, keeps the fork profitable from ~32 lanes
            // up without competing with the scheduler/runtime threads.
            const unsigned hw = std::thread::hardware_concurrency();
            return static_cast<std::size_t>(
                std::min<unsigned>(6, std::max(1u, hw / 4)));
        }();
        return count;
    }

    // Runs fn(i) for i in [0, n) across the workers + the calling thread.
    // Rethrows the first task exception on the caller after the join.
    void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
        const std::size_t workers = worker_count();
        if (n == 0) return;
        if (workers == 0 || n == 1) {
            for (std::size_t i = 0; i < n; ++i) fn(i);
            return;
        }
        ensure_started(workers);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = &fn;
            count_ = n;
            next_.store(0, std::memory_order_relaxed);
            done_.store(0, std::memory_order_relaxed);
            ++epoch_;
        }
        cv_.notify_all();
        run_share();
        while (done_.load(std::memory_order_acquire) != count_) {
            std::this_thread::yield();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = nullptr;
        }
        if (failure_) {
            std::exception_ptr failure = failure_;
            failure_ = nullptr;
            std::rethrow_exception(failure);
        }
    }

  private:
    static constexpr std::size_t kChunk = 8;

    void ensure_started(std::size_t workers) {
        if (!workers_.empty()) return;
        workers_.reserve(workers);
        for (std::size_t index = 0; index < workers; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void run_share() {
        for (;;) {
            const std::size_t start =
                next_.fetch_add(kChunk, std::memory_order_relaxed);
            if (start >= count_) return;
            const std::size_t stop = std::min(count_, start + kChunk);
            for (std::size_t i = start; i < stop; ++i) {
                try {
                    (*task_)(i);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!failure_) failure_ = std::current_exception();
                }
                done_.fetch_add(1, std::memory_order_acq_rel);
            }
        }
    }

    void worker_loop() {
        std::uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || epoch_ != seen; });
                if (stop_) return;
                seen = epoch_;
                if (task_ == nullptr) continue;
            }
            run_share();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    const std::function<void(std::size_t)>* task_ = nullptr;
    std::size_t count_ = 0;
    std::atomic<std::size_t> next_{0};
    std::atomic<std::size_t> done_{0};
    std::exception_ptr failure_;
    std::uint64_t epoch_ = 0;
    bool stop_ = false;
};

struct BoundInstance {
    struct CommitSnapshot {
        std::uint32_t* device = nullptr;
        std::uint32_t* host = nullptr;
        std::uint32_t* host_device = nullptr;
    };

    std::uint64_t program_hash = 0;
    std::uint32_t geometry_class = PIE_GEOMETRY_CLASS_HOST;
    std::uint64_t pacing_wait_id = 0;
    const Trace* trace = nullptr;
    std::vector<std::uint64_t> channel_ids;
    std::unique_ptr<PtirInstance> instance;
    std::shared_ptr<CallbackFence> callback_fence =
        std::make_shared<CallbackFence>();
    std::vector<std::vector<std::uint32_t>> stage_topologies;
    std::array<std::vector<const plan::StagePlan*>, 4> phase_plans;
    cudaEvent_t publish_done = nullptr;
    std::deque<CommitSnapshot> commit_snapshots;
};

struct NotifyContext;

namespace {

constexpr std::uint64_t kNoDescriptorReadyOffset =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::size_t kDescriptorCopiesPerBlock = 8;
constexpr std::size_t kDescriptorCopyChunkBytes = 4096;
constexpr std::size_t kFixedDecodeInitialLanes = 512;
constexpr std::size_t kFixedDecodePortCount = 7;

struct DescriptorPackCopy {
    std::uint64_t source = 0;
    std::uint64_t ready_source = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t ready_offset = kNoDescriptorReadyOffset;
    std::uint32_t byte_count = 0;
    std::uint8_t default_ready = 0;
    std::uint8_t reserved[3] = {};
};

static_assert(std::is_standard_layout_v<DescriptorPackCopy>);
static_assert(std::is_trivially_copyable_v<DescriptorPackCopy>);

__global__ void pack_descriptor_cells(
    const DescriptorPackCopy* copies,
    std::size_t count,
    std::uint8_t* output) {
    const std::size_t warp = threadIdx.x / warpSize;
    const std::size_t lane = threadIdx.x % warpSize;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) *
        kDescriptorCopiesPerBlock;
    for (std::size_t index =
             static_cast<std::size_t>(blockIdx.x) *
                 kDescriptorCopiesPerBlock +
             warp;
         index < count;
         index += stride) {
        const DescriptorPackCopy copy = copies[index];
        const auto* source = reinterpret_cast<const std::uint8_t*>(
            static_cast<std::uintptr_t>(copy.source));
        for (std::size_t byte = lane;
             byte < copy.byte_count;
             byte += warpSize) {
            output[copy.destination_offset + byte] = source[byte];
        }
        if (lane == 0 && copy.ready_offset != kNoDescriptorReadyOffset) {
            const auto* ready =
                reinterpret_cast<const std::uint8_t*>(
                    static_cast<std::uintptr_t>(copy.ready_source));
            output[copy.ready_offset] =
                ready == nullptr ? copy.default_ready : *ready;
        }
    }
}

class DescriptorReadbackArena {
  public:
    ~DescriptorReadbackArena() noexcept {
        if (device_copies_ != nullptr) cudaFree(device_copies_);
        if (host_copies_ != nullptr) cudaFreeHost(host_copies_);
        if (device_bytes_ != nullptr) cudaFree(device_bytes_);
        if (host_bytes_ != nullptr) cudaFreeHost(host_bytes_);
    }

    DescriptorReadbackArena() = default;
    DescriptorReadbackArena(const DescriptorReadbackArena&) = delete;
    DescriptorReadbackArena& operator=(const DescriptorReadbackArena&) =
        delete;

    const std::uint8_t* read(
        std::span<const DescriptorPackCopy> copies,
        std::size_t bytes,
        cudaStream_t stream) {
        if (copies.empty() || bytes == 0) return nullptr;
        reserve_copies(copies.size());
        reserve_bytes(bytes);
        std::memcpy(
            host_copies_, copies.data(),
            copies.size_bytes());

        bool submitted = false;
        try {
            CUDA_CHECK(cudaMemcpyAsync(
                device_copies_, host_copies_, copies.size_bytes(),
                cudaMemcpyHostToDevice, stream));
            submitted = true;
            const std::size_t required_blocks =
                (copies.size() + kDescriptorCopiesPerBlock - 1) /
                kDescriptorCopiesPerBlock;
            const std::uint32_t blocks = static_cast<std::uint32_t>(
                std::min<std::size_t>(required_blocks, 65535));
            pack_descriptor_cells<<<
                blocks,
                kDescriptorCopiesPerBlock * 32,
                0,
                stream>>>(
                device_copies_, copies.size(), device_bytes_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(
                host_bytes_, device_bytes_, bytes,
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (...) {
            if (submitted) {
                static_cast<void>(cudaStreamSynchronize(stream));
            }
            throw;
        }
        return host_bytes_;
    }

  private:
    static std::size_t grown_capacity(
        std::size_t current,
        std::size_t required,
        std::size_t minimum) {
        if (current >= required) return current;
        if (current == 0) return std::max(required, minimum);
        if (current > std::numeric_limits<std::size_t>::max() / 2) {
            return required;
        }
        return std::max(required, current * 2);
    }

    void reserve_copies(std::size_t required) {
        if (required <= copy_capacity_) return;
        const std::size_t capacity =
            grown_capacity(copy_capacity_, required, 64);
        DescriptorPackCopy* host = nullptr;
        DescriptorPackCopy* device = nullptr;
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&host),
            capacity * sizeof(DescriptorPackCopy)));
        try {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&device),
                capacity * sizeof(DescriptorPackCopy)));
        } catch (...) {
            cudaFreeHost(host);
            throw;
        }
        if (device_copies_ != nullptr) cudaFree(device_copies_);
        if (host_copies_ != nullptr) cudaFreeHost(host_copies_);
        device_copies_ = device;
        host_copies_ = host;
        copy_capacity_ = capacity;
    }

    void reserve_bytes(std::size_t required) {
        if (required <= byte_capacity_) return;
        const std::size_t capacity =
            grown_capacity(byte_capacity_, required, 4096);
        std::uint8_t* host = nullptr;
        std::uint8_t* device = nullptr;
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&host), capacity));
        try {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&device), capacity));
        } catch (...) {
            cudaFreeHost(host);
            throw;
        }
        if (device_bytes_ != nullptr) cudaFree(device_bytes_);
        if (host_bytes_ != nullptr) cudaFreeHost(host_bytes_);
        device_bytes_ = device;
        host_bytes_ = host;
        byte_capacity_ = capacity;
    }

    DescriptorPackCopy* host_copies_ = nullptr;
    DescriptorPackCopy* device_copies_ = nullptr;
    std::uint8_t* host_bytes_ = nullptr;
    std::uint8_t* device_bytes_ = nullptr;
    std::size_t copy_capacity_ = 0;
    std::size_t byte_capacity_ = 0;
};

struct FixedDecodeLane {
    std::uint64_t token = 0;
    std::uint64_t position = 0;
    std::uint64_t pages = 0;
    std::uint64_t page_indptr = 0;
    std::uint64_t kv_len = 0;
    std::uint64_t w_slot = 0;
    std::uint64_t w_off = 0;
    std::uint64_t ready[kFixedDecodePortCount] = {};
    std::uint64_t pass_commit = 0;
    std::uint64_t translation = 0;
    std::uint64_t write_lower_bound = 0;
    std::uint64_t write_upper_bound =
        std::numeric_limits<std::uint64_t>::max();
    std::uint32_t translation_len = 0;
    std::uint32_t pages_capacity = 0;
};

struct FixedDecodeOutputs {
    std::uint32_t* token_ids = nullptr;
    std::uint32_t* position_ids = nullptr;
    std::uint32_t* qo_indptr = nullptr;
    std::uint32_t* kv_page_indices = nullptr;
    std::uint32_t* kv_page_indptr = nullptr;
    std::uint32_t* kv_last_page_lens = nullptr;
    std::uint32_t* w_page = nullptr;
    std::uint32_t* w_off = nullptr;
    std::uint8_t* row_valid = nullptr;
    std::int32_t* rs_slot_ids = nullptr;
    std::int32_t* sample_indices = nullptr;
    // Monotonic device counter of fail-stopped lanes (chain kills); the
    // host mirrors it after each batch and reports growth loudly.
    std::uint32_t* chain_kills = nullptr;
    std::uint32_t dummy_page = 0;
    std::uint32_t page_size = 0;
    std::uint32_t device_pages = 0;
};

static_assert(std::is_standard_layout_v<FixedDecodeLane>);
static_assert(std::is_trivially_copyable_v<FixedDecodeLane>);
static_assert(std::is_standard_layout_v<FixedDecodeOutputs>);

template <typename T>
__device__ const T* fixed_decode_pointer(std::uint64_t address) {
    return reinterpret_cast<const T*>(
        static_cast<std::uintptr_t>(address));
}

__global__ void compose_fixed_decode(
    const FixedDecodeLane* lanes,
    std::uint32_t lane_count,
    FixedDecodeOutputs output) {
    extern __shared__ std::uint32_t page_offsets[];
    const std::uint32_t lane = threadIdx.x;
    bool valid = lane < lane_count;
    bool sentinel = false;
    std::uint32_t token = 0;
    const FixedDecodeLane* descriptor =
        valid ? &lanes[lane] : nullptr;

    if (valid) {
        const auto* commit =
            fixed_decode_pointer<std::uint32_t>(
            descriptor->pass_commit);
        valid = commit != nullptr && *commit != 0;
        for (std::size_t port = 0;
             port < kFixedDecodePortCount;
             ++port) {
            const auto* ready =
                fixed_decode_pointer<std::uint8_t>(
                    descriptor->ready[port]);
            if (ready != nullptr && *ready == 0) valid = false;
        }
        const auto* token_source =
            fixed_decode_pointer<std::uint32_t>(descriptor->token);
        if (token_source == nullptr) {
            valid = false;
        } else {
            token = *token_source;
            sentinel =
                token == std::numeric_limits<std::uint32_t>::max();
        }
    }

    std::uint32_t page_count = 1;
    std::uint32_t kv_len = 1;
    std::uint32_t write_page = output.dummy_page;
    std::uint32_t write_offset = 0;
    if (valid && !sentinel) {
        const auto* page_indptr =
            fixed_decode_pointer<std::uint32_t>(
                descriptor->page_indptr);
        const auto* pages =
            fixed_decode_pointer<std::uint32_t>(
                descriptor->pages);
        const auto* translation =
            fixed_decode_pointer<std::uint32_t>(
                descriptor->translation);
        const auto* kv_len_source =
            fixed_decode_pointer<std::uint32_t>(
                descriptor->kv_len);
        const auto* w_slot =
            fixed_decode_pointer<std::uint32_t>(
                descriptor->w_slot);
        const auto* w_off =
            fixed_decode_pointer<std::uint32_t>(
                descriptor->w_off);
        if (page_indptr == nullptr || pages == nullptr ||
            translation == nullptr || kv_len_source == nullptr ||
            w_slot == nullptr || w_off == nullptr ||
            page_indptr[0] != 0) {
            valid = false;
        } else {
            page_count = page_indptr[1];
            kv_len = *kv_len_source;
            write_offset = *w_off;
            const std::uint32_t logical_write_page = *w_slot;
            const std::uint64_t logical_write_position =
                static_cast<std::uint64_t>(logical_write_page) *
                    output.page_size +
                write_offset;
            const std::uint32_t expected_pages =
                kv_len == 0 || output.page_size == 0
                    ? 0
                    : (kv_len + output.page_size - 1) /
                          output.page_size;
            if (page_count == 0 ||
                page_count > descriptor->pages_capacity ||
                page_count > descriptor->translation_len ||
                page_count != expected_pages ||
                logical_write_page >= descriptor->translation_len ||
                write_offset >= output.page_size ||
                logical_write_position < descriptor->write_lower_bound ||
                logical_write_position >=
                    descriptor->write_upper_bound) {
                valid = false;
            } else {
                write_page = translation[logical_write_page];
                if (write_page >= output.device_pages) valid = false;
                for (std::uint32_t page = 0;
                     page < page_count;
                     ++page) {
                    const std::uint32_t logical_page = pages[page];
                    if (logical_page >= descriptor->translation_len ||
                        translation[logical_page] >= output.device_pages) {
                        valid = false;
                        break;
                    }
                }
            }
        }
    }

    if (!valid && lane < lane_count) {
        // Fail-stop: kill the chain (successors dummy-run) AND count the
        // kill so the host reports it loudly — never a silent poison.
        auto* commit = const_cast<std::uint32_t*>(
            fixed_decode_pointer<std::uint32_t>(
                descriptor->pass_commit));
        if (commit != nullptr) *commit = 0;
        if (output.chain_kills != nullptr) {
            atomicAdd(output.chain_kills, 1u);
        }
        page_count = 1;
        kv_len = 1;
        write_page = output.dummy_page;
        write_offset = 0;
    }
    if (sentinel) {
        page_count = 1;
        kv_len = 1;
        write_page = output.dummy_page;
        write_offset = 0;
    }
    if (lane < lane_count) {
        page_offsets[lane] = page_count;
    }
    __syncthreads();

    if (lane == 0) {
        std::uint32_t page_cursor = 0;
        output.qo_indptr[0] = 0;
        output.kv_page_indptr[0] = 0;
        for (std::uint32_t index = 0;
             index < lane_count;
             ++index) {
            const std::uint32_t count = page_offsets[index];
            page_offsets[index] = page_cursor;
            page_cursor += count;
            output.qo_indptr[index + 1] = index + 1;
            output.kv_page_indptr[index + 1] = page_cursor;
        }
    }
    __syncthreads();

    if (lane >= lane_count) return;
    const bool active = valid && !sentinel;
    output.row_valid[lane] = static_cast<std::uint8_t>(active);
    output.token_ids[lane] = active ? token : 0;
    const auto* position =
        fixed_decode_pointer<std::uint32_t>(descriptor->position);
    output.position_ids[lane] =
        active && position != nullptr ? *position : 0;
    output.kv_last_page_lens[lane] =
        active
            ? ((kv_len - 1) % output.page_size) + 1
            : 1;
    output.w_page[lane] = write_page;
    output.w_off[lane] = write_offset;
    if (!active && output.rs_slot_ids != nullptr) {
        output.rs_slot_ids[lane] = -1;
    }
    if (output.sample_indices != nullptr) {
        output.sample_indices[lane] =
            static_cast<std::int32_t>(lane);
    }

    const std::uint32_t page_base = page_offsets[lane];
    if (!active) {
        output.kv_page_indices[page_base] = output.dummy_page;
        return;
    }
    const auto* pages =
        fixed_decode_pointer<std::uint32_t>(
            descriptor->pages);
    const auto* translation =
        fixed_decode_pointer<std::uint32_t>(
            descriptor->translation);
    for (std::uint32_t page = 0;
         page < page_count;
         ++page) {
        output.kv_page_indices[page_base + page] =
            translation[pages[page]];
    }
}

// Pinned host staging depth: single-sourced from runahead.hpp (must
// EXCEED the scheduler's run-ahead, not match it — a depth-equal pool
// blocks every submit in cudaEventSynchronize once the pipe is full).
using pie_cuda_driver::kUploadStagingDepth;

class FixedDecodeUploadArena {
  public:
    FixedDecodeUploadArena() {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &copy_stream_, cudaStreamNonBlocking));
        try {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &upload_done_, cudaEventDisableTiming));
        } catch (...) {
            cudaStreamDestroy(copy_stream_);
            copy_stream_ = nullptr;
            throw;
        }
    }

    ~FixedDecodeUploadArena() noexcept {
        if (copy_stream_ != nullptr) {
            static_cast<void>(cudaStreamSynchronize(copy_stream_));
        }
        if (device_lanes_ != nullptr) cudaFree(device_lanes_);
        if (device_translation_ != nullptr) cudaFree(device_translation_);
        if (device_done_ != nullptr) cudaEventDestroy(device_done_);
        if (upload_done_ != nullptr) cudaEventDestroy(upload_done_);
        for (HostSlot& slot : host_slots_) {
            if (slot.lanes != nullptr) cudaFreeHost(slot.lanes);
            if (slot.translation != nullptr) {
                cudaFreeHost(slot.translation);
            }
            if (slot.copy_done != nullptr) {
                cudaEventDestroy(slot.copy_done);
            }
        }
        if (copy_stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(copy_stream_));
        }
    }

    FixedDecodeUploadArena(const FixedDecodeUploadArena&) = delete;
    FixedDecodeUploadArena& operator=(
        const FixedDecodeUploadArena&) = delete;

    void reserve(
        std::size_t lanes,
        std::size_t translations,
        cudaStream_t stream) {
        if (lanes <= lane_capacity_ &&
            translations <= translation_capacity_) {
            return;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        device_pending_ = false;
        for (HostSlot& slot : host_slots_) slot.pending = false;

        const std::size_t lane_capacity =
            grown_capacity(lane_capacity_, lanes, kFixedDecodeInitialLanes);
        const std::size_t translation_capacity =
            grown_capacity(translation_capacity_, translations, 16384);
        if (lane_capacity >
                std::numeric_limits<std::size_t>::max() /
                    sizeof(FixedDecodeLane) ||
            translation_capacity >
                std::numeric_limits<std::size_t>::max() /
                    sizeof(std::uint32_t)) {
            throw std::runtime_error(
                "fixed-decode upload capacity overflow");
        }
        FixedDecodeLane* device_lanes = nullptr;
        std::uint32_t* device_translation = nullptr;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&device_lanes),
            lane_capacity * sizeof(FixedDecodeLane)));
        try {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&device_translation),
                translation_capacity * sizeof(std::uint32_t)));
        } catch (...) {
            cudaFree(device_lanes);
            throw;
        }
        if (device_lanes_ != nullptr) cudaFree(device_lanes_);
        if (device_translation_ != nullptr) cudaFree(device_translation_);
        device_lanes_ = device_lanes;
        device_translation_ = device_translation;
        lane_capacity_ = lane_capacity;
        translation_capacity_ = translation_capacity;
    }

    const std::uint32_t* translation_at(
        std::size_t offset) const noexcept {
        return device_translation_ + offset;
    }

    const FixedDecodeLane* upload(
        std::span<const FixedDecodeLane> lanes,
        std::span<const std::uint32_t> translation,
        cudaStream_t consumer_stream) {
        if (device_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(
                copy_stream_, device_done_, 0));
        }
        HostSlot& slot = host_slots_[next_slot_];
        next_slot_ = (next_slot_ + 1) % host_slots_.size();
        acquire_host_slot(slot, copy_stream_);
        std::memcpy(
            slot.lanes, lanes.data(), lanes.size_bytes());
        std::memcpy(
            slot.translation, translation.data(),
            translation.size_bytes());

        bool submitted = false;
        try {
            CUDA_CHECK(cudaMemcpyAsync(
                device_translation_, slot.translation,
                translation.size_bytes(),
                cudaMemcpyHostToDevice, copy_stream_));
            submitted = true;
            CUDA_CHECK(cudaMemcpyAsync(
                device_lanes_, slot.lanes, lanes.size_bytes(),
                cudaMemcpyHostToDevice, copy_stream_));
            CUDA_CHECK(cudaEventRecord(
                slot.copy_done, copy_stream_));
            CUDA_CHECK(cudaEventRecord(
                upload_done_, copy_stream_));
            CUDA_CHECK(cudaStreamWaitEvent(
                consumer_stream, upload_done_, 0));
            slot.pending = true;
        } catch (...) {
            if (submitted) {
                static_cast<void>(
                    cudaStreamSynchronize(copy_stream_));
            }
            slot.pending = false;
            throw;
        }
        return device_lanes_;
    }

    void mark_used(cudaStream_t stream) {
        if (device_done_ == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &device_done_, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(device_done_, stream));
        device_pending_ = true;
    }

  private:
    struct HostSlot {
        FixedDecodeLane* lanes = nullptr;
        std::uint32_t* translation = nullptr;
        cudaEvent_t copy_done = nullptr;
        bool pending = false;
        std::size_t lane_capacity = 0;
        std::size_t translation_capacity = 0;
    };

    static std::size_t grown_capacity(
        std::size_t current,
        std::size_t required,
        std::size_t minimum) {
        if (current >= required) return current;
        if (current == 0) return std::max(required, minimum);
        if (current > std::numeric_limits<std::size_t>::max() / 2) {
            return required;
        }
        return std::max(required, current * 2);
    }

    void acquire_host_slot(HostSlot& slot, cudaStream_t stream) {
        if (slot.pending) {
            CUDA_CHECK(cudaEventSynchronize(slot.copy_done));
            slot.pending = false;
        }
        if (slot.lane_capacity < lane_capacity_) {
            FixedDecodeLane* replacement = nullptr;
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&replacement),
                lane_capacity_ * sizeof(FixedDecodeLane)));
            if (slot.lanes != nullptr) cudaFreeHost(slot.lanes);
            slot.lanes = replacement;
            slot.lane_capacity = lane_capacity_;
        }
        if (slot.translation_capacity < translation_capacity_) {
            std::uint32_t* replacement = nullptr;
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&replacement),
                translation_capacity_ * sizeof(std::uint32_t)));
            if (slot.translation != nullptr) {
                cudaFreeHost(slot.translation);
            }
            slot.translation = replacement;
            slot.translation_capacity = translation_capacity_;
        }
        if (slot.copy_done == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &slot.copy_done, cudaEventDisableTiming));
        }
        static_cast<void>(stream);
    }

    std::array<HostSlot, kUploadStagingDepth> host_slots_{};
    FixedDecodeLane* device_lanes_ = nullptr;
    std::uint32_t* device_translation_ = nullptr;
    cudaEvent_t device_done_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t upload_done_ = nullptr;
    bool device_pending_ = false;
    std::size_t lane_capacity_ = 0;
    std::size_t translation_capacity_ = 0;
    std::size_t next_slot_ = 0;
};

constexpr std::size_t kDecodeEnvelopeMaxLanes = 1024;

struct DecodeEnvelopeLane {
    std::uint64_t token_source = 0;
    std::uint64_t position_source = 0;
    std::uint64_t pass_commit = 0;
    // Containment as launch args: the device-resolved write position must
    // land in the declaration's exact [lower, upper) token span.
    std::uint64_t write_lower_bound = 0;
    std::uint64_t write_upper_bound =
        std::numeric_limits<std::uint64_t>::max();
    std::uint32_t token_start = 0;
    std::uint32_t request_index = 0;
    std::uint32_t source_token_start = 0;
    std::uint32_t source_position_start = 0;
    std::uint32_t source_page_begin = 0;
    std::uint32_t source_page_count = 0;
    std::uint32_t passthrough = 0;
};

static_assert(std::is_standard_layout_v<DecodeEnvelopeLane>);
static_assert(std::is_trivially_copyable_v<DecodeEnvelopeLane>);

struct DecodeEnvelopeOutputs {
    std::uint32_t* token_ids = nullptr;
    std::uint32_t* position_ids = nullptr;
    std::uint32_t* kv_page_indices = nullptr;
    std::uint32_t* kv_page_indptr = nullptr;
    std::uint32_t* kv_last_page_lens = nullptr;
    std::uint8_t* row_valid = nullptr;
    std::int32_t* rs_slot_ids = nullptr;
    // Stream-ordered snapshot of the template page table. Lanes read their
    // source spans from here and write compacted spans to
    // `kv_page_indices`, so parallel lanes never race the in-place
    // left-shift that the serial kernel relied on (RV-17).
    const std::uint32_t* template_pages = nullptr;
    // Monotonic device counter of fail-stopped lanes; the host mirrors it
    // after each batch and reports growth loudly (RV-2 diagnostics).
    std::uint32_t* chain_kills = nullptr;
    std::uint32_t dummy_page = 0;
    std::uint32_t page_size = 0;
};

// One thread per batch request. Envelope lanes resolve token/position from
// device channels and are containment-checked; passthrough lanes only carry
// their template page span. Dead lanes (sentinel token or fail-stopped)
// shrink to one dummy page, and the page table compacts through a shared
// prefix scan exactly like `compose_fixed_decode`.
__global__ void compose_decode_envelopes(
    const DecodeEnvelopeLane* lanes,
    std::uint32_t lane_count,
    DecodeEnvelopeOutputs output) {
    extern __shared__ std::uint32_t page_offsets[];
    const std::uint32_t lane_index = threadIdx.x;
    const bool in_range = lane_index < lane_count;
    DecodeEnvelopeLane lane{};
    bool active = false;
    bool killed = false;
    std::uint32_t token = 0;
    std::uint32_t position = 0;
    if (in_range) {
        lane = lanes[lane_index];
        if (lane.passthrough == 0) {
            const auto* tokens =
                fixed_decode_pointer<std::uint32_t>(lane.token_source);
            const auto* positions =
                fixed_decode_pointer<std::uint32_t>(lane.position_source);
            token = tokens[lane.source_token_start];
            active =
                token != std::numeric_limits<std::uint32_t>::max();
            if (active) {
                position = positions[lane.source_position_start];
                const std::uint64_t write_position = position;
                if (write_position < lane.write_lower_bound ||
                    write_position >= lane.write_upper_bound ||
                    position / output.page_size >=
                        lane.source_page_count) {
                    killed = true;
                    active = false;
                }
            }
        }
        page_offsets[lane_index] =
            (lane.passthrough != 0 || active)
                ? lane.source_page_count
                : 1;
    }
    __syncthreads();

    if (lane_index == 0) {
        std::uint32_t page_cursor = 0;
        output.kv_page_indptr[0] = 0;
        for (std::uint32_t index = 0; index < lane_count; ++index) {
            const std::uint32_t count = page_offsets[index];
            page_offsets[index] = page_cursor;
            page_cursor += count;
            output.kv_page_indptr[index + 1] = page_cursor;
        }
    }
    __syncthreads();

    if (!in_range) return;
    const std::uint32_t destination_page_begin =
        page_offsets[lane_index];
    if (lane.passthrough != 0 || active) {
        for (std::uint32_t page = 0;
             page < lane.source_page_count;
             ++page) {
            output.kv_page_indices[destination_page_begin + page] =
                output.template_pages[lane.source_page_begin + page];
        }
    } else {
        output.kv_page_indices[destination_page_begin] =
            output.dummy_page;
        output.kv_last_page_lens[lane.request_index] = 1;
    }
    if (lane.passthrough != 0) return;

    output.token_ids[lane.token_start] = active ? token : 0;
    output.position_ids[lane.token_start] = active ? position : 0;
    output.row_valid[lane.token_start] =
        static_cast<std::uint8_t>(active);
    if (!active && output.rs_slot_ids != nullptr) {
        output.rs_slot_ids[lane.request_index] = -1;
    }
    if (killed) {
        auto* commit = const_cast<std::uint32_t*>(
            fixed_decode_pointer<std::uint32_t>(lane.pass_commit));
        if (commit != nullptr) *commit = 0;
        if (output.chain_kills != nullptr) {
            atomicAdd(output.chain_kills, 1u);
        }
    }
}

class DecodeEnvelopeUploadArena {
  public:
    struct Staged {
        const DecodeEnvelopeLane* lanes = nullptr;
        const std::uint32_t* template_pages = nullptr;
    };

    ~DecodeEnvelopeUploadArena() noexcept {
        if (device_ != nullptr) cudaFree(device_);
        if (pages_device_ != nullptr) cudaFree(pages_device_);
        if (device_done_ != nullptr) cudaEventDestroy(device_done_);
        for (HostSlot& slot : host_slots_) {
            if (slot.host != nullptr) cudaFreeHost(slot.host);
            if (slot.copy_done != nullptr) {
                cudaEventDestroy(slot.copy_done);
            }
        }
    }

    // Stages the lane table AND a device snapshot of the template page
    // table (copied stream-ordered from `template_pages`). The snapshot is
    // what the compose kernel reads its source spans from, so the parallel
    // lanes never alias the compacted `kv_page_indices` they write.
    Staged upload(
        std::span<const DecodeEnvelopeLane> lanes,
        const std::uint32_t* template_pages,
        std::size_t template_page_count,
        cudaStream_t stream) {
        reserve(lanes.size(), template_page_count, stream);
        if (device_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, device_done_, 0));
        }
        HostSlot& slot = host_slots_[next_slot_];
        next_slot_ = (next_slot_ + 1) % host_slots_.size();
        if (slot.pending) {
            CUDA_CHECK(cudaEventSynchronize(slot.copy_done));
            slot.pending = false;
        }
        std::memcpy(slot.host, lanes.data(), lanes.size_bytes());
        bool submitted = false;
        try {
            CUDA_CHECK(cudaMemcpyAsync(
                device_, slot.host, lanes.size_bytes(),
                cudaMemcpyHostToDevice, stream));
            submitted = true;
            CUDA_CHECK(cudaEventRecord(slot.copy_done, stream));
            slot.pending = true;
            CUDA_CHECK(cudaMemcpyAsync(
                pages_device_, template_pages,
                template_page_count * sizeof(std::uint32_t),
                cudaMemcpyDeviceToDevice, stream));
        } catch (...) {
            if (submitted) {
                static_cast<void>(cudaStreamSynchronize(stream));
            }
            slot.pending = false;
            throw;
        }
        return Staged{device_, pages_device_};
    }

    void mark_used(cudaStream_t stream) {
        if (device_done_ == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &device_done_, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(device_done_, stream));
        device_pending_ = true;
    }

  private:
    struct HostSlot {
        DecodeEnvelopeLane* host = nullptr;
        cudaEvent_t copy_done = nullptr;
        bool pending = false;
    };

    void reserve(std::size_t required,
                 std::size_t required_pages,
                 cudaStream_t stream) {
        if (required <= capacity_ && required_pages <= pages_capacity_) {
            return;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t capacity =
            std::max({required, capacity_, kFixedDecodeInitialLanes});
        const std::size_t pages_capacity = std::max(
            {required_pages, pages_capacity_,
             kFixedDecodeInitialLanes});
        if (capacity >
                std::numeric_limits<std::size_t>::max() /
                    sizeof(DecodeEnvelopeLane) ||
            pages_capacity >
                std::numeric_limits<std::size_t>::max() /
                    sizeof(std::uint32_t)) {
            throw std::runtime_error(
                "decode-envelope staging capacity overflow");
        }
        DecodeEnvelopeLane* replacement_device = nullptr;
        std::uint32_t* replacement_pages = nullptr;
        std::array<HostSlot, kUploadStagingDepth> replacement_slots{};
        auto release_replacements = [&] {
            if (replacement_device != nullptr) {
                cudaFree(replacement_device);
                replacement_device = nullptr;
            }
            if (replacement_pages != nullptr) {
                cudaFree(replacement_pages);
                replacement_pages = nullptr;
            }
            for (HostSlot& slot : replacement_slots) {
                if (slot.host != nullptr) {
                    cudaFreeHost(slot.host);
                    slot.host = nullptr;
                }
                if (slot.copy_done != nullptr) {
                    cudaEventDestroy(slot.copy_done);
                    slot.copy_done = nullptr;
                }
            }
        };
        try {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&replacement_device),
                capacity * sizeof(DecodeEnvelopeLane)));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&replacement_pages),
                pages_capacity * sizeof(std::uint32_t)));
            for (HostSlot& slot : replacement_slots) {
                CUDA_CHECK(cudaMallocHost(
                    reinterpret_cast<void**>(&slot.host),
                    capacity * sizeof(DecodeEnvelopeLane)));
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &slot.copy_done, cudaEventDisableTiming));
            }
        } catch (...) {
            release_replacements();
            throw;
        }

        if (device_ != nullptr) cudaFree(device_);
        if (pages_device_ != nullptr) cudaFree(pages_device_);
        for (HostSlot& slot : host_slots_) {
            if (slot.host != nullptr) cudaFreeHost(slot.host);
            if (slot.copy_done != nullptr) {
                cudaEventDestroy(slot.copy_done);
            }
        }
        device_ = replacement_device;
        replacement_device = nullptr;
        pages_device_ = replacement_pages;
        replacement_pages = nullptr;
        host_slots_ = replacement_slots;
        capacity_ = capacity;
        pages_capacity_ = pages_capacity;
        device_pending_ = false;
        next_slot_ = 0;
    }

    std::array<HostSlot, kUploadStagingDepth> host_slots_{};
    DecodeEnvelopeLane* device_ = nullptr;
    std::uint32_t* pages_device_ = nullptr;
    cudaEvent_t device_done_ = nullptr;
    bool device_pending_ = false;
    std::size_t capacity_ = 0;
    std::size_t pages_capacity_ = 0;
    std::size_t next_slot_ = 0;
};

}  // namespace

struct Dispatch::Impl {
    static constexpr std::size_t kSignatureStreamCount = 4;
    static constexpr std::size_t kMaxRetainedInstanceResources = 2048;
    static constexpr std::size_t kMaxSettlementArenas = 8;
    Impl() = default;
    ~Impl();
    PtirProgramCache cache;
    generated::ModuleCache fused_modules;
    generated::GeneratedRuntimeContext generated_runtime;
    std::unordered_map<
        std::uint64_t,
        std::shared_ptr<const GroupedStageStaticPlan>> grouped_plans;
    DeviceChannelRegistry channels;
    std::unordered_map<std::uint64_t, BoundInstance> instances;
    DescriptorReadbackArena descriptor_readback;
    FixedDecodeUploadArena fixed_decode_upload;
    DecodeEnvelopeUploadArena decode_envelope_upload;
    std::vector<cudaEvent_t> available_publish_events;
    // W6: per-wave launch events (source_ready, phase_done, signature_*)
    // are acquired here and returned at StagedLaunch teardown — event
    // create/destroy used to run 3-5x per wave on the lane thread.
    std::vector<cudaEvent_t> available_launch_events;
    // One publication-ordering point per WAVE instead of one per instance:
    // every wave's channel publications ride the same callback stream, so a
    // single event recorded after the wave's publication enqueue subsumes
    // all per-instance ordering (the driver lane serializes launches, so a
    // later wave's wait always observes the intended record). Per-instance
    // `publish_done` remains only for the bind-time seed upload (a
    // different stream), consumed at the instance's first completed wave.
    cudaEvent_t publications_done = nullptr;
    bool publications_recorded = false;
    std::vector<BoundInstance::CommitSnapshot> available_commit_snapshots;
    // W4 exit reaper: a closed instance's resources may only be reclaimed
    // after its callback fence drains and its publication events settle —
    // waits that used to run ON THE LANE, serializing every exit close
    // behind up to a full in-flight wave (measured: one close_instance
    // held the lane 14.7 ms; exit phase = 46% of the run's gap). The
    // reaper thread does ONLY the waiting; everything that touches the
    // registry or the pools (PtirInstance drop -> channel refcounts,
    // snapshot/event returns) comes back via `reaped_ready` and runs on
    // the lane at its next entry point. No new thread-safety surface.
    struct InstanceReapItem {
        BoundInstance bound;
        bool wait_publications = false;
    };
    std::thread instance_reaper;
    std::mutex reaper_mutex;
    std::condition_variable reaper_cv;
    std::deque<InstanceReapItem> reaper_queue;
    std::deque<BoundInstance> reaped_ready;
    bool reaper_stop = false;
    std::vector<std::unique_ptr<NotifyContext>> settlement_arenas;
    std::mutex settlement_mutex;
    std::atomic<bool> shutting_down{false};
    std::atomic<std::uint32_t> force_retry_launches_remaining{
        std::getenv("PIE_CUDA_FORCE_RETRY_ONCE") != nullptr ? 1u : 0u
    };
    DispatchStats stats;
    // W6: per-lane pure-work fork-join (ticket builds, settle tables).
    LaneWorkPool lane_pool;
    mutable std::mutex stats_mutex;
    // Fixed-decode chain-kill diagnostic: a monotonic device counter the
    // compose kernel bumps on fail-stop, mirrored into pinned memory after
    // each batch and reported loudly when it grows.
    std::uint32_t* d_fixed_decode_kills = nullptr;
    std::uint32_t* h_fixed_decode_kills = nullptr;
    std::uint32_t fixed_decode_kills_reported = 0;
    // Same diagnostic for the decode-envelope compose path (RV-16).
    std::uint32_t* d_envelope_kills = nullptr;
    std::uint32_t* h_envelope_kills = nullptr;
    std::uint32_t envelope_kills_reported = 0;
    cudaStream_t output_copy_stream = nullptr;
    cudaStream_t group_streams[2] = {nullptr, nullptr};
    cudaStream_t signature_streams[kSignatureStreamCount] = {};
    bool attention_hook_coverage = false;
    std::uint32_t model_layers = 0;
};

struct StagedLane {
    std::size_t program = 0;
    BoundInstance* bound = nullptr;
    BoundInstance::CommitSnapshot* snapshot = nullptr;
    const std::vector<plan::StagePlan>* plans = nullptr;
    const std::vector<std::uint64_t>* plan_identities = nullptr;
    std::shared_ptr<const generated::FusedProgramExecutable>
        generated_program;
    const std::array<std::vector<const plan::StagePlan*>, 4>*
        phase_plans = nullptr;
    std::vector<DeviceHostChannelTicket> tickets;
    DeviceHostChannelTicket* device_tickets = nullptr;
    std::uint32_t device_ticket_offset = 0;
    std::uint32_t device_ticket_count = 0;
    std::unordered_set<std::uint32_t> prior_put_slots;
    std::unordered_set<std::uint32_t> prior_take_slots;
    std::uint32_t row_offset = 0;
    std::uint32_t sampled_rows = 0;
    std::uint32_t token_start = 0;
    std::uint32_t runtime_row_count = kUnavailableGroupedExtent;
    std::uint32_t token_count = kUnavailableGroupedExtent;
    std::uint32_t kv_len = kUnavailableGroupedExtent;
    std::uint32_t page_count = kUnavailableGroupedExtent;
    std::uint32_t query_len = kUnavailableGroupedExtent;
    std::uint32_t key_len = kUnavailableGroupedExtent;
    std::uint32_t logical_vocab = 0;
    std::vector<std::uint64_t> logits_bf16_rows;
    std::vector<std::uint64_t> mtp_logits_bf16_rows;
    const std::uint8_t* row_valid = nullptr;
    std::uint32_t row_valid_offset = 0;
};

struct StagedLaunch::State {
    Dispatch::Impl* owner = nullptr;
    pie_native::LaunchView view{};
    cudaStream_t stream = nullptr;
    std::vector<std::unique_ptr<StagedLane>> lanes;
    std::vector<std::uint64_t> touched_instances;
    std::vector<DeviceHostChannelTicket> ticket_staging;
    std::vector<PullValidateHostChannelLane> pull_staging;
    // Host-writer ring pulls staged for this launch (bool cells unpack on
    // the CPU into these buffers, which the async H2D copies read). Riding
    // the launch state — which outlives every copy on `stream` — is what
    // lets the pull skip the old whole-device synchronize on the fire path.
    std::vector<std::vector<std::uint8_t>> writer_staging;
    DeviceHostChannelTicket* device_tickets = nullptr;
    std::uint32_t* device_layer = nullptr;
    cudaEvent_t source_ready = nullptr;
    cudaEvent_t phase_done[2] = {nullptr, nullptr};
    cudaEvent_t signature_ready = nullptr;
    cudaEvent_t signature_done[
        Dispatch::Impl::kSignatureStreamCount] = {};
    std::array<std::uint32_t, 4> phase_invocations{};
    bool active = true;
    bool failed = false;
};

namespace {
cudaEvent_t acquire_launch_event(Dispatch::Impl& s);
void release_launch_event(Dispatch::Impl& s, cudaEvent_t event);
}  // namespace

StagedLaunch::StagedLaunch() : state_(std::make_unique<State>()) {}

StagedLaunch::~StagedLaunch() {
    if (!state_) return;
    if (state_->active) {
        cudaStreamSynchronize(state_->stream);
    }
    // W6: these frees run PER WAVE on the lane thread at scope exit —
    // stream-ordered frees and pool returns instead of the old plain
    // cudaFree (potentially device-synchronizing) + event destroys.
    if (state_->device_tickets != nullptr) {
        if (state_->stream != nullptr) {
            cudaFreeAsync(state_->device_tickets, state_->stream);
        } else {
            cudaFree(state_->device_tickets);
        }
        state_->device_tickets = nullptr;
    }
    if (state_->device_layer != nullptr) {
        if (state_->stream != nullptr) {
            cudaFreeAsync(state_->device_layer, state_->stream);
        } else {
            cudaFree(state_->device_layer);
        }
        state_->device_layer = nullptr;
    }
    const auto retire_event = [this](cudaEvent_t& event) {
        if (event == nullptr) return;
        if (state_->owner != nullptr) {
            release_launch_event(*state_->owner, event);
        } else {
            cudaEventDestroy(event);
        }
        event = nullptr;
    };
    retire_event(state_->source_ready);
    for (cudaEvent_t& event : state_->phase_done) {
        retire_event(event);
    }
    retire_event(state_->signature_ready);
    for (cudaEvent_t& event : state_->signature_done) {
        retire_event(event);
    }
}

template <class T>
class PinnedHostVector {
  public:
    static_assert(std::is_trivially_copyable_v<T>);

    ~PinnedHostVector() {
        if (data_ != nullptr) cudaFreeHost(data_);
        for (T* retired : retired_) cudaFreeHost(retired);
    }
    PinnedHostVector() = default;
    PinnedHostVector(const PinnedHostVector&) = delete;
    PinnedHostVector& operator=(const PinnedHostVector&) = delete;

    void clear() noexcept { size_ = 0; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    const T* data() const noexcept { return data_; }
    std::span<const T> values() const noexcept {
        return {data_, size_};
    }

    void reserve(std::size_t required) {
        if (required <= capacity_) return;
        const std::size_t next = std::max(
            required, capacity_ == 0 ? std::size_t{8} : capacity_ * 2);
        T* replacement = nullptr;
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&replacement), next * sizeof(T)));
        if (data_ != nullptr && size_ != 0) {
            std::memcpy(replacement, data_, size_ * sizeof(T));
        }
        if (data_ != nullptr) {
            try {
                retired_.push_back(data_);
            } catch (const std::bad_alloc&) {
                cudaFreeHost(replacement);
                throw;
            }
        }
        data_ = replacement;
        capacity_ = next;
    }

    void push_back(const T& value) {
        reserve(size_ + 1);
        data_[size_++] = value;
    }

  private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
    std::vector<T*> retired_;
};

struct NotifyContext {
    PieRuntimeCallbacks runtime{};
    PieCompletion completion{};
    struct FinalizeEntry {
        struct EndpointUpdate {
            std::uint32_t slot = DeviceChannelRegistry::kBadSlot;
            std::uint64_t target = 0;
            std::uint64_t wait_id = 0;
            // Pinned word block, resolved at enqueue time on the scheduler
            // thread. The completion callback dereferences ONLY this stable
            // pointer (plan §7): registry vectors may be reallocated by a
            // concurrent register_endpoint, but the per-slot pinned block
            // lives until the channel's ordered close.
            std::uint64_t* words = nullptr;
        };

        PieTerminalCell* terminal_cell = nullptr;
        std::uint32_t* commit_host = nullptr;
        bool poison = false;
        std::vector<EndpointUpdate> published;
        std::vector<EndpointUpdate> consumed;
        std::vector<EndpointUpdate> poisoned;
    };
    Dispatch::Impl* impl = nullptr;
    std::vector<FinalizeEntry> entries;
    std::size_t entry_count = 0;
    PinnedHostVector<CommitBumpLane> commit_lanes;
    PinnedHostVector<HostChannelSettlementLane> settlement_lanes;
    std::vector<void*> copy_destinations;
    std::vector<const void*> copy_sources;
    std::vector<std::size_t> copy_sizes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> notifications;
    std::vector<std::shared_ptr<CallbackFence>> callback_fences;
    std::atomic<bool> in_use{false};
    cudaEvent_t copy_ready = nullptr;
    cudaEvent_t copy_done = nullptr;
    cudaEvent_t callback_done = nullptr;
    bool callback_pending = false;
    bool fire_timing_enabled = false;
    fire_timing::Clock::time_point fire_timing_started{};
    std::size_t fire_count = 0;
    std::uint64_t membership_hash = 0;

    ~NotifyContext() {
        if (copy_ready != nullptr) cudaEventDestroy(copy_ready);
        if (copy_done != nullptr) cudaEventDestroy(copy_done);
        if (callback_done != nullptr) cudaEventDestroy(callback_done);
    }

    FinalizeEntry& next_entry() {
        if (entry_count == entries.size()) entries.emplace_back();
        FinalizeEntry& entry = entries[entry_count++];
        entry.terminal_cell = nullptr;
        entry.commit_host = nullptr;
        entry.poison = false;
        entry.published.clear();
        entry.consumed.clear();
        entry.poisoned.clear();
        return entry;
    }

    void reset_for_submission() {
        runtime = {};
        completion = {};
        impl = nullptr;
        entry_count = 0;
        commit_lanes.clear();
        settlement_lanes.clear();
        copy_destinations.clear();
        copy_sources.clear();
        copy_sizes.clear();
        notifications.clear();
        callback_fences.clear();
        callback_pending = false;
        fire_timing_enabled = false;
        fire_timing_started = {};
        fire_count = 0;
        membership_hash = 0;
    }
};

Dispatch::Impl::~Impl() {
    if (d_fixed_decode_kills != nullptr) {
        cudaFree(d_fixed_decode_kills);
    }
    if (h_fixed_decode_kills != nullptr) {
        cudaFreeHost(h_fixed_decode_kills);
    }
    if (d_envelope_kills != nullptr) {
        cudaFree(d_envelope_kills);
    }
    if (h_envelope_kills != nullptr) {
        cudaFreeHost(h_envelope_kills);
    }
}

// Word-pointer variants of DeviceChannelRegistry::finalize_host_publish /
// finalize_host_consume for the completion callback: the callback must not
// index registry vectors (a concurrent register_endpoint may reallocate
// them), so it writes through the pinned word pointers precomputed at
// enqueue. Word layout: [0]=head, [1]=tail, [2]=poison, [3]=closed.
void finalize_publish_words(std::uint64_t* words, std::uint64_t target, bool failed) {
    if (words == nullptr) return;
    if (failed) {
        std::atomic_ref<std::uint64_t>(words[2]).store(
            target == 0 ? 1 : target, std::memory_order_release);
        return;
    }
    std::atomic_ref<std::uint64_t>(words[1]).store(target, std::memory_order_release);
    std::atomic_ref<std::uint64_t>(words[2]).store(0, std::memory_order_release);
}

void release_callback_fences(NotifyContext& context) noexcept {
    for (const auto& fence : context.callback_fences) {
        if (fence->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            fence->pending.notify_all();
        }
    }
    context.callback_fences.clear();
}

void CUDART_CB notify_runtime_callback(void* userdata) {
    auto* ctx = static_cast<NotifyContext*>(userdata);
    if (ctx == nullptr) return;
    const bool notify =
        ctx->runtime.notify != nullptr &&
        (ctx->impl == nullptr ||
         !ctx->impl->shutting_down.load(std::memory_order_acquire));
    ctx->notifications.clear();
    for (std::size_t index = 0; index < ctx->entry_count; ++index) {
        const auto& entry = ctx->entries[index];
        const bool committed =
            entry.commit_host != nullptr && *(entry.commit_host) != 0;
        const bool failed = entry.poison;
        const bool retry = !failed && !committed;
        if (retry && ctx->fire_timing_enabled) {
            // Bounded diagnostic: dump the retried lane's endpoint state so
            // an uncommitted pass names the gate that refused it (ring
            // expectation vs live words; rings all matching implicates the
            // envelope/fixed-decode kill path instead).
            static std::atomic<int> retry_dumps{0};
            if (retry_dumps.fetch_add(1, std::memory_order_relaxed) < 48) {
                std::string line;
                line.reserve(512);
                line += "[pie-fire-timing] {\"schema\":1,\"source\":\"driver\","
                        "\"event\":\"retry_lane\",\"endpoints\":[";
                bool first = true;
                auto dump = [&](const char* kind, const auto& updates) {
                    for (const auto& update : updates) {
                        if (update.words == nullptr) continue;
                        if (!first) line += ",";
                        first = false;
                        char buf[160];
                        std::snprintf(
                            buf, sizeof(buf),
                            "{\"kind\":\"%s\",\"slot\":%u,\"target\":%llu,"
                            "\"head\":%llu,\"tail\":%llu,\"poison\":%llu}",
                            kind, update.slot,
                            static_cast<unsigned long long>(update.target),
                            static_cast<unsigned long long>(
                                std::atomic_ref<std::uint64_t>(update.words[0])
                                    .load(std::memory_order_acquire)),
                            static_cast<unsigned long long>(
                                std::atomic_ref<std::uint64_t>(update.words[1])
                                    .load(std::memory_order_acquire)),
                            static_cast<unsigned long long>(
                                std::atomic_ref<std::uint64_t>(update.words[2])
                                    .load(std::memory_order_acquire)));
                        line += buf;
                    }
                };
                dump("publish", entry.published);
                dump("consume", entry.consumed);
                line += "]}";
                std::fprintf(stderr, "%s\n", line.c_str());
            }
        }
        if (committed) {
            for (const auto& update : entry.published) {
                const std::uint64_t actual =
                    std::atomic_ref<std::uint64_t>(update.words[1]).load(
                        std::memory_order_acquire);
                ctx->notifications.emplace_back(update.wait_id, actual);
            }
            for (const auto& update : entry.consumed) {
                const std::uint64_t actual =
                    std::atomic_ref<std::uint64_t>(update.words[0]).load(
                        std::memory_order_acquire);
                ctx->notifications.emplace_back(update.wait_id, actual);
            }
        }
        if (failed) {
            for (const auto& update : entry.poisoned) {
                finalize_publish_words(update.words, update.target, true);
                ctx->notifications.emplace_back(update.wait_id, update.target);
            }
        }
        if (entry.terminal_cell != nullptr) {
            entry.terminal_cell->reserved0 = 0;
            std::atomic_ref<std::uint32_t>(entry.terminal_cell->outcome).store(
                failed ? PIE_TERMINAL_OUTCOME_FAILED
                       : (retry ? PIE_TERMINAL_OUTCOME_RETRY
                                : PIE_TERMINAL_OUTCOME_SUCCESS),
                std::memory_order_release);
        }
    }
    const std::uint64_t finish_to_settle_us =
        ctx->fire_timing_enabled
            ? fire_timing::duration_us(
                  ctx->fire_timing_started,
                  fire_timing::Clock::now())
            : 0;
    const std::uint64_t settled_monotonic_ns =
        ctx->fire_timing_enabled ? fire_timing::monotonic_ns() : 0;
    if (notify) {
        for (const auto& [wait_id, epoch] : ctx->notifications) {
            if (wait_id != 0 && epoch != 0) {
                ctx->runtime.notify(ctx->runtime.ctx, wait_id, epoch);
            }
        }
    }
    // No native instance/channel state is touched after the batch wake: a woken
    // runtime thread may immediately close the instance.
    if (notify && ctx->completion.wait_id != 0) {
        ctx->runtime.notify(
            ctx->runtime.ctx, ctx->completion.wait_id, ctx->completion.target_epoch);
    }
    if (ctx->fire_timing_enabled) {
        fire_timing::enqueue_settled({
            .wave_id = ctx->completion.wait_id,
            .fire_count = ctx->fire_count,
            .membership_hash = ctx->membership_hash,
            .finish_to_settle_us = finish_to_settle_us,
            .settled_monotonic_ns = settled_monotonic_ns,
        });
    }
    release_callback_fences(*ctx);
    ctx->commit_lanes.clear();
    ctx->settlement_lanes.clear();
    ctx->in_use.store(false, std::memory_order_release);
}

namespace {
void close_bound_instance(
    Dispatch::Impl& s,
    std::uint64_t instance_id,
    bool retain_resources = true);

bool host_publish_destinations_overlap(
    const NotifyContext& context) {
    for (std::size_t left = 0;
         left < context.copy_destinations.size();
         ++left) {
        const auto left_begin = reinterpret_cast<std::uintptr_t>(
            context.copy_destinations[left]);
        const std::size_t left_bytes = context.copy_sizes[left];
        if (left_bytes >
            std::numeric_limits<std::uintptr_t>::max() - left_begin) {
            return true;
        }
        const auto left_end = left_begin + left_bytes;
        for (std::size_t right = left + 1;
             right < context.copy_destinations.size();
             ++right) {
            const auto right_begin = reinterpret_cast<std::uintptr_t>(
                context.copy_destinations[right]);
            const std::size_t right_bytes = context.copy_sizes[right];
            if (right_bytes >
                std::numeric_limits<std::uintptr_t>::max() - right_begin) {
                return true;
            }
            const auto right_end = right_begin + right_bytes;
            if (left_begin < right_end && right_begin < left_end) {
                return true;
            }
        }
    }
    return false;
}

bool can_batch_host_publish_copies(
    const NotifyContext& context,
    cudaStream_t stream) {
#if CUDART_VERSION >= 12080
    return stream != nullptr &&
        context.copy_destinations.size() > 1 &&
        !host_publish_destinations_overlap(context);
#else
    static_cast<void>(context);
    static_cast<void>(stream);
    return false;
#endif
}

void enqueue_host_publish_copies(
    NotifyContext& context,
    cudaStream_t stream,
    bool batched) {
    if (context.copy_destinations.empty()) return;
#if CUDART_VERSION >= 12080
    if (batched) {
        cudaMemcpyAttributes attributes{};
        attributes.srcAccessOrder =
            cudaMemcpySrcAccessOrderStream;
        attributes.flags = cudaMemcpyFlagDefault;
        std::size_t attributes_index = 0;
        constexpr std::size_t kChunk = 1024;
        for (std::size_t offset = 0;
             offset < context.copy_destinations.size();
             offset += kChunk) {
            const std::size_t count = std::min(
                kChunk,
                context.copy_destinations.size() - offset);
#if CUDART_VERSION >= 13000
            CUDA_CHECK(cudaMemcpyBatchAsync(
                context.copy_destinations.data() + offset,
                const_cast<const void**>(
                    context.copy_sources.data() + offset),
                context.copy_sizes.data() + offset,
                count,
                &attributes,
                &attributes_index,
                1,
                stream));
#else
            std::size_t failed = 0;
            CUDA_CHECK(cudaMemcpyBatchAsync(
                context.copy_destinations.data() + offset,
                const_cast<void**>(
                    context.copy_sources.data() + offset),
                context.copy_sizes.data() + offset,
                count,
                &attributes,
                &attributes_index,
                1,
                &failed,
                stream));
#endif
        }
        return;
    }
#else
    static_cast<void>(batched);
#endif
    for (std::size_t index = 0;
         index < context.copy_destinations.size();
         ++index) {
        CUDA_CHECK(cudaMemcpyAsync(
            context.copy_destinations[index],
            context.copy_sources[index],
            context.copy_sizes[index],
            cudaMemcpyDeviceToHost,
            stream));
    }
}

NotifyContext* acquire_notify_context(Dispatch::Impl& owner) {
    const auto try_acquire = [](NotifyContext& context) {
        bool available = false;
        return context.in_use.compare_exchange_strong(
            available, true, std::memory_order_acq_rel);
    };
    for (const auto& context : owner.settlement_arenas) {
        if (try_acquire(*context)) {
            context->reset_for_submission();
            return context.get();
        }
    }
    if (owner.settlement_arenas.size() <
        Dispatch::Impl::kMaxSettlementArenas) {
        auto context = std::make_unique<NotifyContext>();
        CUDA_CHECK(cudaEventCreateWithFlags(
            &context->copy_ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &context->copy_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &context->callback_done, cudaEventDisableTiming));
        context->in_use.store(true, std::memory_order_relaxed);
        NotifyContext* result = context.get();
        owner.settlement_arenas.push_back(std::move(context));
        result->reset_for_submission();
        return result;
    }

    for (const auto& context : owner.settlement_arenas) {
        if (!context->callback_pending) continue;
        CUDA_CHECK(cudaEventSynchronize(context->callback_done));
        if (try_acquire(*context)) {
            context->reset_for_submission();
            return context.get();
        }
    }
    throw std::runtime_error(
        "PTIR settlement arena capacity exceeded by concurrent callers");
}

class NotifyContextLease {
  public:
    NotifyContextLease(
        NotifyContext* context,
        cudaStream_t stream,
        cudaStream_t auxiliary_stream,
        std::unique_lock<std::mutex> lock)
        : context_(context),
          stream_(stream),
          auxiliary_stream_(auxiliary_stream),
          lock_(std::move(lock)) {}
    ~NotifyContextLease() {
        if (context_ != nullptr) {
            const cudaError_t auxiliary_status =
                auxiliary_stream_ == stream_
                ? cudaSuccess
                : cudaStreamSynchronize(auxiliary_stream_);
            const cudaError_t status =
                cudaStreamSynchronize(stream_);
            if (auxiliary_status == cudaSuccess &&
                status == cudaSuccess) {
                release_callback_fences(*context_);
                context_->in_use.store(false, std::memory_order_release);
            } else {
                std::fprintf(
                    stderr,
                    "[pie-driver-cuda] settlement cleanup stream sync failed: %s / %s\n",
                    cudaGetErrorString(auxiliary_status),
                    cudaGetErrorString(status));
            }
        }
    }
    NotifyContextLease(const NotifyContextLease&) = delete;
    NotifyContextLease& operator=(const NotifyContextLease&) = delete;
    void release() noexcept {
        context_ = nullptr;
        lock_.unlock();
    }

  private:
    NotifyContext* context_;
    cudaStream_t stream_;
    cudaStream_t auxiliary_stream_;
    std::unique_lock<std::mutex> lock_;
};

// Batch-level channel budget (§4.3 availability + reader capacity): members
// of one batch that share a channel are validated against the AGGREGATE of
// their planned ring consumes and reader publishes. Checked one-by-one, two
// members could both pass on the last available entry/slot and the second
// would die as a device-side poison instead of a synchronous rejection.
std::vector<DeviceHostChannelTicket> build_channel_tickets(
    const pie_native::LaunchView& view,
    std::size_t program,
    BoundInstance& bound,
    DeviceChannelRegistry& channels) {
    const std::size_t count = bound.trace->channels.size();
    const bool supplied =
        view.channel_ticket_indptr.size() ==
            view.ptir_program_instances.size() + 1 &&
        view.channel_expected_head.size() ==
            view.channel_expected_tail.size();
    if (!supplied) {
        throw std::runtime_error(
            "ptir launch requires runtime-assigned channel tickets");
    }
    std::size_t lo = 0;
    std::size_t hi = 0;
    lo = view.channel_ticket_indptr.data()[program];
    hi = view.channel_ticket_indptr.data()[program + 1];
    if (hi < lo || hi - lo != count ||
        hi > view.channel_expected_head.size()) {
        throw std::runtime_error(
            "ptir launch channel ticket segment does not match instance");
    }

    std::vector<DeviceHostChannelTicket> tickets;
    tickets.reserve(count);
    for (ChannelId dense = 0; dense < count; ++dense) {
        const std::uint32_t slot = bound.instance->view().slot(dense);
        const bool consumes = bound.instance->takes_channel(dense);
        const bool publishes = bound.instance->puts_channel(dense);
        std::uint64_t expected_head = kNoChannelTicket;
        std::uint64_t expected_tail = kNoChannelTicket;
        expected_head = view.channel_expected_head.data()[lo + dense];
        expected_tail = view.channel_expected_tail.data()[lo + dense];

        std::uint32_t flags = 0;
        if (consumes && expected_head != kNoChannelTicket) {
            flags |= kTicketConsume;
        }
        if (publishes && expected_tail != kNoChannelTicket) {
            flags |= kTicketPublish;
        }
        if (channels.host_role(slot) == PIE_CHANNEL_HOST_ROLE_WRITER &&
            !(channels.seed_credit(slot) && expected_head == 0)) {
            flags |= kTicketHostWriter;
        }
        if (channels.dtype(slot) == PIE_CHANNEL_DTYPE_BOOL) {
            flags |= kTicketPackedBool;
        }
        if (bound.instance->requires_channel_input(dense)) {
            flags |= kTicketRequireInput;
        }
        // Sequence-ticket APPLY hoisted to apply_lane_sequence_tickets
        // (W6): this builder runs in parallel across lanes and must not
        // mutate registry state; the applies run afterward in lane order.
        if ((flags & (kTicketConsume | kTicketPublish | kTicketRequireInput)) == 0) {
            continue;
        }
        tickets.push_back(DeviceHostChannelTicket{
            .slot = slot,
            .flags = flags,
            .expected_head = expected_head,
            .expected_tail = expected_tail,
            .words = channels.host_words(slot),
            .mirror = static_cast<const std::uint8_t*>(
                channels.host_mirror(slot)),
            .cells = static_cast<std::uint8_t*>(channels.cell_base(slot)),
            .cap1 = channels.capacity(slot) + 1,
            .wire_bytes = static_cast<std::uint32_t>(
                channels.wire_bytes(slot)),
            .native_bytes = static_cast<std::uint32_t>(
                channels.cell_bytes(slot)),
        });
    }
    return tickets;
}

// The mutation half of the old build_channel_tickets: advance each slot's
// host head/tail to the wire-assigned sequence. Serial, in lane order —
// byte-for-byte the order the fused builder produced (W6).
void apply_lane_sequence_tickets(
    const pie_native::LaunchView& view,
    std::size_t program,
    BoundInstance& bound,
    DeviceChannelRegistry& channels) {
    const std::size_t count = bound.trace->channels.size();
    const std::size_t lo = view.channel_ticket_indptr.data()[program];
    for (ChannelId dense = 0; dense < count; ++dense) {
        channels.apply_sequence_ticket(
            bound.instance->view().slot(dense),
            view.channel_expected_head.data()[lo + dense],
            view.channel_expected_tail.data()[lo + dense]);
    }
}

const DeviceHostChannelTicket* find_publish_ticket(
    const std::vector<DeviceHostChannelTicket>& tickets,
    std::uint32_t slot) {
    auto it = std::find_if(
        tickets.begin(), tickets.end(),
        [slot](const DeviceHostChannelTicket& ticket) {
            return ticket.slot == slot &&
                   (ticket.flags & kTicketPublish) != 0;
        });
    return it == tickets.end() ? nullptr : &*it;
}

std::uint32_t stage_mtp_rows(const plan::StagePlan* stage) {
    if (stage == nullptr) return 0;
    std::uint32_t next_value = 0;
    std::uint32_t rows = 0;
    for (const auto& normalized : stage->ops) {
        const auto& op = normalized.op;
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.intr == PTIR_INTR_MTP_LOGITS) {
            if (next_value >= stage->value_types.size()) {
                throw std::runtime_error(
                    "MtpLogits value is outside the region plan");
            }
            const auto& type = stage->value_types[next_value];
            if (type.dims.size() != 2 || type.dims[0].symbolic ||
                type.dims[0].value == 0) {
                throw std::runtime_error(
                    "MtpLogits requires a static non-empty draft-row extent");
            }
            if (rows != 0 && rows != type.dims[0].value) {
                throw std::runtime_error(
                    "one program declares incompatible MtpLogits row extents");
            }
            rows = type.dims[0].value;
        }
        next_value += op.results;
    }
    return rows;
}

std::uint32_t stage_logits_vocab(
    const plan::StagePlan* stage,
    std::uint32_t fallback) {
    if (stage == nullptr) return fallback;
    std::uint32_t next_value = 0;
    std::uint32_t logical_vocab = 0;
    for (const auto& normalized : stage->ops) {
        const auto& op = normalized.op;
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            (op.intr == PTIR_INTR_LOGITS ||
             op.intr == PTIR_INTR_MTP_LOGITS)) {
            if (next_value >= stage->value_types.size() ||
                stage->value_types[next_value].dims.empty()) {
                throw std::runtime_error(
                    "logits intrinsic has no planned vocabulary dimension");
            }
            const auto& dimension =
                stage->value_types[next_value].dims.back();
            if (dimension.symbolic || dimension.value == 0) {
                throw std::runtime_error(
                    "logits vocabulary dimension must be static");
            }
            if (logical_vocab != 0 &&
                logical_vocab != dimension.value) {
                throw std::runtime_error(
                    "program declares incompatible logits vocabularies");
            }
            logical_vocab = dimension.value;
        }
        next_value += op.results;
    }
    if (logical_vocab == 0) return fallback;
    if (logical_vocab > fallback) {
        throw std::runtime_error(
            "PTIR logical vocabulary exceeds the model row stride");
    }
    return logical_vocab;
}

bool stage_uses_intrinsic(
    const plan::StagePlan& stage,
    std::uint16_t intrinsic) {
    return std::any_of(
        stage.ops.begin(), stage.ops.end(),
        [intrinsic](const plan::NormalizedOp& normalized) {
            return normalized.op.tag == PTIR_OP_INTRINSIC_VAL &&
                normalized.op.intr == intrinsic;
        });
}

std::vector<std::uint32_t> channel_alias_topology(
    const plan::StagePlan& stage,
    PtirInstance& instance) {
    std::vector<std::uint32_t> topology;
    topology.reserve(stage.channel_bindings.size());
    std::vector<std::uint32_t> slots;
    slots.reserve(stage.channel_bindings.size());
    for (std::uint32_t dense : stage.channel_bindings) {
        const std::uint32_t slot = instance.view().slot(dense);
        auto found = std::find(slots.begin(), slots.end(), slot);
        if (found == slots.end()) {
            topology.push_back(static_cast<std::uint32_t>(slots.size()));
            slots.push_back(slot);
        } else {
            topology.push_back(
                static_cast<std::uint32_t>(found - slots.begin()));
        }
    }
    return topology;
}

void record_stage_channel_effects(
    StagedLane& lane,
    const plan::StagePlan& stage) {
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.chan < 0 ||
            (op.tag != PTIR_OP_CHAN_TAKE &&
             op.tag != PTIR_OP_CHAN_PUT)) {
            continue;
        }
        const std::uint32_t local = static_cast<std::uint32_t>(op.chan);
        if (local >= stage.channel_bindings.size()) continue;
        const std::uint32_t slot = lane.bound->instance->view().slot(
            stage.channel_bindings[local]);
        if (op.tag == PTIR_OP_CHAN_TAKE) {
            lane.prior_take_slots.insert(slot);
        } else {
            lane.prior_put_slots.insert(slot);
        }
    }
}

__global__ void cast_query_bf16_to_f32(
    const __nv_bfloat16* source,
    float* destination,
    std::size_t count) {
    for (std::size_t index =
             blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        destination[index] = __bfloat162float(source[index]);
    }
}

}  // namespace

Dispatch::Dispatch() : impl_(std::make_unique<Impl>()) {
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &impl_->output_copy_stream, cudaStreamNonBlocking));
    for (std::size_t index = 0; index < 2; ++index) {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &impl_->group_streams[index], cudaStreamNonBlocking));
    }
    for (cudaStream_t& stream : impl_->signature_streams) {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream, cudaStreamNonBlocking));
    }
}
DispatchStats Dispatch::stats() const {
    DispatchStats result;
    {
        std::lock_guard<std::mutex> lock(impl_->stats_mutex);
        result = impl_->stats;
    }
    // Fold in chain kills whose mirror landed after the last compose folded
    // them (the pinned word is written asynchronously on the launch stream).
    if (impl_->h_fixed_decode_kills != nullptr) {
        const std::uint32_t seen = *impl_->h_fixed_decode_kills;
        if (seen > impl_->fixed_decode_kills_reported) {
            result.fixed_decode_chain_kills +=
                seen - impl_->fixed_decode_kills_reported;
        }
    }
    if (impl_->h_envelope_kills != nullptr) {
        const std::uint32_t seen = *impl_->h_envelope_kills;
        if (seen > impl_->envelope_kills_reported) {
            result.decode_envelope_chain_kills +=
                seen - impl_->envelope_kills_reported;
        }
    }
    const auto generated = impl_->fused_modules.stats();
    result.generated_compilations = generated.compilations;
    result.generated_disk_hits = generated.disk_hits;
    result.generated_disk_writes = generated.disk_writes;
    result.generated_disk_errors = generated.disk_errors;
    result.generated_negative_hits = generated.negative_hits;
    result.generated_stage_cache_entries = generated.stage_entries;
    result.generated_program_cache_entries = generated.program_entries;
    result.generated_negative_cache_entries = generated.negative_entries;
    result.channel_slot_capacity = impl_->channels.capacity_slots();
    return result;
}

std::vector<std::uint32_t> Dispatch::mtp_draft_rows(
    const pie_native::LaunchView& view) const {
    std::vector<std::uint32_t> rows(view.ptir_program_hashes.size(), 0);
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) {
            throw std::runtime_error(
                "MtpLogits layout requested for an unregistered program");
        }
        for (const auto& stage : *plans) {
            const auto stage_rows = stage_mtp_rows(&stage);
            if (stage_rows == 0) continue;
            if (rows[program] != 0 && rows[program] != stage_rows) {
                throw std::runtime_error(
                    "program stages declare incompatible MtpLogits layouts");
            }
            rows[program] = stage_rows;
        }
    }
    return rows;
}

namespace {
// W4 reaper helpers — defined with the close machinery further down; the
// dtor needs them ahead of that block.
void wait_bound_instance_quiescent(
    Dispatch::Impl& s, BoundInstance& bound, bool wait_publications);
void reclaim_bound_instance(
    Dispatch::Impl& s, BoundInstance& bound, bool retain_resources);
}  // namespace

Dispatch::~Dispatch() {
    if (!impl_) return;
    impl_->shutting_down.store(true, std::memory_order_release);
    // W4: retire the exit reaper before anything it references. Items it
    // has not waited out yet are waited here (shutdown is synchronous by
    // design); items it finished are reclaimed like any lane drain.
    if (impl_->instance_reaper.joinable()) {
        {
            std::lock_guard<std::mutex> lock(impl_->reaper_mutex);
            impl_->reaper_stop = true;
        }
        impl_->reaper_cv.notify_one();
        impl_->instance_reaper.join();
        while (!impl_->reaper_queue.empty()) {
            Impl::InstanceReapItem item =
                std::move(impl_->reaper_queue.front());
            impl_->reaper_queue.pop_front();
            wait_bound_instance_quiescent(
                *impl_, item.bound, item.wait_publications);
            reclaim_bound_instance(*impl_, item.bound, false);
        }
        while (!impl_->reaped_ready.empty()) {
            BoundInstance bound = std::move(impl_->reaped_ready.front());
            impl_->reaped_ready.pop_front();
            reclaim_bound_instance(*impl_, bound, false);
        }
    }
    for (cudaStream_t stream : impl_->group_streams) {
        if (stream != nullptr) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    for (cudaStream_t stream : impl_->signature_streams) {
        if (stream != nullptr) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (impl_->output_copy_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(
            impl_->output_copy_stream));
    }
    impl_->generated_runtime.clear();
    CUDA_CHECK(cudaStreamSynchronize(
        sampling_ir::FrameCarrierEngine::instance().copy_stream()));
    for (const auto& context : impl_->settlement_arenas) {
        if (context->callback_pending) {
            CUDA_CHECK(cudaEventSynchronize(context->callback_done));
        }
    }
    while (!impl_->instances.empty()) {
        close_bound_instance(
            *impl_, impl_->instances.begin()->first, false);
    }
    for (cudaEvent_t event : impl_->available_publish_events) {
        if (event != nullptr) CUDA_CHECK(cudaEventDestroy(event));
    }
    for (cudaEvent_t event : impl_->available_launch_events) {
        if (event != nullptr) CUDA_CHECK(cudaEventDestroy(event));
    }
    for (const BoundInstance::CommitSnapshot& snapshot :
         impl_->available_commit_snapshots) {
        if (snapshot.device != nullptr) CUDA_CHECK(cudaFree(snapshot.device));
        if (snapshot.host != nullptr) CUDA_CHECK(cudaFreeHost(snapshot.host));
    }
    for (std::size_t index = 0; index < 2; ++index) {
        if (impl_->group_streams[index] != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(impl_->group_streams[index]));
        }
    }
    for (cudaStream_t& stream : impl_->signature_streams) {
        if (stream != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }
    if (impl_->output_copy_stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(impl_->output_copy_stream));
        impl_->output_copy_stream = nullptr;
    }
    if (impl_->publications_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(impl_->publications_done));
        impl_->publications_done = nullptr;
    }
}

namespace {

std::vector<ChannelValue> copy_seed_values(
    const std::vector<PieChannelValueDesc>& descs) {
    std::vector<ChannelValue> out;
    out.reserve(descs.size());
    for (const PieChannelValueDesc& desc : descs) {
        ChannelValue value;
        value.channel = desc.channel_id;
        if (desc.bytes.ptr != nullptr && desc.bytes.len > 0) {
            value.bytes.assign(desc.bytes.ptr, desc.bytes.ptr + desc.bytes.len);
        }
        out.push_back(std::move(value));
    }
    return out;
}

void ensure_event(cudaEvent_t* event) {
    if (*event == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
}

cudaEvent_t acquire_launch_event(Dispatch::Impl& s) {
    if (!s.available_launch_events.empty()) {
        cudaEvent_t event = s.available_launch_events.back();
        s.available_launch_events.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

// Return a launch event to the pool (lane thread; the event's last use is
// stream-ordered work the caller has already accounted for). Destroys past
// the retention cap — in practice the pool converges on the per-wave set.
void release_launch_event(Dispatch::Impl& s, cudaEvent_t event) {
    if (event == nullptr) return;
    if (s.available_launch_events.size() <
        Dispatch::Impl::kMaxRetainedInstanceResources) {
        try {
            s.available_launch_events.push_back(event);
            return;
        } catch (const std::bad_alloc&) {
        }
    }
    cudaEventDestroy(event);
}

BoundInstance::CommitSnapshot& commit_snapshot(
    Dispatch::Impl& owner,
    BoundInstance& bound,
    std::size_t index) {
    while (bound.commit_snapshots.size() <= index) {
        BoundInstance::CommitSnapshot snapshot{};
        if (!owner.available_commit_snapshots.empty()) {
            snapshot = owner.available_commit_snapshots.back();
            owner.available_commit_snapshots.pop_back();
        }
        try {
            if (snapshot.device == nullptr) {
                CUDA_CHECK(cudaMalloc(
                    reinterpret_cast<void**>(&snapshot.device),
                    sizeof(std::uint32_t)));
            }
            if (snapshot.host == nullptr) {
                int device = 0;
                cudaDeviceProp properties{};
                CUDA_CHECK(cudaGetDevice(&device));
                CUDA_CHECK(cudaGetDeviceProperties(
                    &properties, device));
                const bool try_mapping =
                    properties.canMapHostMemory != 0 &&
                    std::getenv(
                        "PIE_CUDA_DISABLE_MAPPED_COMMITS") == nullptr;
                if (try_mapping) {
                    const cudaError_t host_status = cudaHostAlloc(
                        reinterpret_cast<void**>(&snapshot.host),
                        sizeof(std::uint32_t),
                        cudaHostAllocMapped | cudaHostAllocPortable);
                    if (host_status != cudaSuccess &&
                        host_status != cudaErrorNotSupported) {
                        CUDA_CHECK(host_status);
                    }
                    if (host_status == cudaErrorNotSupported) {
                        static_cast<void>(cudaGetLastError());
                    }
                }
                if (snapshot.host != nullptr) {
                    const cudaError_t mapping_status =
                        cudaHostGetDevicePointer(
                            reinterpret_cast<void**>(
                                &snapshot.host_device),
                            snapshot.host,
                            0);
                    if (mapping_status != cudaSuccess) {
                        cudaFreeHost(snapshot.host);
                        snapshot.host = nullptr;
                        snapshot.host_device = nullptr;
                        if (mapping_status != cudaErrorNotSupported &&
                            mapping_status != cudaErrorInvalidValue) {
                            CUDA_CHECK(mapping_status);
                        }
                        static_cast<void>(cudaGetLastError());
                    }
                }
                if (snapshot.host == nullptr) {
                    CUDA_CHECK(cudaMallocHost(
                        reinterpret_cast<void**>(&snapshot.host),
                        sizeof(std::uint32_t)));
                }
            }
            bound.commit_snapshots.push_back(snapshot);
        } catch (...) {
            if (snapshot.host != nullptr) {
                cudaFreeHost(snapshot.host);
            }
            if (snapshot.device != nullptr) {
                cudaFree(snapshot.device);
            }
            throw;
        }
    }
    return bound.commit_snapshots[index];
}

// Lane-side reclamation of a WAITED-OUT instance (fence drained, events
// settled): pool returns and the PtirInstance drop (channel-view refcount
// release into the registry) — all single-enqueuer bookkeeping.
void reclaim_bound_instance(
    Dispatch::Impl& s,
    BoundInstance& bound,
    bool retain_resources) {
    if (bound.publish_done != nullptr) {
        bool retained = false;
        if (retain_resources &&
            s.available_publish_events.size() <
                Dispatch::Impl::kMaxRetainedInstanceResources) {
            try {
                s.available_publish_events.push_back(bound.publish_done);
                retained = true;
            } catch (const std::bad_alloc&) {
            }
        }
        if (!retained) {
            CUDA_CHECK(cudaEventDestroy(bound.publish_done));
        }
        bound.publish_done = nullptr;
    }
    for (BoundInstance::CommitSnapshot& snapshot : bound.commit_snapshots) {
        bool retained = false;
        if (retain_resources &&
            s.available_commit_snapshots.size() <
                Dispatch::Impl::kMaxRetainedInstanceResources) {
            try {
                s.available_commit_snapshots.push_back(snapshot);
                retained = true;
            } catch (const std::bad_alloc&) {
            }
        }
        if (!retained) {
            if (snapshot.device != nullptr) {
                CUDA_CHECK(cudaFree(snapshot.device));
            }
            if (snapshot.host != nullptr) {
                CUDA_CHECK(cudaFreeHost(snapshot.host));
            }
        }
        snapshot = {};
    }
    // `bound` drops at the caller: PtirInstance's ChannelView releases its
    // registry refcounts there (lane-owned bookkeeping, cheap post-W2).
}

// The reaper's half: block until nothing references the instance's
// resources. Runs OFF the lane; touches no registry or pool state.
void wait_bound_instance_quiescent(
    Dispatch::Impl& s,
    BoundInstance& bound,
    bool wait_publications) {
    for (std::uint32_t pending =
             bound.callback_fence->pending.load(std::memory_order_acquire);
         pending != 0;
         pending =
             bound.callback_fence->pending.load(std::memory_order_acquire)) {
        bound.callback_fence->pending.wait(pending, std::memory_order_acquire);
    }
    if (bound.publish_done != nullptr) {
        CUDA_CHECK(cudaEventSynchronize(bound.publish_done));
    }
    // After an instance's first completed wave its publications are ordered
    // by the shared per-wave event; a close must not outrun them. Syncing
    // the shared handle while the lane re-records it is thread-safe and at
    // worst conservative (waits for a later wave's record).
    if (wait_publications) {
        CUDA_CHECK(cudaEventSynchronize(s.publications_done));
    }
}

void ensure_instance_reaper(Dispatch::Impl& s) {
    if (s.instance_reaper.joinable()) return;
    s.instance_reaper = std::thread([&s]() {
        for (;;) {
            Dispatch::Impl::InstanceReapItem item;
            {
                std::unique_lock<std::mutex> lock(s.reaper_mutex);
                s.reaper_cv.wait(lock, [&s] {
                    return s.reaper_stop || !s.reaper_queue.empty();
                });
                if (s.reaper_queue.empty()) {
                    if (s.reaper_stop) return;
                    continue;
                }
                item = std::move(s.reaper_queue.front());
                s.reaper_queue.pop_front();
            }
            wait_bound_instance_quiescent(
                s, item.bound, item.wait_publications);
            {
                std::lock_guard<std::mutex> lock(s.reaper_mutex);
                s.reaped_ready.push_back(std::move(item.bound));
            }
        }
    });
}

// Lane entry-point drain: destroy every instance the reaper has finished
// waiting on. Called from begin/bind/close — a handful of pool pushes and
// refcount releases per reaped instance, never a blocking wait.
void drain_reaped_instances(Dispatch::Impl& s) {
    for (;;) {
        BoundInstance bound;
        {
            std::lock_guard<std::mutex> lock(s.reaper_mutex);
            if (s.reaped_ready.empty()) return;
            bound = std::move(s.reaped_ready.front());
            s.reaped_ready.pop_front();
        }
        reclaim_bound_instance(s, bound, /*retain_resources=*/true);
    }
}

void close_bound_instance(
    Dispatch::Impl& s,
    std::uint64_t instance_id,
    bool retain_resources) {
    auto it = s.instances.find(instance_id);
    if (it == s.instances.end()) return;
    if (retain_resources) {
        // Steady/exit path (W4): logical retire NOW (the id leaves the
        // map), the blocking waits go to the reaper, destruction returns
        // via drain_reaped_instances. The lane never blocks on a close.
        ensure_instance_reaper(s);
        Dispatch::Impl::InstanceReapItem item{
            std::move(it->second), s.publications_recorded};
        s.instances.erase(it);
        {
            std::lock_guard<std::mutex> lock(s.reaper_mutex);
            s.reaper_queue.push_back(std::move(item));
        }
        s.reaper_cv.notify_one();
        return;
    }
    // Shutdown path: synchronous, as before.
    BoundInstance bound = std::move(it->second);
    s.instances.erase(it);
    wait_bound_instance_quiescent(s, bound, s.publications_recorded);
    reclaim_bound_instance(s, bound, /*retain_resources=*/false);
}

}  // namespace

void Dispatch::reserve_channel_slots(std::uint32_t min_slots) {
    impl_->channels.reserve_slots(min_slots);
}

int Dispatch::register_program(std::uint64_t program_hash,
                                   pie_native::ByteSlice canonical,
                                   pie_native::ByteSlice sidecar,
                                   std::string* err) {
    if (err) err->clear();
    std::string derr;
    const Trace* trace = impl_->cache.get_or_decode(
        program_hash,
        reinterpret_cast<const std::uint8_t*>(canonical.ptr), canonical.size(),
        reinterpret_cast<const std::uint8_t*>(sidecar.ptr), sidecar.size(), &derr);
    if (trace == nullptr) {
        if (err) *err = derr;
        return PIE_STATUS_DRIVER_ERROR;
    }
    for (const Channel& channel : trace->channels) {
        const std::size_t cell_bytes =
            channel.type.shape.numel() * dtype_size(channel.type.dtype);
        if (channel.capacity >= kMaxRing ||
            cell_bytes == 0 ||
            cell_bytes > std::numeric_limits<std::uint32_t>::max()) {
            if (err) *err = "ptir program has an unsupported channel declaration";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    const auto* plans = impl_->cache.plans(program_hash);
    if (plans == nullptr || plans->empty()) {
        if (err) *err = "ptir program has no compiler region plans";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (const plan::StagePlan& stage : *plans) {
        if ((stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
             stage.stage == PTIR_STAGE_ON_ATTN) &&
            !impl_->attention_hook_coverage) {
            if (err) {
                *err =
                    "active CUDA model does not implement PTIR attention hooks";
            }
            return PIE_STATUS_UNSUPPORTED;
        }
        for (const plan::NormalizedOp& normalized : stage.ops) {
            const auto& op = normalized.op;
            if (op.tag == PTIR_OP_SINK_CALL) {
                if (err) {
                    *err =
                        "ptir model sinks are not implemented by the active CUDA model";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
            if (!grouped_supported_tag(op.tag)) {
                if (err) {
                    *err =
                        "ptir region plan contains an unsupported generic CUDA op";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
            if (op.tag != PTIR_OP_INTRINSIC_VAL) continue;
            const bool valid =
                (stage.stage == PTIR_STAGE_EPILOGUE &&
                 (op.intr == PTIR_INTR_LOGITS ||
                  op.intr == PTIR_INTR_MTP_LOGITS)) ||
                ((stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
                  stage.stage == PTIR_STAGE_ON_ATTN) &&
                 (op.intr == PTIR_INTR_QUERY ||
                  op.intr == PTIR_INTR_LAYER));
            if (!valid) {
                if (err) {
                    *err =
                        "ptir intrinsic is unavailable at its declared CUDA phase";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
        }
    }
    generated::CompileFailureKind compile_failure =
        generated::CompileFailureKind::None;
    std::string compile_error;
    const auto compiled_program = impl_->fused_modules.compile_program(
            program_hash,
            *plans,
            compile_failure,
            compile_error);
    if (compiled_program == nullptr) {
        if (err) *err = std::move(compile_error);
        return compile_failure == generated::CompileFailureKind::Deterministic
            ? PIE_STATUS_UNSUPPORTED
            : PIE_STATUS_DRIVER_ERROR;
    }
    if (compiled_program->stages.size() != plans->size()) {
        if (err) *err = "CUDA fused program stage count mismatch";
        return PIE_STATUS_UNSUPPORTED;
    }
    std::vector<std::pair<
        std::uint64_t,
        std::shared_ptr<const GroupedStageStaticPlan>>> staged_group_plans;
    for (std::size_t stage_index = 0;
         stage_index < plans->size();
         ++stage_index) {
        std::string availability_error;
        if (compiled_program->stages[stage_index] == nullptr ||
            !generated::generated_stage_supported(
                *compiled_program->stages[stage_index],
                (*plans)[stage_index],
                &availability_error)) {
            if (err) {
                *err =
                    "CUDA fused registration lacks complete coverage: " +
                    availability_error;
            }
            return PIE_STATUS_UNSUPPORTED;
        }
        const std::uint64_t runtime_id =
            compiled_program->stages[stage_index]->runtime_id;
        const bool already_staged = std::any_of(
            staged_group_plans.begin(),
            staged_group_plans.end(),
            [runtime_id](const auto& entry) {
                return entry.first == runtime_id;
            });
        if (!impl_->grouped_plans.contains(runtime_id) &&
            !already_staged) {
            auto group_plan = std::make_shared<GroupedStageStaticPlan>(
                (*plans)[stage_index]);
            if (!group_plan->valid) {
                if (err) {
                    *err =
                        "CUDA grouped registration lacks complete coverage: " +
                        group_plan->error;
                }
                return PIE_STATUS_UNSUPPORTED;
            }
            staged_group_plans.emplace_back(
                runtime_id, std::move(group_plan));
        }
    }
    for (auto& [runtime_id, group_plan] : staged_group_plans) {
        impl_->grouped_plans.emplace(
            runtime_id, std::move(group_plan));
    }
    return PIE_STATUS_OK;
}

int Dispatch::register_channel(
    const PieChannelDesc& channel,
    PieChannelEndpointBinding* binding,
    std::string* err) {
    if (err) err->clear();
    return impl_->channels.register_endpoint(channel, binding, err)
        ? PIE_STATUS_OK
        : PIE_STATUS_INVALID_ARGUMENT;
}

int Dispatch::bind_instance(std::uint64_t instance_id,
                                std::uint64_t program_hash,
                                std::uint32_t geometry_class,
                                std::uint64_t pacing_wait_id,
                                const std::vector<std::uint64_t>& channel_ids,
                                const std::vector<PieChannelValueDesc>& seed_values,
                                PieInstanceBinding* binding,
                                std::string* err) {
    if (err) err->clear();
    drain_reaped_instances(*impl_);
    // Stage timing (diagnostic, `PIE_FIRE_TIMING`): the engine-side bind
    // breakdown shows `driver_bind_us` — this whole call — at p50 5.7 ms /
    // p90 11 ms under load; the sections below name the payer inside.
    const bool bind_timing = fire_timing::full();
    const auto bind_t0 = bind_timing ? fire_timing::Clock::now()
                                     : fire_timing::Clock::time_point{};
    auto bind_mark = bind_t0;
    std::uint64_t bind_decode_us = 0;
    std::uint64_t bind_instance_us = 0;
    std::uint64_t bind_topology_us = 0;
    std::string derr;
    const Trace* trace = impl_->cache.get_or_decode(
        program_hash, nullptr, 0, nullptr, 0, &derr);
    if (bind_timing) {
        const auto now = fire_timing::Clock::now();
        bind_decode_us = fire_timing::duration_us(bind_mark, now);
        bind_mark = now;
    }
    if (trace == nullptr) {
        if (err) *err = derr;
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    // Classify once: the RUNTIME decides the class; the driver verifies only
    // the EXECUTION invariants the claimed class dereferences (never a
    // re-derivation — Host is the universal wire-driven path and needs none).
    switch (geometry_class) {
        case PIE_GEOMETRY_CLASS_HOST:
            break;
        case PIE_GEOMETRY_CLASS_DECODE_ENVELOPE:
            if (!is_decode_envelope_trace(*trace)) {
                if (err) {
                    *err = "ptir trace cannot execute as a decode envelope";
                }
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            break;
        case PIE_GEOMETRY_CLASS_DEVICE_GEOMETRY:
            if (!is_device_geometry_trace(*trace) &&
                !is_loop_carried_explicit_geometry_trace(*trace)) {
                if (err) {
                    *err = "ptir trace cannot execute with device-resolved "
                           "descriptor geometry";
                }
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            break;
        default:
            if (err) {
                *err = "unknown ptir geometry class " +
                    std::to_string(geometry_class);
            }
            return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (channel_ids.size() != trace->channels.size()) {
        if (err) *err = "ptir instance channel count does not match program";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    // The sampled-token routing mask is one bit per trace channel in a
    // 64-bit word (`sample_output_channel_mask`); a channel index past it
    // would silently fall out of sample routing, so refuse the bind loudly
    // instead (RV-21).
    if (trace->channels.size() > 64) {
        if (err) {
            *err = "ptir trace exceeds the 64-channel sample routing limit";
        }
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (impl_->instances.find(instance_id) != impl_->instances.end()) {
        if (err) *err = "ptir instance id is already bound";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    {
        std::unordered_set<std::uint64_t> unique_ids(
            channel_ids.begin(), channel_ids.end());
        if (unique_ids.size() != channel_ids.size()) {
            if (err) *err = "ptir instance channel ids must be unique";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    std::string ierr;
    if (bind_timing) bind_mark = fire_timing::Clock::now();
    auto inst = std::make_unique<PtirInstance>(
        *trace, &impl_->channels, channel_ids, copy_seed_values(seed_values), &ierr);
    if (bind_timing) {
        const auto now = fire_timing::Clock::now();
        bind_instance_us = fire_timing::duration_us(bind_mark, now);
        bind_mark = now;
    }
    if (!inst->ok()) {
        if (err) *err = ierr;
        return PIE_STATUS_INVALID_ARGUMENT;
    }

    BoundInstance bound;
    bound.program_hash = program_hash;
    bound.geometry_class = geometry_class;
    bound.pacing_wait_id = pacing_wait_id;
    bound.trace = trace;
    bound.channel_ids = channel_ids;
    bound.instance = std::move(inst);
    const auto* plans = impl_->cache.plans(program_hash);
    if (plans == nullptr) {
        if (err) *err = "ptir instance has no registered stage plans";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    bound.stage_topologies.reserve(plans->size());
    for (const plan::StagePlan& stage : *plans) {
        if (stage.stage > PTIR_STAGE_EPILOGUE) {
            if (err) *err = "ptir instance stage has an invalid phase";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        bound.stage_topologies.push_back(
            channel_alias_topology(stage, *bound.instance));
        bound.phase_plans[stage.stage].push_back(&stage);
    }
    if (bind_timing) {
        const auto now = fire_timing::Clock::now();
        bind_topology_us = fire_timing::duration_us(bind_mark, now);
        bind_mark = now;
    }
    if (!impl_->available_publish_events.empty()) {
        bound.publish_done = impl_->available_publish_events.back();
        impl_->available_publish_events.pop_back();
    } else {
        ensure_event(&bound.publish_done);
    }
    CUDA_CHECK(cudaEventRecord(
        bound.publish_done, sampling_ir::FrameCarrierEngine::instance().copy_stream()));
    if (bind_timing) {
        const auto now = fire_timing::Clock::now();
        std::ostringstream record;
        record << R"({"schema":1,"source":"cuda","event":"cuda_bind")"
               << R"(,"instance_id":)" << instance_id
               << R"(,"decode_us":)" << bind_decode_us
               << R"(,"instance_us":)" << bind_instance_us
               << R"(,"topology_us":)" << bind_topology_us
               << R"(,"event_us":)" << fire_timing::duration_us(bind_mark, now)
               << R"(,"total_us":)" << fire_timing::duration_us(bind_t0, now)
               << '}';
        fire_timing::write(record.str());
    }

    if (binding != nullptr) {
        std::memset(binding, 0, sizeof(*binding));
        binding->instance_id = instance_id;
        binding->geometry_class = geometry_class;
    }
    impl_->instances.emplace(instance_id, std::move(bound));
    return PIE_STATUS_OK;
}

void Dispatch::close_instance(std::uint64_t instance_id) {
    drain_reaped_instances(*impl_);
    close_bound_instance(*impl_, instance_id);
}

int Dispatch::close_channel(std::uint64_t channel_id, std::string* err) {
    if (err) err->clear();
    drain_reaped_instances(*impl_);
    const bool has_active_attachment = std::any_of(
        impl_->instances.begin(),
        impl_->instances.end(),
        [channel_id](const auto& entry) {
            const auto& ids = entry.second.channel_ids;
            return std::find(ids.begin(), ids.end(), channel_id) != ids.end();
        });
    return impl_->channels.close_endpoint(
               channel_id, err, !has_active_attachment)
        ? PIE_STATUS_OK
        : (impl_->channels.contains(channel_id)
               ? PIE_STATUS_INVALID_ARGUMENT
               : PIE_STATUS_CLOSED);
}

int Dispatch::validate_launch(
    const pie_native::LaunchView& view,
    std::string* err) {
    if (err) err->clear();
    const std::size_t count = view.ptir_program_hashes.size();
    if (count == 0 ||
        view.ptir_program_instances.size() != count ||
        view.terminal_cells.size() != count) {
        if (err) *err = "ptir launch has inconsistent program arrays";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    try {
        for (std::size_t program = 0; program < count; ++program) {
            const std::uint64_t instance_id =
                view.ptir_program_instances.data()[program];
            auto instance = impl_->instances.find(instance_id);
            if (instance == impl_->instances.end() ||
                instance->second.trace == nullptr ||
                instance->second.program_hash !=
                    view.ptir_program_hashes.data()[program]) {
                if (err) *err = "ptir launch references an incompatible instance";
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    } catch (const std::exception& error) {
        if (err) *err = error.what();
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

void Dispatch::set_attention_hook_coverage(
    bool supported,
    std::uint32_t model_layers) {
    impl_->attention_hook_coverage = supported;
    impl_->model_layers = supported ? model_layers : 0;
}

bool Dispatch::launch_has_attention_stages(
    const pie_native::LaunchView& view) const {
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) continue;
        if (std::any_of(
                plans->begin(), plans->end(),
                [](const plan::StagePlan& stage) {
                    return stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
                        stage.stage == PTIR_STAGE_ON_ATTN;
                })) {
            return true;
        }
    }
    return false;
}

bool Dispatch::has_decode_envelopes(
    const pie_native::LaunchView& view) const {
    if (view.ptir_program_instances.size() !=
        view.ptir_program_hashes.size()) {
        return false;
    }
    for (std::size_t program = 0;
         program < view.ptir_program_instances.size();
         ++program) {
        const auto instance = impl_->instances.find(
            view.ptir_program_instances.data()[program]);
        if (instance != impl_->instances.end() &&
            instance->second.trace != nullptr &&
            instance->second.program_hash ==
                view.ptir_program_hashes.data()[program] &&
            instance->second.geometry_class ==
                PIE_GEOMETRY_CLASS_DECODE_ENVELOPE) {
            return true;
        }
    }
    return false;
}

bool Dispatch::envelope_plan_page_bounds(
    const pie_native::LaunchView& view,
    std::span<const std::uint32_t> program_request_starts,
    std::span<const std::uint32_t> wire_kv_page_indptr,
    std::vector<std::uint32_t>& per_request_pages) const {
    const std::size_t programs = view.ptir_program_hashes.size();
    if (programs == 0 ||
        view.ptir_program_instances.size() != programs ||
        wire_kv_page_indptr.size() < 2) {
        return false;
    }
    const std::size_t requests = wire_kv_page_indptr.size() - 1;
    per_request_pages.assign(requests, 0);
    for (std::size_t request = 0; request < requests; ++request) {
        per_request_pages[request] =
            wire_kv_page_indptr[request + 1] - wire_kv_page_indptr[request];
    }
    const bool has_translation =
        view.kv_translation_indptr.size() == programs + 1;
    bool any = false;
    for (std::size_t program = 0; program < programs; ++program) {
        const auto found = impl_->instances.find(
            view.ptir_program_instances.data()[program]);
        if (found == impl_->instances.end() ||
            found->second.trace == nullptr) {
            continue;
        }
        const Trace& trace = *found->second.trace;
        if (found->second.geometry_class == PIE_GEOMETRY_CLASS_HOST) {
            continue;
        }
        const PortBinding* pages = nullptr;
        for (const PortBinding& binding : trace.ports) {
            if (!binding.is_const && binding.port == kPortPages) {
                pages = &binding;
                break;
            }
        }
        if (pages == nullptr ||
            pages->channel >= trace.channels.size()) {
            continue;
        }
        const auto& shape = trace.channels[pages->channel].type.shape;
        const std::size_t numel = shape.numel();
        const std::size_t program_lanes =
            shape.dims.size() == 2 && shape.dims[0] != 0 ? shape.dims[0] : 1;
        std::uint32_t bound = static_cast<std::uint32_t>(
            program_lanes == 0 ? numel : numel / program_lanes);
        if (has_translation) {
            const std::uint32_t begin =
                view.kv_translation_indptr.data()[program];
            const std::uint32_t end =
                view.kv_translation_indptr.data()[program + 1];
            if (end > begin) {
                bound = std::min(bound, end - begin);
            }
        }
        const std::size_t start =
            program < program_request_starts.size()
                ? program_request_starts[program]
                : program;
        for (std::size_t lane = 0; lane < program_lanes; ++lane) {
            const std::size_t request = start + lane;
            if (request >= requests) break;
            per_request_pages[request] =
                std::max(per_request_pages[request], bound);
            any = true;
        }
    }
    return any;
}

namespace {

std::uint64_t sample_output_channel_mask(
    const StagedLane& lane,
    const plan::StagePlan& stage) {
    if (lane.bound == nullptr || lane.bound->trace == nullptr) return 0;
    std::optional<std::uint32_t> token_channel;
    for (const PortBinding& binding : lane.bound->trace->ports) {
        if (!binding.is_const && binding.port == kPortEmbedTokens) {
            token_channel = binding.channel;
            break;
        }
    }
    if (!token_channel.has_value()) return 0;

    std::vector<std::uint32_t> bases(stage.ops.size(), 0);
    std::uint32_t value_count = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        bases[node] = value_count;
        value_count += stage.ops[node].op.results;
    }
    std::vector<std::uint32_t> aliases(value_count);
    for (std::uint32_t value = 0; value < value_count; ++value) {
        aliases[value] = value;
    }
    auto resolve = [&](std::uint32_t value) {
        while (value < aliases.size() && aliases[value] != value) {
            value = aliases[value];
        }
        return value;
    };
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        const auto& op = stage.ops[node].op;
        if ((op.tag == PTIR_OP_RESHAPE || op.tag == PTIR_OP_CAST) &&
            op.results == 1 && !op.args.empty() &&
            bases[node] < aliases.size() && op.args[0] < aliases.size()) {
            aliases[bases[node]] = resolve(op.args[0]);
        }
    }

    std::unordered_set<std::uint32_t> sampled_values;
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.tag != PTIR_OP_CHAN_PUT || op.chan < 0 ||
            op.args.empty()) {
            continue;
        }
        const auto local = static_cast<std::size_t>(op.chan);
        if (local < stage.channel_bindings.size() &&
            stage.channel_bindings[local] == *token_channel) {
            sampled_values.insert(resolve(op.args[0]));
        }
    }

    std::uint64_t mask = 0;
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.tag != PTIR_OP_CHAN_PUT || op.chan < 0 ||
            op.args.empty()) {
            continue;
        }
        const auto local = static_cast<std::uint32_t>(op.chan);
        if (local < 64 &&
            sampled_values.contains(resolve(op.args[0]))) {
            mask |= std::uint64_t{1} << local;
        }
    }
    return mask;
}

GroupedLaneBinding make_staged_binding(
    StagedLane& lane,
    const plan::StagePlan& stage,
    const float* logits_base,
    std::uint32_t logits_stride,
    const float* query_base,
    std::uint32_t query_columns,
    const std::uint32_t* layer_base) {
    const float* lane_query = nullptr;
    if (query_base != nullptr) {
        lane_query = query_base +
            static_cast<std::size_t>(lane.token_start) * query_columns;
    }
    return GroupedLaneBinding{
        .instance = lane.bound->instance.get(),
        .plan = &stage,
        .plan_identity = lane.plan_identities->at(
            static_cast<std::size_t>(&stage - lane.plans->data())),
        .tickets = &lane.tickets,
        .logits_base = logits_base,
        .query_base = lane_query,
        .layer_base = layer_base,
        .logits_bf16_rows = lane.logits_bf16_rows.empty()
            ? nullptr
            : &lane.logits_bf16_rows,
        .mtp_logits_bf16_rows = lane.mtp_logits_bf16_rows.empty()
            ? nullptr
            : &lane.mtp_logits_bf16_rows,
        .sample_output_channel_mask =
            sample_output_channel_mask(lane, stage),
        .row_valid = lane.row_valid,
        .row_valid_offset = lane.row_valid_offset,
        .prior_put_slots = &lane.prior_put_slots,
        .prior_take_slots = &lane.prior_take_slots,
        .commit_slot = lane.snapshot->device,
        .logits_row_offset = lane.row_offset,
        .logits_row_count = lane.sampled_rows,
        .row_count = lane.runtime_row_count,
        .token_count = lane.token_count,
        .kv_len = lane.kv_len,
        .page_count = lane.page_count,
        .query_len = lane.query_len,
        .key_len = lane.key_len,
        .vocab = lane.logical_vocab,
        .logits_stride = logits_stride,
        .program_index = static_cast<std::uint32_t>(lane.program),
    };
}

void execute_declared_phase(
    StagedLaunch::State& launch,
    std::uint8_t phase,
    const float* logits_base,
    std::uint32_t logits_stride,
    const float* query_base,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream,
    Dispatch::FinishBreakdown* breakdown = nullptr) {
    const bool probing = breakdown != nullptr;
    std::int64_t assemble_total = 0;
    std::int64_t group_total = 0;
    std::int64_t execute_total = 0;
    if (!launch.active || launch.failed) {
        throw std::runtime_error("PTIR staged launch is not active");
    }
    if (phase > PTIR_STAGE_EPILOGUE) {
        throw std::runtime_error("invalid PTIR execution phase");
    }
    if ((phase == PTIR_STAGE_ON_ATTN_PROJ ||
         phase == PTIR_STAGE_ON_ATTN) &&
        layer != launch.phase_invocations[phase]) {
        throw std::runtime_error(
            "PTIR model hook layer order is not exact");
    }
    ++launch.phase_invocations[phase];
    launch.stream = stream;
    const cudaStream_t source_stream = stream;
    const std::size_t bridge_index = phase % 2;
    const bool boundary_phase =
        phase == PTIR_STAGE_PROLOGUE ||
        phase == PTIR_STAGE_EPILOGUE;
    cudaStream_t execution_stream = boundary_phase
        ? source_stream
        : launch.owner->group_streams[bridge_index];
    const bool bridged =
        execution_stream != nullptr && execution_stream != source_stream;
    if (bridged) {
        CUDA_CHECK(cudaEventRecord(
            launch.source_ready, source_stream));
        CUDA_CHECK(cudaStreamWaitEvent(
            execution_stream, launch.source_ready, 0));
        stream = execution_stream;
    }
    struct StreamBridge {
        cudaEvent_t done = nullptr;
        cudaStream_t source = nullptr;
        cudaStream_t execution = nullptr;
        ~StreamBridge() {
            if (done == nullptr) return;
            cudaEventRecord(done, execution);
            cudaStreamWaitEvent(source, done, 0);
        }
    } bridge{
        bridged ? launch.phase_done[bridge_index] : nullptr,
        source_stream,
        execution_stream,
    };
    if (phase == PTIR_STAGE_ON_ATTN_PROJ ||
        phase == PTIR_STAGE_ON_ATTN) {
        CUDA_CHECK(cudaMemcpyAsync(
            launch.device_layer, &layer, sizeof(layer),
            cudaMemcpyHostToDevice, stream));
    }

    std::size_t max_occurrences = 0;
    for (const auto& lane : launch.lanes) {
        max_occurrences = std::max(
            max_occurrences, (*lane->phase_plans)[phase].size());
    }
    for (std::size_t occurrence = 0;
         occurrence < max_occurrences;
         ++occurrence) {
        const auto t_assemble_begin = probing
            ? fire_timing::Clock::now()
            : fire_timing::Clock::time_point{};
        struct Task {
            StagedLane* lane = nullptr;
            const plan::StagePlan* plan = nullptr;
            const generated::FusedStageExecutable* executable = nullptr;
            const GroupedStageStaticPlan* group_plan = nullptr;
            GroupedLaneBinding binding;
            const std::vector<std::uint32_t>* topology = nullptr;
            bool complete = false;
        };
        std::vector<Task> tasks;
        tasks.reserve(launch.lanes.size());
        for (auto& lane_ptr : launch.lanes) {
            StagedLane& lane = *lane_ptr;
            if (occurrence >=
                (*lane.phase_plans)[phase].size()) continue;
            const plan::StagePlan& stage =
                *(*lane.phase_plans)[phase][occurrence];
            if (stage.ops.empty()) continue;
            const std::size_t stage_index =
                static_cast<std::size_t>(&stage - lane.plans->data());
            if (lane.generated_program == nullptr ||
                stage_index >= lane.generated_program->stages.size()) {
                throw std::runtime_error(
                    "PTIR staged launch has no compiled fused stage");
            }
            if (stage_uses_intrinsic(stage, PTIR_INTR_QUERY)) {
                if (query_base == nullptr || query_columns == 0 ||
                    lane.token_count == kUnavailableGroupedExtent ||
                    lane.token_start > query_rows ||
                    lane.token_count > query_rows - lane.token_start) {
                    throw std::runtime_error(
                        "Query intrinsic is outside the current model query span");
                }
            }
            GroupedLaneBinding binding = make_staged_binding(
                lane, stage, logits_base, logits_stride,
                query_base, query_columns, launch.device_layer);
            std::uint32_t value_base = 0;
            for (const auto& normalized : stage.ops) {
                if (normalized.op.tag == PTIR_OP_INTRINSIC_VAL &&
                    normalized.op.intr == PTIR_INTR_QUERY) {
                    if (value_base >= stage.value_types.size() ||
                        grouped_numel(
                            stage.value_types[value_base], binding) >
                            static_cast<std::uint64_t>(lane.token_count) *
                                query_columns) {
                        throw std::runtime_error(
                            "Query intrinsic shape exceeds the current "
                            "program query tensor");
                    }
                }
                value_base += normalized.op.results;
            }
            tasks.push_back(Task{
                .lane = &lane,
                .plan = &stage,
                .executable =
                    lane.generated_program->stages[stage_index].get(),
                .group_plan = launch.owner->grouped_plans.at(
                    lane.generated_program->stages[stage_index]->runtime_id)
                    .get(),
                .binding = binding,
                .topology =
                    &lane.bound->stage_topologies.at(stage_index),
            });
        }

        const auto t_group_begin = probing
            ? fire_timing::Clock::now()
            : fire_timing::Clock::time_point{};
        if (probing) {
            assemble_total +=
                fire_timing::duration_us(t_assemble_begin, t_group_begin);
        }
        struct ExecutionGroup {
            Task* first = nullptr;
            std::vector<Task*> members;
            std::vector<GroupedLaneBinding> bindings;
        };
        std::vector<ExecutionGroup> groups;
        groups.reserve(tasks.size());
        for (std::size_t first_index = 0;
             first_index < tasks.size();
             ++first_index) {
            if (tasks[first_index].complete) continue;
            Task& first = tasks[first_index];
            std::vector<Task*> members;
            std::vector<GroupedLaneBinding> bindings;
            members.reserve(tasks.size() - first_index);
            bindings.reserve(tasks.size() - first_index);
            members.push_back(&first);
            bindings.push_back(first.binding);
            GroupedStageAccumulator accumulator(*first.group_plan);
            std::string reason;
            if (!accumulator.try_add(first.binding, &reason)) {
                throw std::runtime_error(
                    "PTRP stage is not executable by the generic CUDA backend: " +
                    reason);
            }
            for (std::size_t candidate = first_index + 1;
                 candidate < tasks.size();
                 ++candidate) {
                Task& next = tasks[candidate];
                if (next.complete ||
                    next.plan->signature_hash !=
                        first.plan->signature_hash ||
                    next.plan->signature != first.plan->signature ||
                    *next.topology != *first.topology) {
                    continue;
                }
                reason.clear();
                if (!accumulator.try_add(next.binding, &reason)) {
                    if (reason.find("shared") != std::string::npos) {
                        std::lock_guard<std::mutex> lock(
                            launch.owner->stats_mutex);
                        ++launch.owner->stats.shared_slot_exclusions;
                        ++launch.owner->stats.ordered_alias_launches;
                    }
                    continue;
                }
                bindings.push_back(next.binding);
                members.push_back(&next);
            }
            for (Task* member : members) member->complete = true;
            groups.push_back(ExecutionGroup{
                .first = &first,
                .members = std::move(members),
                .bindings = std::move(bindings),
            });
        }

        const GroupedExecutionOptions execution_options{
            .reset_commits = false,
            .pull_tickets = false,
            .finalize = false,
            .time_sections = probing,
        };
        auto execute_group = [&](ExecutionGroup& group,
                                 cudaStream_t target_stream) {
            Task& first = *group.first;
            std::string generated_reason;
            if (first.executable == nullptr ||
                !generated::generated_stage_supported(
                    *first.executable,
                    *first.plan,
                    &generated_reason)) {
                throw std::runtime_error(
                    "registered PTIR stage has no generated execution: " +
                    generated_reason);
            }
            GroupedLaunchResult result =
                generated::run_generated_stage(
                    group.bindings,
                    *first.executable,
                    launch.owner->generated_runtime,
                    target_stream,
                    execution_options);
            if (probing && result.t_build_us >= 0) {
                auto bump = [](std::int64_t& total, std::int64_t part) {
                    total = (total < 0 ? 0 : total) + part;
                };
                bump(breakdown->epilogue_exec_build_us, result.t_build_us);
                bump(breakdown->epilogue_exec_workspace_us,
                     result.t_workspace_us);
                bump(breakdown->epilogue_exec_upload_us,
                     result.t_upload_us);
                bump(breakdown->epilogue_exec_launch_us,
                     result.t_launch_us);
            }
            if (result.device_tickets != nullptr) {
                CUDA_CHECK(cudaFreeAsync(
                    result.device_tickets, target_stream));
            }
            const bool direct_bf16 = std::any_of(
                group.bindings.begin(), group.bindings.end(),
                [](const GroupedLaneBinding& binding) {
                    return binding.logits_bf16_rows != nullptr ||
                        binding.mtp_logits_bf16_rows != nullptr;
                });
            {
                std::lock_guard<std::mutex> lock(
                    launch.owner->stats_mutex);
                ++launch.owner->stats.generated_fused_groups;
                launch.owner->stats.generated_fused_body_launches +=
                    result.body_op_launches;
                launch.owner->stats.grouped_lanes +=
                    group.members.size();
                launch.owner->stats.grouped_body_op_launches +=
                    result.body_op_launches;
                if (direct_bf16) {
                    ++launch.owner->stats.direct_bf16_groups;
                }
                if (result.used_nucleus_library) {
                    ++launch.owner->stats.nucleus_library_groups;
                }
                if (result.used_selection_library) {
                    ++launch.owner->stats.selection_library_groups;
                }
                if (result.large_nucleus_scalable) {
                    ++launch.owner->stats.large_nucleus_scalable_groups;
                }
            }
            for (Task* member : group.members) {
                record_stage_channel_effects(
                    *member->lane, *member->plan);
            }
        };

        bool independent = groups.size() > 1;
        std::unordered_set<std::uint32_t> prior_group_slots;
        for (const auto& group : groups) {
            std::unordered_set<std::uint32_t> group_slots;
            for (const auto& binding : group.bindings) {
                group_slots.insert(
                    binding.instance->view().slots().begin(),
                    binding.instance->view().slots().end());
            }
            for (const std::uint32_t slot : group_slots) {
                if (prior_group_slots.contains(slot)) {
                    independent = false;
                }
            }
            prior_group_slots.insert(
                group_slots.begin(), group_slots.end());
        }
        const auto t_execute_begin = probing
            ? fire_timing::Clock::now()
            : fire_timing::Clock::time_point{};
        if (probing) {
            group_total +=
                fire_timing::duration_us(t_group_begin, t_execute_begin);
        }
        if (!independent) {
            for (auto& group : groups) execute_group(group, stream);
            if (probing) {
                execute_total += fire_timing::duration_us(
                    t_execute_begin, fire_timing::Clock::now());
            }
            continue;
        }

        if (launch.signature_ready == nullptr) {
            launch.signature_ready = acquire_launch_event(*launch.owner);
        }
        CUDA_CHECK(cudaEventRecord(launch.signature_ready, stream));
        const std::size_t used_streams = std::min(
            groups.size(),
            Dispatch::Impl::kSignatureStreamCount);
        for (std::size_t index = 0; index < used_streams; ++index) {
            if (launch.signature_done[index] == nullptr) {
                launch.signature_done[index] =
                    acquire_launch_event(*launch.owner);
            }
            CUDA_CHECK(cudaStreamWaitEvent(
                launch.owner->signature_streams[index],
                launch.signature_ready,
                0));
        }
        struct SignatureStreamJoin {
            StagedLaunch::State& launch;
            cudaStream_t source;
            std::size_t count;
            ~SignatureStreamJoin() {
                for (std::size_t index = 0; index < count; ++index) {
                    const cudaError_t record_status = cudaEventRecord(
                        launch.signature_done[index],
                        launch.owner->signature_streams[index]);
                    const cudaError_t wait_status =
                        record_status == cudaSuccess
                        ? cudaStreamWaitEvent(
                              source,
                              launch.signature_done[index],
                              0)
                        : record_status;
                    if (wait_status != cudaSuccess) {
                        std::fprintf(
                            stderr,
                            "[pie-driver-cuda] failed to rejoin PTIR "
                            "signature stream: %s\n",
                            cudaGetErrorString(wait_status));
                    }
                }
            }
        } signature_join{launch, stream, used_streams};
        for (std::size_t index = 0; index < groups.size(); ++index) {
            execute_group(
                groups[index],
                launch.owner->signature_streams[
                    index % used_streams]);
        }
        if (probing) {
            execute_total += fire_timing::duration_us(
                t_execute_begin, fire_timing::Clock::now());
        }
        {
            std::lock_guard<std::mutex> lock(
                launch.owner->stats_mutex);
            launch.owner->stats.overlapped_groups += groups.size();
        }
    }
    if (probing) {
        breakdown->epilogue_assemble_us = assemble_total;
        breakdown->epilogue_group_us = group_total;
        breakdown->epilogue_execute_us = execute_total;
    }
}

}  // namespace

std::unique_ptr<StagedLaunch> Dispatch::begin(
    const pie_native::LaunchView& view,
    cudaStream_t stream) {
    drain_reaped_instances(*impl_);
    const bool prologue_timing = fire_timing::full();
    const auto prologue_mark = prologue_timing
        ? fire_timing::Clock::now()
        : fire_timing::Clock::time_point{};
    // Structural validation only; the per-instance existence/hash checks
    // are folded into pass A below (W6: the separate validate_launch call
    // duplicated every instance map find).
    {
        const std::size_t programs = view.ptir_program_hashes.size();
        if (programs == 0 ||
            view.ptir_program_instances.size() != programs ||
            view.terminal_cells.size() != programs) {
            throw std::runtime_error(
                "ptir launch has inconsistent program arrays");
        }
    }
    auto launch = std::unique_ptr<StagedLaunch>(new StagedLaunch());
    StagedLaunch::State& state = *launch->state_;
    state.owner = impl_.get();
    state.view = view;
    state.stream = stream;
    CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&state.device_layer),
        sizeof(std::uint32_t),
        stream));
    state.source_ready = acquire_launch_event(*impl_);
    for (cudaEvent_t& event : state.phase_done) {
        event = acquire_launch_event(*impl_);
    }
    const std::size_t count = view.ptir_program_hashes.size();
    std::unordered_map<std::uint64_t, std::size_t> fire_counts;
    state.lanes.reserve(count);
    state.ticket_staging.reserve(view.channel_expected_head.size());
    state.pull_staging.reserve(count);
    // ONE publication-ordering wait for the whole wave (see
    // `Impl::publications_done`): the previous wave's publications all rode
    // the callback stream, so this single wait replaces the per-instance
    // event waits that used to cost ~2 host API calls per lane per wave.
    if (impl_->publications_recorded) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, impl_->publications_done, 0));
    }
    // Pass A (serial): everything ordering- or allocation-sensitive — map
    // lookups, snapshot allocation, CUDA event waits.
    const bool begin_timing = prologue_timing;
    auto begin_mark = begin_timing ? fire_timing::Clock::now()
                                   : fire_timing::Clock::time_point{};
    if (begin_timing) {
        launch->begin_breakdown_.prologue_us =
            fire_timing::duration_us(prologue_mark, begin_mark);
    }
    std::vector<std::unique_ptr<StagedLane>> pending_lanes(count);
    std::vector<std::uint32_t> pending_initial_commit(count, 0);
    // A wave overwhelmingly repeats one program (the bench: 256 lanes,
    // one hash) — memoize the three per-hash cache lookups instead of
    // paying 3·C hash-map probes (W6 pass-A hoist).
    std::uint64_t memo_hash = 0;
    bool memo_valid = false;
    const std::vector<plan::StagePlan>* memo_plans = nullptr;
    const std::vector<std::uint64_t>* memo_identities = nullptr;
    std::shared_ptr<const generated::FusedProgramExecutable> memo_generated;
    for (std::size_t program = 0; program < count; ++program) {
        const std::uint64_t instance_id =
            view.ptir_program_instances.data()[program];
        auto found = impl_->instances.find(instance_id);
        if (found == impl_->instances.end()) {
            throw std::runtime_error("PTIR launch references a missing instance");
        }
        BoundInstance& bound = found->second;
        if (bound.trace == nullptr ||
            bound.program_hash !=
                view.ptir_program_hashes.data()[program]) {
            throw std::runtime_error(
                "ptir launch references an incompatible instance");
        }
        auto lane = std::make_unique<StagedLane>();
        const std::size_t instance_occurrence = fire_counts[instance_id]++;
        lane->program = program;
        lane->bound = &bound;
        lane->snapshot =
            &commit_snapshot(*impl_, bound, instance_occurrence);
        if (!memo_valid || memo_hash != bound.program_hash) {
            memo_hash = bound.program_hash;
            memo_plans = impl_->cache.plans(memo_hash);
            memo_identities = impl_->cache.graph_stage_identities(memo_hash);
            memo_generated = impl_->fused_modules.program(memo_hash);
            memo_valid = true;
        }
        lane->plans = memo_plans;
        lane->plan_identities = memo_identities;
        lane->generated_program = memo_generated;
        if (lane->plans == nullptr || lane->plan_identities == nullptr ||
            lane->plan_identities->size() != lane->plans->size() ||
            lane->generated_program == nullptr ||
            lane->generated_program->stages.size() !=
                lane->plans->size()) {
            throw std::runtime_error("PTIR launch has no compiler region plans");
        }
        lane->phase_plans = &bound.phase_plans;
        // Per-instance ordering survives only for the bind-time seed
        // upload (recorded on the seed copy stream); after the instance's
        // first completed wave the event is retired and the shared
        // `publications_done` wait above carries the ordering.
        if (bound.publish_done != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, bound.publish_done, 0));
        }
        pending_initial_commit[program] =
            instance_occurrence == 0 ? 1u : 0u;
        pending_lanes[program] = std::move(lane);
    }
    if (begin_timing) {
        const auto now = fire_timing::Clock::now();
        launch->begin_breakdown_.pass_a_us =
            fire_timing::duration_us(begin_mark, now);
        begin_mark = now;
    }
    // Pass B (parallel, W6): the per-lane ticket builds are pure functions
    // of the view and bind-time-immutable registry arrays — the single
    // largest exclusive pool of the lane's serial host chain (~0.7 ms at
    // 256 lanes). The sequence-ticket APPLIES are hoisted to pass C so the
    // registry sees them in lane order exactly as before.
    {
        const std::function<void(std::size_t)> build_lane_tickets =
            [&](std::size_t program) {
                StagedLane& lane = *pending_lanes[program];
                lane.tickets = build_channel_tickets(
                    view, program, *lane.bound, impl_->channels);
            };
        impl_->lane_pool.parallel_for(count, build_lane_tickets);
    }
    if (begin_timing) {
        const auto now = fire_timing::Clock::now();
        launch->begin_breakdown_.tickets_us =
            fire_timing::duration_us(begin_mark, now);
        begin_mark = now;
    }
    // Pass C (serial): registry sequence applies in lane order, diagnostic
    // retry forcing, and the staging appends.
    for (std::size_t program = 0; program < count; ++program) {
        std::unique_ptr<StagedLane> lane = std::move(pending_lanes[program]);
        BoundInstance& bound = *lane->bound;
        apply_lane_sequence_tickets(
            view, program, bound, impl_->channels);
        const std::uint32_t initial_commit = pending_initial_commit[program];
        if (impl_->force_retry_launches_remaining.exchange(
                0, std::memory_order_relaxed) != 0) {
            bool forced = false;
            for (DeviceHostChannelTicket& ticket : lane->tickets) {
                if ((ticket.flags & kTicketConsume) != 0) {
                    ++ticket.expected_head;
                    forced = true;
                    break;
                }
                if ((ticket.flags & kTicketPublish) != 0) {
                    ++ticket.expected_tail;
                    forced = true;
                    break;
                }
            }
            if (!forced) {
                impl_->force_retry_launches_remaining.store(
                    1, std::memory_order_relaxed);
            }
        }
        const std::size_t max_ticket_count =
            std::numeric_limits<std::uint32_t>::max();
        if (state.ticket_staging.size() > max_ticket_count ||
            lane->tickets.size() >
                max_ticket_count - state.ticket_staging.size()) {
            throw std::runtime_error(
                "PTIR host channel ticket batch exceeds u32 capacity");
        }
        lane->device_ticket_offset =
            static_cast<std::uint32_t>(state.ticket_staging.size());
        lane->device_ticket_count =
            static_cast<std::uint32_t>(lane->tickets.size());
        state.ticket_staging.insert(
            state.ticket_staging.end(),
            lane->tickets.begin(),
            lane->tickets.end());
        state.pull_staging.push_back(PullValidateHostChannelLane{
            .full = bound.instance->view().d_full(),
            .pass_commit = lane->snapshot->device,
            .ticket_offset = lane->device_ticket_offset,
            .ticket_count = lane->device_ticket_count,
            .initial_commit = initial_commit,
        });
        state.touched_instances.push_back(
            view.ptir_program_instances.data()[program]);
        state.lanes.push_back(std::move(lane));
    }
    if (begin_timing) {
        const auto now = fire_timing::Clock::now();
        launch->begin_breakdown_.pass_c_us =
            fire_timing::duration_us(begin_mark, now);
        begin_mark = now;
    }
    state.device_tickets = launch_pull_validate_host_channels_batch(
        state.ticket_staging,
        state.pull_staging,
        stream);
    if (begin_timing) {
        launch->begin_breakdown_.pull_validate_us = fire_timing::duration_us(
            begin_mark, fire_timing::Clock::now());
    }
    if (state.device_tickets != nullptr) {
        for (auto& lane : state.lanes) {
            if (lane->device_ticket_count != 0) {
                lane->device_tickets =
                    state.device_tickets + lane->device_ticket_offset;
            }
        }
    }
    const bool stateful_rs = rs_launch_requires_readiness_settlement(
        view.rs_slot_ids.size(),
        view.rs_fold_lens.size(),
        view.rs_buffer_slot_ids.size(),
        view.rs_buffer_slot_indptr.size());
    auto settle_readiness = [&](const char* phase) {
        for (const auto& lane : state.lanes) {
            CUDA_CHECK(cudaMemcpyAsync(
                lane->snapshot->host,
                lane->snapshot->device,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost,
                stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (const auto& lane : state.lanes) {
            if (*lane->snapshot->host == 0) {
                throw RetryableLaunchError(
                    std::string("ptir ") + phase +
                    " readiness did not commit");
            }
        }
    };
    try {
        // Stateful model launches cannot discover a ticket miss after the
        // recurrent-state kernels have already mutated their slots. Settle the
        // host/device ticket pull before Prologue, then settle Prologue's own
        // channel readiness before returning to the model forward.
        if (stateful_rs) settle_readiness("channel ticket");
        execute_declared_phase(
            state, PTIR_STAGE_PROLOGUE,
            nullptr, 0, nullptr, 0, 0, 0, stream);
        if (stateful_rs) settle_readiness("prologue");
    } catch (...) {
        abort(*launch, stream);
        throw;
    }
    return launch;
}

void Dispatch::update_launch_geometry(
    StagedLaunch& launch,
    const pie_native::LaunchView& resolved_view,
    std::span<const std::uint32_t> program_token_starts) {
    StagedLaunch::State& state = *launch.state_;
    if (!state.active ||
        resolved_view.ptir_program_hashes.size() != state.lanes.size() ||
        program_token_starts.size() != state.lanes.size()) {
        throw std::runtime_error("invalid staged PTIR geometry update");
    }
    state.view = resolved_view;
    const std::size_t count = state.lanes.size();
    auto extent = [&](const pie_native::Slice<std::uint32_t>& values,
                      std::size_t program) {
        return values.size() == count
            ? values.data()[program]
            : kUnavailableGroupedExtent;
    };
    for (std::size_t program = 0; program < count; ++program) {
        StagedLane& lane = *state.lanes[program];
        lane.token_start = program_token_starts[program];
        if (resolved_view.ptir_sample_starts.size() == count &&
            resolved_view.ptir_sample_counts.size() == count) {
            lane.row_offset =
                resolved_view.ptir_sample_starts.data()[program];
            lane.sampled_rows =
                resolved_view.ptir_sample_counts.data()[program];
        } else if (resolved_view.sampling_indptr.size() == count + 1) {
            lane.row_offset =
                resolved_view.sampling_indptr.data()[program];
            lane.sampled_rows =
                resolved_view.sampling_indptr.data()[program + 1] -
                lane.row_offset;
        }
        lane.runtime_row_count =
            extent(resolved_view.ptir_row_counts, program);
        lane.token_count =
            extent(resolved_view.ptir_token_counts, program);
        lane.kv_len = extent(resolved_view.ptir_kv_lens, program);
        lane.page_count =
            extent(resolved_view.ptir_page_counts, program);
        lane.query_len =
            extent(resolved_view.ptir_query_lens, program);
        lane.key_len =
            extent(resolved_view.ptir_key_lens, program);
        for (const PortBinding& binding : lane.bound->trace->ports) {
            if (binding.is_const || !port_consumes(binding.port)) continue;
            lane.prior_take_slots.insert(
                lane.bound->instance->view().slot(binding.channel));
        }
    }
}

void Dispatch::execute_attention_phase(
    StagedLaunch& launch,
    std::uint8_t phase,
    const void* query_data,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream,
    bool query_is_f32) {
    if (phase != PTIR_STAGE_ON_ATTN_PROJ &&
        phase != PTIR_STAGE_ON_ATTN) {
        throw std::runtime_error("model hook invoked a non-attention PTIR phase");
    }
    StagedLaunch::State& state = *launch.state_;
    bool needs_query = false;
    for (const auto& lane : state.lanes) {
        for (const plan::StagePlan* stage :
             (*lane->phase_plans)[phase]) {
            needs_query =
                needs_query || stage_uses_intrinsic(*stage, PTIR_INTR_QUERY);
        }
    }
    float* query_f32 = nullptr;
    if (needs_query) {
        if (query_data == nullptr || query_rows == 0 || query_columns == 0 ||
            static_cast<std::size_t>(query_rows) >
                std::numeric_limits<std::size_t>::max() / query_columns) {
            throw std::runtime_error("model hook has no valid Query tensor");
        }
        const std::size_t count =
            static_cast<std::size_t>(query_rows) * query_columns;
        if (query_is_f32) {
            query_f32 = const_cast<float*>(
                static_cast<const float*>(query_data));
        } else {
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&query_f32),
                count * sizeof(float), stream));
            const std::uint32_t blocks = static_cast<std::uint32_t>(
                std::min<std::size_t>((count + 255) / 256, 65535));
            cast_query_bf16_to_f32<<<blocks, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(query_data),
                query_f32,
                count);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    try {
        execute_declared_phase(
            state, phase, nullptr, 0, query_f32,
            query_rows, query_columns, layer, stream);
    } catch (...) {
        if (query_f32 != nullptr && !query_is_f32) {
            cudaFreeAsync(query_f32, stream);
        }
        state.failed = true;
        throw;
    }
    if (query_f32 != nullptr && !query_is_f32) {
        CUDA_CHECK(cudaFreeAsync(query_f32, stream));
    }
}

bool Dispatch::finish(
    StagedLaunch& launch,
    const pie_native::LaunchView& view,
    const void* logits,
    std::uint32_t vocab,
    cudaStream_t stream,
    const PieRuntimeCallbacks* runtime,
    PieCompletion completion,
    const std::uint16_t* direct_bf16_logits,
    const std::uint32_t* direct_row_indices,
    std::span<const std::uint32_t> mtp_draft_row_starts,
    std::span<const std::uint32_t> mtp_draft_row_counts,
    std::uint32_t direct_bf16_row_capacity,
    const std::uint8_t* row_valid,
    std::span<const std::uint32_t> row_valid_offsets,
    FinishBreakdown* breakdown) {
    const bool trace_fire_timing = fire_timing::enabled();
    if (trace_fire_timing) {
        fire_timing::ensure_settlement_writer();
    }
    const auto fire_timing_started = trace_fire_timing
        ? fire_timing::Clock::now()
        : fire_timing::Clock::time_point{};
    StagedLaunch::State& state = *launch.state_;
    if (!state.active || state.failed ||
        state.lanes.size() != view.ptir_program_hashes.size()) {
        throw std::runtime_error("invalid PTIR staged finish");
    }
    state.view = view;
    state.stream = stream;
    const std::size_t program_count = state.lanes.size();
    if (row_valid != nullptr &&
        row_valid_offsets.size() != program_count) {
        throw std::runtime_error(
            "PTIR row-valid offsets do not match launched programs");
    }
    for (std::uint8_t phase :
         {std::uint8_t{PTIR_STAGE_ON_ATTN_PROJ},
          std::uint8_t{PTIR_STAGE_ON_ATTN}}) {
        const bool declared = std::any_of(
            state.lanes.begin(), state.lanes.end(),
            [phase](const auto& lane) {
                return !(*lane->phase_plans)[phase].empty();
            });
        if (declared &&
            state.phase_invocations[phase] != impl_->model_layers) {
            throw std::runtime_error(
                "PTIR attention phase did not execute at every model layer");
        }
    }
    for (std::size_t program = 0; program < program_count; ++program) {
        StagedLane& lane = *state.lanes[program];
        std::uint32_t logical_vocab = 0;
        std::uint32_t drafts = 0;
        for (const plan::StagePlan* stage :
             (*lane.phase_plans)[PTIR_STAGE_EPILOGUE]) {
            const std::uint32_t stage_vocab =
                stage_logits_vocab(stage, vocab);
            if (logical_vocab != 0 && stage_vocab != logical_vocab) {
                throw std::runtime_error(
                    "epilogue plans declare incompatible vocabularies");
            }
            logical_vocab = stage_vocab;
            const std::uint32_t stage_drafts = stage_mtp_rows(stage);
            if (stage_drafts != 0 && drafts != 0 &&
                stage_drafts != drafts) {
                throw std::runtime_error(
                    "epilogue plans declare incompatible MtpLogits rows");
            }
            drafts = std::max(drafts, stage_drafts);
        }
        lane.logical_vocab = logical_vocab == 0 ? vocab : logical_vocab;
        lane.logits_bf16_rows.clear();
        lane.mtp_logits_bf16_rows.clear();
        lane.row_valid = row_valid;
        lane.row_valid_offset =
            row_valid == nullptr ? 0 : row_valid_offsets[program];
        if (direct_bf16_logits != nullptr &&
            direct_row_indices != nullptr) {
            lane.logits_bf16_rows.reserve(lane.sampled_rows);
            for (std::uint32_t row = 0; row < lane.sampled_rows; ++row) {
                const std::uint32_t source =
                    direct_row_indices[lane.row_offset + row];
                if (direct_bf16_row_capacity != 0 &&
                    source >= direct_bf16_row_capacity) {
                    throw std::runtime_error(
                        "direct BF16 sampled row exceeds the logits layout");
                }
                lane.logits_bf16_rows.push_back(
                    reinterpret_cast<std::uint64_t>(
                        direct_bf16_logits +
                        static_cast<std::size_t>(source) * vocab));
            }
        }
        if (drafts != 0) {
            if (mtp_draft_row_starts.size() != program_count ||
                mtp_draft_row_counts.size() != program_count ||
                mtp_draft_row_counts[program] != drafts) {
                throw std::runtime_error(
                    "MtpLogits dedicated rows are unavailable");
            }
            const std::uint32_t start =
                mtp_draft_row_starts[program];
            if (start > direct_bf16_row_capacity ||
                drafts > direct_bf16_row_capacity - start) {
                throw std::runtime_error(
                    "MtpLogits dedicated rows exceed the logits layout");
            }
            if (direct_bf16_logits == nullptr) {
                throw std::runtime_error(
                    "generic staged MtpLogits requires direct BF16 rows");
            }
            lane.mtp_logits_bf16_rows.reserve(drafts);
            for (std::uint32_t row = 0; row < drafts; ++row) {
                lane.mtp_logits_bf16_rows.push_back(
                    reinterpret_cast<std::uint64_t>(
                        direct_bf16_logits +
                        static_cast<std::size_t>(start + row) * vocab));
            }
        }
    }
    try {
        execute_declared_phase(
            state,
            PTIR_STAGE_EPILOGUE,
            static_cast<const float*>(logits),
            vocab,
            nullptr,
            0,
            0,
            0,
            stream,
            trace_fire_timing ? breakdown : nullptr);
    } catch (...) {
        state.failed = true;
        throw;
    }
    const auto t_epilogue_done = (trace_fire_timing && breakdown != nullptr)
        ? fire_timing::Clock::now()
        : fire_timing::Clock::time_point{};
    cudaStream_t callback_stream = stream;
    std::unique_lock<std::mutex> settlement_lock(
        impl_->settlement_mutex);
    const auto t_lock_acquired = (trace_fire_timing && breakdown != nullptr)
        ? fire_timing::Clock::now()
        : fire_timing::Clock::time_point{};
    if (trace_fire_timing && breakdown != nullptr) {
        breakdown->epilogue_us =
            fire_timing::duration_us(fire_timing_started, t_epilogue_done);
        breakdown->settle_lock_us =
            fire_timing::duration_us(t_epilogue_done, t_lock_acquired);
    }
    NotifyContext* notify = acquire_notify_context(*impl_);
    NotifyContextLease notify_lease(
        notify,
        callback_stream,
        impl_->output_copy_stream,
        std::move(settlement_lock));
    if (runtime != nullptr) notify->runtime = *runtime;
    notify->completion = completion;
    notify->impl = impl_.get();
    notify->fire_timing_enabled = trace_fire_timing;
    notify->fire_timing_started = fire_timing_started;
    if (trace_fire_timing) {
        const auto logical_fire_ids =
            view.logical_fire_ids.as<std::uint64_t>();
        notify->fire_count = logical_fire_ids.size();
        notify->membership_hash =
            fire_timing::membership_hash(logical_fire_ids);
    }
    notify->commit_lanes.reserve(program_count);
    for (auto& lane_ptr : state.lanes) {
        StagedLane& lane = *lane_ptr;
        PtirInstance& instance = *lane.bound->instance;
        ChannelView& channel_view = instance.view();
        notify->commit_lanes.push_back(CommitBumpLane{
            .full = channel_view.d_full(),
            .head = channel_view.d_head(),
            .tail = channel_view.d_tail(),
            .cap1 = channel_view.d_cap1(),
            .taken = instance.commit_taken_device(),
            .taken_count = instance.commit_taken_count(),
            .put = instance.commit_put_device(),
            .put_count = instance.commit_put_count(),
            .commit = lane.snapshot->device,
        });
    }
    launch_commit_bump_batch(notify->commit_lanes.values(), stream);
    notify->settlement_lanes.reserve(program_count);
    for (auto& lane_ptr : state.lanes) {
        StagedLane& lane = *lane_ptr;
        BoundInstance& bound = *lane.bound;
        auto& entry = notify->next_entry();
        entry.terminal_cell =
            view.terminal_cells.data()[lane.program];
        entry.commit_host = lane.snapshot->host;
        entry.published.reserve(bound.trace->channels.size());
        entry.consumed.reserve(lane.tickets.size());
        entry.poisoned.reserve(bound.trace->channels.size());

        auto outputs = bound.instance->predict_outputs_device();
        HostChannelSettlementLane settlement{
            .full = bound.instance->view().d_full(),
            .head = bound.instance->view().d_head(),
            .cap1 = bound.instance->view().d_cap1(),
            .commit = lane.snapshot->device,
            .host_commit = lane.snapshot->host_device,
            .tickets = lane.device_tickets,
            .ticket_count = lane.device_ticket_count,
        };
        for (auto& output : outputs) {
            const DeviceHostChannelTicket* ticket =
                find_publish_ticket(lane.tickets, output.slot);
            if (ticket == nullptr) continue;
            if (settlement.consume.n ==
                kMaxConditionalConsumeChannels) {
                throw std::runtime_error(
                    "PTIR host output count exceeds settlement capacity");
            }
            output.device_ptr = ticket->cells +
                static_cast<std::size_t>(
                    ticket->expected_tail % ticket->cap1) *
                    ticket->native_bytes;
            const PreparedHostPublish publish =
                impl_->channels.prepare_host_publish_at(
                    output.slot,
                    ticket->expected_tail,
                    output.device_ptr,
                    callback_stream);
            entry.published.push_back({
                .slot = output.slot,
                .target = publish.target_tail,
                .wait_id = impl_->channels.reader_wait_id(output.slot),
                .words = impl_->channels.host_words(output.slot),
            });
            notify->copy_destinations.push_back(
                publish.destination);
            notify->copy_sources.push_back(publish.source);
            notify->copy_sizes.push_back(publish.bytes);
            settlement.consume.slots[settlement.consume.n++] =
                output.slot;
        }
        for (const DeviceHostChannelTicket& ticket : lane.tickets) {
            if ((ticket.flags & (kTicketConsume | kTicketHostWriter)) !=
                (kTicketConsume | kTicketHostWriter)) {
                continue;
            }
            entry.consumed.push_back({
                .slot = ticket.slot,
                .target = ticket.expected_head + 1,
                .wait_id = impl_->channels.writer_wait_id(ticket.slot),
                .words = ticket.words,
            });
        }
        notify->settlement_lanes.push_back(settlement);
        if (lane.snapshot->host_device == nullptr) {
            notify->copy_destinations.push_back(
                lane.snapshot->host);
            notify->copy_sources.push_back(
                lane.snapshot->device);
            notify->copy_sizes.push_back(
                sizeof(std::uint32_t));
        }
        for (std::size_t channel = 0;
             channel < bound.trace->channels.size();
             ++channel) {
            if (!bound.trace->channels[channel].host_visible) continue;
            const std::uint32_t slot =
                impl_->channels.slot_for(bound.channel_ids[channel]);
            if (slot == DeviceChannelRegistry::kBadSlot) continue;
            entry.poisoned.push_back({
                .slot = slot,
                .target = impl_->channels.poison_target(slot),
                .wait_id = impl_->channels.host_wait_id(slot),
                .words = impl_->channels.host_words(slot),
            });
        }
    }
    const bool batch_copies = can_batch_host_publish_copies(
        *notify, impl_->output_copy_stream);
    cudaStream_t settlement_stream = callback_stream;
    if (batch_copies) {
        CUDA_CHECK(cudaEventRecord(
            notify->copy_ready, callback_stream));
        CUDA_CHECK(cudaStreamWaitEvent(
            impl_->output_copy_stream,
            notify->copy_ready,
            0));
        settlement_stream = impl_->output_copy_stream;
    }
    enqueue_host_publish_copies(
        *notify, settlement_stream, batch_copies);
    launch_settle_host_channels_batch(
        notify->settlement_lanes.values(), settlement_stream);
    if (state.device_tickets != nullptr) {
        CUDA_CHECK(cudaFreeAsync(
            state.device_tickets, settlement_stream));
        state.device_tickets = nullptr;
        for (auto& lane : state.lanes) {
            lane->device_tickets = nullptr;
        }
    }
    if (batch_copies) {
        CUDA_CHECK(cudaEventRecord(
            notify->copy_done, settlement_stream));
        CUDA_CHECK(cudaStreamWaitEvent(
            callback_stream,
            notify->copy_done,
            0));
    }
    std::sort(
        state.touched_instances.begin(),
        state.touched_instances.end());
    state.touched_instances.erase(
        std::unique(
            state.touched_instances.begin(),
            state.touched_instances.end()),
        state.touched_instances.end());
    if (state.device_layer != nullptr) {
        CUDA_CHECK(cudaFreeAsync(state.device_layer, stream));
        state.device_layer = nullptr;
    }
    notify->callback_fences.reserve(
        state.touched_instances.size());
    for (std::uint64_t instance_id : state.touched_instances) {
        auto found = impl_->instances.find(instance_id);
        if (found != impl_->instances.end()) {
            // The instance's bind-time seed event is consumed: from here
            // its publication ordering rides the shared per-wave event
            // below. (Retire to the pool; close handles a null event.)
            if (found->second.publish_done != nullptr) {
                if (impl_->available_publish_events.size() <
                    Impl::kMaxRetainedInstanceResources) {
                    impl_->available_publish_events.push_back(
                        found->second.publish_done);
                } else {
                    CUDA_CHECK(cudaEventDestroy(
                        found->second.publish_done));
                }
                found->second.publish_done = nullptr;
            }
            notify->callback_fences.push_back(
                found->second.callback_fence);
            found->second.callback_fence->pending.fetch_add(
                1, std::memory_order_acq_rel);
        }
    }
    // ONE publication-ordering record for the whole wave (replaces the
    // per-instance records): callback_stream has already been joined with
    // the copy stream above, so this point covers every publication.
    ensure_event(&impl_->publications_done);
    CUDA_CHECK(cudaEventRecord(impl_->publications_done, callback_stream));
    impl_->publications_recorded = true;
    const cudaError_t callback_status = cudaLaunchHostFunc(
        settlement_stream, notify_runtime_callback, notify);
    if (callback_status != cudaSuccess) {
        CUDA_CHECK(callback_status);
    }
    const cudaError_t event_status = cudaEventRecord(
        notify->callback_done, settlement_stream);
    if (event_status == cudaSuccess) {
        notify->callback_pending = true;
    } else {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] settlement callback event record failed: %s\n",
            cudaGetErrorString(event_status));
        const cudaError_t sync_status =
            cudaStreamSynchronize(settlement_stream);
        if (sync_status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] settlement callback drain failed: %s\n",
                cudaGetErrorString(sync_status));
        }
    }
    if (trace_fire_timing && breakdown != nullptr) {
        breakdown->settle_prep_us = fire_timing::duration_us(
            t_lock_acquired, fire_timing::Clock::now());
    }
    notify_lease.release();
    state.active = false;
    return true;
}

void Dispatch::abort(
    StagedLaunch& launch,
    cudaStream_t stream) noexcept {
    if (!launch.state_ || !launch.state_->active) return;
    StagedLaunch::State& state = *launch.state_;
    const std::uint32_t zero = 0;
    for (auto& lane : state.lanes) {
        if (lane->snapshot != nullptr &&
            lane->snapshot->device != nullptr) {
            cudaMemcpyAsync(
                lane->snapshot->device,
                &zero,
                sizeof(zero),
                cudaMemcpyHostToDevice,
                stream);
        }
        if (lane->bound != nullptr &&
            lane->bound->publish_done != nullptr) {
            cudaEventRecord(lane->bound->publish_done, stream);
        }
    }
    // Instances whose seed events were already retired order through the
    // shared per-wave event; re-arm it on the abort stream so a retried
    // fire also orders after this cleanup. The abort stream waited on the
    // previous record at begin, so this point transitively covers it.
    if (state.owner != nullptr &&
        state.owner->publications_done != nullptr) {
        cudaEventRecord(state.owner->publications_done, stream);
        state.owner->publications_recorded = true;
    }
    if (state.device_tickets != nullptr) {
        cudaFreeAsync(state.device_tickets, stream);
        state.device_tickets = nullptr;
        for (auto& lane : state.lanes) {
            lane->device_tickets = nullptr;
        }
    }
    if (state.device_layer != nullptr) {
        cudaFreeAsync(state.device_layer, stream);
        state.device_layer = nullptr;
    }
    state.stream = stream;
    state.failed = true;
    state.active = false;
}

bool Dispatch::run(
    const pie_native::LaunchView& view,
    const void* logits,
    std::uint32_t vocab,
    cudaStream_t stream,
    const PieRuntimeCallbacks* runtime,
    PieCompletion completion,
    const std::uint16_t* direct_bf16_logits,
    const std::uint32_t* direct_row_indices,
    std::span<const std::uint32_t> mtp_draft_row_starts,
    std::span<const std::uint32_t> mtp_draft_row_counts,
    std::uint32_t direct_bf16_row_capacity) {
    if (view.ptir_program_hashes.empty()) {
        if (runtime != nullptr && runtime->notify != nullptr &&
            completion.wait_id != 0) {
            runtime->notify(
                runtime->ctx,
                completion.wait_id,
                completion.target_epoch);
        }
        return false;
    }
    auto launch = begin(view, stream);
    try {
        if (launch_has_attention_stages(view)) {
            throw std::runtime_error(
                "PTIR attention stages require launch-scoped model hooks");
        }
        std::vector<std::uint32_t> token_starts(
            view.ptir_program_hashes.size(), 0);
        if (view.ptir_token_counts.size() == token_starts.size()) {
            std::uint32_t cursor = 0;
            for (std::size_t program = 0;
                 program < token_starts.size();
                 ++program) {
                token_starts[program] = cursor;
                const std::uint32_t count =
                    view.ptir_token_counts.data()[program];
                if (count != kUnavailableGroupedExtent) cursor += count;
            }
        }
        update_launch_geometry(*launch, view, token_starts);
        return finish(
            *launch,
            view,
            logits,
            vocab,
            stream,
            runtime,
            completion,
            direct_bf16_logits,
            direct_row_indices,
            mtp_draft_row_starts,
            mtp_draft_row_counts,
            direct_bf16_row_capacity);
    } catch (...) {
        abort(*launch, stream);
        throw;
    }
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
Dispatch::settle_failed_launch(
    const pie_native::LaunchView& view,
    cudaStream_t execution_stream) {
    const cudaError_t execution_status =
        cudaStreamSynchronize(execution_stream);
    if (execution_status != cudaSuccess) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] failed launch stream synchronization: %s\n",
            cudaGetErrorString(execution_status));
    }
    cudaStream_t callback_stream =
        sampling_ir::FrameCarrierEngine::instance().copy_stream();
    if (callback_stream != nullptr && callback_stream != execution_stream) {
        const cudaError_t status = cudaStreamSynchronize(callback_stream);
        if (status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] failed launch callback synchronization: %s\n",
                cudaGetErrorString(status));
        }
    }

    Impl& s = *impl_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> notifications;
    for (std::size_t p = 0; p < view.ptir_program_instances.size(); ++p) {
        const std::uint64_t instance_id =
            view.ptir_program_instances.data()[p];
        auto it = s.instances.find(instance_id);
        if (it == s.instances.end()) continue;
        BoundInstance& bound = it->second;
        for (std::size_t c = 0; c < bound.trace->channels.size(); ++c) {
            if (!bound.trace->channels[c].host_visible) continue;
            const std::uint32_t slot =
                s.channels.slot_for(bound.channel_ids[c]);
            if (slot != DeviceChannelRegistry::kBadSlot) {
                const std::uint64_t poison_epoch =
                    s.channels.poison_target(slot);
                s.channels.finalize_host_publish(slot, poison_epoch, true);
                notifications.emplace_back(
                    s.channels.host_wait_id(slot), poison_epoch);
            }
        }
    }
    return notifications;
}

bool Dispatch::enqueue_decode_envelopes(
    const pie_native::LaunchView& view,
    std::span<const std::uint32_t> program_token_starts,
    std::span<const std::uint32_t> program_request_starts,
    std::span<const std::uint32_t> template_kv_page_indptr,
    const DecodeEnvelopeDeviceBuffers& buffers,
    std::string* err,
    StagedLaunch& launch) {
    if (err != nullptr) err->clear();
    const auto fail = [err](const char* message) {
        if (err != nullptr) *err = message;
        return false;
    };
    Impl& state = *impl_;
    StagedLaunch::State& staged = *launch.state_;
    const std::size_t programs = view.ptir_program_hashes.size();
    if (!staged.active || staged.lanes.size() != programs ||
        program_token_starts.size() != programs ||
        program_request_starts.size() != programs ||
        template_kv_page_indptr.size() < 2 ||
        buffers.token_ids == nullptr ||
        buffers.position_ids == nullptr ||
        buffers.kv_page_indices == nullptr ||
        buffers.kv_page_indptr == nullptr ||
        buffers.kv_last_page_lens == nullptr ||
        buffers.row_valid == nullptr ||
        buffers.page_size == 0) {
        return fail("decode envelope: malformed batch inputs");
    }
    if (view.ptir_kv_write_lower_bounds.size() != programs ||
        view.ptir_kv_write_upper_bounds.size() != programs) {
        return fail(
            "decode envelope: KV write containment bounds missing");
    }

    std::vector<DecodeEnvelopeLane> lanes(
        template_kv_page_indptr.size() - 1);
    if (lanes.size() > kDecodeEnvelopeMaxLanes) {
        return fail("decode envelope: batch exceeds the lane limit");
    }
    for (std::size_t request = 0; request < lanes.size(); ++request) {
        const std::uint32_t begin =
            template_kv_page_indptr[request];
        const std::uint32_t end =
            template_kv_page_indptr[request + 1];
        if (end < begin) {
            return fail(
                "decode envelope: template page indptr not monotonic");
        }
        lanes[request] = DecodeEnvelopeLane{
            .request_index = static_cast<std::uint32_t>(request),
            .source_page_begin = begin,
            .source_page_count = end - begin,
            .passthrough = 1,
        };
    }
    std::size_t envelope_lanes = 0;
    for (std::size_t program = 0; program < programs; ++program) {
        const std::uint64_t instance_id =
            view.ptir_program_instances.data()[program];
        auto found = state.instances.find(instance_id);
        if (found == state.instances.end()) {
            return fail("decode envelope: instance not bound");
        }
        if (found->second.geometry_class !=
            PIE_GEOMETRY_CLASS_DECODE_ENVELOPE) {
            continue;
        }
        if (found->second.trace == nullptr) {
            return fail("decode envelope: instance has no trace");
        }
        const PortBinding* token = nullptr;
        const PortBinding* position = nullptr;
        const PortBinding* embed_indptr = nullptr;
        for (const PortBinding& binding :
             found->second.trace->ports) {
            if (binding.port == kPortEmbedIndptr) {
                embed_indptr = &binding;
            } else if (!binding.is_const) {
                if (binding.port == kPortEmbedTokens) {
                    token = &binding;
                } else if (binding.port == kPortPositions) {
                    position = &binding;
                }
            }
        }
        if (token == nullptr) {
            return fail("decode envelope: no channel-fed token port");
        }
        if (token->channel >= found->second.trace->channels.size()) {
            return fail("decode envelope: token channel out of range");
        }
        const auto& token_shape =
            found->second.trace->channels[token->channel].type.shape.dims;
        if (token_shape.size() != 1 || token_shape[0] == 0) {
            return fail("decode envelope: token channel is not rank-1");
        }
        std::vector<std::uint32_t> qo_indptr;
        if (embed_indptr == nullptr) {
            qo_indptr = {0, token_shape[0]};
        } else if (embed_indptr->is_const) {
            if (embed_indptr->const_data.size() % sizeof(std::uint32_t) != 0) {
                return fail(
                    "decode envelope: const EmbedIndptr is not u32-aligned");
            }
            qo_indptr.resize(
                embed_indptr->const_data.size() / sizeof(std::uint32_t));
            std::memcpy(
                qo_indptr.data(),
                embed_indptr->const_data.data(),
                embed_indptr->const_data.size());
        } else {
            qo_indptr.resize(static_cast<std::size_t>(token_shape[0]) + 1);
            for (std::size_t row = 0; row < qo_indptr.size(); ++row) {
                qo_indptr[row] = static_cast<std::uint32_t>(row);
            }
        }
        if (qo_indptr.size() < 2 ||
            qo_indptr.front() != 0 ||
            qo_indptr.back() != token_shape[0]) {
            return fail(
                "decode envelope: EmbedIndptr does not cover the tokens");
        }
        ChannelView& channel_view =
            found->second.instance->view();
        const auto& pending_slots =
            staged.lanes[program]->prior_put_slots;
        DecodeEnvelopeLane base{};
        base.pass_commit = reinterpret_cast<std::uintptr_t>(
            staged.lanes[program]->snapshot->device);
        base.write_lower_bound =
            view.ptir_kv_write_lower_bounds.data()[program];
        base.write_upper_bound =
            view.ptir_kv_write_upper_bounds.data()[program];
        auto bind_source = [&](
                               const PortBinding& binding,
                               std::uint64_t& source) {
            const std::uint32_t slot =
                channel_view.slot(binding.channel);
            const bool pending = pending_slots.contains(slot);
            source = reinterpret_cast<std::uintptr_t>(
                pending
                    ? channel_view.pending_cell(binding.channel)
                    : channel_view.committed_cell(binding.channel));
        };
        bind_source(
            *token, base.token_source);
        if (position != nullptr) {
            bind_source(*position, base.position_source);
        } else {
            base.position_source =
                reinterpret_cast<std::uintptr_t>(buffers.position_ids);
        }
        for (std::size_t request = 0;
             request + 1 < qo_indptr.size();
             ++request) {
            if (qo_indptr[request + 1] != qo_indptr[request] + 1) {
                return fail(
                    "decode envelope: lane is not single-token");
            }
            DecodeEnvelopeLane lane = base;
            const std::size_t output_request =
                static_cast<std::size_t>(
                    program_request_starts[program]) +
                request;
            if (output_request >= lanes.size() ||
                lanes[output_request].passthrough == 0) {
                return fail(
                    "decode envelope: lane request rows collide");
            }
            if (lanes[output_request].source_page_count == 0) {
                // A decode lane must own at least one leased page to host
                // its KV append; a 0-page template span is broken host
                // geometry, not a device-resolvable condition (RV-17).
                return fail(
                    "decode envelope: lane has a 0-page template span");
            }
            lane.token_start =
                program_token_starts[program] + qo_indptr[request];
            lane.request_index =
                static_cast<std::uint32_t>(output_request);
            lane.source_token_start = qo_indptr[request];
            lane.source_position_start = position != nullptr
                ? qo_indptr[request]
                : lane.token_start;
            lane.source_page_begin =
                lanes[output_request].source_page_begin;
            lane.source_page_count =
                lanes[output_request].source_page_count;
            lane.passthrough = 0;
            lanes[output_request] = lane;
            ++envelope_lanes;
        }
    }
    if (envelope_lanes == 0) {
        return fail("decode envelope: no envelope lanes in the batch");
    }

    if (state.d_envelope_kills == nullptr) {
        CUDA_CHECK(cudaMalloc(&state.d_envelope_kills, sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMemset(state.d_envelope_kills, 0, sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMallocHost(&state.h_envelope_kills, sizeof(std::uint32_t)));
        *state.h_envelope_kills = 0;
    }
    if (const std::uint32_t seen = *state.h_envelope_kills;
        seen > state.envelope_kills_reported) {
        const std::uint32_t fresh = seen - state.envelope_kills_reported;
        state.envelope_kills_reported = seen;
        std::lock_guard<std::mutex> lock(state.stats_mutex);
        state.stats.decode_envelope_chain_kills += fresh;
        std::cerr << "[pie-driver-cuda] decode-envelope compose FAIL-STOPPED "
                  << fresh << " lane(s): device position escaped its "
                  << "containment window or template page span\n";
    }

    const DecodeEnvelopeUploadArena::Staged uploaded =
        state.decode_envelope_upload.upload(
            lanes, buffers.kv_page_indices,
            template_kv_page_indptr[lanes.size()], staged.stream);
    const DecodeEnvelopeOutputs outputs{
        .token_ids = buffers.token_ids,
        .position_ids = buffers.position_ids,
        .kv_page_indices = buffers.kv_page_indices,
        .kv_page_indptr = buffers.kv_page_indptr,
        .kv_last_page_lens = buffers.kv_last_page_lens,
        .row_valid = buffers.row_valid,
        .rs_slot_ids = buffers.rs_slot_ids,
        .template_pages = uploaded.template_pages,
        .chain_kills = state.d_envelope_kills,
        .dummy_page = buffers.dummy_page,
        .page_size = buffers.page_size,
    };
    std::uint32_t threads = 32;
    while (threads < lanes.size()) threads *= 2;
    compose_decode_envelopes<<<
        1,
        threads,
        lanes.size() * sizeof(std::uint32_t),
        staged.stream>>>(
        uploaded.lanes,
        static_cast<std::uint32_t>(lanes.size()),
        outputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(
        state.h_envelope_kills,
        state.d_envelope_kills,
        sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost,
        staged.stream));
    state.decode_envelope_upload.mark_used(staged.stream);
    {
        std::lock_guard<std::mutex> lock(state.stats_mutex);
        ++state.stats.decode_envelope_batches;
        state.stats.decode_envelope_lanes += envelope_lanes;
    }
    return true;
}

bool Dispatch::enqueue_fixed_decode(
    const pie_native::LaunchView& view,
    std::uint32_t page_size,
    std::uint32_t device_pages,
    const FixedDecodeDeviceBuffers& buffers,
    std::string* err,
    StagedLaunch& launch) {
    if (err != nullptr) err->clear();
    Impl& state = *impl_;
    StagedLaunch::State& staged = *launch.state_;
    const std::size_t programs = view.ptir_program_hashes.size();
    if (programs == 0 ||
        view.ptir_program_instances.size() != programs ||
        staged.lanes.size() != programs ||
        !staged.active ||
        page_size == 0 ||
        device_pages == 0 ||
        buffers.dummy_page >= device_pages) {
        if (err != nullptr) {
            *err = "ptir fixed decode preconditions rejected the launch "
                   "(lanes/staging/page shape)";
        }
        return false;
    }
    if (buffers.token_ids == nullptr ||
        buffers.position_ids == nullptr ||
        buffers.qo_indptr == nullptr ||
        buffers.kv_page_indices == nullptr ||
        buffers.kv_page_indptr == nullptr ||
        buffers.kv_last_page_lens == nullptr ||
        buffers.w_page == nullptr ||
        buffers.w_off == nullptr ||
        buffers.row_valid == nullptr ||
        buffers.token_capacity < programs ||
        buffers.request_capacity < programs) {
        if (err != nullptr) {
            *err = "ptir fixed decode output buffers are undersized";
        }
        return false;
    }
    if (view.kv_translation_indptr.size() != programs + 1 ||
        view.kv_translation_indptr.data()[0] != 0 ||
        view.kv_translation_indptr.data()[programs] !=
            view.kv_translation.size()) {
        if (err != nullptr) {
            *err = "ptir fixed decode translation table is malformed";
        }
        return false;
    }
    const bool has_write_bounds =
        view.ptir_kv_write_lower_bounds.size() == programs &&
        view.ptir_kv_write_upper_bounds.size() == programs;
    if ((!view.ptir_kv_write_lower_bounds.empty() ||
         !view.ptir_kv_write_upper_bounds.empty()) &&
        !has_write_bounds) {
        if (err != nullptr) {
            *err = "ptir fixed decode write bounds are incomplete";
        }
        return false;
    }

    constexpr std::uint8_t required_ports[] = {
        kPortEmbedTokens,
        kPortPages,
        kPortPageIndptr,
        kPortKvLen,
        kPortWSlot,
        kPortWOff,
    };
    struct ProgramPorts {
        BoundInstance* instance = nullptr;
        std::array<const PortBinding*, 10> by_tag{};
        std::uint32_t translation_begin = 0;
        std::uint32_t translation_len = 0;
        std::uint32_t pages_capacity = 0;
        std::size_t wire_position_offset =
            std::numeric_limits<std::size_t>::max();
    };
    std::vector<ProgramPorts> ports(programs);
    std::size_t maximum_pages = 0;
    for (std::size_t program = 0; program < programs; ++program) {
        const std::uint64_t instance_id =
            view.ptir_program_instances.data()[program];
        auto found = state.instances.find(instance_id);
        if (found == state.instances.end() ||
            found->second.trace == nullptr ||
            found->second.program_hash !=
                view.ptir_program_hashes.data()[program] ||
            found->second.geometry_class == PIE_GEOMETRY_CLASS_HOST) {
            if (err != nullptr) {
                *err = "ptir fixed decode instance is unknown, stale, or "
                       "host-classified";
            }
            return false;
        }
        const Trace& trace = *found->second.trace;
        ProgramPorts& program_ports = ports[program];
        program_ports.instance = &found->second;
        for (const PortBinding& binding : trace.ports) {
            if (binding.is_const) continue;
            if (binding.port > kPortAttnMask ||
                program_ports.by_tag[binding.port] != nullptr) {
                if (err != nullptr) {
                    *err = "ptir fixed decode port bindings are out of "
                           "range or duplicated";
                }
                return false;
            }
            program_ports.by_tag[binding.port] = &binding;
        }
        if (program_ports.by_tag[kPortAttnMask] != nullptr) {
            if (err != nullptr) {
                *err = "ptir fixed decode cannot compose attention-mask ports";
            }
            return false;
        }
        for (const std::uint8_t port : required_ports) {
            if (program_ports.by_tag[port] == nullptr) {
                if (err != nullptr) {
                    *err = "ptir fixed decode is missing required geometry "
                           "port " + std::to_string(port);
                }
                return false;
            }
        }
        auto channel_numel = [&](std::uint8_t port) {
            const ChannelId channel =
                program_ports.by_tag[port]->channel;
            return channel < trace.channels.size()
                ? trace.channels[channel].type.shape.numel()
                : std::size_t{0};
        };
        auto channel_dtype = [&](std::uint8_t port) {
            const ChannelId channel =
                program_ports.by_tag[port]->channel;
            return trace.channels[channel].type.dtype;
        };
        if (channel_numel(kPortEmbedTokens) != 1 ||
            channel_numel(kPortPageIndptr) != 2 ||
            channel_numel(kPortKvLen) != 1 ||
            channel_numel(kPortWSlot) != 1 ||
            channel_numel(kPortWOff) != 1 ||
            (channel_dtype(kPortEmbedTokens) != DType::I32 &&
             channel_dtype(kPortEmbedTokens) != DType::U32) ||
            channel_dtype(kPortPages) != DType::U32 ||
            channel_dtype(kPortPageIndptr) != DType::U32 ||
            channel_dtype(kPortKvLen) != DType::U32 ||
            channel_dtype(kPortWSlot) != DType::U32 ||
            channel_dtype(kPortWOff) != DType::U32 ||
            (program_ports.by_tag[kPortEmbedIndptr] != nullptr &&
             (channel_numel(kPortEmbedIndptr) != 2 ||
              channel_dtype(kPortEmbedIndptr) != DType::U32)) ||
            (program_ports.by_tag[kPortReadout] != nullptr &&
             (channel_numel(kPortReadout) != 1 ||
              channel_dtype(kPortReadout) != DType::U32))) {
            if (err != nullptr) {
                *err = "ptir fixed decode channel shapes or dtypes do not "
                       "match the envelope";
            }
            return false;
        }
        if (program_ports.by_tag[kPortPositions] != nullptr) {
            if (channel_numel(kPortPositions) != 1 ||
                channel_dtype(kPortPositions) != DType::U32) {
                if (err != nullptr) {
                    *err = "ptir fixed decode positions channel is not a "
                           "u32 scalar";
                }
                return false;
            }
        } else if (view.position_ids.size() != programs) {
            if (err != nullptr) {
                *err = "ptir fixed decode positions are neither "
                       "loop-carried nor wire-supplied";
            }
            return false;
        }
        const ChannelId pages_channel =
            program_ports.by_tag[kPortPages]->channel;
        if (pages_channel >= trace.channels.size()) {
            if (err != nullptr) {
                *err = "ptir fixed decode pages channel is out of range";
            }
            return false;
        }
        const auto& page_dims =
            trace.channels[pages_channel].type.shape.dims;
        if (page_dims.size() != 1 &&
            (page_dims.size() != 2 || page_dims[0] != 1)) {
            if (err != nullptr) {
                *err = "ptir fixed decode pages channel must be a vector";
            }
            return false;
        }
        program_ports.pages_capacity = static_cast<std::uint32_t>(
            channel_numel(kPortPages));

        const std::uint32_t translation_begin =
            view.kv_translation_indptr.data()[program];
        const std::uint32_t translation_end =
            view.kv_translation_indptr.data()[program + 1];
        if (translation_end < translation_begin ||
            translation_end > view.kv_translation.size() ||
            translation_end == translation_begin) {
            if (err != nullptr) {
                *err = "ptir fixed decode lane translation span is empty "
                       "or out of bounds";
            }
            return false;
        }
        program_ports.translation_begin = translation_begin;
        program_ports.translation_len =
            translation_end - translation_begin;
        const std::size_t lane_pages = std::max<std::size_t>(
            1,
            std::min<std::size_t>(
                program_ports.pages_capacity,
                program_ports.translation_len));
        if (lane_pages >
            std::numeric_limits<std::size_t>::max() - maximum_pages) {
            if (err != nullptr) {
                *err = "ptir fixed decode page capacity overflow";
            }
            return false;
        }
        maximum_pages += lane_pages;
    }
    if (maximum_pages > buffers.page_capacity) {
        if (err != nullptr) {
            *err = "ptir fixed decode page output exceeds capacity";
        }
        return false;
    }

    // Pull host-writer rings on the LAUNCH stream: the compose kernel and
    // every stage kernel are ordered behind these copies, and the staging
    // rides the launch state past their completion — no whole-device
    // synchronize on the fire path.
    for (std::size_t program = 0; program < programs; ++program) {
        BoundInstance& instance = *ports[program].instance;
        std::string value_error;
        if (!instance.instance->writer_inputs_available(
                &value_error)) {
            throw RetryableLaunchError(value_error);
        }
        instance.instance->pull_writer_inputs(
            staged.stream, staged.writer_staging);
    }

    std::vector<std::uint32_t> upload_values(
        view.kv_translation.data(),
        view.kv_translation.data() + view.kv_translation.size());
    for (std::size_t program = 0; program < programs; ++program) {
        if (ports[program].by_tag[kPortPositions] != nullptr) continue;
        ports[program].wire_position_offset = upload_values.size();
        upload_values.push_back(view.position_ids.data()[program]);
    }
    state.fixed_decode_upload.reserve(
        programs, upload_values.size(), staged.stream);
    std::vector<FixedDecodeLane> lanes(programs);
    for (std::size_t program = 0; program < programs; ++program) {
        ProgramPorts& program_ports = ports[program];
        ChannelView& channel_view =
            program_ports.instance->instance->view();
        const auto& pending_slots =
            staged.lanes[program]->prior_put_slots;
        FixedDecodeLane& lane = lanes[program];
        const std::uint8_t ports_in_lane[] = {
            kPortEmbedTokens,
            kPortPositions,
            kPortPages,
            kPortPageIndptr,
            kPortKvLen,
            kPortWSlot,
            kPortWOff,
        };
        std::uint64_t* sources[] = {
            &lane.token,
            &lane.position,
            &lane.pages,
            &lane.page_indptr,
            &lane.kv_len,
            &lane.w_slot,
            &lane.w_off,
        };
        for (std::size_t index = 0;
             index < kFixedDecodePortCount;
             ++index) {
            const PortBinding* binding =
                program_ports.by_tag[ports_in_lane[index]];
            if (binding == nullptr) {
                *sources[index] = 0;
                lane.ready[index] = 0;
                continue;
            }
            const std::uint32_t slot =
                channel_view.slot(binding->channel);
            const bool pending = pending_slots.contains(slot);
            *sources[index] = reinterpret_cast<std::uintptr_t>(
                pending
                    ? channel_view.pending_cell(binding->channel)
                    : channel_view.committed_cell(binding->channel));
            lane.ready[index] = pending
                ? 0
                : reinterpret_cast<std::uintptr_t>(
                      channel_view.d_full() +
                      static_cast<std::size_t>(slot) * kMaxRing +
                      channel_view.registry()->host_head(slot));
        }
        lane.pass_commit = reinterpret_cast<std::uintptr_t>(
            staged.lanes[program]->snapshot->device);
        if (program_ports.wire_position_offset !=
            std::numeric_limits<std::size_t>::max()) {
            lane.position = reinterpret_cast<std::uintptr_t>(
                state.fixed_decode_upload.translation_at(
                    program_ports.wire_position_offset));
        }
        lane.translation = reinterpret_cast<std::uintptr_t>(
            state.fixed_decode_upload.translation_at(
                program_ports.translation_begin));
        if (has_write_bounds) {
            lane.write_lower_bound =
                view.ptir_kv_write_lower_bounds.data()[program];
            lane.write_upper_bound =
                view.ptir_kv_write_upper_bounds.data()[program];
        }
        lane.translation_len = program_ports.translation_len;
        lane.pages_capacity = program_ports.pages_capacity;
    }

    // Chain-kill diagnostic plumbing: report growth from earlier batches
    // (the mirror copy below is async on the launch stream), then arm this
    // batch's counter.
    if (state.d_fixed_decode_kills == nullptr) {
        CUDA_CHECK(cudaMalloc(&state.d_fixed_decode_kills, sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMemset(state.d_fixed_decode_kills, 0, sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMallocHost(&state.h_fixed_decode_kills, sizeof(std::uint32_t)));
        *state.h_fixed_decode_kills = 0;
    }
    if (const std::uint32_t seen = *state.h_fixed_decode_kills;
        seen > state.fixed_decode_kills_reported) {
        const std::uint32_t fresh = seen - state.fixed_decode_kills_reported;
        state.fixed_decode_kills_reported = seen;
        std::lock_guard<std::mutex> lock(state.stats_mutex);
        state.stats.fixed_decode_chain_kills += fresh;
        std::cerr << "[pie-driver-cuda] fixed-decode compose FAIL-STOPPED "
                  << fresh << " lane(s): geometry/containment inconsistency; "
                  << "the affected chains are killed (successors dummy-run)\n";
    }

    const FixedDecodeLane* device_lanes =
        state.fixed_decode_upload.upload(
            lanes, upload_values, staged.stream);
    const FixedDecodeOutputs outputs{
        .token_ids = buffers.token_ids,
        .position_ids = buffers.position_ids,
        .qo_indptr = buffers.qo_indptr,
        .kv_page_indices = buffers.kv_page_indices,
        .kv_page_indptr = buffers.kv_page_indptr,
        .kv_last_page_lens = buffers.kv_last_page_lens,
        .w_page = buffers.w_page,
        .w_off = buffers.w_off,
        .row_valid = buffers.row_valid,
        .rs_slot_ids = buffers.rs_slot_ids,
        .sample_indices = buffers.sample_indices,
        .chain_kills = state.d_fixed_decode_kills,
        .dummy_page = buffers.dummy_page,
        .page_size = page_size,
        .device_pages = device_pages,
    };
    std::uint32_t threads = 32;
    while (threads < programs) threads *= 2;
    compose_fixed_decode<<<
        1,
        threads,
        programs * sizeof(std::uint32_t),
        staged.stream>>>(
        device_lanes,
        static_cast<std::uint32_t>(programs),
        outputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(
        state.h_fixed_decode_kills,
        state.d_fixed_decode_kills,
        sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost,
        staged.stream));
    state.fixed_decode_upload.mark_used(staged.stream);
    {
        std::lock_guard<std::mutex> lock(state.stats_mutex);
        ++state.stats.fixed_decode_batches;
        state.stats.fixed_decode_lanes += programs;
    }
    return true;
}

bool Dispatch::resolve_descriptors(const pie_native::LaunchView& view,
                                   std::uint32_t page_size,
                                   std::uint32_t device_pages,
                                   ResolvedPrograms& out,
                                   std::string* err,
                                   bool allow_structured_masks,
                                   StagedLaunch* launch,
                                   bool allow_device_composed) {
    if (err) err->clear();
    out = ResolvedPrograms{};
    if (view.ptir_program_hashes.empty()) return false;
    Impl& s = *impl_;
    const std::size_t n_prog = view.ptir_program_hashes.size();
    if (view.ptir_program_instances.size() != n_prog) {
        if (err) *err = "ptir descriptor resolution instance/hash count mismatch";
        return false;
    }
    StagedLaunch::State* staged =
        launch == nullptr ? nullptr : launch->state_.get();
    if (staged != nullptr) {
        if (!staged->active || staged->lanes.size() != n_prog) {
            if (err) *err = "ptir descriptor resolution has no active launch";
            return false;
        }
    }

    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        if (it == s.instances.end()) {
            if (err) *err = "ptir descriptor resolution missing instance " +
                            std::to_string(iid);
            return false;
        }
        if (it->second.program_hash != view.ptir_program_hashes.data()[p]) {
            if (err) *err = "ptir descriptor resolution instance/hash mismatch";
            return false;
        }
        const Trace* trace = it->second.trace;
        if (trace == nullptr) {
            if (err) *err = "ptir descriptor resolution missing trace";
            return false;
        }
    }

    auto try_device_composed_template = [&]() {
        // Any all-decode lane count qualifies: graph-lattice padding
        // (compose.cpp) takes the composed batch from R to its request
        // bucket with device-side pad rows, so the template no longer
        // requires R to sit exactly on the lattice.
        if (!allow_device_composed || staged == nullptr ||
            page_size == 0 || device_pages == 0 ||
            !view.rs_slot_ids.empty() ||
            !view.rs_fold_lens.empty() ||
            !view.rs_buffer_slot_ids.empty() ||
            view.kv_translation_indptr.size() != n_prog + 1 ||
            view.kv_translation_indptr.data()[0] != 0 ||
            view.kv_translation_indptr.data()[n_prog] !=
                view.kv_translation.size() ||
            view.ptir_kv_write_lower_bounds.size() != n_prog ||
            view.ptir_kv_write_upper_bounds.size() != n_prog) {
            return false;
        }
        ResolvedPrograms candidate;
        candidate.per_program.resize(n_prog);
        candidate.is_device_geometry.assign(n_prog, 0);
        candidate.device_count = 0;
        bool all_decode_envelopes = true;
        auto constant_is = [](const PortBinding* binding,
                              std::span<const std::uint32_t> expected) {
            if (binding == nullptr) return true;
            if (!binding->is_const ||
                binding->const_data.size() != expected.size_bytes()) {
                return false;
            }
            return std::memcmp(
                       binding->const_data.data(),
                       expected.data(),
                       expected.size_bytes()) == 0;
        };
        constexpr std::array<std::uint32_t, 2> default_indptr{0, 1};
        constexpr std::array<std::uint32_t, 1> default_readout{0};
        for (std::size_t p = 0; p < n_prog; ++p) {
            BoundInstance& instance = s.instances.at(
                view.ptir_program_instances.data()[p]);
            if (instance.geometry_class == PIE_GEOMETRY_CLASS_HOST) {
                all_decode_envelopes = false;
                continue;
            }
            if (instance.geometry_class !=
                    PIE_GEOMETRY_CLASS_DECODE_ENVELOPE ||
                instance.trace == nullptr) {
                return false;
            }
            const Trace& trace = *instance.trace;
            std::array<const PortBinding*, 10> dynamic{};
            std::array<const PortBinding*, 10> constants{};
            for (const PortBinding& binding : trace.ports) {
                if (binding.port > kPortAttnMask) return false;
                auto& slot = binding.is_const
                    ? constants[binding.port]
                    : dynamic[binding.port];
                if (slot != nullptr) return false;
                slot = &binding;
            }
            if (dynamic[kPortAttnMask] != nullptr ||
                constants[kPortAttnMask] != nullptr ||
                (dynamic[kPortEmbedIndptr] == nullptr &&
                 !constant_is(
                     constants[kPortEmbedIndptr], default_indptr)) ||
                (dynamic[kPortReadout] == nullptr &&
                 !constant_is(
                     constants[kPortReadout], default_readout))) {
                return false;
            }
            constexpr std::array<std::uint8_t, 7> required{
                kPortEmbedTokens,
                kPortPositions,
                kPortPages,
                kPortPageIndptr,
                kPortKvLen,
                kPortWSlot,
                kPortWOff,
            };
            for (const std::uint8_t port : required) {
                if (dynamic[port] == nullptr ||
                    constants[port] != nullptr) {
                    return false;
                }
            }
            auto channel = [&](std::uint8_t port)
                -> const Channel* {
                const ChannelId id = dynamic[port]->channel;
                return id < trace.channels.size()
                    ? &trace.channels[id]
                    : nullptr;
            };
            const Channel* tokens = channel(kPortEmbedTokens);
            const Channel* positions = channel(kPortPositions);
            const Channel* pages = channel(kPortPages);
            const Channel* page_indptr = channel(kPortPageIndptr);
            const Channel* kv_len = channel(kPortKvLen);
            const Channel* w_slot = channel(kPortWSlot);
            const Channel* w_off = channel(kPortWOff);
            const Channel* embed_indptr =
                dynamic[kPortEmbedIndptr] != nullptr
                    ? channel(kPortEmbedIndptr)
                    : nullptr;
            const Channel* readout =
                dynamic[kPortReadout] != nullptr
                    ? channel(kPortReadout)
                    : nullptr;
            if (tokens == nullptr || positions == nullptr ||
                pages == nullptr || page_indptr == nullptr ||
                kv_len == nullptr || w_slot == nullptr ||
                w_off == nullptr ||
                (dynamic[kPortEmbedIndptr] != nullptr &&
                 embed_indptr == nullptr) ||
                (dynamic[kPortReadout] != nullptr &&
                 readout == nullptr)) {
                return false;
            }
            const auto& page_dims = pages->type.shape.dims;
            if (tokens->type.shape.numel() != 1 ||
                positions->type.shape.numel() != 1 ||
                pages->type.shape.numel() == 0 ||
                (page_dims.size() != 1 &&
                 (page_dims.size() != 2 || page_dims[0] != 1)) ||
                page_indptr->type.shape.numel() != 2 ||
                kv_len->type.shape.numel() != 1 ||
                w_slot->type.shape.numel() != 1 ||
                w_off->type.shape.numel() != 1 ||
                (tokens->type.dtype != DType::I32 &&
                 tokens->type.dtype != DType::U32) ||
                positions->type.dtype != DType::U32 ||
                pages->type.dtype != DType::U32 ||
                page_indptr->type.dtype != DType::U32 ||
                kv_len->type.dtype != DType::U32 ||
                w_slot->type.dtype != DType::U32 ||
                w_off->type.dtype != DType::U32 ||
                (embed_indptr != nullptr &&
                 (embed_indptr->type.dtype != DType::U32 ||
                  embed_indptr->type.shape.dims !=
                      std::vector<std::uint32_t>{2})) ||
                (readout != nullptr &&
                 (readout->type.dtype != DType::U32 ||
                  readout->type.shape.numel() != 1))) {
                return false;
            }
            const std::uint32_t translation_begin =
                view.kv_translation_indptr.data()[p];
            const std::uint32_t translation_end =
                view.kv_translation_indptr.data()[p + 1];
            if (translation_end <= translation_begin ||
                translation_end > view.kv_translation.size() ||
                view.ptir_kv_write_lower_bounds.data()[p] >=
                    view.ptir_kv_write_upper_bounds.data()[p]) {
                return false;
            }
            FireGeometry& geometry = candidate.per_program[p];
            geometry.token_ids = {0};
            geometry.position_ids = {0};
            geometry.qo_indptr = {0, 1};
            geometry.kv_page_indices = {0};
            geometry.kv_page_indptr = {0, 1};
            geometry.kv_last_page_lens = {1};
            geometry.sampling_indices = {0};
            geometry.sampling_indptr = {0, 1};
            geometry.w_page = {0};
            geometry.w_off = {0};
            geometry.has_kv_family = true;
            geometry.has_write_desc = true;
            candidate.is_device_geometry[p] = 1;
            ++candidate.device_count;
        }
        if (candidate.device_count == 0) return false;
        // The placeholder geometry this template stages is consumable ONLY
        // by the fully device-composed path (`enqueue_fixed_decode`, which
        // re-derives every lane's real geometry from device cells). A MIXED
        // batch (Host-class chunk lanes present) routes to
        // `enqueue_decode_envelopes`, which trusts the host page spans —
        // feeding it the 1-page placeholder containment-kills every
        // envelope lane past its first page. Mixed batches must resolve
        // real geometry through the descriptor fallback below.
        if (!all_decode_envelopes) return false;
        candidate.device_composed = true;
        out = std::move(candidate);
        return true;
    };
    if (try_device_composed_template()) return true;

    out.per_program.resize(n_prog);
    out.is_device_geometry.assign(n_prog, 0);
    const bool resolve_device_mask =
        view.has_user_mask && view.flattened_masks.empty();
    bool resolved_mask = false;
    std::vector<detail::PortCellCache> cached_cells(n_prog);
    // Pull host-writer rings on the descriptor stream: the readback pack
    // below is ordered behind these copies and its `read` synchronizes the
    // stream before any host use, so the pull itself never blocks. The bool
    // staging rides the launch state (which outlives the copies); the rare
    // probe call without a staged launch keeps a blocking local pull.
    cudaStream_t descriptor_stream =
        staged == nullptr ? nullptr : staged->stream;
    std::vector<std::vector<std::uint8_t>> local_writer_staging;
    auto& writer_staging = staged != nullptr
        ? staged->writer_staging
        : local_writer_staging;
    bool pulled_writer_input = false;
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid =
            view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        // Classify once: the ACK'd class — not a trace sniff — decides which
        // programs resolve descriptors from device cells (RV-6).
        if (it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST) continue;
        std::string value_error;
        if (!it->second.instance->writer_inputs_available(&value_error)) {
            throw RetryableLaunchError(value_error);
        }
        pulled_writer_input =
            it->second.instance->pull_writer_inputs(
                descriptor_stream, writer_staging) ||
            pulled_writer_input;
    }
    if (pulled_writer_input && staged == nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    struct PortCopy {
        std::size_t program = 0;
        std::uint32_t slot = 0;
        const void* source = nullptr;
        const std::uint8_t* ready_source = nullptr;
        std::size_t payload_offset = 0;
        std::size_t ready_offset = 0;
    };
    std::vector<PortCopy> port_copies;
    std::vector<std::uint32_t> ready(n_prog, 1);
    std::vector<std::size_t> snapshot_offsets(
        n_prog, std::numeric_limits<std::size_t>::max());
    std::size_t packed_bytes = 0;
    auto reserve_packed = [&](std::size_t bytes, std::size_t alignment) {
        if (alignment == 0 ||
            packed_bytes >
                std::numeric_limits<std::size_t>::max() -
                    (alignment - 1)) {
            throw std::runtime_error(
                "ptir descriptor readback size overflow");
        }
        packed_bytes =
            (packed_bytes + alignment - 1) & ~(alignment - 1);
        const std::size_t offset = packed_bytes;
        if (bytes >
            std::numeric_limits<std::size_t>::max() - packed_bytes) {
            throw std::runtime_error(
                "ptir descriptor readback size overflow");
        }
        packed_bytes += bytes;
        return offset;
    };
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid =
            view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        const Trace* trace = it->second.trace;
        const bool mask_only =
            it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST &&
            resolve_device_mask;
        if (it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST &&
            !mask_only) {
            continue;
        }
        const std::unordered_set<std::uint32_t>* pending_slots =
            staged == nullptr
                ? nullptr
                : &staged->lanes[p]->prior_put_slots;
        for (const PortBinding& binding : trace->ports) {
            if (binding.is_const) continue;
            if (mask_only && binding.port != kPortAttnMask) continue;
            ChannelView& channel_view = it->second.instance->view();
            const std::uint32_t slot =
                channel_view.slot(binding.channel);
            auto [cell, inserted] =
                cached_cells[p].try_emplace(slot);
            if (!inserted) continue;
            cell->second.bytes.resize(
                channel_view.cell_bytes(binding.channel));
            const std::size_t ready_offset =
                reserve_packed(sizeof(std::uint8_t), alignof(std::uint8_t));
            const std::size_t payload_offset =
                reserve_packed(
                    cell->second.bytes.size(), alignof(std::uint32_t));
            const bool pending =
                pending_slots != nullptr &&
                pending_slots->contains(slot);
            cell->second.ready = pending ? 1 : 0;
            port_copies.push_back(PortCopy{
                .program = p,
                .slot = slot,
                .source = pending
                    ? channel_view.pending_cell(binding.channel)
                    : channel_view.committed_cell(binding.channel),
                .ready_source = pending
                    ? nullptr
                    : channel_view.d_full() +
                          static_cast<std::size_t>(slot) * kMaxRing +
                          channel_view.registry()->host_head(slot),
                .payload_offset = payload_offset,
                .ready_offset = ready_offset,
            });
        }
        if (staged != nullptr) {
            snapshot_offsets[p] = reserve_packed(
                sizeof(std::uint32_t), alignof(std::uint32_t));
        }
    }

    std::vector<DescriptorPackCopy> pack_copies;
    for (const PortCopy& copy : port_copies) {
        const std::size_t bytes =
            cached_cells[copy.program].at(copy.slot).bytes.size();
        if (bytes == 0) {
            pack_copies.push_back(DescriptorPackCopy{
                .source = reinterpret_cast<std::uintptr_t>(copy.source),
                .ready_source =
                    reinterpret_cast<std::uintptr_t>(copy.ready_source),
                .destination_offset = copy.payload_offset,
                .ready_offset = copy.ready_offset,
                .byte_count = 0,
                .default_ready =
                    static_cast<std::uint8_t>(
                        copy.ready_source == nullptr),
            });
            continue;
        }
        for (std::size_t offset = 0; offset < bytes;) {
            const std::size_t chunk = std::min(
                kDescriptorCopyChunkBytes, bytes - offset);
            pack_copies.push_back(DescriptorPackCopy{
                .source = reinterpret_cast<std::uintptr_t>(
                    static_cast<const std::uint8_t*>(copy.source) + offset),
                .ready_source =
                    reinterpret_cast<std::uintptr_t>(copy.ready_source),
                .destination_offset = copy.payload_offset + offset,
                .ready_offset =
                    offset == 0
                        ? static_cast<std::uint64_t>(copy.ready_offset)
                        : kNoDescriptorReadyOffset,
                .byte_count = static_cast<std::uint32_t>(chunk),
                .default_ready =
                    static_cast<std::uint8_t>(
                        copy.ready_source == nullptr),
            });
            offset += chunk;
        }
    }
    if (staged != nullptr) {
        for (std::size_t p = 0; p < n_prog; ++p) {
            const auto& instance =
                s.instances.at(view.ptir_program_instances.data()[p]);
            if (instance.geometry_class == PIE_GEOMETRY_CLASS_HOST &&
                !resolve_device_mask) {
                continue;
            }
            pack_copies.push_back(DescriptorPackCopy{
                .source = reinterpret_cast<std::uintptr_t>(
                    staged->lanes[p]->snapshot->device),
                .ready_source = 0,
                .destination_offset = snapshot_offsets[p],
                .ready_offset = kNoDescriptorReadyOffset,
                .byte_count = sizeof(std::uint32_t),
                .default_ready = 0,
            });
        }
    }

    const std::uint8_t* packed = nullptr;
    if (!pack_copies.empty()) {
        packed = s.descriptor_readback.read(
            pack_copies, packed_bytes, descriptor_stream);
    }
    for (const PortCopy& copy : port_copies) {
        auto& destination = cached_cells[copy.program].at(copy.slot);
        if (!destination.bytes.empty()) {
            std::memcpy(
                destination.bytes.data(),
                packed + copy.payload_offset,
                destination.bytes.size());
        }
        destination.ready = packed[copy.ready_offset];
    }
    if (staged != nullptr && packed != nullptr) {
        for (std::size_t p = 0; p < n_prog; ++p) {
            if (snapshot_offsets[p] ==
                std::numeric_limits<std::size_t>::max()) {
                continue;
            }
            std::memcpy(
                &ready[p], packed + snapshot_offsets[p],
                sizeof(std::uint32_t));
        }
    }
    if (!pack_copies.empty()) {
        std::lock_guard<std::mutex> lock(s.stats_mutex);
        ++s.stats.descriptor_readback_batches;
        s.stats.descriptor_readback_cells += port_copies.size();
        s.stats.descriptor_readback_bytes += packed_bytes;
    }

    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        const Trace* trace = it->second.trace;
        const bool mask_only =
            it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST &&
            resolve_device_mask;
        if (it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST &&
            !mask_only) {
            continue;
        }

        const std::unordered_set<std::uint32_t>* pending_slots = nullptr;
        if (staged != nullptr) {
            const StagedLane& lane = *staged->lanes[p];
            if (ready[p] == 0) {
                throw RetryableLaunchError(
                    "ptir prologue or channel readiness did not commit");
            }
            pending_slots = &lane.prior_put_slots;
        }

        FireGeometry& fg = out.per_program[p];
        if (mask_only) {
            if (!resolve_attention_mask(
                    *trace, it->second.instance->view(), fg, err,
                    allow_structured_masks, pending_slots,
                    &cached_cells[p])) {
                return false;
            }
            resolved_mask = true;
            continue;
        }
        if (!resolve_fire_geometry(
                *trace, it->second.instance->view(), page_size, fg, err,
                allow_structured_masks, pending_slots,
                &cached_cells[p])) {
            return false;
        }
        if (fg.structured_mask) {
            std::lock_guard<std::mutex> lock(s.stats_mutex);
            if (fg.has_mask) {
                ++s.stats.structured_mask_dense_fallback;
            } else {
                ++s.stats.structured_mask_direct;
            }
        }

        if (view.ptir_kv_write_lower_bounds.size() == n_prog &&
            view.ptir_kv_write_upper_bounds.size() == n_prog &&
            fg.has_write_desc) {
            const std::uint64_t lower =
                view.ptir_kv_write_lower_bounds.data()[p];
            const std::uint64_t upper =
                view.ptir_kv_write_upper_bounds.data()[p];
            if (!validate_kv_write_containment(
                    fg, page_size, lower, upper, err)) {
                return false;
            }
        }

        // WorkingSet page translation (kv_refact.md flattened-table model):
        // channel-resolved `Pages`/`WSlot` values are WorkingSet-RELATIVE
        // indexes — the guest never holds physical ids. Map them through this
        // instance's translation segment (committed mapping overlaid with the
        // fire's prepared write targets, built at prepare). An index past the
        // segment is a reserved-but-unwritten page (a masked-only attention
        // candidate): map it to page 0 — readable garbage the mask discards.
        // An EMPTY segment passes values through (legacy physical geometry).
        if (view.kv_translation_indptr.size() == n_prog + 1) {
            const std::uint32_t lo = view.kv_translation_indptr.data()[p];
            const std::uint32_t hi = view.kv_translation_indptr.data()[p + 1];
            if (hi > lo && hi <= view.kv_translation.size()) {
                const std::uint32_t* tr = view.kv_translation.data() + lo;
                const std::uint32_t tr_len = hi - lo;
                const bool masked_reads =
                    fg.has_mask || static_cast<bool>(fg.structured_mask);
                if (!translate_resolved_page_ids(
                        fg.kv_page_indices,
                        fg.w_page,
                        std::span<const std::uint32_t>(tr, tr_len),
                        masked_reads,
                        err)) {
                    return false;
                }
            }
        }

        if (!validate_fire_geometry(fg, device_pages, page_size, err)) {
            return false;
        }
        out.is_device_geometry[p] = 1;
        ++out.device_count;
    }
    return out.device_count > 0 || resolved_mask;
}

}  // namespace pie_cuda_driver::pipeline

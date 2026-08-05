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
#include <optional>
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
#include "model/attn_observation.hpp"
#include "model/attn_page_mask.hpp"
#include "model/attn_score.hpp"
#include "model/hook_sideband_arena.hpp"
#include "model/lora.hpp"
#include "store/kv_cache.hpp"

namespace pie_cuda_driver::pipeline {

namespace {

// A (stage, region) index over the host's emitted kernel table.
//
// The host (`compiler/codegen/src/cuda/`) is the only emitter, so both the
// source *and* the entry symbol it named are carried through unchanged. An
// entry whose `source` is empty is a *recorded* failure, not an omission --
// the host is saying it could not emit that region -- so it is left absent
// here and registration refuses the program.
class HostEmittedKernels {
public:
    explicit HostEmittedKernels(PieEmittedKernelSlice slice) {
        if (slice.ptr == nullptr) return;
        for (std::size_t i = 0; i < slice.len; ++i) {
            const PieEmittedKernel& kernel = slice.ptr[i];
            if (kernel.kind != PIE_KERNEL_FUSED || kernel.source.len == 0) {
                continue;
            }
            sources_.emplace(
                Key{kernel.stage_index, kernel.region_index},
                Region{
                    to_string(kernel.entry_name),
                    to_string(kernel.source)});
        }
    }

    // The host's region analysis rides the same table, keyed the same way.
    // It is the other half of one contract: the kernel above was emitted from
    // these answers, and the packer fills its side tables from them.
    void adopt(PieRegionAnalysisSlice slice) {
        if (slice.ptr == nullptr) return;
        for (std::size_t i = 0; i < slice.len; ++i) {
            const PieRegionAnalysis& region = slice.ptr[i];
            generated::StageRegionAnalysis analysis;
            analysis.flags = region.flags;
            analysis.direct_argmax.reserve(region.direct_argmax.len);
            for (std::size_t j = 0; j < region.direct_argmax.len; ++j) {
                const PieDirectArgmax& record = region.direct_argmax.ptr[j];
                analysis.direct_argmax.push_back(
                    generated::StageRegionArgmax{
                        record.node,
                        record.source_value,
                        record.intrinsic,
                        record.requires_single_row});
            }
            analysis.skipped.assign(
                region.skipped.ptr, region.skipped.ptr + region.skipped.len);
            regions_.emplace(
                Key{region.stage_index, region.region_index},
                std::move(analysis));
        }
    }

    static generated::ModuleCache::HostRegion lookup(
        void* context, std::size_t stage_index, std::size_t region_index) {
        auto* self = static_cast<HostEmittedKernels*>(context);
        const auto found = self->sources_.find(Key{
            static_cast<std::uint32_t>(stage_index),
            static_cast<std::uint32_t>(region_index)});
        if (found == self->sources_.end()) return {};
        return {&found->second.entry, &found->second.source};
    }

    static const generated::StageRegionAnalysis* lookup_region(
        void* context, std::size_t stage_index, std::size_t region_index) {
        auto* self = static_cast<HostEmittedKernels*>(context);
        const auto found = self->regions_.find(Key{
            static_cast<std::uint32_t>(stage_index),
            static_cast<std::uint32_t>(region_index)});
        if (found == self->regions_.end()) return nullptr;
        return &found->second;
    }

private:
    template <typename Slice>
    static std::string to_string(const Slice& slice) {
        if (slice.ptr == nullptr || slice.len == 0) return {};
        return std::string(
            reinterpret_cast<const char*>(slice.ptr), slice.len);
    }

    struct Region {
        std::string entry;
        std::string source;
    };
    struct Key {
        std::uint32_t stage;
        std::uint32_t region;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const {
            return (static_cast<std::size_t>(key.stage) << 32) ^ key.region;
        }
    };
    std::unordered_map<Key, Region, KeyHash> sources_;
    std::unordered_map<Key, generated::StageRegionAnalysis, KeyHash> regions_;
};

}  // namespace

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives under pie::driver (driver/common); bring both the
// fire and launch namespaces into scope so the CUDA-side tier-0/1 code below
// can use them unqualified.
using namespace pie::driver::launch;
using namespace pie::driver::fire;

// `store/kv_cache.hpp` (pulled in for the `envelope_dot` KV geometry) declares
// its own `pie_cuda_driver::DType`, which sits closer in the lookup chain than
// the using-directive above and would silently retarget every unqualified
// `DType` in this file. Pin the PTIR one explicitly; the cache's own dtype is
// spelled `pie_cuda_driver::DType` where it is needed.
using DType = pie::driver::launch::DType;

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
    // Two adjacent u32 words per fire occurrence (ringed):
    //   [0] pass_commit — seeded by the pull-validate kernel, ANDed by the
    //       readiness/ticket checks, zeroed by fail-stops.
    //   [1] kill — set ONLY by a compose fail-stop (fixed-decode/envelope
    //       chain kill). Settlement reads it to classify the lane FAILED
    //       (deterministic fault, channels poisoned) instead of RETRY
    //       (which v14 reserves for host staging-contract violations).
    struct CommitSnapshot {
        static constexpr std::size_t kWords = 2;
        std::uint32_t* device = nullptr;
        std::uint32_t* host = nullptr;
        std::uint32_t* host_device = nullptr;
        // Set once a wave's pull-validate has seeded the words below.
        // FramePrepare reads the commit word BEFORE this wave's
        // pull-validate runs, so an unseeded snapshot carries no verdict —
        // gating on it would refuse the lane for whatever the pool (or
        // `cudaMalloc`) happened to leave behind.
        bool ever_validated = false;
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

// PIE_DEBUG_PULL_VALIDATE=1 makes the pull-validate kernel name the ticket
// that vetoed a fire, which is otherwise reported only as the generic
// "ptir prologue or channel readiness did not commit".
const std::uint32_t kDiagnosePullValidate =
    std::getenv("PIE_DEBUG_PULL_VALIDATE") != nullptr ? 1u : 0u;

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
    // Ordered sub-batch offsets (mixed [wire][envelope] steps): the wire
    // sub-batch's totals. Lane i composes request `request_base + i` at
    // token row `row_base + i`, its pages landing at CSR base
    // `page_base`. Zero for the all-envelope whole-step form, whose lane
    // 0 also writes the CSR heads.
    std::uint32_t row_base = 0;
    std::uint32_t request_base = 0;
    std::uint32_t page_base = 0;
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
        // Fail-stop: kill the chain (successors dummy-run), mark the lane's
        // kill word so settlement classifies it FAILED, and count the kill
        // so the host reports it loudly — never a silent poison.
        auto* commit = const_cast<std::uint32_t*>(
            fixed_decode_pointer<std::uint32_t>(
                descriptor->pass_commit));
        if (commit != nullptr) {
            commit[0] = 0;
            commit[1] = 1;
        }
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
        std::uint32_t page_cursor = output.page_base;
        if (output.request_base == 0) {
            output.qo_indptr[0] = 0;
            output.kv_page_indptr[0] = 0;
        }
        for (std::uint32_t index = 0;
             index < lane_count;
             ++index) {
            const std::uint32_t count = page_offsets[index];
            page_offsets[index] = page_cursor;
            page_cursor += count;
            output.qo_indptr[output.request_base + index + 1] =
                output.row_base + index + 1;
            output.kv_page_indptr[output.request_base + index + 1] =
                page_cursor;
        }
    }
    __syncthreads();

    if (lane >= lane_count) return;
    const bool active = valid && !sentinel;
    const std::uint32_t row = output.row_base + lane;
    const std::uint32_t request = output.request_base + lane;
    output.row_valid[row] = static_cast<std::uint8_t>(active);
    output.token_ids[row] = active ? token : 0;
    const auto* position =
        fixed_decode_pointer<std::uint32_t>(descriptor->position);
    output.position_ids[row] =
        active && position != nullptr ? *position : 0;
    output.kv_last_page_lens[request] =
        active
            ? ((kv_len - 1) % output.page_size) + 1
            : 1;
    output.w_page[row] = write_page;
    output.w_off[row] = write_offset;
    if (!active && output.rs_slot_ids != nullptr) {
        output.rs_slot_ids[request] = -1;
    }
    if (output.sample_indices != nullptr) {
        output.sample_indices[request] =
            static_cast<std::int32_t>(row);
    }

    const std::uint32_t page_cursor_base = page_offsets[lane];
    if (!active) {
        output.kv_page_indices[page_cursor_base] = output.dummy_page;
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
        output.kv_page_indices[page_cursor_base + page] =
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
        if (commit != nullptr) {
            commit[0] = 0;
            commit[1] = 1;
        }
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
    // Settlement notification runs on its OWN stream. `cudaLaunchHostFunc`
    // blocks the stream it is enqueued on until the driver's callback thread
    // wakes and returns, so hosting it on the compute stream stalled the GPU
    // between one wave's `k_settle_host_channels_batch` and the next wave's
    // `k_pull_validate_host_channels_batch` — measured 111 us fixed (callback
    // thread wakeup) + 0.44 us per lane, i.e. 224 us at 256 lanes. The next
    // wave was already submitted, so no amount of frame lookahead (k) could
    // hide it: a blocking node cannot be jumped by pre-queueing.
    //
    // `settlement_ready` orders the callback after the wave's settlement;
    // `settlement_callbacks_done` re-establishes the ordering the compute
    // stream used to get for free (see the wait before this wave's host
    // publications in `Dispatch::finish`).
    cudaEvent_t settlement_ready = nullptr;
    cudaEvent_t settlement_callbacks_done = nullptr;
    bool settlement_callbacks_recorded = false;
    std::vector<BoundInstance::CommitSnapshot> available_commit_snapshots;
    // Backing storage for `available_commit_snapshots`. Snapshots are two
    // words each, but they were allocated ONE PER INSTANCE, and each
    // `cudaHostAlloc` page-locks and maps a fresh page: 128 fresh lanes cost a
    // measured ~47ms of `begin_pass_a` on the fleet's first decode wave (steady
    // state is 17us). They are carved out of these slabs instead, so a whole
    // fleet costs two allocations. Snapshots therefore do NOT own their memory
    // — only these slabs are freed.
    std::vector<void*> commit_snapshot_device_slabs;
    std::vector<void*> commit_snapshot_host_slabs;
    // Private, non-blocking stream for seeding commit snapshots. The seed used
    // to be a blocking `cudaMemcpy`, which orders against the LEGACY DEFAULT
    // STREAM — so seeding the first decode wave's 128 lanes blocked the
    // submitting thread until the still-running prefill finished. That was the
    // entire measured 47ms `begin_pass_a` spike. A fresh (or quiesced,
    // recycled) snapshot is referenced by no launched kernel, so seeding it off
    // to the side and waiting only for that copy is both correct and free.
    cudaStream_t commit_seed_stream = nullptr;
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
    // One wave's settlement tail — batch completion notify, `cuda_settled`
    // emission, instance-close fence release — captured into locals before
    // the notify arena releases (the arena may be reset by the next wave).
    struct SettleRecord {
        PieRuntimeCallbacks runtime{};
        PieCompletion completion{};
        bool fire_timing_enabled = false;
        std::uint64_t finish_to_settle_us = 0;
        std::uint64_t settled_monotonic_ns = 0;
        std::size_t fire_count = 0;
        std::uint64_t membership_hash = 0;
        std::vector<std::shared_ptr<CallbackFence>> fences;
    };
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
    cudaStream_t notify_stream = nullptr;
    cudaStream_t group_streams[2] = {nullptr, nullptr};
    cudaStream_t signature_streams[kSignatureStreamCount] = {};
    bool attention_hook_coverage = false;
    std::uint32_t model_layers = 0;
    bool kv_envelopes_available = false;
    bool attn_page_mask_available = false;
    bool lora_available = false;
    std::function<void()> enable_kv_envelopes;
    // Hook-graph replay (stage 6 increment 4): a static device table of
    // layer indices [0, 1, …, model_layers). In prepared mode each
    // occurrence's lane metadata points `layer_base` at `&table[L]` instead
    // of relying on the eager path's per-invocation H2D memcpy of a STACK
    // variable — a captured memcpy node would re-read that dead stack slot
    // on every replay. Same values, address-stable, contents immutable.
    std::uint32_t* hook_layer_table = nullptr;
    std::uint32_t hook_layer_table_len = 0;
    // Hook-side device indirection (the Peel campaign's second half): the
    // score-pad gathers' lane→request indices, ONE u32 per pad in build
    // order, re-uploaded by every prepare pass into this address-stable
    // table. The captured pad kernel bakes `&table[ordinal]` instead of the
    // request VALUE, so a hooked lane changing rows (the split moving)
    // replays the same exec — the fingerprint keys the table base, not the
    // per-fire indices. Grows only (capacity doubling); growth moves the
    // base and honestly recaptures once.
    std::uint32_t* hook_pad_requests = nullptr;
    std::uint32_t hook_pad_request_capacity = 0;
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
    // The Prologue's statically-known put effects, computed at begin_host
    // for FramePrepare-time consumers (descriptor resolution, the stage_*
    // composition tables): they historically ran post-begin — after the
    // Prologue executed and recorded into `prior_put_slots` — but under
    // the frame split they run before the Prologue is enqueued. The live
    // sets above still fill at execution time only (the Prologue's own
    // stage-metadata build must NOT see its own effects).
    std::unordered_set<std::uint32_t> prologue_put_slots;
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
    std::vector<std::uint64_t> presampled_token_rows;
    const std::uint8_t* row_valid = nullptr;
    std::uint32_t row_valid_offset = 0;
};

// ── Hook prepared mode (stage 6 increment 4 + eager unification) ────────────
// One fire's hoisted attention-phase work, in body order. Filled by
// `Dispatch::prepare_attention_phases` — run by the batch engine for EVERY
// pure-decode hook fire, eager and graph alike — consumed by the
// prepared-mode branch of `execute_attention_phase` — exactly once per
// invocation, cursor-checked.

// A body-side gather that materializes one lane's padded `[kv_max]` AttnScore
// row from the layer's folded capture. Replaces the eager path's host-sized
// cudaMallocAsync + memset + D2D copy (`resolve_lane_attn_score`), whose
// capture-time sizes and offsets would go stale as the KV grows: here the
// live extent comes from the DEVICE CSR the prepare pass refreshes per fire,
// and the grid is a function of the program-declared ceiling alone.
struct HookScorePadLaunch {
    const float* folded = nullptr;              // arena score slot (stable)
    const std::uint32_t* folded_indptr = nullptr;  // arena score-rows CSR
    float* row = nullptr;                       // arena score-rows row
    // Index into Impl::hook_pad_requests, assigned in build order. The
    // kernel reads the lane's request from `&table[ordinal]` — uploaded
    // fresh by every prepare pass — so the captured launch survives the
    // lane changing rows (device indirection, the Peel campaign's second
    // half). The request VALUE lives only in the per-fire host vector.
    std::uint32_t ordinal = 0;
    std::uint32_t kv_max = 0;
};

struct HookPreparedGroup {
    std::unique_ptr<generated::GeneratedStagePrepared> prepared;
    std::vector<HookScorePadLaunch> score_pads;
};

struct HookPreparedInvocation {
    std::uint32_t layer = 0;
    std::vector<HookPreparedGroup> groups;
    // OnAttn invocations whose stages read AttnScore: at capture time the
    // prepared-mode execute validates the model actually published this
    // layer's capture INTO the planned arena slot — the one moment the
    // model's branch choice (windowed layer, xqa path, prefill capture) is
    // observable. A mismatch is a loud throw, never a silently-stale row.
    bool expects_scores = false;
    const float* expected_folded = nullptr;
    // Invocations whose stages write `attn_page_mask`: the prepared-mode
    // execute must perform the resolver's HOST half — tagging the model's
    // sink with `written_layer` so the body's compact branch fires — and
    // validates the model's sink is the arena block the pass planned
    // against (the addresses the captured kernels bake).
    bool marks_mask = false;
    const std::uint8_t* expected_mask_keep = nullptr;
    std::uint32_t expected_mask_stride = 0;
};

struct StagedLaunch::State {
    Dispatch::Impl* owner = nullptr;
    pie::driver::fire::LaunchView view{};
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
    // Score-pad lane→request indices staged by `prepare_attention_phases`
    // for the async upload into Impl::hook_pad_requests (the source must
    // outlive the pass; see the device-indirection note there).
    std::vector<std::uint32_t> pad_request_staging;
    DeviceHostChannelTicket* device_tickets = nullptr;
    // Frame split: device-composition lane tables staged at FramePrepare
    // (`stage_fixed_decode` / `stage_decode_envelopes` — the tables read
    // live registry ring cursors and the wave's channel-effect sets, valid
    // only at this wave's position in begin_host order). StepEnqueue's
    // `enqueue_*` halves claim the upload arena, patch the arena-relative
    // pointers, and launch the compose kernel.
    bool fixed_decode_staged = false;
    std::vector<FixedDecodeLane> fixed_decode_lanes;
    std::vector<std::uint32_t> fixed_decode_upload_values;
    std::vector<std::uint32_t> fixed_decode_translation_begin;
    std::vector<std::size_t> fixed_decode_position_offset;
    std::uint32_t fixed_decode_page_size = 0;
    std::uint32_t fixed_decode_device_pages = 0;
    Dispatch::FixedDecodeScope fixed_decode_scope{};
    bool decode_envelopes_staged = false;
    std::vector<DecodeEnvelopeLane> decode_envelope_lanes;
    std::size_t decode_envelope_lane_count = 0;
    std::uint32_t decode_envelope_template_pages = 0;
    // The `lora` sink's begin-time resolution: one entry per lane whose
    // program carries the sink, rebuilt each time the prologue executes.
    // Launch-owned because the prologue runs in `begin`, before the model
    // body (and its buffers) exist — `launch_lora_table` hands the frame a
    // borrowed view of exactly this storage. `lora_lane_sources` is parallel:
    // the lane each entry came from, so `update_launch_geometry` — which the
    // frame calls AFTER the prologue under the frame split — can re-stamp the
    // entries' token spans with the resolved geometry instead of leaving the
    // begin_host-time spans to go stale.
    std::vector<model::LoraLaneView> lora_lanes;
    std::vector<const StagedLane*> lora_lane_sources;
    std::uint32_t* device_layer = nullptr;
    cudaEvent_t source_ready = nullptr;
    cudaEvent_t phase_done[2] = {nullptr, nullptr};
    cudaEvent_t signature_ready = nullptr;
    cudaEvent_t signature_done[
        Dispatch::Impl::kSignatureStreamCount] = {};
    std::array<std::uint32_t, 4> phase_invocations{};
    bool active = true;
    bool failed = false;
    // Hook-graph prepared mode (stage 6 increment 4). When set, every
    // attention-phase invocation was prepared at fire level and
    // `execute_attention_phase` only replays launches, cursor-checked
    // against the exact (phase, layer) order the prepare pass recorded.
    // `prepared_attn[phase - PTIR_STAGE_ON_ATTN_PROJ]`.
    bool hook_graph_prepared = false;
    std::array<std::vector<HookPreparedInvocation>, 2> prepared_attn;
    std::array<std::size_t, 2> prepared_cursor{};
    // Host CSR of folded score offsets, uploaded to the arena's score-rows
    // block by the prepare pass; owned here so the upload's source outlives
    // the enqueue.
    std::vector<std::uint32_t> hook_folded_offsets_h;
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
    // Scatter-kernel descriptors, materialized from the three vectors below
    // only when that transport is selected. Pinned so the upload is a real
    // async copy rather than a staging round trip on the lane thread.
    PinnedHostVector<HostPublishCopy> publish_copies;
    // Reused [begin, end) scratch for the destination-overlap test, so the
    // per-wave check sorts without reallocating.
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> overlap_scratch;
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
        publish_copies.clear();
        overlap_scratch.clear();
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
    if (hook_layer_table != nullptr) {
        cudaFree(hook_layer_table);
    }
    if (hook_pad_requests != nullptr) {
        cudaFree(hook_pad_requests);
    }
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

// Settle one wave's record: batch completion notify, `cuda_settled`
// emission, instance-close fence release. Publication (endpoint wakes,
// terminal cells, doorbells) happened in the wave's own callback and never
// rides these records.
void settle_wave_record(
    Dispatch::Impl* impl,
    Dispatch::Impl::SettleRecord& record) noexcept {
    const bool notify =
        record.runtime.notify != nullptr &&
        (impl == nullptr ||
         !impl->shutting_down.load(std::memory_order_acquire));
    if (notify && record.completion.wait_id != 0) {
        record.runtime.notify(
            record.runtime.ctx,
            record.completion.wait_id,
            record.completion.target_epoch);
    }
    if (record.fire_timing_enabled) {
        fire_timing::enqueue_settled({
            .wave_id = record.completion.wait_id,
            .fire_count = record.fire_count,
            .membership_hash = record.membership_hash,
            .finish_to_settle_us = record.finish_to_settle_us,
            .settled_monotonic_ns = record.settled_monotonic_ns,
        });
    }
    for (const auto& fence : record.fences) {
        if (fence->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            fence->pending.notify_all();
        }
    }
    record.fences.clear();
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
            entry.commit_host != nullptr && entry.commit_host[0] != 0;
        // Word [1] of the snapshot: a compose fail-stop (chain kill) is a
        // deterministic per-lane fault — FAILED with poisoned channels,
        // never RETRY (v14 reserves RETRY for host staging violations).
        const bool killed =
            entry.commit_host != nullptr && entry.commit_host[1] != 0;
        const bool failed = entry.poison || killed;
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
    // Capture the settlement tail into locals before the arena releases:
    // once `in_use` clears, `ctx` may be reset by the next wave on the lane
    // thread. No native instance/channel state is touched after a batch
    // wake: a woken runtime thread may immediately close the instance.
    Dispatch::Impl* impl = ctx->impl;
    Dispatch::Impl::SettleRecord self{};
    self.runtime = ctx->runtime;
    self.completion = ctx->completion;
    self.fire_timing_enabled = ctx->fire_timing_enabled;
    self.finish_to_settle_us = finish_to_settle_us;
    self.settled_monotonic_ns = settled_monotonic_ns;
    self.fire_count = ctx->fire_count;
    self.membership_hash = ctx->membership_hash;
    self.fences = std::move(ctx->callback_fences);
    ctx->callback_fences.clear();
    ctx->commit_lanes.clear();
    ctx->settlement_lanes.clear();
    ctx->publish_copies.clear();
    ctx->in_use.store(false, std::memory_order_release);
    settle_wave_record(impl, self);
}

namespace {
void close_bound_instance(
    Dispatch::Impl& s,
    std::uint64_t instance_id,
    bool retain_resources = true);

// True if any two publication destinations name overlapping bytes, in which
// case only a sequential enqueue gives them a defined order.
//
// Sorting rather than comparing every pair: a 512-wide wave publishes 512
// cells, and the quadratic form cost ~130 µs of lane-thread time per wave —
// the same order as the whole settlement prologue it sits in.
bool host_publish_destinations_overlap(NotifyContext& context) {
    const std::size_t count = context.copy_destinations.size();
    if (count < 2) return false;
    context.overlap_scratch.clear();
    context.overlap_scratch.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto begin =
            reinterpret_cast<std::uintptr_t>(context.copy_destinations[index]);
        const std::size_t bytes = context.copy_sizes[index];
        if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
            return true;
        }
        context.overlap_scratch.emplace_back(begin, begin + bytes);
    }
    std::sort(
        context.overlap_scratch.begin(), context.overlap_scratch.end());
    for (std::size_t index = 1; index < count; ++index) {
        // Sorted by start, so a range can only overlap its predecessor's
        // reach; empty ranges never overlap.
        if (context.overlap_scratch[index].first <
            context.overlap_scratch[index - 1].second) {
            return true;
        }
    }
    return false;
}

// How a wave's host-visible output cells reach their pinned mirrors.
//
// `Scatter` is the default for the decode-shaped waves that dominate: one
// kernel writes every cell straight into the mapped mirrors, so a wave costs
// a single launch instead of one copy-engine transfer per (lane, host-read
// output channel). `Batched` keeps the copy engine for cells big enough that
// DMA bandwidth beats scattered PCIe writes. `Sequential` is the
// always-available fallback, and the only path that gives overlapping
// destinations a defined last-writer-wins order.
enum class HostPublishTransport {
    Sequential,
    Batched,
    Scatter,
};

// Cells at or below this size are published by the scatter kernel. A
// copy-engine D2H costs ~2 µs of GPU time almost independently of size, so
// below a few KB that fixed cost, not bandwidth, decides; above it the copy
// engine wins and the work goes back to DMA.
constexpr std::size_t kMaxScatterCellBytes = 4096;

HostPublishTransport select_host_publish_transport(
    NotifyContext& context,
    cudaStream_t batch_stream) {
    if (context.copy_destinations.size() <= 1) {
        return HostPublishTransport::Sequential;
    }
    // Neither aggregated transport orders its entries against each other.
    if (host_publish_destinations_overlap(context)) {
        return HostPublishTransport::Sequential;
    }
    const bool cells_are_small = std::all_of(
        context.copy_sizes.begin(),
        context.copy_sizes.end(),
        [](std::size_t bytes) { return bytes <= kMaxScatterCellBytes; });
    if (cells_are_small) return HostPublishTransport::Scatter;
#if CUDART_VERSION >= 12080
    if (batch_stream != nullptr) return HostPublishTransport::Batched;
#else
    static_cast<void>(batch_stream);
#endif
    return HostPublishTransport::Sequential;
}

void enqueue_host_publish_copies(
    NotifyContext& context,
    cudaStream_t stream,
    HostPublishTransport transport) {
    if (context.copy_destinations.empty()) return;
    if (transport == HostPublishTransport::Scatter) {
        context.publish_copies.clear();
        for (std::size_t index = 0;
             index < context.copy_destinations.size();
             ++index) {
            context.publish_copies.push_back(HostPublishCopy{
                .destination = context.copy_destinations[index],
                .source = context.copy_sources[index],
                .bytes =
                    static_cast<std::uint32_t>(context.copy_sizes[index]),
            });
        }
        launch_scatter_host_publish_copies(
            context.publish_copies.values(), stream);
        return;
    }
#if CUDART_VERSION >= 12080
    if (transport == HostPublishTransport::Batched) {
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
        cudaStream_t notify_stream,
        std::unique_lock<std::mutex> lock)
        : context_(context),
          stream_(stream),
          auxiliary_stream_(auxiliary_stream),
          notify_stream_(notify_stream),
          lock_(std::move(lock)) {}
    ~NotifyContextLease() {
        if (context_ != nullptr) {
            const cudaError_t auxiliary_status =
                auxiliary_stream_ == stream_
                ? cudaSuccess
                : cudaStreamSynchronize(auxiliary_stream_);
            // The settlement callback rides its own stream now, so an
            // exception thrown after it was enqueued must drain that stream
            // too before the fences it releases are touched here.
            const cudaError_t notify_status =
                (notify_stream_ == nullptr || notify_stream_ == stream_)
                ? cudaSuccess
                : cudaStreamSynchronize(notify_stream_);
            const cudaError_t status =
                cudaStreamSynchronize(stream_);
            if (auxiliary_status == cudaSuccess &&
                notify_status == cudaSuccess &&
                status == cudaSuccess) {
                release_callback_fences(*context_);
                context_->in_use.store(false, std::memory_order_release);
            } else {
                std::fprintf(
                    stderr,
                    "[pie-driver-cuda] settlement cleanup stream sync failed: %s / %s / %s\n",
                    cudaGetErrorString(auxiliary_status),
                    cudaGetErrorString(notify_status),
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
    cudaStream_t notify_stream_;
    std::unique_lock<std::mutex> lock_;
};

// Batch-level channel budget (§4.3 availability + reader capacity): members
// of one batch that share a channel are validated against the AGGREGATE of
// their planned ring consumes and reader publishes. Checked one-by-one, two
// members could both pass on the last available entry/slot and the second
// would die as a device-side poison instead of a synchronous rejection.
std::vector<DeviceHostChannelTicket> build_channel_tickets(
    const pie::driver::fire::LaunchView& view,
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
        //
        // Flag-free tickets (read-only channels) are KEPT: every device-
        // side consumer (pull-validate, settle, publish lookup) is
        // flag-gated, so they are inert there — but they carry the wave's
        // channel-cursor positions to the stage-metadata builders, which
        // under the frame split may run after LATER steps' sequence
        // applies have moved the registry mirrors (a live-mirror fallback
        // there binds another wave's cells).
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
    const pie::driver::fire::LaunchView& view,
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

// Does this stage name a second-party kernel? Used to decide whether the fire's
// KV geometry has to be resolved for the lane at all: resolving it is cheap but
// it THROWS when unavailable, so it must only run for stages that need it.
bool stage_calls_kernel(
    const plan::StagePlan& stage,
    std::string_view name) {
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.tag != PTIR_OP_KERNEL_CALL) continue;
        if (op.name_idx < stage.names.size() &&
            stage.names[op.name_idx] == name) {
            return true;
        }
    }
    return false;
}

bool stage_calls_sink(
    const plan::StagePlan& stage,
    std::string_view name) {
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.tag != PTIR_OP_SINK_CALL) continue;
        if (op.name_idx < stage.names.size() &&
            stage.names[op.name_idx] == name) {
            return true;
        }
    }
    return false;
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

// FramePrepare-time channel-cursor resolution: the wave's window comes
// from its TICKETS (engine-sequenced expected positions), never from the
// live registry mirrors — under the frame split the mirrors advance at
// each wave's enqueue position, which has not happened yet at prepare
// time. Channels the engine left unsequenced (kNoChannelTicket) are
// apply-invariant, so their live mirror IS their window.
struct PreparedCursor {
    std::uint32_t head_index = 0;
    std::uint32_t tail_index = 0;
};

PreparedCursor lane_ticket_window(
    const StagedLane& lane,
    std::uint32_t slot,
    const DeviceChannelRegistry& channels) {
    PreparedCursor cursor{
        channels.host_head(slot),
        channels.host_tail(slot),
    };
    for (const DeviceHostChannelTicket& ticket : lane.tickets) {
        if (ticket.slot != slot) continue;
        if (ticket.expected_head != kNoChannelTicket) {
            cursor.head_index = static_cast<std::uint32_t>(
                ticket.expected_head % ticket.cap1);
        }
        if (ticket.expected_tail != kNoChannelTicket) {
            cursor.tail_index = static_cast<std::uint32_t>(
                ticket.expected_tail % ticket.cap1);
        }
        break;
    }
    return cursor;
}

// Hook-graph mode's device-resolved AttnScore materialization (stage 6
// increment 4). One padded `[kv_max]` row per (layer, lane), gathered from
// the layer's folded capture INSIDE the body: the live extent comes from the
// device CSR (refreshed before every replay by the captured
// `LayerScoreCapture` upload reading prepare-refreshed host storage), the
// grid from the program-declared ceiling alone — so the recorded launch is
// replay-stable while the eager path's host-sized malloc+memset+memcpy
// (`resolve_lane_attn_score`) is not. Slack beyond the live extent reads as
// 0.0, the intrinsic's defined value for positions that do not exist —
// byte-identical to the eager path's zero-filled row.
__global__ void k_hook_attn_score_pad(
    const float* __restrict__ folded,
    const std::uint32_t* __restrict__ folded_indptr,
    const std::uint32_t* __restrict__ request_d,
    float* __restrict__ row,
    std::uint32_t kv_max) {
    // Device-indirected lane→request mapping (Peel campaign half two):
    // the prepare pass re-uploads the index per fire, so a captured
    // launch replays across row splits instead of baking the request.
    const std::uint32_t request = *request_d;
    const std::uint32_t begin = folded_indptr[request];
    const std::uint32_t end = folded_indptr[request + 1];
    const std::uint32_t kv_len = end >= begin ? end - begin : 0u;
    for (std::uint32_t index =
             blockIdx.x * blockDim.x + threadIdx.x;
         index < kv_max;
         index += gridDim.x * blockDim.x) {
        row[index] = index < kv_len ? folded[begin + index] : 0.0f;
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
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &impl_->notify_stream, cudaStreamNonBlocking));
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
    result.generated_host_sources = generated.host_sources;
    result.generated_driver_sources = generated.driver_sources;
    result.generated_stage_cache_entries = generated.stage_entries;
    result.generated_program_cache_entries = generated.program_entries;
    result.generated_negative_cache_entries = generated.negative_entries;
    result.region_host_supplied = impl_->cache.region_stats().host_supplied;
    result.channel_slot_capacity = impl_->channels.capacity_slots();
    return result;
}

std::vector<std::uint32_t> Dispatch::mtp_draft_rows(
    const pie::driver::fire::LaunchView& view) const {
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
    if (impl_->notify_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(impl_->notify_stream));
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
    for (void* slab : impl_->commit_snapshot_device_slabs) {
        if (slab != nullptr) CUDA_CHECK(cudaFree(slab));
    }
    for (void* slab : impl_->commit_snapshot_host_slabs) {
        if (slab != nullptr) CUDA_CHECK(cudaFreeHost(slab));
    }
    if (impl_->commit_seed_stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(impl_->commit_seed_stream));
        impl_->commit_seed_stream = nullptr;
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
    if (impl_->notify_stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(impl_->notify_stream));
        impl_->notify_stream = nullptr;
    }
    if (impl_->publications_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(impl_->publications_done));
        impl_->publications_done = nullptr;
    }
    if (impl_->settlement_ready != nullptr) {
        CUDA_CHECK(cudaEventDestroy(impl_->settlement_ready));
        impl_->settlement_ready = nullptr;
    }
    if (impl_->settlement_callbacks_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(impl_->settlement_callbacks_done));
        impl_->settlement_callbacks_done = nullptr;
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

// Carve a slab's worth of commit snapshots into the pool. Each snapshot is
// two words, but neighbouring lanes write theirs concurrently from both host
// and device, so they are spread `kCommitSnapshotStride` bytes apart (one
// device cache line) rather than packed — packing them would trade the
// allocation cost for false sharing on the settle path.
constexpr std::size_t kCommitSnapshotStride = 128;
constexpr std::size_t kCommitSnapshotChunk = 256;

void refill_commit_snapshot_pool(Dispatch::Impl& owner) {
    static_assert(
        BoundInstance::CommitSnapshot::kWords * sizeof(std::uint32_t) <=
            kCommitSnapshotStride,
        "commit snapshot does not fit its stride");
    const std::size_t bytes = kCommitSnapshotStride * kCommitSnapshotChunk;

    void* device_slab = nullptr;
    CUDA_CHECK(cudaMalloc(&device_slab, bytes));
    try {
        owner.commit_snapshot_device_slabs.push_back(device_slab);
    } catch (...) {
        cudaFree(device_slab);
        throw;
    }

    // Same mapped-host preference as before, just once per slab instead of
    // once per instance.
    static const bool try_mapping = [] {
        int device = 0;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            return false;
        }
        return properties.canMapHostMemory != 0 &&
               std::getenv("PIE_CUDA_DISABLE_MAPPED_COMMITS") == nullptr;
    }();

    void* host_slab = nullptr;
    void* host_device_slab = nullptr;
    if (try_mapping) {
        const cudaError_t status = cudaHostAlloc(
            &host_slab, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (status != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            host_slab = nullptr;
        } else if (cudaHostGetDevicePointer(&host_device_slab, host_slab, 0) !=
                   cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            cudaFreeHost(host_slab);
            host_slab = nullptr;
            host_device_slab = nullptr;
        }
    }
    if (host_slab == nullptr) {
        CUDA_CHECK(cudaMallocHost(&host_slab, bytes));
        host_device_slab = nullptr;
    }
    try {
        owner.commit_snapshot_host_slabs.push_back(host_slab);
        owner.available_commit_snapshots.reserve(
            owner.available_commit_snapshots.size() + kCommitSnapshotChunk);
    } catch (...) {
        cudaFreeHost(host_slab);
        throw;
    }

    auto* device_base = static_cast<std::byte*>(device_slab);
    auto* host_base = static_cast<std::byte*>(host_slab);
    auto* host_device_base = static_cast<std::byte*>(host_device_slab);
    for (std::size_t i = 0; i < kCommitSnapshotChunk; ++i) {
        const std::size_t offset = i * kCommitSnapshotStride;
        BoundInstance::CommitSnapshot snapshot{};
        snapshot.device =
            reinterpret_cast<std::uint32_t*>(device_base + offset);
        snapshot.host = reinterpret_cast<std::uint32_t*>(host_base + offset);
        snapshot.host_device =
            host_device_base == nullptr
                ? nullptr
                : reinterpret_cast<std::uint32_t*>(host_device_base + offset);
        owner.available_commit_snapshots.push_back(snapshot);
    }
}

BoundInstance::CommitSnapshot& commit_snapshot(
    Dispatch::Impl& owner,
    BoundInstance& bound,
    std::size_t index) {
    while (bound.commit_snapshots.size() <= index) {
        if (owner.available_commit_snapshots.empty()) {
            refill_commit_snapshot_pool(owner);
        }
        BoundInstance::CommitSnapshot snapshot =
            owner.available_commit_snapshots.back();
        owner.available_commit_snapshots.pop_back();
        // A pooled word still holds the previous instance's verdict, and a
        // freshly carved one holds nothing at all. Either way this snapshot
        // carries no information about any wave until a pull-validate seeds
        // it, so no reader may treat its contents as a verdict.
        snapshot.ever_validated = false;
        // W1.6 reads this word at PREPARE, i.e. before the fire's own
        // pull-validate runs, so it is really the PREVIOUS fire's commit.
        // A ring index used for the first time has no previous fire — the
        // host's bind-time seeds are its predecessor — so it starts READY.
        // Recycled snapshots carry a retired instance's value and must be
        // re-seeded for the same reason.
        const std::uint32_t seed[BoundInstance::CommitSnapshot::kWords] = {
            1u, 0u};
        if (owner.commit_seed_stream == nullptr) {
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &owner.commit_seed_stream, cudaStreamNonBlocking));
        }
        CUDA_CHECK(cudaMemcpyAsync(
            snapshot.device,
            seed,
            sizeof(seed),
            cudaMemcpyHostToDevice,
            owner.commit_seed_stream));
        CUDA_CHECK(cudaStreamSynchronize(owner.commit_seed_stream));
        std::memcpy(snapshot.host, seed, sizeof(seed));
        bound.commit_snapshots.push_back(snapshot);
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
            // Slab-carved (see `refill_commit_snapshot_pool`): the snapshot
            // owns nothing, so dropping it past the pool cap just leaks its
            // slot back to no one. The slabs are freed with the Impl.
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
    // The reaper waits on CUDA events, and CUDA's current device is
    // thread-local. Capture the device of whichever rank is starting the
    // thread; a fresh thread would otherwise default to device 0 and, under
    // TP, synchronize the wrong context.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    s.instance_reaper = std::thread([&s, device]() {
        CUDA_CHECK(cudaSetDevice(device));
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
                                   const PieLaunchPackage& package,
                                   PieEmittedKernelSlice emitted,
                                   PieRegionAnalysisSlice region_analysis,
                                   std::string* err) {
    if (err) err->clear();
    std::string derr;
    const Trace* trace = impl_->cache.adopt(program_hash, package, &derr);
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
    std::string region_error;
    if (!impl_->cache.adopt_host_region_analysis(
            program_hash,
            region_analysis.ptr,
            region_analysis.len,
            &region_error)) {
        if (err) *err = region_error;
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    bool needs_kv_envelopes = false;
    for (const plan::StagePlan& stage : *plans) {
        // The host already decided whether the grouped interpreter can run
        // this stage, and said so in the envelope. Re-deriving it here is how
        // the two verdicts drift.
        if (!stage.grouped.valid) {
            if (err) {
                *err = stage.grouped.error.empty()
                    ? std::string("ptir stage is not executable by the CUDA "
                                  "grouped runtime")
                    : stage.grouped.error;
            }
            return PIE_STATUS_UNSUPPORTED;
        }
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
                // Sinks are named like second-party kernels, and the same rule
                // applies: a name the backend cannot honour has to be refused
                // at bind rather than silently skipped, because a skipped
                // configuration sink is a program whose selection never took
                // effect while every test still passes.
                const std::string& name =
                    op.name_idx < stage.names.size()
                        ? stage.names[op.name_idx]
                        : std::string();
                if (name == "attn_page_mask") {
                    if (stage.stage != PTIR_STAGE_ON_ATTN_PROJ) {
                        // `SinkScope::Attention` also admits `Prologue`, but a
                        // prologue mask cannot name a layer, and this consumer
                        // is per-layer. Refusing is better than honouring it
                        // for layer 0 only.
                        if (err) {
                            *err =
                                "attn_page_mask is only available at the "
                                "on_attn_proj stage";
                        }
                        return PIE_STATUS_UNSUPPORTED;
                    }
                    if (!impl_->attn_page_mask_available) {
                        if (err) {
                            *err =
                                "attn_page_mask requires a decode attention "
                                "path whose plan does not depend on the page "
                                "count";
                        }
                        return PIE_STATUS_UNSUPPORTED;
                    }
                    continue;
                }
                if (name == "lora") {
                    if (stage.stage != PTIR_STAGE_PROLOGUE) {
                        // Pass-wide sinks are confined to the prologue by the
                        // compiler (T11), and this runtime additionally
                        // RESOLVES the sink at begin time, when the prologue
                        // runs. Any other placement would resolve after the
                        // projections it configures.
                        if (err) {
                            *err =
                                "lora is only available at the prologue "
                                "stage";
                        }
                        return PIE_STATUS_UNSUPPORTED;
                    }
                    if (!impl_->lora_available) {
                        // Refused at bind, not skipped at fire: a lora-naming
                        // program on a model that does not apply the delta is
                        // a request whose adapter would silently never take
                        // effect while every sample still returns.
                        if (err) {
                            *err =
                                "lora requires a model whose projection path "
                                "applies the low-rank delta (capability "
                                "has_lora); the active CUDA model does not "
                                "advertise it";
                        }
                        return PIE_STATUS_UNSUPPORTED;
                    }
                    continue;
                }
                if (err) {
                    *err =
                        "ptir model sinks are not implemented by the active CUDA model";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
            if (op.tag == PTIR_OP_KERNEL_CALL) {
                // Second-party kernels are named, not generic: the fused
                // runtime dispatches them by name, so a name it cannot launch
                // has to be refused at bind rather than silently skipped.
                const std::string& name =
                    op.name_idx < stage.names.size()
                        ? stage.names[op.name_idx]
                        : std::string();
                if (name != "envelope_dot") {
                    if (err) {
                        *err =
                            "ptir second-party kernel is not implemented by "
                            "the active CUDA model";
                    }
                    return PIE_STATUS_UNSUPPORTED;
                }
                if (!impl_->kv_envelopes_available) {
                    if (err) {
                        *err =
                            "envelope_dot requires a native bf16 NHD kv cache "
                            "and a post-rope query on this model";
                    }
                    return PIE_STATUS_UNSUPPORTED;
                }
                needs_kv_envelopes = true;
                if (stage.stage != PTIR_STAGE_ON_ATTN_PROJ &&
                    stage.stage != PTIR_STAGE_ON_ATTN) {
                    if (err) {
                        *err =
                            "envelope_dot is only available at an attention "
                            "stage";
                    }
                    return PIE_STATUS_UNSUPPORTED;
                }
                continue;
            }
            if (op.tag != PTIR_OP_INTRINSIC_VAL) continue;
            const bool valid =
                (stage.stage == PTIR_STAGE_EPILOGUE &&
                 (op.intr == PTIR_INTR_LOGITS ||
                  op.intr == PTIR_INTR_MTP_LOGITS)) ||
                ((stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
                  stage.stage == PTIR_STAGE_ON_ATTN) &&
                 (op.intr == PTIR_INTR_QUERY ||
                  op.intr == PTIR_INTR_LAYER)) ||
                (stage.stage == PTIR_STAGE_ON_ATTN &&
                 op.intr == PTIR_INTR_ATTN_SCORE);
            if (!valid) {
                if (err) {
                    *err =
                        "ptir intrinsic is unavailable at its declared CUDA phase";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
        }
    }
    if (needs_kv_envelopes && impl_->enable_kv_envelopes) {
        // Envelopes cost 4/page_size of the KV cache, so they are allocated
        // LAZILY: the first program that names `envelope_dot` pays for them and
        // every model that never observes attention pays nothing. Registration
        // is a control-plane call, so allocating + seeding here (which
        // synchronizes) does not race a fire.
        try {
            impl_->enable_kv_envelopes();
        } catch (const std::exception& e) {
            if (err) {
                *err = std::string("kv envelopes are unavailable: ") + e.what();
            }
            return PIE_STATUS_UNSUPPORTED;
        }
    }
    generated::CompileFailureKind compile_failure =
        generated::CompileFailureKind::None;
    std::string compile_error;
    // Index the host's kernels by (stage, region) so the compiler can look one
    // up without rescanning; an empty table leaves every lookup null and the
    // in-driver emitter runs exactly as before.
    HostEmittedKernels host_kernels(emitted);
    host_kernels.adopt(region_analysis);
    const auto compiled_program = impl_->fused_modules.compile_program(
            program_hash,
            *plans,
            compile_failure,
            compile_error,
            HostEmittedKernels::lookup,
            &host_kernels,
            HostEmittedKernels::lookup_region);
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
    const Trace* trace = impl_->cache.find(program_hash, &derr);
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
    // Always defer-if-attached: the registry's per-slot refcount already
    // tracks live instance attachments exactly (bound at view bind, released
    // at instance destruction), and `release()` retires a pending-close slot
    // at refcount zero. The previous any_of scan over every live instance's
    // channel list was O(instances × channels) per close — a cohort teardown
    // paid ~9k comparisons × 36.9k closes — and it passed the flag INVERTED
    // (attached → defer=false), so a close racing an instance's teardown
    // hard-failed, the engine dropped it, and the slot leaked permanently:
    // ~15k zombie slots per 2048-fleet run starved the retained-storage pool
    // and turned steady-state registrations back into fresh allocations.
    return impl_->channels.close_endpoint(
               channel_id, err, /*defer_if_attached=*/true)
        ? PIE_STATUS_OK
        : (impl_->channels.contains(channel_id)
               ? PIE_STATUS_INVALID_ARGUMENT
               : PIE_STATUS_CLOSED);
}

int Dispatch::validate_launch(
    const pie::driver::fire::LaunchView& view,
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

void Dispatch::set_kv_envelopes_available(
    bool available,
    std::function<void()> enable) {
    impl_->kv_envelopes_available = available;
    impl_->enable_kv_envelopes = available ? std::move(enable) : nullptr;
}

void Dispatch::set_attn_page_mask_available(bool available) {
    impl_->attn_page_mask_available = available;
}

void Dispatch::set_lora_available(bool available) {
    impl_->lora_available = available;
}

model::LoraTable Dispatch::launch_lora_table(
    const StagedLaunch& launch) const {
    const StagedLaunch::State& state = *launch.state_;
    return model::LoraTable{
        .lanes = state.lora_lanes.empty() ? nullptr : state.lora_lanes.data(),
        .count = static_cast<std::uint32_t>(state.lora_lanes.size()),
    };
}

void Dispatch::set_attention_hook_coverage(
    bool supported,
    std::uint32_t model_layers) {
    impl_->attention_hook_coverage = supported;
    impl_->model_layers = supported ? model_layers : 0;
}

bool Dispatch::launch_has_attention_stages(
    const pie::driver::fire::LaunchView& view) const {
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

std::uint32_t Dispatch::launch_hook_free_prefix_rows(
    const pie::driver::fire::LaunchView& view) const {
    const std::uint32_t derived = derive_hook_free_prefix_rows(view);
    // B (the fire planner's first consumed lowering): when the scheduler
    // sent its planned prefix — fire_plan's qkv_postprocess site,
    // converted to wire rows through the same attribution CSR — the plan
    // OWNS the answer, and this derivation becomes the cross-check: the
    // admission-time hook stamps and the compiled stage plans must agree,
    // and a drift is a bug to refuse loudly, not a case to split.
    if (view.planned_hook_free_prefix_rows ==
        PIE_HOOK_FREE_PREFIX_UNPLANNED) {
        return derived;
    }
    if (view.planned_hook_free_prefix_rows != derived) {
        throw std::runtime_error(
            "planned hook-free prefix (" +
            std::to_string(view.planned_hook_free_prefix_rows) +
            " wire rows) disagrees with the compiled-plan derivation (" +
            std::to_string(derived) +
            "): the scheduler's hook stamps and the driver's stage plans "
            "drifted");
    }
    // Positive engagement evidence for the parity harness: without it a
    // wiring bug that never sends a plan would green-run vacuously on the
    // derivation alone.
    if (std::getenv("PIE_DECLARED_FORWARD_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "[hook-prefix-plan] planned=%u (cross-checked)\n",
                     view.planned_hook_free_prefix_rows);
    }
    return view.planned_hook_free_prefix_rows;
}

std::uint32_t Dispatch::derive_hook_free_prefix_rows(
    const pie::driver::fire::LaunchView& view) const {
    const std::size_t n_prog = view.ptir_program_hashes.size();
    // Per-program row attribution is the only way to LOCATE a hook, so its
    // absence means "no fast prefix", not "no hooks".
    if (view.ptir_program_row_indptr.size() != n_prog + 1) return 0;
    const std::uint32_t* row_indptr = view.ptir_program_row_indptr.data();
    const std::uint32_t total_rows = row_indptr[n_prog];
    if (total_rows == 0) return 0;

    const auto has_attention_stages = [&](std::size_t program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) return false;
        return std::any_of(
            plans->begin(), plans->end(),
            [](const plan::StagePlan& stage) {
                return stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
                    stage.stage == PTIR_STAGE_ON_ATTN;
            });
    };

    std::uint32_t first_hook_row = total_rows;
    for (std::size_t program = 0; program < n_prog; ++program) {
        if (!has_attention_stages(program)) continue;
        const std::uint32_t lo = row_indptr[program];
        const std::uint32_t hi = row_indptr[program + 1];
        // A hook-carrying program with an empty wire span (device-resolved
        // geometry placeholder) cannot be located among the rows, so no row
        // can be proven hook-free.
        if (hi <= lo) return 0;
        first_hook_row = std::min(first_hook_row, lo);
    }
    // Every row below the minimum hook-program row start is hook-free by
    // construction (spans are contiguous [lo, hi)), so the minimum IS the
    // fast prefix. Hook-free rows that happen to sit above it fall into the
    // slow tail — a lost optimisation, not a correctness question; making
    // them fast is the row-ordering half of this work (scheduler-side).
    return first_hook_row;
}

bool Dispatch::launch_wants_attn_score(
    const pie::driver::fire::LaunchView& view) const {
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) continue;
        for (const plan::StagePlan& stage : *plans) {
            if (stage.stage != PTIR_STAGE_ON_ATTN) continue;
            if (stage_uses_intrinsic(stage, PTIR_INTR_ATTN_SCORE)) {
                return true;
            }
        }
    }
    return false;
}

bool Dispatch::launch_epilogue_is_greedy_argmax(
    const pie::driver::fire::LaunchView& view,
    std::uint32_t vocab) const {
    bool saw_epilogue = false;
    // The verdict is a property of the program, and a launch is overwhelmingly
    // the same guest replicated across every lane, so the same hash would
    // otherwise be re-analysed once per lane -- op scans plus two vector
    // allocations each, hundreds of times per fire.
    std::vector<std::uint64_t> analysed;
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const std::uint64_t hash = view.ptir_program_hashes.data()[program];
        if (std::find(analysed.begin(), analysed.end(), hash) !=
            analysed.end()) {
            continue;
        }
        analysed.push_back(hash);
        const auto* plans = impl_->cache.plans(hash);
        if (plans == nullptr) return false;
        const auto generated = impl_->fused_modules.program(hash);
        if (generated == nullptr ||
            generated->stages.size() != plans->size()) {
            return false;
        }
        for (std::size_t index = 0; index < plans->size(); ++index) {
            const plan::StagePlan& stage = (*plans)[index];
            if (stage.stage != PTIR_STAGE_EPILOGUE) continue;
            // A program may declare a vocabulary narrower than the weight's,
            // and the materialising path honours it by scanning only that many
            // columns. The fused reduction runs over the whole weight, so it
            // could return a token the narrowed program can never emit.
            //
            // `stage_logits_vocab` throws on a malformed program. Declining is
            // the right answer here rather than propagating: this is a
            // question about an optimisation, and `finish` still rejects the
            // same program at the same place it always did.
            std::uint32_t logical_vocab = 0;
            try {
                logical_vocab = stage_logits_vocab(&stage, vocab);
            } catch (const std::exception&) {
                return false;
            }
            if (logical_vocab != vocab) return false;
            const auto& executable = generated->stages[index];
            if (executable == nullptr) return false;
            std::string unused;
            if (!generated::generated_stage_supported(
                    *executable, stage, &unused)) {
                return false;
            }
            if (!generated::generated_stage_is_compact_argmax(
                    stage, *executable)) {
                return false;
            }
            saw_epilogue = true;
        }
    }
    return saw_epilogue;
}

bool Dispatch::launch_wants_page_mask(
    const pie::driver::fire::LaunchView& view) const {
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) continue;
        for (const plan::StagePlan& stage : *plans) {
            if (stage.stage != PTIR_STAGE_ON_ATTN_PROJ) continue;
            for (const plan::NormalizedOp& normalized : stage.ops) {
                if (normalized.op.tag != PTIR_OP_SINK_CALL) continue;
                const std::uint32_t name_idx = normalized.op.name_idx;
                if (name_idx < stage.names.size() &&
                    stage.names[name_idx] == "attn_page_mask") {
                    return true;
                }
            }
        }
    }
    return false;
}

bool Dispatch::launch_wants_lora(
    const pie::driver::fire::LaunchView& view) const {
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) continue;
        for (const plan::StagePlan& stage : *plans) {
            if (stage.stage != PTIR_STAGE_PROLOGUE) continue;
            for (const plan::NormalizedOp& normalized : stage.ops) {
                if (normalized.op.tag != PTIR_OP_SINK_CALL) continue;
                const std::uint32_t name_idx = normalized.op.name_idx;
                if (name_idx < stage.names.size() &&
                    stage.names[name_idx] == "lora") {
                    return true;
                }
            }
        }
    }
    return false;
}

bool Dispatch::has_decode_envelopes(
    const pie::driver::fire::LaunchView& view) const {
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
    const pie::driver::fire::LaunchView& view,
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

// Resolve one lane's slice of the fire's KV geometry so `envelope_dot` can
// score the pages this lane's request actually owns.
//
// The lane->request mapping goes through `qo_indptr_h`: `make_staged_binding`
// already offsets the query by `token_start * query_columns`, and
// `qo_indptr_h[r]` IS request r's token start, so the two agree by
// construction. A lane whose token start is not a request boundary is a wiring
// bug and throws rather than scoring another request's pages.
GroupedLaneEnvelope resolve_lane_envelope(
    const StagedLane& lane,
    const float* query_base,
    std::uint32_t query_columns,
    std::uint32_t layer,
    const model::StageHookSideband& sideband) {
    const model::AttentionObservation* obs = sideband.observation;
    if (obs == nullptr || !obs->usable()) {
        throw std::runtime_error(
            "envelope_dot ran outside a model body with kv geometry");
    }
    if (query_base == nullptr || query_columns == 0 ||
        lane.token_count == kUnavailableGroupedExtent ||
        lane.token_count == 0) {
        throw std::runtime_error("envelope_dot has no query to score with");
    }
    const KvCacheLayerView view =
        obs->kv->layer_view(static_cast<int>(layer));
    if (!view.has_envelopes()) {
        throw std::runtime_error(
            "envelope_dot ran on a layer without kv envelopes");
    }
    if (view.head_dim <= 0 || view.num_kv_heads <= 0 ||
        query_columns % static_cast<std::uint32_t>(view.head_dim) != 0) {
        throw std::runtime_error(
            "envelope_dot query width is not a multiple of head_dim");
    }

    int request = -1;
    for (int r = 0; r < obs->num_requests; ++r) {
        if (obs->qo_indptr_h[r] == lane.token_start) {
            request = r;
            break;
        }
    }
    if (request < 0) {
        throw std::runtime_error(
            "envelope_dot lane does not start at a request boundary");
    }

    const std::uint32_t page_begin = obs->kv_page_indptr_h[request];
    const std::uint32_t page_end = obs->kv_page_indptr_h[request + 1];
    if (page_end < page_begin) {
        throw std::runtime_error("envelope_dot saw a malformed kv page CSR");
    }
    // A BOUND, not the count. It sizes the result row; the kernel takes the
    // real count from the device CSR. See `GroupedLaneEnvelope`.
    const std::uint32_t page_bound = page_end - page_begin;
    const std::uint32_t page_size =
        static_cast<std::uint32_t>(view.page_size);
    const std::uint32_t qo_len =
        obs->qo_indptr_h[request + 1] - obs->qo_indptr_h[request];

    if (const char* dbg = std::getenv("PIE_QUEST_DEBUG"); dbg && *dbg == '1') {
        // The fire's request count, reported once per new maximum. A test that
        // wants to prove the R>1 path was exercised has no other way to see it:
        // co-batching is a scheduling outcome, not something a program asks for.
        static std::atomic<int> seen_requests{0};
        int prev = seen_requests.load(std::memory_order_relaxed);
        while (obs->num_requests > prev &&
               !seen_requests.compare_exchange_weak(prev, obs->num_requests)) {
        }
        if (obs->num_requests > prev) {
            std::fprintf(stderr, "[quest] R_max=%d\n", obs->num_requests);
        }
        static std::atomic<int> shots{0};
        if (shots.fetch_add(1) < 4) {
            std::fprintf(stderr,
                "[quest] req=%d/%d layer=%u tok_start=%u tok_count=%u "
                "page_begin(bound)=%u page_bound=%u page_size=%u "
                "qo=[%u,%u] kv_heads=%d head_dim=%d qcols=%u\n",
                request, obs->num_requests, layer, lane.token_start,
                lane.token_count, page_begin, page_bound, page_size,
                obs->qo_indptr_h[request], obs->qo_indptr_h[request + 1],
                view.num_kv_heads, view.head_dim, query_columns);
        }
    }

    return GroupedLaneEnvelope{
        .env_min = reinterpret_cast<const __nv_bfloat16*>(view.k_env_min),
        .env_max = reinterpret_cast<const __nv_bfloat16*>(view.k_env_max),
        .query = query_base +
            static_cast<std::size_t>(lane.token_start + lane.token_count - 1) *
                query_columns,
        .page_ids = obs->kv_page_indices_d,
        .page_indptr = obs->kv_page_indptr_d,
        .last_page_lens = obs->kv_last_page_lens_d,
        .request = static_cast<std::uint32_t>(request),
        .qo_len = qo_len,
        .page_size = page_size,
        .page_bound = page_bound,
        .num_q_heads =
            query_columns / static_cast<std::uint32_t>(view.head_dim),
        .num_kv_heads = static_cast<std::uint32_t>(view.num_kv_heads),
        .head_dim = static_cast<std::uint32_t>(view.head_dim),
    };
}

// Materialize one lane's `[declared_kv_max]` attention-probability row from the
// layer capture published by the model body.
//
// The capture is ragged and exactly `kv_len` wide; the program declared a static
// ceiling (it cannot know `kv_len` at compile time, exactly like `envelope_dot`'s
// `p_max`). So this pads to the declared width with zeros -- a position that does
// not exist received no attention, which is already the correct value and sorts
// to the bottom of every eviction ranking without needing a sentinel.
//
// Every disagreement here throws. Truncating a row that overflows the ceiling
// would silently make the tail of the context un-evictable, and a zero row would
// be indistinguishable from "evict everything": both are the class of failure
// this driver's unchecked-contract list exists to prevent.
// Resolve where this lane's `attn_page_mask` sink should write. Unlike every
// other lane resolution this hands back a *mutable* slice: the buffer belongs
// to the model body, which allocated it for the whole fire and re-seeds it to
// "keep everything" before each layer's hook.
GroupedLanePageMask resolve_lane_page_mask(
    const StagedLane& lane,
    std::uint32_t layer,
    const model::StageHookSideband& sideband) {
    model::AttentionMaskSink* sink = sideband.mask_sink;
    if (sink == nullptr || !sink->usable()) {
        throw std::runtime_error(
            "attn_page_mask ran on a fire with no page-mask destination");
    }
    const model::AttentionObservation* obs = sideband.observation;
    if (obs == nullptr || !obs->usable()) {
        throw std::runtime_error(
            "attn_page_mask ran outside a model body with kv geometry");
    }
    if (static_cast<std::uint32_t>(obs->num_requests) != sink->num_requests) {
        throw std::runtime_error(
            "attn_page_mask destination and kv geometry disagree on request "
            "count");
    }

    int request = -1;
    for (int r = 0; r < obs->num_requests; ++r) {
        if (obs->qo_indptr_h[r] == lane.token_start) {
            request = r;
            break;
        }
    }
    if (request < 0) {
        throw std::runtime_error(
            "attn_page_mask lane does not start at a request boundary");
    }

    // Row base is the request index times a fixed stride -- never the page
    // CSR. The lane's mask has to line up with the scores the lane read, and
    // those are indexed "slot p of request r" too; going through a CSR would
    // make both depend on a host table that the decode-envelope path only
    // bounds. See `AttentionMaskSink`.
    const std::uint32_t begin =
        static_cast<std::uint32_t>(request) * sink->stride;

    // Tag here rather than after the launch: the sink kernel is enqueued on the
    // same stream immediately below, so by the time the model body's decode
    // call reads the buffer the write has happened. Tagging on the host is what
    // lets the consumer distinguish "this layer wrote a mask" from "a previous
    // layer did" without a device round trip.
    sink->written_layer = static_cast<int>(layer);
    return GroupedLanePageMask{sink->keep + begin, sink->stride};
}

// Resolve one lane's `lora` sink into a launch-owned table entry.
//
// This deliberately does NOT follow `attn_page_mask`'s shape. The mask sink's
// effect is a device write into a body-owned buffer at the layer hook — the
// model body exists (and delivered its sideband) by the time it fires. The
// lora sink fires in the PROLOGUE, which `Dispatch::begin` executes BEFORE
// the model body runs, so a body-owned destination cannot exist yet; and its
// effect is host-side configuration, not device computation — the program
// hands the backend where the adapter weights live (A/B channel cells) and
// where they apply (SITES), and the body's projection GEMMs consume that for
// the whole forward. So resolving the sink IS executing it: the generated
// runtime performs no device stores for it (`fused_runtime.cuh`).
//
// Every disagreement throws. A silently unresolved lora is a request whose
// adapter never applied while every sample still returns — the exact failure
// class the sink-name bind gate exists to prevent.
// PER-SITE PAIRS (the adapter rung, north-star-dsl.md): a prologue may
// carry MULTIPLE lora sinks — one (A, B, SITES) per projection site
// set — and the lane contributes one table entry per sink over the
// same token span. Site sets must be DISJOINT across the lane's sinks
// (one pair per site; the consumer's per-entry width checks then bind
// each site to its own d_out — the q+v case).
std::vector<model::LoraLaneView> resolve_lane_lora_sinks(
    const StagedLane& lane,
    const plan::StagePlan& stage);

model::LoraLaneView resolve_lane_lora_one(
    const StagedLane& lane,
    const plan::StagePlan& stage,
    const plan::PlanOp* sink) {
    // Value id -> producing op, via the stage's flat result numbering (the
    // same walk the fused packer uses for its `bases` table).
    std::vector<std::uint32_t> bases(stage.ops.size());
    std::uint32_t values = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        bases[node] = values;
        values += stage.ops[node].op.results;
    }
    auto producer = [&](std::uint32_t value) -> const plan::PlanOp& {
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const std::uint32_t results = stage.ops[node].op.results;
            if (value >= bases[node] && value < bases[node] + results) {
                return stage.ops[node].op;
            }
        }
        throw std::runtime_error(
            "lora sink argument has no producing op in its stage");
    };

    if (sink == nullptr) {
        throw std::runtime_error(
            "lora resolution ran on a stage without the sink");
    }
    if (sink->args.size() != 3 && sink->args.size() != 2) {
        throw std::runtime_error(
            "lora sink is neither the (A, B, SITES) low-rank shape nor "
            "the (L, SITES) scale shape");
    }
    const bool scale_form = sink->args.size() == 2;

    // A and B are channel CONTENTS (an adapter swap is a re-seed, never a
    // re-trace), so the harvested address is the channel's committed cell —
    // resolved exactly the way the fused packer positions a read: an
    // engine-sequenced ticket pins the cell at its expected head, otherwise
    // the live registry mirror is the head.
    auto channel_address = [&](std::uint32_t value,
                               const char* which) -> const void* {
        const plan::PlanOp& op = producer(value);
        if (op.tag != PTIR_OP_CHAN_READ || op.chan < 0) {
            // The design admits in-graph adapter computation (a scaled or
            // merged adapter feeding the sink); this begin-time resolver does
            // not — a prologue value has no device buffer that outlives
            // `execute_declared_phase`'s temporaries. Refuse loudly rather
            // than harvest a pointer that dangles by the first projection.
            throw std::runtime_error(
                std::string("lora sink ") + which +
                " argument is not a direct channel read; computed adapter "
                "weights are not supported by the CUDA begin-time resolver");
        }
        const auto local = static_cast<std::size_t>(op.chan);
        if (local >= stage.channel_bindings.size()) {
            throw std::runtime_error(
                std::string("lora sink ") + which +
                " argument reads a channel outside the stage's bindings");
        }
        const std::uint32_t dense = stage.channel_bindings[local];
        auto& view = lane.bound->instance->view();
        const std::uint32_t slot = view.slot(dense);
        const auto ticket = std::find_if(
            lane.tickets.begin(),
            lane.tickets.end(),
            [slot](const DeviceHostChannelTicket& candidate) {
                return candidate.slot == slot;
            });
        if (ticket != lane.tickets.end()) {
            const std::uint64_t head =
                ticket->expected_head == kNoChannelTicket
                    ? view.registry()->host_head(slot)
                    : ticket->expected_head;
            return ticket->cells +
                static_cast<std::size_t>(head % ticket->cap1) *
                    ticket->native_bytes;
        }
        return view.committed_cell(dense);
    };

    // SITES is trace-known placement (a `Tensor::constant` bitmask over the
    // model's site vocabulary): structure, not contents, so it lives in the
    // plan as a literal rather than behind a channel.
    const plan::PlanOp& sites =
        producer(sink->args[scale_form ? 1 : 2]);
    if (sites.tag != PTIR_OP_CONST) {
        throw std::runtime_error(
            "lora SITES argument is not a trace-known constant");
    }
    if (sites.lit_dtype != PTIR_DT_U32 && sites.lit_dtype != PTIR_DT_I32) {
        throw std::runtime_error(
            "lora SITES constant is not an integer site bitmask");
    }

    // Adapter geometry comes from the sink arguments' declared value types —
    // the same flat SSA numbering the fused runtime types against. The rank
    // is trace-known shape (a different rank = a different traced program,
    // §6.5), so the plan carries it; a symbolic or non-rank-3 dim would mean
    // the trace did not commit to a geometry the forward can loop over, and
    // is refused here rather than mis-sliced per layer.
    auto static_dims_3 =
        [&](std::uint32_t value, const char* which) -> const plan::ValueType& {
        if (value >= stage.value_types.size()) {
            throw std::runtime_error(
                std::string("lora sink ") + which +
                " argument has no declared value type");
        }
        const plan::ValueType& type = stage.value_types[value];
        if (type.dtype != PTIR_DT_F32) {
            throw std::runtime_error(
                std::string("lora sink ") + which +
                " argument is not f32 (the channel wire dtype the consumer "
                "expects to cast from)");
        }
        if (type.dims.size() != 3) {
            throw std::runtime_error(
                std::string("lora sink ") + which +
                " argument is not rank-3 ([num_layers, R, d_in] / "
                "[num_layers, d_out, R])");
        }
        for (const auto& dim : type.dims) {
            if (dim.symbolic || dim.value == 0) {
                throw std::runtime_error(
                    std::string("lora sink ") + which +
                    " argument has a symbolic or zero dimension; adapter "
                    "geometry must be trace-known");
            }
        }
        return type;
    };
    if (scale_form) {
        // The SCALE form (IA3): `l` is [num_layers, d_out] f32, applied
        // as `y = l ⊙ y` at the declared sites. `a` carries the l
        // address; rank/d_in rest at zero (no GEMM, no scratch).
        if (sink->args[0] >= stage.value_types.size()) {
            throw std::runtime_error(
                "lora scale sink L argument has no declared value type");
        }
        const plan::ValueType& l_type = stage.value_types[sink->args[0]];
        if (l_type.dtype != PTIR_DT_F32 || l_type.dims.size() != 2) {
            throw std::runtime_error(
                "lora scale sink L argument is not f32 rank-2 "
                "([num_layers, d_out])");
        }
        for (const auto& dim : l_type.dims) {
            if (dim.symbolic || dim.value == 0) {
                throw std::runtime_error(
                    "lora scale sink L argument has a symbolic or zero "
                    "dimension");
            }
        }
        return model::LoraLaneView{
            .a = channel_address(sink->args[0], "L"),
            .b = nullptr,
            .sites_bits = sites.lit_bits,
            .token_start = lane.token_start,
            .token_count = lane.token_count,
            .num_layers = l_type.dims[0].value,
            .rank = 0,
            .d_in = 0,
            .d_out = l_type.dims[1].value,
            .form = model::LoraLaneView::Form::Scale,
        };
    }
    const plan::ValueType& a_type = static_dims_3(sink->args[0], "A");
    const plan::ValueType& b_type = static_dims_3(sink->args[1], "B");
    const std::uint32_t num_layers = a_type.dims[0].value;
    const std::uint32_t rank = a_type.dims[1].value;
    const std::uint32_t d_in = a_type.dims[2].value;
    const std::uint32_t d_out = b_type.dims[1].value;
    if (b_type.dims[0].value != num_layers || b_type.dims[2].value != rank) {
        throw std::runtime_error(
            "lora sink A and B disagree on num_layers/rank: A is [" +
            std::to_string(num_layers) + ", " + std::to_string(rank) + ", " +
            std::to_string(d_in) + "], B is [" +
            std::to_string(b_type.dims[0].value) + ", " +
            std::to_string(d_out) + ", " +
            std::to_string(b_type.dims[2].value) + "]");
    }

    return model::LoraLaneView{
        .a = channel_address(sink->args[0], "A"),
        .b = channel_address(sink->args[1], "B"),
        .sites_bits = sites.lit_bits,
        .token_start = lane.token_start,
        .token_count = lane.token_count,
        .num_layers = num_layers,
        .rank = rank,
        .d_in = d_in,
        .d_out = d_out,
        .form = model::LoraLaneView::Form::LowRank,
    };
}

std::vector<model::LoraLaneView> resolve_lane_lora_sinks(
    const StagedLane& lane,
    const plan::StagePlan& stage) {
    std::vector<model::LoraLaneView> views;
    // Disjointness is PER FORM: two low-rank pairs (or two scales) on
    // one site are ambiguous and refuse; a low-rank + a scale on the
    // SAME site compose in program order (DoRA: s ⊙ (y + B(Ax)) — the
    // consumer applies every scale after every delta).
    std::uint64_t claimed_lowrank = 0;
    std::uint64_t claimed_scale = 0;
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.tag != PTIR_OP_SINK_CALL) continue;
        if (op.name_idx >= stage.names.size() ||
            stage.names[op.name_idx] != "lora") {
            continue;
        }
        model::LoraLaneView view = resolve_lane_lora_one(lane, stage, &op);
        std::uint64_t& claimed =
            view.form == model::LoraLaneView::Form::Scale
                ? claimed_scale
                : claimed_lowrank;
        if ((view.sites_bits & claimed) != 0) {
            throw std::runtime_error(
                "lora sinks claim overlapping sites of one form in one "
                "prologue (bits " +
                std::to_string(view.sites_bits & claimed) + ")");
        }
        claimed |= view.sites_bits;
        views.push_back(view);
    }
    if (views.empty()) {
        throw std::runtime_error(
            "lora resolution ran on a stage without the sink");
    }
    return views;
}

const float* resolve_lane_attn_score(
    const StagedLane& lane,
    std::uint32_t layer,
    std::uint64_t declared_kv_max,
    cudaStream_t stream,
    std::vector<void*>& temporaries,
    const model::StageHookSideband& sideband) {
    const model::AttentionScores* scores = sideband.scores;
    if (scores == nullptr || !scores->usable()) {
        throw std::runtime_error(
            "attn_score ran on a fire that captured no attention scores");
    }
    if (scores->layer != layer) {
        throw std::runtime_error(
            "attn_score saw a capture from a different layer");
    }
    const model::AttentionObservation* obs = sideband.observation;
    if (obs == nullptr || !obs->usable()) {
        throw std::runtime_error(
            "attn_score ran outside a model body with kv geometry");
    }
    if (static_cast<std::uint32_t>(obs->num_requests) !=
        scores->num_requests) {
        throw std::runtime_error(
            "attn_score capture and kv geometry disagree on request count");
    }
    if (declared_kv_max == 0) {
        throw std::runtime_error("attn_score declares an empty row");
    }

    int request = -1;
    for (int r = 0; r < obs->num_requests; ++r) {
        if (obs->qo_indptr_h[r] == lane.token_start) {
            request = r;
            break;
        }
    }
    if (request < 0) {
        throw std::runtime_error(
            "attn_score lane does not start at a request boundary");
    }

    const std::uint32_t begin = scores->offsets_h[request];
    const std::uint32_t end = scores->offsets_h[request + 1];
    if (end < begin) {
        throw std::runtime_error("attn_score saw a malformed capture CSR");
    }
    const std::uint32_t kv_len = end - begin;
    if (static_cast<std::uint64_t>(kv_len) > declared_kv_max) {
        throw std::runtime_error(
            "attn_score declared kv_max " + std::to_string(declared_kv_max) +
            " but the request holds " + std::to_string(kv_len) +
            " kv positions; raise the program's ceiling");
    }

    float* row = nullptr;
    CUDA_CHECK(cudaMallocAsync(
        &row, declared_kv_max * sizeof(float), stream));
    temporaries.push_back(row);
    CUDA_CHECK(cudaMemsetAsync(
        row, 0, declared_kv_max * sizeof(float), stream));
    if (kv_len > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            row, scores->values + begin,
            static_cast<std::size_t>(kv_len) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }
    return row;
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
        .presampled_token_rows = lane.presampled_token_rows.empty()
            ? nullptr
            : &lane.presampled_token_rows,
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
    const model::StageHookSideband& sideband = {},
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
    if (phase == PTIR_STAGE_PROLOGUE) {
        // The lora table is a begin-time product of exactly this phase:
        // rebuilt whenever the prologue runs so a stale resolution can never
        // outlive the launch geometry it was harvested against.
        launch.lora_lanes.clear();
        launch.lora_lane_sources.clear();
    }
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
    // Per-lane intrinsic rows materialized below. Freed stream-ordered after
    // every occurrence has been launched and every side stream rejoined, so
    // the frees cannot outrun the kernels that read them.
    struct PhaseTemporaries {
        std::vector<void*> pointers;
        cudaStream_t stream = nullptr;
        ~PhaseTemporaries() {
            for (void* pointer : pointers) {
                cudaFreeAsync(pointer, stream);
            }
        }
    } phase_temporaries{{}, stream};
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
            if (stage_calls_kernel(stage, "envelope_dot")) {
                binding.envelope = resolve_lane_envelope(
                    lane, query_base, query_columns, layer, sideband);
            }
            if (stage_calls_sink(stage, "attn_page_mask")) {
                binding.page_mask =
                    resolve_lane_page_mask(lane, layer, sideband);
            }
            if (phase == PTIR_STAGE_PROLOGUE &&
                stage_calls_sink(stage, "lora")) {
                // Resolving IS the sink's execution: the entry lands in the
                // launch-owned table the frame later hands the model body,
                // and the generated runtime performs no device work for the
                // sink itself. See `resolve_lane_lora`. One lane gets ONE
                // configuration for the whole forward — a prologue carrying
                // the sink twice would otherwise silently produce two table
                // entries with the same span and leave the consumer to pick.
                if (std::find(
                        launch.lora_lane_sources.begin(),
                        launch.lora_lane_sources.end(),
                        &lane) != launch.lora_lane_sources.end()) {
                    throw std::runtime_error(
                        "lora sink resolved twice for one lane");
                }
                for (const model::LoraLaneView& view :
                     resolve_lane_lora_sinks(lane, stage)) {
                    // The sources array is parallel PER ENTRY (the span
                    // re-stamp indexes it 1:1), so a multi-sink lane
                    // contributes one source per view.
                    launch.lora_lanes.push_back(view);
                    launch.lora_lane_sources.push_back(&lane);
                }
            }
            std::uint64_t attn_score_kv_max = 0;
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
                if (normalized.op.tag == PTIR_OP_INTRINSIC_VAL &&
                    normalized.op.intr == PTIR_INTR_ATTN_SCORE) {
                    if (value_base >= stage.value_types.size()) {
                        throw std::runtime_error(
                            "AttnScore intrinsic has no declared value type");
                    }
                    const std::uint64_t declared = grouped_numel(
                        stage.value_types[value_base], binding);
                    // One occurrence per stage may declare several reads; they
                    // must agree, since one buffer backs them all.
                    if (attn_score_kv_max != 0 &&
                        attn_score_kv_max != declared) {
                        throw std::runtime_error(
                            "AttnScore is read at two different widths in one "
                            "stage");
                    }
                    attn_score_kv_max = declared;
                }
                value_base += normalized.op.results;
            }
            if (attn_score_kv_max != 0) {
                binding.attn_score_base = resolve_lane_attn_score(
                    lane, layer, attn_score_kv_max, stream,
                    phase_temporaries.pointers, sideband);
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
        const bool attention_phase =
            phase == PTIR_STAGE_ON_ATTN_PROJ ||
            phase == PTIR_STAGE_ON_ATTN;
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
            GroupedLaunchResult result;
            if (attention_phase) {
                // Attention phases through the prepare/body seam with stable
                // per-stage buffers (stage 6 increment 1), interleaved at
                // hook time. Since the eager unification this branch runs
                // only for fires the fire-level prepare pass VETOED (see
                // `execute_attention_phase`'s fallback comment) — prepared
                // pure-decode hook fires consume the fire-level cursor
                // instead and never come through here. Same seam either way:
                // `prepare_generated_stage` does every piece of host work
                // (metadata build, channel-cursor reads, elision analysis,
                // side-table uploads, pack + upload) and
                // `launch_generated_stage` is a host-work-free body reading
                // only prepared state — no rotating rings and no
                // cudaEventSynchronize on this path.
                const auto prepared = generated::prepare_generated_stage(
                    group.bindings,
                    *first.executable,
                    launch.owner->generated_runtime,
                    target_stream,
                    execution_options,
                    generated::PreparedBufferMode::kStablePerStage);
                result = generated::launch_generated_stage(*prepared);
            } else {
                // Prologue / Epilogue: the ring-backed combined wrapper,
                // deliberately untouched by the eager unification.
                // TODO(stage6-plan.md increment 1): migrate them onto the
                // prepared kStablePerStage path too and retire the rotating
                // rings entirely.
                result = generated::run_generated_stage_ring(
                    group.bindings,
                    *first.executable,
                    launch.owner->generated_runtime,
                    target_stream,
                    execution_options);
            }
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

std::unique_ptr<StagedLaunch> Dispatch::begin_host(
    const pie::driver::fire::LaunchView& view,
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
    state.source_ready = acquire_launch_event(*impl_);
    for (cudaEvent_t& event : state.phase_done) {
        event = acquire_launch_event(*impl_);
    }
    const std::size_t count = view.ptir_program_hashes.size();
    std::unordered_map<std::uint64_t, std::size_t> fire_counts;
    state.lanes.reserve(count);
    state.ticket_staging.reserve(view.channel_expected_head.size());
    state.pull_staging.reserve(count);
    // Pass A (serial): everything ordering- or allocation-sensitive — map
    // lookups, snapshot allocation. (The CUDA event waits that used to
    // interleave here are stream work and live in `begin_enqueue`.)
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
        // Prologue put effects for FramePrepare-time consumers (see the
        // field's comment) — the LIVE effect sets stay empty until the
        // phases execute.
        for (const plan::StagePlan* stage :
             (*lane->phase_plans)[PTIR_STAGE_PROLOGUE]) {
            for (const auto& normalized : stage->ops) {
                const auto& op = normalized.op;
                if (op.chan < 0 || op.tag != PTIR_OP_CHAN_PUT) continue;
                const std::uint32_t local =
                    static_cast<std::uint32_t>(op.chan);
                if (local >= stage->channel_bindings.size()) continue;
                lane->prologue_put_slots.insert(
                    bound.instance->view().slot(
                        stage->channel_bindings[local]));
            }
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
    // Pass C (serial): diagnostic retry forcing and the staging appends.
    // The registry sequence APPLIES moved to `begin_enqueue`: host mirrors
    // must advance at each wave's ENQUEUE position (the pre-frame-split
    // timeline every execution-time mirror reader was written against),
    // not at frame entry. FramePrepare-time consumers read the wave's
    // window from its tickets instead.
    for (std::size_t program = 0; program < count; ++program) {
        std::unique_ptr<StagedLane> lane = std::move(pending_lanes[program]);
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
            .full = lane->bound->instance->view().d_full(),
            .pass_commit = lane->snapshot->device,
            .ticket_offset = lane->device_ticket_offset,
            .ticket_count = lane->device_ticket_count,
            .initial_commit = initial_commit,
            .diagnose = kDiagnosePullValidate,
        });
        state.touched_instances.push_back(
            view.ptir_program_instances.data()[program]);
        state.lanes.push_back(std::move(lane));
    }
    if (begin_timing) {
        const auto now = fire_timing::Clock::now();
        launch->begin_breakdown_.pass_c_us =
            fire_timing::duration_us(begin_mark, now);
    }
    return launch;
}

namespace {

// The churn fix (driver-only; north-star "CONSTRAINT, STATED" — the
// runtime is untouchable): absorb transient guest lag at the LAST host
// point before a wave commits. The engine's run-ahead expectations
// assume consumes and ring space a guest a few ms behind has not yet
// provided; since v14 deleted the RETRY outcome, one unready lane
// fails its whole frame step and the failed fire's committed
// reservations poison the instance (the churn fault, root-caused
// 2026-08-04: 63-90% lane death under 9-lane 35ms churn). Readiness
// is MONOTONE toward the expectations (heads advance only by prior
// waves already on the stream and by guest takes; ring space only
// grows), so a bounded host poll is sound: wait up to
// PIE_CHANNEL_READY_WAIT_MS (default 25; 0 disables) for every gate
// to hold, then proceed either way — an exhausted budget keeps
// today's loud failure semantics.
bool lane_tickets_ready(const std::vector<DeviceHostChannelTicket>& tickets) {
    for (const DeviceHostChannelTicket& ticket : tickets) {
        if (ticket.words == nullptr) continue;
        const std::uint64_t head =
            std::atomic_ref<const std::uint64_t>(ticket.words[0])
                .load(std::memory_order_acquire);
        const std::uint64_t tail =
            std::atomic_ref<const std::uint64_t>(ticket.words[1])
                .load(std::memory_order_acquire);
        if ((ticket.flags & kTicketConsume) != 0 &&
            head != ticket.expected_head) {
            return false;
        }
        if ((ticket.flags & kTicketRequireInput) != 0 && !(tail > head)) {
            return false;
        }
        if ((ticket.flags & kTicketPublish) != 0) {
            const std::uint64_t same_fire_consume =
                (ticket.flags & kTicketConsume) != 0 ? 1u : 0u;
            if (tail != ticket.expected_tail) return false;
            if (!(tail - head <
                  static_cast<std::uint64_t>(ticket.cap1 - 1) +
                      same_fire_consume)) {
                return false;
            }
        }
    }
    return true;
}

void wait_for_wave_readiness(
    const std::vector<std::unique_ptr<StagedLane>>& lanes) {
    // DEFAULT OFF after measurement: a 2-minute churn soak with the
    // wait armed (25 ms) still failed 88% of lanes across 12k waits —
    // the dominant failure is NOT front-loaded enqueue-time lag, so
    // the wait only added stall. Kept as an env-armed experiment
    // (PIE_CHANNEL_READY_WAIT_MS=<ms>); the real fix is the upstream
    // dev integration (its scheduler keeps channel-bound lanes out of
    // shared steps — the blast-radius half of the fault).
    static const int budget_ms = [] {
        const char* v = std::getenv("PIE_CHANNEL_READY_WAIT_MS");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    if (budget_ms <= 0) return;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(budget_ms);
    bool waited = false;
    for (const auto& lane : lanes) {
        while (!lane_tickets_ready(lane->tickets)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                if (std::getenv("PIE_READY_WAIT_TRACE") != nullptr) {
                    std::fprintf(stderr,
                                 "[ready-wait] budget exhausted "
                                 "(%d ms) — proceeding to the loud "
                                 "gate\n",
                                 budget_ms);
                }
                return;
            }
            waited = true;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    if (waited && std::getenv("PIE_READY_WAIT_TRACE") != nullptr) {
        std::fprintf(stderr, "[ready-wait] wave absorbed guest lag\n");
    }
}

}  // namespace

void Dispatch::begin_enqueue(StagedLaunch& launch) {
    StagedLaunch::State& state = *launch.state_;
    if (!state.active || state.device_layer != nullptr) {
        throw std::runtime_error(
            "staged PTIR launch enqueued twice or after abort");
    }
    cudaStream_t stream = state.stream;
    const pie::driver::fire::LaunchView& view = state.view;
    const bool begin_timing = fire_timing::full();
    auto begin_mark = begin_timing ? fire_timing::Clock::now()
                                   : fire_timing::Clock::time_point{};
    // The churn fix: bounded readiness absorption before this wave's
    // expectations harden into the frame (see wait_for_wave_readiness).
    wait_for_wave_readiness(state.lanes);
    // Registry sequence applies, in lane order, at the wave's ENQUEUE
    // position: every execution-time mirror reader (stage-metadata
    // builders, settlement prep) was written against the pre-frame-split
    // timeline where mirrors reflect exactly the waves enqueued so far.
    // FramePrepare-time consumers never read the mirrors — they use the
    // wave's tickets.
    for (const auto& lane : state.lanes) {
        apply_lane_sequence_tickets(
            view, lane->program, *lane->bound, impl_->channels);
    }
    // Order this wave after any pending bind-time initialization work (ring
    // metadata, seed uploads, baked-list uploads) still riding the registry's
    // initialization stream — binds no longer host-sync it (RV-28: fires must
    // never observe a slot whose ring metadata or seeds are still in flight).
    impl_->channels.order_after_initialization(stream);
    CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&state.device_layer),
        sizeof(std::uint32_t),
        stream));
    // ONE publication-ordering wait for the whole wave (see
    // `Impl::publications_done`): the previous wave's publications all rode
    // the callback stream, so this single wait replaces the per-instance
    // event waits that used to cost ~2 host API calls per lane per wave.
    if (impl_->publications_recorded) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, impl_->publications_done, 0));
    }
    // Per-instance ordering survives only for the bind-time seed upload
    // (recorded on the seed copy stream); after the instance's first
    // completed wave the event is retired and the shared
    // `publications_done` wait above carries the ordering.
    for (const auto& lane : state.lanes) {
        if (lane->bound != nullptr &&
            lane->bound->publish_done != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(
                stream, lane->bound->publish_done, 0));
        }
    }
    state.device_tickets = launch_pull_validate_host_channels_batch(
        state.ticket_staging,
        state.pull_staging,
        stream);
    for (const auto& lane : state.lanes) {
        if (lane->snapshot != nullptr) lane->snapshot->ever_validated = true;
    }
    if (begin_timing) {
        launch.begin_breakdown_.pull_validate_us = fire_timing::duration_us(
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
        abort(launch, stream);
        throw;
    }
}

std::unique_ptr<StagedLaunch> Dispatch::begin(
    const pie::driver::fire::LaunchView& view,
    cudaStream_t stream) {
    auto launch = begin_host(view, stream);
    begin_enqueue(*launch);
    return launch;
}

void Dispatch::update_launch_geometry(
    StagedLaunch& launch,
    const pie::driver::fire::LaunchView& resolved_view,
    std::span<const std::uint32_t> program_token_starts) {
    StagedLaunch::State& state = *launch.state_;
    if (!state.active ||
        resolved_view.ptir_program_hashes.size() != state.lanes.size() ||
        program_token_starts.size() != state.lanes.size()) {
        throw std::runtime_error("invalid staged PTIR geometry update");
    }
    state.view = resolved_view;
    const std::size_t count = state.lanes.size();
    auto extent = [&](const pie::driver::Slice<std::uint32_t>& values,
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
    // The prologue already ran under the frame split, so any lora entries it
    // resolved carry begin_host-time token spans; re-stamp them from the
    // resolved geometry just written above. The A/B addresses and SITES need
    // no refresh — channel cells and trace constants do not move with the
    // wire geometry.
    if (state.lora_lanes.size() != state.lora_lane_sources.size()) {
        throw std::runtime_error(
            "lora table and its lane attribution are out of step");
    }
    for (std::size_t entry = 0; entry < state.lora_lanes.size(); ++entry) {
        state.lora_lanes[entry].token_start =
            state.lora_lane_sources[entry]->token_start;
        state.lora_lanes[entry].token_count =
            state.lora_lane_sources[entry]->token_count;
    }
}

namespace {

// FNV-ish accumulator for the hook-graph replay fingerprint. Everything a
// captured body BAKES — device addresses recorded as kernel arguments and
// grid geometry — flows through this; the batch engine recaptures whenever
// the value changes, which is the growth/generation/instance-churn
// invalidation in one comparison.
inline void hook_fp_mix(std::uint64_t& hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 12) + (hash >> 4);
    hash *= 0x100000001b3ull;
}

inline void hook_fp_mix_ptr(std::uint64_t& hash, const void* pointer) {
    hook_fp_mix(hash, reinterpret_cast<std::uintptr_t>(pointer));
}

void hook_fp_mix_shape(std::uint64_t& hash, const GroupedRowShape& shape) {
    hook_fp_mix(hash, shape.max_rows);
    hook_fp_mix(hash, shape.max_columns);
    hook_fp_mix(hash, shape.rows.max_numel);
    hook_fp_mix(hash, shape.rows.elements_per_extent);
    hook_fp_mix(hash, shape.rows.extent);
    hook_fp_mix(hash, shape.columns.max_numel);
    hook_fp_mix(hash, shape.columns.elements_per_extent);
    hook_fp_mix(hash, shape.columns.extent);
}

// Mix every field of a prepared launch that the captured body bakes into a
// recorded kernel node: grids, block-shape scalars, value ids, and the
// device-side argument block addresses.
void hook_fp_mix_prepared(
    std::uint64_t& hash,
    const generated::GeneratedStagePrepared& prepared) {
    hook_fp_mix(hash, prepared.lane_count);
    hook_fp_mix(hash, prepared.padded_lane_count);
    hook_fp_mix(hash, prepared.value_count);
    hook_fp_mix_ptr(hash, prepared.device_header);
    hook_fp_mix_ptr(hash, prepared.device_lanes);
    hook_fp_mix_ptr(hash, prepared.device_readiness);
    hook_fp_mix_ptr(hash, prepared.device_readiness_slots);
    hook_fp_mix_ptr(hash, prepared.device_scratch);
    hook_fp_mix_ptr(hash, prepared.ring_full);
    hook_fp_mix_ptr(hash, prepared.ring_head);
    hook_fp_mix_ptr(hash, prepared.ring_tail);
    hook_fp_mix_ptr(hash, prepared.ring_cap1);
    hook_fp_mix(hash, prepared.readiness_diagnose);
    hook_fp_mix(hash, prepared.arg_header);
    hook_fp_mix(hash, prepared.arg_lanes);
    hook_fp_mix(hash, prepared.arg_channels);
    hook_fp_mix(hash, prepared.arg_descriptors);
    hook_fp_mix(hash, prepared.arg_params);
    hook_fp_mix(hash, prepared.arg_offsets);
    hook_fp_mix(hash, prepared.arg_scratch);
    hook_fp_mix(hash, prepared.arg_value_count);
    hook_fp_mix(hash, prepared.arg_scratch_stride);
    hook_fp_mix(hash, prepared.arg_temporary_offset);
    hook_fp_mix(hash, prepared.arg_pending);
    hook_fp_mix(hash, prepared.arg_intrinsic_bases);
    hook_fp_mix(hash, prepared.arg_intrinsic_modes);
    hook_fp_mix(hash, prepared.arg_intrinsic_widths);
    hook_fp_mix(hash, prepared.arg_intrinsic_strides);
    hook_fp_mix(hash, prepared.arg_intrinsic_offsets);
    hook_fp_mix(hash, prepared.regions.size());
    for (const auto& region : prepared.regions) {
        hook_fp_mix(hash, static_cast<std::uint64_t>(region.kind));
        hook_fp_mix_ptr(hash, region.region);
        hook_fp_mix(hash, region.nucleus_segments);
        hook_fp_mix(hash, region.nucleus.rows);
        hook_fp_mix(hash, region.nucleus.len);
        hook_fp_mix(hash, region.nucleus.logits_kind);
        hook_fp_mix_ptr(hash, region.device_dests);
        hook_fp_mix(hash, region.mask_dtype);
        hook_fp_mix_ptr(hash, region.device_envelopes);
        hook_fp_mix(hash, region.max_pages);
        hook_fp_mix(hash, region.scan_blocks);
        hook_fp_mix_shape(hash, region.shape_a);
        hook_fp_mix_shape(hash, region.shape_b);
        hook_fp_mix(hash, region.dynamic_shape.max_numel);
        hook_fp_mix(hash, region.dynamic_shape.elements_per_extent);
        hook_fp_mix(hash, region.dynamic_shape.extent);
        hook_fp_mix(hash, region.value_a);
        hook_fp_mix(hash, region.value_b);
        hook_fp_mix(hash, region.out_a);
        hook_fp_mix(hash, region.out_b);
        hook_fp_mix(hash, region.matmul_grid);
        hook_fp_mix(hash, region.topk_rows);
        hook_fp_mix(hash, region.topk_length);
        hook_fp_mix(hash, region.topk_k);
        hook_fp_mix(hash, region.topk_is_sort ? 1u : 0u);
        hook_fp_mix(hash, region.topk_vocab);
    }
}

// The eager path's per-stage AttnScore width walk (mirrors
// `execute_declared_phase`'s value_base loop): the declared ceiling the pad
// row is sized by. Returns 0 when the stage reads no scores; returns
// nullopt on a width conflict the eager path would refuse.
std::optional<std::uint64_t> hook_stage_attn_score_width(
    const plan::StagePlan& stage,
    const GroupedLaneBinding& binding) {
    std::uint64_t attn_score_kv_max = 0;
    std::uint32_t value_base = 0;
    for (const auto& normalized : stage.ops) {
        if (normalized.op.tag == PTIR_OP_INTRINSIC_VAL &&
            normalized.op.intr == PTIR_INTR_ATTN_SCORE) {
            if (value_base >= stage.value_types.size()) {
                return std::nullopt;
            }
            const std::uint64_t declared = grouped_numel(
                stage.value_types[value_base], binding);
            if (attn_score_kv_max != 0 && attn_score_kv_max != declared) {
                return std::nullopt;
            }
            attn_score_kv_max = declared;
        }
        value_base += normalized.op.results;
    }
    return attn_score_kv_max;
}

}  // namespace

std::uint64_t Dispatch::prepare_attention_phases(
    StagedLaunch& launch,
    const HookReplayPrepare& in) {
    StagedLaunch::State& state = *launch.state_;
    // Veto diagnostics ([hook-graph] evidence): every 0 return names its
    // reason under PIE_HOOK_GRAPH_TRACE, so an eager fallback is always
    // attributable.
    auto veto = [](const char* reason) -> std::uint64_t {
        static const bool trace = [] {
            const char* v = std::getenv("PIE_HOOK_GRAPH_TRACE");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        if (trace) {
            std::fprintf(
                stderr, "[hook-graph] prepare veto reason=%s\n", reason);
        }
        return 0;
    };
    if (!state.active || state.failed || state.hook_graph_prepared) {
        return veto("launch state");
    }
    // NOTE: `in.stream` may be the legacy default stream (nullptr) — the
    // engine's submission stream — which is a valid stream handle for every
    // enqueue this pass performs.
    if (in.observation == nullptr || !in.observation->usable()) {
        return veto("no usable observation");
    }
    const std::uint32_t layers = in.planned_layers == 0xffffffffu
        ? impl_->model_layers
        : std::min(impl_->model_layers, in.planned_layers);
    if (layers == 0 || !impl_->attention_hook_coverage) {
        return veto("no attention hook coverage");
    }
    const model::AttentionObservation& obs = *in.observation;
    if (obs.total_tokens != obs.num_requests) {
        // Decode-only by contract (the caller gates on `is_pure_decode` too;
        // this is the seam's own restatement): the sideband planners below
        // are decode-shaped — `prepare_decode_score_capture` sizes one query
        // row per request, while a prefill fire's body publishes the PREFILL
        // score capture, whose window-row carve lands the folded row at a
        // different arena address than the plan would bake. Non-decode hook
        // fires run the legacy interleaved eager body.
        return veto("non-decode fire");
    }

    constexpr std::uint8_t kPhases[2] = {
        PTIR_STAGE_ON_ATTN_PROJ, PTIR_STAGE_ON_ATTN};

    // ── Feasibility scan — read-only; a 0 return here has no side effects
    // and the caller runs the eager body (which reproduces, loudly, any
    // refusal the eager path would have made anyway). Excluded from v0
    // capture: Query readers (the bf16→f32 cast allocates in the body),
    // page-mask sinks and envelope_dot (per-fire device side tables +
    // host-side mask control flow), scalable nucleus (per-fire workspace
    // allocations), and logits-family intrinsics (no logits exist in an
    // attention phase).
    bool any_attn_score = false;
    bool any_page_mask = false;
    for (const std::uint8_t phase : kPhases) {
        for (const auto& lane_ptr : state.lanes) {
            const StagedLane& lane = *lane_ptr;
            if (lane.plans == nullptr || lane.plan_identities == nullptr ||
                lane.generated_program == nullptr) {
                return veto("lane has no compiled plans");
            }
            for (const plan::StagePlan* stage :
                 (*lane.phase_plans)[phase]) {
                if (stage->ops.empty()) continue;
                if (stage_uses_intrinsic(*stage, PTIR_INTR_QUERY) ||
                    stage_uses_intrinsic(*stage, PTIR_INTR_LOGITS) ||
                    stage_uses_intrinsic(*stage, PTIR_INTR_MTP_LOGITS) ||
                    stage_uses_intrinsic(*stage, PTIR_INTR_MTP_DRAFTS)) {
                    return veto("stage reads Query/logits intrinsics");
                }
                if (stage_calls_kernel(*stage, "envelope_dot") ||
                    stage_calls_sink(*stage, "lora")) {
                    return veto("stage calls envelope_dot/lora");
                }
                if (stage_calls_sink(*stage, "attn_page_mask")) {
                    // The mask SINK is replayable: its destination rows are
                    // arena-stable, its consumer branch in the model body
                    // (`FirePageMask::written_for`) is host-STRUCTURAL —
                    // tagged by the resolver whenever the stage carries the
                    // sink, never read back from device state — and the
                    // compaction that honours it is device-resolved against
                    // the live CSR. (stage6-plan.md's v0 sketch excluded
                    // this class on the belief the branch read device-era
                    // state; it does not — see resolve_lane_page_mask.)
                    any_page_mask = true;
                }
                for (const auto& region : stage->fused.regions) {
                    if (region.library &&
                        region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE &&
                        !grouped_nucleus_library_supported(*stage, region)) {
                        return veto("scalable nucleus region");
                    }
                }
                const std::size_t stage_index = static_cast<std::size_t>(
                    stage - lane.plans->data());
                if (stage_index >= lane.generated_program->stages.size() ||
                    lane.generated_program->stages[stage_index] == nullptr) {
                    return veto("stage has no generated executable");
                }
                if (stage_uses_intrinsic(*stage, PTIR_INTR_ATTN_SCORE)) {
                    // Scores exist only at OnAttn; anywhere else the eager
                    // resolver throws.
                    if (phase != PTIR_STAGE_ON_ATTN) {
                        return veto("AttnScore outside OnAttn");
                    }
                    any_attn_score = true;
                }
            }
        }
    }
    if (any_attn_score &&
        (!in.wants_attn_score || in.arena == nullptr ||
         in.num_q_heads == 0)) {
        return veto("score fire without capture wiring");
    }
    if (any_page_mask && (!in.wants_page_mask || in.arena == nullptr)) {
        return veto("mask fire without mask wiring");
    }

    // ── Page-mask sideband plan: mirror `FirePageMask`'s carve so the
    // per-lane sink destinations (and the compaction outputs the captured
    // attention reads) are known — and the arena pre-grown — before the
    // body constructs the real thing.
    model::PageMaskCapturePlan mask_plan;
    if (any_page_mask) {
        mask_plan = model::prepare_page_mask_capture(
            in.arena, obs, in.stream);
        if (!mask_plan.ok ||
            mask_plan.num_requests !=
                static_cast<std::uint32_t>(obs.num_requests)) {
            return veto("page-mask plan failed");
        }
    }

    // Request-boundary resolution per lane (mirrors the eager resolvers): a
    // sideband-consuming lane that does not start at a request boundary
    // vetoes here, and the eager body then makes the same refusal loudly.
    std::vector<int> lane_request(state.lanes.size(), -1);
    for (std::size_t lane_index = 0;
         lane_index < state.lanes.size();
         ++lane_index) {
        const StagedLane& lane = *state.lanes[lane_index];
        for (int r = 0; r < obs.num_requests; ++r) {
            if (obs.qo_indptr_h[r] == lane.token_start) {
                lane_request[lane_index] = r;
                break;
            }
        }
    }
    if (any_page_mask) {
        for (const std::uint8_t phase : kPhases) {
            for (std::size_t lane_index = 0;
                 lane_index < state.lanes.size();
                 ++lane_index) {
                const StagedLane& lane = *state.lanes[lane_index];
                for (const plan::StagePlan* stage :
                     (*lane.phase_plans)[phase]) {
                    if (!stage->ops.empty() &&
                        stage_calls_sink(*stage, "attn_page_mask") &&
                        lane_request[lane_index] < 0) {
                        return veto("mask lane off request boundary");
                    }
                }
            }
        }
    }

    // ── Score sideband plan (still side-effect-free on the launch): size
    // the pad rows, refresh the capture's host CSR, pre-grow the arena so
    // nothing grows inside a captured region.
    struct ScoreRowNeed {
        std::size_t occurrence = 0;
        std::size_t lane = 0;
        std::uint64_t kv_max = 0;
        std::uint32_t request = 0;
        std::size_t row_offset = 0;  // filled after the carve
    };
    std::vector<ScoreRowNeed> score_rows;
    model::DecodeScoreCapturePlan score_plan;
    const std::uint32_t* folded_indptr_d = nullptr;
    std::uint8_t* score_rows_base = nullptr;
    if (any_attn_score) {
        score_plan = model::prepare_decode_score_capture(
            in.arena, obs, in.num_q_heads, in.stream);
        if (!score_plan.ok ||
            score_plan.num_requests !=
                static_cast<std::uint32_t>(obs.num_requests)) {
            return veto("score capture plan failed");
        }
        std::size_t max_occurrences = 0;
        for (const auto& lane_ptr : state.lanes) {
            max_occurrences = std::max(
                max_occurrences,
                (*lane_ptr->phase_plans)[PTIR_STAGE_ON_ATTN].size());
        }
        for (std::size_t occurrence = 0;
             occurrence < max_occurrences;
             ++occurrence) {
            for (std::size_t lane_index = 0;
                 lane_index < state.lanes.size();
                 ++lane_index) {
                StagedLane& lane = *state.lanes[lane_index];
                const auto& stages =
                    (*lane.phase_plans)[PTIR_STAGE_ON_ATTN];
                if (occurrence >= stages.size()) continue;
                const plan::StagePlan& stage = *stages[occurrence];
                if (stage.ops.empty()) continue;
                if (!stage_uses_intrinsic(stage, PTIR_INTR_ATTN_SCORE)) {
                    continue;
                }
                const GroupedLaneBinding sizing_binding =
                    make_staged_binding(
                        lane, stage, nullptr, 0, nullptr, 0, nullptr);
                const auto width =
                    hook_stage_attn_score_width(stage, sizing_binding);
                if (!width.has_value() || *width == 0) {
                    return veto("AttnScore width conflict");
                }
                // Request boundary + declared-ceiling checks, mirroring
                // `resolve_lane_attn_score` — re-run before EVERY replay,
                // so a request outgrowing its program's ceiling falls back
                // to the eager body and its loud refusal.
                int request = -1;
                for (int r = 0; r < obs.num_requests; ++r) {
                    if (obs.qo_indptr_h[r] == lane.token_start) {
                        request = r;
                        break;
                    }
                }
                if (request < 0) {
                    return veto("score lane off request boundary");
                }
                const std::uint32_t kv_len =
                    score_plan.folded_offsets_h[request + 1] -
                    score_plan.folded_offsets_h[request];
                if (static_cast<std::uint64_t>(kv_len) > *width) {
                    return veto("kv_len exceeds program score ceiling");
                }
                score_rows.push_back(ScoreRowNeed{
                    occurrence, lane_index, *width,
                    static_cast<std::uint32_t>(request), 0});
            }
        }
        if (score_rows.empty()) return veto("no score rows sized");

        // Carve the score-rows block: the folded-offset device CSR, then
        // one padded row per (occurrence, lane) — shared by every layer
        // (stream order serializes pad → consume → next layer's pad).
        auto align256 = [](std::size_t n) {
            return (n + 255u) & ~static_cast<std::size_t>(255u);
        };
        const std::size_t csr_bytes =
            (static_cast<std::size_t>(obs.num_requests) + 1) *
            sizeof(std::uint32_t);
        std::size_t total = align256(csr_bytes);
        for (auto& need : score_rows) {
            need.row_offset = total;
            total += align256(
                static_cast<std::size_t>(need.kv_max) * sizeof(float));
        }
        score_rows_base = static_cast<std::uint8_t*>(in.arena->acquire(
            model::HookSidebandArena::Region::ScoreRows, total, in.stream));
        if (score_rows_base == nullptr) {
            return veto("score-rows arena acquire failed");
        }
        in.arena->release(model::HookSidebandArena::Region::ScoreRows);
        folded_indptr_d =
            reinterpret_cast<const std::uint32_t*>(score_rows_base);
        state.hook_folded_offsets_h.assign(
            score_plan.folded_offsets_h,
            score_plan.folded_offsets_h + obs.num_requests + 1);
        CUDA_CHECK(cudaMemcpyAsync(
            score_rows_base,
            state.hook_folded_offsets_h.data(),
            csr_bytes,
            cudaMemcpyHostToDevice,
            in.stream));
    }

    // ── Layer table (device-resident layer intrinsic; see Impl comment). ──
    if (impl_->hook_layer_table_len < layers) {
        std::vector<std::uint32_t> iota(layers);
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            iota[layer] = layer;
        }
        std::uint32_t* table = nullptr;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&table),
            static_cast<std::size_t>(layers) * sizeof(std::uint32_t)));
        CUDA_CHECK(cudaMemcpy(
            table, iota.data(),
            static_cast<std::size_t>(layers) * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice));
        if (impl_->hook_layer_table != nullptr) {
            cudaFree(impl_->hook_layer_table);
        }
        impl_->hook_layer_table = table;
        impl_->hook_layer_table_len = layers;
    }

    // ── Commit pass: prepare every (layer, phase, occurrence, group) in the
    // exact order the body will consume them, applying channel effects at
    // the same points the eager execute would. From here on, failure is a
    // failed launch (never a silent eager fallback: cursors and overlays
    // have advanced).
    std::uint64_t fingerprint = 0xcbf29ce484222325ull;
    hook_fp_mix(fingerprint, layers);
    hook_fp_mix(fingerprint, state.lanes.size());
    // The row split (hook_free_prefix_rows) is deliberately NOT mixed:
    // since the Peel device-window campaign the captured body reads the
    // split from pi.peel_window at replay (devwin kernels, full-N
    // grids), so a mixed fire's exec is split-independent and must not
    // churn when lane composition moves the split.
    hook_fp_mix(fingerprint, in.num_q_heads);
    hook_fp_mix(fingerprint, in.wants_attn_score ? 1u : 0u);
    hook_fp_mix_ptr(fingerprint, impl_->hook_layer_table);
    hook_fp_mix_ptr(fingerprint, score_plan.folded);
    hook_fp_mix_ptr(fingerprint, score_plan.indptr_d);
    hook_fp_mix_ptr(fingerprint, score_plan.indptr_h_data);
    hook_fp_mix_ptr(fingerprint, score_rows_base);
    // The mask carve: every address and the stride are baked by the captured
    // sink kernel, the seeding memset, the compaction kernel and the
    // attention that reads its outputs. Stride grows with the page count, so
    // page growth recaptures — the honest cost of a host-sized carve.
    hook_fp_mix_ptr(fingerprint, mask_plan.keep);
    hook_fp_mix(fingerprint, mask_plan.stride);
    hook_fp_mix_ptr(fingerprint, mask_plan.out_indices);
    hook_fp_mix_ptr(fingerprint, mask_plan.out_indptr);
    hook_fp_mix_ptr(fingerprint, mask_plan.out_last_lens);
    for (const auto& lane_ptr : state.lanes) {
        hook_fp_mix(fingerprint, lane_ptr->bound->program_hash);
        hook_fp_mix_ptr(fingerprint, lane_ptr->bound);
        // `token_start` (the lane's fire row) is deliberately NOT mixed —
        // the campaign's second half: every captured consumer of the row
        // is device-indirected. The query/logits/score intrinsic bases go
        // through the per-fire uploaded lane metadata tables, and the
        // score-pad gather reads its lane→request index from
        // `hook_pad_requests` (uploaded below). A hooked lane changing
        // rows therefore replays the same exec.
    }

    // The pads' per-fire lane→request indices, in build order; uploaded
    // into the address-stable table after the commit pass (the pads bake
    // `&table[ordinal]`, never the value). Staged on the launch state so
    // the async upload's pageable source outlives this pass.
    std::vector<std::uint32_t>& pad_request_values =
        state.pad_request_staging;
    pad_request_values.clear();

    std::uint32_t stable_slot = 1;
    state.prepared_attn[0].clear();
    state.prepared_attn[1].clear();
    state.prepared_cursor = {0, 0};
    try {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            for (const std::uint8_t phase : kPhases) {
                HookPreparedInvocation invocation;
                invocation.layer = layer;
                std::size_t max_occurrences = 0;
                for (const auto& lane_ptr : state.lanes) {
                    max_occurrences = std::max(
                        max_occurrences,
                        (*lane_ptr->phase_plans)[phase].size());
                }
                for (std::size_t occurrence = 0;
                     occurrence < max_occurrences;
                     ++occurrence) {
                    struct PreparedTask {
                        StagedLane* lane = nullptr;
                        std::size_t lane_index = 0;
                        const plan::StagePlan* plan = nullptr;
                        const generated::FusedStageExecutable* executable =
                            nullptr;
                        const GroupedStageStaticPlan* group_plan = nullptr;
                        GroupedLaneBinding binding;
                        const std::vector<std::uint32_t>* topology = nullptr;
                        const HookScorePadLaunch* pad = nullptr;
                        bool complete = false;
                    };
                    std::vector<PreparedTask> tasks;
                    std::vector<HookScorePadLaunch> task_pads;
                    tasks.reserve(state.lanes.size());
                    task_pads.reserve(score_rows.size());
                    // Two passes so `task_pads` never reallocates under the
                    // `pad` pointers.
                    for (std::size_t lane_index = 0;
                         lane_index < state.lanes.size();
                         ++lane_index) {
                        StagedLane& lane = *state.lanes[lane_index];
                        const auto& stages = (*lane.phase_plans)[phase];
                        if (occurrence >= stages.size()) continue;
                        const plan::StagePlan& stage = *stages[occurrence];
                        if (stage.ops.empty()) continue;
                        const std::size_t stage_index =
                            static_cast<std::size_t>(
                                &stage - lane.plans->data());
                        GroupedLaneBinding binding = make_staged_binding(
                            lane, stage, nullptr, 0, nullptr, 0,
                            impl_->hook_layer_table + layer);
                        const HookScorePadLaunch* pad = nullptr;
                        if (phase == PTIR_STAGE_ON_ATTN &&
                            stage_uses_intrinsic(
                                stage, PTIR_INTR_ATTN_SCORE)) {
                            const auto need = std::find_if(
                                score_rows.begin(), score_rows.end(),
                                [&](const ScoreRowNeed& candidate) {
                                    return candidate.occurrence ==
                                               occurrence &&
                                           candidate.lane == lane_index;
                                });
                            if (need == score_rows.end()) {
                                throw std::runtime_error(
                                    "hook-graph prepare lost a score row "
                                    "it sized");
                            }
                            auto* row = reinterpret_cast<float*>(
                                score_rows_base + need->row_offset);
                            binding.attn_score_base = row;
                            const std::uint32_t pad_ordinal =
                                static_cast<std::uint32_t>(
                                    pad_request_values.size());
                            pad_request_values.push_back(need->request);
                            task_pads.push_back(HookScorePadLaunch{
                                score_plan.folded,
                                folded_indptr_d,
                                row,
                                pad_ordinal,
                                static_cast<std::uint32_t>(need->kv_max),
                            });
                            pad = &task_pads.back();
                        }
                        if (stage_calls_sink(stage, "attn_page_mask")) {
                            // The resolver's device half
                            // (resolve_lane_page_mask minus the host tag,
                            // which the prepared-mode execute applies):
                            // request-strided row into the planned mask
                            // carve.
                            binding.page_mask = GroupedLanePageMask{
                                mask_plan.keep +
                                    static_cast<std::size_t>(
                                        lane_request[lane_index]) *
                                        mask_plan.stride,
                                mask_plan.stride};
                        }
                        tasks.push_back(PreparedTask{
                            .lane = &lane,
                            .lane_index = lane_index,
                            .plan = &stage,
                            .executable = lane.generated_program
                                              ->stages[stage_index]
                                              .get(),
                            .group_plan =
                                impl_->grouped_plans
                                    .at(lane.generated_program
                                            ->stages[stage_index]
                                            ->runtime_id)
                                    .get(),
                            .binding = binding,
                            .topology = &lane.bound->stage_topologies.at(
                                stage_index),
                            .pad = pad,
                        });
                    }
                    // Group identically to `execute_declared_phase`:
                    // signature + channel topology + accumulator admission.
                    for (std::size_t first_index = 0;
                         first_index < tasks.size();
                         ++first_index) {
                        if (tasks[first_index].complete) continue;
                        PreparedTask& first = tasks[first_index];
                        std::vector<PreparedTask*> members;
                        std::vector<GroupedLaneBinding> bindings;
                        members.push_back(&first);
                        bindings.push_back(first.binding);
                        GroupedStageAccumulator accumulator(
                            *first.group_plan);
                        std::string reason;
                        if (!accumulator.try_add(first.binding, &reason)) {
                            throw std::runtime_error(
                                "PTRP stage is not executable by the "
                                "generic CUDA backend: " + reason);
                        }
                        for (std::size_t candidate = first_index + 1;
                             candidate < tasks.size();
                             ++candidate) {
                            PreparedTask& next = tasks[candidate];
                            if (next.complete ||
                                next.plan->signature_hash !=
                                    first.plan->signature_hash ||
                                *next.topology != *first.topology) {
                                continue;
                            }
                            reason.clear();
                            if (!accumulator.try_add(
                                    next.binding, &reason)) {
                                continue;
                            }
                            bindings.push_back(next.binding);
                            members.push_back(&next);
                        }
                        for (PreparedTask* member : members) {
                            member->complete = true;
                        }
                        const GroupedExecutionOptions execution_options{
                            .reset_commits = false,
                            .pull_tickets = false,
                            .finalize = false,
                            .time_sections = false,
                        };
                        HookPreparedGroup group;
                        group.prepared = generated::prepare_generated_stage(
                            bindings,
                            *first.executable,
                            impl_->generated_runtime,
                            in.stream,
                            execution_options,
                            generated::PreparedBufferMode::kStablePerStage,
                            stable_slot++);
                        if (!group.prepared->allocations.values.empty()) {
                            // The feasibility scan is supposed to exclude
                            // every per-fire-allocating stage shape; a
                            // capture over stream-ordered frees would
                            // replay dangling addresses.
                            throw std::runtime_error(
                                "hook-graph prepare produced per-fire "
                                "device allocations; stage shape is not "
                                "replayable");
                        }
                        for (PreparedTask* member : members) {
                            if (member->pad != nullptr) {
                                group.score_pads.push_back(*member->pad);
                            }
                            record_stage_channel_effects(
                                *member->lane, *member->plan);
                        }
                        hook_fp_mix(fingerprint, first.plan->signature_hash);
                        hook_fp_mix(
                            fingerprint, first.executable->runtime_id);
                        hook_fp_mix(fingerprint, members.size());
                        for (const auto& pad : group.score_pads) {
                            hook_fp_mix_ptr(fingerprint, pad.folded);
                            hook_fp_mix_ptr(fingerprint, pad.folded_indptr);
                            hook_fp_mix(fingerprint, pad.ordinal);
                            hook_fp_mix_ptr(fingerprint, pad.row);
                            hook_fp_mix(fingerprint, pad.kv_max);
                        }
                        hook_fp_mix_prepared(fingerprint, *group.prepared);
                        invocation.groups.push_back(std::move(group));
                    }
                }
                invocation.expects_scores =
                    phase == PTIR_STAGE_ON_ATTN &&
                    std::any_of(
                        invocation.groups.begin(),
                        invocation.groups.end(),
                        [](const HookPreparedGroup& group) {
                            return !group.score_pads.empty();
                        });
                invocation.expected_folded = score_plan.folded;
                if (any_page_mask) {
                    bool marks = false;
                    for (const auto& lane_ptr : state.lanes) {
                        const auto& stages =
                            (*lane_ptr->phase_plans)[phase];
                        for (const plan::StagePlan* stage : stages) {
                            if (!stage->ops.empty() &&
                                stage_calls_sink(
                                    *stage, "attn_page_mask")) {
                                marks = true;
                            }
                        }
                    }
                    invocation.marks_mask = marks;
                    invocation.expected_mask_keep = mask_plan.keep;
                    invocation.expected_mask_stride = mask_plan.stride;
                }
                state
                    .prepared_attn[phase - PTIR_STAGE_ON_ATTN_PROJ]
                    .push_back(std::move(invocation));
            }
        }
    } catch (...) {
        state.failed = true;
        throw;
    }
    // Upload this fire's lane→request indices into the address-stable pad
    // table (device indirection — see Impl::hook_pad_requests). Ordered on
    // the fire stream, so both the capture-time launches and every replay
    // consume THIS fire's mapping. Growth (first fire, or more pads than
    // ever) moves the base; the fingerprint mixes it, so that one fire
    // honestly recaptures.
    if (!pad_request_values.empty()) {
        const auto needed =
            static_cast<std::uint32_t>(pad_request_values.size());
        if (impl_->hook_pad_request_capacity < needed) {
            std::uint32_t capacity =
                impl_->hook_pad_request_capacity == 0
                    ? 64u
                    : impl_->hook_pad_request_capacity;
            while (capacity < needed) capacity *= 2u;
            std::uint32_t* grown = nullptr;
            CUDA_CHECK(cudaMalloc(
                &grown, static_cast<std::size_t>(capacity) *
                            sizeof(std::uint32_t)));
            if (impl_->hook_pad_requests != nullptr) {
                cudaFree(impl_->hook_pad_requests);
            }
            impl_->hook_pad_requests = grown;
            impl_->hook_pad_request_capacity = capacity;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            impl_->hook_pad_requests, pad_request_values.data(),
            static_cast<std::size_t>(needed) * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, in.stream));
    }
    hook_fp_mix_ptr(fingerprint, impl_->hook_pad_requests);
    // The body's per-layer hook invocations are accounted here — the
    // prepared-mode `execute_attention_phase` only replays launches, and on
    // a graph REPLAY it does not run at all. `finish`'s coverage check
    // (every declared attention phase ran at every layer) is thereby
    // asserted by this pass's construction instead.
    state.phase_invocations[PTIR_STAGE_ON_ATTN_PROJ] = layers;
    state.phase_invocations[PTIR_STAGE_ON_ATTN] = layers;
    state.hook_graph_prepared = true;
    return fingerprint == 0 ? 1 : fingerprint;
}

void Dispatch::verify_hook_capture_consumed(StagedLaunch& launch) const {
    StagedLaunch::State& state = *launch.state_;
    if (!state.hook_graph_prepared) {
        throw std::runtime_error(
            "hook-graph capture verification on an unprepared launch");
    }
    for (std::size_t phase = 0; phase < 2; ++phase) {
        if (state.prepared_cursor[phase] !=
            state.prepared_attn[phase].size()) {
            throw std::runtime_error(
                "hook-graph capture consumed " +
                std::to_string(state.prepared_cursor[phase]) + " of " +
                std::to_string(state.prepared_attn[phase].size()) +
                " prepared attention invocations (phase index " +
                std::to_string(phase) +
                "): the model body did not invoke its hooks at every "
                "layer, so the captured graph is incomplete");
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
    bool query_is_f32,
    const model::StageHookSideband& sideband) {
    if (phase != PTIR_STAGE_ON_ATTN_PROJ &&
        phase != PTIR_STAGE_ON_ATTN) {
        throw std::runtime_error("model hook invoked a non-attention PTIR phase");
    }
    StagedLaunch::State& state = *launch.state_;
    if (state.hook_graph_prepared) {
        // Prepared mode — THE path for pure-decode hook fires, eager and
        // graph alike (stage 6 increment 4; eager unification in
        // `run_forward_dispatch`): the fire-level pass already did every
        // piece of host work for this invocation, in this exact order. What
        // remains — what an eager body executes and what a capturing stream
        // records — is launches against prepared state, on the stream the
        // model body handed the hook (the capture stream, under capture).
        // Retrieval is exact-order by construction: a cursor per phase,
        // loud throws on any divergence from the prepared sequence.
        const std::size_t phase_index =
            static_cast<std::size_t>(phase - PTIR_STAGE_ON_ATTN_PROJ);
        auto& invocations = state.prepared_attn[phase_index];
        const std::size_t at = state.prepared_cursor[phase_index];
        if (at >= invocations.size() || invocations[at].layer != layer) {
            state.failed = true;
            throw std::runtime_error(
                "hook-graph prepared phase order mismatch: phase " +
                std::to_string(static_cast<int>(phase)) + " layer " +
                std::to_string(layer) + " does not match prepared entry " +
                std::to_string(at) + " of " +
                std::to_string(invocations.size()));
        }
        ++state.prepared_cursor[phase_index];
        HookPreparedInvocation& invocation = invocations[at];
        try {
            if (invocation.marks_mask) {
                // The resolver's host half (resolve_lane_page_mask): tag
                // the model's sink so its compact branch fires for this
                // layer — and prove the model carved the SAME arena block
                // the prepared kernels bake, the one moment that is
                // observable (capture time; a replayed body never runs
                // this code, and the tag then lives inside the recorded
                // branch structure).
                model::AttentionMaskSink* sink = sideband.mask_sink;
                if (sink == nullptr || !sink->usable() ||
                    sink->keep != invocation.expected_mask_keep ||
                    sink->stride != invocation.expected_mask_stride) {
                    throw std::runtime_error(
                        "hook-graph prepared attn_page_mask expected the "
                        "planned sideband carve; the model body published "
                        "a different mask sink");
                }
                sink->written_layer = static_cast<int>(layer);
            }
            if (invocation.expects_scores) {
                const model::AttentionScores* scores = sideband.scores;
                if (scores == nullptr || !scores->usable() ||
                    scores->layer != layer ||
                    scores->values != invocation.expected_folded) {
                    throw std::runtime_error(
                        "hook-graph prepared OnAttn expected this layer's "
                        "score capture in the planned sideband slot; the "
                        "model body published " +
                        std::string(scores == nullptr ? "nothing"
                                                      : "a different capture"));
                }
            }
            for (auto& group : invocation.groups) {
                for (const HookScorePadLaunch& pad : group.score_pads) {
                    const std::uint32_t blocks =
                        std::min<std::uint32_t>(
                            (pad.kv_max + 255u) / 256u, 65535u);
                    k_hook_attn_score_pad<<<blocks, 256, 0, stream>>>(
                        pad.folded, pad.folded_indptr,
                        impl_->hook_pad_requests + pad.ordinal,
                        pad.row, pad.kv_max);
                    CUDA_CHECK(cudaGetLastError());
                }
                generated::launch_generated_stage(*group.prepared, stream);
            }
        } catch (...) {
            state.failed = true;
            throw;
        }
        return;
    }
    // Legacy interleaved fallback: reached only when the fire-level prepare
    // pass VETOED this fire (`prepare_attention_phases` returned 0 — Query
    // readers like quest, lora/envelope_dot, scalable nucleus, non-decode
    // fires, off-boundary lanes) or when the frame never wired the seam.
    // Prepared pure-decode hook fires — eager or graph — never reach this
    // branch. It cannot be deleted while those veto classes exist: a
    // Query-reading stage is unhoistable by construction (the Query tensor
    // is produced by THIS layer's kernels, mid-body, and the bf16→f32 cast
    // below allocates per fire).
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
            query_rows, query_columns, layer, stream, sideband);
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
    const pie::driver::fire::LaunchView& view,
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
    const std::int32_t* presampled_tokens,
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
    // A depth-truncated launch (region-table uniform k) legitimately runs
    // its attention phases at only the planned layers. A DEPTH-SPLIT
    // launch (full-depth rows present, truncated suffix) still walks the
    // full model — the split narrows rows, not the layer walk.
    const std::uint32_t expected_layers =
        view.planned_max_layers == 0xffffffffu ||
                view.planned_full_depth_rows > 0
            ? impl_->model_layers
            : std::min(impl_->model_layers, view.planned_max_layers);
    for (std::uint8_t phase :
         {std::uint8_t{PTIR_STAGE_ON_ATTN_PROJ},
          std::uint8_t{PTIR_STAGE_ON_ATTN}}) {
        const bool declared = std::any_of(
            state.lanes.begin(), state.lanes.end(),
            [phase](const auto& lane) {
                return !(*lane->phase_plans)[phase].empty();
            });
        if (declared &&
            state.phase_invocations[phase] != expected_layers) {
            throw std::runtime_error(
                "PTIR attention phase did not execute at every model layer");
        }
    }
    if (state.hook_graph_prepared) {
        // Prepared mode: 0 consumed = a graph REPLAY (the recorded body ran
        // on the GPU, not through these cursors); fully consumed = the
        // capture fire. Anything in between means the model body stopped
        // invoking hooks mid-fire — a structural failure the eager
        // invocation counter above can no longer see (the prepare pass
        // pre-credits it).
        for (std::size_t phase_index = 0; phase_index < 2; ++phase_index) {
            const std::size_t consumed = state.prepared_cursor[phase_index];
            const std::size_t total =
                state.prepared_attn[phase_index].size();
            if (consumed != 0 && consumed != total) {
                throw std::runtime_error(
                    "hook-graph prepared attention phases were partially "
                    "consumed (" + std::to_string(consumed) + " of " +
                    std::to_string(total) + ")");
            }
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
        lane.presampled_token_rows.clear();
        lane.row_valid = row_valid;
        lane.row_valid_offset =
            row_valid == nullptr ? 0 : row_valid_offsets[program];
        if (direct_bf16_logits != nullptr &&
            direct_row_indices != nullptr) {
            if (presampled_tokens != nullptr) {
                lane.presampled_token_rows.reserve(lane.sampled_rows);
            } else {
                lane.logits_bf16_rows.reserve(lane.sampled_rows);
            }
            for (std::uint32_t row = 0; row < lane.sampled_rows; ++row) {
                const std::uint32_t source =
                    direct_row_indices[lane.row_offset + row];
                if (direct_bf16_row_capacity != 0 &&
                    source >= direct_bf16_row_capacity) {
                    throw std::runtime_error(
                        "direct BF16 sampled row exceeds the logits layout");
                }
                if (presampled_tokens != nullptr) {
                    // Same `source` row, one token wide instead of a
                    // vocabulary. Exclusive with the BF16 table rather than
                    // alongside it: on this path `direct_bf16_logits` holds
                    // slab scratch, so a row table into it would be a live
                    // pointer to something nobody wrote.
                    lane.presampled_token_rows.push_back(
                        reinterpret_cast<std::uint64_t>(
                            presampled_tokens + source));
                } else {
                    lane.logits_bf16_rows.push_back(
                        reinterpret_cast<std::uint64_t>(
                            direct_bf16_logits +
                            static_cast<std::size_t>(source) * vocab));
                }
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
            /*sideband=*/{},
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
        impl_->notify_stream,
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
                BoundInstance::CommitSnapshot::kWords *
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
    const HostPublishTransport publish_transport =
        select_host_publish_transport(*notify, impl_->output_copy_stream);
    // Only the copy-engine batch moves to the dedicated copy stream; the
    // scatter kernel stays on `callback_stream` so plain stream order already
    // places it ahead of the settlement kernel that publishes the tails.
    const bool batch_copies =
        publish_transport == HostPublishTransport::Batched;
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
    // Commit snapshots are pooled per (instance, wave-occurrence) and reused
    // every wave, so this wave's D2H publications write the very buffers the
    // PREVIOUS wave's settlement callback reads. That callback used to sit on
    // the compute stream, which enforced the ordering implicitly; it now runs
    // on `notify_stream`, so state the dependency explicitly. Placed here
    // rather than at the wave's head it costs nothing: a full forward pass
    // separates the two, and the callback has long since retired.
    if (impl_->settlement_callbacks_recorded) {
        CUDA_CHECK(cudaStreamWaitEvent(
            settlement_stream,
            impl_->settlement_callbacks_done,
            0));
    }
    enqueue_host_publish_copies(
        *notify, settlement_stream, publish_transport);
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
    // Hand the settlement callback to a stream of its own: `cudaLaunchHostFunc`
    // holds its stream until the driver's callback thread runs the function, so
    // leaving it here would stall the compute stream for the wakeup latency
    // even though the next wave is already queued behind it.
    ensure_event(&impl_->settlement_ready);
    CUDA_CHECK(cudaEventRecord(
        impl_->settlement_ready, settlement_stream));
    CUDA_CHECK(cudaStreamWaitEvent(
        impl_->notify_stream, impl_->settlement_ready, 0));
    const cudaError_t callback_status = cudaLaunchHostFunc(
        impl_->notify_stream, notify_runtime_callback, notify);
    if (callback_status != cudaSuccess) {
        CUDA_CHECK(callback_status);
    }
    const cudaError_t event_status = cudaEventRecord(
        notify->callback_done, impl_->notify_stream);
    if (event_status == cudaSuccess) {
        notify->callback_pending = true;
        ensure_event(&impl_->settlement_callbacks_done);
        CUDA_CHECK(cudaEventRecord(
            impl_->settlement_callbacks_done, impl_->notify_stream));
        impl_->settlement_callbacks_recorded = true;
    } else {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] settlement callback event record failed: %s\n",
            cudaGetErrorString(event_status));
        const cudaError_t sync_status =
            cudaStreamSynchronize(impl_->notify_stream);
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
    static constexpr std::uint32_t kZeroWords
        [BoundInstance::CommitSnapshot::kWords] = {};
    for (auto& lane : state.lanes) {
        if (lane->snapshot != nullptr &&
            lane->snapshot->device != nullptr) {
            cudaMemcpyAsync(
                lane->snapshot->device,
                kZeroWords,
                sizeof(kZeroWords),
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
    const pie::driver::fire::LaunchView& view,
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
    const pie::driver::fire::LaunchView& view,
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

bool Dispatch::stage_decode_envelopes(
    const pie::driver::fire::LaunchView& view,
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
        // FramePrepare-time consumer: the Prologue has not executed yet;
        // its statically-known put effects stand in for the live set.
        const auto& pending_slots =
            staged.lanes[program]->prologue_put_slots;
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
            const PreparedCursor cursor = lane_ticket_window(
                *staged.lanes[program], slot, state.channels);
            DeviceChannelRegistry& registry = *channel_view.registry();
            source = reinterpret_cast<std::uintptr_t>(
                static_cast<std::uint8_t*>(registry.cell_base(slot)) +
                static_cast<std::size_t>(
                    pending ? cursor.tail_index : cursor.head_index) *
                    registry.cell_bytes(slot));
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
    staged.decode_envelope_lanes = std::move(lanes);
    staged.decode_envelope_lane_count = envelope_lanes;
    staged.decode_envelope_template_pages =
        template_kv_page_indptr[staged.decode_envelope_lanes.size()];
    staged.decode_envelopes_staged = true;
    return true;
}

bool Dispatch::enqueue_decode_envelopes(
    const DecodeEnvelopeDeviceBuffers& buffers,
    std::string* err,
    StagedLaunch& launch) {
    if (err != nullptr) err->clear();
    Impl& state = *impl_;
    StagedLaunch::State& staged = *launch.state_;
    if (!staged.decode_envelopes_staged || !staged.active) {
        if (err != nullptr) {
            *err = "ptir decode envelopes were not staged for this launch";
        }
        return false;
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
            staged.decode_envelope_lanes, buffers.kv_page_indices,
            staged.decode_envelope_template_pages, staged.stream);
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
    const std::size_t lane_count = staged.decode_envelope_lanes.size();
    std::uint32_t threads = 32;
    while (threads < lane_count) threads *= 2;
    compose_decode_envelopes<<<
        1,
        threads,
        lane_count * sizeof(std::uint32_t),
        staged.stream>>>(
        uploaded.lanes,
        static_cast<std::uint32_t>(lane_count),
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
        state.stats.decode_envelope_lanes +=
            staged.decode_envelope_lane_count;
    }
    return true;
}

bool Dispatch::stage_fixed_decode(
    const pie::driver::fire::LaunchView& view,
    std::uint32_t page_size,
    std::uint32_t device_pages,
    const FixedDecodeDeviceBuffers& buffers,
    std::string* err,
    StagedLaunch& launch,
    const FixedDecodeScope& scope) {
    if (err != nullptr) err->clear();
    Impl& state = *impl_;
    StagedLaunch::State& staged = *launch.state_;
    const std::size_t total_programs = view.ptir_program_hashes.size();
    const std::size_t program_begin = scope.program_begin;
    const std::size_t programs = scope.program_count == 0
        ? total_programs
        : scope.program_count;
    if (programs == 0 || program_begin + programs > total_programs ||
        view.ptir_program_instances.size() != total_programs ||
        staged.lanes.size() != total_programs ||
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
    if (view.kv_translation_indptr.size() != total_programs + 1 ||
        view.kv_translation_indptr.data()[0] != 0 ||
        view.kv_translation_indptr.data()[total_programs] !=
            view.kv_translation.size()) {
        if (err != nullptr) {
            *err = "ptir fixed decode translation table is malformed";
        }
        return false;
    }
    const bool has_write_bounds =
        view.ptir_kv_write_lower_bounds.size() == total_programs &&
        view.ptir_kv_write_upper_bounds.size() == total_programs;
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
    for (std::size_t lane_index = 0; lane_index < programs; ++lane_index) {
        const std::size_t program = program_begin + lane_index;
        ProgramPorts& program_ports = ports[lane_index];
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

    // Host-writer input availability is a stage-time (admission) check;
    // the ring PULL itself is stream work and runs in the enqueue half at
    // the step's stream position.
    for (std::size_t lane_index = 0; lane_index < programs; ++lane_index) {
        BoundInstance& instance = *ports[lane_index].instance;
        std::string value_error;
        if (!instance.instance->writer_inputs_available(
                &value_error)) {
            throw RetryableLaunchError(value_error);
        }
    }

    std::vector<std::uint32_t> upload_values(
        view.kv_translation.data(),
        view.kv_translation.data() + view.kv_translation.size());
    for (std::size_t lane_index = 0; lane_index < programs; ++lane_index) {
        if (ports[lane_index].by_tag[kPortPositions] != nullptr) continue;
        ports[lane_index].wire_position_offset = upload_values.size();
        upload_values.push_back(
            view.position_ids.data()[program_begin + lane_index]);
    }
    std::vector<FixedDecodeLane> lanes(programs);
    for (std::size_t lane_index = 0; lane_index < programs; ++lane_index) {
        const std::size_t program = program_begin + lane_index;
        ProgramPorts& program_ports = ports[lane_index];
        ChannelView& channel_view =
            program_ports.instance->instance->view();
        // FramePrepare-time consumer: the Prologue has not executed yet;
        // its statically-known put effects stand in for the live set.
        const auto& pending_slots =
            staged.lanes[program]->prologue_put_slots;
        FixedDecodeLane& lane = lanes[lane_index];
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
            const PreparedCursor cursor = lane_ticket_window(
                *staged.lanes[program], slot, state.channels);
            DeviceChannelRegistry& registry = *channel_view.registry();
            auto* cell_base =
                static_cast<std::uint8_t*>(registry.cell_base(slot));
            const std::size_t cell_bytes = registry.cell_bytes(slot);
            *sources[index] = reinterpret_cast<std::uintptr_t>(
                cell_base +
                static_cast<std::size_t>(
                    pending ? cursor.tail_index : cursor.head_index) *
                    cell_bytes);
            lane.ready[index] = pending
                ? 0
                : reinterpret_cast<std::uintptr_t>(
                      channel_view.d_full() +
                      static_cast<std::size_t>(slot) * kMaxRing +
                      cursor.head_index);
        }
        lane.pass_commit = reinterpret_cast<std::uintptr_t>(
            staged.lanes[program]->snapshot->device);
        if (has_write_bounds) {
            lane.write_lower_bound =
                view.ptir_kv_write_lower_bounds.data()[program];
            lane.write_upper_bound =
                view.ptir_kv_write_upper_bounds.data()[program];
        }
        lane.translation_len = program_ports.translation_len;
        lane.pages_capacity = program_ports.pages_capacity;
    }
    // Arena-relative pointers (lane.translation / wire-position source)
    // are patched at enqueue, after the upload-arena claim.
    staged.fixed_decode_translation_begin.resize(programs);
    staged.fixed_decode_position_offset.resize(programs);
    for (std::size_t lane_index = 0; lane_index < programs; ++lane_index) {
        staged.fixed_decode_translation_begin[lane_index] =
            ports[lane_index].translation_begin;
        staged.fixed_decode_position_offset[lane_index] =
            ports[lane_index].wire_position_offset;
    }
    staged.fixed_decode_lanes = std::move(lanes);
    staged.fixed_decode_upload_values = std::move(upload_values);
    staged.fixed_decode_page_size = page_size;
    staged.fixed_decode_device_pages = device_pages;
    staged.fixed_decode_scope = scope;
    staged.fixed_decode_staged = true;
    return true;
}

bool Dispatch::enqueue_fixed_decode(
    const FixedDecodeDeviceBuffers& buffers,
    std::string* err,
    StagedLaunch& launch) {
    if (err != nullptr) err->clear();
    Impl& state = *impl_;
    StagedLaunch::State& staged = *launch.state_;
    if (!staged.fixed_decode_staged || !staged.active) {
        if (err != nullptr) {
            *err = "ptir fixed decode was not staged for this launch";
        }
        return false;
    }
    const std::size_t programs = staged.fixed_decode_lanes.size();
    const Dispatch::FixedDecodeScope& scope = staged.fixed_decode_scope;
    // Pull host-writer rings on the LAUNCH stream: the compose kernel and
    // every stage kernel are ordered behind these copies, and the staging
    // rides the launch state past their completion — no whole-device
    // synchronize on the fire path.
    for (std::size_t lane_index = 0; lane_index < programs; ++lane_index) {
        staged.lanes[scope.program_begin + lane_index]
            ->bound->instance->pull_writer_inputs(
                staged.stream, staged.writer_staging);
    }
    state.fixed_decode_upload.reserve(
        programs, staged.fixed_decode_upload_values.size(), staged.stream);
    for (std::size_t program = 0; program < programs; ++program) {
        FixedDecodeLane& lane = staged.fixed_decode_lanes[program];
        if (staged.fixed_decode_position_offset[program] !=
            std::numeric_limits<std::size_t>::max()) {
            lane.position = reinterpret_cast<std::uintptr_t>(
                state.fixed_decode_upload.translation_at(
                    staged.fixed_decode_position_offset[program]));
        }
        lane.translation = reinterpret_cast<std::uintptr_t>(
            state.fixed_decode_upload.translation_at(
                staged.fixed_decode_translation_begin[program]));
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
            staged.fixed_decode_lanes,
            staged.fixed_decode_upload_values,
            staged.stream);
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
        .page_size = staged.fixed_decode_page_size,
        .device_pages = staged.fixed_decode_device_pages,
        .row_base = scope.row_base,
        .request_base = scope.request_base,
        .page_base = scope.page_base,
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
    // One line per process, on the first templated batch: whether this build's
    // decode template is carrying recurrent state. There is no cheaper way to
    // observe it -- `DispatchStats` is wired only into fire-timing debug, and
    // the whole effect of admitting an RS fire here is that a slower path is
    // NOT taken, which no assertion downstream can see. If the RS guard above
    // is ever re-tightened, this is what says so.
    //
    // The env read is a function-local static, not a per-batch `getenv`: this
    // runs once per decode step on the latency path, and reading the
    // environment there measurably costs (~1.5% of single-request tok/s).
    static const bool trace_template =
        std::getenv("PIE_FIXED_DECODE_TRACE") != nullptr;
    if (trace_template) {
        static std::once_flag once;
        std::call_once(once, [&] {
            std::cerr << "[pie-driver-cuda] fixed-decode template active: "
                      << "programs=" << programs << " recurrent_state="
                      << (buffers.rs_slot_ids != nullptr ? "yes" : "no")
                      << "\n";
        });
    }
    return true;
}

// A lane whose pass-commit word came back zero was refused by exactly one of
// the gates in `k_pull_validate_host_channels_batch`. Replay those gates on
// the host (the ticket words are mapped pinned memory the CPU can read) so
// the retry names the channel and the predicate that failed instead of the
// whole class of causes.
std::string describe_uncommitted_lane(
    std::uint64_t instance_id,
    const BoundInstance& bound,
    const StagedLane& lane) {
    std::string message =
        "ptir prologue or channel readiness did not commit";
    if (bound.trace == nullptr) return message;
    auto channel_label = [&](std::uint32_t slot) -> std::string {
        for (std::size_t c = 0; c < bound.trace->channels.size() &&
                                c < bound.channel_ids.size();
             ++c) {
            if (bound.instance == nullptr) break;
            if (bound.instance->view().slot(bound.trace->channels[c].id) !=
                slot) {
                continue;
            }
            const auto& channel = bound.trace->channels[c];
            std::string label = "chan#" + std::to_string(channel.id);
            if (!channel.extern_name.empty()) {
                label += "(" + channel.extern_name + ")";
            }
            return label;
        }
        return "chan?";
    };
    std::string refused;
    std::size_t refused_count = 0;
    for (const DeviceHostChannelTicket& ticket : lane.tickets) {
        if (ticket.words == nullptr) continue;
        const std::uint64_t head =
            std::atomic_ref<const std::uint64_t>(ticket.words[0])
                .load(std::memory_order_acquire);
        const std::uint64_t tail =
            std::atomic_ref<const std::uint64_t>(ticket.words[1])
                .load(std::memory_order_acquire);
        const char* reason = nullptr;
        if ((ticket.flags & kTicketConsume) != 0 &&
            head != ticket.expected_head) {
            reason = "consume-head-moved";
        } else if ((ticket.flags & kTicketRequireInput) != 0 &&
                   !(tail > head)) {
            reason = "required-input-empty";
        } else if ((ticket.flags & kTicketPublish) != 0) {
            const std::uint64_t same_fire_consume =
                (ticket.flags & kTicketConsume) != 0 ? 1u : 0u;
            if (tail != ticket.expected_tail) {
                reason = "publish-tail-moved";
            } else if (!(tail - head <
                         static_cast<std::uint64_t>(ticket.cap1 - 1) +
                             same_fire_consume)) {
                reason = "publish-ring-full";
            }
        }
        if (reason == nullptr) continue;
        ++refused_count;
        if (refused_count > 4) continue;
        if (!refused.empty()) refused += ", ";
        refused += channel_label(ticket.slot);
        refused += " slot=" + std::to_string(ticket.slot);
        refused += " ";
        refused += reason;
        refused += " head=" + std::to_string(head);
        refused += "/exp" + std::to_string(ticket.expected_head);
        refused += " tail=" + std::to_string(tail);
        refused += "/exp" + std::to_string(ticket.expected_tail);
        refused += " cap1=" + std::to_string(ticket.cap1);
    }
    message += " (instance=" + std::to_string(instance_id);
    message += " geometry_class=" +
               std::to_string(static_cast<int>(bound.geometry_class));
    message += " tickets=" + std::to_string(lane.tickets.size());
    message += " snapshot_seeded=" +
               std::string(lane.snapshot != nullptr &&
                                   lane.snapshot->ever_validated
                               ? "yes"
                               : "no");
    if (refused.empty()) {
        // Every ring gate agrees: the refusal came from the device-side
        // prologue itself (a compose fail-stop or an envelope kill), not
        // from host channel staging.
        message += " refused_by=none-of-the-ring-gates)";
    } else {
        message += " refused=" + std::to_string(refused_count);
        message += " [" + refused + "])";
    }
    return message;
}

int Dispatch::dense_mask_scope_violation(const pie::driver::fire::LaunchView& view,
                                         bool allow_structured_masks) const {
    const std::size_t n_prog = view.ptir_program_hashes.size();
    if (n_prog <= 1 || view.ptir_program_instances.size() != n_prog) {
        return -1;
    }
    // NS-2 (the spatial mask fire): a multi-program step whose scheduler
    // PLANNED an unmasked prefix split may carry dense-masked programs —
    // the split body serves the masked suffix with the custom kernel and
    // the frame packs the mask at its composed suffix positions. Deeper
    // shape checks (single masked program, suffix placement) fail loud at
    // the frame's pack; admission only answers the scope question.
    static const bool spatial_on = [] {
        const char* v = std::getenv("PIE_SPATIAL_MASK");
        return v == nullptr || v[0] != '0';
    }();
    if (spatial_on &&
        view.planned_unmasked_prefix_rows != PIE_UNMASKED_PREFIX_UNPLANNED) {
        // planned == 0 is the all-masked composed fire: the suffix covers
        // every row and the custom kernel serves the whole step.
        return -1;
    }
    const Impl& s = *impl_;
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        const auto it = s.instances.find(iid);
        // Unknown instances / missing traces fail elsewhere with their own
        // diagnostics; this check answers only the mask-scope question.
        if (it == s.instances.end() || it->second.trace == nullptr) continue;
        // The ACK'd class — not a trace sniff — decides which programs
        // resolve descriptors from device cells (RV-6, mirrors
        // `resolve_descriptors`).
        if (it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST) continue;
        const Trace& trace = *it->second.trace;
        for (const PortBinding& binding : trace.ports) {
            if (binding.port != kPortAttnMask || binding.is_const) continue;
            // Mirror `resolve_attention_mask`: a statically recognized
            // structured mask lowers to a runtime window override and never
            // packs a dense device mask.
            const auto descriptor =
                structured_mask_descriptor(trace, binding.channel);
            const bool direct =
                allow_structured_masks &&
                (descriptor.kind == StructuredMaskKind::Causal ||
                 (descriptor.kind == StructuredMaskKind::SlidingWindow &&
                  descriptor.window > 0) ||
                 (descriptor.kind == StructuredMaskKind::SinkWindow &&
                  descriptor.window > 0));
            if (!direct) return static_cast<int>(p);
        }
    }
    return -1;
}

bool Dispatch::resolve_descriptors(const pie::driver::fire::LaunchView& view,
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
        // (frame.cpp) takes the composed batch from R to its request
        // bucket with device-side pad rows, so the template no longer
        // requires R to sit exactly on the lattice.
        // Plain per-request RS slots ride through untouched: this template
        // resolves *geometry* (tokens, positions, pages), and the slot->request
        // attribution is host data that `compose_forward_batch` copies and that
        // graph padding extends with `graph_pad_slot`. A fold or a buffered
        // working set is different -- both change the batch's shape -- so those
        // two still refuse.
        //
        // Refusing on `rs_slot_ids` too is what shut every recurrent-state
        // family out of device composition, which is the only path that can
        // resolve a chained descriptor: the host readback fallback cannot see a
        // value the producing fire has not committed yet, so a decode step that
        // reads the prefill's sampled token never became ready.
        const bool trace_compose = [] {
            const char* v = std::getenv("PIE_TRACE_DEVICE_COMPOSE");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        auto refuse = [&](const std::string& why) {
            if (trace_compose) {
                std::fprintf(stderr, "[compose] refused: %s\n", why.c_str());
                std::fflush(stderr);
            }
            return false;
        };
        if (staged == nullptr) return refuse("no staged launch");
        if (page_size == 0 || device_pages == 0) return refuse("no kv pages");
        // A *bound* fold-length array is not a fold. The runtime carries one
        // entry per request for every recurrent-state family, and on an
        // ordinary decode step every entry is zero -- the state advances by
        // the step's own token, with nothing to replay. Only a non-zero
        // length reshapes the batch, and only that has to refuse.
        {
            const auto folds = view.rs_fold_lens.as<std::uint32_t>();
            std::size_t nonzero = 0;
            for (std::size_t i = 0; i < folds.size(); ++i) {
                if (folds[i] != 0) ++nonzero;
            }
            if (nonzero != 0) {
                if (trace_compose) {
                    std::fprintf(stderr,
                                 "[compose] refused: rs fold (%zu/%zu non-zero)\n",
                                 nonzero, folds.size());
                    std::fflush(stderr);
                }
                return false;
            }
        }
        if (!view.rs_buffer_slot_ids.empty()) return refuse("rs buffer");
        // A DEVICE-RESIDENT fold length (`PIE_RS_FLAG_FOLD_LEN_DEVICE`). Its
        // value is substituted during descriptor resolution and clamped
        // against a host bound; this template resolves nothing, so the wire
        // array would still hold that bound -- folding more than the accepted
        // prefix, and tokens absorbed into the recurrence are unrecoverable.
        // frame.cpp refuses the combination outright, so it must refuse here
        // rather than later.
        //
        // The non-zero fold check above happens to catch this today, but only
        // because the host's bound is non-zero. The flag is what actually
        // carries the rule, so the flag is what is tested.
        if (std::any_of(view.rs_slot_flags.data(),
                        view.rs_slot_flags.data() + view.rs_slot_flags.size(),
                        [](std::uint8_t f) {
                            return (f & PIE_RS_FLAG_FOLD_LEN_DEVICE) != 0;
                        })) {
            return refuse("rs device-resident fold length");
        }
        if (view.kv_translation_indptr.size() != n_prog + 1)
            return refuse("kv_translation_indptr size");
        if (view.kv_translation_indptr.data()[0] != 0)
            return refuse("kv_translation_indptr[0]");
        if (view.kv_translation_indptr.data()[n_prog] !=
            view.kv_translation.size())
            return refuse("kv_translation_indptr tail");
        if (view.ptir_kv_write_lower_bounds.size() != n_prog)
            return refuse("kv_write_lower_bounds size " +
                          std::to_string(view.ptir_kv_write_lower_bounds.size()) +
                          " != n_prog " + std::to_string(n_prog));
        if (view.ptir_kv_write_upper_bounds.size() != n_prog)
            return refuse("kv_write_upper_bounds size");
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
                return refuse("geometry class not decode-envelope");
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
            // Mixed steps reserve the FULL envelope width in the composed
            // CSRs (the offset fixed-decode compose writes actual counts
            // in place on device); the all-envelope whole-step form keeps
            // the 1-page placeholder (enqueue_fixed_decode rewrites every
            // CSR from lane 0).
            const std::uint32_t envelope_width =
                static_cast<std::uint32_t>(std::max<std::size_t>(
                    1,
                    std::min<std::size_t>(
                        pages->type.shape.numel(),
                        translation_end - translation_begin)));
            geometry.kv_page_indices.assign(envelope_width, 0);
            geometry.kv_page_indptr = {0, envelope_width};
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
        if (!all_decode_envelopes) {
            // Mixed [wire][envelope] step: the envelope lanes' shape
            // templates (full-width reserves above) route through the
            // OFFSET fixed-decode compose after the ordinary wire refill
            // — never `enqueue_decode_envelopes` (which trusts host page
            // spans) and never the synchronizing readback fallback below
            // (chained values do not exist host-side).
            candidate.mixed_envelope = true;
            out = std::move(candidate);
            return true;
        }
        // The whole-step 1-page placeholder form is consumable ONLY by
        // `enqueue_fixed_decode`, whose graph-bucket planning the
        // `allow_device_composed` gate guards.
        if (!allow_device_composed) return false;
        candidate.device_composed = true;
        out = std::move(candidate);
        return true;
    };
    if (try_device_composed_template()) return true;

    out.per_program.resize(n_prog);
    out.is_device_geometry.assign(n_prog, 0);
    const bool resolve_device_mask =
        view.has_user_mask && view.flattened_masks.empty();
    // The fold length is the one RS quantity the host may never have seen, so
    // a program that is otherwise ORDINARY -- host geometry, wire-composed --
    // still has one port that must be read off the device. Resolve just that
    // port for such a program, exactly as a device mask is resolved for an
    // otherwise host-geometry attention pass.
    const bool resolve_device_fold_len = std::any_of(
        view.rs_slot_flags.data(),
        view.rs_slot_flags.data() + view.rs_slot_flags.size(),
        [](std::uint8_t f) {
            return (f & PIE_RS_FLAG_FOLD_LEN_DEVICE) != 0;
        });
    bool resolved_mask = false;
    std::vector<detail::PortCellCache> cached_cells(n_prog);
    // Pull host-writer rings on the descriptor stream: the readback pack
    // below is ordered behind these copies and its `read` synchronizes the
    // stream before any host use, so the pull itself never blocks. The bool
    // staging rides the launch state (which outlives the copies); the rare
    // probe call without a staged launch keeps a blocking local pull.
    cudaStream_t descriptor_stream =
        staged == nullptr ? nullptr : staged->stream;
    // Bind-time channel work (ring metadata, seeded-cell full bits, seed
    // payloads) rides the initialization stream and is only awaited in
    // `begin_enqueue` (RV-28). Prepare-time descriptor resolution reads those
    // very cells EARLIER than that, so it has to take the same edge or a
    // first fire sees its own seeds as "not yet produced".
    s.channels.order_after_initialization(descriptor_stream);
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
        const bool host_class =
            it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST;
        const bool mask_only = host_class && resolve_device_mask;
        const bool fold_len_only =
            host_class && !mask_only && resolve_device_fold_len;
        if (host_class && !mask_only && !fold_len_only) {
            continue;
        }
        const std::unordered_set<std::uint32_t>* pending_slots =
            staged == nullptr
                ? nullptr
                : &staged->lanes[p]->prologue_put_slots;
        for (const PortBinding& binding : trace->ports) {
            if (binding.is_const) continue;
            if (mask_only && binding.port != kPortAttnMask) continue;
            if (fold_len_only && binding.port != kPortRsFoldLen) continue;
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
            // FramePrepare-time read: cursor positions from the wave's
            // tickets (live mirrors advance only at enqueue); the rare
            // staged-less probe keeps the live read.
            const PreparedCursor cursor = staged != nullptr
                ? lane_ticket_window(
                      *staged->lanes[p], slot, s.channels)
                : PreparedCursor{
                      s.channels.host_head(slot),
                      s.channels.host_tail(slot),
                  };
            DeviceChannelRegistry& registry = *channel_view.registry();
            port_copies.push_back(PortCopy{
                .program = p,
                .slot = slot,
                .source =
                    static_cast<std::uint8_t*>(registry.cell_base(slot)) +
                    static_cast<std::size_t>(
                        pending ? cursor.tail_index : cursor.head_index) *
                        registry.cell_bytes(slot),
                .ready_source = pending
                    ? nullptr
                    : channel_view.d_full() +
                          static_cast<std::size_t>(slot) * kMaxRing +
                          cursor.head_index,
                .payload_offset = payload_offset,
                .ready_offset = ready_offset,
            });
        }
        // FramePrepare runs BEFORE this wave's `begin_enqueue`, so the
        // commit word cannot describe this wave. It only carries a verdict
        // once some wave's pull-validate has seeded it; before that it is
        // uninitialized (or a pooled leftover) and must not gate anything.
        if (staged != nullptr && staged->lanes[p]->snapshot->ever_validated) {
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
                !resolve_device_mask && !resolve_device_fold_len) {
                continue;
            }
            if (snapshot_offsets[p] ==
                std::numeric_limits<std::size_t>::max()) {
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
        const bool host_class =
            it->second.geometry_class == PIE_GEOMETRY_CLASS_HOST;
        const bool mask_only = host_class && resolve_device_mask;
        const bool fold_len_only =
            host_class && !mask_only && resolve_device_fold_len;
        if (host_class && !mask_only && !fold_len_only) {
            continue;
        }

        const std::unordered_set<std::uint32_t>* pending_slots = nullptr;
        if (staged != nullptr) {
            const StagedLane& lane = *staged->lanes[p];
            if (lane.snapshot->ever_validated && ready[p] == 0) {
                throw RetryableLaunchError(
                    describe_uncommitted_lane(iid, it->second, lane));
            }
            pending_slots = &lane.prologue_put_slots;
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
        if (fold_len_only) {
            // ONE port, and nothing else: this program's geometry is host
            // composed and correct already. `is_device_geometry` stays 0 so
            // composition keeps taking the wire arrays for everything except
            // the fold length, which it substitutes per flagged row.
            if (!resolve_rs_fold_len(
                    *trace, it->second.instance->view(), fg, err,
                    pending_slots, &cached_cells[p])) {
                return false;
            }
            if (fg.has_rs_fold_len) resolved_mask = true;
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

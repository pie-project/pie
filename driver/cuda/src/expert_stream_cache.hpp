#pragma once

// SSD expert streaming for routed MoE experts.
//
// When `[model].stream_routed_experts` is on, the Rust weight-loader compiler
// excludes routed-expert tensors from the resident schedule and emits a
// deferred `StreamPlan`: one reusable ExtentWrite template (slot-relative
// destinations) plus per-(layer, expert) source bindings. At forward time:
//
//   * `streamed_expert_table_from_program` materializes that plan into a
//     host-side extent table while the compiled program view is still live.
//   * `ExpertStreamCache` owns one bounded GPU slab plus per-shard O_RDONLY
//     fds. On a miss, `ensure_resident` executes the deferred ExtentWrites
//     (`pread` → pinned staging → async H2D) with plain LRU eviction and
//     returns per-section device pointers.
//
// Section count and layout come from the stream plan (arch-specific naming
// lives in the weight loader / model forward, not here).
//
// Scope (v1): on-demand only — no prefetch, no popularity preload, tp=1.
// The table/slot bookkeeping (`ExpertSlotIndex`) is host-only so it can be
// unit-tested without a GPU.

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../weight_loader/include/weight_loader.h"

namespace pie_cuda_driver {

struct ExpertSectionExtent {
    std::uint32_t shard = 0;        // index into StreamedExpertTable::shard_paths
    std::uint64_t file_offset = 0;  // absolute byte offset within the shard
};

struct StreamedExpertExtents {
    // One entry per section of the stream template (order matches the plan).
    std::vector<ExpertSectionExtent> sections;
};

// Runtime view of the compiler's deferred stream plan: per-(layer, expert)
// source bindings plus the slot layout the template ExtentWrites target.
struct StreamedExpertTable {
    int num_layers = 0;
    int num_experts = 0;
    int sections_per_expert = 0;
    std::vector<std::uint64_t> section_bytes;
    std::vector<std::uint64_t> section_offsets;
    std::uint64_t slot_bytes = 0;  // 0 = cache computes from section_bytes
    std::vector<std::filesystem::path> shard_paths;
    std::vector<StreamedExpertExtents> extents;  // [num_layers * num_experts]

    bool empty() const noexcept { return extents.empty(); }
    std::uint64_t payload_bytes_per_expert() const noexcept;
    std::uint64_t total_payload_bytes() const noexcept;
    const StreamedExpertExtents& at(int layer, int expert) const;
};

// Build the runtime extent table from the storage program's deferred stream
// plan. Throws if the plan is empty or malformed.
StreamedExpertTable streamed_expert_table_from_program(
    const pie_weight_loader::PieLoaderStorageProgramView& program);

// Host-only LRU slot bookkeeping for the expert slab: a flat
// [num_layers × num_experts] → slot map plus per-slot back-references and
// ages. Free slots have age 0 and are therefore picked before any touched
// slot; ties broken by lowest slot id. Slots pinned by the current
// `ensure_resident` batch are never chosen as victims.
class ExpertSlotIndex {
public:
    ExpertSlotIndex() = default;
    ExpertSlotIndex(int num_layers, int num_experts, int num_slots);

    int num_slots() const noexcept { return static_cast<int>(slots_.size()); }

    // Slot currently holding (layer, expert), or -1.
    int find(int layer, int expert) const;

    // LRU-touch a slot and pin it for the current batch.
    void touch_and_pin(int slot);

    // Map (layer, expert) to a slot: a free slot if any, else the LRU
    // unpinned slot (evicting its previous mapping). Touches + pins the
    // returned slot. Throws if every slot is pinned.
    struct Acquired {
        int slot = -1;
        bool evicted = false;
    };
    Acquired acquire(int layer, int expert);

    // Release the pins taken by the current batch.
    void unpin_all();

    std::uint64_t evictions() const noexcept { return evictions_; }

private:
    std::size_t key(int layer, int expert) const;

    int num_layers_ = 0;
    int num_experts_ = 0;
    std::vector<std::int32_t> slot_of_;  // [num_layers * num_experts], -1 = absent

    struct Slot {
        std::int32_t layer = -1;
        std::int32_t expert = -1;
        std::uint64_t age = 0;  // 0 = never used (free wins LRU)
        bool pinned = false;
    };
    std::vector<Slot> slots_;
    std::uint64_t tick_ = 0;
    std::uint64_t evictions_ = 0;
};

// Device pointers for one resident expert, in stream-plan section order.
struct ExpertSectionPointers {
    std::vector<const void*> section;

    int num_sections() const noexcept {
        return static_cast<int>(section.size());
    }
    const std::uint8_t* at(int i) const {
        return static_cast<const std::uint8_t*>(section.at(static_cast<std::size_t>(i)));
    }
};

// Bounded GPU cache of streamed expert weights. Single-threaded (the one
// forward thread); `ensure_resident` blocks until every requested expert
// is resident and its uploads have landed.
class ExpertStreamCache {
public:
    // `budget_bytes` bounds the slab; the slot count is clamped to
    // [1, num_layers × num_experts]. Throws if the budget cannot fit one
    // slot, a shard cannot be opened, or CUDA allocation fails.
    ExpertStreamCache(StreamedExpertTable table,
                      std::uint64_t budget_bytes,
                      bool verbose);
    ~ExpertStreamCache();

    ExpertStreamCache(const ExpertStreamCache&) = delete;
    ExpertStreamCache& operator=(const ExpertStreamCache&) = delete;

    int num_slots() const noexcept { return index_.num_slots(); }
    int sections_per_expert() const noexcept {
        return table_.sections_per_expert;
    }
    std::uint64_t slot_stride_bytes() const noexcept { return slot_stride_; }
    std::uint64_t slab_bytes() const noexcept {
        return slot_stride_ * static_cast<std::uint64_t>(num_slots());
    }
    const StreamedExpertTable& table() const noexcept { return table_; }

    // Make every expert in `experts` (all from `layer`) resident and return
    // its device section pointers in `out` (resized to `experts.size()`,
    // same order). At most `num_slots()` experts per call. Misses first
    // synchronize `compute_stream` so no in-flight kernel can still be
    // reading a victim slot, then execute the deferred stream-template
    // ExtentWrites (`pread` → pinned staging → H2D) on a dedicated upload
    // stream (double-buffered) and block until done.
    void ensure_resident(int layer,
                         std::span<const int> experts,
                         cudaStream_t compute_stream,
                         std::vector<ExpertSectionPointers>& out);

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t evictions = 0;
        std::uint64_t bytes_read = 0;
        double pread_ms = 0.0;
        double upload_wait_ms = 0.0;
    };
    Stats stats() const;
    void log_stats(const char* tag) const;

private:
    void* slot_base(int slot) const noexcept {
        return static_cast<std::uint8_t*>(slab_) +
               static_cast<std::uint64_t>(slot) * slot_stride_;
    }
    ExpertSectionPointers slot_pointers(int slot) const;
    // Execute the deferred stream-template ExtentWrites for one expert into
    // pinned staging buffer `buf` (section payloads at slot-relative offsets).
    void execute_stream_template(int layer, int expert, std::uint8_t* buf);

    StreamedExpertTable table_;
    ExpertSlotIndex index_;
    std::vector<std::uint64_t> section_offsets_;
    std::uint64_t slot_stride_ = 0;

    void* slab_ = nullptr;
    std::vector<int> shard_fds_;

    // Double-buffered pinned staging + dedicated upload stream.
    std::vector<std::uint8_t*> staging_;
    std::vector<cudaEvent_t> staging_done_;
    cudaStream_t upload_stream_ = nullptr;

    Stats stats_;
    bool verbose_ = false;
};

}  // namespace pie_cuda_driver

#pragma once

// A bounded slab of group instances, paged in on demand.
//
// It holds *several* groups, not one, and that is forced by the contract
// vocabulary rather than chosen: a name template carries one `{}`, so a
// checkpoint that names its experts `layers.<L>.experts.<E>.w1` is L groups of
// arity E, not one group of arity L*E. The slab has to span them anyway --
// budgeting per layer would size every layer for its worst step -- so the key
// here is (group, instance) flattened, and every group must agree on the slot
// stride, which for a real architecture they do because every layer's experts
// have the same shape.
//
// The loader compiles a group to one plan plus `arity` source bindings and
// says nothing about residency, because running the plan `arity` times into
// `arity` destinations and running it on demand into a few slots are the same
// program. This is the second reading. `GroupSlotIndex` decides which instance
// goes where; this moves the bytes and hands back the tensors.
//
// Two facts from the loader's uniformity proof do all the work here:
//
//   * every instance's plan has the same `memory.persistent_bytes`, so one
//     slot stride serves all of them, and the slab is just `slots × stride`;
//   * every instance's buffers land at the same offsets within that arena, so
//     a slot's tensor pointers are fixed at slot creation and a page-in
//     changes only the bytes behind them. A caller that captured a pointer
//     while its instance was resident is safe for exactly as long as the pin
//     lasts, and no longer -- which is what `end_batch` marks.
//
// Because a slot's plan is a whole `LoadPlanView`, the executor that runs the
// resident load runs it verbatim: transforms, tile maps and casts all work on
// the page-in path, and nothing here needs to know what the group contains.
//
// Single-threaded, like the forward pass that calls it. `ensure_resident`
// blocks until the instance is there.

#include <chrono>
#include <ostream>
#include <iostream>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Unconditionally CUDA, unlike the executor it wraps: a slab is device memory
// and there is no host-side reading of this component to preserve. The part
// that is worth testing without a device is `GroupSlotIndex`, which is why it
// is a separate header.
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/checkpoint_source.hpp"

#include "loader/group_slot_index.hpp"
#include "loader/load_plan_executor.hpp"
#include "loader/loader_config.hpp"
#include "loader/weight_copy_engine.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

class GroupStreamCache {
public:
    /// `budget_bytes` bounds the slab; the slot count is clamped to
    /// `[1, arity]`, so a budget large enough for the whole group degenerates
    /// to residency with no page-ins after the first sweep. Throws if the
    /// budget cannot hold one slot.
    GroupStreamCache(
        pie_loader::CheckpointSource& loader,
        pie_loader::PieLoaderGroupSlice groups,
        std::uint64_t budget_bytes,
        bool verbose = false)
        : loader_(loader), groups_(groups), verbose_(verbose)
    {
        if (groups.len == 0) {
            throw std::runtime_error(
                "group stream cache: nothing to page (no groups)");
        }
        std::uint64_t total_instances = 0;
        for (std::size_t g = 0; g < groups.len; ++g) {
            const auto& group = groups.ptr[g];
            const std::string name = pie_loader::bytes_to_string(group.name);
            if (group.plan == nullptr) {
                throw std::runtime_error(
                    "group stream cache: group \"" + name + "\" has no plan");
            }
            const std::uint64_t bytes = group.plan->memory.persistent_bytes;
            if (bytes == 0) {
                throw std::runtime_error(
                    "group stream cache: group \"" + name + "\" occupies no "
                    "persistent memory, so there is nothing to page");
            }
            // One stride for the whole slab, so a slot can hold any instance
            // of any group. Every layer's experts have the same shape, so a
            // disagreement here means the groups were not meant to share a
            // cache and sizing one slab for them would be a guess.
            if (slot_bytes_ == 0) {
                slot_bytes_ = bytes;
            } else if (bytes != slot_bytes_) {
                throw std::runtime_error(
                    "group stream cache: group \"" + name + "\" wants " +
                    std::to_string(bytes) + " bytes per instance but \"" +
                    pie_loader::bytes_to_string(groups.ptr[0].name) +
                    "\" wants " + std::to_string(slot_bytes_) +
                    ", so they cannot share a slab");
            }
            group_base_.push_back(total_instances);
            total_instances += group.arity;
        }
        if (budget_bytes < slot_bytes_) {
            throw std::runtime_error(
                "group stream cache: one instance needs " +
                std::to_string(slot_bytes_) + " bytes but the budget is " +
                std::to_string(budget_bytes));
        }
        std::uint64_t slots = budget_bytes / slot_bytes_;
        if (slots > total_instances) slots = total_instances;
        index_ = GroupSlotIndex(
            static_cast<std::uint32_t>(total_instances),
            static_cast<std::uint32_t>(slots));
        slot_stores_.resize(slots);
        slot_filled_.assign(slots, false);
        allocate_slab();
    }

    ~GroupStreamCache() {
        // Also on a bare env var, not only under the boot config's verbose:
        // measuring paging is a thing one does to an otherwise ordinary run,
        // and the rest of verbose is a wall of load-time noise.
        if (verbose_ || loader_config::env_truthy("PIE_CUDA_STREAM_STATS")) {
            report(std::cerr);
        }
        free_slab();
    }

    /// What paging actually cost, in the terms the next decision needs.
    ///
    /// Hit rate says whether the slab is big enough; ns/miss and MiB/s say
    /// whether the source is fast enough. They point at different fixes -- a
    /// larger slab versus a faster tier or a prefetch -- and a run that is
    /// slow for the second reason will not improve by any amount of the first.
    void report(std::ostream& out) const {
        const Stats s = stats();
        const std::uint64_t accesses = s.hits + s.misses;
        if (accesses == 0) return;
        // Warm-up held out of both the rate and the per-miss cost.
        const std::uint64_t steady_ns =
            s.page_in_ns > s.first_page_in_ns ? s.page_in_ns - s.first_page_in_ns : 0;
        const std::uint64_t steady_misses = s.misses > 1 ? s.misses - 1 : 0;
        const double page_in_ms = static_cast<double>(steady_ns) / 1e6;
        const double secs = static_cast<double>(steady_ns) / 1e9;
        out << "[pie-driver-cuda] group stream cache: " << accesses
            << " accesses, " << (100.0 * static_cast<double>(s.hits) /
                                 static_cast<double>(accesses))
            << "% hit, " << s.misses << " page-ins ("
            << (static_cast<double>(s.first_page_in_ns) / 1e6)
            << " ms of warm-up held out), "
            << (steady_misses == 0
                    ? 0.0
                    : static_cast<double>(steady_ns) / 1e3 /
                          static_cast<double>(steady_misses))
            << " us each, of "
            << (s.misses == 0 ? 0.0
                              : static_cast<double>(s.bytes_paged_in) /
                                    static_cast<double>(s.misses) / 1048576.0)
            << " MiB in " << page_in_ms << " ms ("
            << (secs == 0.0 ? 0.0
                            : static_cast<double>(s.bytes_paged_in) / 1048576.0 / secs)
            << " MiB/s), " << (static_cast<double>(s.evict_wait_ns) / 1e6)
            << " ms waiting to evict; of the page-in, alloc " << s.alloc_ms
            << " ms, transfer " << s.transfer_ms << " ms, transform "
            << s.transform_ms << " ms\n";
    }

    GroupStreamCache(const GroupStreamCache&) = delete;
    GroupStreamCache& operator=(const GroupStreamCache&) = delete;

    std::size_t num_groups() const noexcept { return groups_.len; }
    /// The group named `name`, or `kNoGroup`. A bind path holds names, not
    /// indices, and resolving once at bind keeps the forward pass indexing.
    static constexpr std::size_t kNoGroup = static_cast<std::size_t>(-1);
    std::size_t find_group(std::string_view name) const noexcept {
        for (std::size_t g = 0; g < groups_.len; ++g) {
            const auto& group = groups_.ptr[g];
            if (std::string_view(
                    reinterpret_cast<const char*>(group.name.ptr),
                    group.name.len) == name) {
                return g;
            }
        }
        return kNoGroup;
    }
    std::uint32_t arity(std::size_t group) const {
        if (group >= groups_.len) {
            throw std::out_of_range("group stream cache: no such group");
        }
        return groups_.ptr[group].arity;
    }
    /// Every instance of every group, which is what the slab is sized against.
    std::uint32_t total_instances() const noexcept { return index_.arity(); }
    std::uint32_t num_slots() const noexcept { return index_.num_slots(); }
    std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    std::uint64_t slab_bytes() const noexcept {
        return slot_bytes_ * num_slots();
    }
    /// True when the slab holds the whole group, so no page-in can ever miss
    /// after the first sweep. The caller may use this to keep CUDA graph
    /// capture on, since nothing will call into the host mid-forward.
    bool fully_resident() const noexcept {
        return num_slots() == total_instances();
    }

    /// Make `instance` resident and return its tensors, keyed by the runtime
    /// names its plan finalizes.
    ///
    /// A miss synchronizes `compute_stream` before it writes, because the
    /// victim slot may still be under a kernel that was launched while its old
    /// instance was pinned. The instance stays pinned -- and its pointers stay
    /// valid -- until `end_batch`.
    const WeightStore& ensure_resident(
        std::size_t group,
        std::uint32_t instance,
        cudaStream_t compute_stream)
    {
        const std::uint32_t key = flatten(group, instance);
        const auto found = index_.find(key);
        if (found != GroupSlotIndex::kAbsent) {
            const auto slot = static_cast<std::uint32_t>(found);
            index_.touch_and_pin(slot);
            ++stats_.hits;
            return slot_stores_[slot];
        }

        const auto acquired = index_.acquire(key);
        ++stats_.misses;
        if (acquired.evicted) {
            const auto started = Clock::now();
            sync_stream(compute_stream);
            stats_.evict_wait_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - started).count());
        }
        fill(acquired.slot, group, instance);
        return slot_stores_[acquired.slot];
    }

    /// End the batch: every slot becomes evictable again. Pointers handed out
    /// since the last call must not be used past this point.
    void end_batch() noexcept { index_.unpin_all(); }

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t bytes_paged_in = 0;
        /// Wall time inside `fill`, which is the whole of a miss: reading the
        /// checkpoint, any transform, and the copy. Blocking, so it is time
        /// the forward pass did not spend computing.
        std::uint64_t page_in_ns = 0;
        /// Wall time spent waiting on the compute stream before overwriting a
        /// slot. Separated because it is the cost of the slab being too small
        /// rather than of the bytes moving, and the two want different fixes.
        std::uint64_t evict_wait_ns = 0;
        /// The page-in split the way a fix would be: allocating the plan's
        /// buffers, moving the bytes, running the transform. A miss of a few
        /// megabytes is not bandwidth-bound, so which of these dominates is
        /// the whole question.
        double alloc_ms = 0;
        double transfer_ms = 0;
        double transform_ms = 0;
        /// The first page-in, held out of the rest. It pays for the copy
        /// engine's streams and staging pool, and at a few hundred
        /// microseconds against a steady-state miss of a few tens it would
        /// otherwise dominate the average and hide what a miss really costs.
        std::uint64_t first_page_in_ns = 0;
    };

    Stats stats() const noexcept {
        Stats out = stats_;
        return out;
    }
    std::uint64_t evictions() const noexcept { return index_.evictions(); }

private:
    std::uint8_t* slot_base(std::uint32_t slot) const noexcept {
        return slab_ + static_cast<std::uint64_t>(slot) * slot_bytes_;
    }

    /// Run instance `instance`'s plan into `slot`.
    ///
    /// The executor is the one the resident load uses, told two things: the
    /// destination arena is this slot, and the sources are this instance's.
    /// Its output goes to a scratch store; on a slot's first fill that store
    /// becomes the slot's, and afterwards it is checked against it and thrown
    /// away, because the layout is what the loader proved constant, not the
    /// bytes.
    std::uint32_t flatten(std::size_t group, std::uint32_t instance) const
    {
        if (group >= groups_.len) {
            throw std::out_of_range(
                "group stream cache: group " + std::to_string(group) +
                " outside " + std::to_string(groups_.len));
        }
        return static_cast<std::uint32_t>(group_base_[group]) + instance;
    }

    void fill(std::uint32_t slot, std::size_t group_index, std::uint32_t instance)
    {
        const auto& group = groups_.ptr[group_index];
        WeightStore scratch;
        WeightStoreBuilder builder(scratch);
        // One copy engine for the life of the cache, not one per page-in: it
        // creates copy streams and a pinned pool on first use and drops them
        // with itself, and at a few megabytes a miss that setup is most of
        // the cost.
        LoadPlanExecutor executor(loader_, builder, {}, &copy_engine_);

        LoadPlanExecution how;
        how.persistent_arena = slot_base(slot);
        how.persistent_arena_bytes = slot_bytes_;
        const std::size_t per_instance = group.bindings_per_instance;
        if (per_instance != 0) {
            const std::size_t start =
                static_cast<std::size_t>(instance) * per_instance;
            if (start + per_instance > group.bindings.len) {
                throw std::runtime_error(
                    "group stream cache: group \"" +
                    pie_loader::bytes_to_string(group.name) + "\" instance " +
                    std::to_string(instance) + " has no bindings");
            }
            how.source_bindings = group.bindings.ptr + start;
            how.source_binding_count = per_instance;
        }

        const auto started = Clock::now();
        const auto stats = executor.execute(*group.plan, how);
        stats_.bytes_paged_in += stats.h2d_copy_bytes;
        const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - started).count());
        if (stats_.misses == 1) stats_.first_page_in_ns = elapsed_ns;
        stats_.alloc_ms += stats.phase_alloc_ms;
        stats_.transfer_ms += stats.phase_transfer_ms;
        stats_.transform_ms += stats.phase_transform_ms;
        stats_.page_in_ns += elapsed_ns;

        if (!slot_filled_[slot]) {
            slot_stores_[slot] = std::move(scratch);
            slot_filled_[slot] = true;
        } else {
            check_layout_held(slot, scratch, flatten(group_index, instance));
        }
    }

    /// The invariant the whole design rests on: a page-in replaces bytes, not
    /// addresses. If this ever fires, a caller holding a pointer from an
    /// earlier instance is reading the wrong memory, and that is the one bug
    /// this component could produce silently.
    void check_layout_held(
        std::uint32_t slot,
        const WeightStore& fresh,
        std::uint32_t instance) const
    {
        const auto& held = slot_stores_[slot];
        for (const auto& [name, record] : fresh) {
            const auto it = held.find(name);
            if (it == held.end() || it->second.data() != record.data()) {
                throw std::runtime_error(
                    "group stream cache: instance " +
                    std::to_string(instance) + " laid \"" + name +
                    "\" out differently from the instance already in slot " +
                    std::to_string(slot) +
                    ", so the group's instances are not interchangeable");
            }
        }
    }

    void allocate_slab()
    {
        const cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&slab_), slab_bytes());
        if (err != cudaSuccess) {
            slab_ = nullptr;
            throw std::runtime_error(
                "group stream cache: could not allocate " +
                std::to_string(slab_bytes()) + " bytes: " +
                cudaGetErrorString(err));
        }
    }

    void free_slab() noexcept
    {
        if (slab_ != nullptr) {
            cudaFree(slab_);
            slab_ = nullptr;
        }
    }

    static void sync_stream(cudaStream_t stream)
    {
        if (stream != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    using Clock = std::chrono::steady_clock;

    pie_loader::CheckpointSource& loader_;
    WeightCopyEngine copy_engine_{loader_};
    pie_loader::PieLoaderGroupSlice groups_;
    /// Where each group's instances start in the one flat key space.
    std::vector<std::uint64_t> group_base_;
    bool verbose_ = false;

    std::uint64_t slot_bytes_ = 0;
    std::uint8_t* slab_ = nullptr;

    GroupSlotIndex index_;
    std::vector<WeightStore> slot_stores_;
    std::vector<bool> slot_filled_;
    Stats stats_;
};

}  // namespace pie_cuda_driver

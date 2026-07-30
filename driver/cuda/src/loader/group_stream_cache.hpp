#pragma once

// A bounded slab of group instances, paged in on demand.
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

#include <cstdint>
#include <string>
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
        const pie_loader::PieLoaderGroupView& group,
        std::uint64_t budget_bytes,
        bool verbose = false)
        : loader_(loader),
          group_(group),
          name_(pie_loader::bytes_to_string(group.name)),
          verbose_(verbose)
    {
        if (group.plan == nullptr) {
            throw std::runtime_error(
                "group stream cache: group \"" + name_ + "\" has no plan");
        }
        slot_bytes_ = group.plan->memory.persistent_bytes;
        if (slot_bytes_ == 0) {
            throw std::runtime_error(
                "group stream cache: group \"" + name_ +
                "\" occupies no persistent memory, so there is nothing to page");
        }
        if (budget_bytes < slot_bytes_) {
            throw std::runtime_error(
                "group stream cache: group \"" + name_ + "\" needs " +
                std::to_string(slot_bytes_) + " bytes per instance but the "
                "budget is " + std::to_string(budget_bytes));
        }
        std::uint64_t slots = budget_bytes / slot_bytes_;
        if (slots > group.arity) slots = group.arity;
        index_ = GroupSlotIndex(group.arity, static_cast<std::uint32_t>(slots));
        slot_stores_.resize(slots);
        slot_filled_.assign(slots, false);
        allocate_slab();
    }

    ~GroupStreamCache() { free_slab(); }

    GroupStreamCache(const GroupStreamCache&) = delete;
    GroupStreamCache& operator=(const GroupStreamCache&) = delete;

    const std::string& name() const noexcept { return name_; }
    std::uint32_t arity() const noexcept { return group_.arity; }
    std::uint32_t num_slots() const noexcept { return index_.num_slots(); }
    std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    std::uint64_t slab_bytes() const noexcept {
        return slot_bytes_ * num_slots();
    }
    /// True when the slab holds the whole group, so no page-in can ever miss
    /// after the first sweep. The caller may use this to keep CUDA graph
    /// capture on, since nothing will call into the host mid-forward.
    bool fully_resident() const noexcept { return num_slots() == group_.arity; }

    /// Make `instance` resident and return its tensors, keyed by the runtime
    /// names its plan finalizes.
    ///
    /// A miss synchronizes `compute_stream` before it writes, because the
    /// victim slot may still be under a kernel that was launched while its old
    /// instance was pinned. The instance stays pinned -- and its pointers stay
    /// valid -- until `end_batch`.
    const WeightStore& ensure_resident(
        std::uint32_t instance,
        cudaStream_t compute_stream)
    {
        const auto found = index_.find(instance);
        if (found != GroupSlotIndex::kAbsent) {
            const auto slot = static_cast<std::uint32_t>(found);
            index_.touch_and_pin(slot);
            ++stats_.hits;
            return slot_stores_[slot];
        }

        const auto acquired = index_.acquire(instance);
        ++stats_.misses;
        if (acquired.evicted) {
            sync_stream(compute_stream);
        }
        fill(acquired.slot, instance);
        return slot_stores_[acquired.slot];
    }

    /// End the batch: every slot becomes evictable again. Pointers handed out
    /// since the last call must not be used past this point.
    void end_batch() noexcept { index_.unpin_all(); }

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t bytes_paged_in = 0;
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
    void fill(std::uint32_t slot, std::uint32_t instance)
    {
        WeightStore scratch;
        WeightStoreBuilder builder(scratch);
        LoadPlanExecutor executor(loader_, builder, {});

        LoadPlanExecution how;
        how.persistent_arena = slot_base(slot);
        how.persistent_arena_bytes = slot_bytes_;
        const std::size_t per_instance = group_.bindings_per_instance;
        if (per_instance != 0) {
            const std::size_t start =
                static_cast<std::size_t>(instance) * per_instance;
            if (start + per_instance > group_.bindings.len) {
                throw std::runtime_error(
                    "group stream cache: group \"" + name_ + "\" instance " +
                    std::to_string(instance) + " has no bindings");
            }
            how.source_bindings = group_.bindings.ptr + start;
            how.source_binding_count = per_instance;
        }

        const auto stats = executor.execute(*group_.plan, how);
        stats_.bytes_paged_in += stats.h2d_copy_bytes;

        if (!slot_filled_[slot]) {
            slot_stores_[slot] = std::move(scratch);
            slot_filled_[slot] = true;
        } else {
            check_layout_held(slot, scratch, instance);
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
                    "group stream cache: group \"" + name_ + "\" instance " +
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
                std::to_string(slab_bytes()) + " bytes for group \"" + name_ +
                "\": " + cudaGetErrorString(err));
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

    pie_loader::CheckpointSource& loader_;
    const pie_loader::PieLoaderGroupView& group_;
    std::string name_;
    bool verbose_ = false;

    std::uint64_t slot_bytes_ = 0;
    std::uint8_t* slab_ = nullptr;

    GroupSlotIndex index_;
    std::vector<WeightStore> slot_stores_;
    std::vector<bool> slot_filled_;
    Stats stats_;
};

}  // namespace pie_cuda_driver

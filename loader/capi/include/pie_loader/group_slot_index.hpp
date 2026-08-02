#pragma once

// Which instance of a group is in which slot.
//
// A `PieLoaderGroupView` is `arity` interchangeable instances of one plan. The
// loader stops there on purpose: making all `arity` of them resident and paging
// a few of them through a bounded slab are the same program, and choosing
// between those is policy, which lives here.
//
// This is the bookkeeping half of that policy and nothing else. It holds no
// device memory, opens no files, and names no API, so the eviction rule -- the
// part that decides whether streaming thrashes -- is testable on a machine
// with no GPU. The half that moves bytes is each backend's own: CUDA pages
// into a device slab, Metal into a heap slot, and neither disagrees with the
// other about which instance was supposed to be there.
//
// It lives beside the plan rather than in a driver for that last reason. Two
// backends deciding residency by two eviction rules is two ways for the same
// checkpoint to thrash, and one of them would be found much later than the
// other.
//
// The rule: a free slot always beats a used one (free slots have age 0, and
// nothing else ever does), ties go to the lowest slot id, and among used slots
// the least recently touched loses. Slots pinned by the batch in flight are
// never victims, because a kernel may still be reading one; a batch that wants
// more instances than there are slots is a configuration error and says so
// rather than corrupting a slot out from under a running kernel.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_loader {

class GroupSlotIndex {
public:
    GroupSlotIndex() = default;

    /// `arity` is the group's instance count, `num_slots` how many fit the
    /// budget. `num_slots` may exceed `arity` -- the caller clamps if it cares
    /// -- but neither may be zero.
    GroupSlotIndex(std::uint32_t arity, std::uint32_t num_slots)
        : arity_(arity)
    {
        if (arity == 0 || num_slots == 0) {
            throw std::invalid_argument(
                "GroupSlotIndex: arity and slot count must be positive");
        }
        slot_of_.assign(arity, kAbsent);
        slots_.resize(num_slots);
    }

    std::uint32_t arity() const noexcept { return arity_; }
    std::uint32_t num_slots() const noexcept {
        return static_cast<std::uint32_t>(slots_.size());
    }

    static constexpr std::int32_t kAbsent = -1;

    /// The slot holding `instance`, or `kAbsent`.
    std::int32_t find(std::uint32_t instance) const {
        return slot_of_[checked(instance)];
    }

    /// Mark a slot as used by the batch in flight. A hit still has to say so,
    /// or a later miss in the same batch could evict what this one just found.
    void touch_and_pin(std::uint32_t slot) {
        auto& s = slots_.at(slot);
        s.age = ++tick_;
        s.pinned = true;
    }

    struct Acquired {
        std::uint32_t slot = 0;
        /// Whether the slot held a different instance, which the caller must
        /// treat as invalidating any pointer it handed out for that instance.
        bool evicted = false;
    };

    /// Give `instance` a slot, evicting the LRU unpinned one if no slot is
    /// free. The returned slot is touched and pinned. The caller still has to
    /// fill it: this says where the instance goes, not that it is there.
    Acquired acquire(std::uint32_t instance) {
        const std::size_t k = checked(instance);

        std::uint32_t victim = 0;
        bool found = false;
        std::uint64_t best_age = std::numeric_limits<std::uint64_t>::max();
        for (std::uint32_t i = 0; i < num_slots(); ++i) {
            const auto& s = slots_[i];
            if (s.pinned) continue;
            if (!found || s.age < best_age) {
                best_age = s.age;
                victim = i;
                found = true;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "GroupSlotIndex: all " + std::to_string(num_slots()) +
                " slots are pinned by the batch in flight, so it wants more "
                "instances at once than the cache holds");
        }

        auto& s = slots_[victim];
        Acquired out{.slot = victim, .evicted = s.instance != kAbsent};
        if (out.evicted) {
            slot_of_[static_cast<std::size_t>(s.instance)] = kAbsent;
            ++evictions_;
        }
        s.instance = static_cast<std::int64_t>(instance);
        s.age = ++tick_;
        s.pinned = true;
        slot_of_[k] = static_cast<std::int32_t>(victim);
        return out;
    }

    /// Give back the pin an `acquire` took, leaving the slot's contents and
    /// its recency alone.
    ///
    /// For a caller that acquires a slot to write it rather than to read it,
    /// and so has no batch to hold it against. `place_all` is the one that
    /// does: it walks every instance in turn, and pinning each as it went
    /// would run the index out of slots before it finished -- which is a
    /// failure, not a slowdown, since `acquire` has nothing left to evict.
    void release(std::uint32_t slot) noexcept {
        if (slot < slots_.size()) slots_[slot].pinned = false;
    }

    /// End the batch. Until this is called every slot the batch touched is
    /// off limits as a victim.
    void unpin_all() noexcept {
        for (auto& s : slots_) s.pinned = false;
    }

    std::uint64_t evictions() const noexcept { return evictions_; }

private:
    std::size_t checked(std::uint32_t instance) const {
        if (instance >= arity_) {
            throw std::out_of_range(
                "GroupSlotIndex: instance " + std::to_string(instance) +
                " outside arity " + std::to_string(arity_));
        }
        return instance;
    }

    struct Slot {
        std::int64_t instance = kAbsent;
        std::uint64_t age = 0;  // 0 = never filled, so free slots win the LRU
        bool pinned = false;
    };

    std::uint32_t arity_ = 0;
    std::vector<std::int32_t> slot_of_;
    std::vector<Slot> slots_;
    std::uint64_t tick_ = 0;
    std::uint64_t evictions_ = 0;
};

}  // namespace pie_loader

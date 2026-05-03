#include "slot_allocator.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver {

SlotAllocator::SlotAllocator(int max_slots) {
    reset(max_slots);
}

void SlotAllocator::reset(int max_slots) {
    if (max_slots < 1) {
        throw std::invalid_argument(
            "SlotAllocator::reset: max_slots must be >= 1, got " +
            std::to_string(max_slots));
    }
    max_slots_ = max_slots;
    by_ctx_.clear();
    by_slot_.assign(static_cast<std::size_t>(max_slots), 0u);
    lru_order_.clear();
    lru_order_.reserve(static_cast<std::size_t>(max_slots));
    in_use_this_fire_.assign(static_cast<std::size_t>(max_slots), false);
    free_slots_.clear();
    free_slots_.reserve(static_cast<std::size_t>(max_slots));
    // Push free slots in reverse so acquire pops slot 0 first — keeps
    // single-request workloads on slot 0, matching the legacy layout.
    for (int s = max_slots - 1; s >= 0; --s) {
        free_slots_.push_back(s);
    }
}

SlotAllocator::Acquired SlotAllocator::acquire(std::uint64_t ctx_id) {
    if (max_slots_ < 1) {
        throw std::runtime_error(
            "SlotAllocator::acquire: allocator not initialised (max_slots=0)");
    }

    auto it = by_ctx_.find(ctx_id);
    if (it != by_ctx_.end()) {
        const int slot = it->second;
        // Bump to back of LRU (most-recent).
        auto pos = std::find(lru_order_.begin(), lru_order_.end(), slot);
        if (pos != lru_order_.end()) {
            lru_order_.erase(pos);
        }
        lru_order_.push_back(slot);
        in_use_this_fire_[static_cast<std::size_t>(slot)] = true;
        return {slot, /*is_fresh=*/false};
    }

    // Need a fresh slot. Prefer free slots; else evict LRU.
    int slot;
    if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
    } else {
        // Evict the oldest non-in-use slot. Walking lru_order_ from the
        // front, the first slot whose flag is false is the victim.
        slot = -1;
        for (std::size_t i = 0; i < lru_order_.size(); ++i) {
            const int candidate = lru_order_[i];
            if (!in_use_this_fire_[static_cast<std::size_t>(candidate)]) {
                slot = candidate;
                lru_order_.erase(lru_order_.begin() +
                                 static_cast<std::ptrdiff_t>(i));
                break;
            }
        }
        if (slot < 0) {
            throw std::runtime_error(
                "SlotAllocator::acquire: every slot is in use this fire "
                "(R > max_slots=" + std::to_string(max_slots_) + ")");
        }
        // Drop the evicted ctx from the forward map.
        const std::uint64_t old_ctx = by_slot_[static_cast<std::size_t>(slot)];
        if (old_ctx != 0u) {
            by_ctx_.erase(old_ctx);
        }
    }

    by_ctx_[ctx_id] = slot;
    by_slot_[static_cast<std::size_t>(slot)] = ctx_id;
    lru_order_.push_back(slot);
    in_use_this_fire_[static_cast<std::size_t>(slot)] = true;
    return {slot, /*is_fresh=*/true};
}

void SlotAllocator::end_of_fire() {
    std::fill(in_use_this_fire_.begin(), in_use_this_fire_.end(), false);
}

void SlotAllocator::release(std::uint64_t ctx_id) {
    auto it = by_ctx_.find(ctx_id);
    if (it == by_ctx_.end()) return;
    const int slot = it->second;
    by_ctx_.erase(it);
    by_slot_[static_cast<std::size_t>(slot)] = 0u;
    auto pos = std::find(lru_order_.begin(), lru_order_.end(), slot);
    if (pos != lru_order_.end()) {
        lru_order_.erase(pos);
    }
    free_slots_.push_back(slot);
}

}  // namespace pie_cuda_driver

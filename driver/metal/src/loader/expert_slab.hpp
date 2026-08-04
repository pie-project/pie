#pragma once

/// A bounded set of expert slots, filled from the checkpoint on demand.
///
/// This is the mechanism that lets a model outgrow the machine, and it exists
/// because the obvious alternative does not work here. Mapping a bank and
/// asking for residency does not page it lazily: `requestResidency` wires every
/// page whether or not a kernel reads it, measured at 18.4 GB wired for a
/// streamed Qwen3-30B-A3B against 1.5 GB at rest. An Apple Silicon GPU has no
/// demand paging, and a kernel that touched a page the residency set had let go
/// would abort its command buffer rather than fault it back. So the bytes the
/// GPU can reach have to be a fixed, wired region whose CONTENTS change, and
/// changing them is an explicit copy the host makes.
///
/// What makes that copy cheap is the same fact the mapping work leaned on: a
/// converted checkpoint holds each expert bank as plain bytes in the file, so
/// paging an expert in is a `memcpy` from a mapping and not a transform. There
/// is no decode step here and no plan to run, which is why this takes a byte
/// range rather than a `LoadPlan`.
///
/// Two things are deliberately NOT here:
///
///   * The Metal allocation. The slab is a destination pointer and a stride,
///     so the eviction rule and the byte movement are both testable against
///     ordinary host memory with no device. `GroupSlotIndex` was split out of
///     the CUDA cache for the same reason and this reuses it rather than
///     restating it.
///   * Which expert a token wants. That is the router's answer, read back
///     between segments, and it belongs to the caller.
///
/// Single-threaded, like the forward pass that drives it.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pie_loader/group_slot_index.hpp>

namespace pie::metal {

/// One stack of interchangeable experts, as bytes in a mapped file.
///
/// `arity` bands of `band_bytes` each, back to back from `file_offset`. That
/// layout is not an assumption about checkpoints in general -- it is what the
/// destination buffer already is, an expert-major `[experts * rows, cols]`
/// tensor -- so band `e` is the range the kernel would have read at
/// `e * rows * cols` had the whole bank been resident.
struct ExpertBank {
    const std::uint8_t* file_base = nullptr;
    std::uint64_t file_offset = 0;
    std::uint64_t band_bytes = 0;
    std::uint32_t arity = 0;
    /// For diagnostics only; the slab never matches on it.
    std::string name;
};

class ExpertSlab {
  public:
    /// `slab` must hold `slots * band_bytes`, and every bank must agree on
    /// `band_bytes` -- one stride serves all of them, which is what lets slots
    /// be shared across layers instead of budgeted per layer.
    ///
    /// Throws rather than clamping when the budget cannot hold one band: a
    /// caller that asked for less than an expert has not configured a small
    /// cache, it has configured one that cannot run.
    ExpertSlab(std::vector<ExpertBank> banks, void* slab, std::uint32_t slots)
        : banks_(std::move(banks)), slab_(static_cast<std::uint8_t*>(slab)) {
        if (banks_.empty()) throw std::invalid_argument("ExpertSlab: no banks");
        if (slab_ == nullptr || slots == 0) {
            throw std::invalid_argument("ExpertSlab: the slab is empty");
        }
        band_bytes_ = banks_.front().band_bytes;
        std::uint64_t instances = 0;
        for (const ExpertBank& b : banks_) {
            if (b.band_bytes != band_bytes_) {
                throw std::invalid_argument(
                    "ExpertSlab: bank '" + b.name + "' bands " +
                    std::to_string(b.band_bytes) + " bytes against " +
                    std::to_string(band_bytes_) + " for the rest, so one slot "
                    "stride cannot serve them");
            }
            if (b.arity == 0 || b.file_base == nullptr) {
                throw std::invalid_argument("ExpertSlab: bank '" + b.name +
                                            "' has nothing to page");
            }
            instances += b.arity;
        }
        if (band_bytes_ == 0) throw std::invalid_argument("ExpertSlab: zero-byte bands");
        if (instances > 0xFFFFFFFFull) {
            throw std::invalid_argument("ExpertSlab: too many expert instances");
        }
        if (slots > instances) slots = static_cast<std::uint32_t>(instances);
        index_ = pie_loader::GroupSlotIndex(static_cast<std::uint32_t>(instances), slots);
    }

    std::uint32_t num_slots() const noexcept { return index_.num_slots(); }
    std::uint64_t band_bytes() const noexcept { return band_bytes_; }
    std::uint64_t slab_bytes() const noexcept { return std::uint64_t(num_slots()) * band_bytes_; }

    /// True when the slab can hold every expert, so nothing can ever be
    /// evicted. The caller that knows this can skip the routing readback
    /// entirely -- which is the whole cost of streaming once the slab is warm.
    bool fully_resident() const noexcept { return num_slots() == index_.arity(); }

    /// The slot holding `(bank, expert)`, copying it in if it is not there.
    ///
    /// The slot stays pinned -- and so does whatever the caller binds to it --
    /// until `end_batch`. A batch that wants more experts at once than the slab
    /// has slots throws rather than evicting a slot a kernel may still read.
    std::uint32_t ensure_resident(std::uint32_t bank, std::uint32_t expert) {
        const std::uint32_t key = flatten(bank, expert);
        const auto found = index_.find(key);
        if (found != pie_loader::GroupSlotIndex::kAbsent) {
            const auto slot = static_cast<std::uint32_t>(found);
            index_.touch_and_pin(slot);
            ++hits_;
            return slot;
        }
        const auto acquired = index_.acquire(key);
        ++misses_;
        fill(acquired.slot, bank, expert);
        return acquired.slot;
    }

    /// Every slot becomes evictable again. Pointers and slot indices handed out
    /// since the last call must not be used past this point.
    void end_batch() noexcept { index_.unpin_all(); }

    std::uint64_t hits() const noexcept { return hits_; }
    std::uint64_t misses() const noexcept { return misses_; }
    std::uint64_t bytes_paged_in() const noexcept { return misses_ * band_bytes_; }

    /// Where slot `s` lives, for binding and for tests.
    const std::uint8_t* slot_data(std::uint32_t slot) const {
        return slab_ + std::uint64_t(slot) * band_bytes_;
    }

  private:
    std::uint32_t flatten(std::uint32_t bank, std::uint32_t expert) const {
        if (bank >= banks_.size()) {
            throw std::out_of_range("ExpertSlab: bank " + std::to_string(bank) +
                                    " of " + std::to_string(banks_.size()));
        }
        std::uint32_t base = 0;
        for (std::uint32_t b = 0; b < bank; ++b) base += banks_[b].arity;
        if (expert >= banks_[bank].arity) {
            throw std::out_of_range(
                "ExpertSlab: expert " + std::to_string(expert) + " outside bank '" +
                banks_[bank].name + "' of " + std::to_string(banks_[bank].arity));
        }
        return base + expert;
    }

    void fill(std::uint32_t slot, std::uint32_t bank, std::uint32_t expert) {
        const ExpertBank& b = banks_[bank];
        std::memcpy(slab_ + std::uint64_t(slot) * band_bytes_,
                    b.file_base + b.file_offset + std::uint64_t(expert) * band_bytes_,
                    std::size_t(band_bytes_));
    }

    std::vector<ExpertBank> banks_;
    std::uint8_t* slab_ = nullptr;
    std::uint64_t band_bytes_ = 0;
    pie_loader::GroupSlotIndex index_;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
};

}  // namespace pie::metal

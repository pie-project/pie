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
/// is no decode step here and no plan to run, which is why this takes byte
/// ranges rather than a `LoadPlan`.
///
/// ## A slot is a set of parallel bands, not one band
///
/// An expert is not one tensor. Qwen3-MoE routes to `gate_proj`, `up_proj` and
/// `down_proj` together; gpt-oss adds scales beside each. They have DIFFERENT
/// band sizes, so they cannot share a stride -- but they must share a slot
/// NUMBER, because the kernels read one `expert_ids` buffer and every routed
/// projection in the layer indexes with it. A per-tensor cache would hand
/// `gate_proj` slot 3 and `down_proj` slot 1 for the same expert, and there is
/// no single number to write into `expert_ids` that means both.
///
/// So there is one slot index over (layer, expert) instances and one slab per
/// TENSOR KIND, all addressed by the same slot number. Occupying a slot moves
/// every tensor of that expert, which is also what makes the residency claim
/// true: a slot is resident or it is not, and no kernel can find half of one.
///
/// Two things are deliberately NOT here:
///
///   * The Metal allocation. The slabs are pointers and strides, so the
///     eviction rule and the byte movement are both testable against ordinary
///     host memory with no device. `GroupSlotIndex` was split out of the CUDA
///     cache for the same reason and this reuses it rather than restating it.
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

/// One routed tensor kind across every mixture layer.
///
/// `layer_base[l]` points at expert 0 of layer `l`, and expert `e` follows at
/// `e * band_bytes`. That layout is not an assumption about checkpoints in
/// general -- it is what the destination buffer already is, an expert-major
/// `[experts * rows, cols]` tensor -- so the band is the range the kernel would
/// have read at `e * rows * cols` had the whole bank been resident.
struct SlabTensor {
    /// The runtime name past the layer prefix, e.g. `mlp.experts.gate_proj`.
    /// Diagnostics only; nothing here matches on it.
    std::string suffix;
    std::uint64_t band_bytes = 0;
    std::vector<const std::uint8_t*> layer_base;
};

class ExpertSlab {
  public:
    /// `slab_base[t]` must hold `slots * tensors[t].band_bytes`.
    ///
    /// Throws rather than clamping when the budget cannot hold one slot: a
    /// caller that asked for less than an expert has not configured a small
    /// cache, it has configured one that cannot run.
    ExpertSlab(std::vector<SlabTensor> tensors, std::uint32_t experts_per_layer,
               std::vector<std::uint8_t*> slab_base, std::uint32_t slots)
        : tensors_(std::move(tensors)), slab_(std::move(slab_base)),
          experts_(experts_per_layer) {
        if (tensors_.empty()) throw std::invalid_argument("ExpertSlab: no routed tensors");
        if (slab_.size() != tensors_.size()) {
            throw std::invalid_argument("ExpertSlab: one slab per tensor kind, got " +
                                        std::to_string(slab_.size()) + " for " +
                                        std::to_string(tensors_.size()));
        }
        if (experts_ == 0 || slots == 0) {
            throw std::invalid_argument("ExpertSlab: an empty cache cannot run");
        }
        layers_ = static_cast<std::uint32_t>(tensors_.front().layer_base.size());
        if (layers_ == 0) throw std::invalid_argument("ExpertSlab: no mixture layers");
        for (std::size_t t = 0; t < tensors_.size(); ++t) {
            const SlabTensor& s = tensors_[t];
            // Not a shape check for its own sake: a tensor present on fewer
            // layers than the rest would silently make (layer, expert) mean a
            // different expert for it than for its siblings.
            if (s.layer_base.size() != layers_) {
                throw std::invalid_argument(
                    "ExpertSlab: '" + s.suffix + "' spans " +
                    std::to_string(s.layer_base.size()) + " layers against " +
                    std::to_string(layers_) + " for the rest");
            }
            if (s.band_bytes == 0) {
                throw std::invalid_argument("ExpertSlab: '" + s.suffix + "' has zero-byte bands");
            }
            for (const std::uint8_t* p : s.layer_base) {
                if (p == nullptr) {
                    throw std::invalid_argument("ExpertSlab: '" + s.suffix +
                                                "' has a layer with nothing to page");
                }
            }
            if (slab_[t] == nullptr) {
                throw std::invalid_argument("ExpertSlab: '" + s.suffix + "' has no slab");
            }
            slot_bytes_ += s.band_bytes;
        }
        const std::uint64_t instances = std::uint64_t(layers_) * experts_;
        if (slots > instances) slots = static_cast<std::uint32_t>(instances);
        index_ = pie_loader::GroupSlotIndex(static_cast<std::uint32_t>(instances), slots);
    }

    std::uint32_t num_slots() const noexcept { return index_.num_slots(); }
    std::uint32_t layers() const noexcept { return layers_; }
    std::uint32_t experts_per_layer() const noexcept { return experts_; }
    /// Every routed tensor of one expert, which is what one slot costs.
    std::uint64_t slot_bytes() const noexcept { return slot_bytes_; }
    std::uint64_t slab_bytes() const noexcept { return std::uint64_t(num_slots()) * slot_bytes_; }

    /// True when the slab can hold every expert of every layer, so nothing can
    /// ever be evicted. The caller that knows this can skip the routing
    /// readback entirely -- which is the whole cost of streaming once the slab
    /// is warm.
    bool fully_resident() const noexcept { return num_slots() == index_.arity(); }

    /// The slot holding layer `layer`'s expert `expert`, copying every one of
    /// its tensors in if it is not there.
    ///
    /// The slot stays pinned until `end_batch`. A batch that wants more experts
    /// at once than the slab has slots throws rather than evicting a slot a
    /// kernel may still read.
    std::uint32_t ensure_resident(std::uint32_t layer, std::uint32_t expert) {
        const std::uint32_t key = flatten(layer, expert);
        const auto found = index_.find(key);
        if (found != pie_loader::GroupSlotIndex::kAbsent) {
            const auto slot = static_cast<std::uint32_t>(found);
            index_.touch_and_pin(slot);
            ++hits_;
            return slot;
        }
        const auto acquired = index_.acquire(key);
        ++misses_;
        for (std::size_t t = 0; t < tensors_.size(); ++t) {
            const SlabTensor& s = tensors_[t];
            std::memcpy(slab_[t] + std::uint64_t(acquired.slot) * s.band_bytes,
                        s.layer_base[layer] + std::uint64_t(expert) * s.band_bytes,
                        std::size_t(s.band_bytes));
        }
        return acquired.slot;
    }

    /// Every slot becomes evictable again. Slot numbers handed out since the
    /// last call must not be used past this point.
    void end_batch() noexcept { index_.unpin_all(); }

    std::uint64_t hits() const noexcept { return hits_; }
    std::uint64_t misses() const noexcept { return misses_; }
    std::uint64_t bytes_paged_in() const noexcept { return misses_ * slot_bytes_; }

    /// Where tensor `t`'s band for slot `slot` lives, for binding and for tests.
    const std::uint8_t* slot_data(std::size_t t, std::uint32_t slot) const {
        return slab_[t] + std::uint64_t(slot) * tensors_[t].band_bytes;
    }
    const std::vector<SlabTensor>& tensors() const noexcept { return tensors_; }

  private:
    std::uint32_t flatten(std::uint32_t layer, std::uint32_t expert) const {
        if (layer >= layers_ || expert >= experts_) {
            throw std::out_of_range("ExpertSlab: (" + std::to_string(layer) + ", " +
                                    std::to_string(expert) + ") outside " +
                                    std::to_string(layers_) + " x " + std::to_string(experts_));
        }
        return layer * experts_ + expert;
    }

    std::vector<SlabTensor> tensors_;
    std::vector<std::uint8_t*> slab_;
    std::uint32_t experts_ = 0;
    std::uint32_t layers_ = 0;
    std::uint64_t slot_bytes_ = 0;
    pie_loader::GroupSlotIndex index_;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
};

}  // namespace pie::metal

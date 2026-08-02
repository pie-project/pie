#pragma once

/// Firing a step whose routed experts do not all fit on the device.
///
/// The mechanism, once, for every family that has a routed bank. It was
/// written inside the llama engine first and gpt-oss needed the same thing;
/// the second copy would have been the same eighty lines with different type
/// names, and the two would have drifted at the first fix applied to one.
///
/// What it does is decided by one fact about the hardware: an Apple Silicon
/// GPU has no demand paging, and `requestResidency` wires every page of a
/// mapping whether a kernel reads it or not. So a bank bigger than the working
/// set cannot be mapped -- it has to be read through a region that stays wired
/// while its CONTENTS change, and the host has to change them between
/// dispatches. That makes a step several ordered command buffers instead of
/// one, with the host awake between them:
///
///   [ ... router of layer 0 ]  -> host pages layer 0's experts in
///   [ ... router of layer 1 ]  -> host pages layer 1's experts in
///   ...
///   [ the tail ]                  pages nothing
///
/// and the ids the router wrote are rewritten IN PLACE from expert numbers to
/// slot numbers. That rewrite is why no kernel changed for any of this: a
/// routed matvec does `base += expert_ids[sel] * stride`, and a slot number in
/// a shorter stack is the same instruction against different bytes.
///
/// The cost is one submit-and-wait per mixture layer. It is a bad trade for a
/// model that fits and the only trade there is for a model that does not.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "loader/expert_slab.hpp"
#include "mtl4_context.hpp"

namespace pie::metal::batch {

class ExpertPaging {
  public:
    /// One mixture layer's cut and the buffer its router writes.
    ///
    /// `ids` is per LAYER, not one for the stack. A single handle assumed
    /// every layer's routing decision colours onto the same pool slot, which
    /// is a property of a greedy colouring rather than a guarantee -- and the
    /// day it stopped holding the host would have rewritten a buffer the next
    /// segment does not read, which is wrong tokens and not an error.
    struct Cut {
        std::size_t end = 0;  ///< one past the last dispatch of this segment
        SlotHandle ids{};
    };

    /// Returns false and fills `err` when the shape cannot be paged. Every
    /// refusal here is a case that would otherwise fail mid-decode, where the
    /// diagnosis is right and nothing can be done about it.
    bool plan(std::shared_ptr<ExpertSlab> slab, std::vector<Cut> cuts, std::size_t dag_size,
              int n_experts, int experts_per_token, int max_rows, const char* who,
              std::string* err) {
        slab_ = std::move(slab);
        cuts_ = std::move(cuts);
        n_experts_ = n_experts;
        experts_per_token_ = experts_per_token;
        const std::string tag = std::string(who) + " expert slab: ";
        if (cuts_.size() != std::size_t(slab_->layers())) {
            if (err) {
                *err = tag + std::to_string(cuts_.size()) + " routed layers in the DAG against " +
                       std::to_string(slab_->layers()) + " staged";
            }
            return false;
        }
        for (std::size_t k = 0; k < cuts_.size(); ++k) {
            if (k > 0 && cuts_[k].end <= cuts_[k - 1].end) {
                if (err) *err = tag + "two routers in one concurrency run";
                return false;
            }
            if (cuts_[k].end > dag_size) {
                if (err) *err = tag + "a cut past the end of the step";
                return false;
            }
            if (!cuts_[k].ids.valid() || cuts_[k].ids.contents() == nullptr) {
                if (err) {
                    *err = tag + "the routing decision of mixture layer " + std::to_string(k) +
                           " is not host-readable";
                }
                return false;
            }
        }
        // Every expert a single fire can route to must be resident AT ONCE:
        // one dispatch reads them all, so there is no order in which a smaller
        // cache could serve it. Refused with the number that would work.
        //
        // It is also why the slab never needs to be bigger than ONE layer's
        // bank however many layers the model has -- the whole reduction.
        const auto worst = std::uint32_t(
            std::min<std::int64_t>(n_experts, std::int64_t(max_rows) * experts_per_token));
        if (slab_->num_slots() < worst) {
            if (err) {
                *err = "expert_slab_bytes is too small: a fire of " + std::to_string(max_rows) +
                       " rows can route to " + std::to_string(worst) +
                       " distinct experts at once, which needs " +
                       std::to_string(std::uint64_t(worst) * slab_->slot_bytes()) +
                       " bytes, but the budget holds " + std::to_string(slab_->num_slots()) +
                       " experts";
            }
            return false;
        }
        cuts_.push_back(Cut{dag_size, SlotHandle{}});  // the tail, which pages nothing
        std::fprintf(stderr,
                     "[pie-metal] %s experts paged through %u slots of %llu, %.2f GB on the device\n",
                     who, slab_->num_slots(),
                     (unsigned long long)(std::uint64_t(n_experts) * slab_->layers()),
                     double(slab_->slab_bytes()) / 1e9);
        return true;
    }

    bool active() const { return !cuts_.empty(); }

    /// `encode(se, begin, end)` walks the dispatches in `[begin, end)`. The
    /// caller recognises the first segment by `begin == 0` and the last by
    /// `end == dag_size`, which is where any pre/post hook belongs.
    StepTiming fire(RawMetalContext& ctx, int rows,
                    const std::function<void(StepEncoder&, std::size_t, std::size_t)>& encode) {
        std::vector<std::function<void(StepEncoder&)>> segs;
        segs.reserve(cuts_.size());
        std::size_t begin = 0;
        for (const Cut& c : cuts_) {
            const std::size_t end = c.end;
            segs.push_back([&encode, begin, end](StepEncoder& se) { encode(se, begin, end); });
            begin = end;
        }
        const std::size_t ids = std::size_t(rows) * std::size_t(experts_per_token_);
        return ctx.run_segments(segs, [this, ids](std::size_t k) {
            // Give the previous segment's pins back FIRST: its command buffer
            // has completed, which is exactly the condition that makes its
            // slots reusable. Doing it after the page-in would hold two layers
            // at once and need twice the budget.
            slab_->end_batch();
            if (k + 1 >= cuts_.size()) return;  // the tail owns no experts
            auto* p = static_cast<std::int32_t*>(cuts_[k].ids.contents());
            for (std::size_t i = 0; i < ids; ++i) {
                const std::int32_t e = p[i];
                if (e < 0 || e >= n_experts_) continue;  // a padded slot
                p[i] = std::int32_t(slab_->ensure_resident(std::uint32_t(k), std::uint32_t(e)));
            }
        });
    }

  private:
    std::shared_ptr<ExpertSlab> slab_{};
    /// One per mixture layer in DAG order -- which is ascending layer number,
    /// and so the order the slab indexes its layer axis in -- plus a final
    /// entry for the tail.
    std::vector<Cut> cuts_{};
    int n_experts_ = 0;
    int experts_per_token_ = 0;
};

}  // namespace pie::metal::batch

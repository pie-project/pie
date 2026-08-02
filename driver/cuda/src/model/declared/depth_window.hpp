#pragma once

// The DEPTH AXIS's per-op rebind — family-neutral, and provably so: it
// reads the op's STATED role (`depth_role`, `layer`) and the prepare's
// stamped bands, and nothing about which family traced the plan.
//
// It lived inside llama_like's executor because that is where the depth
// axis landed first; qwen3_5's executor has no depth machinery at all,
// which is the same duplication story from the other side (one family
// carries the mechanism, the other simply cannot serve the axis).
//
// Three postures, in the order the walk tests them:
//
//   BANDED   2..3 distinct truncations (the prepare stamped k/rows,
//            deepest-first). At an op's layer the live rows are the
//            first band whose k the layer has reached; `live == 0` ends
//            that op — nothing lives that deep.
//   UNION    one boundary with full-depth rows beside it: tail-layer ops
//            run over the full-depth PREFIX (rows [0, split)).
//   UNIFORM  one boundary, every row truncated: tail-layer ops are
//            SKIPPED (the epilogue, layer -1, is the logit-lens head and
//            is never a tail op).
//
// A plan whose declaration does not state the axis (`stated == false`)
// leaves every op at the fire's own extents — which is what a family
// that has not adopted the axis passes.

#include <cstdint>

#include "pie_forward.h"

using pie_forward::PieForwardOp;

namespace pie_cuda_driver::model::declared {

// What the fire's prepare and plan derived about depth. All of it is
// stated by the trace or stamped by the prepare; none of it is a family
// fact.
struct DepthFacts {
    // The declaration states the depth axis (`plan.view().depth_window`).
    bool stated = false;
    // Uniform/union truncation, or -1 when the fire is full depth.
    int k = -1;
    // A union fire: `split` full-depth rows sit before the truncated ones.
    bool union_fire = false;
    int split = 0;
    // Banded fire: deepest-first k/rows arrays from the prepare.
    std::uint32_t band_count = 0;
    const std::uint32_t* band_k = nullptr;
    const std::uint32_t* band_rows = nullptr;
};

class DepthWindow {
   public:
    DepthWindow(const DepthFacts& facts, int n_fire, int r_fire)
        : facts_(facts),
          n_fire_(n_fire),
          r_fire_(r_fire),
          n_(n_fire),
          r_(r_fire),
          banded_(facts.stated && facts.band_count >= 2),
          uniform_or_union_(facts.stated && !banded_ && facts.k >= 0) {}

    // Rebinds the window for one op. False = the op is skipped entirely.
    bool enter(const PieForwardOp& op) {
        band_index_ = -1;
        tail_active_ = false;
        if (banded_) {
            int live = n_fire_;
            if (op.depth_role != 0) {
                // Deepest-first: the first band whose k this layer has
                // reached is the deepest interval containing it.
                for (std::uint32_t j = 0; j < facts_.band_count; ++j) {
                    if (op.layer >= static_cast<std::int32_t>(
                                        facts_.band_k[j])) {
                        live = static_cast<int>(facts_.band_rows[j]);
                        band_index_ = static_cast<int>(j);
                        break;
                    }
                }
            }
            if (live == 0) return false;
            n_ = live;
            r_ = live;
            if (band_index_ >= 0 && live == n_fire_) {
                band_index_ = -1;  // degenerate: every row lives
            }
            return true;
        }
        if (uniform_or_union_) {
            // Membership comes from the op's STATED role, not a
            // re-derived layer-tag rule: the one function every walker
            // shares is the trace itself.
            const bool tail_op =
                op.depth_role != 0 && op.layer >= facts_.k;
            if (tail_op && !facts_.union_fire) return false;
            tail_active_ = facts_.union_fire && tail_op;
            n_ = tail_active_ ? facts_.split : n_fire_;
            r_ = tail_active_ ? facts_.split : r_fire_;
            return true;
        }
        n_ = n_fire_;
        r_ = r_fire_;
        return true;
    }

    int n() const { return n_; }
    int r() const { return r_; }
    // The band whose plan this op's attention must use, or -1 for the
    // fire's own plan (full rows).
    int band_index() const { return band_index_; }
    // A union fire's tail op: running over the full-depth prefix.
    bool tail_active() const { return tail_active_; }
    bool banded() const { return banded_; }

   private:
    DepthFacts facts_;
    int n_fire_;
    int r_fire_;
    int n_;
    int r_;
    int band_index_ = -1;
    bool tail_active_ = false;
    bool banded_ = false;
    bool uniform_or_union_ = false;
};

}  // namespace pie_cuda_driver::model::declared

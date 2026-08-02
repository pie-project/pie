#pragma once

/// Gemma 4's activation dataflow: which value every dispatch's buffers touch.
///
/// Pure — no Metal, no allocation. Colouring the live ranges onto a buffer pool
/// is the shared half and happens after.

#include <cstdint>
#include <vector>

#include "decode_step.hpp"
#include "geometry.hpp"

namespace pie::metal::gemma4 {

/// One buffer of one dispatch, and the activation value it carries.
struct Use {
    int ordinal = 0;
    std::uint8_t bind_index = 0;
    int value = 0;
    bool is_write = false;
};

struct ScratchPlan {
    std::vector<Use> uses;
    int value_count = 0;
    /// The value the logits land in — what the sampler reads.
    int logits_value = -1;
};

ScratchPlan build_gemma4_scratch(const std::vector<Dispatch>& dag, const Gemma4Geometry& g);

/// `run_ends[i]`: the last ordinal of the concurrency run containing `i`. What
/// the colourer needs to know about which barriers the encoder will drop; the
/// encoder (`encode_gemma4_step`) reads the SAME derivation so the two cannot
/// disagree.
std::vector<int> gemma4_run_ends(const std::vector<Dispatch>& dag);

struct ScratchBind {
    std::uint8_t bind_index = 0;
    int color = -1;
};

struct ScratchColoring {
    std::vector<std::vector<ScratchBind>> per_dispatch;
    int colors_used = 0;
    bool hazard_free = false;
};

/// Colour the dataflow's live ranges onto pool buffers, honouring the barriers
/// the encoder will drop.
ScratchColoring color_gemma4_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                     bool no_recycle = false);

}  // namespace pie::metal::gemma4

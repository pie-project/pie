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

}  // namespace pie::metal::gemma4

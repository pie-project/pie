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
///
/// `index` is the dispatch's POSITION in the DAG, not its argument-table
/// ordinal. The two coincide on the decode path and diverge on the prefill one,
/// whose ordinals are shifted clear of it -- and position is what this means:
/// the colouring uses it as a time axis, alongside concurrency runs that are
/// themselves indexed by position.
struct Use {
    int index = 0;
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

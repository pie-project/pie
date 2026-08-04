#pragma once

/// GPT-OSS's activation dataflow: which value every dispatch's buffers touch.
///
/// Pure — no Metal, no allocation. Colouring the live ranges onto a buffer pool
/// is the shared half and happens after.

#include <cstdint>
#include <vector>

#include "decode_step.hpp"
#include "geometry.hpp"

namespace pie::metal::gptoss {

/// One buffer of one dispatch, and the activation value it carries.
///
/// `index` is the dispatch's POSITION in the DAG, not its argument-table
/// ordinal. The two coincide on the decode path and diverge on the prefill one;
/// position is what the colouring's time axis means.
struct Use {
    int index = 0;
    std::uint8_t bind_index = 0;
    int value = 0;
    bool is_write = false;
};

struct ScratchPlan {
    std::vector<Use> uses;
    int value_count = 0;
    /// The value the routing decision lands in. The host reads and rewrites it
    /// when the experts are paged: the ids the router produced are expert
    /// numbers, and the kernels beside a bounded slab need slot numbers.
    std::vector<int> expert_ids_by_layer;
    /// The value the logits land in — what the sampler reads.
    int logits_value = -1;
};

ScratchPlan build_gptoss_scratch(const std::vector<Dispatch>& dag, const GptOssGeometry& g);

}  // namespace pie::metal::gptoss

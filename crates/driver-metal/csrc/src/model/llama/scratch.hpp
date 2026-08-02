#pragma once

/// The llama families' activation dataflow: which value every dispatch touches.
///
/// Pure -- no Metal, no allocation. Colouring the live ranges onto a buffer
/// pool is the shared half and happens after.

#include <cstdint>
#include <vector>

#include "../family_coloring.hpp"
#include "../shared_kernels.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"

namespace pie::metal::llama {

/// One buffer of one dispatch, and the activation value it carries.
///
/// `index` is the dispatch's POSITION in the DAG, not its argument-table
/// ordinal. The two coincide on the decode path and diverge on the prefill one,
/// whose ordinals are shifted clear of it -- and position is what this means:
/// the colouring uses it as a time axis.
struct Use {
    int index = 0;
    std::uint8_t bind_index = 0;
    int value = 0;
    bool is_write = false;
};

struct ScratchPlan {
    std::vector<Use> uses;
    int value_count = 0;
    /// The value the logits land in -- what the sampler reads.
    int logits_value = -1;
    /// The routing decision of each mixture layer, in DAG order. Empty on a
    /// dense model. Per LAYER rather than one value for the stack, because the
    /// host rewrites this buffer when the experts are paged and has to rewrite
    /// the one the NEXT segment will read. A single value silently assumed
    /// every layer's ids colour onto the same pool slot -- true today, a
    /// property of a greedy colouring rather than a guarantee, and wrong tokens
    /// rather than an error on the day it stops holding.
    std::vector<int> expert_ids_by_layer;
    /// The routing WEIGHTS of each mixture layer, same axis.
    std::vector<int> expert_weights_by_layer;
};

/// Element width, in units of the model's hidden size, of each activation.
///
/// The pool sizes its slots from this. A routed model's expert stack is the
/// only place a value is `experts_per_token` times wider than its dense
/// counterpart, and that is exactly the fact a colouring pass cannot infer.
struct ValueExtent {
    int elems = 0;   // per row
    int rows_are_slots = 0;  // 1 if the value is [rows * experts_per_token, elems]
    /// 1 if the value is [`llama_moe_sorted_rows(rows)`, elems] -- the expert-
    /// major order the batched mixture works in, which is TALLER than the slot
    /// stack because every expert's run is padded to a whole tile.
    int rows_are_sorted = 0;
};

/// How many expert-sorted rows a batch of `rows` tokens produces, at worst.
///
/// The pool is sized from this, and a batch smaller than the one it was sized
/// for produces fewer -- the bound rises with the row count on both sides of
/// the point where the sort starts padding.
int llama_moe_sorted_rows(const LlamaGeometry& g, int rows);

/// The bind index `ExpertSort` writes its permutation to.
///
/// Exported because the golden-tap dump has to READ that permutation to publish
/// the routed intermediates in the slot-major layout every reference speaks,
/// and a second spelling of the number would be a second thing to keep true.
constexpr std::uint8_t kMoeSortPermBind = 1;

ScratchPlan build_llama_scratch(const std::vector<Dispatch>& dag, const LlamaGeometry& g);

/// Per-value extents, indexed by value id, sized `plan.value_count`.
std::vector<ValueExtent> llama_value_extents(const std::vector<Dispatch>& dag,
                                             const ScratchPlan& plan, const LlamaGeometry& g);

/// How many elements each pool colour must hold, indexed by colour.
///
/// Separate from the extents because a colour is shared: its size is the widest
/// value that landed in it, and `col.color_of_value` is what says which those
/// are.
std::vector<std::size_t> llama_pool_elems(const std::vector<Dispatch>& dag,
                                          const ScratchPlan& plan,
                                          const model::ScratchColoring& col,
                                          const LlamaGeometry& g, int rows = 1,
                                          int head_rows = 1);

}  // namespace pie::metal::llama

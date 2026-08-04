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
    /// The routing decision of each mixture layer, in DAG order; empty on a
    /// dense model. Per LAYER because paging rewrites this buffer between
    /// segments, and has to rewrite the one the NEXT segment reads.
    std::vector<int> expert_ids_by_layer;
    /// The routing WEIGHTS of each mixture layer, same axis.
    std::vector<int> expert_weights_by_layer;
};

ScratchPlan build_gemma4_scratch(const std::vector<Dispatch>& dag, const Gemma4Geometry& g);

/// How many expert-sorted rows a batch of `rows` tokens produces, at worst:
/// every (row, expert) pair, with each expert's run padded to a whole tile.
int gemma4_moe_sorted_rows(const Gemma4Geometry& g, int rows);

/// How wide one row of a value is, and what counts as a row.
///
/// The mixture is the reason this exists. A routed layer's expert stack has one
/// row per (token, slot) PAIR, padded to whole tiles, and it shares pool
/// colours with dense tensors that have one row per token -- so a colour sized
/// by the dispatch that happened to be tapped is either too small for the stack
/// or, under a golden dump's no-recycle colouring, wide enough for every value
/// at once. Measured: sizing by dispatch asked for 20 GB.
struct ValueExtent {
    int elems = 0;  // per row
    /// 1 if the value is [`gemma4_moe_sorted_rows(rows)`, elems].
    int rows_are_sorted = 0;
};

std::vector<ValueExtent> gemma4_value_extents(const std::vector<Dispatch>& dag,
                                              const ScratchPlan& plan,
                                              const Gemma4Geometry& g);

/// The tiling the sort pads every expert's run to.
///
/// One number, read by the launch shape and by the params struct the sort
/// itself reads -- a second spelling would let a tile disagree with the
/// padding, which reads one expert's weights for another's rows.
int gemma4_moe_tile_rows(const Gemma4Geometry& g, int rows);

}  // namespace pie::metal::gemma4

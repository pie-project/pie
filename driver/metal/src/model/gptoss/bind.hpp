#pragma once

/// Binding gpt-oss's argument tables.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../batch/decode_abi.hpp"
#include "../../loader/heap_bind.hpp"
#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"
#include "scratch.hpp"

namespace pie::metal::gptoss {

/// One layer's KV pages.
struct KvPages {
    SlotHandle k{};
    SlotHandle v{};
};

/// Everything staged before a step can be bound.
struct BoundGptOss {
    std::unordered_map<std::string, SlotHandle> weights;
    std::vector<SlotHandle> io;    // indexed by IoSlot
    std::vector<KvPages> kv;       // indexed by LAYER; every layer owns its own
    std::vector<SlotHandle> pool;  // activation buffers, indexed by colour
};

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
ScratchColoring color_gptoss_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                     bool no_recycle = false);

/// The paged binder: the same DAG against page-addressed KV and the CSR.
///
/// Not an M>1 path -- gpt-oss has none, because its MoE picks experts per ROW.
/// What paging buys is SEVERAL SEQUENCES: each one's history is its own page
/// list, so a second resident sequence no longer clobbers the first.
void bind_gptoss_dag_paged(RawMetalContext& ctx, const BoundGptOss& b,
                           const std::vector<Dispatch>& dag, const GptOssGeometry& g,
                           const ScratchColoring& scratch,
                           const std::vector<SlotHandle>& k_pages,
                           const std::vector<SlotHandle>& v_pages, int ordinal_base = 0);

void bind_gptoss_dag(RawMetalContext& ctx, const BoundGptOss& b, const std::vector<Dispatch>& dag,
                     const GptOssGeometry& g, const ScratchColoring& scratch,
                     int ordinal_base = 0);

/// Bytes of k (== bytes of v) one layer needs for `max_ctx` tokens.
inline std::size_t gptoss_kv_bytes_per_layer(const GptOssGeometry& g, int max_ctx,
                                             int act_dtype_bytes) {
    return std::size_t(g.n_kv_heads) * std::size_t(max_ctx) * std::size_t(g.head_dim) *
           std::size_t(act_dtype_bytes);
}

/// The widest activation any pool buffer must hold, in elements.
///
/// NOT `hidden`: the MoE's operands are a k-wide stack, so the expert gate/up
/// are `experts_per_token * intermediate` and the logits are the vocabulary.
inline int gptoss_widest_elems(const GptOssGeometry& g) {
    int widest = g.hidden;
    widest = widest < g.vocab ? g.vocab : widest;
    const int stack = g.experts_per_token * g.intermediate;
    widest = widest < stack ? stack : widest;
    const int qd = g.q_dim();
    widest = widest < qd ? qd : widest;
    return widest;
}

}  // namespace pie::metal::gptoss

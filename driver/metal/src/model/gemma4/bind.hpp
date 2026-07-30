#pragma once

/// Binding gemma4's argument tables.
///
/// The KV part is this family's own: layers past `first_kv_shared()` attend the
/// pages an earlier layer of the same attention type wrote, so what a dispatch
/// binds comes from `kv_source(L)` and the region is sized for `n_kv_owning()`.

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

namespace pie::metal::gemma4 {

/// One owning layer's KV pages.
struct KvPages {
    SlotHandle k{};
    SlotHandle v{};
};

/// Everything staged before a step can be bound.
struct BoundGemma4 {
    std::unordered_map<std::string, SlotHandle> weights;
    std::vector<SlotHandle> io;    // indexed by IoSlot
    std::vector<KvPages> kv;       // indexed by LAYER; only owning layers are filled
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
ScratchColoring color_gemma4_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                     bool no_recycle = false);

void bind_gemma4_dag(RawMetalContext& ctx, const BoundGemma4& b, const std::vector<Dispatch>& dag,
                     const Gemma4Geometry& g, const ScratchColoring& scratch,
                     int ordinal_base = 0);

}  // namespace pie::metal::gemma4

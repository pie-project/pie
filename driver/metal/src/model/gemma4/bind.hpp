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


/// Bytes of k (== bytes of v) one KV-owning layer needs for `max_ctx` tokens.
///
/// Sized per LAYER, not once for the stack: gemma 4's full-attention layers
/// carry head_dim 512 against the sliding layers' 256, so one number for all of
/// them either wastes half the region or truncates half the cache. Shared
/// layers own nothing and are not asked.
inline std::size_t gemma4_kv_bytes_per_layer(const Gemma4Geometry& g, int layer,
                                             int max_ctx, int act_dtype_bytes) {
    return std::size_t(g.n_kv_heads) * std::size_t(max_ctx) *
           std::size_t(g.head_dim_of(layer)) * std::size_t(act_dtype_bytes);
}

/// The whole KV region: k and v over the layers that own KV.
///
/// `n_kv_owning()` is 15 of E2B's 35 layers, so sizing this the way the GDN
/// family does -- count full-attention layers -- would ask for the wrong region
/// twice over: wrong count, and wrong per-layer size.
inline std::size_t gemma4_kv_region_bytes(const Gemma4Geometry& g, int max_ctx,
                                          int act_dtype_bytes) {
    std::size_t total = 0;
    for (int L = 0; L < g.n_layers; ++L) {
        if (g.is_kv_shared(L)) continue;
        total += 2 * gemma4_kv_bytes_per_layer(g, L, max_ctx, act_dtype_bytes);
    }
    return total;
}

}  // namespace pie::metal::gemma4

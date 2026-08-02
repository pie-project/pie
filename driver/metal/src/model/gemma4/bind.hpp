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
#include "../family_coloring.hpp"
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
    /// Zeros, for the routed matvec's bias slot. The shared routed template
    /// DECLARES a bias buffer and never reads it at BIASED=false, but Metal
    /// still wants the slot filled -- gemma 4's expert projections have no
    /// bias tensor to fill it with.
    SlotHandle zero_bias{};
};

// Shared: the colouring adapter and its result are identical across families.
using model::ScratchBind;
using model::ScratchColoring;

/// Colour the dataflow's live ranges onto pool buffers, honouring the barriers
/// the encoder will drop.
ScratchColoring color_gemma4_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                     bool no_recycle = false);

void bind_gemma4_dag(RawMetalContext& ctx, const BoundGemma4& b, const std::vector<Dispatch>& dag,
                     const Gemma4Geometry& g, const ScratchColoring& scratch,
                     int ordinal_base = 0);

/// Where in the per-row IO and the logits buffer this bind's token sits.
///
/// The prefill path binds the same DAG once per token row, so the only thing
/// that changes between them is an offset.
struct MbBindOffsets {
    std::size_t token_row = 0;
    std::size_t logits_bytes = 0;
};

/// The M>1 binder: the same DAG, against per-row IO and a paged KV cache.
///
/// Deliberately a second function rather than a flag on the first. What differs
/// is not a value but an ABI -- `Kind::Sdpa` and `Kind::KvAppend` run a
/// different kernel at M>1, whose slots mean different things -- and a binder
/// that switched on a bool in six places would be two implementations sharing a
/// name.
///
/// `k_pages`/`v_pages` are indexed by LAYER and only the owning ones are read;
/// a KV-shared layer's attention binds `kv_source(L)`'s pages, exactly as the
/// decode binder does.
void bind_gemma4_dag_mb(RawMetalContext& ctx, const BoundGemma4& b,
                        const std::vector<Dispatch>& dag, const Gemma4Geometry& g,
                        const ScratchColoring& scratch,
                        const std::vector<SlotHandle>& k_pages,
                        const std::vector<SlotHandle>& v_pages,
                        const MbBindOffsets& offsets = {}, int ordinal_base = 0);


/// One row of the dense allow-mask, in bytes: one entry per addressable KV slot.
///
/// Same quantity qwen3.5's `paged_attention_mask_pitch_bytes` computes, restated
/// against this family's geometry rather than shared, because the two structs
/// do not have a common base and threading one through would couple the
/// families for a product of two integers.
inline std::size_t gemma4_mask_pitch_bytes(const Gemma4Geometry& g) {
    if (g.total_pages <= 0 || g.kv_page_size <= 0) return 0;
    return std::size_t(g.total_pages) * std::size_t(g.kv_page_size);
}

/// Bytes of k (== bytes of v) one KV-owning layer needs for `max_ctx` tokens.
///
/// Sized per LAYER, not once for the stack: gemma 4's full-attention layers
/// carry head_dim 512 against the sliding layers' 256, so one number for all of
/// them either wastes half the region or truncates half the cache. Shared
/// layers own nothing and are not asked.
inline std::size_t gemma4_kv_bytes_per_layer(const Gemma4Geometry& g, int layer,
                                             int max_ctx, int act_dtype_bytes) {
    return std::size_t(g.n_kv_heads_of(layer)) * std::size_t(max_ctx) *
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

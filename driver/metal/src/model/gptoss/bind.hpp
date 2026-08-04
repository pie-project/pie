#pragma once

/// Binding gpt-oss's argument tables.

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

namespace pie::metal::gptoss {

/// One layer's KV pages.
struct KvPages {
    SlotHandle k{};
    SlotHandle v{};
};

/// The router's quantization width, solved from the two staged tensors.
///
/// `layers.0.mlp.router.weight` holds `n_experts * packed_cols` u32 words and
/// `.scales` holds `n_experts * groups` bf16, so with a group of 64 the width
/// falls straight out of the byte counts:
///
///     bits = 32 * packed_cols / (groups * 64) = weight_bytes / (4 * scale_bytes)
///
/// The same solve the contract does per tensor, done here because the PSO is
/// chosen on this side. Returns 0 when either tensor is missing or the ratio is
/// not a width these kernels have, so the caller refuses rather than guessing.
int router_bits_from_extents(std::uint64_t weight_bytes, std::uint64_t scale_bytes);

/// The same, looked up by name in a staged weight set.
/// Whether the staged expert bank is native MXFP4 (no `.biases`) or the affine
/// U4 the loader used to convert it to.
bool mxfp4_experts_from_weights(const std::unordered_map<std::string, SlotHandle>& weights);

int router_bits_from_weights(const std::unordered_map<std::string, SlotHandle>& weights);

/// Everything staged before a step can be bound.
struct BoundGptOss {
    std::unordered_map<std::string, SlotHandle> weights;
    std::vector<SlotHandle> io;    // indexed by IoSlot
    std::vector<KvPages> kv;       // indexed by LAYER; every layer owns its own
    std::vector<SlotHandle> pool;  // activation buffers, indexed by colour
};

// Shared: the colouring adapter and its result are identical across families.
using model::ScratchBind;
using model::ScratchColoring;

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

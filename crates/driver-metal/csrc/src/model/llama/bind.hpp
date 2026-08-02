#pragma once

/// Binding the llama families' argument tables.
///
/// Uniform layers, so nothing here redirects the way gemma4's KV-shared layers
/// do: layer L appends to layer L's pages and reads them back. What this file
/// does carry is the routed FFN's extra IO -- and that turns out to be nothing
/// either, because the router's ids and weights are ACTIVATIONS. They come out
/// of the pool like any other value, which is why the MoE path needed no new
/// IO slot and no special case in the binder.

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

namespace pie::metal::llama {

using model::ScratchBind;
using model::ScratchColoring;

/// One layer's KV pages.
struct KvPages {
    SlotHandle k{};
    SlotHandle v{};
};

/// Everything staged before a step can be bound.
struct BoundLlama {
    std::unordered_map<std::string, SlotHandle> weights;
    std::vector<SlotHandle> io;    // indexed by IoSlot
    std::vector<KvPages> kv;       // indexed by layer
    std::vector<SlotHandle> pool;  // activation buffers, indexed by colour
    /// A zeroed buffer for the routed matvec's bias slot.
    ///
    /// `affine_qmv_routed` and `affine_qmv_routed_bias` are ONE template with
    /// `BIASED` flipped, so the unbiased instantiation still DECLARES
    /// `buffer(7)` and never dereferences it. Qwen's experts have no bias
    /// tensor to put there, and leaving the slot unbound would leave whatever
    /// that ordinal held last -- which is fine only for as long as the shader
    /// keeps not reading it. Binding zeros costs one small allocation and makes
    /// the guarantee independent of the kernel's internals.
    SlotHandle zero_bias{};
    SlotHandle argmax_params{};
    SlotHandle eos_flag{};
};

/// Colour the dataflow's live ranges onto pool buffers, honouring the barriers
/// the encoder will drop.
ScratchColoring color_llama_scratch(const std::vector<Dispatch>& dag, const ScratchPlan& plan,
                                    bool no_recycle = false);

/// Bind one step's argument tables.
///
/// `paged` selects the ATTENTION ABI, and it is one binder rather than two.
/// gpt-oss carries a second, near-identical `bind_*_dag_paged`, and the only
/// thing that genuinely differs between the two is a pair of switch arms: the
/// paged KV kernels take the page tables and the per-token request map, where
/// the ring takes a single base pointer and a length. Everything else -- the
/// weights, the activations, the embedding and rope IO -- is the same code
/// twice. Llama does not need the sink-slot remap that motivated the split
/// there, so it takes the flag instead.
///
/// It must agree with the `paged` passed to `bind_llama_consts`, which selects
/// the matching constant layout. Disagreeing is not a crash: the ring's
/// `KHeadStride` lands on the paged ABI's `KvPageIndices`, and the kernel
/// walks a page table made of a stride.
void bind_llama_dag(RawMetalContext& ctx, const BoundLlama& b, const std::vector<Dispatch>& dag,
                    const LlamaGeometry& g, const ScratchColoring& scratch,
                    int ordinal_base = 0, bool paged = false);

/// Bytes of k (== bytes of v) one layer needs for `max_ctx` tokens.
inline std::size_t llama_kv_bytes_per_layer(const LlamaGeometry& g, int max_ctx,
                                            int act_dtype_bytes) {
    return std::size_t(g.n_kv_heads) * std::size_t(max_ctx) * std::size_t(g.head_dim) *
           std::size_t(act_dtype_bytes);
}

/// The whole KV region: k and v over every layer.
///
/// One number for the stack, unlike gemma4 -- llama's layers all have the same
/// head width and all own their pages, so there is nothing to sum over.
inline std::size_t llama_kv_region_bytes(const LlamaGeometry& g, int max_ctx,
                                         int act_dtype_bytes) {
    return std::size_t(g.n_layers) * 2 * llama_kv_bytes_per_layer(g, max_ctx, act_dtype_bytes);
}

}  // namespace pie::metal::llama

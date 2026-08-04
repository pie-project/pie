#pragma once
// heap_bind_metal.hpp — the Metal-side staging/bind surface (declarations) for decode_run.
//
// heap_bind.hpp stays PURE (no Metal) so heap_bind_probe can verify the name registry
// against the real checkpoint with zero Metal deps. The actual heap_alloc/memcpy/arg_bind
// surface (which needs alpha's RawMetalContext + SlotHandle) lives here, included by
// heap_bind.cpp (definitions) and decode_run.cpp (the headline driver).

#include <unordered_map>
#include <vector>
#include <string>

#include "decode_abi.hpp"
#include "heap_layout.hpp"      // HeapPlan
#include "loader/expert_slab.hpp"
#include "mtl4_context.hpp"     // RawMetalContext, SlotHandle
#include "pie_loader/plan.hpp"

namespace pie_loader {
class CheckpointSource;
}

namespace pie::metal {

struct Dispatch;  // beta: decode_step.hpp

// Every slot delta allocates: the bind pass + the beta scratch handoff read from this.
struct BoundDecode {
    HeapPlan plan;
    SlotHandle weights_region;
    /// Keeps alive whatever the weight slots point into when they point
    /// outside the heap -- the stream pack, or the checkpoint's own mapping.
    /// It MUST outlive every slot in `weights`: unmapping it under them reads
    /// as weights of exactly zero, which is a model that runs and answers
    /// nothing rather than a model that fails.
    std::shared_ptr<void> weight_mapping;

    // load-once weights, keyed by HF tensor name (tied lm_head appears once).
    std::unordered_map<std::string, SlotHandle> weights;

    // GDN persistent state, per layer (only GDN layers populated).
    struct GdnState { SlotHandle conv_state, conv_state_out, recurrent_state, conv_bias_zero; };
    std::vector<GdnState> gdn;        // size n_layers; full-attn entries unused

    // paged KV, per layer (only full-attn layers populated).
    struct KvSlots { SlotHandle k_pages, v_pages; };
    std::vector<KvSlots> kv;          // size n_layers; GDN entries unused

    // IO region (I1 per-token buffers + I3 logits). Indexed by IoSlot.
    SlotHandle io[kIoSlotCount];

    // device-argmax substrate (I3; allocated always, bound + read only when with_argmax):
    // argmax_params = ArgmaxParams const (vocab + inline EOS ids; Shared, runtime-writable),
    // eos_flag = u32[max_tokens] OUT (1 if NextToken[r] ∈ eos_ids). Inert at M=1/no-argmax.
    SlotHandle argmax_params;
    SlotHandle eos_flag;

    // activation ping-pong pool handed to beta (he assigns X/Out per dispatch).
    SlotHandle scratch[SCRATCH_POOL];
};

// Stage all weights/state/KV/IO/scratch into the single resident heap (allocation order
// follows the region plan). The view must outlive nothing (zero-copy mmap is read here).
/// The weights a load plan names, staged into the heap and keyed by runtime
/// name. Family-independent: the plan is authored by the model contract, and
/// the transforms it carries (dequantize, fill, band copies) are exactly where a
/// second implementation would quietly diverge.
struct StagedWeights {
    /// How many bytes were bound over a pack instead of copied into the heap.
    std::uint64_t streamed_bytes = 0;
    /// Keeps alive whatever the weight slots point into when they point
    /// outside the heap. Two things can be on the other end and they are the
    /// same obligation: the stream pack, or -- when the checkpoint already
    /// places its tensors where a device pointer may point -- the
    /// `CheckpointSource` whose mmap the slots slice directly.
    std::shared_ptr<void> weight_mapping;
    SlotHandle weights_region{};
    std::unordered_map<std::string, SlotHandle> weights;
    /// The routed experts' paging cache, when one was asked for. Null
    /// otherwise, and null is the ordinary case: a model that fits is bound
    /// whole and never calls back.
    std::shared_ptr<ExpertSlab> slab;
};
StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes);

/// How many bytes `stage_plan_weights` would stream for `streams`.
///
/// Pure -- it reads the plan and nothing else -- because the answer is needed
/// BEFORE the context exists: the heap has to be created without them, and a
/// heap sized for weights that then also get mapped is the footprint doubled
/// rather than halved.
/// `slab_paging` says the routed experts will be PAGED rather than mapped.
/// Mapping subsumes streaming -- when every weight can be bound where it lies
/// there is no pack to carve out -- but paging does not: it turns mapping off,
/// so only the routed banks leave the heap and the dense weights are copied
/// into it as they would be from a checkpoint nothing could map.
std::uint64_t streamable_plan_bytes(const pie_loader::LoadPlan& load,
                                    const std::function<bool(const std::string&)>& streams,
                                    bool slab_paging = false);

/// Stage with weight STREAMING for the tensors `streams` names.
///
/// Those tensors are copied ONCE into a page-aligned pack beside the
/// checkpoint, and thereafter bound over that pack's mapping. The arena is
/// rebuilt around what is left, so the streamed bytes leave the heap rather
/// than being duplicated beside it. On Apple silicon the pack's pages are
/// demand-faulted under GPU access and stay clean, so the kernel evicts them
/// under pressure: a model reads the weights it touches and pays for no more.
/// gpt-oss reads 4 of 32 experts per layer.
///
/// The pack exists because a stock safetensors checkpoint does not put tensors
/// where a device pointer may point: of gpt-oss's 400 expert tensors, 196 start
/// on a 4-byte boundary, 36 on 16, and none on 32.
/// How many weight bytes are resident, and how many one decoded token reads.
///
/// Measured from the tensors that were actually staged, not computed from a
/// geometry. The formula this replaces read `cfg.llama` for every family, so
/// it printed `0.00 GiB` for gemma4 and gpt-oss -- and it could not have been
/// merely repaired family by family: qwen3.5's stack is mostly linear
/// attention rather than the four projections a formula assumes, and gpt-oss's
/// expert bank is 4 bits under a one-byte exponent per 32 rather than a pair
/// of bf16 per 64. All of it is already a byte count in the heap.
///
/// Only two things are read in part rather than in whole, and both are named
/// for what they ARE rather than for which family holds them:
///
///  * the routed expert bank, of which a token pulls `experts_per_token` of
///    `n_experts`. That is the whole difference between a mixture's size and
///    its cost: counting the bank in full would say Qwen3-30B-A3B moves 30B of
///    weights per token when it moves about 3B.
///  * an embedding table that is only GATHERED -- one row a token, which rounds
///    to nothing next to a decoder layer. A TIED table is not one of these: the
///    head reads every row of it, so it counts whole.
struct WeightBytes {
    std::uint64_t resident = 0;
    /// Fractional because a mixture's active share need not land on a whole
    /// byte, and rounding it would be rounding the answer.
    double per_token = 0.0;
    /// How much of `resident` is the routed bank. Reported rather than kept
    /// private so a caller can CHECK the discount above against the top-k the
    /// checkpoint's own config declares -- two independent sources, which is
    /// the only way this can be checked at all. Everything else here is the
    /// heap describing itself, and a heap agrees with itself by construction.
    std::uint64_t routed_resident = 0;
};
WeightBytes weight_bytes(const std::unordered_map<std::string, SlotHandle>& weights,
                         int n_experts, int experts_per_token);

StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes,
    const std::function<bool(const std::string&)>& streams);

/// Page the routed experts through a bounded slab instead of binding the bank.
///
/// This is the only arrangement under which a model can exceed the machine.
/// Mapping does not achieve it -- `requestResidency` wires every page of a
/// mapping, and an Apple Silicon GPU has no demand paging to make it lazy --
/// so the routed bytes the GPU can reach must be a fixed region whose contents
/// the host swaps. Asking for this therefore turns mapping OFF: the dense
/// weights are copied into the heap as they would be from a checkpoint that
/// could not be mapped, and only the mmap's host pointers are kept, as the
/// source the slab copies from.
///
/// `budget_bytes` is what the slab may occupy in total. Slots are
/// `budget_bytes / slot_bytes`, floored, and at least one -- a budget below a
/// single expert is refused rather than rounded up, because a caller in that
/// position has not asked for a small cache.
struct ExpertSlabRequest {
    /// The mixture's arity. Each streamed bank is this many equal bands.
    int n_experts = 0;
    std::uint64_t budget_bytes = 0;
    bool valid() const { return n_experts > 1 && budget_bytes > 0; }
};

StagedWeights stage_plan_weights(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view,
    const pie_loader::LoadPlan& load,
    std::size_t weights_bytes,
    const std::function<bool(const std::string&)>& streams,
    const ExpertSlabRequest& slab);

BoundDecode stage_decode_storage(
    RawMetalContext& ctx,
    std::shared_ptr<pie_loader::CheckpointSource> view,
    const pie_loader::LoadPlan& load_plan,
    const DecodeGeometry& g,
    const HeapPlan& heap_plan,
    const std::function<bool(const std::string&)>& streams = {});

// Walk beta's DAG; bind delta's weight/state/KV/IO slots for each dispatch by ordinal.
void bind_decode_dag(RawMetalContext& ctx, const BoundDecode& b,
                     const std::vector<Dispatch>& dag, const DecodeGeometry& g,
                     bool gdn_prep = false);

}  // namespace pie::metal

#pragma once

/// A decode engine for a model family with no recurrent state.
///
/// `MetalExecutor::Impl` is qwen3.5's: its state is `DecodeGeometry`, that
/// family's `Dispatch`, `ScratchSchedule` and `BoundDecode`, plus GDN conv and
/// recurrent slots and the ping-pong that advances them. gemma4 and gpt-oss
/// share none of that, and both are the same simpler shape:
///
///   * one DAG, one activation pool, per-layer KV;
///   * no state to carry between tokens beyond the KV itself, so a fresh
///     sequence is a memset and nothing else.
///
/// So this is not a second executor. It is the family-shaped part of one, and
/// `Impl` keeps everything that is not family-shaped: the context, the logits
/// staging, the sequence bookkeeping.
///
/// Both families are PAGED, so several sequences are resident at once, each
/// attending its own page list. They differ in what a multi-row fire COSTS.
///
/// gemma4 has an M>1 path: one schedule serves decode and prefill, and a fire
/// of R rows is one wider launch. `gemma4_prefill_numerics_test` measures that
/// a decode step is a fire of one row and lands exactly where firing the whole
/// prompt at once does, which is what lets one schedule do both.
///
/// gpt-oss does not, and will not soon: its MoE picks experts per ROW, so a
/// batched routed matmul is a different kernel rather than a wider launch. A
/// fire of R rows is R PASSES, each a full DAG at M=1. It is correct and it
/// batches nothing, so concurrency costs it throughput rather than buying any
/// -- `max_rows()` is a time budget there and a memory one for gemma4.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mtl4_context.hpp"
#include "loader/heap_bind_metal.hpp"
#include "loader/load_plan.hpp"
#include "model/contract.hpp"
#include "forward.hpp"

namespace pie::metal::batch {

class SimpleFamilyEngine {
  public:
    virtual ~SimpleFamilyEngine();

    /// Build the engine for `family` against an already-created context.
    ///
    /// Returns null and fills `err` when the config describes a shape this
    /// driver cannot schedule — the geometry refuses rather than guessing.
    static std::unique_ptr<SimpleFamilyEngine> create(model::ModelFamily family,
                                                      RawMetalContext& ctx,
                                                      const std::string& kernels_dir,
                                                      const SetupConfig& cfg,
                                                      const pie_loader::LoadPlan& load_plan,
                                                      int max_ctx, std::string* err);

    /// How much heap the family needs on top of its weights, for a `max_ctx`
    /// context. Answered before the context exists, because it sizes it.
    static std::size_t extra_heap_bytes(model::ModelFamily family, const SetupConfig& cfg,
                                        int max_ctx);

    /// The most rows per fire this family can afford, for a `budget_bytes`
    /// activation pool.
    ///
    /// Derived by asking `extra_heap_bytes` — the same function that will do
    /// the allocating — rather than from a per-row price. A price has to be
    /// guessed once and then goes stale: the shipped one charges every prefill
    /// row a `vocab * 2` slice of logits, which was true before `RowGather`
    /// and has not been true since. Bisecting the real function cannot go
    /// stale, and costs microseconds because the DAG, the dataflow and the
    /// colouring are all pure.
    static std::uint32_t max_forward_tokens_for_budget(model::ModelFamily family,
                                                       const SetupConfig& cfg, int max_ctx,
                                                       std::uint64_t budget_bytes);

    /// Which of this family's tensors are worth streaming, or null if none are.
    ///
    /// Streaming binds them over a page-aligned pack instead of copying them
    /// into the heap, so the heap must be created WITHOUT them -- which is why
    /// this is answered before the context exists, alongside `extra_heap_bytes`.
    /// Takes the switch as a bit rather than a `SetupConfig`: qwen3.5 asks
    /// this from a path that has a `DecodeGeometry` and no config, and the
    /// predicate never wanted more than the one bit anyway.
    /// `slab_paging` says the bank will be PAGED rather than mapped, which
    /// changes one thing: a per-expert bias must join the slab instead of
    /// staying resident, because paging renumbers the buffer that indexes it.
    static std::function<bool(const std::string&)> stream_predicate(
        bool stream_routed_experts, bool slab_paging = false);

    /// One fire: several requests' new tokens sharing a command buffer.
    ///
    /// This is the shape a mixed prefill+decode batch has. A prefill request
    /// contributes its whole prompt; a decode request contributes one token.
    /// Nothing in the fire distinguishes them — `qo_indptr` says who owns which
    /// rows, and that is the only difference.
    ///
    /// `w_page` is PHYSICAL (the page the runtime allocated), matching
    /// `kv_append_paged`'s port; `kv_page_indices`/`kv_page_indptr` are the CSR
    /// the attention walks for each request's history.
    struct FireCsr {
        std::vector<std::uint32_t> token_ids;
        std::vector<std::uint32_t> position_ids;
        std::vector<std::uint32_t> req_of_token;
        std::vector<std::uint32_t> w_page;
        std::vector<std::uint32_t> w_off;
        std::vector<std::uint32_t> qo_indptr;
        std::vector<std::uint32_t> kv_page_indices;
        std::vector<std::uint32_t> kv_page_indptr;
        /// Which rows this fire will sample, in readout order. The tail runs
        /// over these and no others -- a prefill computes every row of the body
        /// and reads one per request, and the LM head is the step's most
        /// expensive dispatch by two orders of magnitude.
        std::vector<std::uint32_t> sample_rows;
        bool run_argmax = false;
    };

    virtual int vocab() const = 0;
    virtual int n_layers() const = 0;

    /// What the heap actually holds, and how much of it a token reads.
    ///
    /// The counting RULE lives once, in `pie::metal::weight_bytes`. What each
    /// family supplies is the two numbers only it knows -- the width of its
    /// expert bank and how much of it a token pulls -- because a mixture's size
    /// and its cost are different questions and nothing in the tensor names
    /// answers the second.
    virtual WeightBytes weight_bytes() const = 0;

    /// Whether this family stores its KV in pages the runtime allocates, and so
    /// can hold several sequences at once. False means a single-sequence ring:
    /// `fire` is refused and `step` replays one token at a time.
    virtual bool paged() const { return false; }

    /// The most rows one `fire` may carry, and the page geometry the runtime
    /// must allocate against. Meaningless unless `paged()`.
    virtual int max_rows() const { return 1; }
    /// The most logits rows one fire may produce. Bounded by the request count
    /// rather than the token count: a request samples one row.
    virtual int max_sampled_rows() const { return 1; }
    virtual int page_size() const { return 1; }
    virtual int total_pages() const { return 0; }

    /// Fire `csr`. The logits slot holds one row per SAMPLED row, in
    /// `csr.sample_rows` order.
    ///
    /// `pre`/`post` are encoded into the same command buffer, before and after
    /// the model — PTIR's device program rides along that way rather than in a
    /// second submission.
    using EncodeHook = std::function<void(StepEncoder&)>;
    virtual StepTiming fire(RawMetalContext& ctx, const FireCsr& csr,
                            const EncodeHook& pre = {}, const EncodeHook& post = {}) {
        (void)ctx;
        (void)csr;
        (void)pre;
        (void)post;
        return StepTiming{};
    }

    /// A fresh sequence: the KV is the only thing carried between tokens, so
    /// zeroing it is the whole reset.
    virtual void reset() = 0;

    /// One token. `position` is absolute; the attention reads `position+1` keys.
    virtual StepTiming step(RawMetalContext& ctx, std::uint32_t token_id,
                            std::uint32_t position) = 0;

    /// Where this step's logits land: bf16, `vocab` wide. The tail's last
    /// dispatch writes here directly, so nothing copies between the model and
    /// the sampler.
    virtual SlotHandle logits_slot() const = 0;

    /// Optional u32 greedy tokens, one per sampled row from the last fire.
    virtual SlotHandle greedy_tokens_slot() const { return {}; }

    /// Dump every tapped activation of the LAST step, under the names
    /// `tests/parity/gemma4_mlx_taps.py` writes.
    ///
    /// The engine is a second configuration of a path the raw tests already
    /// check, and a second configuration is exactly where the two can disagree
    /// while both look reasonable. Every numerical defect in this driver's three
    /// families was found by walking taps forward to the first disagreement;
    /// this is that walk pointed at the ENGINE rather than at the raw path.
    ///
    /// Requires `PIE_METAL_GOLDEN_DIR` and the pool it names: under a tap dump
    /// the engine colours with `no_recycle`, so nothing is overwritten before it
    /// is read. Off, it is not called and costs nothing.
    virtual void dump_taps(int rows) const { (void)rows; }
};

}  // namespace pie::metal::batch

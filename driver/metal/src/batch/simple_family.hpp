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
/// The two families are not at the same place. gemma4 is PAGED: one schedule
/// serves decode and prefill, several sequences are resident at once, and a
/// fire carries as many rows as the batch has —
/// `gemma4_prefill_numerics_test` measures that a decode step is a fire of one
/// row and lands exactly where firing the whole prompt at once does, which is
/// what lets one schedule do both.
///
/// gpt-oss is not: its KV is a position-indexed ring holding ONE sequence, and
/// it has no M>1 path at all. Its MoE picks experts per ROW, so a batched
/// routed matmul is a different kernel rather than a wider launch, and its
/// attention sink has no paged variant. `paged()` says which a family is, and
/// `MetalExecutor` refuses a second resident sequence for the ring-backed one
/// rather than serving it wrongly.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mtl4_context.hpp"
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
    };

    virtual int vocab() const = 0;
    virtual int n_layers() const = 0;

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

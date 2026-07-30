#pragma once

/// A decode engine for a model family with no recurrent state.
///
/// `MetalExecutor::Impl` is qwen3.5's: its state is `DecodeGeometry`, that
/// family's `Dispatch`, `ScratchSchedule` and `BoundDecode`, plus GDN conv and
/// recurrent slots and the ping-pong that advances them. gemma4 and gpt-oss
/// share none of that, and both are the same simpler shape:
///
///   * one decode DAG, one activation pool, contiguous per-layer KV;
///   * no state to carry between tokens beyond the KV itself, so a fresh
///     sequence is a memset and nothing else;
///   * M=1 per dispatch, which is what `run_member_forward` already drives —
///     it replays a prompt one token at a time and reads the logits after each.
///
/// So this is not a second executor. It is the family-shaped part of one, and
/// `Impl` keeps everything that is not family-shaped: the context, the logits
/// staging, the sequence bookkeeping.
///
/// What it deliberately does NOT do is the paged multi-request path. That needs
/// a page table per request and a prefill that fires M>1, and neither family has
/// been through an M>1 numerics walk yet. `MetalExecutor` refuses those fires
/// for these families rather than running them untested.

#include <cstdint>
#include <memory>
#include <string>

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

    virtual int vocab() const = 0;
    virtual int n_layers() const = 0;

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

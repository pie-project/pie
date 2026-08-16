//! Reading an FA2 plan cache, and preparing a launch out of one.
//!
//! # STATUS: the params filling went DOWN, and this is what stayed
//!
//! This module was *"the FA2 dispatches: params filling and the fire"*. The
//! filling is [`crate::attn::fa2::params`]' now -- `make_decode_params`,
//! `make_prefill_params`, `make_paged_kv`, `sm_scale_or_default` and the
//! `Buffers`/`Partials` pair -- and the five arm cascades are
//! [`crate::attn::fa2`]'s, beside the `DecodeArm`/`PrefillArm` they
//! answer with.
//!
//! **The move is the point, and the hazard is why.** The mirrors are pinned to
//! the layout `shim/cuda/cmath`'s `__fast_div_modulo` produces --
//! `{u32 @0, u64 @8}`, align 8, putting `paged_kv_t::num_heads` at **+24**.
//! The deleted `attention_flashinfer.cu` compiled against real CCCL, whose
//! `uint_fastdiv` is `{u32,u32,u32,i32}` align 4 and puts the same field at
//! **+20**, with `sizeof` reconverging at 96 under both. Both were correct for
//! their own reader and **a block filled on one side and read on the other is
//! a silent wrong answer, not a crash**. There is exactly one filler and one
//! reader, and the filler now lives in the crate whose `offset_of!` assertions
//! ARE the layout -- which a filler in this crate could not be checked by.
//!
//! # What is here
//!
//! 1. **The reading of a plan cache.** [`decode_plan_of`] and
//!    [`prefill_plan_of`] destructure a [`DecodePlanCache`] /
//!    [`PrefillPlanCache`] -- which owns `Vec`s and is re-planned once per
//!    fire -- into the `Copy` descriptor a routine takes. That is the whole of
//!    what a launch reads out of a cache, and it is this crate's vocabulary
//!    because the cache is.
//! 2. **The prepare-a-launch layer**, [`prefill`], which returns a
//!    [`PrefillDispatch`] and does not fire. `driver-cuda/src/tower/qwen3_vl`
//!    reaches the FA2 lattice this way, by path rather than through a trace
//!    statement, and [`super::plan::fire_prefill`] is what it hands the
//!    result to.
//! 3. **[`Partials::merge_job`]**, the split-KV fold as
//!    [`crate::cascade::merge_states`]' job.
//!
//! # WHAT §6.3 FOUND HERE, which is the reason the descent was worth doing
//!
//! (2) was FIVE functions -- `decode`, `decode_capture`, `prefill`,
//! `prefill_capture`, `prefill_custom` -- and four of them had no caller
//! anywhere in the workspace. That on its own would be ordinary dead code.
//! What made it a defect is what they did: each read a plan cache, called
//! `make_*_params`, ran an arm cascade and resolved a lattice point --
//! **which is exactly, line for line, what [`crate::attn::fa2`]'s six
//! routines do.** Two compositions of one params block, in two crates.
//!
//! This module's own header claims the FA2 hazard closed because there is
//! *"exactly one filler and one reader"*. There were two fillers, and the
//! crate boundary is what hid the second: the routines were down here and
//! these were up there, so no reader ever saw both at once. `decode_at` and
//! `prefill_at` made it plainest -- both existed twice, **byte for byte
//! identical**, once in each file.
//!
//! The four uncalled preparers are deleted, and with them `Capture`,
//! `CustomMask`, `DecodeDispatch`, both local `*_at` copies, and (in
//! [`super::plan`]) `fire_decode` and `MlaPlanCache`, none of which had a
//! producer left. `prefill` stays because the ViT tower needs the prepare and
//! the fire as two steps -- it must reach [`Fired::Split`]'s partials to fold
//! them -- which is the one thing a routine's single call cannot offer.
//!
//! The six trace symbols are NOT here any more: they are
//! [`crate::attn::fa2`]'s routines and `bind/arms/fa2.rs`'s arms. The
//! banner at the bottom of this file says what went with them.
//!
//! # Refusal
//!
//! The layer at (2) returns [`Fired`], which is `#[must_use]` and spells
//! *"declined"* differently from *"ran"*, per `fire/gemv.rs`'s precedent.
//! What a [`Decline`] can name is what this module can see: an empty plan
//! cache, an SM90 plan, a capture with no sink and a capture that composes
//! with nothing. The routines make the same four refusals as
//! `kernels::Refusal`s, because that is what an arm answers with.
//!
//! **A head_dim the lattice does not carry is not one of them**, and that is
//! where the lattice went rather than a loss: the point is resolved at the
//! fire, by the routine that also derives the rectangle, and refuses there
//! with the detail in the log. A `Decline` arm that could only be produced by
//! re-deriving the geometry a second time would be two derivations of one
//! fact.
//!
//! # What is deliberately not here
//!
//! - **No launch.** (2) prepares one; the stream belongs to the caller.
//!   [`Fired::Split`] is how it says a fire left partial results that
//!   something else must merge, and [`Partials`] carries everything the fold
//!   needs.
//! - **No SM90 prefill.** `dispatch_attention_flashinfer_prefill_bf16:783-798`
//!   forwarded to a separate hopper launcher when `cache.use_sm90`; that
//!   launcher is in the archive's own tree, not in the deleted file, and
//!   §44.7's rule stands -- every sm_90 claim in this migration is argued from
//!   the call graph and none from a run. [`Decline::Sm90Unported`] says so
//!   rather than firing an FA2 symbol at a plan that was not built for it.

use super::geometry::Device as FaDevice;
use crate::attn::fa2::params::{PrefillPagedParams, make_prefill_params};
use crate::attn::fa2::{PrefillArm, PrefillPoint};

use super::plan::{DecodePlanCache, PrefillPlanCache};

// THE RE-EXPORT BLOCK IS GONE, AND THE MOVE IS WHY.
//
// Nine names stood here — `Buffers`, `DecodePlan`, `Partials`, `PrefillPlan`
// and the five `*_arm` cascades — re-exported out of `attn::fa2` and
// `attn::fa2::params`. The stated reason was *"that is where every caller
// already reaches for them"*, which was true while this file was
// `driver-cuda`'s: a driver caller wrote `fa2d::Buffers` and the alternative
// was a rename in files the move had nothing else to say to.
//
// One crate down the reason inverts. This module is a SIBLING of the two it
// was re-exporting from, so the block became a module re-exporting its
// parent's items under a second path — one name reachable by two routes, with
// nothing to choose between them, which is the shape this crate refuses
// everywhere else. Seven of the nine had no external caller at all; the two
// that did (`Buffers` and `prefill_arm`, both `tower/qwen3_vl/attn.rs`'s)
// name their own module now, which is one line each and is the address a
// reader wants anyway.
use crate::attn::fa2::params::{Buffers, DecodePlan, Partials, PrefillPlan};

/// The outcome of a dispatch: what to launch, or why not.
///
/// `fire/gemv.rs`'s `#[must_use] enum Gemv { Launched, Declined(Decline) }`,
/// with one extra state the FA2 lattice needs and GEMV does not — a split fire
/// has not produced the answer yet — and a payload, because this module
/// prepares a launch rather than performing one.
///
/// **`Whole` and `Split` are not interchangeable.** Returning
/// `Split(d, partials)` means: after `d` runs, `o` is *not* the answer —
/// `d.params.o` has been redirected to `partials.tmp_v` and `d.params.lse` to
/// `partials.tmp_s` — and `VariableLengthMergeStates` has to run before the
/// caller's `o` means anything. That is
/// [`crate::cascade::merge_states::variable_length`], over
/// [`Partials::merge_job`], on the same stream, after the fire. A `bool` field
/// would not have made forgetting that a type error.
///
/// It was a `panic!` at all seven call sites while the fold had no unit. It
/// is a launch now, and the flag that kept the planner from ever producing a
/// `Split` — `fire::flashinfer_fa2`'s `disable_split_kv` — is off.
#[must_use]
#[derive(Clone, Copy, Debug)]
pub enum Fired<D> {
    /// The kernel below produces the whole answer in `o`.
    Whole(D),
    /// The kernel below produces partials that must be merged. See above.
    Split(D, Partials),
    /// Nothing to launch.
    Declined(Decline),
}

impl<D> Fired<D> {
    /// The launch, if there is one. `None` is a decline, and the caller that
    /// wants to know *why* should match instead.
    pub fn dispatch(&self) -> Option<&D> {
        match self {
            Self::Whole(d) | Self::Split(d, _) => Some(d),
            Self::Declined(_) => None,
        }
    }

    /// The split staging pointers, if the fire will produce partials.
    #[must_use]
    pub fn partials(&self) -> Option<Partials> {
        match *self {
            Self::Split(_, p) => Some(p),
            _ => None,
        }
    }
}

/// Why a dispatch launched nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The plan cache was never filled, or was invalidated.
    ///
    /// `attention_flashinfer.cu:504-508` threw here. A refusal rather than a
    /// panic because an empty cache is a caller-ordering mistake, which is
    /// recoverable, and not a broken JIT, which is not.
    Unplanned,
    /// A score-capturing dispatch asked for with a soft cap or a window.
    ///
    /// `attention_flashinfer.cu:551-560` threw here, in these words: the two
    /// capture arms are instantiated over `AttnScoreCapture` and
    /// `AttnScoreCaptureFull` only, and neither composes with the soft-cap or
    /// sliding-window variants — there is no such instantiation to name.
    CaptureVariantUnsupported,
    /// A score-capturing dispatch with no sink to capture into.
    ///
    /// `attention_flashinfer.cu:546-549` and `:849-856`. `score_out` or
    /// `score_indptr` null, or (prefill) `score_window` zero, which would make
    /// the kernel write every row it was asked to observe to a null base.
    CaptureSinkMissing,
    /// The plan was built for the SM90 launcher, which this lattice has not
    /// ported.
    ///
    /// `dispatch_attention_flashinfer_prefill_bf16:783-798` forwarded to
    /// `dispatch_attention_flashinfer_prefill_sm90_bf16` when
    /// `cache.use_sm90`. That launcher lived in the archive crate
    /// `kernels-cuda`'s hopper unit and was never part of the deleted file,
    /// so this is a **routing** gap and not a numerics one: an FA2 symbol
    /// fired against an SM90 plan reads
    /// a different `PrefillPlanInfo` layout. §44.7 — every sm_90 claim in this
    /// migration is argued from the call graph and none from a run — is why
    /// this refuses rather than guesses.
    ///
    /// # `attention_flashinfer_hopper.cu` IS DELETED, AND THIS IS WHERE ITS
    /// MEASUREMENTS LIVE NOW
    ///
    /// The 392-line FA3 body — a five-level template funnel over
    /// `::flashinfer::BatchPrefillWithPagedKVCacheDispatched`, plus the
    /// scheduler call that filled `HopperPrefillPlan` — went in the pass that
    /// emptied the archive's `attn/` of everything but XQA and MLA. It was
    /// unreachable **by call graph** and not merely by [`super::plan`]'s
    /// unconditional `use_sm90 = false`: the three functions it defined have
    /// no table row (`launch_abi.rs` pins `plan_…` as `NoRow::Prepare` and
    /// `dispatch_…` as `NoRow::KernelsInternal`), so no `pie_k_*` entry and no
    /// `ffi` arm ever existed, and its one C++ caller went with
    /// `driver-cuda/csrc/`. That distinction is the whole of why it could go
    /// without a replacement, and a decline that survives its subject has to
    /// carry the subject's evidence.
    ///
    /// **The one real run behind that file, kept because a port that consumes
    /// a measurement is a regression even if it compiles.** Its
    /// extended-shapes predicate — head_dim 256, a sliding window and
    /// decode-shaped fires all routed to FA3, on by default — was justified
    /// by: *gemma-4-26B-A4B at 1k context, routing the sliding layers' decode
    /// to the Hopper path takes attention from **4.19 ms to 2.73 ms** and the
    /// model from **122.5 to 144.1 tok/s**, output unchanged.* That is the
    /// prize this decline is currently declining, stated in the units it was
    /// measured in, so that whoever ports FA3 knows what it is worth.
    ///
    /// Three smaller facts from the same header, each of which cost an
    /// argument to establish:
    ///
    /// * The predicate **was** `getenv("PIE_CUDA_DISABLE_HOPPER_EXTENDED")`,
    ///   read per call, and became a named constant because it answers a
    ///   CAPABILITY — "does this deployment serve head_dim 256, a sliding
    ///   window, a decode-shaped prefill" — and a capability answered per
    ///   launch is answered too late to be refusable. Nothing in the
    ///   repository ever set the variable, so `true` is exactly
    ///   `getenv(...) == nullptr` and every deployment got the same answer.
    /// * A `void set_hopper_extended_shapes(bool)` seam was written and
    ///   **withdrawn** for having no caller — *"a symbol added ahead of its
    ///   caller is not a decision, it is a backlog entry wearing a
    ///   signature"*. Its intended home was `driver-cuda`'s `Boot`. If FA3
    ///   returns, the fact belongs there and not in a header.
    /// * The head_dim funnel dispatched **128 and 256 only**, and 256 was
    ///   there for a reason worth keeping: Gemma-4's sliding layers and
    ///   Qwen3.6's full-attention layers both attend at 256, which are the
    ///   shapes this driver serves from FA2 today and the ones vLLM serves
    ///   from FlashAttention.
    /// * §36's audit of the `getenv` fold was compiled for sm_90 with **SASS
    ///   identical before and after**, and the audit box was an L40S (sm_89)
    ///   where `PIE_HAS_SM90` is false — so the file was not merely
    ///   unreachable there, it was not in the binary. That is the shape of
    ///   every sm_90 claim in this migration and the reason §44.7 exists.
    Sm90Unported,
}

impl core::fmt::Display for Decline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Unplanned => {
                write!(f, "flashinfer fa2 dispatch: the plan cache is empty; plan before firing")
            }
            Self::CaptureVariantUnsupported => write!(
                f,
                "flashinfer fa2 score capture: not instantiated with a logits soft cap \
                 or a sliding window"
            ),
            Self::CaptureSinkMissing => write!(
                f,
                "flashinfer fa2 score capture: requires score_out, score_indptr and a \
                 non-zero window"
            ),
            Self::Sm90Unported => write!(
                f,
                "flashinfer fa2 prefill: this plan is an SM90 plan and the SM90 launcher \
                 is not part of this lattice"
            ),
        }
    }
}

/// The fold a [`Fired::Split`] needs, as [`crate::cascade::merge_states`]'
/// job.
///
/// **A method, and it was a free function.** The reason it could not be one
/// is in its own former doc: *"an inherent `impl` on it cannot be written on
/// this side of the crate boundary"* — [`Partials`] is
/// [`crate::attn::fa2::params`]' and this file was `driver-cuda`'s. §6.3
/// moved this file down and [`crate::cascade::merge_states`] with it, so both
/// types are this crate's and the boundary that forced the shape is gone.
///
/// The RENAME is the whole risk in the conversion and is what the test below
/// pins: `tmp_v` is the merge's INPUT `v` and `o` is its OUTPUT `v_merged`,
/// and the two are both `u64`, so a transposition type-checks.
impl Partials {
    /// This split's operands, in `cascade.cuh`'s names.
    #[must_use]
    pub const fn merge_job(self) -> crate::cascade::merge_states::VarLen {
        crate::cascade::merge_states::VarLen {
            v: self.tmp_v,
            s: self.tmp_s,
            indptr: self.indptr,
            v_merged: self.o,
            s_merged: self.lse,
            max_seq_len: self.max_seq_len,
            seq_len: self.seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        }
    }
}

/// What a decode launch reads out of a plan cache, as a value.
///
/// **This is the destructuring the crossing turns on.** A `DecodePlanCache`
/// owns `Vec`s and is re-planned once per fire, so it can never be an
/// argument; what a launch reads out of one is a descriptor, seven shape
/// scalars and two flags, all `Copy` and all fixed for the fire. Reading them
/// HERE is what keeps `flashinfer_fa2`'s cache the driver's own type while
/// the params filling lives beside the mirrors it fills.
///
/// The device rides along for the reason
/// [`crate::attn::fa2::params::DecodePlan`] gives: the geometry is
/// derived against it and the plan was sized against the same part.
#[must_use]
pub fn decode_plan_of(cache: &DecodePlanCache, device: FaDevice) -> DecodePlan {
    DecodePlan {
        info: cache.plan_info,
        device,
        num_requests: cache.num_requests,
        num_q_heads: cache.num_q_heads,
        num_kv_heads: cache.num_kv_heads,
        head_dim: cache.head_dim,
        page_size: cache.page_size,
        int_base_bytes: cache.int_base_bytes as u64,
        hnd_layout: cache.hnd_layout,
        full_attention_variant: cache.full_attention_variant,
        valid: cache.valid,
    }
}

/// [`decode_plan_of`]'s twin.
///
/// `cta_tile_q` is read back from the plan rather than recomputed: the planner
/// SPLIT the batch against this tile, so a fire that chose its own would index
/// a work list built for a different one.
#[must_use]
pub fn prefill_plan_of(cache: &PrefillPlanCache, device: FaDevice) -> PrefillPlan {
    PrefillPlan {
        info: cache.plan_info,
        device,
        num_requests: cache.num_requests,
        num_q_heads: cache.num_q_heads,
        num_kv_heads: cache.num_kv_heads,
        head_dim: cache.head_dim,
        page_size: cache.page_size,
        cta_tile_q: cache.cta_tile_q,
        window_left: cache.window_left,
        hnd_layout: cache.hnd_layout,
        full_attention_variant: cache.full_attention_variant,
        causal_mask: cache.causal_mask,
        use_sm90: cache.use_sm90,
        valid: cache.valid,
    }
}

/// Everything a prefill launch needs: which lattice point, which arm, the
/// grid's two extents, and the one `__grid_constant__` argument.
///
/// There was a `DecodeDispatch` beside it and there is no decode caller left
/// -- see this module's header for what the descent found and removed.
#[derive(Clone, Copy, Debug)]
pub struct PrefillDispatch<P = PrefillPagedParams> {
    /// Which lattice point, which arm, and the grid's two extents.
    pub at: PrefillPoint,
    /// The second of the kernel's two template arguments, by value.
    pub params: P,
}

/// The prefill dispatch.
///
/// Replaces `dispatch_attention_flashinfer_prefill_bf16`
/// (`attention_flashinfer.cu:776-836`).
pub fn prefill(
    cache: &PrefillPlanCache,
    bufs: &Buffers,
    device: FaDevice,
    arm: PrefillArm,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> Fired<PrefillDispatch> {

    /// The two plan-validity refusals, in the order the C++ made them.
    ///
    /// Shared by the three prefill entry points so that all three make them the
    /// same way: `:780` tests `cache.valid` and `:783` tests `cache.use_sm90`, and
    /// `dispatch_attention_flashinfer_prefill_custom_bf16:1132` tests both in one
    /// `if`. Both read the PLAN, which is why they stayed here when the lattice
    /// lookup and the geometry went down to [`crate::attn::fa2`].
    fn prefill_plan_usable(cache: &PrefillPlanCache) -> Result<(), Decline> {
    if !cache.valid {
    return Err(Decline::Unplanned);
    }
    if cache.use_sm90 {
    return Err(Decline::Sm90Unported);
    }
    Ok(())
    }

    if let Err(why) = prefill_plan_usable(cache) {
        return Fired::Declined(why);
    }
    let plan = prefill_plan_of(cache, device);

    let (params, partials) = make_prefill_params(&plan, bufs, logits_soft_cap, sm_scale);

    let ready = PrefillDispatch { at: super::prefill_at(&plan, arm, params.padded_batch_size), params };
    if cache.plan_info.split_kv { Fired::Split(ready, partials) } else { Fired::Whole(ready) }
}
#[cfg(test)]
mod tests {
    // The five arm cascades and `offset_ptr` moved to `kernels-cuda` with
    // the enums and the mirrors they answer for, and their tests moved with
    // them -- `x::fa2::tests` and `fa2::params::tests`. What is still this
    // module's is the seam: a half-answer that must not read like an answer.
    use super::{Fired, Partials};

    /// `Fired` distinguishes an answer from a half-answer at the type level,
    /// and [`Partials::merge_job`] carries every field the fold needs.
    #[test]
    fn a_split_fire_is_not_an_answer() {
        let staging = Partials {
            tmp_v: 1,
            tmp_s: 2,
            indptr: 3,
            o: 4,
            lse: 5,
            max_seq_len: 6,
            seq_len: 7,
            num_heads: 8,
            head_dim: 128,
        };
        let split: Fired<u8> = Fired::Split(7, staging);
        let whole: Fired<u8> = Fired::Whole(7);
        assert_eq!(split.dispatch(), Some(&7));
        assert_eq!(whole.dispatch(), Some(&7));
        assert!(split.partials().is_some(), "a split fire hands back its staging");
        assert!(whole.partials().is_none(), "a whole fire has nothing to merge");

        // The rename is the whole risk in this conversion: `tmp_v` is the
        // merge's INPUT `v` and `o` is its OUTPUT `v_merged`, and the two
        // are both `u64`. Nine distinct values, so a transposition of any
        // pair is a failure here.
        let job = staging.merge_job();
        assert_eq!(job.v, staging.tmp_v);
        assert_eq!(job.s, staging.tmp_s);
        assert_eq!(job.indptr, staging.indptr);
        assert_eq!(job.v_merged, staging.o);
        assert_eq!(job.s_merged, staging.lse);
        assert_eq!(job.max_seq_len, staging.max_seq_len);
        assert_eq!(job.seq_len, staging.seq_len);
        assert_eq!(job.num_heads, staging.num_heads);
        assert_eq!(job.head_dim, staging.head_dim);
    }
}
// ───────────────────────────────────────────────────────────────────────────
// WHERE THE SIX ENTRY POINTS WENT
//
// `attn_dispatch_attention_flashinfer_{decode,decode_capture,prefill_bf16,
// prefill_capture_bf16,prefill_custom}` and `attn_attention_flashinfer_prefill`
// stood here: six `unsafe fn`s that took a plan handle, a layer view and a
// stream, and did the dequant prelude, the params filling, the fire and the
// split fold in that order. They are `crate::attn::fa2`'s six
// routines now, and `driver-cuda/src/bind/arms/fa2.rs` is the arm that reads
// a statement into one.
//
// **Two host programs for one symbol was the thing to delete.** This module's
// own header argues that the FA2 hazard closed because there is *"exactly one
// filler and one reader"*; an entry point that composed the same filling
// beside a routine that composed it again would have reopened exactly that,
// one layer up. `fa2_buffers` went with them: it was their operand widening,
// and a routine takes the pointers loose.
//
// What is left above is what a plan cache is: the reading of one
// ([`decode_plan_of`], [`prefill_plan_of`]), and the prepare-a-launch layer
// the ViT still calls by path ([`prefill`], [`Fired`], [`PrefillDispatch`]).
// ───────────────────────────────────────────────────────────────────────────

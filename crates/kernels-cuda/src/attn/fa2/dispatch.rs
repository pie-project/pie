//! Reading an FA2 plan cache, and preparing a launch out of one.
//!
//! Three things live here: [`decode_plan_of`] / [`prefill_plan_of`], which
//! destructure a [`DecodePlanCache`] / [`PrefillPlanCache`] — `Vec`-owning,
//! re-planned once per fire — into the `Copy` descriptor a routine takes;
//! [`prefill`], which prepares a [`PrefillDispatch`] and does not fire, because
//! `driver-cuda/src/tower/qwen3_vl` reaches the lattice by path and must get at
//! [`Fired::Split`]'s partials to fold them, which a routine's single call
//! cannot offer; and [`Partials::merge_job`], the split-KV fold as
//! [`crate::cascade::merge_states`]' job.
//!
//! **Params filling is [`crate::attn::fa2::params`]', not this module's, and
//! the hazard is why.** The mirrors are pinned to the layout
//! `shim/cuda/cmath`'s `__fast_div_modulo` produces — `{u32 @0, u64 @8}`,
//! align 8, putting `paged_kv_t::num_heads` at **+24**; real CCCL's
//! `uint_fastdiv` is `{u32,u32,u32,i32}` align 4 and puts it at **+20**, with
//! `sizeof` reconverging at 96 under both. A block filled on one side and read
//! on the other is a silent wrong answer, not a crash, so there is exactly one
//! filler and one reader, and the filler lives in the crate whose `offset_of!`
//! assertions ARE the layout.
//!
//! **Refusal.** [`Fired`] is `#[must_use]` and spells *"declined"* differently
//! from *"ran"*. A [`Decline`] names what this module can see: an empty plan
//! cache, an SM90 plan, a capture with no sink, a capture that composes with
//! nothing. A head_dim the lattice does not carry is deliberately NOT among
//! them — the point is resolved at the fire by the routine that also derives
//! the rectangle, and a `Decline` arm for it would be two derivations of one
//! fact. No launch happens here; the stream belongs to the caller.

use super::geometry::Device as FaDevice;
use crate::attn::fa2::params::{PrefillPagedParams, make_prefill_params};
use crate::attn::fa2::{PrefillArm, PrefillPoint};

use super::plan::{DecodePlanCache, PrefillPlanCache};

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
    /// `dispatch_attention_flashinfer_prefill_sm90_bf16` when `cache.use_sm90`.
    /// That launcher reads a different `PrefillPlanInfo` layout, so this is a
    /// **routing** gap and not a numerics one. §44.7 — every sm_90 claim in this
    /// migration is argued from the call graph and none from a run — is why it
    /// refuses rather than guesses.
    ///
    /// # Where `attention_flashinfer_hopper.cu`'s measurements live now
    ///
    /// The FA3 body was unreachable **by call graph**, not merely by
    /// [`super::plan`]'s unconditional `use_sm90 = false`: its three functions
    /// have no table row (`launch_abi.rs` pins `plan_…` as `NoRow::Prepare`,
    /// `dispatch_…` as `NoRow::KernelsInternal`), so no `pie_k_*` entry and no
    /// `ffi` arm ever existed. Its one measured run is what this decline is
    /// declining, in the units it was measured in: *gemma-4-26B-A4B at 1k
    /// context, routing the sliding layers' decode to the Hopper path takes
    /// attention from **4.19 ms to 2.73 ms** and the model from **122.5 to
    /// 144.1 tok/s**, output unchanged.*
    ///
    /// Three facts from the same header, each of which cost an argument:
    ///
    /// * The extended-shapes predicate was `getenv("PIE_CUDA_DISABLE_HOPPER_EXTENDED")`
    ///   read per call, and became a constant because it answers a CAPABILITY —
    ///   head_dim 256, a sliding window, a decode-shaped prefill — and a
    ///   capability answered per launch is answered too late to be refusable.
    /// * A `void set_hopper_extended_shapes(bool)` seam was **withdrawn** for
    ///   having no caller. If FA3 returns, that fact belongs in `driver-cuda`'s
    ///   `Boot` and not in a header.
    /// * head_dim dispatched **128 and 256 only**, and 256 earns its place:
    ///   Gemma-4's sliding layers and Qwen3.6's full-attention layers both
    ///   attend at 256, which are the shapes this driver serves from FA2 today.
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

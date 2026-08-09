//! The FA2 dispatches: params filling and the fire.
//!
//! # STATUS: the three seams are closed and the C++ is deleted
//!
//! `driver-cuda/csrc/attn/attention_flashinfer.cu` and `plan_lifecycle.cpp`
//! **are gone**, and with them `build.rs`'s last `.cuda(true)`. What used to
//! stand here was a list of three seams; this is what each became, in the
//! order the list had them:
//!
//! 1. **The launch.** [`decode`], [`prefill`], [`decode_capture`] and
//!    [`prefill_capture`] still return a [`DecodeDispatch`] /
//!    [`PrefillDispatch`] and still do not fire — that return type is the
//!    point, not the gap. The fire is
//!    [`super::flashinfer_fa2::fire_decode`] and
//!    [`super::flashinfer_fa2::fire_prefill`], which own the stream and hand
//!    the point and the params to [`kernels_cuda_new::x::fa2`].
//! 2. **`bind/mod.rs`'s `DecodePlan`/`PrefillPlan`** now own a
//!    [`DecodePlanCache`] / [`PrefillPlanCache`] directly. The seven
//!    `pie_x_*` entry points and their `plan_lifecycle.cpp` bodies are
//!    deleted; the C++ existed for *"a `unique_ptr` with a custom deleter"*
//!    and the deleter is now [`Drop`].
//! 3. **`Plan::int_upload`'s H2D** is [`super::flashinfer_fa2::upload_int_plan`],
//!    called by the two fire functions immediately before the launch that
//!    reads it — which is where north star §5 step 7 puts it.
//!
//! # The hazard that closed with the `cc::Build`
//!
//! [`params`] is pinned to the layout `csrc/shim/cuda/cmath`'s
//! `__fast_div_modulo` produces — `{u32 @0, u64 @8}`, align 8, putting
//! `paged_kv_t::num_heads` at **+24**. `attention_flashinfer.cu` compiled
//! against real CCCL, whose `uint_fastdiv` is `{u32,u32,u32,i32}` align 4 and
//! puts the same field at **+20**, with `sizeof` reconverging at 96 under
//! both. Both were correct for their own reader and **a block filled on one
//! side and read on the other is a silent wrong answer, not a crash**. There
//! is now exactly one filler and one reader, so the question cannot be asked.
//!
//! The other half of [`super::flashinfer_fa2`]. That module plans; this one
//! turns a plan plus a batch into the single `__grid_constant__` struct each
//! FA2 `__global__` takes and works out where to put it.
//!
//! # What this replaces
//!
//! `attention_flashinfer.cu`'s four `switch (cache.head_dim)` dispatches
//! (`:490`, `:537`, `:776`, `:837`) and the `AttnHd<HD>` member templates they
//! delegated to — `run_decode` (`attention_flashinfer_common.cuh:567-686`),
//! `dispatch_decode` (`:687-722`), `make_prefill_params`
//! (`attention_flashinfer.cu:693-775`) — plus the custom-mask dispatch at
//! `:1115-1224` and the planless prefill at `:1077-1113`.
//!
//! The switch itself does not survive as a switch. It existed because
//! `AttnHd<HD>` is a class template and `HD` had to be a compile-time constant
//! to name an instantiation; `#include "kernels.def"` expanded one `case` per
//! head_dim so that a runtime integer could reach one. Under the JIT the
//! head_dim is a *lookup key* —
//! [`kernels_cuda_new::x::fa2::decode_root`] returns a root or [`None`] — so
//! the four switches become four calls and the `kernels.def` expansion has no
//! remaining job. The lookup itself is one layer further down than it was:
//! this module names the POINT and the routine resolves it, because the same
//! function that picks the root is the one that derives the rectangle.
//!
//! # The variant selection is NOT a switch on head_dim and never was
//!
//! `dispatch_decode` (`:697-722`) branches on the *request*, not the
//! geometry: full attention when `full_attention_variant && window_left < 0 &&
//! logits_soft_cap <= 0`, soft-cap when `logits_soft_cap > 0`, and the sliding
//! window otherwise — **in that order**, which matters, because a windowed
//! layer with a soft cap takes the soft-cap arm. [`decode_arm`] is that
//! cascade with its order preserved, and [`prefill_arm`] is `prefill`'s
//! (`attention_flashinfer_common.cuh`), whose non-full branch is causal-only.
//!
//! # Refusal
//!
//! Everything here returns [`Fired`], which is `#[must_use]` and spells
//! *"declined"* differently from *"ran"*, per `fire/gemv.rs`'s precedent.
//! What a [`Decline`] can still name is what this module can still see: an
//! empty plan cache, an SM90 plan, a capture with no sink and a capture that
//! composes with nothing.
//!
//! **A head_dim the lattice does not carry is no longer one of them**, and
//! that is where the lattice went rather than a loss: the point is resolved
//! at the fire, by the routine that also derives the rectangle, and refuses
//! there with the detail in the log. A `Decline` arm that could only be
//! produced by re-deriving the geometry a second time would be two
//! derivations of one fact.
//!
//! # What is deliberately not here
//!
//! - **No `tmp_v`/`tmp_s` merge launch — but there is now a fold to call.**
//!   `BatchDecodeWithPagedKVCacheDispatched` launches the FA2 kernel *and
//!   then*, on the split path, `VariableLengthMergeStates` — a second kernel,
//!   from `attention/cascade.cuh`, which is not part of THIS lattice. It has
//!   a unit of its own now (`families::cascade`) and a host program in
//!   `fire/merge_states.rs`. This module still does not launch it: it
//!   prepares launches and performs none, so [`Fired::Split`] remains the way
//!   it says a fire left partial results that something else must merge —
//!   and [`Partials`] now carries everything that merge needs.
//! - **No stream ordering.** It belongs to the caller that owns the stream;
//!   see [`super::flashinfer_fa2::fire_decode`].
//! - **No SM90 prefill.** `dispatch_attention_flashinfer_prefill_bf16:783-798`
//!   forwarded to a separate hopper launcher when `cache.use_sm90`; that
//!   launcher is in `kernels-cuda`'s own tree, not in the deleted file, and
//!   §44.7's rule stands — every sm_90 claim in this migration is argued from
//!   the call graph and none from a run. [`Decline::Sm90Unported`] says so
//!   rather than firing an FA2 symbol at a plan that was not built for it.

use kernels_cuda_new::fa2::Device as FaDevice;
use kernels_cuda_new::fa2::params::{
    DecodeParams, DecodeScoreParams, PagedKv, PrefillPagedParams, PrefillScoreParams, UintFastdiv,
};
use kernels_cuda_new::x::fa2::{DecodeArm, DecodePoint, PrefillArm, PrefillPoint};

use super::flashinfer_fa2::{DecodePlanCache, PrefillPlanCache};

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
/// `crate::fire::merge_states::variable_length`, over
/// [`Partials::merge`], on the same stream, after the fire. A `bool` field
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
    /// `cache.use_sm90`. That launcher lived in `kernels-cuda`'s hopper unit
    /// and was never part of the deleted file, so this is a **routing** gap
    /// and not a numerics one: an FA2 symbol fired against an SM90 plan reads
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
    /// unreachable **by call graph** and not merely by [`crate::fire::flashinfer_fa2`]'s
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

/// Where a fire reads and writes. Every field is a device address.
///
/// One struct rather than fourteen positional arguments, which is what
/// `dispatch_attention_flashinfer_decode_bf16` (`:490-503`) had. The C++ could
/// not do this without a header both sides included; Rust can.
#[derive(Clone, Copy, Debug, Default)]
pub struct Buffers {
    /// `[num_tokens, num_q_heads, head_dim]`, or one row broadcast — see
    /// `broadcast_q`.
    pub q: u64,
    /// The paged K cache.
    pub k_pages: u64,
    /// The paged V cache.
    pub v_pages: u64,
    /// `[num_tokens, num_q_heads, head_dim]`, written.
    pub o: u64,
    /// `[nnz_pages]`.
    pub kv_page_indices: u64,
    /// `[batch_size + 1]`.
    pub kv_page_indptr: u64,
    /// `[batch_size]`.
    pub kv_last_page_lens: u64,
    /// `[batch_size + 1]` QO row offsets. Prefill only; decode has one row per
    /// request and passes 0.
    pub qo_indptr: u64,
    /// Optional `[num_tokens, num_q_heads]` log-sum-exp output, or 0.
    pub lse: u64,
    /// The plan's int workspace. `Plan::int_upload` was copied here.
    pub int_buffer: u64,
    /// The plan's float workspace, where the split path stages partials.
    pub float_buffer: u64,
}

/// The split-path staging pointers a [`Fired::Split`] leaves behind, and
/// everything the fold that consumes them needs.
///
/// # Why this carries nine fields and not two
///
/// It carried `tmp_v` and `tmp_s` alone while `Fired::Split` was a panic. Now
/// that the fold exists, the question is where the OTHER seven values come
/// from — `o`, `lse`, the indptr, the row count, the head count, the head
/// dim — and the answer decides whether the seam is safe.
///
/// They are filled here, in [`make_decode_params`] and
/// [`make_prefill_params`], because this is where they are unambiguous: the
/// same function that redirects `params.o` to `tmp_v` is the one that knows
/// what `o` used to be. A call site in `bind/service.rs` also holds an `o`
/// and a head count, and would be filling them from a second reading of the
/// same request — two derivations of one fact, with nothing checking that
/// they agree. `prefill.cuh:4339-4342` makes the same choice for the same
/// reason: it saves `o` and `lse` into locals immediately before overwriting
/// the fields.
///
/// The two dispatches disagree about three of the nine, and the disagreement
/// is upstream's:
///
/// | | prefill (`prefill.cuh:4350-4352`) | decode (`decode.cuh:822-824`) |
/// |---|---|---|
/// | `indptr`      | `params.merge_indptr` | `params.o_indptr` |
/// | `max_seq_len` | `params.max_total_num_rows` | `params.paged_kv.batch_size` |
/// | `seq_len`     | `params.total_num_rows` | `nullptr` |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Partials {
    /// Partial outputs, `plan_info.v_offset` into the float workspace.
    ///
    /// **`params.o` points here too**, after the redirect. That is the whole
    /// mechanism: the attention kernel writes its per-chunk answers where the
    /// merge will read them.
    pub tmp_v: u64,
    /// Partial log-sum-exps, `plan_info.s_offset`. `params.lse` points here.
    pub tmp_s: u64,
    /// Where each merged row's partials start.
    ///
    /// `plan_info.merge_indptr_offset` for prefill, `plan_info.o_indptr_offset`
    /// for decode — see the table above.
    pub indptr: u64,
    /// The caller's real output, saved before `params.o` was redirected.
    pub o: u64,
    /// The caller's real log-sum-exp output, or 0. Saved likewise.
    pub lse: u64,
    /// Rows to fold: `plan_info.total_num_rows` for prefill, the request
    /// count for decode.
    pub max_seq_len: u32,
    /// A DEVICE `uint32_t*` overriding `max_seq_len`, or 0.
    ///
    /// Taken from `params.total_num_rows` verbatim rather than recomputed, so
    /// the fold folds exactly the rows the attention kernel wrote. See
    /// [`make_prefill_params`] for the one case where that is currently 0 and
    /// upstream's would not be.
    pub seq_len: u64,
    /// Query heads.
    pub num_heads: u32,
    /// 64, 128, 256 or 512.
    pub head_dim: u32,
}

impl Partials {
    /// The fold this split needs, as [`crate::fire::merge_states`]' job.
    ///
    /// A method rather than a `From`, because the direction is one-way and
    /// naming it `merge` is what makes the call site at each of the seven
    /// dispatches read as a sentence.
    #[must_use]
    pub fn merge(&self) -> crate::fire::merge_states::VarLen {
        crate::fire::merge_states::VarLen {
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

/// The score sink a capturing dispatch writes, `attention_score_capture.cuh`.
///
/// Separate from [`Buffers`] because it is the whole difference between the
/// capturing and non-capturing arms, and because a zero here is a refusal
/// ([`Decline::CaptureSinkMissing`]) rather than a default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capture {
    /// `float*` — the ragged score sink.
    pub score_out: u64,
    /// `const IdType*`, `batch + 1` entries.
    pub score_indptr: u64,
    /// Prefill only: the observation window in query rows. Decode ignores it,
    /// because a decode step has exactly one query row per request.
    pub score_window: u32,
}

/// The arbitrary mask a custom prefill takes,
/// `dispatch_attention_flashinfer_prefill_custom_bf16:1150-1155`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CustomMask {
    /// `const uint8_t*` — one bit per (qo_row, kv_pos), packed.
    pub mask: u64,
    /// `const int32_t*` — per-request bit offsets into `mask`.
    pub mask_indptr: u64,
}

/// `GetPtrFromBaseOffset`, `attention_flashinfer_common.cuh:174-177`:
/// `(base + offset_bytes)`.
///
/// A saturating add rather than a wrapping one. Upstream's is a pointer
/// arithmetic that would be UB on overflow; here an overflow can only come
/// from a corrupt plan, and saturating to `u64::MAX` produces an address the
/// device faults on immediately rather than one that aliases the workspace.
const fn offset_ptr(base: u64, off: i64) -> u64 {
    if off < 0 { base } else { base.saturating_add(off as u64) }
}

/// `sm_scale > 0 ? sm_scale : 1/sqrt(head_dim)`.
///
/// `attention_flashinfer_common.cuh:603-605` and
/// `attention_flashinfer.cu:735-737`, identically in both.
fn sm_scale_or_default(sm_scale: f32, head_dim: i32) -> f32 {
    if sm_scale > 0.0 { sm_scale } else { 1.0 / (head_dim as f32).sqrt() }
}

/// `paged_kv_t`'s guarded `__host__` constructor, `page.cuh:103-120`.
///
/// The three strides are computed here because that constructor is
/// `#ifndef __CUDACC_RTC__` and device code never runs one — its `// PIE:`
/// marker says *"Under the JIT this struct is filled by the Rust caller"*.
/// This is that caller.
///
/// `hnd_layout` is `QKVLayout::kHND`; false is `kNHD`. `page.cuh:118-119`:
///
/// ```text
/// stride_page = num_heads * page_size * head_dim
/// stride_n    = kHND ? head_dim            : num_heads * head_dim
/// stride_h    = kHND ? page_size * head_dim : head_dim
/// ```
#[allow(clippy::too_many_arguments)]
fn make_paged_kv(
    num_heads: u32,
    page_size: u32,
    head_dim: u32,
    batch_size: u32,
    hnd_layout: bool,
    k_data: u64,
    v_data: u64,
    indices: u64,
    indptr: u64,
    last_page_len: u64,
) -> PagedKv {
    PagedKv {
        page_size: UintFastdiv::new(page_size),
        num_heads,
        head_dim,
        batch_size,
        stride_page: num_heads.wrapping_mul(page_size).wrapping_mul(head_dim),
        stride_n: if hnd_layout { head_dim } else { num_heads.wrapping_mul(head_dim) },
        stride_h: if hnd_layout { page_size.wrapping_mul(head_dim) } else { head_dim },
        k_data,
        v_data,
        indices,
        indptr,
        last_page_len,
        // Left null, and `run_decode`'s comment at
        // `attention_flashinfer_common.cuh:614-619` is why it must stay null:
        // `PieScoreCapture` records `kv_idx` verbatim and the kernel derives
        // it from `rope_pos_offset` (`decode.cuh:541`), so a non-null value
        // would silently land every captured score at the wrong position.
        // The C++ asserted this at runtime on the capture path; here it is
        // structural, because nothing writes the field.
        rope_pos_offset: 0,
    }
}

/// Which decode variant a request selects, `dispatch_decode`
/// (`attention_flashinfer_common.cuh:697-722`).
///
/// **The order is load-bearing.** A windowed layer that also has a soft cap
/// takes the soft-cap arm, because the soft-cap test comes second and the
/// window arm is the fallthrough. Reordering these three `if`s is a silent
/// numerics change.
#[must_use]
pub fn decode_arm(
    full_attention_variant: bool,
    window_left: i32,
    logits_soft_cap: f32,
) -> DecodeArm {
    if full_attention_variant && window_left < 0 && logits_soft_cap <= 0.0 {
        return DecodeArm::Full;
    }
    if logits_soft_cap > 0.0 {
        return DecodeArm::Softcap;
    }
    DecodeArm::Window
}

/// Which capturing decode arm a request selects,
/// `dispatch_decode_capture` (`attention_flashinfer_common.cuh`).
///
/// Two arms, not three: the capture variants are instantiated over
/// `AttnScoreCaptureFull` and `AttnScoreCapture` only. `None` is the C++'s
/// `throw` — a soft cap or a window on a capturing dispatch names an
/// instantiation that was never built, and there is nothing to fall back to.
#[must_use]
pub fn decode_capture_arm(
    full_attention_variant: bool,
    window_left: i32,
    logits_soft_cap: f32,
) -> Option<DecodeArm> {
    if logits_soft_cap > 0.0 || window_left >= 0 {
        return None;
    }
    Some(if full_attention_variant { DecodeArm::CaptureFull } else { DecodeArm::CaptureWindow })
}

/// Which prefill variant a request selects, `prefill`
/// (`attention_flashinfer_common.cuh`).
///
/// **The asymmetry is upstream's and is kept.** The full-attention branch has
/// all four combinations of causal × soft-cap; the windowed branch has only
/// the causal ones, because a bidirectional windowed prefill is not
/// instantiated. A caller that asks for one lands on `CausalWindow`, exactly
/// as the C++ did — which is a numerics difference, not a fault, and is the
/// reason it is written out here rather than folded into a table.
#[must_use]
pub fn prefill_arm(full_attention_variant: bool, causal: bool, logits_soft_cap: f32) -> PrefillArm {
    if full_attention_variant {
        return match (causal, logits_soft_cap > 0.0) {
            (true, true) => PrefillArm::CausalFullSoftcap,
            (false, true) => PrefillArm::NoneFullSoftcap,
            (true, false) => PrefillArm::CausalFull,
            (false, false) => PrefillArm::NoneFull,
        };
    }
    if logits_soft_cap > 0.0 { PrefillArm::CausalSoftcap } else { PrefillArm::CausalWindow }
}

/// Which capturing prefill arm a request selects, `prefill_capture`.
///
/// [`decode_capture_arm`]'s counterpart, with the same `None`: soft cap or
/// window is an instantiation that does not exist.
#[must_use]
pub fn prefill_capture_arm(
    causal: bool,
    window_left: i32,
    logits_soft_cap: f32,
) -> Option<PrefillArm> {
    if logits_soft_cap > 0.0 || window_left >= 0 {
        return None;
    }
    Some(if causal { PrefillArm::CausalCapture } else { PrefillArm::NoneCapture })
}

/// Which custom-mask prefill arm a request selects, `prefill_custom`.
///
/// Two arms and no causal axis: the mask *is* the causality, so a custom
/// dispatch that also set `CAUSAL` would mask twice.
#[must_use]
pub fn prefill_custom_arm(logits_soft_cap: f32) -> PrefillArm {
    if logits_soft_cap > 0.0 { PrefillArm::CustomSoftcap } else { PrefillArm::Custom }
}

/// `run_decode`'s params filling, `attention_flashinfer_common.cuh:581-641`.
///
/// Returns the struct and the split-path staging pointers, which the C++
/// returned through two out-references (`tmp_v`, `tmp_s`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn make_decode_params(
    cache: &DecodePlanCache,
    bufs: &Buffers,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    broadcast_q: bool,
) -> (DecodeParams, Partials) {
    let info = &cache.plan_info;
    let mut p = DecodeParams {
        q: bufs.q,
        // `:583` — always null on this path. Fused rope is not wired.
        q_rope_offset: 0,
        paged_kv: make_paged_kv(
            cache.num_kv_heads as u32,
            cache.page_size as u32,
            cache.head_dim as u32,
            cache.num_requests as u32,
            cache.hnd_layout,
            bufs.k_pages,
            bufs.v_pages,
            bufs.kv_page_indices,
            bufs.kv_page_indptr,
            bufs.kv_last_page_lens,
        ),
        o: bufs.o,
        lse: bufs.lse,
        // `:586`.
        maybe_alibi_slopes: 0,
        num_qo_heads: cache.num_q_heads as u32,
        // `:588-590`. **Zero, not a stride**, when `broadcast_q`: one query row
        // is read by every token. That is how a single decoded token feeds a
        // batch, and a mirror that "fixed" the zero would read past the row.
        q_stride_n: if broadcast_q { 0 } else { cache.num_q_heads * cache.head_dim },
        q_stride_h: cache.head_dim,
        window_left,
        logits_soft_cap,
        sm_scale: sm_scale_or_default(sm_scale, cache.head_dim),
        // `:598-599`. Rope is not fused into FA2 here; it ran earlier as its
        // own kernel, so both reciprocals are 1 and the kernel's rope path is
        // an identity.
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        ..DecodeParams::default()
    };

    // `:632-633`. The decode int base is NOT the workspace base: several
    // layers' plans share one int buffer and `set_decode_plan_int_base`
    // (`attention_flashinfer.cu:215-217`) says where this one starts.
    let int_buf = bufs.int_buffer.saturating_add(cache.int_base_bytes as u64);
    p.request_indices = offset_ptr(int_buf, info.request_indices_offset);
    p.kv_tile_indices = offset_ptr(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = offset_ptr(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = offset_ptr(int_buf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;

    let mut partials = Partials::default();
    if info.split_kv {
        partials.tmp_v = offset_ptr(bufs.float_buffer, info.v_offset);
        partials.tmp_s = offset_ptr(bufs.float_buffer, info.s_offset);
        // `decode.cuh:809-812`. **The redirect, and it is the whole split
        // mechanism**: the attention kernel writes per-chunk partials to
        // `tmp_v`/`tmp_s` and `o`/`lse` are filled by the merge afterwards.
        // Without these two lines the kernel writes partial answers straight
        // into the caller's output and the merge folds a buffer nothing
        // staged.
        partials.o = p.o;
        partials.lse = p.lse;
        p.o = partials.tmp_v;
        p.lse = partials.tmp_s;
        // `decode.cuh:823`. Decode's indptr is `o_indptr` and its row count
        // is the batch size; there is no `merge_indptr` on this path because
        // a decode step has exactly one query row per request.
        partials.indptr = p.o_indptr;
        partials.max_seq_len = cache.num_requests as u32;
        // `decode.cuh:823` passes `nullptr`.
        partials.seq_len = 0;
        partials.num_heads = cache.num_q_heads as u32;
        partials.head_dim = cache.head_dim as u32;
        // `:648-651`. Only under graph capture: outside it the grid is exactly
        // the work list and every block is valid.
        if info.enable_cuda_graph {
            p.block_valid_mask = offset_ptr(int_buf, info.block_valid_mask_offset);
        }
    }
    (p, partials)
}

/// `make_prefill_params`, `attention_flashinfer.cu:693-775`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn make_prefill_params(
    cache: &PrefillPlanCache,
    bufs: &Buffers,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> (PrefillPagedParams, Partials) {
    let info = &cache.plan_info;
    let group =
        if cache.num_kv_heads > 0 { (cache.num_q_heads / cache.num_kv_heads) as u32 } else { 1 };
    let mut p = PrefillPagedParams {
        q: bufs.q,
        paged_kv: make_paged_kv(
            cache.num_kv_heads as u32,
            cache.page_size as u32,
            cache.head_dim as u32,
            cache.num_requests as u32,
            cache.hnd_layout,
            bufs.k_pages,
            bufs.v_pages,
            bufs.kv_page_indices,
            bufs.kv_page_indptr,
            bufs.kv_last_page_lens,
        ),
        // `:722` — null. The causal mask is a KernelTraits constant, not a
        // buffer; `maybe_custom_mask` is upstream's arbitrary-mask path and
        // pie does not use it.
        maybe_custom_mask: 0,
        q_indptr: bufs.qo_indptr,
        maybe_mask_indptr: 0,
        maybe_q_rope_offset: 0,
        o: bufs.o,
        lse: bufs.lse,
        maybe_alibi_slopes: 0,
        // `:728-729`. **24 bytes, and a computed magic** — see
        // [`UintFastdiv`]. The prefill kernel divides by the GQA group on
        // every row, which is why upstream carries a reciprocal rather than a
        // divisor.
        group_size: UintFastdiv::new(group),
        num_qo_heads: cache.num_q_heads as u32,
        // `:731`. Note there is no `broadcast_q` here: prefill always has a
        // real QO row per token, so the decode path's zero stride has no
        // prefill analogue.
        q_stride_n: cache.num_q_heads * cache.head_dim,
        q_stride_h: cache.head_dim,
        // `:733` — from the CACHE, not the argument. The window was fixed at
        // planning time because the split was sized against it.
        window_left: cache.window_left,
        logits_soft_cap,
        sm_scale: sm_scale_or_default(sm_scale, cache.head_dim),
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        ..PrefillPagedParams::default()
    };

    // `:742`. Prefill reads the workspace base directly — there is no prefill
    // analogue of `int_base_bytes`, because one prefill plan serves one fire.
    let int_buf = bufs.int_buffer;
    p.request_indices = offset_ptr(int_buf, info.request_indices_offset);
    p.qo_tile_indices = offset_ptr(int_buf, info.qo_tile_indices_offset);
    p.kv_tile_indices = offset_ptr(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = offset_ptr(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = offset_ptr(int_buf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;
    // `:753`. A VALUE, and the field below it is a POINTER — the names differ
    // by two characters and the types by eight bytes.
    p.max_total_num_rows = info.total_num_rows as u32;
    p.total_num_rows = 0;

    let mut partials = Partials::default();
    if info.split_kv {
        p.merge_indptr = offset_ptr(int_buf, info.merge_indptr_offset);
        partials.tmp_v = offset_ptr(bufs.float_buffer, info.v_offset);
        partials.tmp_s = offset_ptr(bufs.float_buffer, info.s_offset);
        // `prefill.cuh:4339-4342` — the redirect. See [`make_decode_params`]
        // for what it is for; the two are the same three lines.
        partials.o = p.o;
        partials.lse = p.lse;
        p.o = partials.tmp_v;
        p.lse = partials.tmp_s;
        // `prefill.cuh:4351`. Prefill folds by ROW rather than by request —
        // `merge_indptr` has `total_num_rows + 1` entries
        // (`plan/prefill.rs:124`) — which is why it has an indptr of its own
        // where decode reuses `o_indptr`.
        partials.indptr = p.merge_indptr;
        partials.max_seq_len = p.max_total_num_rows;
        // **Verbatim, not recomputed.** `prefill.cuh:4352` passes
        // `params.total_num_rows`, and the field is written 0 four lines
        // above. So the fold uses `max_total_num_rows`, which is exactly what
        // the attention kernel used, and the two cannot disagree.
        //
        // Under `enable_cuda_graph` upstream would have a real pointer here —
        // `plan/prefill.rs:414-416` allocates `total_num_rows_offset` only in
        // that mode, for a grid captured with a dummy row count. This driver
        // does not fill it, on either side, and reading it here would make
        // the merge fold a different row count from the kernel that produced
        // the partials. That gap is FA2's and predates the split; it is
        // recorded here because this is where a reader will look for it.
        partials.seq_len = p.total_num_rows;
        partials.num_heads = cache.num_q_heads as u32;
        partials.head_dim = cache.head_dim as u32;
        if info.enable_cuda_graph {
            p.block_valid_mask = offset_ptr(int_buf, info.block_valid_mask_offset);
        }
    }
    (p, partials)
}

/// Everything a decode launch needs, and nothing that needs a CUDA context.
///
/// [`decode`] returns this rather than firing, and that is deliberate rather
/// than unfinished: the lattice point and the params are the whole of what
/// this module can know, and the stream a launch is ordered on belongs to the
/// caller. Handing them back makes the launch one
/// [`kernels_cuda_new::x::fa2::decode`] call at the site that owns the
/// stream — and makes *this* testable without a GPU, which nothing that
/// called `cudaLaunchKernel` inline ever was.
///
/// **The rectangle is not here any more.** The grid, the block and the shared
/// allocation are derived from `at` by the routine that fires it, because
/// they are the kernel's arithmetic rather than the plan's; what this carries
/// is which point, which arm, and the two grid extents a plan supplies.
///
/// `P` is the `__grid_constant__` struct's type and defaults to
/// [`DecodeParams`]; [`decode_capture`] returns
/// `DecodeDispatch<DecodeScoreParams>`. A type parameter rather than a second
/// struct because the difference between the two is exactly the params type
/// and nothing else — same point, same arm axis, same derivation.
#[derive(Clone, Copy, Debug)]
pub struct DecodeDispatch<P = DecodeParams> {
    /// Which lattice point, which arm, and the grid's two extents.
    pub at: DecodePoint,
    /// The single `__grid_constant__` argument.
    pub params: P,
}

/// Everything a prefill launch needs. See [`DecodeDispatch`].
#[derive(Clone, Copy, Debug)]
pub struct PrefillDispatch<P = PrefillPagedParams> {
    /// Which lattice point, which arm, and the grid's two extents.
    pub at: PrefillPoint,
    /// The second of the kernel's two template arguments, by value.
    pub params: P,
}

/// The decode dispatch.
///
/// Replaces `dispatch_attention_flashinfer_decode_bf16`
/// (`attention_flashinfer.cu:490-522`) and everything it called.
#[allow(clippy::too_many_arguments)]
pub fn decode(
    cache: &DecodePlanCache,
    bufs: &Buffers,
    device: FaDevice,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    broadcast_q: bool,
) -> Fired<DecodeDispatch> {
    if !cache.valid {
        return Fired::Declined(Decline::Unplanned);
    }
    let arm = decode_arm(cache.full_attention_variant, window_left, logits_soft_cap);

    let (params, partials) =
        make_decode_params(cache, bufs, window_left, logits_soft_cap, sm_scale, broadcast_q);

    let ready =
        DecodeDispatch { at: decode_at(cache, device, arm, params.padded_batch_size), params };
    if cache.plan_info.split_kv { Fired::Split(ready, partials) } else { Fired::Whole(ready) }
}

/// The score-capturing decode dispatch.
///
/// Replaces `dispatch_attention_flashinfer_decode_capture_bf16`
/// (`attention_flashinfer.cu:537-607`).
///
/// The params are [`DecodeParams`] plus the sink — `PieScoreParams` derives
/// from `BatchDecodeParams` — so every field [`make_decode_params`] fills is
/// filled by the same call, and the difference is two pointers and the symbol.
/// **`logits_soft_cap` is forced to zero** in the struct, matching the C++,
/// which passed `0.f` after refusing a non-zero one: the capture arms are not
/// instantiated over the soft-cap variant, so a value the kernel could not
/// honour would be a lie in the params rather than an error.
#[allow(clippy::too_many_arguments)]
pub fn decode_capture(
    cache: &DecodePlanCache,
    bufs: &Buffers,
    capture: &Capture,
    device: FaDevice,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    broadcast_q: bool,
) -> Fired<DecodeDispatch<DecodeScoreParams>> {
    if !cache.valid {
        return Fired::Declined(Decline::Unplanned);
    }
    // `:546-549`, before the variant test, and in that order.
    if capture.score_out == 0 || capture.score_indptr == 0 {
        return Fired::Declined(Decline::CaptureSinkMissing);
    }
    let Some(arm) = decode_capture_arm(cache.full_attention_variant, window_left, logits_soft_cap)
    else {
        return Fired::Declined(Decline::CaptureVariantUnsupported);
    };

    let (base, partials) = make_decode_params(cache, bufs, window_left, 0.0, sm_scale, broadcast_q);
    let params = DecodeScoreParams {
        base,
        score_out: capture.score_out,
        score_indptr: capture.score_indptr,
    };

    let ready =
        DecodeDispatch { at: decode_at(cache, device, arm, params.base.padded_batch_size), params };
    if cache.plan_info.split_kv { Fired::Split(ready, partials) } else { Fired::Whole(ready) }
}

/// The lattice point a decode fire lands on, as the routine takes it.
///
/// Factored out of [`decode`] and [`decode_capture`] because the two differ
/// only in the arm, and a second copy of the GQA division is a second place
/// for it to be wrong.
///
/// This no longer resolves anything: whether the lattice holds the point, and
/// what geometry it has, are [`kernels_cuda_new::x::fa2`]'s and are answered
/// at the fire. What is left here is the reading of the PLAN — the head dim,
/// the GQA group and the two grid extents — which is this crate's vocabulary.
fn decode_at(
    cache: &DecodePlanCache,
    device: FaDevice,
    arm: DecodeArm,
    padded_batch_size: u32,
) -> DecodePoint {
    DecodePoint {
        head_dim: cache.head_dim as u32,
        group_size: if cache.num_kv_heads > 0 {
            (cache.num_q_heads / cache.num_kv_heads) as u32
        } else {
            1
        },
        arm,
        padded_batch_size,
        num_kv_heads: cache.num_kv_heads as u32,
        device,
    }
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
    if let Err(why) = prefill_plan_usable(cache) {
        return Fired::Declined(why);
    }

    let (params, partials) = make_prefill_params(cache, bufs, logits_soft_cap, sm_scale);

    let ready =
        PrefillDispatch { at: prefill_at(cache, device, arm, params.padded_batch_size), params };
    if cache.plan_info.split_kv { Fired::Split(ready, partials) } else { Fired::Whole(ready) }
}

/// The score-capturing prefill dispatch.
///
/// Replaces `dispatch_attention_flashinfer_prefill_capture_bf16`
/// (`attention_flashinfer.cu:837-934`).
///
/// As [`decode_capture`]: the params derive from [`PrefillPagedParams`], so
/// the base is filled by the same [`make_prefill_params`] and the soft cap is
/// forced to zero after the refusal.
#[allow(clippy::too_many_arguments)]
pub fn prefill_capture(
    cache: &PrefillPlanCache,
    bufs: &Buffers,
    capture: &Capture,
    device: FaDevice,
    causal: bool,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> Fired<PrefillDispatch<PrefillScoreParams>> {
    // `:849-856`. The window is part of the sink here and not part of the
    // variant: a zero window is a sink with no rows, which the kernel would
    // still index into.
    if capture.score_out == 0 || capture.score_indptr == 0 || capture.score_window == 0 {
        return Fired::Declined(Decline::CaptureSinkMissing);
    }
    let Some(arm) = prefill_capture_arm(causal, cache.window_left, logits_soft_cap) else {
        return Fired::Declined(Decline::CaptureVariantUnsupported);
    };
    if let Err(why) = prefill_plan_usable(cache) {
        return Fired::Declined(why);
    }

    let (base, partials) = make_prefill_params(cache, bufs, 0.0, sm_scale);
    let params = PrefillScoreParams {
        base,
        score_out: capture.score_out,
        score_indptr: capture.score_indptr,
        score_window: capture.score_window,
    };

    let ready = PrefillDispatch {
        at: prefill_at(cache, device, arm, params.base.padded_batch_size),
        params,
    };
    if cache.plan_info.split_kv { Fired::Split(ready, partials) } else { Fired::Whole(ready) }
}

/// The custom-mask prefill dispatch.
///
/// Replaces `dispatch_attention_flashinfer_prefill_custom_bf16`
/// (`attention_flashinfer.cu:1115-1224`).
///
/// **`window_left` is `-1` here and NOT the cache's** — `:1163` sets it
/// literally, because the mask states the visibility and a window on top of it
/// would mask twice. This is the one place [`make_prefill_params`]'s
/// cache-sourced window is overwritten, and it is overwritten after the call
/// rather than parameterised into it so that the deviation is visible.
#[allow(clippy::too_many_arguments)]
pub fn prefill_custom(
    cache: &PrefillPlanCache,
    bufs: &Buffers,
    mask: &CustomMask,
    device: FaDevice,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> Fired<PrefillDispatch> {
    let arm = prefill_custom_arm(logits_soft_cap);
    if let Err(why) = prefill_plan_usable(cache) {
        return Fired::Declined(why);
    }

    let (mut params, partials) = make_prefill_params(cache, bufs, logits_soft_cap, sm_scale);
    // `:1150-1155`, `:1163`.
    params.maybe_custom_mask = mask.mask;
    params.maybe_mask_indptr = mask.mask_indptr;
    params.window_left = -1;

    let ready =
        PrefillDispatch { at: prefill_at(cache, device, arm, params.padded_batch_size), params };
    if cache.plan_info.split_kv { Fired::Split(ready, partials) } else { Fired::Whole(ready) }
}

/// The two plan-validity refusals, in the order the C++ made them.
///
/// Shared by the three prefill entry points so that all three make them the
/// same way: `:780` tests `cache.valid` and `:783` tests `cache.use_sm90`, and
/// `dispatch_attention_flashinfer_prefill_custom_bf16:1132` tests both in one
/// `if`. Both read the PLAN, which is why they stayed here when the lattice
/// lookup and the geometry went down to [`kernels_cuda_new::x::fa2`].
fn prefill_plan_usable(cache: &PrefillPlanCache) -> Result<(), Decline> {
    if !cache.valid {
        return Err(Decline::Unplanned);
    }
    if cache.use_sm90 {
        return Err(Decline::Sm90Unported);
    }
    Ok(())
}

/// The lattice point a prefill fire lands on, as the routine takes it.
///
/// [`decode_at`]'s twin, and the same division of labour: `NUM_MMA_KV` is not
/// here, because it is derived from the shared-memory budget by
/// [`kernels_cuda_new::fa2::PrefillGeometry::derive`] and naming it twice is a
/// second place for it to be wrong.
fn prefill_at(
    cache: &PrefillPlanCache,
    device: FaDevice,
    arm: PrefillArm,
    padded_batch_size: u32,
) -> PrefillPoint {
    PrefillPoint {
        head_dim: cache.head_dim as u32,
        cta_tile_q: cache.cta_tile_q,
        arm,
        padded_batch_size,
        num_kv_heads: cache.num_kv_heads as u32,
        device,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Fired, Partials, decode_arm, decode_capture_arm, offset_ptr, prefill_arm,
        prefill_capture_arm, prefill_custom_arm,
    };
    use kernels_cuda_new::x::fa2::{DecodeArm, PrefillArm};

    /// The cascade's ORDER, which is the part that can be broken silently.
    ///
    /// A windowed layer with a soft cap takes the soft-cap arm, not the window
    /// arm — `attention_flashinfer_common.cuh:697-722` tests the cap second
    /// and falls through to the window. Written as a test because reordering
    /// the three `if`s compiles and changes the kernel.
    #[test]
    fn a_windowed_layer_with_a_soft_cap_takes_the_softcap_arm() {
        assert_eq!(decode_arm(true, -1, 0.0), DecodeArm::Full);
        assert_eq!(decode_arm(false, -1, 0.0), DecodeArm::Window);
        assert_eq!(decode_arm(true, 4096, 0.0), DecodeArm::Window);
        assert_eq!(decode_arm(true, 4096, 30.0), DecodeArm::Softcap);
        assert_eq!(decode_arm(true, -1, 30.0), DecodeArm::Softcap);
    }

    /// A corrupt plan produces an address that faults, not one that aliases.
    #[test]
    fn a_negative_offset_does_not_walk_backwards() {
        assert_eq!(offset_ptr(4096, -8), 4096);
        assert_eq!(offset_ptr(4096, 8), 4104);
        assert_eq!(offset_ptr(u64::MAX, 8), u64::MAX);
    }

    /// Prefill's windowed branch is CAUSAL ONLY, and that is upstream's.
    ///
    /// A bidirectional windowed prefill is not instantiated, so the request
    /// lands on `CausalWindow`. Written down because it is the one place a
    /// caller can ask for something and get something else, and because the
    /// ViT path (`tower/qwen3_vl`) asks for `causal = false` — it reaches
    /// `NoneFull` only because it also passes `full_attention_variant = false`
    /// with `window_left = -1`... which is exactly this fallthrough. The
    /// assertion below is what keeps that in view.
    #[test]
    fn a_bidirectional_windowed_prefill_falls_through_to_causal() {
        assert_eq!(prefill_arm(true, true, 0.0), PrefillArm::CausalFull);
        assert_eq!(prefill_arm(true, false, 0.0), PrefillArm::NoneFull);
        assert_eq!(prefill_arm(true, true, 30.0), PrefillArm::CausalFullSoftcap);
        assert_eq!(prefill_arm(true, false, 30.0), PrefillArm::NoneFullSoftcap);
        assert_eq!(prefill_arm(false, true, 30.0), PrefillArm::CausalSoftcap);
        assert_eq!(prefill_arm(false, true, 0.0), PrefillArm::CausalWindow);
        assert_eq!(prefill_arm(false, false, 0.0), PrefillArm::CausalWindow);
    }

    /// The capture arms compose with neither a soft cap nor a window.
    ///
    /// `None` rather than a nearest arm: there is no instantiation, so the
    /// only honest answers are this and a throw, and the C++ threw.
    #[test]
    fn capture_does_not_compose_with_softcap_or_window() {
        assert_eq!(decode_capture_arm(true, -1, 0.0), Some(DecodeArm::CaptureFull));
        assert_eq!(decode_capture_arm(false, -1, 0.0), Some(DecodeArm::CaptureWindow));
        assert_eq!(decode_capture_arm(true, -1, 30.0), None);
        assert_eq!(decode_capture_arm(true, 4096, 0.0), None);
        assert_eq!(prefill_capture_arm(true, -1, 0.0), Some(PrefillArm::CausalCapture));
        assert_eq!(prefill_capture_arm(false, -1, 0.0), Some(PrefillArm::NoneCapture));
        assert_eq!(prefill_capture_arm(true, -1, 30.0), None);
        assert_eq!(prefill_capture_arm(true, 0, 0.0), None);
        assert_eq!(prefill_custom_arm(0.0), PrefillArm::Custom);
        assert_eq!(prefill_custom_arm(30.0), PrefillArm::CustomSoftcap);
    }

    /// `Fired` distinguishes an answer from a half-answer at the type level,
    /// and `Partials::merge` carries every field the fold needs.
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
        let job = staging.merge();
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
// THE SIX ENTRY POINTS, MOVED OUT OF `bind::service`.
//
// They were `execution::RUST_SERVED` entries -- functions the generated C
// shim called, which called back into Rust. They are not that any more: the
// six symbols crossed to `x::attn`'s `contract!`s as driver ops, so the rows
// are gone, `emit_c_shim` emits nothing for them, and their only caller is
// `bind::dispatch`'s driver-op table.
//
// **They moved because `bind::service` is `bridge`-gated** (`f38d199c2`) and
// these must outlive the feature. The gate was the honest fix there -- that
// module is *"the consumer that makes the classification cost the C++ its
// body"*, which is `bridge`'s whole subject. These six are not: each is a
// thin resolution over the planner beside them and this file's own params
// filling, neither of which has ever needed the shim. Same shape
// `dequant_kv_cache_layer_to_bf16_active` took out of `fire/kv_paged.rs` and
// `attn_plan` took out of `dispatch_generated`: **the body moves to the
// surviving side.**
//
// `_ctx: &DispatchCtx` went with the move and is not missed -- it was unused
// in all six, and `DispatchCtx` is one of the `bridge`-gated types. The
// stream each needs is still a parameter, as it always was.
// ───────────────────────────────────────────────────────────────────────────
// ── FlashInfer FA2 — north star §5 step 7's six rows ────────────────────────
//
// `attention_flashinfer.cu` (1,258 lines) and `plan_lifecycle.cpp` (105) are
// DELETED, and these six functions are the whole of what stood behind them.
// The measured census that justified it: `__global__` 0, `__device__` 0, one
// real `<<<>>>` and that one was `device::attn_score_fold_heads`, ours and
// already rowed (`fire/attn_score.rs:279`).
//
// The split is deliberate and is `fa2-nvrtc`'s: `fire/flashinfer_fa2.rs`
// plans, `fire/flashinfer_fa2_dispatch.rs` decides a symbol, a grid and a
// filled params block, and only these functions -- which own a module and a
// stream -- launch. Everything above the launch is testable without a GPU,
// which nothing calling `cudaLaunchKernel` inline ever was.
//
// # Why every refusal here is a panic
//
// The C++ threw `std::runtime_error` / `std::invalid_argument` from exactly
// these points, and a generated dispatch arm returns `()`. `Decline` is a
// type one layer down, where it can be asserted about in a unit test; this is
// the boundary where it stops being one, and it stops being one loudly.
//
// # `Fired::Split` FOLDS, and this is the record of the pass where it did not
//
// A split fire leaves partials in `tmp_v`/`tmp_s` that
// `VariableLengthMergeStates` has to fold into `o`. That kernel came from
// `attention/cascade.cuh` compiled INTO `attention_flashinfer.cu`, and when
// that file was deleted it had no unit and no row -- so for one pass every
// arm below was a `panic!`, prefill was kept away from it by
// `plan_prefill` setting `disable_split_kv` unconditionally, and decode
// could still reach it. Firing anyway would have put un-merged partials in
// `o`: a silent wrong answer, which is the one outcome worse than a stop.
//
// It runs now. `kernels_cuda_new::families::cascade` compiles
// `PersistentVariableLengthMergeStatesKernel` out of the vendored
// `cascade.cuh` under NVRTC, and `fire/merge_states.rs` fires it. Every
// function below that can split does this, in this order:
//
//   1. `ffa2::fire_{decode,prefill}` -- the attention kernel, writing
//      partials, because `make_*_params` redirected `params.o`/`params.lse`
//      to `tmp_v`/`tmp_s` (`prefill.cuh:4339-4342`, `decode.cuh:809-812`).
//   2. `merge_states::variable_length` -- the fold, same stream, writing the
//      caller's real `o` and `lse` (`prefill.cuh:4350-4352`,
//      `decode.cuh:822-824`).
//
// **Two things the old note got wrong, recorded so the next reader does not
// inherit them.** Decode did NOT reach the panic only through the env-gated
// windowed planner: `DecodePlanCache::can_use_static_nonsplit` covers
// batches of 512 or fewer on cc >= 8, so any batch ABOVE 512 took the real
// planner and could split. And `disable_split_kv` is a PREFILL flag, so
// flipping it was never going to be the whole fix -- the decode arms had to
// fold too.

use std::ffi::c_void;

use kernels_cuda_new::x::KvLayer;

use super::flashinfer_fa2 as ffa2;
use super::merge_states;
use crate::bind::abi::{AttentionWorkspaceView, KvCacheLayerView};

/// The workspace addresses every FA2 fire reads, widened once.
fn fa2_buffers(
    q: *const c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    o: *mut c_void,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    qo_indptr: *const u32,
    lse: *mut f32,
    workspace: AttentionWorkspaceView,
) -> Buffers {
    Buffers {
        q: q as u64,
        k_pages: k_pages as u64,
        v_pages: v_pages as u64,
        o: o as u64,
        kv_page_indices: kv_page_indices as u64,
        kv_page_indptr: kv_page_indptr as u64,
        kv_last_page_lens: kv_last_page_lens as u64,
        qo_indptr: qo_indptr as u64,
        lse: lse as u64,
        int_buffer: workspace.int_buffer as u64,
        float_buffer: workspace.float_buffer as u64,
    }
}

/// `dispatch_attention_flashinfer_decode`, `attention_flashinfer.cu:660-684`.
///
/// Two statements, in the C++'s order: dequantise the layer's active pages
/// into `k_bf16_pages`/`v_bf16_pages`, then fire FA2 over those. The KV width
/// axis is why -- `KvWidth::BF16` is the only width the lattice instantiates,
/// so every scheme is widened before FA2 sees a page.
///
/// # Panics
///
/// If the plan is empty (`:504-508`'s `throw`), if the head dim or GQA group
/// has no unit, or if the plan splits -- see this section's banner.
///
/// # Safety
///
/// `cache` is a live [`crate::fire::flashinfer_fa2::DecodePlanCache`]; every
/// other pointer is a device address the caller keeps live across the launch;
/// `stream` is the fire's stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_decode(
    cache: *const c_void,
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: the caller's contract -- `bind::DecodePlan::as_ptr` is the only
    // producer of this pointer and it hands out its own boxed cache.
    let plan = unsafe { &*cache.cast::<ffa2::DecodePlanCache>() };

    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged; `:675`.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(stream) };
        let _ = kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
            &ctx,
            &l,
            kv_page_indices_d,
            plan.num_pages_in_batch,
        );
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        core::ptr::null(),
        lse_out,
        workspace,
    );
    let fired = decode(
        plan,
        &bufs,
        ffa2::fa_device(),
        window_left,
        logits_soft_cap,
        sm_scale,
        // `attention_flashinfer.hpp:136`'s default; the outer dispatch never
        // passed it.
        false,
    );
    let (mut dispatch, partials) = match fired {
        Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        Fired::Split(d, split) => (d, Some(split)),
        Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_decode declined: {why}")
        }
    };
    // SAFETY: the caller's contract, plus the plan's own: `int_upload` was
    // carved against `workspace.int_bytes` by the planner that filled it.
    unsafe {
        ffa2::fire_decode(
            &mut dispatch,
            ffa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_decode: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `decode.cuh:822-824` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_decode");
    }
}

/// `dispatch_attention_flashinfer_decode_capture`, `:631-658`.
///
/// [`attn_dispatch_attention_flashinfer_decode`] writing the pre-softmax
/// logits to a ragged sink as it goes. The params block is
/// [`kernels_cuda_new::fa2::params::DecodeScoreParams`] rather than
/// `DecodeParams`, which is why this is a separate function and not a flag.
///
/// The C++ threw on a null sink BEFORE choosing a variant, and so does the
/// arm helper: [`crate::fire::flashinfer_fa2_dispatch::Decline::CaptureSinkMissing`].
///
/// The post-kernels (`attn::attn_score_normalize`, `attn::attn_score_fold_heads`)
/// are NOT fired here and were not fired by the C++ either -- they belong to
/// `fire/attn_score.rs`' `LayerScoreCapture::publish`, on this stream,
/// immediately after this returns.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_decode`], plus: a soft cap, a
/// window, or a null score sink, none of which compose with capture.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`]; `score_out` addresses
/// `score_indptr[batch]` floats and `score_indptr` addresses `batch + 1`
/// `i32`s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_decode_capture(
    cache: *const c_void,
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    score_out: *mut f32,
    score_indptr_d: *const i32,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: as above.
    let plan = unsafe { &*cache.cast::<ffa2::DecodePlanCache>() };

    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged; `:648`.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(stream) };
        let _ = kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
            &ctx,
            &l,
            kv_page_indices_d,
            plan.num_pages_in_batch,
        );
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        core::ptr::null(),
        lse_out,
        workspace,
    );
    let capture = Capture {
        score_out: score_out as u64,
        score_indptr: score_indptr_d as u64,
        // A decode step has exactly one query row per request, so there is no
        // window to observe. The C++ capture params for decode carry no
        // `score_window` field at all -- see `DecodeScoreParams`.
        score_window: 0,
    };
    let fired = decode_capture(
        plan,
        &bufs,
        &capture,
        ffa2::fa_device(),
        window_left,
        logits_soft_cap,
        sm_scale,
        false,
    );
    let (mut dispatch, partials) = match fired {
        Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        Fired::Split(d, split) => (d, Some(split)),
        Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_decode_capture declined: {why}")
        }
    };
    // SAFETY: as above.
    unsafe {
        ffa2::fire_decode(
            &mut dispatch,
            ffa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_decode_capture: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `decode.cuh:822-824` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_decode_capture");
    }
}

/// `dispatch_attention_flashinfer_prefill_bf16`, `:775-810`.
///
/// The one FA2 row whose KV comes in ALREADY bf16: the fire states `k_pages`
/// and `v_pages` rather than a [`KvCacheLayerView`], so there is no dequant
/// here and there was none in the C++ either.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_decode`], plus
/// [`crate::fire::flashinfer_fa2_dispatch::Decline::Sm90Unported`] if the
/// plan ever names the Hopper route. It cannot today --
/// `fire::flashinfer_fa2::plan_prefill` writes `use_sm90 = false` -- and the
/// refusal is kept so that wiring an SM90 family is one conditional and not
/// an audit.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_prefill_bf16(
    cache: *const c_void,
    q: *const c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: `bind::PrefillPlan::as_ptr` is the only producer.
    let plan = unsafe { &*cache.cast::<ffa2::PrefillPlanCache>() };
    let bufs = fa2_buffers(
        q,
        k_pages,
        v_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    // `:786-790`. The arm reads the plan's own variant and mask flags, which
    // is what lets one row serve a causal decoder layer and a bidirectional
    // ViT: `tower/qwen3_vl` plans with `causal_mask: false` and fires this.
    let arm = prefill_arm(plan.full_attention_variant, plan.causal_mask, logits_soft_cap);
    let fired = prefill(plan, &bufs, ffa2::fa_device(), arm, logits_soft_cap, sm_scale);
    let (mut dispatch, partials) = match fired {
        Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        Fired::Split(d, split) => (d, Some(split)),
        Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_prefill_bf16 declined: {why}")
        }
    };
    // SAFETY: as above.
    unsafe {
        ffa2::fire_prefill(
            &mut dispatch,
            ffa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_prefill_bf16: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_prefill_bf16");
    }
}

/// `dispatch_attention_flashinfer_prefill_capture_bf16`, `:1255-1258` onwards.
///
/// [`attn_dispatch_attention_flashinfer_prefill_bf16`] with the score sink and
/// the observation window, on
/// [`kernels_cuda_new::fa2::params::PrefillScoreParams`].
///
/// `folded_out` is bound by the row and **not read here**: folding is
/// `attn::attn_score_fold_heads`, a separate row fired by
/// `fire/attn_score.rs`' `LayerPrefillScoreCapture::publish` after this
/// returns. It stays in the signature because the row states it and because
/// dropping it would make the operand list disagree with `table/attn.rs`.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_prefill_bf16`], plus a soft cap, a
/// window, a null sink, or a zero window -- the C++'s four `throw`s.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode_capture`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_prefill_capture_bf16(
    cache: *const c_void,
    q: *const c_void,
    k_pages: *mut c_void,
    v_pages: *mut c_void,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    score_out: *mut f32,
    folded_out: *mut f32,
    score_indptr_d: *const i32,
    window: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    let _ = folded_out;
    // SAFETY: as above.
    let plan = unsafe { &*cache.cast::<ffa2::PrefillPlanCache>() };
    let bufs = fa2_buffers(
        q,
        k_pages,
        v_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    let capture = Capture {
        score_out: score_out as u64,
        score_indptr: score_indptr_d as u64,
        score_window: window.max(0) as u32,
    };
    let fired = prefill_capture(
        plan,
        &bufs,
        &capture,
        ffa2::fa_device(),
        plan.causal_mask,
        logits_soft_cap,
        sm_scale,
    );
    let (mut dispatch, partials) = match fired {
        Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        Fired::Split(d, split) => (d, Some(split)),
        Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_prefill_capture_bf16 declined: {why}")
        }
    };
    // SAFETY: as above.
    unsafe {
        ffa2::fire_prefill(
            &mut dispatch,
            ffa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| {
        panic!("attn::dispatch_attention_flashinfer_prefill_capture_bf16: {why}")
    });

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_prefill_capture_bf16");
    }
}

/// `dispatch_attention_flashinfer_prefill_custom`, `:1225-1252`.
///
/// The arbitrary-mask prefill: the fire supplies a packed bit per
/// `(qo_row, kv_pos)` and the kernel reads it instead of deriving causality.
/// Dequantises like decode -- it takes a [`KvCacheLayerView`], not raw pages
/// -- with `num_pages_in_batch` read off the plan's own KV indptr tail rather
/// than off a device pointer, exactly as `:1244` did.
///
/// `window_left` is **not** a parameter and is not read from the plan:
/// `:1163` writes `params.window_left = -1` literally, because a custom mask
/// already says everything a window would.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_prefill_bf16`]. The C++'s
/// *"custom prefill dispatch requires a prepared non-SM90 plan"* is
/// `Decline::Unplanned` / `Decline::Sm90Unported` here.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`]; `mask_d` addresses the
/// packed bits `mask_indptr_d` indexes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_dispatch_attention_flashinfer_prefill_custom(
    cache: *const c_void,
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    mask_d: *const u8,
    mask_indptr_d: *const i32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    // SAFETY: as above.
    let plan = unsafe { &*cache.cast::<ffa2::PrefillPlanCache>() };

    // `:1244`, whole: the page count comes off the plan's widened KV indptr,
    // because the device copy cannot be read from the host.
    let num_pages_in_batch = if plan.num_requests > 0 {
        plan.kv_h_buf.get(plan.num_requests as usize).copied().unwrap_or(0)
    } else {
        0
    };
    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(stream) };
        let _ = kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
            &ctx,
            &l,
            kv_page_indices_d,
            num_pages_in_batch,
        );
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    let mask = CustomMask { mask: mask_d as u64, mask_indptr: mask_indptr_d as u64 };
    let fired = prefill_custom(plan, &bufs, &mask, ffa2::fa_device(), logits_soft_cap, sm_scale);
    let (mut dispatch, partials) = match fired {
        Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        Fired::Split(d, split) => (d, Some(split)),
        Fired::Declined(why) => {
            panic!("attn::dispatch_attention_flashinfer_prefill_custom declined: {why}")
        }
    };
    // SAFETY: as above.
    unsafe {
        ffa2::fire_prefill(
            &mut dispatch,
            ffa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::dispatch_attention_flashinfer_prefill_custom: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::dispatch_attention_flashinfer_prefill_custom");
    }
}

/// `attention_flashinfer_prefill`, `:1077-1113` — the PLANLESS prefill.
///
/// `Prepare::FireWide` with `whole = true`: no cache crosses, so this plans
/// into a cache of its own and throws it away. The C++ did the same with a
/// function-local `PrefillPlanInfo` and two `std::vector<IdType>`; the only
/// difference here is that the vectors live on a `PrefillPlanCache` that is
/// dropped at the end of the call, which costs one allocation per fire and
/// buys sharing every line of the planned path.
///
/// `:1063-1067` fixes three flags this path never varies:
/// `enable_cuda_graph = false`, `full_attention_variant = false`,
/// `causal_mask = true`. So the arm is always
/// `prefill_arm(false, true, soft_cap)` -- `CausalSoftcap` or `CausalWindow`.
///
/// # Panics
///
/// As [`attn_dispatch_attention_flashinfer_prefill_bf16`]. `num_requests <= 0`
/// is `Decline::NoRequests` and not a silent return, because the C++ reached
/// `PrefillPlan` with it and `PrefillPlan` failed.
///
/// # Safety
///
/// As [`attn_dispatch_attention_flashinfer_decode`], plus: `qo_indptr_h` and
/// `kv_page_indptr_h` address `num_requests + 1` readable HOST `u32`s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attn_attention_flashinfer_prefill(
    q: *const c_void,
    kv_layer: KvCacheLayerView,
    o: *mut c_void,
    qo_indptr_d: *const u32,
    kv_page_indices_d: *const u32,
    kv_page_indptr_d: *const u32,
    kv_last_page_lens_d: *const u32,
    qo_indptr_h: *const u32,
    kv_page_indptr_h: *const u32,
    total_tokens: i32,
    num_requests: i32,
    num_q_heads: i32,
    workspace: AttentionWorkspaceView,
    stream: *mut c_void,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    lse_out: *mut f32,
) {
    if num_requests <= 0 {
        panic!("attn::attention_flashinfer_prefill declined: empty batch");
    }
    let n = num_requests as usize + 1;
    // SAFETY: the caller's contract -- both are host CSRs of `num_requests + 1`
    // entries, which is what `Prepare::FireWide` publishes.
    let (qo_h, kv_h) = unsafe {
        (
            core::slice::from_raw_parts(qo_indptr_h, n),
            core::slice::from_raw_parts(kv_page_indptr_h, n),
        )
    };

    // `:1098`.
    let num_pages_in_batch = kv_h[num_requests as usize] as i32;
    // The dequant prelude, moved. A layer whose dtype `KvDType` does not
    // name skips the prelude and the attention below still runs — which is
    // the shape the `Declined` it used to return already had, because every
    // one of these four call sites consumed that return with `let _ =`.
    if let Ok(l) = KvLayer::try_from(&kv_layer) {
        // SAFETY: forwarded unchanged.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(stream) };
        let _ = kernels_cuda_new::x::attn::kv_paged::dequant_kv_cache_layer_to_bf16_active(
            &ctx,
            &l,
            kv_page_indices_d,
            num_pages_in_batch,
        );
    }

    let mut plan = ffa2::PrefillPlanCache::new();
    let device = ffa2::plan_device();
    let planned = ffa2::plan_prefill(
        &mut plan,
        qo_h,
        kv_h,
        total_tokens,
        num_requests,
        num_q_heads,
        kv_layer.num_kv_heads,
        kv_layer.head_dim,
        kv_layer.page_size,
        kernels_cuda_new::plan::Workspace {
            float_bytes: workspace.float_bytes,
            int_bytes: workspace.int_bytes,
        },
        &device,
        // `:1000`.
        false,
        window_left,
        // `:1066-1067`.
        false,
        kv_layer.hnd_layout,
        true,
        false,
        false,
    );
    if let ffa2::Planned::Declined(why) = planned {
        panic!("attn::attention_flashinfer_prefill declined: {why}");
    }

    let bufs = fa2_buffers(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        qo_indptr_d,
        lse_out,
        workspace,
    );
    let arm = prefill_arm(false, true, logits_soft_cap);
    let fired = prefill(&plan, &bufs, ffa2::fa_device(), arm, logits_soft_cap, sm_scale);
    let (mut dispatch, partials) = match fired {
        Fired::Whole(d) => (d, None),
        // The plan split KV. The fire writes per-chunk partials --
        // `make_*_params` pointed `params.o`/`params.lse` at them -- and
        // the fold after the launch below turns them into the caller's
        // `o`. Both are on this stream, in this order.
        Fired::Split(d, split) => (d, Some(split)),
        Fired::Declined(why) => {
            panic!("attn::attention_flashinfer_prefill declined: {why}")
        }
    };
    // SAFETY: as above. `plan` outlives the H2D because the copy is issued
    // from a pageable source, which `cudaMemcpyAsync` stages synchronously --
    // see `fire::flashinfer_fa2::upload_int_plan`'s note. That is what makes a
    // function-local plan legal here and it is the reason the note exists.
    unsafe {
        ffa2::fire_prefill(
            &mut dispatch,
            ffa2::PlanUpload {
                bytes: &plan.int_upload,
                int_buffer: workspace.int_buffer as u64,
                int_base_bytes: plan.int_base_bytes,
            },
            stream,
        )
    }
    .unwrap_or_else(|why| panic!("attn::attention_flashinfer_prefill: {why}"));

    if let Some(split) = partials {
        // SAFETY: `split` names the plan's own float workspace and the
        // `o`/`lse` this call was handed; the stream is the caller's, as
        // above. `prefill.cuh:4350-4352` fires exactly this, in exactly this position.
        unsafe { merge_states::variable_length(split.merge(), stream) }
            .expect_launched("attn::attention_flashinfer_prefill");
    }
}

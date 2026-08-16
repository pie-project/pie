//! The FlashInfer FA2 host program's plan half, in Rust.
//!
//! North-star §5 step 7's three caches and their factories.
//! `driver-cuda/csrc/attn/attention_flashinfer.cu` calls itself a host program
//! in its own header — *"What it holds is a HOST PROGRAM: six plan factories,
//! two plan-cache lifetimes, and four dispatches"* — and the census agrees:
//! `__global__` 0, `__device__` 0, and one launch, which is
//! `pie::attn_score_fold_heads`, ours and already rowed. This module is the
//! planning half of that program; [`crate::fire::kv_paged`] holds the dequant
//! switch its dispatches open with, and the launches themselves resolve
//! against [`crate::attn::fa2`]'s 56 roots.
//!
//! # What is here and what is deliberately not
//!
//! Here: the three caches as Rust structs, the two real planner factories over
//! [`crate::attn::plan`], the static non-split decode short-circuit with
//! both of its environment gates, the geometry gate that turns an unsupported
//! (head dim, `CTA_TILE_Q`) pair into a named refusal, and — north star §5
//! step 7's last two seams — the plan H2D ([`upload_int_plan`]) and the fire
//! ([`fire_decode`], [`fire_prefill`]).
//!
//! `attention_flashinfer.cu` and `plan_lifecycle.cpp` **are deleted**, and
//! `build.rs`'s last `.cuda(true)` with them. The params filling and the arm
//! cascades are [`crate::fire::flashinfer_fa2_dispatch`]; the six table rows
//! that used to reach the C++ through `pie_k_attn_*` now reach
//! [`crate::bind::service`] through
//! [`crate::execution::RUST_SERVED`].
//!
//! Not here: the SM90 prefill launcher. [`PrefillPlanCache::sm90_plan`] is
//! still planned and still recorded, and the dispatch refuses with
//! `Decline::Sm90Unported` rather than firing an FA2 symbol at it — see
//! §44.7, quoted below.
//!
//! # The measurements this file inherits
//!
//! These came from `attention_flashinfer.cu` and are the reason the code below
//! is shaped the way it is. A port that consumes a measurement is a regression
//! even if it compiles.
//!
//! * **The gemma-4 planner regression.** Re-running FlashInfer's full planner
//!   per decode batch was a hundredfold cost, and
//!   [`plan_static_nonsplit_decode`] exists to skip it. It is legal only
//!   because the work estimator has already forced `split_kv` off for the TP1
//!   latency shapes: *"the schedule is independent of KV lengths, so avoid
//!   rerunning the full FlashInfer planner for every decode batch"*
//!   (`attention_flashinfer.cu:105-115`).
//! * **The roofline note that bounds it** (`:241-242`): *"the static plan is
//!   unsplit by construction, which is what leaves a sliding layer at
//!   batch\*kv_heads CTAs (8 on 148 SMs for gemma-4) and ~50x off its
//!   bandwidth roofline."* This is why a windowed layer takes the real
//!   planner. A port that dropped the windowed branch would be fully correct
//!   and 50x slower on exactly one layer type, which is the worst shape a
//!   regression can have.
//! * **The unbounded-split cost the window hint existed to avoid.**
//!   `attention_flashinfer_common.cuh:250-260` records it: a sliding layer's
//!   split is bounded by the window and is cheap, *"while an unbounded one at
//!   1k context took a 256-token generation from 22 s to over 2400 s."* The
//!   C++ carried that fact into upstream's estimator through a `thread_local`
//!   `decode_window_hint()`, because the estimator is upstream's code and
//!   could not take another argument. **That side channel is gone and the
//!   fact is not**: [`plan_decode`] makes the same branch at the call site,
//!   before choosing a planner, which is where a branch on a caller's flag
//!   belongs.
//! * **`head_dim_supports_cascade_merge`'s `{64, 128, 256, 512}`** (`:376`,
//!   `:985`) is **upstream's** set, and it agrees with
//!   [`crate::attn::fa2::HEAD_DIMS`] by shared origin rather
//!   than by construction. Two facts that happen to match; whoever changes one
//!   says which.
//! * **The arch coverage this deletion RECOVERS, and it is worth naming**
//!   because the measurement it recovers from was a regression.
//!   `attention_flashinfer.cu`'s header said:
//!
//!   > `build.rs` gives the `pie_attn_flashinfer` unit `-gencode
//!   > arch=compute_89,code=sm_89`, copying the towers beside it, because
//!   > sm_89 is the box this tree is developed on. On an sm_90 part the three
//!   > post-kernels below would fail to launch with *"no kernel image is
//!   > available for execution on the device"*. That is a REGRESSION IN
//!   > COVERAGE against the archive build, which reads its arch list from
//!   > CMake.
//!
//!   NVRTC compiles for the loaded device, so the FA2 lattice has no
//!   `gencode` list to be narrow. The regression is closed by the deletion
//!   rather than by a fix, and it is recorded here because *"it stopped being
//!   true"* and *"it was never true"* are different claims.
//! * **§44.7's rule, which the arch note above is an instance of, and which
//!   still binds every line below**: *every sm_90 claim in this migration is
//!   argued from the call graph and none from a run.* Nothing here has been
//!   run on Hopper. [`PrefillPlanCache::sm90_plan`] is planned and refused at
//!   the fire for exactly that reason.

use std::ffi::c_void;
use std::sync::OnceLock;

use super::geometry::Device as FaDevice;
use crate::attn::plan::info::{
    DecodePlanInfo, PrefillPlanInfo, PrefillPlanSm90Info,
};
use crate::attn::plan::{self, Device, Workspace};
use crate::attn::fa2 as lattice;

use kernels::Refusal;

use super::dispatch::PrefillDispatch;

/// Whether a plan was built, and by which planner.
///
/// `#[must_use]` for `fire/gemv.rs`' reason, which is the rule for every
/// refusal in this tree: a function that can say no must not be callable in a
/// way that spells *"it declined"* the same as *"it ran"*. The two `Planned`
/// arms are distinguished because they are not interchangeable — see
/// [`DecodePlanCache::page_count_independent`].
#[must_use]
pub enum Planned {
    /// FlashInfer's own planner ran and the cache holds its descriptor.
    Full,
    /// [`plan_static_nonsplit_decode`] ran instead. The descriptor is
    /// `request_indices[r] = r`, `kv_tile_indices = 0`, `o_indptr[r] = r`,
    /// `split_kv = false`.
    StaticNonsplit,
    /// Nothing was planned, and the reason.
    Declined(Decline),
}

/// Every way planning refuses.
///
/// Each arm names the fact that produced it, so a caller meeting one meets a
/// statement rather than a crash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// `head_dim` is not one of `{64, 128, 256, 512}`.
    ///
    /// `attention_flashinfer.cu:236-238` checked this up front rather than in
    /// the dispatch switch, and its comment says why: *"the static non-split
    /// plan short-circuits past the dispatch entirely, so an unsupported
    /// head_dim would otherwise be reported as a valid plan and only fail
    /// later inside the kernel launch."* The check keeps that position.
    HeadDim {
        /// The head dim asked for.
        head_dim: u32,
    },
    /// The lattice holds no unit for this (head dim, `CTA_TILE_Q`) pair.
    ///
    /// # The one pair this can actually name today
    ///
    /// **Head dim 256 with `CTA_TILE_Q` 128**, and it is upstream's own
    /// exclusion rather than a gap in the lattice.
    /// `KernelTraits::IsInvalid()` (`prefill.cuh:221-232`) has the clause
    ///
    /// ```text
    /// NUM_MMA_Q * (8 * NUM_MMA_D_VO_TILE + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256
    /// ```
    ///
    /// and at head dim 256 (`NUM_MMA_D_VO` 16) with `CTA_TILE_Q` 128
    /// (`NUM_MMA_Q` 2) the left side is `2 * (128 + 8 * NUM_MMA_KV)`, which is
    /// at least 256 **for every `NUM_MMA_KV` including zero**. There is no
    /// valid instantiation, so `x::fa2` names no root for it.
    ///
    /// # And the archive did not hit it — the guard is one line of arithmetic
    ///
    /// An earlier pass flagged this as a latent `FLASHINFER_ERROR` in the
    /// archive. **That was wrong, and the correction belongs here rather than
    /// in a report nobody reads**:
    /// [`crate::attn::plan::arith::fa2_determine_cta_tile_q`] is
    ///
    /// ```text
    /// arith.rs:95   if avg_packed_qo_len > 64 && head_dim < 256 { 128 }
    /// ```
    ///
    /// — `head_dim < 256`, so 128 is unreachable at 256 and above, and
    /// `arith.rs:186` pins it: `fa2_determine_cta_tile_q(65, 256, 8) == 64`.
    /// The pair cannot arise from the chooser.
    ///
    /// The refusal stays anyway, because `cta_tile_q` is a *parameter* here
    /// and a caller may compute it some other way. It is cheaper than a
    /// launch that fails inside NVRTC with a static assertion.
    HeadDimTile {
        /// The head dim asked for.
        head_dim: u32,
        /// The `CTA_TILE_Q` asked for.
        cta_tile_q: u32,
    },
    /// The batch is empty. `num_requests <= 0`.
    NoRequests,
    /// [`crate::attn::plan`] refused, and which planner did.
    ///
    /// The planner's own [`plan::Error`] is **not** carried, and that is a
    /// real loss rather than a simplification: `Error::WorkspaceOverflow`
    /// names the array that did not fit and `Error::IndptrTooShort` names the
    /// length, and both are gone here. It is this way because `Decline` is
    /// `Copy` and `plan::Error` is not, and because every other refusal in
    /// this module is a fact about the request rather than about the
    /// workspace. **If a workspace overflow ever needs diagnosing from this
    /// path, this arm is what has to grow** — a `Box<plan::Error>` and the
    /// loss of `Copy`.
    /// The planner refused, naming which one.
    Planner(&'static str),
    /// The int workspace cannot hold this plan's descriptor.
    ///
    /// `plan_static_nonsplit_decode` (`attention_flashinfer.cu:173-176`) threw
    /// *"flashinfer decode static plan: attention int workspace too small"*.
    /// The static planner carves the descriptor itself rather than going
    /// through [`crate::attn::plan`]'s allocator, so this is the one
    /// overflow this module can name precisely — and it does, where
    /// [`Decline::Planner`] cannot.
    WorkspaceTooSmall {
        /// Bytes the plan needs.
        needed: usize,
        /// Bytes the caller supplied.
        have: usize,
    },
    /// A score-capturing prefill plan was asked for with a sliding window.
    ///
    /// `attention_flashinfer.cu:299-311`, transcribed whole because the
    /// argument is not obvious: FlashInfer applies `LogitsMask` **after**
    /// `LogitsTransform`, so a windowed variant's mask runs on logits the
    /// capture has already written out. The captured tensor would hold scores
    /// for positions the kernel then discards — not a scaled answer but a
    /// different one, and silently. The C++ threw; this refuses.
    ScoreCaptureWindow {
        /// The window the caller asked for.
        window_left: i32,
    },
}

impl core::fmt::Display for Decline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HeadDim { head_dim } => {
                write!(
                    f,
                    "flashinfer fa2: unsupported head_dim {head_dim}; the lattice holds {{64, 128, 256, 512}}"
                )
            }
            Self::HeadDimTile { head_dim, cta_tile_q } => write!(
                f,
                "flashinfer fa2 prefill: head_dim {head_dim} with CTA_TILE_Q {cta_tile_q} has no \
                 valid KernelTraits -- `IsInvalid()` (prefill.cuh:221-232) is true for every \
                 NUM_MMA_KV, so no unit exists and none can",
            ),
            Self::NoRequests => write!(f, "flashinfer fa2: empty batch"),
            Self::Planner(which) => {
                write!(f, "flashinfer fa2: the {which} planner refused")
            }
            Self::WorkspaceTooSmall { needed, have } => write!(
                f,
                "flashinfer decode static plan: attention int workspace too small -- \
                 {needed} bytes needed, {have} granted",
            ),
            Self::ScoreCaptureWindow { window_left } => write!(
                f,
                "flashinfer prefill plan: score capture is not available with a sliding window \
                 (window_left {window_left}) -- LogitsMask runs after LogitsTransform, so the \
                 captured scores would include positions the kernel discards",
            ),
        }
    }
}

/// The head dims the FA2 lattice is instantiated over.
///
/// `kernels.def`'s list, and upstream's `head_dim_supports_cascade_merge` set.
/// Stated here as well as in [`crate::attn::fa2::HEAD_DIMS`]
/// because this is the *gate* and that is the *lattice*: they must agree, and
/// [`the_gate_and_the_lattice_agree`] is what makes them.
///
/// # `kernels.def` HAS NO READER LEFT, so the values are carried and not cited
///
/// That file states its own consumers — *"Consumed twice … C++ … CMake"* —
/// and both are gone. `csrc/CMakeLists.txt` is DELETED, and the C++ side is
/// `kernels_manifest.hpp`, whose includers are down to one: a `#include` line
/// inside `driver-cuda/tests/hf_config_dump/generate.py`'s template, whose
/// output `.cpp` is not on disk and has no build. The other was
/// `attention_flashinfer_common.cuh` — zero includers of its own, which is
/// why it sat in `spec/` rather than `csrc/` — and it has since been deleted
/// with that directory, taking the count from two dead readers to one. It is
/// not swept into the generated production shim either: the archive's
/// `kernels-cuda/build.rs::includes()` took every `*.hpp` in the FAMILY
/// directories, and `kernels_manifest.hpp` sits at `kernels/` top level.
///
/// So `kernels.def` is the archive's last piece of pure text and it is
/// deleted with `crates/kernels-cuda`. This citation is therefore to a file
/// that will not resolve, which is fine for the list — the four values ARE
/// the content and they are right here — and not fine for the one fact the
/// citation was carrying alone. That fact, verbatim from `kernels.def:53-56`
/// so it survives the file:
///
/// > 96 is deliberately absent: the prefill dispatch never had a 96 case, so
/// > a checkpoint that truly reached the kernels with 96 would already fail
/// > there — which is what made the decode side's `<96>` instantiations
/// > detectable as dead.
///
/// Phi-3-mini's 96 arrives as 128; the head dim is rounded up to one of the
/// four before it reaches a kernel.
const HEAD_DIMS: [u32; 4] = [64, 128, 256, 512];

/// `attn_head_dim_instantiated`, `attention_flashinfer.cu:236`.
#[must_use]
pub fn head_dim_instantiated(head_dim: u32) -> bool {
    HEAD_DIMS.contains(&head_dim)
}

// ── The environment gates, `attention_flashinfer.cu:105-115`, `:244` ────────

/// `PIE_CUDA_FORCE_SPLIT_KV_SMALL` — forces the real planner on small batches.
///
/// A debugging escape that disables [`plan_static_nonsplit_decode`] wholesale.
/// Read per call rather than cached in a `OnceLock`: it is off in production,
/// the read is a `getenv` on a path that already builds a plan, and a cached
/// value cannot be changed between two runs in one process, which is exactly
/// what a bisect wants to do.
#[must_use]
pub fn force_split_kv_small_enabled() -> bool {
    truthy("PIE_CUDA_FORCE_SPLIT_KV_SMALL")
}

/// `PIE_CUDA_WINDOW_SPLIT_KV` — routes windowed layers to the real planner.
///
/// See the roofline measurement in this module's header: with this off, a
/// sliding-window layer runs at `batch * kv_heads` CTAs, which was 8 on 148
/// SMs for gemma-4 and ~50x off the KV bandwidth roofline.
#[must_use]
pub fn window_split_kv_enabled() -> bool {
    truthy("PIE_CUDA_WINDOW_SPLIT_KV")
}

/// The C++'s truthiness, which is not Rust's and not `bool::from_str`'s.
///
/// `attention_flashinfer_common.cuh`'s helper treats an unset variable and the
/// literal `"0"` as false and everything else as true. Transcribed rather than
/// improved: a knob that answered differently in the two languages would make
/// a bisect across this port lie.
fn truthy(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => v != "0",
        Err(_) => false,
    }
}

// ── The decode cache ────────────────────────────────────────────────────────

/// FlashInfer's decode plan cache — `attention_flashinfer_common.cuh:341-374`.
///
/// A Rust struct and not a handle. The C++ type was incomplete on purpose
/// (`struct DecodePlanCache;`) and `driver-cuda`'s `bind::DecodePlan` wrapped
/// a `*mut c_void` created by `pie_x_make_decode_plan` and destroyed by a
/// custom deleter, because *"Rust was never available — the whole reason
/// these exist is a `unique_ptr` with a custom deleter"* (`plan_lifecycle.cpp`
/// lines 9-16). Rust is available now, the deleter is [`Drop`] and there is
/// nothing to release: the fields below are `Vec`s and plain data.
///
/// `bind::DecodePlan` still owns this cache's LIFETIME, behind a
/// `Box::into_raw` handle, and that is the one thing the descent did not
/// move: when a plan is rebuilt is a property of a fire, which is the
/// driver's to know.
#[derive(Clone, Debug, Default)]
pub struct DecodePlanCache {
    /// The descriptor the kernel reads.
    pub plan_info: DecodePlanInfo,
    /// Exactly the bytes upstream's `cudaMemcpyAsync(int_buffer, ...)` would
    /// have copied. **Held, not uploaded.** North-star §5 step 7 puts the H2D
    /// beside the launch that reads it rather than at the end of the planner,
    /// because the planner has no stream ordering to offer and the launch
    /// does.
    pub int_upload: Vec<u8>,
    /// The batch this plan was built for.
    pub num_requests: i32,
    /// Query heads.
    pub num_q_heads: i32,
    /// KV heads. `num_q_heads / num_kv_heads` is the GQA group that picks the
    /// decode unit.
    pub num_kv_heads: i32,
    /// Per-head width. One of `HEAD_DIMS`, this module’s private copy of the
    /// four the lattice carries.
    pub head_dim: i32,
    /// Tokens per page.
    pub page_size: i32,
    /// `kv_page_indptr_h[num_requests]` — how many pages the batch touches,
    /// and the count the dequant switch is given.
    pub num_pages_in_batch: i32,
    /// Programmatic dependent launch.
    pub enable_pdl: bool,
    /// Whether the plan was built for the full-attention variant.
    pub full_attention_variant: bool,
    /// HND page layout.
    pub hnd_layout: bool,
    /// Whether anything above was written.
    pub valid: bool,
    /// Whether the schedule is independent of the page counts it was planned
    /// with.
    ///
    /// The C++'s twenty-line comment is the contract and is kept whole,
    /// because everything that reads this field depends on all of it:
    ///
    /// > True when the plan was built by `plan_static_nonsplit_decode`, whose
    /// > descriptor is `request_indices[r] = r`, `kv_tile_indices = 0`,
    /// > `o_indptr[r] = r`, `split_kv = false` -- a schedule that does NOT
    /// > depend on the page counts it was planned with. That independence is
    /// > what lets a caller hand the *launch* a different (compacted) page
    /// > list than the one it planned against, which is how `attn_page_mask`
    /// > restricts a layer's attention without a replan. Under any other plan
    /// > the arrays ARE derived from page counts, and substituting a shorter
    /// > list is silently wrong.
    pub page_count_independent: bool,
    /// Byte offset of this plan's descriptor inside the shared int workspace.
    ///
    /// Also the C++'s, also kept whole: *"Every planner otherwise carves from
    /// offset 0, which is safe only while the live plans hold identical bytes
    /// -- and two plans over different REQUEST COUNTS never do, because the
    /// field offsets are derived from that count. A caller holding two plans
    /// at once sets this to keep them apart, the way the sm90 wrapper takes
    /// `int_base_bytes`."*
    pub int_base_bytes: usize,
    /// The request count the static vectors below were last refreshed for.
    pub static_nonsplit_num_requests: i32,
    /// `request_indices[r] = r`.
    pub static_request_indices: Vec<i32>,
    /// `kv_tile_indices[r] = 0`.
    pub static_kv_tile_indices: Vec<i32>,
    /// `o_indptr[r] = r`, with a trailing `num_requests`.
    pub static_o_indptr: Vec<i32>,
    /// The host page indptr, widened to `i32` for the planner.
    ///
    /// The C++ kept this buffer on the cache to avoid reallocating per step;
    /// the same reason applies here, and [`plan::decode::Request::kv_indptr`]
    /// borrows it.
    pub indptr_h_buf: Vec<i32>,
}

impl DecodePlanCache {
    /// A fresh, unplanned cache. `make_decode_plan()`, `:97-99`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// `set_decode_plan_int_base`, `:215`.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.int_base_bytes = bytes;
    }

    /// `can_use_static_nonsplit_decode_plan`, `:105-115`.
    ///
    /// Every clause is upstream's, in upstream's order. `cc_major >= 8` is a
    /// device fact and a parameter here rather than a query, for the reason
    /// `plan::Device` exists.
    #[must_use]
    pub fn can_use_static_nonsplit(num_requests: i32, cc_major: i32) -> bool {
        !force_split_kv_small_enabled() && cc_major >= 8 && num_requests > 0 && num_requests <= 512
    }

    /// `refresh_static_nonsplit_decode_vectors`, `:117-133`.
    ///
    /// Rebuilds the three descriptor arrays only when the request count moved,
    /// which is the whole point: the arrays depend on nothing else.
    fn refresh_static_vectors(&mut self, num_requests: i32) {
        if self.static_nonsplit_num_requests == num_requests {
            return;
        }
        self.static_nonsplit_num_requests = num_requests;
        let n = num_requests.max(0) as usize;
        self.static_request_indices.clear();
        self.static_request_indices.extend((0..n).map(|r| r as i32));
        self.static_kv_tile_indices.clear();
        self.static_kv_tile_indices.resize(n, 0);
        self.static_o_indptr.clear();
        self.static_o_indptr.extend((0..=n).map(|r| r as i32));
    }
}

// ── The prefill cache ───────────────────────────────────────────────────────

/// FlashInfer's prefill plan cache — `attention_flashinfer_common.cuh:376-400`.
///
/// [`DecodePlanCache`]'s twin, owned the same way for the same reasons.
#[derive(Clone, Debug, Default)]
pub struct PrefillPlanCache {
    /// The FA2 descriptor.
    pub plan_info: PrefillPlanInfo,
    /// The SM90 descriptor, when [`PrefillPlanCache::use_sm90`] is set.
    ///
    /// `Option<PrefillPlanSm90Info>` and not a `plan::sm90::Plan`: the sm90
    /// planner returns a `Plan<PrefillPlanSm90Info>` whose `int_upload` this
    /// cache keeps in its own [`PrefillPlanCache::int_upload`], so only the
    /// descriptor differs between the two routes.
    ///
    /// **The SM90 route has never been run in this migration.** §44.7's rule
    /// holds: every sm_90 claim here is argued from the call graph and none
    /// from a run, and the field exists because the C++ had one.
    pub sm90_plan: Option<PrefillPlanSm90Info>,
    /// Bytes for the H2D, held for the launch. See
    /// [`DecodePlanCache::int_upload`].
    pub int_upload: Vec<u8>,
    /// QO rows in the batch.
    pub total_tokens: i32,
    /// Requests in the batch.
    pub num_requests: i32,
    /// Query heads.
    pub num_q_heads: i32,
    /// KV heads.
    pub num_kv_heads: i32,
    /// Per-head width.
    pub head_dim: i32,
    /// Tokens per page.
    pub page_size: i32,
    /// Sliding-window span, `-1` for full attention.
    pub window_left: i32,
    /// Whether the plan was built for the full-attention variant.
    pub full_attention_variant: bool,
    /// Whether the mask is causal.
    pub causal_mask: bool,
    /// HND page layout.
    pub hnd_layout: bool,
    /// Whether the SM90 route was chosen.
    pub use_sm90: bool,
    /// Programmatic dependent launch.
    pub enable_pdl: bool,
    /// Whether anything above was written.
    pub valid: bool,
    /// Whether the executor may capture and replay the dispatch.
    ///
    /// The C++'s doc, kept: *"The plan ran in graph mode
    /// (content-independent launch geometry) on the FA2 causal path — the
    /// executor may capture/replay the dispatch. False when graph mode was
    /// requested but demoted (SM90 route, split disabled for the head dim, or
    /// the graph carve exceeding the float workspace grant)."*
    pub graph_capturable: bool,
    /// The host QO indptr, widened for the planner.
    pub qo_h_buf: Vec<i32>,
    /// The host KV page indptr, widened for the planner.
    pub kv_h_buf: Vec<i32>,
    /// The `CTA_TILE_Q` this plan was built at, which names the prefill unit.
    ///
    /// **Not a field of the C++ cache**, and not a recomputation either: it is
    /// `PrefillPlanInfo::cta_tile_q` (`plan/info.rs:128`) copied out where the
    /// dispatch can reach it without unpacking the plan.
    ///
    /// The archive did not need it. It instantiated all four `NUM_MMA_KV`
    /// points, so `DISPATCH_NUM_MMA_KV` was a free switch on a device query
    /// and the tile never had to leave the launcher. Under the JIT the tile
    /// **names the unit**, so it is a planning output — see `families/fa2.rs`
    /// on what that switch cost the archive in compiled points.
    pub cta_tile_q: u32,
    /// Byte offset of this plan's descriptor inside the shared int workspace.
    ///
    /// [`DecodePlanCache::int_base_bytes`]'s twin, and **not** a field of the
    /// C++ cache: the prefill planner always carved from offset zero because
    /// only one prefill plan was ever live. The field exists because
    /// [`upload_int_plan`] takes a base and taking it as a constant zero here
    /// would be an asymmetry a reader has to go and check. It defaults to 0,
    /// which is the C++'s behaviour exactly.
    pub int_base_bytes: usize,
}

impl PrefillPlanCache {
    /// A fresh, unplanned cache. `make_prefill_plan()`, `:101-103`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// [`DecodePlanCache::set_int_base`]'s twin.
    pub fn set_int_base(&mut self, bytes: usize) {
        self.int_base_bytes = bytes;
    }
}

// ── The MLA cache ───────────────────────────────────────────────────────────

// ── The device facts both halves need ───────────────────────────────────────

/// The four device attributes the FA2 planners and geometries read, queried
/// once.
///
/// [`crate::attn::plan::Device`] and
/// [`crate::attn::fa2::geometry::Device`] are deliberately different structs — a
/// planner needs the SM count and a geometry needs the shared-memory budget —
/// and this is the one place that fills both, so a fire cannot pair one
/// device's SM count with another's smem limit.
///
/// Through [`crate::jit::device`]'s memos rather than a second
/// `cudaDeviceGetAttribute` written here, `OnceLock` so two fires in one
/// process cannot disagree, and a **named fallback** rather than a panic on a
/// failed query.
///
/// It read `driver-cuda`'s `pie::Device` until §6.3 brought this file down
/// a crate. The four queries it made are the four `jit::device` already
/// memoised or grew to memoise, so what the descent removed here is a
/// `cudaSetDevice` and a runtime-version check that a plan has no business
/// making — this function ASKS about a device, it does not bind one.
///
/// # The fallback is `Device::L40S` and that is a claim, not a default
///
/// `fa2::Device::L40S` is *"not a default"* by its own doc, because the wrong
/// shared-memory budget produces a valid-looking `NUM_MMA_KV` and a kernel
/// that is quietly one CTA per SM. It is used here anyway, and the difference
/// is that this is a *failure* path: a machine where
/// `cudaDeviceGetAttribute` fails is a machine where the launch is going to
/// fail too, and answering with the box this tree was developed on gives the
/// failure a name in the log instead of a `None` three frames up.
fn facts() -> (Device, FaDevice) {
    static FACTS: OnceLock<(Device, FaDevice)> = OnceLock::new();
    *FACTS.get_or_init(|| {
        let Some((num_sm, cc_major, smem_sm, smem_block)) = queried() else {
            return (Device::new(148, 8), FaDevice::L40S);
        };
        (
            // `Device::new` takes the capability as `i32`, which is what the
            // planner's own arithmetic compares against.
            Device::new(num_sm.max(1), cc_major.cast_signed()),
            FaDevice {
                cc_major,
                max_smem_per_sm: smem_sm,
                max_smem_per_block_optin: smem_block,
            },
        )
    })
}

/// The four attributes, or `None` if any of them could not be had.
///
/// All four or none, deliberately: [`facts`] fills two structs from them and
/// the whole point of doing it in one place is that a fire cannot pair one
/// device's SM count with another's shared-memory budget. A partial answer
/// topped up from the fallback would be exactly that pairing.
#[cfg(feature = "_cuda")]
fn queried() -> Option<(u32, u32, u32, u32)> {
    use crate::jit::device;

    Some((
        device::multiprocessors().ok()?,
        device::compute_capability_major()?,
        device::max_shared_memory_per_sm().ok()?,
        device::max_shared_memory_per_block_optin().ok()?,
    ))
}

/// No CUDA runtime was selected, so there is nothing to ask.
///
/// The fallback that follows is not a stub for this build. This file is layer
/// 2 — it compiles and is readable without a driver — and a planner that
/// cannot ask a device anything is in exactly the position of one whose
/// queries failed, which the doc above already argues is a claim rather than
/// a default. Nothing it plans can fire in such a build regardless:
/// [`crate::jit::Ctx::launch`] refuses first, as a value.
#[cfg(not(feature = "_cuda"))]
const fn queried() -> Option<(u32, u32, u32, u32)> {
    None
}

/// The planner's device facts — SM count and compute-capability major.
#[must_use]
pub fn plan_device() -> Device {
    facts().0
}

/// The geometry's device facts — compute capability and both shared-memory
/// budgets.
#[must_use]
pub fn fa_device() -> FaDevice {
    facts().1
}

/// `max_grid_size` for [`plan_decode`], from the kernel that will run.
///
/// Upstream reads
/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` **on the decode kernel**
/// (`decode.cuh:715-718`) and multiplies by the SM count. That is a per-cubin
/// fact, so under the JIT it is
/// [`crate::attn::fa2::decode_blocks_per_sm`] over the `CUfunction`
/// the root produced — which means the point is compiled at PLAN time, before
/// the fire. Which arm is probed, and why `num_threads` rather than the
/// product of the block dims, are that function's own doc.
///
/// # When the query cannot be made this answers `num_sm`
///
/// One block per SM: a conservative answer rather than a wrong one.
/// `max_grid_size` bounds `plan::decode::estimate`'s split, so too small
/// yields fewer, larger KV chunks — the unsplit end of the range, which is
/// the plan the static short-circuit uses anyway. Too LARGE would be the
/// dangerous direction and cannot happen here.
#[must_use]
pub fn decode_max_grid_size(head_dim: i32, num_q_heads: i32, num_kv_heads: i32) -> u32 {
    let (device, fa) = facts();
    let floor = device.num_sm.max(1);
    if head_dim < 0 || !head_dim_instantiated(head_dim as u32) {
        return floor;
    }
    let group = if num_kv_heads > 0 { (num_q_heads / num_kv_heads).max(1) } else { 1 };
    match lattice::decode_blocks_per_sm(head_dim as u32, group as u32, fa) {
        Some(per_sm) => per_sm.max(1).saturating_mul(floor),
        None => floor,
    }
}

// ── The factories ───────────────────────────────────────────────────────────

/// `plan_static_nonsplit_decode`, `attention_flashinfer.cu:135-211`.
///
/// The hundredfold short-circuit. Writes the descriptor directly instead of
/// running FlashInfer's planner, which is legal only under the condition this
/// module's header records: the work estimator has already forced `split_kv`
/// off for these shapes, so the schedule does not depend on the KV lengths.
///
/// Sets [`DecodePlanCache::page_count_independent`], which is the field's only
/// producer.
///
/// # It carves its own descriptor, and that is the part that is easy to lose
///
/// `:145-171` is a four-call bump allocator over the int workspace — three
/// `IdType` arrays at 16-byte alignment and one scalar at 1 — and `:178-190`
/// fills the page-locked mirror from the three static vectors plus
/// `page_size` as the KV chunk size. Neither goes through
/// [`crate::attn::plan`]: that module implements upstream's *planner*,
/// and this function's whole purpose is not to run it.
///
/// The bytes land in [`DecodePlanCache::int_upload`] rather than in a
/// `cudaMemcpyAsync`, because §5 step 7 puts the H2D beside the launch. See
/// [`upload_int_plan`].
#[allow(clippy::too_many_arguments)]
pub fn plan_static_nonsplit_decode(
    cache: &mut DecodePlanCache,
    kv_page_indptr_h: &[u32],
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    workspace: Workspace,
    cc_major: i32,
    enable_cuda_graph: bool,
    full_attention_variant: bool,
    hnd_layout: bool,
) -> Planned {
    /// The `std::memcpy` at `attention_flashinfer.cu:178-190`, as a slice write.
    ///
    /// Native byte order, because the C++ `memcpy`'d host `int32_t`s and the
    /// device reads them back as `int32_t`. `to_ne_bytes` is that `memcpy`; a
    /// `to_le_bytes` here would be correct on every machine this runs on and
    /// wrong as a statement.
    fn put_i32s(dst: &mut [u8], at: i64, src: &[i32]) {
    let at = at.max(0) as usize;
    for (i, v) in src.iter().enumerate() {
    let lo = at + i * 4;
    dst[lo..lo + 4].copy_from_slice(&v.to_ne_bytes());
    }
    }

    /// `align_up_bytes` then bump — the `alloc` lambda at
    /// `attention_flashinfer.cu:146-151`.
    ///
    /// Returns the offset as `i64` because that is what a `PlanInfo` offset is:
    /// upstream's descriptor carries signed offsets and `offset_ptr` reads a
    /// negative one as "absent".
    fn carve(cursor: &mut usize, bytes: usize, alignment: usize) -> i64 {
    *cursor = cursor.next_multiple_of(alignment);
    let offset = *cursor;
    *cursor += bytes;
    offset as i64
    }

    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }
    cache.refresh_static_vectors(num_requests);

    let n = num_requests as usize;
    // `:145-151`. `sizeof(IdType)` — `IdType` is `int32_t` throughout this
    // lattice, which is also why the three static vectors are `Vec<i32>`.
    const ID: usize = 4;
    let mut cursor = 0usize;
    let request_indices_offset = carve(&mut cursor, ID * n, 16);
    let kv_tile_indices_offset = carve(&mut cursor, ID * n, 16);
    let o_indptr_offset = carve(&mut cursor, ID * (n + 1), 16);
    let kv_chunk_size_ptr_offset = carve(&mut cursor, ID, 1);

    // `:173-176`. The base is included because several layers' plans share one
    // int buffer and this one starts at `int_base_bytes`.
    let needed = cursor.saturating_add(cache.int_base_bytes);
    if needed > workspace.int_bytes {
        return Planned::Declined(Decline::WorkspaceTooSmall { needed, have: workspace.int_bytes });
    }

    // `:178-190`. The C++ wrote into the page-locked mirror at
    // `page_locked_int + int_base_bytes` and copied `cursor` bytes from there;
    // this builds the same `cursor` bytes and hands them to the fire. The
    // offsets are relative to the base in both cases, which is why the buffer
    // starts at zero and the H2D adds `int_base_bytes` to the destination.
    cache.int_upload.clear();
    cache.int_upload.resize(cursor, 0);
    put_i32s(&mut cache.int_upload, request_indices_offset, &cache.static_request_indices);
    put_i32s(&mut cache.int_upload, kv_tile_indices_offset, &cache.static_kv_tile_indices);
    put_i32s(&mut cache.int_upload, o_indptr_offset, &cache.static_o_indptr);
    // `:189-190`. **A chunk size in TOKENS**, and for an unsplit plan the
    // chunk is one page — which is `page_size` tokens, not 1.
    put_i32s(&mut cache.int_upload, kv_chunk_size_ptr_offset, &[page_size]);

    cache.plan_info = DecodePlanInfo {
        enable_cuda_graph,
        split_kv: false,
        padded_batch_size: i64::from(num_requests),
        request_indices_offset,
        kv_tile_indices_offset,
        o_indptr_offset,
        kv_chunk_size_ptr_offset,
        // `:153-154`, `:159` — all three explicitly zero, and they must be:
        // nothing carves the float workspace on an unsplit plan and there is
        // no padding for a valid mask to describe.
        v_offset: 0,
        s_offset: 0,
        block_valid_mask_offset: 0,
        ..DecodePlanInfo::default()
    };

    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.num_pages_in_batch =
        kv_page_indptr_h.get(num_requests as usize).copied().unwrap_or(0) as i32;
    // `:207` — `current_device_supports_pdl()`, which
    // `crate::attn::xqa::xqa_decode_bf16` states as `major >= 9` and
    // does not apply either. RECORDED AND NOT APPLIED: programmatic dependent launch
    // is a `cudaLaunchKernelEx` attribute and this lattice fires through
    // `cuLaunchKernel`, as every other JIT row in this driver does. The field
    // is kept so that the day a launch attribute path exists, the plan already
    // says whether the device would take one.
    cache.enable_pdl = cc_major >= 9;
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout = hnd_layout;
    cache.page_count_independent = true;
    cache.valid = true;

    Planned::StaticNonsplit
}

/// `plan_attention_flashinfer_decode_bf16`, `attention_flashinfer.cu:218-285`.
///
/// The head_dim check is first and stays first — `:232-238`'s comment is the
/// reason, and it is quoted on [`Decline::HeadDim`]. Then the windowed test,
/// then the short-circuit, then the real planner.
///
/// `max_grid_size` is a parameter and not a query. Upstream reads it from
/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` **on the decode kernel** —
/// a per-cubin fact — and under the JIT that is
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor`
/// over `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the `CUfunction` the
/// unit produced, times the SM count. The caller asks the module, because the
/// module is what holds the function.
#[allow(clippy::too_many_arguments)]
pub fn plan_decode(
    cache: &mut DecodePlanCache,
    kv_page_indptr_h: &[u32],
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    workspace: Workspace,
    device: &Device,
    max_grid_size: u32,
    enable_cuda_graph: bool,
    full_attention_variant: bool,
    hnd_layout: bool,
    window_left: i32,
) -> Planned {
    if head_dim < 0 || !head_dim_instantiated(head_dim as u32) {
        return Planned::Declined(Decline::HeadDim { head_dim: head_dim.max(0) as u32 });
    }
    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }

    // `:240-245`. A windowed layer wants the real planner; see the roofline
    // measurement in this module's header for what the static plan costs it.
    let windowed_split = window_split_kv_enabled() && window_left >= 0;
    if !windowed_split && DecodePlanCache::can_use_static_nonsplit(num_requests, device.cc_major) {
        return plan_static_nonsplit_decode(
            cache,
            kv_page_indptr_h,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            workspace,
            device.cc_major,
            enable_cuda_graph,
            full_attention_variant,
            hnd_layout,
        );
    }

    // The planner takes `i32`; the fire's indptr is `u32`. Widened into the
    // cache's own buffer rather than allocated per step, which is what the
    // C++'s `indptr_h_buf` was for.
    cache.indptr_h_buf.clear();
    cache
        .indptr_h_buf
        .extend(kv_page_indptr_h.iter().take(num_requests as usize + 1).map(|&v| v as i32));

    let gqa_group_size = if num_kv_heads > 0 { num_q_heads / num_kv_heads } else { 0 };
    let req = plan::decode::Request {
        kv_indptr: &cache.indptr_h_buf,
        batch_size: num_requests as u32,
        num_qo_heads: num_q_heads as u32,
        gqa_group_size: gqa_group_size as u32,
        page_size: page_size as u32,
        head_dim: head_dim as u32,
        enable_cuda_graph,
    };

    let planned = match plan::decode::plan(&req, max_grid_size, workspace) {
        Ok(p) => p,
        Err(_) => return Planned::Declined(Decline::Planner("decode")),
    };

    cache.plan_info = planned.info;
    cache.int_upload = planned.int_upload;
    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.num_pages_in_batch =
        kv_page_indptr_h.get(num_requests as usize).copied().unwrap_or(0) as i32;
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout = hnd_layout;
    // `:283`. Recorded, not applied — see `plan_static_nonsplit_decode`.
    cache.enable_pdl = device.cc_major >= 9;
    // The real planner's arrays ARE derived from page counts. See the field.
    cache.page_count_independent = false;
    cache.valid = true;

    Planned::Full
}

/// `plan_attention_flashinfer_prefill_bf16`, `attention_flashinfer.cu:287-373`.
///
/// Adds one thing the C++ did not record: [`PrefillPlanCache::cta_tile_q`].
/// The archive recomputed the tile at every dispatch because all four
/// `NUM_MMA_KV` points were instantiated and the switch was free; under the
/// JIT the tile names the unit, so it is a planning output.
#[allow(clippy::too_many_arguments)]
pub fn plan_prefill(
    cache: &mut PrefillPlanCache,
    qo_indptr_h: &[u32],
    kv_page_indptr_h: &[u32],
    total_tokens: i32,
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    workspace: Workspace,
    device: &Device,
    enable_cuda_graph: bool,
    window_left: i32,
    full_attention_variant: bool,
    hnd_layout: bool,
    causal_mask: bool,
    custom_mask: bool,
    wants_prefill_score: bool,
) -> Planned {
    if head_dim < 0 || !head_dim_instantiated(head_dim as u32) {
        return Planned::Declined(Decline::HeadDim { head_dim: head_dim.max(0) as u32 });
    }
    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }

    // `:299-317`. Both halves, in order: the window is refused, then the
    // variant is PROMOTED to full.
    //
    // The promotion is numerics-neutral and the C++ says why: with
    // `window_left < 0` the windowed variant's `LogitsMask` is `true` for
    // every position, so `Full` and `Window` compute the same logits — but
    // only `Full` has capture instantiations. Promoting therefore changes
    // which unit runs and not what it computes.
    let full_attention_variant = if wants_prefill_score {
        if window_left >= 0 {
            return Planned::Declined(Decline::ScoreCaptureWindow { window_left });
        }
        true
    } else {
        full_attention_variant
    };

    cache.qo_h_buf.clear();
    cache.qo_h_buf.extend(qo_indptr_h.iter().take(num_requests as usize + 1).map(|&v| v as i32));
    cache.kv_h_buf.clear();
    cache
        .kv_h_buf
        .extend(kv_page_indptr_h.iter().take(num_requests as usize + 1).map(|&v| v as i32));

    let req = plan::prefill::Request {
        qo_indptr: &cache.qo_h_buf,
        kv_indptr: &cache.kv_h_buf,
        total_num_rows: total_tokens.max(0) as u32,
        batch_size: num_requests as u32,
        num_qo_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim_qk: head_dim as u32,
        head_dim_vo: head_dim as u32,
        page_size: page_size as u32,
        enable_cuda_graph,
        sizeof_dtype_o: 2,
        window_left,
        fixed_split_size: -1,
        // `:381`'s `disable_split_kv = !head_dim_supports_cascade_merge(head_dim)`,
        // restored — and this comment is the record of what took it away and
        // what brought it back, because the reason mattered.
        //
        // The C++ asked *"can `VariableLengthMergeStates` run for this head
        // dim?"*, because a split plan leaves partial outputs in `tmp_v` and
        // partial log-sum-exps in `tmp_s` that the cascade merge has to fold
        // into `o`. Its answer was per-head-dim because the merge kernel was
        // instantiated for some dims and not others (`:980-985`).
        //
        // For one migration pass the answer was **no for every head dim**.
        // `attn::merge_attention_states_bf16` was deleted by
        // `new-horizon.md` §38 — correctly, its DSL wrapper had no caller —
        // but the C++ that actually ran was compiled into
        // `driver-cuda/csrc/attn/attention_flashinfer.cu`, and closing the
        // FA2 seams deleted that file. There was nothing left to fold the
        // partials with, so a split plan was not a slower answer, it was no
        // answer, and this said `true`. That was a REAL performance
        // regression on short prompts and small batches, which are exactly
        // the shapes the prefill scheduler splits.
        //
        // It is fixed. `crate::cascade` carries
        // `PersistentVariableLengthMergeStatesKernel` at all four head dims,
        // compiled by NVRTC out of the VENDORED `cascade.cuh`, and
        // `fire/merge_states.rs` is the Rust host program that fires it.
        // Every dispatch that returns
        // `flashinfer_fa2_dispatch::Fired::Split` now folds before it
        // returns, so the predicate is upstream's again and its input is the
        // lattice's own head-dim set. `x::cascade::HEAD_DIMS` and
        // `x::fa2::HEAD_DIMS` are the same four values in two places, and
        // whoever changes one says which.
        disable_split_kv: !head_dim_instantiated(head_dim.max(0) as u32),
        num_colocated_ctas: 0,
    };

    // `:390-399`, the graph-mode demotion, in the shape the Rust planner
    // makes available.
    //
    // The C++ recomputed a worst-case `max_qo_len` BEFORE planning and turned
    // `enable_cuda_graph` off when the resulting tile would not fit. The Rust
    // planner does that same arithmetic internally (`prefill.rs:225`) and
    // reports the overflow as an error instead, so the demotion is written
    // here as a retry: plan in graph mode, and if the carve does not fit, plan
    // again eagerly. Same two outcomes, same order of preference, and the
    // second attempt is the one whose failure is fatal.
    let (planned, capturable) = match plan::prefill::plan(&req, device, workspace) {
        Ok(p) => (p, enable_cuda_graph),
        Err(_) if enable_cuda_graph => {
            let req = plan::prefill::Request { enable_cuda_graph: false, ..req };
            match plan::prefill::plan(&req, device, workspace) {
                Ok(p) => (p, false),
                Err(_) => return Planned::Declined(Decline::Planner("prefill")),
            }
        }
        Err(_) => return Planned::Declined(Decline::Planner("prefill")),
    };

    // The tile, read back from the plan rather than recomputed.
    //
    // This was a reconstruction — `total_tokens * group / num_requests` fed to
    // `plan::arith::fa2_determine_cta_tile_q` — and it was the one number in
    // this module without a citation. It is now gone, because the recompute
    // was not merely uncited, it was **unnecessary**: `plan::prefill` at
    // `prefill.rs:238-239` computes `avg_packed_qo_len` from
    // `sum_packed_qo_len / batch_size` (a sum over the QO indptr, not a
    // product of totals — my reconstruction had the wrong numerator) and
    // publishes the answer as `PrefillPlanInfo::cta_tile_q`, `info.rs:128`,
    // pinned at offset 24 by `info.rs:158`.
    //
    // Reading it back rather than recomputing it also removes a class of bug
    // rather than a line: the planner **split the batch against this tile**
    // (`prefill.rs:241`, `:289`), so a fire that chose its own would index a
    // work list built for a different one. There is no arithmetic here that
    // can drift from the planner's, because there is no arithmetic here.
    //
    // The graph-mode branch at `attention_flashinfer.cu:390-399` is a
    // different quantity and is not this one: it is `max_qo_len`, the
    // worst case where one request holds every token
    // (`max(1, total_tokens - num_requests + 1) * max(1, gqa_group)`), used
    // only to bound a workspace that must be sized before the batch is known.
    // `plan::prefill` has that branch too, at `prefill.rs:225`, and takes it
    // under the same condition.
    let cta_tile_q = u32::try_from(planned.info.cta_tile_q).unwrap_or(0);

    // The gate. See `Decline::HeadDimTile`: with the tile now coming from the
    // planner, `arith.rs:95`'s `head_dim < 256` makes this unreachable — it is
    // a statement kept against an upstream edit, not a live branch.
    if head_dim as u32 >= 256 && cta_tile_q == 128 {
        return Planned::Declined(Decline::HeadDimTile { head_dim: head_dim as u32, cta_tile_q });
    }

    cache.plan_info = planned.info;
    cache.int_upload = planned.int_upload;
    cache.total_tokens = total_tokens;
    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.window_left = window_left;
    cache.full_attention_variant = full_attention_variant;
    cache.causal_mask = causal_mask;
    cache.hnd_layout = hnd_layout;
    cache.cta_tile_q = cta_tile_q;
    // `:319-321`. Three fields set together because they are one decision.
    //
    // The C++ chose the SM90 route at `:325-333` when the layer was
    // uncustomised, unrolled, unscored and Hopper. **This lattice never takes
    // it**, so `use_sm90` is a written `false` rather than an unset field, and
    // `sm90_plan` is `None` rather than stale: `families/fa2.rs` holds FA2
    // units only, so there is no `sm90` unit for a plan to name and a `true`
    // here would produce a cache the fire can only refuse. The refusal is
    // `flashinfer_fa2_dispatch::Decline::Sm90Unported`, and it exists so that
    // a future SM90 family can be wired by making this line conditional and
    // nothing else.
    cache.use_sm90 = false;
    cache.sm90_plan = None;
    // `:390-399`, the other half of the demotion above: graph capture is
    // available only if the plan that survived was the graph-mode one.
    cache.graph_capturable = capturable;
    // `custom_mask` reached the C++ planner for exactly one purpose — it
    // vetoed the SM90 route at `:325`. With that route unreachable the flag
    // has no planning consequence left, and it is taken as a parameter anyway
    // so that the call site keeps stating it and so this comment has somewhere
    // to live. The mask POINTERS are a dispatch input, not a plan input:
    // `flashinfer_fa2_dispatch::prefill_custom`.
    let _ = custom_mask;
    // `:362`. Recorded, not applied — see `plan_static_nonsplit_decode`.
    cache.enable_pdl = device.cc_major >= 9;
    cache.valid = true;

    Planned::Full
}

// ── The H2D and the fire — north star §5 step 7's last two seams ────────────

/// The plan's descriptor, host to device.
///
/// `attention_flashinfer.cu:193-198`, moved to where the launch is. The C++
/// issued it at the end of the planner; this is issued immediately before the
/// fire that reads it, which is what §5 step 7 asks for and is also the only
/// version that is correct without further argument: the planner has no
/// stream to order against, and a descriptor uploaded on one stream and read
/// on another is a race the C++ avoided by having exactly one stream and not
/// by saying so.
///
/// `int_base_bytes` is added to the DESTINATION and not to the offsets.
/// [`DecodePlanCache::int_upload`] is carved from zero, so a plan that shares
/// an int workspace with another moves as a block and the descriptor's own
/// offsets stay relative — which is what makes
/// [`DecodePlanCache::set_int_base`] a one-field change.
///
/// # The one behaviour difference from the C++, stated because it is real
///
/// The C++ copied from `workspace.page_locked_int`, so its `cudaMemcpyAsync`
/// was fully asynchronous. This copies from a `Vec<u8>` the cache owns, which
/// is pageable, and a pageable H2D `cudaMemcpyAsync` stages through a driver
/// pinned buffer and is only *partially* async. That is a latency difference
/// on a per-plan path, not a correctness one, and it buys the deletion of the
/// page-locked staging slot's exclusive owner. **The source must outlive the
/// copy** and does: `int_upload` belongs to the cache, and the cache outlives
/// the fire.
///
/// # Errors
///
/// The copy faulted.
///
/// # Safety
///
/// `int_buffer` must name at least `int_base_bytes + bytes.len()` writable
/// device bytes, and `stream` must outlive the copy.
pub unsafe fn upload_int_plan(
    bytes: &[u8],
    int_buffer: u64,
    int_base_bytes: usize,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    if bytes.is_empty() {
        return Ok(());
    }
    let dst = (int_buffer as usize).saturating_add(int_base_bytes) as *mut c_void;
    #[cfg(feature = "_cuda")]
    {
        // SAFETY: the caller's contract, forwarded verbatim.
        unsafe { crate::jit::device::upload(dst, bytes, stream) }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (dst, stream);
        Err(Refusal::Device { why: "this build selected no CUDA runtime" })
    }
}

/// Where a plan's descriptor has to land before the launch reads it.
///
/// Three values that always travel together, named once so that the fire's
/// signature does not grow a `u64` and a `usize` whose order is guessable.
#[derive(Clone, Copy, Debug)]
pub struct PlanUpload<'a> {
    /// [`DecodePlanCache::int_upload`] or [`PrefillPlanCache::int_upload`].
    pub bytes: &'a [u8],
    /// `AttentionWorkspaceView::int_buffer`.
    pub int_buffer: u64,
    /// [`DecodePlanCache::int_base_bytes`]. **Zero for prefill** — one prefill
    /// plan serves one fire, so `make_prefill_params` reads the workspace base
    /// directly and there is no prefill analogue of the decode offset.
    pub int_base_bytes: usize,
}

/// Upload the plan, then fire one prefill dispatch.
///
/// Seam 1 and seam 3 in one function, in that order, because the order is the
/// point: the grid indexes the work list being uploaded.
///
/// `P` is the params type — [`crate::attn::fa2::params::PrefillPagedParams`]
/// or [`crate::attn::fa2::params::PrefillScoreParams`] — so the capturing and
/// non-capturing fires are one function.
///
/// **There was a `fire_decode` beside it, and it had no caller.** It is
/// deleted, with the four uncalled preparers in [`super::dispatch`] that were
/// its only possible source of a `DecodeDispatch`; that module's header is
/// the account. The one caller of this pair is `driver-cuda`'s ViT tower,
/// which reaches FA2 by path rather than through a trace statement and does
/// prefill only.
///
/// # Errors
///
/// The descriptor upload faulted, or the routine refused.
///
/// # Safety
///
/// Every device address in `dispatch.params` must name memory of the extent
/// the kernel reads or writes, `upload.int_buffer` must be the workspace the
/// params' offsets were computed against, and `stream` must outlive the
/// launch.
pub unsafe fn fire_prefill<P: lattice::PrefillBlock>(
    dispatch: &mut PrefillDispatch<P>,
    upload: PlanUpload<'_>,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // SAFETY: the caller's contract, forwarded.
    unsafe {
        upload_int_plan(upload.bytes, upload.int_buffer, upload.int_base_bytes, stream)?;
    }
    // SAFETY: as above.
    let ctx = unsafe { crate::jit::Ctx::on(stream) };
    lattice::prefill(&ctx, dispatch.at, &dispatch.params)
}

#[cfg(test)]
mod tests {
    use super::{HEAD_DIMS, head_dim_instantiated};

    /// The gate and the lattice name the same head dims.
    ///
    /// Two lists in two crates, and nothing but this makes them agree. The
    /// failure it prevents is quiet in the worst way: a head dim this gate
    /// admits and the lattice does not instantiate is a JIT miss at the fire,
    /// with the symbol named but no unit to compile it.
    #[test]
    fn the_gate_and_the_lattice_agree() {
        assert_eq!(HEAD_DIMS.as_slice(), crate::attn::fa2::HEAD_DIMS);
        for hd in HEAD_DIMS {
            assert!(head_dim_instantiated(hd));
        }
        assert!(
            !head_dim_instantiated(96),
            "96 is deliberately absent — the prefill dispatch never had a 96 \
             case, so a checkpoint reaching the kernels with 96 fails there. \
             The reason is carried in `HEAD_DIMS`' doc; `kernels.def` is \
             deleted with the archive and has no reader."
        );
        assert!(!head_dim_instantiated(0));
    }

    /// The pair the prefill gate exists for cannot come from the chooser.
    ///
    /// `arith.rs:95`'s `head_dim < 256` is the guard, and this is the
    /// assertion that an upstream edit removing it does not pass silently.
    #[test]
    fn the_tile_chooser_never_asks_for_128_at_256() {
        use crate::attn::plan::arith::fa2_determine_cta_tile_q;
        for &hd in &[256u32, 512] {
            for qo in [1i64, 16, 17, 64, 65, 4096] {
                assert_ne!(
                    fa2_determine_cta_tile_q(qo, hd, 8),
                    128,
                    "head_dim {hd} at avg_packed_qo_len {qo} asked for CTA_TILE_Q 128, \
                     which `KernelTraits::IsInvalid()` rejects for every NUM_MMA_KV",
                );
            }
        }
    }

    /// The static plan's descriptor is carved and filled, not merely declared.
    ///
    /// The regression this guards is the one the seam was left at: a
    /// `plan_info` with the right `padded_batch_size` and four zero offsets
    /// looks planned, launches, and reads request 0's work item for every
    /// block. Three arrays at 16-byte alignment and one scalar at 1 —
    /// `attention_flashinfer.cu:145-151` — for four requests is 16 + 16 + 32 +
    /// 4, and the KV chunk size is `page_size` TOKENS.
    #[test]
    fn the_static_plan_carves_and_fills_its_descriptor() {
        let mut cache = super::DecodePlanCache::new();
        let indptr = [0u32, 2, 4, 6, 8];
        let planned = super::plan_static_nonsplit_decode(
            &mut cache,
            &indptr,
            4,
            8,
            2,
            128,
            16,
            crate::attn::plan::Workspace { float_bytes: 1 << 20, int_bytes: 1 << 20 },
            8,
            false,
            true,
            false,
        );
        assert!(matches!(planned, super::Planned::StaticNonsplit));
        let info = &cache.plan_info;
        assert_eq!(info.request_indices_offset, 0);
        assert_eq!(info.kv_tile_indices_offset, 16);
        assert_eq!(info.o_indptr_offset, 32);
        assert_eq!(info.kv_chunk_size_ptr_offset, 52);
        assert_eq!(cache.int_upload.len(), 56);
        assert!(!info.split_kv);
        assert_eq!(info.padded_batch_size, 4);

        let at =
            |off: usize| i32::from_ne_bytes(cache.int_upload[off..off + 4].try_into().unwrap());
        assert_eq!([at(0), at(4), at(8), at(12)], [0, 1, 2, 3], "request_indices[r] = r");
        assert_eq!([at(16), at(20), at(24), at(28)], [0; 4], "kv_tile_indices = 0");
        assert_eq!([at(32), at(36), at(40), at(44), at(48)], [0, 1, 2, 3, 4], "o_indptr");
        assert_eq!(at(52), 16, "the KV chunk size is page_size TOKENS, not 1");
    }

    /// A workspace that cannot hold the descriptor is a named refusal.
    #[test]
    fn a_short_workspace_declines_rather_than_truncating() {
        let mut cache = super::DecodePlanCache::new();
        cache.set_int_base(4096);
        let indptr = [0u32, 1];
        let planned = super::plan_static_nonsplit_decode(
            &mut cache,
            &indptr,
            1,
            8,
            2,
            128,
            16,
            crate::attn::plan::Workspace { float_bytes: 0, int_bytes: 4096 },
            8,
            false,
            true,
            false,
        );
        assert!(matches!(
            planned,
            super::Planned::Declined(super::Decline::WorkspaceTooSmall { have: 4096, .. })
        ));
        assert!(!cache.valid, "a declined plan is not a valid one");
    }
}

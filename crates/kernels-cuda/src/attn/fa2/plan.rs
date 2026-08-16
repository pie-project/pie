//! The FlashInfer FA2 host program's plan half, in Rust.
//!
//! The plan caches as Rust structs, the two planner factories over
//! [`crate::attn::plan`], the static non-split decode short-circuit and its
//! gates, the geometry gate that turns an unsupported (head dim,
//! `CTA_TILE_Q`) pair into a named refusal, and the plan H2D
//! ([`upload_int_plan`]) and fire. Launches resolve against
//! [`crate::attn::fa2`]'s roots; [`crate::fire::kv_paged`] holds the dequant
//! switch. Params filling and the arm cascades are
//! [`crate::fire::flashinfer_fa2_dispatch`].
//!
//! Not here: the SM90 prefill launcher. [`PrefillPlanCache::sm90_plan`] is
//! planned and recorded, and the dispatch refuses with
//! `Decline::Sm90Unported` rather than firing an FA2 symbol at it. Every
//! sm_90 claim in this migration is argued from the call graph and none from
//! a run; nothing here has been run on Hopper.
//!
//! # The measurements this file inherits
//!
//! Consuming one of these is a regression even if it compiles.
//!
//! * Re-running the full planner per decode batch was a hundredfold cost, so
//!   [`plan_static_nonsplit_decode`] skips it. That is legal only because the
//!   work estimator has already forced `split_kv` off for the TP1 latency
//!   shapes, which makes the schedule independent of KV lengths.
//! * The static plan is unsplit by construction, which leaves a sliding layer
//!   at `batch * kv_heads` CTAs (8 on 148 SMs for gemma-4) and ~50x off its
//!   bandwidth roofline. That is why a windowed layer takes the real planner:
//!   dropping the windowed branch is fully correct and 50x slower on exactly
//!   one layer type.
//! * A sliding layer's split is bounded by the window and is cheap; an
//!   unbounded one at 1k context took a 256-token generation from 22 s to
//!   over 2400 s. The C++ carried that through a `thread_local`
//!   `decode_window_hint()` because upstream's estimator could not take
//!   another argument. [`plan_decode`] makes the same branch at the call
//!   site, before choosing a planner.
//! * `head_dim_supports_cascade_merge`'s `{64, 128, 256, 512}` is upstream's
//!   set, and it agrees with [`crate::attn::fa2::HEAD_DIMS`] by shared origin
//!   rather than by construction. Whoever changes one says which.

use std::ffi::c_void;
use std::sync::OnceLock;

use super::geometry::Device as FaDevice;
use crate::attn::plan::info::{
    DecodePlanInfo, PrefillPlanInfo, PrefillPlanSm90Info,
};
use crate::attn::plan::{self, Device, Workspace};
use crate::attn::fa2 as lattice;

use kernels::Refusal;

use crate::jit::PinnedBytes;

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
    /// The one pair this can name is head dim 256 with `CTA_TILE_Q` 128, and
    /// it is upstream's own exclusion. `KernelTraits::IsInvalid()`
    /// (`prefill.cuh:221-232`) rejects
    ///
    /// ```text
    /// NUM_MMA_Q * (8 * NUM_MMA_D_VO_TILE + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256
    /// ```
    ///
    /// and at head dim 256 (`NUM_MMA_D_VO` 16) with `CTA_TILE_Q` 128
    /// (`NUM_MMA_Q` 2) the left side is `2 * (128 + 8 * NUM_MMA_KV)`, at least
    /// 256 for every `NUM_MMA_KV` including zero. No valid instantiation
    /// exists, so `x::fa2` names no root for it.
    ///
    /// The chooser cannot produce the pair —
    /// [`crate::attn::plan::arith::fa2_determine_cta_tile_q`] gates 128 on
    /// `head_dim < 256` — but `cta_tile_q` is a *parameter* here and a caller
    /// may compute it some other way. The refusal is cheaper than a launch
    /// that fails inside NVRTC on a static assertion.
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
    /// The planner's own [`plan::Error`] is **not** carried, which loses the
    /// array `Error::WorkspaceOverflow` names and the length
    /// `Error::IndptrTooShort` names. `Decline` is `Copy` and `plan::Error`
    /// is not. If a workspace overflow ever needs diagnosing from this path,
    /// this arm is what has to grow — a `Box<plan::Error>` and the loss of
    /// `Copy`.
    Planner(&'static str),
    /// The int workspace cannot hold this plan's descriptor.
    ///
    /// `plan_static_nonsplit_decode` threw *"attention int workspace too
    /// small"*: the static planner carves the descriptor itself rather than
    /// going through [`crate::attn::plan`]'s allocator, so this is the one
    /// overflow this module can name precisely.
    WorkspaceTooSmall {
        /// Bytes the plan needs.
        needed: usize,
        /// Bytes the caller supplied.
        have: usize,
    },
    /// A score-capturing prefill plan was asked for with a sliding window.
    ///
    /// FlashInfer applies `LogitsMask` **after** `LogitsTransform`, so a
    /// windowed variant's mask runs on logits the capture has already written
    /// out. The captured tensor would hold scores for positions the kernel
    /// then discards — not a scaled answer but a different one, and silently.
    ScoreCaptureWindow {
        /// The window the caller asked for.
        window_left: i32,
    },
    /// The descriptor's page-locked buffer could not be taken.
    ///
    /// The plan itself succeeded; what failed is the pin the H2D's source
    /// must have to be capturable — see [`PinnedBytes`]. A refusal rather
    /// than a fallback to pageable bytes, because the fallback is the
    /// use-after-free §6.1 F4 names.
    Pin,
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
            Self::Pin => write!(
                f,
                "flashinfer fa2: the plan descriptor's page-locked buffer could not be taken, and \
                  a pageable one is not capturable",
            ),
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
/// Upstream's `head_dim_supports_cascade_merge` set. Stated here as well as
/// in [`crate::attn::fa2::HEAD_DIMS`] because this is the *gate* and that is
/// the *lattice*: they must agree, and [`the_gate_and_the_lattice_agree`] is
/// what makes them.
///
/// 96 is deliberately absent: the prefill dispatch never had a 96 case, so a
/// checkpoint that truly reached the kernels with 96 would already fail
/// there — which is what made the decode side's `<96>` instantiations
/// detectable as dead. Phi-3-mini's 96 arrives as 128; the head dim is
/// rounded up to one of the four before it reaches a kernel.
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
/// A Rust struct and not a handle: the C++ type was incomplete on purpose and
/// existed only to hang a `unique_ptr` deleter on. There is nothing to
/// release here; the fields are `Vec`s and plain data.
///
/// `bind::DecodePlan` still owns this cache's LIFETIME behind a
/// `Box::into_raw` handle: when a plan is rebuilt is a property of a fire,
/// which is the driver's to know.
// NOT `Clone`: `int_upload` is pinned, and a captured graph holds its
// address. A clone would be a second buffer that every replay ignores.
#[derive(Debug, Default)]
pub struct DecodePlanCache {
    /// The descriptor the kernel reads.
    pub plan_info: DecodePlanInfo,
    /// Exactly the bytes upstream's `cudaMemcpyAsync(int_buffer, ...)` would
    /// have copied. **Held, not uploaded**: the H2D sits beside the launch
    /// that reads it, because the planner has no stream ordering to offer and
    /// the launch does.
    ///
    /// [`PinnedBytes`] and not a `Vec`, because a captured graph bakes this
    /// buffer's ADDRESS. The capacity is taken once, on the first plan, from
    /// the largest descriptor this cache's geometry can produce, and a plan
    /// that would exceed it is refused rather than moved.
    pub int_upload: PinnedBytes,
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
    /// True when the plan was built by `plan_static_nonsplit_decode`, whose
    /// descriptor is `request_indices[r] = r`, `kv_tile_indices = 0`,
    /// `o_indptr[r] = r`, `split_kv = false` — a schedule that does NOT
    /// depend on the page counts it was planned with. That independence is
    /// what lets a caller hand the *launch* a compacted page list instead of
    /// the one it planned against, which is how `attn_page_mask` restricts a
    /// layer's attention without a replan. Under any other plan the arrays
    /// ARE derived from page counts, and substituting a shorter list is
    /// silently wrong.
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

    /// `can_use_static_nonsplit_decode_plan`, `:105-115`, plus the capture
    /// clause upstream did not need. `cc_major >= 8` is a device fact and a
    /// parameter here rather than a query.
    ///
    /// # Why `!enable_cuda_graph`
    ///
    /// The static plan carves every offset from `num_requests`, so its
    /// offsets MOVE WITH THE BATCH. A captured graph bakes the addresses
    /// those offsets resolve to and re-uploads only the VALUES at them, so a
    /// plan whose layout is a function of the fire is a plan a capture cannot
    /// hold. Decode is always `<= 512` requests, so without this clause the
    /// static path is the path that runs and the graph bakes offsets from
    /// whichever batch happened to capture. The real planner's layout is a
    /// function of head geometry alone once `enable_cuda_graph` is set, which
    /// is the constancy capture needs.
    #[must_use]
    pub fn can_use_static_nonsplit(
        num_requests: i32,
        cc_major: i32,
        enable_cuda_graph: bool,
    ) -> bool {
        !enable_cuda_graph
            && !force_split_kv_small_enabled()
            && cc_major >= 8
            && num_requests > 0
            && num_requests <= 512
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
// NOT `Clone`, for [`DecodePlanCache`]'s reason.
#[derive(Debug, Default)]
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
    /// [`DecodePlanCache::int_upload`], including why it is pinned.
    pub int_upload: PinnedBytes,
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
/// [`crate::attn::plan::Device`] and [`crate::attn::fa2::geometry::Device`]
/// are deliberately different structs — a planner needs the SM count, a
/// geometry needs the shared-memory budget — and this is the one place that
/// fills both, so a fire cannot pair one device's SM count with another's
/// smem limit. `OnceLock` so two fires in one process cannot disagree.
///
/// The fallback is `Device::L40S`, which that type's own doc calls *"not a
/// default"* because the wrong shared-memory budget produces a valid-looking
/// `NUM_MMA_KV` and a kernel that is quietly one CTA per SM. It is used here
/// anyway because this is a *failure* path: a machine where
/// `cudaDeviceGetAttribute` fails is a machine where the launch is going to
/// fail too, and answering with a named box puts the failure in the log
/// instead of a `None` three frames up.
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
/// Occupancy is a per-cubin fact, so under the JIT it is
/// [`crate::attn::fa2::decode_blocks_per_sm`] over the `CUfunction` the root
/// produced — which means the point is compiled at PLAN time, before the
/// fire.
///
/// When the query cannot be made this answers `num_sm`: one block per SM, a
/// conservative answer rather than a wrong one. `max_grid_size` bounds
/// `plan::decode::estimate`'s split, so too small yields fewer, larger KV
/// chunks — the unsplit end of the range, which the static short-circuit uses
/// anyway. Too LARGE would be the dangerous direction and cannot happen here.
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
/// module's header records: the estimator has already forced `split_kv` off
/// for these shapes, so the schedule does not depend on the KV lengths.
///
/// Sets [`DecodePlanCache::page_count_independent`], its only producer.
///
/// It carves its own descriptor — a four-call bump allocator over the int
/// workspace, three `IdType` arrays at 16-byte alignment and one scalar at 1,
/// then a page-locked mirror filled from the three static vectors plus
/// `page_size` as the KV chunk size. Neither step goes through
/// [`crate::attn::plan`]: that module implements upstream's *planner*, and
/// this function's whole purpose is not to run it. The bytes land in
/// [`DecodePlanCache::int_upload`] rather than in a `cudaMemcpyAsync`; see
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
    //
    // Carved into a scratch `Vec` and then copied into the cache's pinned
    // buffer in one go: the pin is the H2D's source and its address is what a
    // capture bakes, so it is written whole and never resized under a graph.
    let mut staging = vec![0u8; cursor];
    put_i32s(&mut staging, request_indices_offset, &cache.static_request_indices);
    put_i32s(&mut staging, kv_tile_indices_offset, &cache.static_kv_tile_indices);
    put_i32s(&mut staging, o_indptr_offset, &cache.static_o_indptr);
    // `:189-190`. **A chunk size in TOKENS**, and for an unsplit plan the
    // chunk is one page — which is `page_size` tokens, not 1.
    put_i32s(&mut staging, kv_chunk_size_ptr_offset, &[page_size]);
    if cache.int_upload.fill(&staging).is_err() {
        return Planned::Declined(Decline::Pin);
    }

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
/// The head_dim check is first and stays first; the reason is quoted on
/// [`Decline::HeadDim`]. Then the windowed test, then the short-circuit, then
/// the real planner.
///
/// `max_grid_size` is a parameter and not a query: occupancy is a per-cubin
/// fact, and the caller asks the module that holds the function.
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
    if !windowed_split
        && DecodePlanCache::can_use_static_nonsplit(
            num_requests,
            device.cc_major,
            enable_cuda_graph,
        )
    {
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
    if cache.int_upload.fill(&planned.int_upload).is_err() {
        return Planned::Declined(Decline::Pin);
    }
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
        // `:381`'s `disable_split_kv = !head_dim_supports_cascade_merge(head_dim)`.
        //
        // A split plan leaves partial outputs in `tmp_v` and partial
        // log-sum-exps in `tmp_s` that a cascade merge has to fold into `o`,
        // and the predicate asks whether that merge exists for this head dim.
        // `crate::cascade` carries `PersistentVariableLengthMergeStatesKernel`
        // at all four, compiled by NVRTC out of the vendored `cascade.cuh`,
        // and `fire/merge_states.rs` fires it; every dispatch that returns
        // `Fired::Split` folds before it returns. `x::cascade::HEAD_DIMS` and
        // `x::fa2::HEAD_DIMS` are the same four values in two places, and
        // whoever changes one says which.
        disable_split_kv: !head_dim_instantiated(head_dim.max(0) as u32),
        num_colocated_ctas: 0,
    };
    // `:390-399`, the graph-mode demotion, in the shape the Rust planner
    // makes available. The C++ recomputed a worst-case `max_qo_len` before
    // planning and turned `enable_cuda_graph` off when the tile would not
    // fit; the Rust planner does that arithmetic internally and reports the
    // overflow as an error, so the demotion is a retry: plan in graph mode,
    // and if the carve does not fit, plan again eagerly. The second attempt
    // is the one whose failure is fatal.
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
    // `plan::prefill` computes `avg_packed_qo_len` from
    // `sum_packed_qo_len / batch_size` — a sum over the QO indptr, not a
    // product of totals — and publishes the answer as
    // `PrefillPlanInfo::cta_tile_q`, pinned at offset 24.
    //
    // Reading it back rather than recomputing removes a class of bug rather
    // than a line: the planner **split the batch against this tile**, so a
    // fire that chose its own would index a work list built for a different
    // one. There is no arithmetic here that can drift from the planner's.
    //
    // The graph-mode branch is a different quantity: `max_qo_len`, the worst
    // case where one request holds every token, used only to bound a
    // workspace that must be sized before the batch is known.
    let cta_tile_q = u32::try_from(planned.info.cta_tile_q).unwrap_or(0);

    // The gate. See `Decline::HeadDimTile`: with the tile now coming from the
    // planner, `arith.rs:95`'s `head_dim < 256` makes this unreachable — it is
    // a statement kept against an upstream edit, not a live branch.
    if head_dim as u32 >= 256 && cta_tile_q == 128 {
        return Planned::Declined(Decline::HeadDimTile { head_dim: head_dim as u32, cta_tile_q });
    }

    cache.plan_info = planned.info;
    if cache.int_upload.fill(&planned.int_upload).is_err() {
        return Planned::Declined(Decline::Pin);
    }
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
/// `attention_flashinfer.cu:193-198`, moved to where the launch is: issued
/// immediately before the fire that reads it. The planner has no stream to
/// order against, and a descriptor uploaded on one stream and read on another
/// is a race the C++ avoided by having exactly one stream and not by saying
/// so.
///
/// `int_base_bytes` is added to the DESTINATION and not to the offsets.
/// [`DecodePlanCache::int_upload`] is carved from zero, so a plan sharing an
/// int workspace with another moves as a block and the descriptor's own
/// offsets stay relative — which makes [`DecodePlanCache::set_int_base`] a
/// one-field change.
///
/// The source is a [`PinnedBytes`] and not a `Vec<u8>`: graph capture records
/// the source ADDRESS and performs no copy at all, so a pageable buffer that
/// merely stages before returning is not enough. Fixed capacity, refilled in
/// place, and the address a captured node bakes is the address the next fire
/// writes into. The source must outlive the copy and every replay of it, and
/// does: `int_upload` belongs to the cache, and the cache outlives the graph.
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
/// non-capturing fires are one function. There is no `fire_decode` beside it:
/// the one caller of this pair is `driver-cuda`'s ViT tower, which reaches
/// FA2 by path rather than through a trace statement and does prefill only.
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
        assert_eq!(cache.int_upload.as_slice().len(), 56);
        assert!(!info.split_kv);
        assert_eq!(info.padded_batch_size, 4);

        let at =
            |off: usize| {
                i32::from_ne_bytes(
                    cache.int_upload.as_slice()[off..off + 4].try_into().unwrap(),
                )
            };
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

//! The FlashInfer FA2 host program's plan half, in Rust.
//!
//! North-star §5 step 7's three caches and their factories.
//! `driver-cuda/csrc/attn/attention_flashinfer.cu` calls itself a host program
//! in its own header — *"What it holds is a HOST PROGRAM: six plan factories,
//! two plan-cache lifetimes, and four dispatches"* — and the census agrees:
//! `__global__` 0, `__device__` 0, and one launch, which is
//! `device::attn_score_fold_heads`, ours and already rowed. This module is the
//! planning half of that program; [`crate::fire::kv_paged`] holds the dequant
//! switch its dispatches open with, and the launches themselves resolve
//! against [`kernels_cuda_new::families::fa2`]'s 56 units.
//!
//! # What is here and what is deliberately not
//!
//! Here: the three caches as Rust structs, the two real planner factories over
//! [`kernels_cuda_new::plan`], the static non-split decode short-circuit with
//! both of its environment gates, and the geometry gate that turns an
//! unsupported (head dim, `CTA_TILE_Q`) pair into a named refusal.
//!
//! Not here: the four `switch (cache.head_dim)` dispatches. They need a
//! `#[repr(C)]` mirror of `BatchDecodeParams` and `BatchPrefillPagedParams` —
//! roughly 25 and 40 fields of upstream layout — passed by value through
//! [`kernels_cuda_new::runtime::module::KernelModule::fire_raw`], and a
//! mirror that is wrong in one field is a kernel reading a pointer out of an
//! integer. **Until they exist the C++ file stays and still runs**; nothing in
//! this module is on a live path yet, and the archive is not deleted. That
//! ordering is deliberate: a half-written cache that leaves the file in place
//! costs nothing, and a deleted file with no cache behind it does not link.
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
//! * **`head_dim_supports_cascade_merge`'s `{64, 128, 256, 512}`** (`:376`,
//!   `:985`) is **upstream's** set, and it agrees with
//!   [`kernels_cuda_new::families::fa2::HEAD_DIMS`] by shared origin rather
//!   than by construction. Two facts that happen to match; whoever changes one
//!   says which.

use kernels_cuda_new::plan::{self, Device, Workspace};
use kernels_cuda_new::plan::info::{
    DecodePlanInfo, MlaPlanInfo, PrefillPlanInfo, PrefillPlanSm90Info,
};

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
    HeadDim { head_dim: u32 },
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
    /// valid instantiation, so `families::fa2` names no unit for it.
    ///
    /// # And the archive did not hit it — the guard is one line of arithmetic
    ///
    /// An earlier pass flagged this as a latent `FLASHINFER_ERROR` in the
    /// archive. **That was wrong, and the correction belongs here rather than
    /// in a report nobody reads**:
    /// [`kernels_cuda_new::plan::arith::fa2_determine_cta_tile_q`] is
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
    HeadDimTile { head_dim: u32, cta_tile_q: u32 },
    /// The batch is empty. `num_requests <= 0`.
    NoRequests,
    /// [`kernels_cuda_new::plan`] refused, and which planner did.
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
    Planner(&'static str),
}

impl core::fmt::Display for Decline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HeadDim { head_dim } => {
                write!(f, "flashinfer fa2: unsupported head_dim {head_dim}; the lattice holds {{64, 128, 256, 512}}")
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
        }
    }
}

/// The head dims the FA2 lattice is instantiated over.
///
/// `kernels.def`'s list, and upstream's `head_dim_supports_cascade_merge` set.
/// Stated here as well as in [`kernels_cuda_new::families::fa2::HEAD_DIMS`]
/// because this is the *gate* and that is the *lattice*: they must agree, and
/// [`the_gate_and_the_lattice_agree`] is what makes them.
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
/// (`struct DecodePlanCache;`) and `bind/mod.rs`'s [`DecodePlan`] wrapped a
/// `*mut c_void` created by `pie_x_make_decode_plan` and destroyed by a
/// custom deleter, because *"Rust was never available — the whole reason
/// these exist is a `unique_ptr` with a custom deleter"* (`plan_lifecycle.cpp`
/// lines 9-16). Rust is available now, the deleter is [`Drop`] and there is
/// nothing to release: the fields below are `Vec`s and plain data.
///
/// [`DecodePlan`]: crate::bind::DecodePlan
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
    /// Per-head width. One of [`HEAD_DIMS`].
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
        !force_split_kv_small_enabled()
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
    /// **Not a field of the C++ cache.** It was recomputed at every dispatch
    /// from `plan_info.total_num_rows` because the archive instantiated all
    /// four `NUM_MMA_KV` points and the switch was free. Under the JIT the
    /// tile chooses the UNIT, so it is a planning output and is recorded where
    /// the plan is — see `families/fa2.rs`'s note on what
    /// `DISPATCH_NUM_MMA_KV` cost the archive.
    pub cta_tile_q: u32,
}

impl PrefillPlanCache {
    /// A fresh, unplanned cache. `make_prefill_plan()`, `:101-103`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

// ── The MLA cache ───────────────────────────────────────────────────────────

/// The MLA plan cache, which never had a Rust owner.
///
/// North-star §5 step 7 names it explicitly as the third cache. It had no
/// `pie_x_*` lifecycle and no `bind/mod.rs` handle — the MLA path built its
/// plan inline — so this is the first time it is a type rather than a set of
/// locals, and it is here so that the three caches are one shape.
///
/// **The MLA kernels are not in the FA2 lattice** and are not probed;
/// `families::fa2` holds paged decode and paged prefill only. This struct
/// plans; nothing in this crate yet launches from it.
#[derive(Clone, Debug, Default)]
pub struct MlaPlanCache {
    /// The descriptor.
    pub plan_info: MlaPlanInfo,
    /// Bytes for the H2D, held for the launch.
    pub int_upload: Vec<u8>,
    /// Requests in the batch.
    pub num_requests: i32,
    /// Query heads.
    pub num_q_heads: i32,
    /// Whether anything above was written.
    pub valid: bool,
    /// Byte offset of this plan's descriptor inside the shared int workspace.
    pub int_base_bytes: usize,
}

impl MlaPlanCache {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

// ── The factories ───────────────────────────────────────────────────────────

/// `plan_static_nonsplit_decode`, `attention_flashinfer.cu:135-160`.
///
/// The hundredfold short-circuit. Writes the descriptor directly instead of
/// running FlashInfer's planner, which is legal only under the condition this
/// module's header records: the work estimator has already forced `split_kv`
/// off for these shapes, so the schedule does not depend on the KV lengths.
///
/// Sets [`DecodePlanCache::page_count_independent`], which is the field's only
/// producer.
pub fn plan_static_nonsplit_decode(
    cache: &mut DecodePlanCache,
    kv_page_indptr_h: &[u32],
    num_requests: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    page_size: i32,
    enable_cuda_graph: bool,
    full_attention_variant: bool,
    hnd_layout: bool,
) -> Planned {
    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }
    cache.refresh_static_vectors(num_requests);

    cache.plan_info = DecodePlanInfo {
        enable_cuda_graph,
        split_kv: false,
        padded_batch_size: i64::from(num_requests),
        ..DecodePlanInfo::default()
    };

    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.num_pages_in_batch =
        kv_page_indptr_h.get(num_requests as usize).copied().unwrap_or(0) as i32;
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
/// [`kernels_cuda_new::runtime::module::KernelModule::max_active_blocks_per_sm`]
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
    if !windowed_split
        && DecodePlanCache::can_use_static_nonsplit(num_requests, device.cc_major)
    {
        return plan_static_nonsplit_decode(
            cache,
            kv_page_indptr_h,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            enable_cuda_graph,
            full_attention_variant,
            hnd_layout,
        );
    }

    // The planner takes `i32`; the fire's indptr is `u32`. Widened into the
    // cache's own buffer rather than allocated per step, which is what the
    // C++'s `indptr_h_buf` was for.
    cache.indptr_h_buf.clear();
    cache.indptr_h_buf.extend(
        kv_page_indptr_h.iter().take(num_requests as usize + 1).map(|&v| v as i32),
    );

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
) -> Planned {
    if head_dim < 0 || !head_dim_instantiated(head_dim as u32) {
        return Planned::Declined(Decline::HeadDim { head_dim: head_dim.max(0) as u32 });
    }
    if num_requests <= 0 {
        return Planned::Declined(Decline::NoRequests);
    }

    cache.qo_h_buf.clear();
    cache
        .qo_h_buf
        .extend(qo_indptr_h.iter().take(num_requests as usize + 1).map(|&v| v as i32));
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
        // `attention_flashinfer.cu` never sets either: it does not run POD
        // attention, and it lets the planner search. Stated rather than
        // defaulted because `plan::prefill::Request` derives no `Default` —
        // deliberately, per its own doc: every field is a decision.
        disable_split_kv: false,
        num_colocated_ctas: 0,
    };

    let planned = match plan::prefill::plan(&req, device, workspace) {
        Ok(p) => p,
        Err(_) => return Planned::Declined(Decline::Planner("prefill")),
    };

    // The tile the plan implies, from the same arithmetic the launcher used.
    // `avg_packed_qo_len` is the packed QO rows over the requests they cover;
    // `plan::arith` owns the chooser and this is its only caller here.
    let group = if num_kv_heads > 0 { num_q_heads / num_kv_heads } else { 1 };
    let avg_packed_qo_len =
        i64::from(total_tokens.max(0)) * i64::from(group.max(1)) / i64::from(num_requests.max(1));
    let cta_tile_q =
        plan::arith::fa2_determine_cta_tile_q(avg_packed_qo_len, head_dim as u32, device.cc_major);

    // The gate. See `Decline::HeadDimTile`: this cannot fire from the chooser
    // above, and fires only if a caller substitutes its own tile.
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
    cache.valid = true;

    Planned::Full
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
        assert_eq!(HEAD_DIMS.as_slice(), kernels_cuda_new::families::fa2::HEAD_DIMS);
        for hd in HEAD_DIMS {
            assert!(head_dim_instantiated(hd));
        }
        assert!(!head_dim_instantiated(96), "96 is deliberately absent; see kernels.def");
        assert!(!head_dim_instantiated(0));
    }

    /// The pair the prefill gate exists for cannot come from the chooser.
    ///
    /// `arith.rs:95`'s `head_dim < 256` is the guard, and this is the
    /// assertion that an upstream edit removing it does not pass silently.
    #[test]
    fn the_tile_chooser_never_asks_for_128_at_256() {
        use kernels_cuda_new::plan::arith::fa2_determine_cta_tile_q;
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
}

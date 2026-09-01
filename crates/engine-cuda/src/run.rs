//! The §8 `Run`, on cuda's stream: one fire's dispatch state, and the one
//! function that turns a plan id into a device handle.
//!
//! Long-lived state — the weight and arena tables, the cache pools — arrives
//! pre-built and borrowed: building those tables is the shell's binding
//! business, not this layer's. Fire-lived state — the input bindings, the
//! plan payloads — is owned: a `Run` is constructed per fire and dropped
//! with it.
//!
//! Everything here answers to one rule: a [`KernelError`] is about the
//! backend, never about the plan (`model_exec::error`). A hole in a table,
//! a cache id in a tensor seat, a plan consumed before its plan op — those
//! are integrity failures of the shell or the compiler, and they panic with
//! a sentence instead of dressing up as a backend refusal.
//!
//! # And one word decides whether a region is being RECORDED
//!
//! A `Run` is built by the router with three facts the fire path cannot
//! recompute — `bodied`, the load's `shifted` slice and this composition's
//! [`Admit`] table ([`Run::bodied`]) — and every ceiling in this file hangs
//! off the single predicate they resolve to, [`Run::captured`]:
//!
//! ```text
//! tier 1/2, a CAPTURED region   the key's grid (carve_rows, carve_lanes),
//!                               the key's schedule carve (planning), the
//!                               plane's base pointers (plane_base) and an
//!                               armed live-rows seat (live_at)
//! tier 2, an ISLAND             none of them: this fire's own live geometry,
//!                               byte for byte the launch the eager walk makes
//! tier 3, an eager fire         the same, because `bodied` is false
//! ```
//!
//! **THE TWO BOTTOM ROWS ARE ONE ANSWER, AND THAT IS THE WHOLE OF WHAT TIER 2
//! COST THIS FILE.** Until the tier-2 campaign a composition with one region
//! a graph could not hold was refused admission outright, so `bodied` alone
//! was a region-level answer by accident of the gate above it; a body is cut
//! around its islands now, so the question every ceiling actually wants is
//! per REGION. The predicate is only ever NARROWER than what stood before it
//! ([`Run::captured`] argues that clause by clause), so no composition tier 1
//! could already serve computes anything different.
//!
//! [`KernelError`]: model_exec::KernelError

use std::cell::Cell;

use kernels_cuda::attn::plan::{
    DecodePlan, Device, Live, MlaPlan, PrefillPlan, PrefillPlanSm90, Shape, Toggles, Workspace,
};
use kernels_cuda::linear::lora::Segments;
use kernels_cuda::linear::moe::{ExpertTable, GroupSeat};
use kernels_cuda::{Ctx, KvPool, Pad, RaggedTensor, RecurrentPool, Tensor};
use model_ir::{Def, Dim, GeomKind, Node, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

use crate::dispatch::copy::CopyPlan;
use crate::record::Carve;
use crate::window::{Admit, At, Window, Windows};

/// One loader-resolved weight. Most rows are one dense handle; an mxfp4 bank
/// is two device planes under one `Def::Weight` id. Both shells seat the form
/// the same way now — the table names it instead of refusing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    /// One dense handle, resolved by [`Run::tensor`].
    Dense(Tensor),

    /// A split-plane quantized bank — e2m1 codes beside e8m0 exponents, or
    /// MLX affine codes beside bf16 scales AND zero points — resolved by
    /// [`Run::planes`], never as one tensor.
    ///
    /// **`seat` IS WHICH TIER THE GROUP IS ON RIGHT NOW** (alto streaming §3
    /// item 3, wave B7). The two handles carry the addresses the group was
    /// SEATED at, which is what a `{:?}` prints and what the null-cell arm
    /// reads; a streamed group also carries a base cell the kernel loads the
    /// live pair out of, and a usage counter it notes the routing in. Both are
    /// zero — [`GroupSeat::RESIDENT`] — for a group the store holds whole,
    /// which is every group of a fully-resident load, and then the launch is
    /// byte for byte the launch this row made before the ladder existed.
    Planes {
        codes: Tensor,
        scales: Tensor,
        /// The zero points, for an affine bank whose element is
        /// `code * scale + bias`; `None` for a scheme whose block centres
        /// itself (mxfp4).
        biases: Option<Tensor>,
        seat: GroupSeat,
    },

    /// **A STREAMED ROUTED-EXPERT BANK** (alto design §7, wave D2): a device
    /// slab of fewer slots than the bank has experts, plus the two device
    /// addresses the select kernel needs to read it — the indirection table
    /// (`expert_id -> base address`, entries pointing into the slab or at
    /// pinned host bytes over UVA) and the per-expert usage counters the fire
    /// path `atomicAdd`s into.
    ///
    /// It is a dense row with two numbers on it rather than a second kind of
    /// weight, and that is deliberate: [`Run::tensor`] resolves it to `slab`
    /// like any other handle, so every op that reads a bank as a plain
    /// rectangle keeps working, and only [`Run::expert_bank`] — the MoE
    /// select's own resolution — asks about the two addresses.
    Streamed {
        /// The device slab: `resident` slots at the store's own address.
        slab: Tensor,
        /// Device address of this bank's indirection table.
        table: u64,
        /// Device address of this bank's usage counters.
        counts: u64,
    },
}

/// Loader-resolved weights, one row per `Trace::params` entry —
/// `Def::Weight(i)` resolves to row `i`. `None` marks a param the shell has
/// not bound; resolving such a row is a binding bug and panics.
#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

/// Arena slots at the compiler's offsets, `ValueId`-indexed. Op outputs and
/// merges alike land here: the compiler aliased every merge arm onto one
/// slot and wrote that slot at the merged id's row too, so a φ resolves like
/// any op output. Rows for ids that own no arena slot (inputs, weights,
/// caches, structs) stay `None`.
#[derive(Clone, Debug, Default)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

/// One resolved cache space — the storage pointer and nothing else (design
/// §7); its geometry rides in [`FireBindings`] as declared inputs. On this
/// plane the [`KvPool`] row also carries the graph-padding `row_valid` the
/// writer kernels read — the shell derives it from the same declared inputs
/// when it builds the row. (The write tables the row used to smuggle are
/// gone: `write_page`/`write_offset` are op-named inputs now.)
#[derive(Clone, Copy, Debug)]
pub enum CachePool {
    /// A paged kv space (`CacheRow::Kv`), and which geometry space's tables
    /// address it.
    ///
    /// **THE SPACE RIDES ALONG BECAUSE A GATHERED WINDOW HAS ITS OWN.** A
    /// pool's page bounds and last-page fills are the fire's, seated per
    /// geometry space at bind; a `Fallback::Copy` re-cuts them for its
    /// compacted lanes and seats the result per space too
    /// (`window::GatheredSpace`), so resolving one means knowing which space
    /// this row belongs to. The number is the cache row's own declaration
    /// (`CacheRow::Kv { space }`) — restated here rather than looked up,
    /// because `Run::pool` holds the pool and not the plan's cache table.
    Kv {
        /// The `CacheRow::Kv` space this row's geometry comes from.
        space: u32,
        /// The storage and the tables addressing it.
        pool: KvPool,
    },
    /// A recurrent state space (`CacheRow::State`).
    Recurrent(RecurrentPool),
}

/// Cache-index-indexed pools, aligned with `Trace::caches`.
#[derive(Clone, Debug, Default)]
pub struct CacheTable(pub Vec<CachePool>);

/// The host half of one cache space's geometry — THE cuda duality. The IR
/// names `kv_indptr` as a device input, and [`Run::tensor`] serves it to
/// launches; but the plan builders are host functions that walk its
/// *contents*, and a device handle cannot be read host-side. So the shell
/// binds the same vector twice, and this seat is the host twin.
///
/// **THE TWINS ARE THE SPACE'S; THE READING IS THE SCHEDULE'S.** Shape,
/// window and workspace used to sit here and moved to [`ScheduleSeat`] in
/// C1b, because a page-id space says which page a token lands in and nothing
/// about how wide that row is or how far back a reader may look — and gemma
/// states two answers to both over one sequence.
///
/// Bound only for cache spaces a plan op names; a plan op firing over a
/// space with no planning seat is a binding bug and panics.
#[derive(Clone, Debug)]
pub struct CachePlanning {
    /// Host copy of the space's `GeomKind::Indptr` contents — what
    /// `plan_decode`/`plan_prefill` walk (the builders' `MENLO-SEAM`).
    pub kv_indptr: Vec<i32>,

    /// Host copy of the space's `GeomKind::KvLen` contents — per-request kv
    /// lengths in tokens, the op-named input the sm90 and mla builders walk
    /// (the fa2 builders take it and leave it unread). The same duality as
    /// `kv_indptr`: the device tensor serves launches, this twin serves the
    /// host planners.
    pub kv_len: Vec<i32>,
}

/// What ONE attention schedule is carved for, and where it is staged.
///
/// **KEYED BY THE PLAN VALUE, NOT BY THE SPACE** (build log 20's first
/// blocker). A page-id space says which page a token lands in; it says
/// nothing about how wide that row is or how far back a reader may look, and
/// gemma states two answers to both over one sequence. So the reading and the
/// grant belong to the schedule that was carved for them, and a family that
/// carves two mints two plan values — which is also what makes each one's
/// staged int image its own.
///
/// The reading itself comes off the PLAN OP: `Attention::PlanDecode` and its
/// two siblings state the query heads, kv heads, head width and window their
/// schedule is carved for, and `store::kv::probe` seats what they say. The
/// launches restate their share and the shell refuses a disagreement.
#[derive(Clone, Copy, Debug)]
pub struct ScheduleSeat {
    /// The kv-side shape this schedule is carved at, at the FIRE's lanes;
    /// [`Run::planning`] narrows `num_requests` to the asking node's window.
    /// The consuming ops restate `head_dim`/`kv_heads` and the entries refuse
    /// a disagreement; for a latent (mla) schedule, `head_dim` is the output
    /// head width it sizes at.
    pub shape: Shape,

    /// The sliding window this schedule carved its kv spans for; the entries
    /// check each consumer's stated window against it.
    pub window: Option<u32>,

    /// Where the built schedule's staged image lands.
    pub workspace: Workspace,
}

/// One cache space's planning twin, CUT TO THE WINDOW of the node asking.
///
/// A plan build is per-window work: an all-decode fire's prefill schedule is
/// never built (the walk skips the empty window, design §5 step 4), and a
/// MIXED fire builds both — each over its own lanes. So the builders must not
/// see the fire's whole geometry, or the decode schedule would carve requests
/// for the prefill lanes and the launch would read a schedule wider than the
/// rectangle it was handed.
///
/// The slices borrow the fire-wide host twins rather than copying them: the
/// window is contiguous in lanes (seriation, design §3) and the builders read
/// DIFFERENCES of the boundary vectors, so a slice is the whole adaptation.
/// `shape` is the one field that is rewritten — `num_requests` is the window's
/// lanes, not the fire's, or the BUCKET'S LANE CEILING for a schedule the
/// bodies path is carving whole-fire (`Run::planning`'s `ceiling`), and then
/// the slices run out that far too.
#[derive(Clone, Copy, Debug)]
pub struct Planning<'a> {
    /// The window's slice of `GeomKind::Indptr`'s host contents.
    pub kv_indptr: &'a [i32],
    /// The window's slice of `GeomKind::KvLen`'s.
    pub kv_len: &'a [i32],
    /// The kv-side shape, at this window's request count and origin — or at
    /// the KEY's lane ceiling and lane origin, on the one path that takes
    /// them (`Run::planning`). The STRUCTURE half of what a builder takes.
    pub shape: Shape,
    /// **AND THE ORIGIN AND EXTENT HALF** ([`Live`]) — the same three
    /// numbers plus this window's rows, on the channel that reaches the
    /// device through the staged image and never through a hashed payload
    /// field. Equal to their [`shape`](Planning::shape) twins on every path
    /// but the one that raises `num_requests` and `lane_offset` to the
    /// key's ceilings, which raises those twins and leaves every field here
    /// alone — this is always what the FIRE brought.
    pub live: Live,
    /// **THE ROW COUNT THE SCHEDULE IS CARVED AT** — this window's own rows,
    /// or the sum of its classes' lattice rungs capped at the fire's bucket,
    /// for the three plan kinds that read a row total (`Run::planning`,
    /// chunks 4 and 5 and the ceiling design's Option B; decode is the one
    /// that does not).
    /// The prefill and latent builders' row argument;
    /// [`live`](Planning::live)'s `rows` is the twin this fire
    /// brought, and the two part exactly where `shape.num_requests` and
    /// `live.requests` do — on a hashed field, so that a body's schedule
    /// stops moving when the row total does.
    pub rows: u32,
    /// The sliding window this schedule is carved for.
    pub window: Option<u32>,
    /// Where the built schedule's staged image lands.
    pub workspace: Workspace,
}

/// The geometry one cache space declared: the device seats the ops read,
/// and the host planning twin beside them (the duality [`CachePlanning`]
/// names). Only what the plan names gets bound, so every seat is optional;
/// resolving an unbound seat is a binding bug and panics.
#[derive(Clone, Debug, Default)]
pub struct CacheGeometry {
    /// `RuntimeInput::Geometry { kind: Indptr }`.
    pub indptr: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: Indices }`.
    pub indices: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: SeqLens }`.
    pub seq_lens: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: LastPageLen }`.
    pub last_page_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: KvLen }`.
    pub kv_len: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RowValid }`.
    pub row_valid: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: RequestOfToken }`.
    pub request_of_token: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WritePage }`.
    pub write_page: Option<Tensor>,

    /// `RuntimeInput::Geometry { kind: WriteOffset }`.
    pub write_offset: Option<Tensor>,

    /// `RuntimeInput::Mask`: this space's packed `u8` mask bits, for
    /// `attention.masked` — one `rows x (held + rows)` rectangle per masked
    /// lane, end to end, each starting on a byte boundary, with the causal
    /// bound already folded in ([`crate::mask`]). `None` for a fire no lane
    /// masked, which is what makes the entry's own refusal reachable.
    pub mask: Option<Tensor>,

    /// The host twin the plan builders walk, bound for spaces a plan op
    /// names.
    pub planning: Option<CachePlanning>,
}

/// The dsv4 compressor state `attention.pool_gather` reads beside its cache.
// MENLO-SEAM: no IR seat — the engine binds the slabs it staged for the
// pooled space (the marker at `kernels_cuda::attn::pool::gather`).
#[derive(Clone, Copy, Debug)]
pub struct PoolSlabs {
    /// The rolling kv window state.
    pub state_kv: Tensor,

    /// The rolling score state.
    pub state_score: Tensor,

    /// The absolute-position-embedding plane.
    pub ape: Tensor,
}

/// The engine-bound extras the cuda entries want beside the ops' named
/// operands. No op names these — every seat here is the engine side of a
/// `MENLO-SEAM` marker in `kernels_cuda`, bound from fire state by the
/// arm that carries the matching comment. (The seats the IR reclaimed —
/// `row_valid`, `request_of_token`, the mask bits — resolve as declared
/// inputs on [`CacheGeometry`] now.)
#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    /// `i32`, `[lanes + 1]`: the span table of the mask bits
    /// `attention.masked` names — each FIRE lane's byte offset into the slab
    /// [`CacheGeometry::mask`] holds — bound onto the prefill plan at build
    /// (`plan_prefill`'s `mask_indptr` seam). `None` when this fire carries
    /// no mask; a masked consumer then gets the entry's own typed refusal,
    /// not a panic — mask-lessness is a run-time fact, not a binding hole.
    ///
    /// **FIRE-WIDE HERE, WINDOW-SLICED AT THE SEAM.** The table is indexed by
    /// the SCHEDULE's request number, so the plan-building arm takes
    /// [`Run::mask_indptr`] — this vector cut to that node's lanes — and the
    /// byte offsets inside it stay ABSOLUTE, because the slab they address is
    /// handed to the launch whole. It is the shape `GeomKind::Indices` and
    /// its bounds vector already have, for the same reason: a table whose
    /// entries are not fire rows cannot be sliced by one.
    pub mask_indptr: Option<Tensor>,

    /// The dsv4 compressor slabs, bound when a pooled space exists. One
    /// fire-wide seat: a plan carries at most one pooled space today; this
    /// moves onto [`CacheGeometry`] the day one carries two.
    pub pool_state: Option<PoolSlabs>,
}

/// **What one lane's recurrent state does with the buffer this fire**
/// (`engine::RsVerb`, resolved to addressing).
///
/// The verb minus everything the shell already decided. `Fold` is the absence
/// of a move — the plain path, and the only shape that graph-replays — so it
/// is not a variant a copy has to test for at every layer.
///
/// **THE RUN IS A LIST AND NOT A RANGE** (wave F3-tail). It was
/// `(first_page, pages)` and the page a buffer token lived in was
/// `first_page + token / page_tokens`, which is an arithmetic the runtime
/// cannot honour: its recurrent store materializes a buffer page when the
/// first write reaches it and copies it on write after a fork, so a lane's
/// run is contiguous only by luck. `RsVerb::Buffer::pages` is the physical
/// slot ids in buffer order, and the addressing is one indexing —
/// `pages[token / page_tokens]` — with the same capacity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsMove<'a> {
    /// Fold in the forward: nothing is copied.
    None,
    /// **Scatter this lane's rows into its buffer** (`RsVerb::Buffer`): row
    /// `t` of the lane lands at buffer token `at + t`.
    Scatter {
        /// The lane's buffer run: physical page slot ids in buffer order.
        pages: &'a [u32],
        /// Which buffer token this fire's first row lands at.
        at: u32,
        /// **How many of this fire's rows this lane also FOLDS** (alto design
        /// §6's fused collapse, wave F3b).
        ///
        /// `0` is the pure scatter — the draft whose rejection is host
        /// bookkeeping, and the only value F3 served. Anything else is the
        /// MIXED ROW: the fire writes every row into the buffer AND lands the
        /// durable state on row `fold`, so the next window's speculation
        /// starts from the accepted boundary without a second fire.
        ///
        /// `fold == rows` degenerates to the single-call folding path and
        /// `fold == 0` to the single-call scatter; only a boundary STRICTLY
        /// inside the row takes the two-launch split.
        fold: u32,
    },
    /// **Gather this lane's buffer over its rows** (`RsVerb::FoldBuffered`):
    /// buffer token `t` becomes row `t`, for as many rows as the lane has.
    ///
    /// The lane's row count IS the host's `bound`; `commit_len` truncates the
    /// scan at the accepted length inside it, which is the whole of why the
    /// gather does not need to know that length.
    Gather {
        /// The lane's buffer run, addressed exactly as `Scatter` wrote it.
        pages: &'a [u32],
        /// **Which buffer token the replay starts at** — the buffer's head
        /// (wave F3b).
        ///
        /// The gap F3 documented and did not close: a fold absorbs tokens off
        /// the FRONT of the buffer but can only release whole covered pages,
        /// so a mid-page fold leaves the survivors physically offset and a
        /// replay from buffer token zero would re-fold what the last fold
        /// already took. `Scatter`'s `at` is the same number from the same
        /// origin, which is what lets one page list serve both.
        at: u32,
    },
}

/// **The buffered-activation plane, seated for one fire.**
#[derive(Debug, Clone, Copy)]
pub struct RsSeat<'a> {
    /// The pool and the plan's reading of it.
    pub buffers: &'a crate::store::rs::Buffers,
    /// One verb per lane, in FIRE (seriated) order.
    pub lanes: &'a [RsMove<'a>],
}

impl RsSeat<'_> {
    /// Move one plane's rows for every lane of one window.
    ///
    /// `bounds` is the window-rebased row CSR (`[lanes + 1]`), `rows` the
    /// plane's rectangle as this window sees it, and `lane_offset` what turns
    /// a window request number into a fire lane.
    fn run(
        &self,
        stream: *mut core::ffi::c_void,
        plane: crate::store::rs::Plane,
        lane_offset: u32,
        bounds: &[i32],
        rows: Tensor,
    ) -> crate::error::Result<()> {
        let page_tokens = self.buffers.page_tokens();
        let elem = model_compiler::arena::elem_bytes(crate::store::rs::PLANE_DTYPE)
            .expect("the buffered planes are bf16, which has an element size");
        if u64::from(rows.width) != plane.width {
            return Err(crate::error::Fault::Unbound {
                what: format!(
                    "a buffered plane reserved at {} elements a token is bound {} wide this \
                     fire",
                    plane.width, rows.width
                ),
            });
        }
        for (at, pair) in bounds.windows(2).enumerate() {
            let Some(&verb) = self.lanes.get(lane_offset as usize + at) else {
                continue;
            };
            let (pages, from, count) = match verb {
                RsMove::None => continue,
                RsMove::Scatter { pages, at, .. } => (pages, at, (pair[1] - pair[0]) as u32),
                RsMove::Gather { pages, at } => (pages, at, (pair[1] - pair[0]) as u32),
            };
            if count == 0 {
                continue;
            }
            let capacity = pages.len() as u64 * u64::from(page_tokens);
            if u64::from(from) + u64::from(count) > capacity {
                return Err(crate::error::Fault::Ceiling {
                    what: "rs buffer tokens",
                    need: u64::from(from) + u64::from(count),
                    have: capacity,
                });
            }
            // One contiguous run per (page, plane): the page-major layout is
            // chosen so that this loop is memcpys and not a strided kernel.
            let mut done = 0u32;
            while done < count {
                let token = from + done;
                let page = token / page_tokens;
                let in_page = token % page_tokens;
                let take = (page_tokens - in_page).min(count - done);
                // THE ONE LINE THE LIST CHANGES: the run's page `page` is
                // whatever slot the list names, not `first_page + page`.
                let page_slot = *pages.get(page as usize).ok_or(crate::error::Fault::Ceiling {
                    what: "rs buffer pages",
                    need: u64::from(page) + 1,
                    have: pages.len() as u64,
                })?;
                let slab = self.buffers.row(plane, page_slot, in_page)?;
                let rows_at = rows.ptr
                    + (u64::from(pair[0] as u32) + u64::from(done)) * plane.width * elem;
                let bytes = usize::try_from(u64::from(take) * plane.width * elem)
                    .unwrap_or(usize::MAX);
                let (dst, src) = match verb {
                    RsMove::Scatter { .. } => (slab, rows_at),
                    _ => (rows_at, slab),
                };
                crate::device::copy_d2d(stream, dst, src, bytes)?;
                done += take;
            }
        }
        Ok(())
    }
}

/// What the engine binds each fire, owned by the [`Run`] for its lifetime.
///
/// `tokens`, `positions`, and `geometry` are the op-visible inputs —
/// `RuntimeInput` routes onto them in [`Run::tensor`]. The rest is ambient:
/// the seam `tables`, the pre-probed `device` facts, the once-read
/// `toggles`, and the shell's `capture` policy word the builders take.
///
/// **THE QO BOUNDARIES ARE NOT HERE, AND THAT IS THE MIXED FIRE.** Design §5
/// removed `qo_indptr` as a named input, so ragged views assemble from an
/// ambient boundary vector — but a windowed consumer's boundaries are its
/// OWN, rebased to its sub-rectangle's zero, and one fire-wide vector would
/// be a lie in a fire whose lanes fall in more than one class. So the device
/// handle and the host twin the prefill/mla builders walk both live per
/// window ([`Window`](crate::window::Window)), reached through
/// [`Run::qo_indptr`] and [`Run::qo_indptr_host`].
///
/// **THE FACT WORD IS NOT HERE EITHER.** It used to ride along as one `u64`
/// for the whole fire, for a `Guard::holds` the walk stopped asking when
/// regions grew masks: which classes run a node is `Region::mask`, resolved
/// per region against the window table, and nothing on this side ever read
/// the word. A fire-wide word is exactly the collapse design §0's vocabulary
/// note warns about ("only the old execution contract collapsed the word to
/// per-fire"), so it is gone rather than windowed.
#[derive(Clone, Debug)]
pub struct FireBindings {
    /// `RuntimeInput::Tokens`: ragged `i32`, one id per token.
    pub tokens: Tensor,

    /// `RuntimeInput::Positions`: ragged `i32`, one absolute position per
    /// token.
    pub positions: Tensor,

    /// `RuntimeInput::AdapterRoutes`: `i32`, one adapter id per token row,
    /// `-1` for a row whose lane registered none.
    ///
    /// **HERE AND NOT IN [`FireTables`], BECAUSE AN OP NAMES IT.** Everything
    /// on `tables` is a seat no `Operands` impl mentions — the engine side of
    /// a `MENLO-SEAM` marker. This one is a declared `RuntimeInput` that
    /// `linear.lora_correct` lists among its inputs, so it stands beside
    /// `tokens` and `positions`, which are the other two fire-wide op-visible
    /// vectors keyed by nothing.
    ///
    /// `None` for a fire no lane carried an adapter into, and that absence is
    /// load-bearing: it is the same statement `mask` makes — nothing staged,
    /// no seat bound, and the axis costs the fire zero bytes and zero
    /// launches, because its window is empty and the walk skips it.
    pub adapter_routes: Option<Tensor>,

    /// **THE SECOND ROW AXIS'S THREE SEATS** (multimodal §2, §5.4), and all
    /// three `None` for a fire no lane submitted an image into.
    ///
    /// `patches` is `RuntimeInput::Patches` — the pre-unfolded patch rows,
    /// `[patch_rows, C·T·P²]`; `patch_segments` is the patch axis's own
    /// indptr, `i32`, `images + 1` entries; `patch_routes` says which token
    /// row each tower row scatters into, `i32`, one per patch row.
    ///
    /// **NONE OF THE THREE RIDES THE PINNED STAGING RING.** The ring is
    /// depth-multiplied and its layout is fixed at load, so reserving a patch
    /// rectangle in it would make every text-only load pay `12 MiB × depth`
    /// of pinned memory for a vector it never fills — the §5.4 finding, and
    /// the reason gate (a) is an arithmetic property here rather than a
    /// measurement. These are written inside `enqueue` from pageable memory
    /// (which makes `cudaMemcpyAsync` synchronous in the SOURCE, so the
    /// `Vec` may be dropped immediately) and consumed by kernels the same
    /// `enqueue` launched behind them on the same stream. It is the argument
    /// `Inputs`' schedule grants already make, one axis over.
    ///
    /// The absence is load-bearing exactly as `adapter_routes`' is: nothing
    /// staged, no seat bound, and the tower's window is empty so the walk
    /// never reaches a launch that would read one.
    pub patches: Option<Tensor>,
    /// The patch axis's indptr — see [`patches`](FireBindings::patches).
    pub patch_segments: Option<Tensor>,
    /// The embed merge's destination rows — see
    /// [`patches`](FireBindings::patches). Checked against this fire's token
    /// row count BEFORE the launch (`Fault::PatchRoute`), because
    /// `layout.scatter_rows` cannot check it and an out-of-range entry is an
    /// out-of-bounds device write.
    pub patch_routes: Option<Tensor>,
    /// **THE TOWER'S ROTATION STREAM** (multimodal §6.3):
    /// `RuntimeInput::PatchPositions`, `i32`, `[patch rows, 3]` — one
    /// `(t, h, w)` per patch row, each patch's own `(h, w)` in its image's
    /// grid. Cut from the same submission as the three above and staged in
    /// the same `enqueue`; `None` on the same terms.
    pub patch_positions: Option<Tensor>,
    /// **THE LEARNED POSITION TABLE'S GATHER INDICES** (multimodal §9.2):
    /// `RuntimeInput::PatchEmbedRows`, `i32`, `[patch rows, taps]` — 1 tap on
    /// the native grid, 4 for bilinear, 16 for bicubic. `None` for a plan that
    /// declares no learned position table.
    pub patch_embed_rows: Option<Tensor>,
    /// **AND HOW MUCH OF EACH TAP**: `RuntimeInput::PatchEmbedWeights`, `f32`,
    /// `[patch rows, taps]`. `None` for a NATIVE-grid plan, which reads one
    /// table row per patch through the plain `layout.embed` and weights it by
    /// nothing — the cheap path being the absence of this seat rather than a
    /// rectangle of ones.
    pub patch_embed_weights: Option<Tensor>,
    /// **THE TRUNK'S ROTATION STREAM** (multimodal §6.3):
    /// `RuntimeInput::MropePositions`, `i32`, `[token rows, 3]`.
    ///
    /// On the FIRST axis and not the second, which is why it stands apart
    /// from the four above: the trunk is one region over the whole token
    /// rectangle, so every row of an mrope-declaring fire carries a triple —
    /// a lane with no image carries `(p, p, p)`, which is scalar rope to the
    /// last bit. `None` for a plan that does not declare the stream, which is
    /// every text this engine served before the towers.
    pub mrope_positions: Option<Tensor>,

    /// Per cache space, aligned with `Trace::caches`:
    /// `RuntimeInput::Geometry { space, kind }` routes to that space, and
    /// the plan-building arms route to row `cache`'s planning twin.
    pub geometry: Vec<CacheGeometry>,

    /// Per (RUN, PLAN VALUE), flat at `run * plan_values + value`: the reading
    /// that schedule is carved for and the grant it stages into. `None` for
    /// every value that is not a plan struct some launch consumes.
    ///
    /// **THE RUN IS P4'S FALLBACK REACHING THE GRANT** (design §3). A region
    /// the layout could not seat carves one schedule per interval of its class
    /// set, every one of them built in the prepare pass and read in the
    /// capture pass, so every one of them needs its own staged image — see
    /// [`Inputs::plans`](crate::inputs) for which half of a grant is
    /// per-run and which is shared. For an artifact P4 seated whole this is
    /// one run wide and is the table it always was.
    pub schedules: Vec<Option<ScheduleSeat>>,
    /// How many plan values one run's slice of
    /// [`schedules`](FireBindings::schedules) holds.
    pub plan_values: usize,

    /// The seam extras the arms bind beside the ops' named operands.
    pub tables: FireTables,

    /// **THE OBSERVABILITY SEAT** (`.wiki/alto/attn-score.md` §4), `None` for
    /// a load whose plan declares no `attn.scores` export and for every fire
    /// of a load whose lanes all asked for nothing.
    ///
    /// A `MENLO-SEAM` in the strict sense — no `Operands` impl mentions the
    /// slab, and no `Operands` impl should: the score write is not a value
    /// the graph computes for another node, it is an OBSERVATION the graph
    /// makes on its way past. What the IR names is the capture arm, and the
    /// capture arm is `attention.prefill_lse`, which the plan already
    /// carried.
    ///
    /// It stands beside [`tables`](FireBindings::tables) rather than on it
    /// because the seat carries a list (which value is which plane) and
    /// [`FireTables`] is `Copy`.
    pub scores: Option<crate::scores::ScoreSeat>,

    /// The device facts every builder takes — pre-probed by the shell
    /// (`Device::probe` once at boot, or a stated fallback); the builders
    /// themselves never probe, purity is their design.
    pub device: Device,

    /// The operator toggles `plan_decode` takes — resolved by the shell
    /// once ([`Toggles::from_env`], like `device`'s one probe) and carried
    /// here so no arm ever reads the environment per fire.
    pub toggles: Toggles,

    /// The shell's graph policy word: whether this fire's capture phase
    /// will be captured as a CUDA graph. Builders carve graph-shaped,
    /// padded schedules under it; `PrefillPlan::graph_capturable` answers
    /// whether they managed. Policy stays the shell's — this word only
    /// carries it to the builders.
    pub capture: bool,
}

/// One built plan payload. An enum over the four kinds this plane can be
/// asked to build, not `Box<dyn Any>`: the IR's `StructKind` is closed, and
/// this crate names every payload type at compile time — erasure would buy
/// no generality, only a silent-downcast failure mode. Here a wrong kind is
/// a named panic.
#[derive(Clone, Debug)]
pub enum StructSlot {
    /// `StructKind::AttnDecodePlan`.
    Decode(DecodePlan),

    /// `StructKind::AttnPrefillPlan`.
    Prefill(PrefillPlan),

    /// `StructKind::AttnPrefillPlanSm90` — built when a trace declares its
    /// prefill plan at this kind; the consumer entry (`attn::prefill_sm90`)
    /// still answers a typed refusal, as the old plane did.
    PrefillSm90(PrefillPlanSm90),

    /// `StructKind::MlaPlan`.
    Mla(MlaPlan),
}

/// One fire's dispatch state: the stream context, the resolution tables,
/// the fire bindings, and the plan payloads this fire builds. The shell
/// constructs one per fire and drives the substrate's walk
/// (`model_exec::fire::walk`) over it — prepare phase first (outside any
/// capture), so every plan payload exists and is staged before its
/// consumers enqueue.
pub struct Run<'c> {
    /// The stream and its companions (cuBLAS handle, communicator, jit
    /// cache behind it). Everything this crate does to the device goes
    /// through it, enqueue only.
    ctx: &'c Ctx,

    /// The routing: `Trace::values`, read by [`Run::tensor`] to send each id
    /// to its table.
    values: &'c [ValueDecl],

    /// `Trace::nodes`, for the one thing a resolution cannot do from a value
    /// id alone: read a whole REGION's operands at once.
    ///
    /// A `Fallback::Copy` compacts every rectangle the region touches into
    /// one slab, so it has to enumerate them before the first node runs —
    /// which is a question about the region, not about any one op. Nothing
    /// else here reads it; the walk still hands each arm its own `&Node`.
    nodes: &'c [Node],

    /// `Def::Weight` rows, loader-resolved.
    weights: &'c WeightTable,

    /// `Def::Op` / `Def::Merge` rows, carved at the compiler's offsets.
    arena: &'c SlotTable,

    /// `Def::Cache` rows — pool pointers, resolved through [`Run::pool`] and
    /// [`Run::recurrent`], never through [`Run::tensor`].
    caches: &'c CacheTable,

    /// Plan payloads: filled by the plan-building arms in the prepare phase,
    /// read by the consuming arms afterwards.
    ///
    /// **KEYED BY `(RUN, VALUE)`, NOT BY VALUE**, and the extra key is the
    /// split's other half. A schedule is carved for ONE window — how many
    /// requests it batches, where each one's query rows start, how its work
    /// items divide the kv — so a region P4 could not seat carves one per
    /// interval, and the prepare phase builds all of them before the capture
    /// phase reads any of them. One slot per value would let run 1's builder
    /// overwrite run 0's, and run 0's launch would then read a schedule
    /// describing run 1's requests: not a fault, just wrong logits for the
    /// lanes in the first interval.
    ///
    /// Flat rather than nested, at `run * values + value`: a plan has
    /// thousands of values and a `Vec` per value would be thousands of
    /// allocations per fire, where this is one. The width is
    /// [`Windows::max_runs`] — `1` for every artifact P4 seated whole, which
    /// is the layout this table had before the split existed.
    structs: Vec<Option<StructSlot>>,
    /// **WHICH REGION BUILT EACH SLOT** — parallel to
    /// [`structs`](Run::structs), written by the one writer ([`Run::put`]),
    /// `u32::MAX` for a slot nothing built. [`Run::schedule_shape`] is the
    /// one reader: an ISLAND region's plan is rebuilt every fire and consumed
    /// only by island launches (the mask-family weld — `record::widen`'s
    /// third rule — is what makes "only" a theorem), so its numbers reach no
    /// captured launch and hashing them would demote a body for a difference
    /// no replay can see.
    struct_region: Vec<u32>,
    /// How many values one run's slice of [`structs`](Run::structs) holds.
    values_wide: usize,

    /// This fire's bindings.
    fire: FireBindings,

    /// **The buffered-activation plane**, for a fire that carries one — a
    /// per-lane RS verb and the pool it addresses ([`Run::buffered`]).
    rs: Option<RsSeat<'c>>,

    /// Every region's window, resolved once per fire from the composition's
    /// class table.
    windows: &'c Windows,

    /// Which region the walk is inside and which run of its window, written
    /// by [`Cursor`](crate::window::Cursor) — the shell's `Sink` — before the
    /// region's nodes are dispatched and before each launch of them. **THIS
    /// IS THE WHOLE MIXED-FIRE MECHANISM**: it turns every resolution below
    /// from "the fire's rectangle" into "this node's window of it".
    place: &'c At,

    /// P6's side-stream contexts, in stream order: `side[0]` is stream 1.
    ///
    /// **EMPTY IS THE EAGER MODE AND IT IS NOT A DEGRADATION.** A `Run` built
    /// with no side contexts fires everything on the main one, which is the
    /// serialization of the compiler's dependency DAG — a legal schedule of it
    /// by construction, and the golden every recorded fire is diffed against.
    side: &'c [&'c Ctx],

    /// Which stream the walk is on, written by the same `Cursor` that writes
    /// [`place`](Run::place), at the same instant and for the same reason.
    stream: &'c Cell<u32>,

    /// **THE CONTEXT A CONDITIONAL BODY'S LAUNCHES LAND ON**, when this load
    /// opened one (`Context::open_conditional`).
    ///
    /// Not a member of [`side`](Run::side) even though it is built exactly
    /// like one, because it is not a stream ASSIGNMENT: no region names it and
    /// `model_compiler::stream` cannot reach it. The cursor writes
    /// [`window::BODY`](crate::window::BODY) into the cell for exactly the
    /// span between a conditional's `cond_begin` and its `cond_end`, which is
    /// the span whose launches belong in the child graph.
    body: Option<&'c Ctx>,

    /// The `Fallback::Copy` currently in force: which rectangles the region
    /// the walk is inside compacted, and where in the scratch slab each one
    /// landed.
    ///
    /// **WRITTEN BY `Serve::gather`, READ BY [`Run::cut`], AND IT CANNOT GO
    /// STALE.** The walk brackets a copied region's nodes with `gather` and
    /// `scatter`, so the plan is rebuilt immediately before the first node
    /// and is still the current one at the last; `CopyPlan::region` carries
    /// which region it was built for, and `cut` refuses a mismatch rather
    /// than reading another region's slab offsets. A region that is not
    /// copied never consults it — `Window::gathered` is `None` and the cut is
    /// the slice it always was.
    copy: CopyPlan,

    /// **D4'S TWO NUMBERS, AND THE WALK IS WHERE THEY ARE GATED**
    /// (`.wiki/palo/cuda-abi.md` §3, refined form): this fire's total rows and
    /// the lattice point above them. `Pad::default()` — a bucket that is not
    /// above the rows — is a shell that padded nothing, which is
    /// `PIE_CUDA_PAD=off`, a deployment with no lattice, and every `Run` built
    /// before this field existed.
    ///
    /// **HELD HERE RATHER THAN STAMPED ON THE CONTEXT ONCE PER FIRE**, because
    /// the question padding turns on is not "which fire" but "which REGION" —
    /// see [`Run::ctx`]. A pad the shell wrote onto the context at the top of
    /// the fire would still be there when the walk stepped into a windowed
    /// region, and the entry would then be left inferring from an extent
    /// whether its launch owns the fire's tail. An extent cannot answer that:
    /// a window's rows and a fire's rows are both `u32`, and they can be
    /// equal. The window can answer it, and the walk is holding one.
    pad: Pad,

    /// **IS THIS FIRE A BODY'S?** (the bodies design's chunk 2b-ii) — the
    /// shell's `Prepared::bodied`, carried in rather than re-asked, because
    /// the answer decided in `prepare` is the same answer that put the
    /// live-rows words into the staging slot.
    ///
    /// `false` is every fire the EAGER path serves, and it is the SHORT
    /// CIRCUIT of [`Run::plane_base`]: a fire that staged no seat resolves
    /// exactly the pointers it always resolved, so the eager walk is byte for
    /// byte the walk it was.
    bodied: bool,

    /// **WHICH REGIONS ADDRESS OFF THE SEAT'S START** — `exports::regions_shifting`
    /// read once at load (`Shell::shifted`), one entry per TEMPLATE REGION:
    /// `true` when every op in it is named by [`crate::SHIFTED`] and therefore
    /// computes over plane rows `[start, start + count)` given the plane's own
    /// base pointers.
    ///
    /// Empty is the safe reading and is what a `Run` nobody told answers: an
    /// out-of-range region reads as NOT shifting, which is
    /// `exports::regions_shifting`'s own rule and
    /// [`Windows::admits`](crate::window::Windows::admits)'s.
    shifted: &'c [bool],

    /// **WHICH REGIONS THIS FIRE'S BODY ACTUALLY HOLDS** —
    /// [`Windows::admits`](crate::window::Windows::admits) as `prepare`
    /// computed it, one entry per TEMPLATE REGION (the tier-2 campaign).
    ///
    /// **THE CORRECTNESS HEART OF SEGMENTED CAPTURE, AND IT IS ONE SLICE.**
    /// A body no longer needs every region to be replayable: the stretches
    /// that are get captured, and the ISLANDS between them are re-issued
    /// eagerly on the same stream, fire after fire. An island's launches are
    /// therefore this fire's launches — they plan, grid and address at the
    /// live geometry the walk is standing in — and every ceiling machine in
    /// this file has to STAND DOWN in one, because a ceiling is a promise
    /// about a key and an island keeps no such promise.
    ///
    /// [`captured`](Run::captured) is the one gate that spends it, and every
    /// consumer asks that rather than [`bodied`](Run::bodied) directly:
    /// [`plane_base`](Run::plane_base), [`live_at`](Run::live_at),
    /// [`carve_rows`](Run::carve_rows), [`carve_lanes`](Run::carve_lanes) and
    /// [`planning`](Run::planning)'s two ceilings. An island that took a
    /// ceiling would bake what must move; an island that armed a seat would
    /// retire rows the launch owns.
    ///
    /// Empty is the safe reading and is what a `Run` nobody told answers: an
    /// out-of-range region is NOT captured, which turns every ceiling off and
    /// leaves the walk exactly the eager walk. It is the same rule
    /// [`shifted`](Run::shifted) above states, in the same direction.
    admits: &'c [Admit],

    /// **WHAT THIS FIRE'S BODY KEY SAYS EACH CLASS MAY BE CARVED OVER** — the
    /// key's [`Ladder`](crate::record::Ladder) beside this fire's own class
    /// table (the ceiling design's Option B).
    ///
    /// `None` for every fire off the bodies path, and then [`Run::planning`]
    /// takes no ceiling at all — which is what keeps the EAGER path byte for
    /// byte the path it was. Set beside
    /// [`bodied`](Run::bodied) and by the same builder, because it is decided
    /// at the same instant and for the same fire: `prepare` builds the key,
    /// and the ladder in the key is the ladder the launches are carved on.
    carve: Option<Carve<'c>>,
}

impl<'c> Run<'c> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        ctx: &'c Ctx,
        values: &'c [ValueDecl],
        nodes: &'c [Node],
        weights: &'c WeightTable,
        arena: &'c SlotTable,
        caches: &'c CacheTable,
        fire: FireBindings,
        windows: &'c Windows,
        place: &'c At,
    ) -> Self {
        Self {
            ctx,
            values,
            nodes,
            weights,
            arena,
            caches,
            structs: vec![None; values.len() * windows.max_runs() as usize],
            struct_region: vec![u32::MAX; values.len() * windows.max_runs() as usize],
            values_wide: values.len(),
            fire,
            rs: None,
            windows,
            place,
            side: &[],
            stream: &place.region,
            body: None,
            copy: CopyPlan::default(),
            pad: Pad::default(),
            bodied: false,
            shifted: &[],
            admits: &[],
            carve: None,
        }
    }

    /// The same `Run`, told this fire's row count and the bucket it rounds up
    /// to — D4's quantization, armed.
    ///
    /// ADDITIVE, AND THE SHELL CHOOSES: a `Run` never handed one pads nothing,
    /// which is the `PIE_CUDA_PAD=off` arm and every caller that predates the
    /// design. `pad.rows` must be the COMPOSITION's rows, not a window's —
    /// [`Run::ctx`] compares a window's span against it and the comparison is
    /// the whole safety argument.
    #[must_use]
    pub fn quantized(mut self, pad: Pad) -> Self {
        self.pad = pad;
        self
    }

    /// **The same `Run`, told that this fire is a BODY'S, which of its regions
    /// a graph actually holds, and which of them can move their own base** —
    /// the launch-plane half of the bodies design's chunk 2b (its gate half is
    /// `Windows::admits`).
    ///
    /// ADDITIVE, AND THE SHELL CHOOSES, on [`quantized`](Run::quantized)'s
    /// terms: a `Run` never handed this pre-shifts every windowed operand and
    /// arms the seat only where [`whole_fire`](Run::whole_fire) holds, which
    /// is what every path but the bodies one wants and what every caller that
    /// predates this chunk got.
    ///
    /// **FOUR FACTS AND NOT ONE**, because they are answered at four
    /// different instants. `bodied` is this FIRE's — `Prepared::bodied`,
    /// decided in `prepare` beside the staging that wrote the seat's words.
    /// `shifted` is the LOAD's, one entry per template region, read once when
    /// the artifact was baked. `admits` is this COMPOSITION's — the same
    /// `Windows::admits` table the gate above was decided on, which is what
    /// tells the walk which regions the body will actually hold and which ones
    /// it will re-issue eagerly ([`captured`](Run::captured), the tier-2
    /// campaign). Handing them together is what makes the gate's question and
    /// the launch's the same question: the shell cuts a body on
    /// `admits(rows, shifted)` and the walk carves, seats and shifts on the
    /// same two slices, so a region the gate says a graph holds is a region
    /// the walk moves — and a region it calls an ISLAND is one the walk leaves
    /// exactly where the eager path leaves it.
    ///
    /// `carve` is the KEY's — `Prepared::ladder`, built in `prepare` off the
    /// same composition the key was built off — and it is handed here for the
    /// same reason: the ceilings [`planning`](Run::planning) carves at have to
    /// be the ceilings the key spells, and a second reading of the lattice on
    /// this side is a second answer waiting to disagree with the one the
    /// cache is keyed on. `None` beside `bodied == false` is the only
    /// combination any caller states, and both halves short-circuit the same
    /// block.
    #[must_use]
    pub fn bodied(
        mut self,
        bodied: bool,
        shifted: &'c [bool],
        admits: &'c [Admit],
        carve: Option<Carve<'c>>,
    ) -> Self {
        self.bodied = bodied;
        self.shifted = shifted;
        self.admits = admits;
        self.carve = carve;
        self
    }

    /// **MAY THE REGION THE WALK IS STANDING IN BE HELD BY A GRAPH?** — this
    /// fire's `bodied` word AND this region's
    /// [`Admit`](crate::window::Admit), and the one gate every ceiling in
    /// this file hangs off (the tier-2 campaign).
    ///
    /// **IT IS `bodied` PER REGION, AND THAT IS THE WHOLE CHANGE.** Until
    /// tier 2 a fire was a body's or it was not, because a composition with
    /// one unrecordable region was refused admission outright — so `bodied`
    /// alone was a region-level answer by accident of the gate above it. It
    /// is not any more: a body is captured in SEGMENTS around its islands,
    /// and an island's launches are re-issued eagerly every fire. So the
    /// question every ceiling actually wants is this one, and asking
    /// `bodied` would hand an island the key's grid, the key's schedule
    /// carve and an armed seat — a launch gridded past what the fire brought,
    /// over a scratch rectangle sized at what it brought, with a seat telling
    /// the kernel to start somewhere the gather did not put anything.
    ///
    /// **AND IT IS ONLY EVER NARROWER THAN THE PREDICATES BELOW IT.** A
    /// region [`plane_base`](Run::plane_base) admits is Captured by
    /// construction — its window is not gathered, carries no segment list and
    /// its region shifts, which is `Admit::Captured`'s second arm — and so is
    /// one [`whole_fire`](Run::whole_fire) admits. So this clause changes no
    /// answer on any composition tier 1 could already serve, and turns every
    /// ceiling off on exactly the regions tier 2 newly admits INTO a body
    /// without being able to record them.
    fn captured(&self) -> bool {
        self.bodied
            && matches!(
                self.admits.get(self.place.region.get() as usize),
                Some(Admit::Captured)
            )
    }

    /// **The buffered-activation plane, for a fire that carries one** (alto
    /// design §6, wave F3).
    ///
    /// A builder rather than a [`FireBindings`] field because it is the one
    /// piece of a fire that is not bound per VALUE: it is a per-LANE verb plus
    /// a pool, read by exactly the two dispatch arms that touch an
    /// in-projection plane, and a fire whose every lane folds carries none of
    /// it at all.
    pub fn buffered(mut self, rs: RsSeat<'c>) -> Self {
        self.rs = Some(rs);
        self
    }

    /// **Scatter or gather this operand's rows, if it is a buffered plane.**
    ///
    /// Called by the dispatch arms of the two ops that READ an in-projection
    /// plane, immediately before they read it. A value no plan buffers, or a
    /// fire with no buffered lane, is one `Option` test and nothing else.
    ///
    /// The window arithmetic is the same one the launch does: this window's
    /// request `r` is fire lane `lane_offset + r`, and its rows begin at the
    /// window-rebased `qo_indptr_host()[r]`.
    ///
    /// # Errors
    ///
    /// The shell's fault for a page slot past the pool, a buffer run too
    /// short for the tokens the verb named, or the copy.
    pub(crate) fn rs_move(
        &self,
        op: &'static str,
        id: ValueId,
        rows: Tensor,
    ) -> Result<(), kernels_cuda::Error> {
        let Some(seat) = self.rs.as_ref() else {
            return Ok(());
        };
        let Some(plane) = seat.buffers.planes().of(id) else {
            return Ok(());
        };
        let span = self.window().span;
        let bounds = self.qo_indptr_host();
        // **AND THE RECTANGLE IS THE WINDOW'S, EVEN WHERE THE OP'S IS NOT**
        // (the chunked-arm wave). `bounds` is the window-REBASED CSR and
        // `RsSeat::run` adds it to `rows.ptr`, so the pointer beside it has to
        // count from the window's first row. A shifting region hands its ops
        // the PLANE's base (`Run::cut` under `plane_base`) because the KERNEL
        // adds `win[1]`; this copy is host arithmetic and adds nothing, so it
        // takes the window back — `Run::windowed`'s exact reason, and the
        // identity on every path but a plane-based one.
        let rows = self.windowed(rows);
        // The shell's fault becomes the KERNEL plane's, because that is the
        // channel this arm answers on: the dispatch arm that calls this is
        // typed `Result<(), kernels_cuda::Error>` like every entry beside it,
        // `error::kernel` lifts the whole arm into the contract, and the walk
        // turns the contract's `KernelError` back into a `Fault` one frame up
        // (`Fault::from`). Nothing is lost across the three but the variant,
        // and the sentence — which is what a caller reads — is carried whole.
        seat.run(self.ctx.stream(), plane, span.lane_offset, bounds, rows)
            .map_err(|fault| kernels_cuda::Error::Backend {
                op,
                detail: fault.to_string(),
            })
    }

    /// `Trace::values`, for the copy plan's own shape reading.
    pub(crate) fn values(&self) -> &'c [ValueDecl] {
        self.values
    }

    /// `Trace::nodes`, for the same.
    pub(crate) fn nodes(&self) -> &'c [Node] {
        self.nodes
    }

    /// Which region of the template the walk is inside.
    pub(crate) fn at_region(&self) -> u32 {
        self.place.region.get()
    }

    /// The FIRE-WIDE rectangle a value names, before any window is applied —
    /// what a copy's gather reads from and its scatter writes back to.
    pub(crate) fn uncut(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// Take the copy plan the region's gather just built.
    pub(crate) fn seat_copy(&mut self, plan: CopyPlan) {
        self.copy = plan;
    }

    /// The same `Run`, told where P6's side streams are.
    ///
    /// ADDITIVE, AND THE CALLER CHOOSES PER PASS. `record.rs` builds one `Run`
    /// per fire and walks it twice — eagerly for the numbers, then capturing —
    /// and only the capturing walk is handed the streams, because a capture is
    /// where an event is a graph edge and an eager fire is where it would be a
    /// real synchronization bought for nothing.
    #[must_use]
    pub fn across(mut self, side: &'c [&'c Ctx], stream: &'c Cell<u32>) -> Self {
        self.side = side;
        self.stream = stream;
        self
    }

    /// The same `Run`, told which context a conditional body's launches go on.
    ///
    /// **IT SEATS THE STREAM CELL TOO, AND THAT IS NOT A CONVENIENCE.**
    /// `Run::new` parks the region cell in the stream field so the type has
    /// something to hold, and [`Run::ctx`] reads the stream field only when
    /// somebody has since replaced it — reading a region index as a stream
    /// index is the bug that guard exists to make impossible. A `Run` given a
    /// body context is one whose cursor writes stream numbers into `stream`,
    /// so the two arrive together or neither does.
    #[must_use]
    pub fn conditional(mut self, body: &'c Ctx, stream: &'c Cell<u32>) -> Self {
        self.body = Some(body);
        self.stream = stream;
        self
    }

    /// The stream context the node being dispatched fires on, for the arms.
    ///
    /// **THE ONE PLACE A SIDE STREAM ENTERS THE DISPATCH PLANE**, and it is a
    /// lookup rather than a decision: which stream a region belongs on was
    /// decided once, at compile, by `model_compiler::stream`, and the `Cursor`
    /// wrote it into a cell before this region's first node. A stream the load
    /// never opened resolves back to the main one — the cursor has already
    /// stopped the walk with `Fault::Unbound` naming it, and answering a null
    /// handle here would turn that refusal into a launch.
    pub(crate) fn ctx(&self) -> &'c Ctx {
        // The eager reading, asked first and by the field that MEANS it: a
        // `Run` with no side contexts was never given a stream cell either,
        // and `new` seats the region cell there so the type has something to
        // hold. Reading a region index as a stream index is exactly the bug
        // this line exists to make impossible.
        let ctx = match self.body {
            // **THE CONDITIONAL BODY IS ASKED FIRST**, because it is the one
            // reading that is not a stream ASSIGNMENT: the sentinel is written
            // for the span between a `cond_begin` and its `cond_end` and says
            // "this launch belongs in the child graph", which outranks
            // whatever stream the region was baked onto. A conditional region
            // is never forked anyway (`model_compiler::stream::forkable` has
            // read `lowering == AlwaysLaunch` since D1), so the two can never
            // both have something to say.
            Some(body) if self.stream.get() == crate::window::BODY => body,
            _ if self.side.is_empty() => self.ctx,
            _ => match self.stream.get() {
                0 => self.ctx,
                n => self.side.get(n as usize - 1).copied().unwrap_or(self.ctx),
            },
        };
        // **AND THE SECOND THING THIS LOOKUP ANSWERS: MAY THIS REGION PAD?**
        // (`.wiki/palo/cuda-abi.md` §3's boundary.) Every dispatch arm reaches
        // its context through here and does so with the cursor already on the
        // node it is about to launch, so this is the last instant at which the
        // WINDOW is still in hand and the first at which an entry could ask.
        // Deciding it here rather than in the entry is not tidiness: an entry
        // sees an extent, and an extent cannot distinguish the fire's rows
        // from a window that happens to hold as many.
        ctx.arm(self.here());
        // **AND THE THIRD: WHERE THIS REGION'S LIVE ROW COUNT IS READ FROM**
        // (bodies design). The same instant and the same argument as the pad —
        // the cursor is on the node, the window is in hand, and an entry can
        // see neither. `0` is the disarmed seat and is what every fire this
        // shell serves today arms, because no caller stages the words yet.
        ctx.arm_stage(self.live_at());
        ctx
    }

    /// **THE PAD THIS REGION IS ALLOWED**, and the argument for every clause.
    ///
    /// D4 pads an opaque callee's `M` up to the bucket so that cuBLASLt's
    /// unpublished shape→kernel table stops following the batch. What that
    /// buys is bought with the rows `[rows, bucket)`, and those rows are safe
    /// to scribble on exactly when they are the FIRE's tail — reserved by the
    /// arena at `max_tokens`, promised by P0 to be above every bucket, and
    /// spoken for by nobody. A WINDOWED launch's tail is a different thing
    /// entirely: the rows above a window are the next class's rows of the same
    /// column, and under `ArenaMap::co_tenants` or a merge those are somebody's
    /// real bytes. Padding one is a clobber that computes and never faults.
    ///
    /// So a region pads only if its window IS the whole fire:
    ///
    /// * **`row_offset == 0` and `rows >= the fire's`** — the mask covers every
    ///   class that has rows, read off the window the shell built FROM the
    ///   mask (`Windows::of`). This is the clause the entry could not make: it
    ///   compares a window's span against the composition, where an entry can
    ///   only compare one extent against another.
    /// * **not gathered** — a `Fallback::Copy` compacted its rows into a
    ///   scratch slab cut to the rows it gathered, so past them is the next
    ///   thing in the slab rather than a reserved tail. (A gathered window
    ///   cannot span the whole fire anyway — gathering needs two intervals and
    ///   two intervals need a gap — but the clause is written because that is
    ///   an argument, and this is a bounds check.)
    /// * **no segment list** — a `Fallback::Grouped` window's span is the UNION
    ///   of its intervals and the gaps hold foreign rows, so its span covering
    ///   the fire says nothing about the rows it owns.
    ///
    /// A region that fails any clause gets `Pad::default()`, which
    /// [`Ctx::opaque_rows`] answers with the extent it was handed.
    fn here(&self) -> Pad {
        if self.pad.bucket <= self.pad.rows {
            return Pad::default();
        }
        if self.whole_fire() { self.pad } else { Pad::default() }
    }

    /// **IS THIS REGION'S WINDOW THE WHOLE FIRE?** — the three clauses
    /// [`here`](Run::here) argues, on their own because two callers ask them.
    ///
    /// A predicate about the WINDOW and not about the pad: the bucket
    /// precondition stays in `here`, where it belongs, because whether a
    /// deployment declared a lattice has nothing to do with whether this
    /// launch owns the fire's tail.
    fn whole_fire(&self) -> bool {
        let window = self.window();
        window.span.row_offset == 0
            && window.span.rows >= self.pad.rows
            && window.gathered.is_none()
            && window.segs() == 0
    }

    /// **DOES THIS REGION GET ITS PLANE'S BASE INSTEAD OF ITS WINDOW'S?** —
    /// the bodies design's chunk 2b-ii, and the ONE predicate the two halves
    /// of that chunk share.
    ///
    /// [`whole_fire`](Run::whole_fire) is the question "is there nothing to
    /// shift"; this is the question "may the shift be left to the DEVICE". The
    /// two are answered at the same instant off the same window and they are
    /// deliberately not each other's negation: a whole-fire window has
    /// `row_offset == 0` and shifts by nothing either way, so a fire can
    /// satisfy both and the callers below arm on the union.
    ///
    /// **FOUR CLAUSES, AND EVERY ONE OF THEM IS LOAD-BEARING.**
    ///
    /// * **this region is one a graph HOLDS** ([`captured`](Run::captured)).
    ///   The seat's words are only in the staging slot for a fire the shell
    ///   routed to a body (`inputs::Fire::live`), and a launch handed a plane
    ///   base under a DISARMED seat reads `win == nullptr`, takes the whole
    ///   extent from row zero of the plane, and computes somebody else's
    ///   rows. So this is asked first and it is what keeps the EAGER path
    ///   byte-identical — and since the tier-2 campaign it asks per REGION,
    ///   because an ISLAND of a segmented body is re-issued eagerly and must
    ///   resolve exactly the pointers the eager walk resolves. The two SHAPE
    ///   clauses below are inside `Admit::Captured` as well, so this is
    ///   narrower than it looks; they stay spelled because this is the
    ///   launch's own reading.
    /// * **`shifted[this region]`** — every op in it is on [`crate::SHIFTED`],
    ///   so every one of them reads `win[1]` and addresses `start + r`. One
    ///   guard-only op in the region and the whole region's launches want the
    ///   pre-shifted pointer, which is why `exports::regions_shifting` asks
    ///   ALL and why a region index this slice does not hold reads as `false`.
    /// * **not gathered** — a `Fallback::Copy`'s rows were compacted into a
    ///   scratch slab and numbered from ITS zero. There is no offset into the
    ///   fire's plane that names them, so `start + r` has nothing to mean.
    /// * **no segment list** — a `Fallback::Grouped` window's span is a UNION
    ///   with foreign rows in the gaps, and `(count, start)` describes one
    ///   interval, which a union of intervals is not.
    ///
    /// The last two are `Windows::admits`'s two SHAPE refusals, spelled again
    /// here rather than inherited, because this is the launch's reading and
    /// that is the host's — and the day they disagreed this one would be the
    /// one that ran.
    fn plane_base(&self) -> bool {
        if !self.captured() {
            return false;
        }
        if !self
            .shifted
            .get(self.place.region.get() as usize)
            .copied()
            .unwrap_or(false)
        {
            return false;
        }
        let window = self.window();
        window.gathered.is_none() && window.segs() == 0
    }

    /// **WHERE THIS REGION'S LIVE ROW COUNT IS READ FROM**, or `0` — the
    /// staged-geometry seat's address, and [`here`](Run::here)'s twin.
    ///
    /// `here` says how far ABOVE its rows a launch may write; this says how
    /// far below its extent a launch may stop. Both are answered off the same
    /// window at the same instant by the same lookup, and both are answered
    /// here rather than in an entry for the same reason: an entry sees one
    /// extent, and an extent cannot tell a fire's rows from a window that
    /// happens to hold as many.
    ///
    /// **AND THE BOUNDARY IS [`here`](Run::here)'S IN TWO CLAUSES AND WIDER IN
    /// THE THIRD.** The pair this address points at is `(count, start)`, and a
    /// window whose SHAPE is not an interval is not a window either word
    /// describes:
    ///
    /// * **gathered** — a `Fallback::Copy`'s span is the COMPACTED rectangle's
    ///   rows, numbered from its own zero in a scratch slab. A kernel retiring
    ///   `r >= win[0]` off that count would be reading the wrong rectangle's
    ///   geometry, so a gathered window arms nothing.
    /// * **grouped** — a `Fallback::Grouped` window's span is the UNION of its
    ///   intervals and the gaps hold foreign rows, so the count says nothing
    ///   about the rows the launch owns; [`Run::segments`] is what carries
    ///   that, and it carries it as an operand.
    /// * **and a window whose count is not its own rows** — which is what the
    ///   first two are instances of, and is the whole of what survives. An
    ///   OFFSET window is no longer refused: chunk 2b-ii moved the refusal
    ///   from "the window must begin at the fire's zero" to "the region's
    ///   every op must read the seat's start", which is
    ///   [`plane_base`](Run::plane_base). A region that does gets its plane's
    ///   base from [`Run::cut`] and the offset from `win[1]`, so arming the
    ///   seat there is not a widening of what a launch may touch — it is the
    ///   only thing that tells the launch where its rows are.
    ///
    /// **THE SAME STAGED WORDS EITHER WAY**, which is why one address serves
    /// both arms: `Windows::live` writes the window's own `[rows, row_offset,
    /// lanes, lane_offset]` at every (region, run), and for a whole-fire
    /// window both offsets are zero — so a region admitted by `whole_fire`
    /// reads a start of 0 on either axis and addresses exactly the plane rows
    /// and the fire lanes its pre-shifted-by-nothing pointers already named.
    /// Nothing about the whole-fire arm moves.
    ///
    /// The seat is asked FIRST and the window second, which is the cheap
    /// order: an unbound fire answers `0` without resolving a window at all,
    /// so a shell nobody armed pays this lookup nothing per dispatched node.
    fn live_at(&self) -> u64 {
        let at = self
            .windows
            .live_at(self.place.region.get(), self.place.run.get());
        if at == 0 || !self.captured() || !(self.whole_fire() || self.plane_base()) {
            0
        } else {
            at
        }
    }

    /// **HOW MANY ROWS A LAUNCH IN THIS REGION IS GRIDDED OVER**, or `None`
    /// for every launch that is gridded at its window's own live span — which
    /// is every fire off the bodies path and every deployment with no lattice.
    ///
    /// **THIS IS THE GRID-AT-CEILING SEAM** (the tier-1 key-collapse wave,
    /// chunk B), and it is the sentence that turns `record::Body::grids` from
    /// a measurement into a function of the key. A body is captured once and
    /// replayed by every fire of its `record::BodyKey`; a launch RECORDED at
    /// this fire's live rows can only serve a fire with no more of them,
    /// because a grid is baked into the graph and no staged word can add a
    /// block to it. So a key whose fires wander in row count used to climb —
    /// capture small, miss, re-capture larger — and since the rungs became
    /// canonical ceilings a key holds a whole LATTICE STEP's worth of splits,
    /// which made the climb long enough to be the dominant cost. Issuing the
    /// launch at the ceiling the key already spells retires the climb: every
    /// fire of the key grids the same, so no in-key fire can outgrow a
    /// recorded grid and `record::grew_past` goes back to being a belt.
    ///
    /// **AND THE SEAT IS THE WHOLE CORRECTNESS ARGUMENT.** The rows between
    /// this fire's own and the ceiling are launched and then RETIRED: every
    /// seated entry's first line is `r >= win[0]` against the live-rows word
    /// this fire staged (`crate::window::Windows::live`,
    /// `kernels_cuda::Ctx::arm_stage`), so a block standing above the fire's
    /// count returns before it addresses anything. What makes that available
    /// exactly here is [`live_at`](Run::live_at): the seat is armed for a
    /// region that is [`whole_fire`](Run::whole_fire) or
    /// [`plane_base`](Run::plane_base) and for no other, which is precisely
    /// the gate below. The two are one predicate written twice on purpose —
    /// the day they part, a launch would be gridded past a count nothing
    /// retires, and that is the shape of the bug this note exists to prevent.
    ///
    /// **THREE CLAUSES, AND THE THIRD IS WHERE THE NUMBER COMES FROM.**
    ///
    /// * **this region is one a graph HOLDS** ([`captured`](Run::captured)).
    ///   Nothing else staged a seat it may read, so nothing else has a
    ///   retirement; the EAGER path grids exactly what it always gridded, and
    ///   so does an ISLAND of a segmented body, which is the same statement
    ///   one wave further on — an island's launches are re-issued every fire
    ///   at the fire's own rows, and a ceiling there would grid past a
    ///   scratch rectangle sized at those rows. **AND IT NOW
    ///   CARRIES THE PAD WITH IT**: `Shell::prepare`'s gate refuses to record
    ///   a body at all on a shell whose pad is off, so `bodied` implies an
    ///   armed lattice point and there is no second clause to ask. The clause
    ///   that used to stand here — `pad.bucket > pad.rows`, "the shell
    ///   actually quantized" — was written for `PIE_CUDA_PAD=off` and could
    ///   only ever be reached by that arm and by ONE other fire: the padded
    ///   fire whose rows land exactly on its bucket. That fire is a fire of
    ///   the same `record::BodyKey` as every other split of the point, so
    ///   disarming its ceilings made its grids and its schedules follow its
    ///   own split — which is precisely what this wave exists to stop, and it
    ///   is the split `Shell::arm_bodies` synthesizes by construction.
    /// * **and the region owns a retirement**, which is `whole_fire ||
    ///   plane_base` above.
    ///
    /// **THE NUMBER ITSELF IS THE CARVE'S, AND IT IS TWO ANSWERS BECAUSE A
    /// WINDOW IS TWO THINGS.** A WHOLE-FIRE window takes `pad.bucket` — the
    /// same number [`Ctx::opaque_rows`](kernels_cuda::Ctx) has padded this
    /// region's GEMM `M` to since D4, so the ceiling is not a new promise
    /// about the arena's tail but the one that number already made. A WINDOWED
    /// one takes its own classes' rungs — `record::Carve::ceiling`'s `own`,
    /// capped at the bucket, which is `Planning::rows`'s expression exactly,
    /// because a window's carve and its grid have to be the same rectangle.
    /// Both are functions of the `record::BodyKey`; neither reads this fire's
    /// split.
    ///
    /// **AND NEITHER ANSWER ADDS THE PREFIX.** `Carve::ceiling` also hands
    /// back how many rows stand IN FRONT of this window, and the row axis
    /// never spends it: the pointer a shifting region is handed is the
    /// PLANE's base and the kernel adds this fire's own `win[1]`, so the rows
    /// a launch touches are `[row_offset, row_offset + live)` and the extent
    /// is only what the grid is sized at. That is why the top of the lattice
    /// is not a corner here: `own.min(bucket)` never exceeds the bucket, the
    /// arena column is carved at the bucket
    /// (`Shell::enqueue_on`'s `FireRows`), and `before + own` — which CAN
    /// reach `bucket + min(lane_ceiling, bucket)` — is a LANE-axis reach and
    /// is spent, clamped, in [`carve_lanes`](Run::carve_lanes) and
    /// [`planning`](Run::planning).
    ///
    /// The `filter` is the belt: a ceiling under this window's own rows would
    /// be a launch that stopped short of the fire, which is the one direction
    /// that is not merely wasteful, so a carve that cannot dominate the span
    /// is not taken at all.
    fn carve_rows(&self) -> Option<u32> {
        if !self.captured() {
            return None;
        }
        debug_assert!(
            self.pad.bucket >= self.pad.rows,
            "a bodied fire carries an armed pad, and an armed bucket holds the \
             fire's {} rows; this one spells {}",
            self.pad.rows,
            self.pad.bucket,
        );
        let whole = self.whole_fire();
        if !(whole || self.plane_base()) {
            return None;
        }
        let span = self.window().span;
        let rows = if whole {
            self.pad.bucket
        } else {
            let (_, own) = self.carve.and_then(|carve| carve.ceiling(span))?;
            own.min(self.pad.bucket)
        };
        (rows >= span.rows).then_some(rows)
    }

    /// **HOW MANY REQUESTS A LANE-GRIDDED LAUNCH IN THIS REGION IS GRIDDED
    /// OVER**, or `None` for the window's own lane count —
    /// [`carve_rows`](Run::carve_rows)'s twin on the other axis, and the
    /// grid-at-ceiling wave's second half.
    ///
    /// **FOUR LAUNCHES IN THE WHOLE TREE COUNT THEIR WORK IN REQUESTS**, and
    /// they are the four chunked recurrent arms — `ssm_causal_conv1d_chunked`,
    /// `ssm_gated_delta_chunked`, `ssm_kda_chunked`, `ple_ngram_ids_chunked`.
    /// Every other seated entry grids on ROWS and takes the ceiling above.
    /// Those four take theirs from the LENGTH of the ragged CSR they are
    /// handed (`indptr.rows - 1`), which is why this answer is delivered as a
    /// wider vector rather than as a number: [`ragged_lanes`](Run::ragged_lanes)
    /// is the one caller and the dispatch arms name it.
    ///
    /// **THE VECTOR STAYS THE WINDOW'S OWN REBASED CSR, AND THAT IS NOT THE
    /// TREATMENT [`ragged_q`](Run::ragged_q) TAKES.** FA2 reads its request
    /// number out of a STAGED datum, so its lane axis could be moved to the
    /// fire's own numbering wholesale; these four read their request off
    /// `blockIdx`, and `attn/ssm.cuh` splits what that ordinal indexes: the
    /// window's rebased CSR at `r`, the FIRE's per-lane tables at `r + win[3]`,
    /// the activation planes at `+ win[1]`. Hand them the ABSOLUTE vector and
    /// the split breaks in the one place it cannot be seen — `qo_absolute[r]`
    /// would be fire lane `r`'s row where the kernel means this window's
    /// `r`-th, and `win[1]` would be added to it on top. The two agree only
    /// where `lane_offset` is zero, which is exactly the window that needed no
    /// help. So what moves is the DECLARED LENGTH and nothing else.
    ///
    /// **AND THE PADDED ENTRIES ARE NEVER DEREFERENCED.** `win[2]` — the
    /// window's live lane count — retires block `r` before the kernel reads
    /// `qo_indptr[r]` at all, in every one of the four. The bytes behind the
    /// live bounds are the window slot's own tail, which
    /// `crate::window::Windows::packed` now pads out to `Slots::tail()` so
    /// that a wider reading names staged bytes rather than whatever the last
    /// fire left; the `debug_assert` below is what says the widening cannot
    /// walk out of this window's slot into the next one's.
    ///
    /// **THE NUMBER IS `Run::planning`'S LANE CEILING, EXPRESSION FOR
    /// EXPRESSION**, and it has to be: a schedule carved at one lane count
    /// and a scan gridded at another would be two readings of one key. So the
    /// ladder's `own` — [`record::Carve::lanes`](crate::record::Carve::lanes),
    /// the LANE reading of the rungs, each one capped at the load's lane
    /// ceiling because a lane needs a seat — is capped again by what step 4d
    /// actually STAGED (`min(lane reach, max_lanes)`, read back off the padded
    /// vector rather than recomputed) less the prefix in front of this window.
    /// A window whose prefix already consumed the staging takes no ceiling and
    /// grids at its own lanes; that costs nothing arithmetically and it is a
    /// SEALED load's silent thrash, because a grid that follows the batch is a
    /// hashed plan payload that follows the batch — see
    /// [`record::Ladder::lane_reach`](crate::record::Ladder::lane_reach) for
    /// the deployment inequality that keeps it out of reach.
    fn carve_lanes(&self) -> Option<u32> {
        if !self.captured() || !self.plane_base() {
            return None;
        }
        let span = self.window().span;
        let (before, own) = self.carve.and_then(|carve| carve.lanes(span))?;
        let staged = self
            .windows
            .qo_absolute()
            .map_or(0, |bounds| bounds.rows.saturating_sub(1));
        let lanes = own.min(staged.checked_sub(before)?);
        debug_assert!(
            u64::from(lanes) + 1 <= self.windows.slots().stride(),
            "a ceiling grid of {lanes} requests wants {} boundary words, and a window \
             slot holds {}",
            lanes + 1,
            self.windows.slots().stride(),
        );
        (lanes >= span.lanes).then_some(lanes)
    }

    /// The fire bindings, for the plan-building arms' seam.
    pub(crate) fn bindings(&self) -> &FireBindings {
        &self.fire
    }

    /// The window the node being dispatched runs over — this region's, cut at
    /// the run the walk is on.
    pub(crate) fn window(&self) -> &'c Window {
        self.windows.at(self.place.region.get(), self.place.run.get())
    }

    /// **THE ROWS OF THIS WINDOW THAT ARE ACTUALLY THE NODE'S**, for a region
    /// P4 answered `Fallback::Grouped` for — and `None` for every other
    /// region, which is every region of every artifact P4 seated whole.
    ///
    /// A grouped window's span is the UNION of the intervals its mask covers,
    /// so [`Run::cut`] hands the arm a rectangle with foreign rows standing in
    /// the gaps; this is the list that keeps the launch off them. An arm that
    /// takes it MUST honour it — reaching a grouped window without asking is
    /// running the correction over rows whose lanes never asked for one — and
    /// an arm whose kernel cannot honour it must never have been named in
    /// `DeviceProfile::grouped`, because that is the word that made the
    /// compiler write the row.
    pub(crate) fn segments(&self) -> Option<Segments> {
        let window = self.window();
        let count = window.segs();
        if count == 0 {
            return None;
        }
        Some(Segments {
            list: window.segments,
            count,
            cap: window.segment_cap,
            max_rows: window.segment_rows(),
        })
    }

    /// Where this run's payload for `id` sits in [`structs`](Run::structs).
    ///
    /// The run comes off the same cell the window does, so a schedule is
    /// stored and read at the same key by construction — a builder cannot
    /// carve for one interval and a launch read another.
    fn struct_at(&self, id: ValueId) -> usize {
        self.place.run.get() as usize * self.values_wide + id.0 as usize
    }

    /// This window's qo boundaries, staged — what a ragged view is cut by.
    ///
    /// Rebased, and unchanged: [`qo_indptr_absolute`](Run::qo_indptr_absolute)
    /// is a SECOND reading of the same boundaries, absolute in value, taken by
    /// the FA2 params under a plane base — [`mask_indptr`](Run::mask_indptr)'s
    /// precedent.
    pub(crate) fn qo_indptr(&self) -> Tensor {
        self.window().indptr
    }

    /// Their host twin, for the prefill and mla builders that walk the
    /// contents. Rebased: entry 0 is 0, because the rectangle they bound is
    /// this window's, not the fire's.
    ///
    /// **AND IT STAYS REBASED EVEN WHERE THE DEVICE READING GOES ABSOLUTE**,
    /// which is chunk 2c-a's whole design choice and not an oversight. Both
    /// prefill builders read DIFFERENCES (`sched_prefill::spans`) and the one
    /// absolute read they make is `qo_indptr[batch_size]` staged as the total
    /// row count — so a schedule built off the rebased vector is
    /// shift-invariant and its numbers are byte-for-byte today's, while the
    /// DEVICE params beside it carry
    /// [`qo_indptr_absolute`](Run::qo_indptr_absolute). Feeding the builders
    /// the absolute vector instead would need `qo[batch] - qo[0]` where they
    /// take the raw last entry, which is a `kernels-cuda` edit for no gain.
    /// The mla builders make the same choice mandatory: `sched_mla` indexes
    /// `qo_indptr[i]` absolutely, into a rectangle that starts at the
    /// window's zero.
    ///
    /// [`qo_indptr_absolute_host`](Run::qo_indptr_absolute_host) is the
    /// sibling, kept for the day a builder wants it.
    pub(crate) fn qo_indptr_host(&self) -> &'c [i32] {
        &self.window().indptr_host
    }

    /// **THE SAME BOUNDARIES, READ ABSOLUTELY** — the FIRE's whole qo vector,
    /// `[fire lanes + 1]` entries with nothing subtracted — or `None` for a
    /// fire that staged no such vector, which is every fire the shell did not
    /// route to a body.
    ///
    /// **WHOLE, AND NOT SLICED BY LANE, WHICH IS WHERE THIS PARTS COMPANY WITH
    /// [`mask_indptr`](Run::mask_indptr).** That table may be cut at
    /// `lane_offset` because its consumer's request number IS the
    /// launch-local ordinal and nothing bakes the address. This one is reached
    /// under [`plane_base`](Run::plane_base), which is the bodies path, and a
    /// body BAKES the pointer it is handed — while `lane_offset` is the sum of
    /// the lanes of the classes in front of the window, which a
    /// `record::BodyKey` deliberately does not fix. `base + lane_offset * 4`
    /// is therefore a stale address on every replay but the one it was
    /// recorded at, which is exactly the staleness this seam exists to remove.
    /// The fire vector's own base is a function of the LOAD, and it is the
    /// only address a recording may keep.
    ///
    /// What decides between the two readings is what the POINTER beside them
    /// is: [`qo_indptr`](Run::qo_indptr) is rebased because a launch cut at a
    /// window is handed the window's first row, and this one is not because a
    /// region on [`crate::SHIFTED`] under a body is handed the PLANE's base
    /// ([`cut`](Run::cut)) and counts from the plane's zero.
    pub(crate) fn qo_indptr_absolute(&self) -> Option<Tensor> {
        // The invariant the two readings owe each other, checked where both
        // are in reach: this window's LANE SLICE of the fire's vector, minus
        // its own first bound, IS the rebased vector beside it. The slice is a
        // HOST reading and stays one — the sibling below says why nothing
        // device-side may take it.
        debug_assert!(
            {
                let absolute = self.qo_indptr_absolute_host();
                let rebased = self.qo_indptr_host();
                rebased.is_empty()
                    || absolute.is_empty()
                    || (absolute.len() == rebased.len()
                        && absolute
                            .iter()
                            .zip(rebased)
                            .all(|(there, here)| there - absolute[0] == *here))
            },
            "a window's two readings of its qo boundaries disagree",
        );
        // **HANDED WHOLE, AND THE OMISSION OF A LANE SLICE IS THE POINT.**
        // A `skip(table, span.lane_offset, ..)` here would be a pointer that
        // moves between fires of one `BodyKey` — `lane_offset` is the sum of
        // preceding classes' lanes, which the key deliberately does not fix —
        // and a body bakes what it is handed. The fire-wide base is the one
        // address that is a function of the key. Which LANE a launch reads is
        // therefore the plan's business, not the pointer's: today the plan's
        // request indices are window-local, so a consumer that reaches this
        // vector before the plan learns absolute ids trips the lane-count
        // refusal at the FA2 door (`q.indptr.rows - 1 == num_requests`) —
        // loud, typed, and exactly the tripwire wanted until that chunk
        // lands. The sliced form would instead read a neighbouring window's
        // rows with no detector at all.
        self.windows.qo_absolute()
    }

    /// Its host twin, and the ONE place a lane slice of the absolute reading
    /// is legal: this window's `[lanes + 1]` entries of the fire's vector,
    /// un-subtracted. Empty for a fire whose table holds no such vector.
    ///
    /// **SLICED HERE AND NOT DEVICE-SIDE**, which is not an inconsistency but
    /// the whole distinction. A host slice is read NOW, by the fire that made
    /// it, and nothing bakes it; the device pointer beside it is read by a
    /// replay of a graph recorded at another fire, where `lane_offset` is not
    /// a function of the key ([`qo_indptr_absolute`](Run::qo_indptr_absolute)
    /// carries the argument).
    ///
    /// Host-side this is always available where the fire-wide vector is,
    /// because the vector is what every window's rebased copy was made out of
    /// ([`Windows::qo_absolute_host`](crate::window::Windows::qo_absolute_host));
    /// the DEVICE reading beside it is the one a fire has to have staged.
    pub(crate) fn qo_indptr_absolute_host(&self) -> &'c [i32] {
        let span = self.window().span;
        let first = span.lane_offset as usize;
        let last = first + span.lanes as usize;
        self.windows
            .qo_absolute_host()
            .get(first..=last)
            .unwrap_or_default()
    }

    /// How many token rows this window carries — the `total_num_rows` the
    /// prefill builders take.
    pub(crate) fn total_tokens(&self) -> u32 {
        self.window().span.rows
    }

    /// The mask span table this window's schedule should carry, or `None`
    /// for a fire no lane masked.
    ///
    /// **SLICED BY LANE, ABSOLUTE IN VALUE** — the opposite of the qo
    /// boundaries beside it, and the difference is what each one bounds. A
    /// window's qo indptr cuts the window's OWN rectangle, so it is rebased;
    /// this one names byte offsets into a fire-wide slab the consumer takes
    /// whole, so rebasing it would send request 0 of a later window to the
    /// first lane's bits. Same shape as `GeomKind::Indices` and its bounds.
    ///
    /// `[lanes + 1]` entries, because the schedule's last request needs an
    /// upper bound as much as the ones before it.
    pub(crate) fn mask_indptr(&self) -> Option<Tensor> {
        // **AND WHOLE UNDER A PLANE BASE**, on the lane axis's own terms
        // (`Run::pool_absolute`): a schedule built under `plane_base` stages
        // FIRE lane ids, so the table its consumer indexes with them has to
        // be the fire's. Slicing it there would send request `lane_offset` to
        // lane zero's bits — and the pointer would move between fires of one
        // `record::BodyKey` besides, which `Run::schedule_shape` now names.
        if self.plane_base() {
            return self.fire.tables.mask_indptr;
        }
        let span = self.window().span;
        self.fire
            .tables
            .mask_indptr
            .map(|table| skip(table, span.lane_offset, span.lanes + 1))
    }

    /// Whether any lane of THIS WINDOW carries more than one token — the mla
    /// builder's `causal` word, derived rather than seated: multi-token lanes
    /// attend causally within themselves, single-token (decode) lanes have
    /// nothing to order.
    pub(crate) fn multi_token(&self) -> bool {
        self.qo_indptr_host()
            .windows(2)
            .any(|span| span[1] - span[0] > 1)
    }

    /// One value's rectangle, cut to the window of the node asking for it.
    ///
    /// **EVERY ROW-SHAPED TABLE IN THIS SHELL IS INDEXED BY ABSOLUTE FIRE
    /// ROW** — the arena carve gives a `Dim::Tokens` value one column at the
    /// fire's row count, the geometry vectors one entry per fire lane — so a
    /// window is a slice, and which slice is read off the value's own leading
    /// `Dim`. A `Dim::Const` column (a weight plane, a bias) is not fire
    /// -aligned and is handed over whole.
    ///
    /// `GeomKind::Indices` is the one declared shape that is not what it
    /// says: the IR spells the flat page-id list `Dim::Lanes` because it has
    /// no page symbol, and its entries are pages rather than lanes. Slicing it
    /// by a lane offset would hand a windowed consumer somebody else's pages,
    /// so it is excluded here — and its bounds vector stays absolute, which is
    /// exactly what makes a sliced `Indptr` still address the whole list.
    ///
    /// `RuntimeInput::Mask` is the SECOND of those, for the same reason and
    /// with the same remedy. The IR spells the custom-mask slab `Dim::Tokens`
    /// because it has no bit symbol either, and its entries are (query, key)
    /// BITS: one lane of the fire occupies `rows x (held + rows)` of them, so
    /// a row offset is not a byte offset and a slice would land mid-lane.
    /// The slab goes over whole and `FireTables::mask_indptr` — absolute byte
    /// offsets, sliced by lane — is what puts a windowed launch on its own
    /// rectangle ([`crate::mask`] argues both halves).
    fn cut(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        // **THE OTHER ANSWER, ASKED FIRST** (design §3). A gathered window's
        // rows do not lie in the arena at all — they were compacted into a
        // scratch slab before the region's first node — so a slice of the
        // fire-wide column is not a narrower reading of the same bytes, it is
        // the wrong bytes. `compacted` is the whole of what a copy changes
        // about resolution.
        if self.window().gathered.is_some() {
            return self.compacted(id, handle);
        }
        if matches!(
            self.values[at].def,
            Def::Input(RuntimeInput::Mask { .. })
                | Def::Input(RuntimeInput::Geometry {
                    kind: GeomKind::Indices,
                    ..
                })
        ) {
            return handle;
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        let seated = self.window();
        let window = seated.span;
        let patch = seated.patch;
        // **THE PLANE BASE, FOR A REGION THAT MOVES ITS OWN** (bodies design,
        // chunk 2b-ii). Asked once, before the match, because it is a fact
        // about the REGION and not about the value: `plane_base` short-circuits
        // on `bodied`, so a fire on any other path pays one `bool` test here.
        let plane = self.plane_base();
        // **AND HOW MANY ROWS THE LAUNCH BESIDE IT IS GRIDDED OVER** — the
        // KEY's ceiling for a region that owns a retirement, and `None` (this
        // window's own rows) for every other fire and every other region. Asked
        // once here for the same reason and on the same terms: it is a fact
        // about the REGION, `carve_rows` short-circuits on `bodied`, and the
        // three other paths pay one `bool` test. [`carve_rows`](Run::carve_rows)
        // carries the whole argument, seat and all.
        let ceiling = self.carve_rows();
        let rows = ceiling.unwrap_or(window.rows);
        let (skip, keep) = match shape.first() {
            // **THE EXTENT IS THE LAUNCH'S, THE POINTER IS THE PLANE'S, AND
            // `win[1]` IS THE BRIDGE.** A shifting region's ops address
            // `start + r` off whatever base they are handed, so the base has
            // to stay the plane's — skipping nothing — while the row count is
            // what grids and loops are sized at. Hand it a shifted pointer and
            // `win[1]` would shift it twice.
            //
            // **AND THE ROW COUNT IS THE KEY'S CEILING SINCE THE
            // GRID-AT-CEILING WAVE, NOT THIS WINDOW'S LIVE SPAN** — which is
            // the one sentence in this arm that changed and the whole of what
            // makes `record::Body::grids` a function of the key. The extent
            // used to be `window.rows` on the argument that "a bigger extent
            // would admit rows the launch does not own"; that argument was
            // about a launch with nothing to retire it, and this region has
            // something: `Run::live_at` armed the seat, every seated entry
            // opens with `r >= win[0]`, and the blocks between the fire's rows
            // and the ceiling return before they address a byte.
            // [`carve_rows`](Run::carve_rows) is that number and carries the
            // argument in full; `None` from it puts this arm back on
            // `window.rows` exactly, which is every fire off the bodies path
            // and every deployment with no lattice.
            //
            // The LANE axis is not here, and that is [`crate::SHIFTED`]'s own
            // caveat: the ops on that list index their per-lane tables by the
            // launch-local ordinal, so those stay sliced at `lane_offset` —
            // with ONE family excepted, whose request number is a staged
            // datum rather than a grid coordinate and which therefore asks
            // for its tables through [`Run::pool_absolute`] instead. Nothing
            // about THIS resolution changes either way: the lane pairs below
            // are the sliced reading on every path, and a caller that wants
            // the other one names it. Only the ROW axis moves here. The patch
            // pair below is untouched for
            // the same reason with a second name: nothing on the list reads a
            // patch seat, so a tower rectangle is cut where it always was.
            Some(Dim::Tokens) if plane => (0, rows),
            Some(Dim::TokensTimes(k)) if plane => (0, rows * k),
            // **AND THE WHOLE-FIRE ARM TAKES THE SAME CEILING WITHOUT TAKING
            // THE PLANE BASE**, which is why `rows` is computed above the
            // match rather than inside the `plane` arms. A region whose window
            // IS the fire skips nothing anyway (`whole_fire` has
            // `row_offset == 0`), so the two arms differ in the OFFSET and not
            // in the extent — and `carve_rows` answers `Some` for exactly the
            // union of the two, because that is the union `Run::live_at` arms
            // the seat over.
            Some(Dim::Tokens) => (window.row_offset, rows),
            Some(Dim::TokensTimes(k)) => (window.row_offset * k, rows * k),
            Some(Dim::Lanes) => (window.lane_offset, window.lanes),
            Some(Dim::LanesPlus(k)) => (window.lane_offset, window.lanes + k),
            Some(Dim::Const(_)) | None => return handle,
            // **THE SECOND ROW AXIS, CUT AT ITS OWN WINDOW** (multimodal
            // §5.1). `Window::patch` is this region's mask read against the
            // PATCH table — patch rows where `span` has token rows and IMAGES
            // where it has lanes — so a tower rectangle is cut by the
            // seriation that placed it and never by the token one. The two
            // pairs are carried separately rather than chosen between,
            // because the embed merge is a TOKEN region that reads a patch
            // column and needs both in the same resolution.
            Some(Dim::Patches) => (patch.row_offset, patch.rows),
            Some(Dim::Images) => (patch.lane_offset, patch.lanes),
            Some(Dim::ImagesPlus(k)) => (patch.lane_offset, patch.lanes + k),
        };
        if skip == 0 && keep >= handle.rows {
            return handle;
        }
        let stride = u64::from(handle.width)
            * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
                panic!(
                    "value {at} is stored as {:?}, which has no element size and so no \
                     row to step by",
                    handle.dtype
                )
            });
        Tensor::new(
            handle.ptr + u64::from(skip) * stride,
            keep.min(handle.rows.saturating_sub(skip)),
            handle.width,
            handle.dtype,
        )
    }

    /// [`Run::cut`]'s other half: what a `Fallback::Copy` resolves to.
    ///
    /// **THREE ANSWERS, AND THEY ARE THE THREE `window::copyable` ADMITS.**
    /// A row-shaped value is the slab rectangle the region's gather laid it
    /// in; the four kv geometry vectors are the twins re-cut for the gathered
    /// lanes ([`GatheredSpace`](crate::window::GatheredSpace)); everything
    /// window-free is handed over whole, exactly as a split hands it over.
    /// Nothing else can arrive — `Windows::of` declines to gather a region
    /// naming anything else, and the region then takes the split, which is
    /// always correct.
    fn compacted(&self, id: ValueId, handle: Tensor) -> Tensor {
        let at = id.0 as usize;
        let gathered = self
            .window()
            .gathered
            .as_ref()
            .expect("`compacted` is reached only through a gathered window");
        if let Def::Input(RuntimeInput::Geometry { space, kind }) = &self.values[at].def {
            let Some(space) = gathered.spaces.get(*space as usize) else {
                return handle;
            };
            return match kind {
                GeomKind::Indptr => space.page_indptr,
                GeomKind::Indices => space.page_indices,
                GeomKind::LastPageLen => space.last_page_lens,
                GeomKind::KvLen => space.kv_len,
                // `window::copyable` admits no other kind into a copied
                // region, so this is the arm nothing reaches — and it
                // answers the fire-wide vector rather than a wrong window,
                // which is the conservative direction.
                _ => handle,
            };
        }
        let Ty::Tensor { shape, .. } = &self.values[at].ty else {
            return handle;
        };
        match shape.first() {
            Some(Dim::Tokens | Dim::TokensTimes(_)) => {
                assert_eq!(
                    self.copy.region,
                    self.place.region.get(),
                    "value {at} is being resolved inside a copied region whose gather \
                     has not run; `model_exec::fire::walk` brackets a copied region's \
                     nodes and this is what says the bracket was lost",
                );
                self.copy.tight(handle.ptr).unwrap_or_else(|| {
                    panic!(
                        "value {at} is row-shaped and its column was not compacted; the \
                         copy plan is built from the same region's operands the walk is \
                         dispatching, so a miss is a plan and a template built apart"
                    )
                })
            }
            _ => handle,
        }
    }

    /// The crate's heart: one plan id in, one device handle out, routed on
    /// the id's `Def`, cut to the asking node's window ([`Run::cut`]). Every
    /// dispatch arm resolves through here, so provenance handling — and
    /// windowing — exists exactly once.
    ///
    /// Cache ids never resolve to a tensor — a cache is a pool pointer and
    /// resolves through [`Run::pool`] or [`Run::recurrent`]; a cache id
    /// arriving here is a dispatch-arm bug, answered with a panic. So is a
    /// split-plane weight: two planes resolve through [`Run::planes`].
    pub(crate) fn tensor(&self, id: ValueId) -> Tensor {
        self.cut(id, self.whole(id))
    }

    /// **THE FIRE-WIDE RECTANGLE, ASKED FOR BY NAME** — what a consumer takes
    /// when its own index vector is already absolute.
    ///
    /// The embed merge is the one op that needs it. `layout.scatter_rows` and
    /// `layout.scatter_live_rows` address their TOKEN destination with
    /// `RuntimeInput::PatchRoutes`, which `serve::prepare` rebases to absolute
    /// fire rows and which `Fault::PatchRoute` checks against the fire's own
    /// row count. Cutting that destination at the region's window would make
    /// the routes relative to the window and absolute at the same time, and
    /// the offset would be counted twice.
    ///
    /// **THE SAME EXEMPTION `GeomKind::Indices` AND `RuntimeInput::Mask`
    /// ALREADY TAKE** ([`Run::cut`] argues both): a `Dim`-shaped table whose
    /// entries are not the rows that `Dim` counts goes over whole, and what
    /// windows the launch is the vector beside it. It is spelled as an
    /// accessor rather than as a third arm of `cut` because this one is a
    /// property of the OP and not of the value — the same embedding column is
    /// read cut by every other node that touches it.
    pub(crate) fn fire_wide(&self, id: ValueId) -> Tensor {
        self.whole(id)
    }

    /// The same resolution, uncut — the fire-wide rectangle a value names.
    fn whole(&self, id: ValueId) -> Tensor {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Input(RuntimeInput::Tokens) => self.fire.tokens,
            Def::Input(RuntimeInput::Positions) => self.fire.positions,
            Def::Input(RuntimeInput::Mask { space }) => {
                let seat = self.geometry(at, *space);
                seat.mask.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads the mask bits of cache space {space}, which \
                         this fire left unbound"
                    )
                })
            }
            // THE ADAPTER AXIS'S ONE RUNTIME INPUT (design §8). Bound only
            // when a lane of this fire carried an adapter — a fire none did
            // stages nothing, and nothing can reach this arm either, because
            // the correction's window is empty and the walk skips a zero-row
            // region before it dispatches a node. So the panic is not a hole:
            // it is the same "unbound seat" statement the mask makes one arm
            // up, and reaching it means a word said `has_adapter` where the
            // submission said no adapter — which `Fault::AdapterWord` refuses
            // before anything launches.
            Def::Input(RuntimeInput::AdapterRoutes) => {
                self.fire.adapter_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's adapter ids, which no lane of it                          carried"
                    )
                })
            }
            // **THE SECOND ROW AXIS'S THREE RUNTIME INPUTS** (multimodal
            // §2), bound from what `enqueue` wrote — outside the staging ring,
            // for the reason [`FireBindings::patches`] states at length.
            //
            // The panic is the `AdapterRoutes` statement one arm up, on the
            // other axis: a fire whose lanes submitted no image binds no patch
            // seat, and a plan that reads one has a class whose window is
            // empty, so the walk never dispatches the node that would ask.
            // `compose_axes` is what makes that true rather than hoped for —
            // a lane carrying images against an artifact with no patch axis is
            // `Fault::Towerless`, refused before a byte is staged.
            Def::Input(RuntimeInput::Patches) => self.fire.patches.unwrap_or_else(|| {
                panic!(
                    "value {at} reads this fire's patch rows, which no lane of it                      submitted"
                )
            }),
            Def::Input(RuntimeInput::PatchSegments) => {
                self.fire.patch_segments.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's image boundaries, which no lane of                          it submitted"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchRoutes) => {
                self.fire.patch_routes.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads where this fire's tower rows land, and no lane                          of it submitted an image"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchEmbedRows) => {
                self.fire.patch_embed_rows.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads which position-table rows this fire's patches gather,                          and no lane of it submitted an image"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchEmbedWeights) => {
                self.fire.patch_embed_weights.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's interpolation weights, which a                          native-grid plan declares none of"
                    )
                })
            }
            Def::Input(RuntimeInput::PatchPositions) => {
                self.fire.patch_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads where this fire's patches sit in their grids,                          and no lane of it submitted an image"
                    )
                })
            }
            // **THE TRUNK'S TRIPLE, ON THE TOKEN AXIS.** Unlike the four
            // above it this is not gated on a lane carrying an image: the
            // trunk's region covers the whole rectangle, so a plan that
            // declares the stream reads it in every fire, and a text lane's
            // rows carry `(p, p, p)`. Reaching the panic therefore means the
            // load reserved no stream for a plan that names one, which
            // `Inputs::reserve` decides from the trace itself.
            Def::Input(RuntimeInput::MropePositions) => {
                self.fire.mrope_positions.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads this fire's (t, h, w) token positions, which this                          load reserved no stream for"
                    )
                })
            }
            Def::Input(RuntimeInput::Geometry { space, kind }) => {
                let seat = self.geometry(at, *space);
                let bound = match kind {
                    GeomKind::Indptr => seat.indptr,
                    GeomKind::Indices => seat.indices,
                    GeomKind::SeqLens => seat.seq_lens,
                    GeomKind::LastPageLen => seat.last_page_len,
                    GeomKind::KvLen => seat.kv_len,
                    GeomKind::RowValid => seat.row_valid,
                    GeomKind::RequestOfToken => seat.request_of_token,
                    GeomKind::WritePage => seat.write_page,
                    GeomKind::WriteOffset => seat.write_offset,
                };
                bound.unwrap_or_else(|| {
                    panic!(
                        "value {at} reads {kind:?} of cache space {space}, which this \
                         fire left unbound"
                    )
                })
            }
            Def::Weight(w) => {
                let row = *w as usize;
                match self.weights.0.get(row).copied().flatten() {
                    Some(WeightRow::Dense(handle) | WeightRow::Streamed { slab: handle, .. }) => {
                        handle
                    }
                    Some(WeightRow::Planes { .. }) => panic!(
                        "value {at} is weight {row}, a split-plane bank; it resolves \
                         through `Run::planes`, never as one dense handle"
                    ),
                    None => panic!("value {at} is weight {row}, which the shell has not bound"),
                }
            }
            // A φ resolves like the op output it merges: the compiler
            // aliased every arm onto one arena slot, written at this id's
            // row — so `Merge` is the same read as `Op`.
            Def::Op(_) | Def::Merge(_) => {
                self.arena.0.get(at).copied().flatten().unwrap_or_else(|| {
                    panic!("value {at} has no arena slot, which the compiler should have cut")
                })
            }
            Def::Cache(_) => panic!(
                "value {at} is a cache space; it resolves to a pool through `Run::pool`, \
                 never to a tensor"
            ),
        }
    }

    /// A fire-aligned value viewed through THIS WINDOW's boundaries. The
    /// indptr is ambient (design §5): no op names it, and this pairing is
    /// where it re-enters.
    ///
    /// The boundaries are the window's own, rebased: `data` already points at
    /// the window's first row, so a fire-wide vector would send every ragged
    /// entry past the end of the rectangle it was handed by exactly the number
    /// of rows the classes before it occupy.
    ///
    /// **AND IT STAYS REBASED EVEN UNDER A PLANE BASE**, which is not an
    /// oversight but [`crate::SHIFTED`]'s second convention spelled as code.
    /// Every op on that list reads `win[1]` and addresses `start + r`, and it
    /// indexes its per-lane tables — this CSR among them — with the
    /// launch-local ordinal. `start + qo_rebased[lane]` IS the fire row; hand
    /// such a kernel the absolute vector instead and it shifts twice. So the
    /// reading a shifting region wants here is the one it already had, and the
    /// absolute one belongs to the consumer that reads NO seat:
    /// [`ragged_q`](Run::ragged_q).
    ///
    /// **AND THE LENGTH IS THIS WINDOW'S TOO, WHICH IS A THIRD READING AND
    /// [`ragged_lanes`](Run::ragged_lanes) IS WHERE IT PARTS.** A consumer
    /// gridded on ROWS reads this vector to find which lane a row belongs to
    /// and could not use a longer one; the four CHUNKED arms are gridded on
    /// REQUESTS and count them off the length, so under the grid-at-ceiling
    /// wave they take the same pointer with the key's lane ceiling declared
    /// over it. Same bytes, same rebasing, one number wider — and named by the
    /// arm rather than decided here, because every other caller of this
    /// function would be taking a grid it does not own.
    pub(crate) fn ragged(&self, id: ValueId) -> RaggedTensor {
        RaggedTensor {
            data: self.tensor(id),
            indptr: self.qo_indptr(),
        }
    }

    /// **THE FA2 QUERY AXIS'S OWN READING OF THE SAME PAIRING** — `data` as
    /// [`ragged`](Run::ragged) resolves it, and the boundaries chosen by
    /// whether this region moves its own plane.
    ///
    /// FA2 takes one by-value params block with NO seat parameter in the
    /// ATTENTION itself: the kernel computes `q + q_indptr[req] * stride` and
    /// has no `win[1]` to add. So its CSR has to count from wherever its
    /// POINTER counts from — the window's zero with a sliced `q`, the fire's
    /// zero with a plane-base one ([`cut`](Run::cut) under
    /// [`plane_base`](Run::plane_base)). That is the exact opposite of the
    /// seated ops' rule above, and it is why the two readings are two methods
    /// rather than one flag on one. (The one launch of the family that DOES
    /// take the seat is the cascade fold behind a split schedule, which is
    /// the only one that writes the fire's own planes — chunk 2c-b's other
    /// half.)
    ///
    /// **THE PREDICATE IS THE ONE `cut` AND [`live_at`](Run::live_at) SWITCHED
    /// ON IN CHUNK 2b-ii, AND FIVE SWITCHES ON ONE PREDICATE IS THE
    /// INVARIANT** — pointer, seat, CSR, the per-lane tables
    /// ([`pool_absolute`](Run::pool_absolute),
    /// [`mask_indptr`](Run::mask_indptr)) and the lane ids the SCHEDULE
    /// stages ([`planning`](Run::planning)) move together or not at all. Chunk
    /// 2c-b added the last two: a launch reading fire-wide lane tables with
    /// window-local request numbers reads the wrong lane's pages, and a
    /// schedule staging fire lane ids into a launch handed sliced tables does
    /// the same thing from the other end. A sixth reader that learns to move
    /// without asking `plane_base` is the shape of the bug this note exists
    /// to prevent.
    ///
    /// **AND THE VECTOR GOES OVER WHOLE**, never cut at `lane_offset`: a body
    /// bakes this pointer and `lane_offset` moves between fires of one key, so
    /// a sliced absolute reading would be stale on every replay — the exact
    /// staleness the seam exists to remove
    /// ([`qo_indptr_absolute`](Run::qo_indptr_absolute) carries the argument).
    /// WHICH entries are this launch's requests is the SCHEDULE's business —
    /// staged per fire into the plan workspace, taught absolute ids by a later
    /// chunk — and not the pointer's.
    ///
    /// **AND IT IS LIVE SINCE CHUNK 2c-b**, which is what that chunk was for:
    /// the five FA2 names are on [`crate::SHIFTED`] now, so a region carrying
    /// one can be `plane_base` and this can answer the fire's vector where
    /// `ragged` answers the window's. The obstacle this note used to name —
    /// `attn::prefill`'s arity refusal, which compared `q.indptr.rows - 1`
    /// against the schedule's request count and would see the FIRE's lanes
    /// where a windowed schedule was planned at the WINDOW's — was the first
    /// thing that chunk had to answer, and it answered it by teaching the
    /// door both readings rather than by removing it (`attn::lanes_carry`).
    ///
    /// The fallback is unreachable rather than defensive: `plane_base`
    /// short-circuits on `bodied`, and a bodied fire is exactly the fire that
    /// staged the absolute vector (`inputs::Fire::qo_absolute`).
    pub(crate) fn ragged_q(&self, id: ValueId) -> RaggedTensor {
        let indptr = if self.plane_base() {
            self.qo_indptr_absolute().unwrap_or_else(|| self.qo_indptr())
        } else {
            self.qo_indptr()
        };
        RaggedTensor {
            data: self.tensor(id),
            indptr,
        }
    }

    /// **THE SAME PAIRING, WITH THE BOUNDARY VECTOR DECLARED OUT TO THE KEY'S
    /// LANE CEILING** — what the four CHUNKED recurrent arms take where every
    /// other consumer takes [`ragged`](Run::ragged).
    ///
    /// Those four are the only launches in the tree whose grid counts
    /// REQUESTS, and they count them off this vector's length
    /// (`attn::ssm::requests`, `attn_ple`'s `lanes`), so this is where their
    /// grid is decided. [`carve_lanes`](Run::carve_lanes) is the number and
    /// carries the argument: why the vector stays the window's own REBASED
    /// boundaries rather than taking `ragged_q`'s absolute treatment, why the
    /// padded entries are never dereferenced, and why the count has to be the
    /// same one [`planning`](Run::planning) carves a schedule at.
    ///
    /// `None` from that — every fire off the bodies path, every region with no
    /// retirement, every deployment with no lattice — hands back exactly what
    /// `ragged` hands back, which is what keeps the EAGER path byte for byte
    /// the path it was.
    ///
    /// A SECOND DOOR AND NOT A WIDENING OF THE FIRST, on
    /// [`recurrent_absolute`](Run::recurrent_absolute)'s terms: the per-STEP
    /// scans beside these four are gridded on ROWS and read their CSR to find
    /// a row's lane, so a longer vector would move a grid they do not own.
    pub(crate) fn ragged_lanes(&self, id: ValueId) -> RaggedTensor {
        let indptr = self.qo_indptr();
        let indptr = match self.carve_lanes() {
            Some(lanes) if lanes + 1 > indptr.rows => {
                Tensor::new(indptr.ptr, lanes + 1, indptr.width, indptr.dtype)
            }
            _ => indptr,
        };
        RaggedTensor {
            data: self.tensor(id),
            indptr,
        }
    }

    /// The `(codes, scales)` planes of a split-plane bank and the seat that
    /// says where they are RIGHT NOW — the resolution
    /// `linear.moe_matmul_select_bias` needs where [`Run::tensor`] would
    /// have to lie with one handle.
    ///
    /// The seat is [`GroupSeat::RESIDENT`] — two zeros — for every group the
    /// store holds whole; see [`WeightRow::Planes`].
    /// The split-plane resolution a DENSE-reading op asks first: `Some` for
    /// a bank the loader landed as planes (the n-gram table under its affine
    /// triplet), `None` for an ordinary handle — so a gather can serve both
    /// landings without a second op.
    pub(crate) fn maybe_planes(
        &self,
        id: ValueId,
    ) -> Option<(Tensor, Tensor, Option<Tensor>, GroupSeat)> {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            return None;
        };
        match self.weights.0.get(*w as usize).copied().flatten() {
            Some(WeightRow::Planes {
                codes,
                scales,
                biases,
                seat,
            }) => Some((codes, scales, biases, seat)),
            _ => None,
        }
    }

    pub(crate) fn planes(&self, id: ValueId) -> (Tensor, Tensor, Option<Tensor>, GroupSeat) {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and split-plane banks live in the weight table");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Planes {
                codes,
                scales,
                biases,
                seat,
            }) => (codes, scales, biases, seat),
            Some(WeightRow::Dense(_) | WeightRow::Streamed { .. }) => panic!(
                "value {at} is weight {row}, bound as one dense handle, and this op reads \
                 a split-plane bank"
            ),
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    /// **A ROUTED EXPERT BANK, AND WHERE ITS EXPERTS ARE** (alto design §7,
    /// wave D2) — the resolution `linear.moe_matmul_select` uses in place of
    /// [`Run::tensor`].
    ///
    /// The handle is the same rectangle `tensor` would answer with. What comes
    /// beside it is the pair of device addresses the select kernel resolves
    /// each expert's weights through: for a fully-resident load they are
    /// [`ExpertTable::RESIDENT`] — two nulls — and the kernel does the
    /// `bank_base + expert * stride` arithmetic it always did, which is why a
    /// load that streams nothing pays nothing for this door existing.
    pub(crate) fn expert_bank(&self, id: ValueId) -> (Tensor, ExpertTable) {
        let at = id.0 as usize;
        let Def::Weight(w) = &self.values[at].def else {
            panic!("value {at} is not a weight, and a routed bank is a weight row");
        };
        let row = *w as usize;
        match self.weights.0.get(row).copied().flatten() {
            Some(WeightRow::Dense(handle)) => (self.cut(id, handle), ExpertTable::RESIDENT),
            Some(WeightRow::Streamed {
                slab,
                table,
                counts,
            }) => (self.cut(id, slab), ExpertTable { table, hits: counts }),
            Some(WeightRow::Planes { .. }) => panic!(
                "value {at} is weight {row}, a split-plane bank; a dense routed select \
                 does not read one"
            ),
            None => panic!("value {at} is weight {row}, which the shell has not bound"),
        }
    }

    /// The paged kv pool a cache id names, with its LANE-INDEXED tables cut
    /// to the asking node's window.
    ///
    /// The storage is the whole pool — pages are the model's state and outlive
    /// every fire — but the tables that address it are this fire's: the page
    /// bounds and last-page fills are one entry per lane, and the padding mask
    /// one per row. A windowed attention launches request `r` in `0..lanes` of
    /// ITS window, so those three are sliced and the page-id list they index
    /// is not (its bounds stay absolute).
    ///
    /// **AND `row_valid` CO-MOVES WITH [`Run::cut`]'S ROW AXIS** (bodies
    /// design, chunk 2b-ii). It is the one table here indexed by ROW rather
    /// than by lane, so a region the walk hands its plane BASE has to be
    /// handed the plane's `row_valid` too: the kernels that shift read
    /// `row_valid[row]` at the same shifted row they read their planes at
    /// (the wave-2a manifest), and a table sliced at `row_offset` under a
    /// launch that then adds `win[1]` would be read twice-shifted. The two
    /// LANE-indexed tables stay sliced under exactly the same condition, which
    /// is [`crate::SHIFTED`]'s lane-axis caveat and is the law here.
    pub(crate) fn pool(&self, id: ValueId) -> KvPool {
        match self.cache(id) {
            CachePool::Kv { space, pool } => {
                let window = self.window().span;
                // **A GATHERED WINDOW'S TABLES ARE RE-CUT, NOT SLICED**, and
                // the page-id list is the reason: gathered lanes own spans of
                // it with other lanes' pages standing between them, and no
                // `[lanes + 1]` bounds vector over the whole list can name
                // two such spans as requests 0 and 1
                // (`window::GatheredSpace`). So the list is compacted with
                // the lanes and the bounds are a fresh prefix sum over it.
                //
                // `row_valid` is the one table a gather does NOT need: its
                // entries are 1 for a row this fire means and 0 for a
                // bucket's padding, every gathered row is one the fire
                // means, and the count is the gathered count — so the
                // window's own prefix of the fire-wide vector holds exactly
                // the values a gathered one would.
                if let Some(gathered) = &self.window().gathered {
                    let seat = gathered.spaces.get(*space as usize);
                    return KvPool {
                        page_indptr: seat.map_or(pool.page_indptr, |seat| seat.page_indptr),
                        page_indices: seat.map_or(pool.page_indices, |seat| seat.page_indices),
                        last_page_lens: seat.map_or(pool.last_page_lens, |seat| seat.last_page_lens),
                        row_valid: skip(pool.row_valid, 0, window.rows),
                        ..*pool
                    };
                }
                KvPool {
                    page_indptr: skip(pool.page_indptr, window.lane_offset, window.lanes + 1),
                    last_page_lens: skip(pool.last_page_lens, window.lane_offset, window.lanes),
                    row_valid: if self.plane_base() {
                        // The plane's own table, whole — see this method's
                        // doc: the launch reaches its rows through `win[1]`.
                        pool.row_valid
                    } else {
                        skip(pool.row_valid, window.row_offset, window.rows)
                    },
                    ..*pool
                }
            }
            CachePool::Recurrent(_) => panic!(
                "value {} is a recurrent state space, and this op walks a paged kv pool",
                id.0
            ),
        }
    }

    /// **THE SAME POOL, WITH ITS PER-LANE TABLES READ ABSOLUTELY** — what
    /// the five FA2 entries take in place of [`pool`](Run::pool), and nothing
    /// else does (bodies design, chunk 2c-b).
    ///
    /// **WHY THIS IS A SECOND DOOR AND NOT A WIDENING OF THE FIRST.** Every
    /// other op on [`crate::SHIFTED`] indexes its per-lane tables with the
    /// LAUNCH-LOCAL ordinal — `attention.index_topk` and `attention.pool_lse`
    /// read `page_indptr[r]` at the `r` their own grid counts, and the
    /// recurrent scans read `slot_ids[r]` the same way — so handing THEM the
    /// fire's tables would break them silently. FA2 is the one family whose
    /// request number is a datum rather than a grid coordinate: it comes off
    /// `request_indices[bx]`, which the schedule staged, and chunk 2c-b
    /// taught the schedule to stage `lane_offset + r`
    /// ([`Run::planning`]). Absolute ids and fire-wide tables are one change
    /// with two halves, and this method is the second half.
    ///
    /// Under anything but a plane base this IS [`pool`](Run::pool): the ids
    /// are window-local there, and so are the tables.
    pub(crate) fn pool_absolute(&self, id: ValueId) -> KvPool {
        let pool = self.pool(id);
        if !self.plane_base() {
            return pool;
        }
        // The page-id list and its bounds were already a pair of absolute
        // readings — the list is handed over whole and its bounds index it —
        // so what moves here is only WHICH bound a request reads. The
        // last-page fills ride with the bounds because the kernels read them
        // at the same request number (`paged_kv_t::get_length`).
        match self.cache(id) {
            CachePool::Kv { pool: whole, .. } => KvPool {
                page_indptr: whole.page_indptr,
                last_page_lens: whole.last_page_lens,
                ..pool
            },
            CachePool::Recurrent(_) => pool,
        }
    }

    /// **THE OBSERVATION'S READING, AND IT IS THE WINDOW'S** — the `q`
    /// rectangle `attention.score_capture` is fired over.
    ///
    /// The score kernel is not an IR op, so [`crate::SHIFTED`] cannot name it
    /// and `exports::regions_shifting` never sees it; it is a second launch
    /// the `prefill_lse` arm makes beside the one the node states. Its grid is
    /// `[requests, heads]` and it reads `q + indptr[blockIdx.x] * stride` off
    /// the REBASED boundaries with a `lane_offset` argument for the slab —
    /// launch-local through and through. So where the arm beside it takes the
    /// plane's base, this one takes the window back: a rectangle cut at
    /// `row_offset`, which is what `Run::tensor` answers on every path but a
    /// plane-based one.
    pub(crate) fn windowed(&self, handle: Tensor) -> Tensor {
        if !self.plane_base() {
            return handle;
        }
        let window = self.window().span;
        skip(handle, window.row_offset, window.rows)
    }

    /// The recurrent state pool a cache id names, with its slot map cut to the
    /// asking node's window.
    ///
    /// A recurrent bank is addressed by SLOT and the scan reads its slot from
    /// `slot_ids[lane]` — where `lane` counts from the launch's own zero. So a
    /// windowed scan gets the window's lanes and nothing else; the slabs
    /// themselves are the model's state, whole.
    ///
    /// **THE CHUNKED ARMS ASK ELSEWHERE** — see
    /// [`recurrent_absolute`](Run::recurrent_absolute): their pointer is baked
    /// by a body and their lane comes off `win[3]` instead.
    pub(crate) fn recurrent(&self, id: ValueId) -> RecurrentPool {
        // **THE HEAD SEGMENT**, which for every fire but a splitting one is
        // the whole row: the origin seat is cleared here and bound only by
        // [`Run::recurrent_tail_absolute`], so no launch ever carries both ends of one
        // boundary and the plain path hands the null pointer it always did.
        RecurrentPool {
            begin_at: Tensor::ABSENT,
            ..self.recurrent_cut(id, false)
        }
    }

    /// **THE SAME POOL WITH ITS PER-LANE SEATS READ ABSOLUTELY** — what the
    /// four CHUNKED arms take in place of [`recurrent`](Run::recurrent), and
    /// nothing else does (the chunked-arm wave). [`pool_absolute`](Run::pool_absolute)'s
    /// twin on the recurrent axis, and a SECOND DOOR for that method's exact
    /// reason.
    ///
    /// **WHY IT CANNOT BE A WIDENING OF THE FIRST.** The per-step scans —
    /// `attention.ssm_causal_conv1d`, `attention.ssm_gated_delta`,
    /// `attention.ssm_kda_step`, `attention.ple_ngram_ids` — are on
    /// [`crate::SHIFTED`] already, and every one of them reads `slot_ids[r]`
    /// at the ordinal its own grid counts, which for a decode launch is a ROW.
    /// Handing THOSE the fire's slot map would send each of them to the wrong
    /// bank the moment a class stood in front of their window. So the
    /// window-local reading stays the default and the absolute one is asked
    /// for by name, by the launchers that learned to add `win[3]`.
    ///
    /// **AND WHY THE CHUNKED ARMS HAVE TO ASK.** Their request number IS a
    /// grid coordinate, so on the face of it they belong with the steppers —
    /// but a BODY bakes the pointer they are handed, and `lane_offset` is the
    /// sum of the lanes of the classes in front of the window, which a
    /// `record::BodyKey` deliberately does not fix. `base + lane_offset * 4`
    /// is therefore stale on every replay but its recording one — the same
    /// staleness [`qo_indptr_absolute`](Run::qo_indptr_absolute) exists to
    /// remove. So the tables go over WHOLE and the kernel indexes them at
    /// `r + win[3]`, which is a number the FIRE staged.
    ///
    /// Under anything but a plane base this IS [`recurrent`](Run::recurrent):
    /// the seat is null there, the kernels add nothing, and the tables are the
    /// window's.
    pub(crate) fn recurrent_absolute(&self, id: ValueId) -> RecurrentPool {
        RecurrentPool {
            begin_at: Tensor::ABSENT,
            ..self.recurrent_cut(id, true)
        }
    }

    /// **The tail segment of a row whose fold boundary is interior**, or
    /// `None` for every fire that does not split one (alto design §6's 2R
    /// split, wave F3b).
    ///
    /// A row that folds a prefix of the tokens it is writing cannot be one
    /// launch: `commit_len` TRUNCATES, so the tokens past the boundary would
    /// get no outputs at all. So the arm fires twice on the one stream — the
    /// head `[0, n)` with the length seat and the fold, whose end-of-sequence
    /// writeback lands ON the boundary, then this tail `[n, rows)` reading
    /// the state the head just wrote, producing the rest of the outputs and
    /// moving nothing.
    ///
    /// **THE SEAT'S PRESENCE IS THE STATEMENT.** The shell binds the origin
    /// only for a fire some row splits, so an absent one here is exactly "one
    /// launch, as before" and there is no second flag to fall out of step
    /// with it. Every lane of a splitting fire gets a boundary: a lane that
    /// folds everything begins its tail past its own last row and returns,
    /// and one that folds nothing begins at zero and runs whole.
    ///
    /// **AND IT IS THE ABSOLUTE READING, BECAUSE ONLY A CHUNKED ARM SPLITS.**
    /// The origin seat is bound for a row whose fold boundary falls inside its
    /// own tokens, which is a PREFILL fact; every caller of this is therefore
    /// one of the four chunked arms, and they take their lane axis through
    /// [`recurrent_absolute`](Run::recurrent_absolute). A window-local twin of
    /// this would be a door nobody walks through.
    pub(crate) fn recurrent_tail_absolute(&self, id: ValueId) -> Option<RecurrentPool> {
        let cut = self.recurrent_cut(id, true);
        if cut.begin_at.is_absent() {
            return None;
        }
        Some(RecurrentPool {
            // The tail moves no state: the boundary is where the head left
            // it, and a second writeback would carry it to the row's end.
            write_state: false,
            commit_len: Tensor::ABSENT,
            ..cut
        })
    }

    /// The pool with every per-request seat cut to the asking node's window,
    /// before either segment claims its end of the boundary.
    ///
    /// `absolute` is which READING of the lane axis the asking op wants, and
    /// it is spent only under a plane base: `false` slices at `lane_offset`,
    /// which is what every per-step scan on [`crate::SHIFTED`] has always
    /// taken, and `true` hands the FIRE's vectors whole to a chunked arm that
    /// adds `win[3]` itself ([`recurrent_absolute`](Run::recurrent_absolute)
    /// carries the argument). Off a plane base the two answers are the same
    /// vector, because there is no seat to add and the ids are window-local.
    fn recurrent_cut(&self, id: ValueId, absolute: bool) -> RecurrentPool {
        match self.cache(id) {
            CachePool::Recurrent(pool) => {
                let window = self.window().span;
                // **ONE PREDICATE FOR ALL FOUR VECTORS**, because a launch
                // reads them at ONE index: `slot_ids`, the fold predicate, the
                // commit length and the segment origin are all `[r]` in
                // `attn/ssm.cuh` and `attn/ple.cuh`, so a fire-wide slot map
                // beside a sliced predicate would fold the wrong lane's rows.
                // An absent seat stays absent on either reading — `skip` of a
                // null handle is a null handle, and a null handle handed whole
                // is still null.
                let lanes = |table: Tensor| {
                    if absolute && self.plane_base() {
                        table
                    } else {
                        skip(table, window.lane_offset, window.lanes)
                    }
                };
                RecurrentPool {
                    slot_ids: lanes(pool.slot_ids),
                    write_state_mask: lanes(pool.write_state_mask),
                    commit_len: lanes(pool.commit_len),
                    begin_at: lanes(pool.begin_at),
                    ..*pool
                }
            }
            CachePool::Kv { .. } => panic!(
                "value {} is a paged kv space, and this op scans a recurrent state pool",
                id.0
            ),
        }
    }

    fn cache(&self, id: ValueId) -> &CachePool {
        let at = id.0 as usize;
        match &self.values[at].def {
            Def::Cache(c) => {
                let row = *c as usize;
                self.caches.0.get(row).unwrap_or_else(|| {
                    panic!(
                        "value {at} is cache space {row}, and the shell binds {} pools",
                        self.caches.0.len()
                    )
                })
            }
            _ => panic!("value {at} is not a cache space; tensors resolve through `Run::tensor`"),
        }
    }

    fn geometry(&self, at: usize, space: u32) -> &CacheGeometry {
        let space = space as usize;
        self.fire.geometry.get(space).unwrap_or_else(|| {
            panic!(
                "value {at} names cache space {space}, and this fire binds {} geometry spaces",
                self.fire.geometry.len()
            )
        })
    }

    /// Everything a plan op's builder takes, in one place: the host geometry
    /// twins of the cache space its `kv_indptr` names, cut to the op's own
    /// window, beside the reading and the grant of the SCHEDULE it defines.
    ///
    /// Two keys, because they are two facts. `kv_indptr` is a device value
    /// whose `Def` says which space it is, and the space's
    /// [`CachePlanning`] holds the twin the builders actually walk — the
    /// duality, routed in one place and windowed in the same one. `plan` is
    /// the struct value the op DEFINES, and its [`ScheduleSeat`] holds the
    /// head width, the query heads and the window this carving is for; a
    /// family that reads its one page-id space at two of those mints two plan
    /// values (build log 20's first blocker).
    pub(crate) fn planning(&self, geom: ValueId, plan: ValueId) -> Planning<'_> {
        let at = geom.0 as usize;
        let Def::Input(RuntimeInput::Geometry { space, .. }) = &self.values[at].def else {
            panic!(
                "value {at} is not declared cache geometry, and a plan op routes to its \
                 cache space through its geometry input"
            );
        };
        let seat = self.geometry(at, *space);
        let seat = seat.planning.as_ref().unwrap_or_else(|| {
            panic!(
                "cache space {space} carries no planning seat; the shell binds the host \
                 geometry twins before a plan op can fire"
            )
        });
        let run = self.place.run.get() as usize;
        let schedule = self
            .fire
            .schedules
            .get(run * self.fire.plan_values + plan.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| {
                panic!(
                    "plan value {} carries no schedule seat for run {run} of its \
                     window; the shell reads every schedule's reading off the plan op \
                     that defines it and carves one grant per run of the region that \
                     builds it, so a plan op firing without one is a value \
                     `store::kv::probe` never walked",
                    plan.0
                )
            });
        let window = self.window();
        let span = window.span;
        // **AND HOW WIDE THIS SCHEDULE MAY BE CARVED** — the KEY's ceilings
        // for any plan the bodies path can serve, and `None` (this window's
        // own lanes and rows) for every other fire. The plan-at-bucket-ceiling
        // design, chunks 3, 4 and 5, finished by the ceiling design's
        // Option B.
        //
        // **THE POINT OF IT.** `Run::schedule_shape` hashes `Shape`, so a
        // count that follows the fire's lanes is a hash that follows the
        // fire's lanes, and a body captured at one batch size is demoted the
        // moment the next batch size arrives. Nothing in a `record::BodyKey`
        // moves inside that key, so a schedule carved at a number the key
        // spells has a hashed image that does not either — and the lanes and
        // rows between the fire's own and the ceiling are chunk 2's GENUINELY
        // EMPTY ones (flat page bounds, zero lengths, zero rows), so widening
        // the carve adds work items that read emptiness and are retired
        // rather than work items that read the last fire's bytes.
        //
        // **AND THE NUMBER THE KEY SPELLS IS PER CLASS** (Option B). Chunks 3
        // to 5 had one number for the whole fire — `Composition::bucket` —
        // and a total says nothing about the split, so the carve could only
        // be taken where the window WAS the whole fire. A `record::Ladder`
        // carries a rung per present class in the order the rows stand, so
        // three numbers become key functions at once and a WINDOWED class can
        // take all three: how many rows stand in front of it (the prefix sum
        // of the rungs before it), how many rows it may be carved over (the
        // sum of its own classes' rungs), and how many lanes.
        // `Carve::ceiling` is that arithmetic on the ROW axis, and the line
        // under it is the same walk read as LANES.
        let carve = self
            .captured()
            .then(|| self.carve.and_then(|carve| carve.ceiling(span)))
            .flatten();
        // **AND THE SAME ARITHMETIC READ AS LANES**, which is a second number
        // and not a second call of the first: a rung is a ROW ceiling, and the
        // most LANES a class of this key can bring is that rung AND the seats
        // this load holds, whichever is smaller
        // (`record::Carve::lanes`). Both are functions of the
        // `record::BodyKey` and of load constants, so nothing about the split
        // is in either — and taking the tighter one on the lane axis is what
        // keeps a mixed key's prefix from consuming the whole of what step 4d
        // could stage and leaving the class behind it with no ceiling at all.
        let carve_lanes = self
            .captured()
            .then(|| self.carve.and_then(|carve| carve.lanes(span)))
            .flatten();
        // **HOW MANY LANES, AND WHERE THEY COUNT FROM.** `(origin, count)`, or
        // `None` for this window's own pair.
        //
        // **THREE CLAUSES, AND EACH ONE IS A THING THE CEILING NEEDS.**
        //
        // **AND NEITHER THE KIND NOR THE WHOLE-FIRE RESTRICTION IS ONE OF
        // THEM ANY MORE** (chunk 5, then Option B). Chunks 3 and 4 named the
        // two builders they had an argument for — decode, then fa2 prefill —
        // and kept sm90 and mla literally unable to reach this block. Chunk 5
        // removed the KIND clause, because the other two want the same number
        // for the same reason and neither can be hurt by it: `sched_sm90`
        // reads the carved lane count only through `max_num_works_per_head`
        // (an allocation bound, and its work items are staged behind a CSR the
        // launch walks), and `sched_mla` reads it only as the divisor of
        // `avg_packed_qo_len` (the cluster split). Neither builder's launch
        // reads `num_requests` at all: the mla arms take their lane count from
        // their own `q.indptr` (`kernels_cuda::attn::mla`), and the sm90
        // launcher answers a typed refusal before it launches anything.
        //
        // Option B removes the WHOLE-FIRE clause, which was the one thing
        // chunk 3 could not argue away: "a first-of-two-classes window widened
        // to the ceiling would swallow the NEXT class's lanes, which are live
        // and not empty". It cannot now, because the ceiling it widens to is
        // its OWN class's rung and the class behind it begins at the prefix
        // sum of the rungs — the two carves are disjoint by construction, and
        // the lanes between them are the padded ones step 4d wrote.
        //
        // * **this REGION is one a graph holds, and the ladder resolved this
        //   span** ([`captured`](Run::captured)). Nothing off the bodies path
        //   staged a padded lane table at all, so there is no ceiling to carve
        //   at and the EAGER path answers exactly what it answered before this
        //   block existed — and neither has an ISLAND of a segmented body,
        //   which is re-issued eagerly at this fire's own lanes every fire
        //   (the tier-2 campaign). A span the ladder answers `None` for is a
        //   gathered or grouped one, which the clause below refuses again for
        //   its own reason.
        // * **the launch beside it takes the PLANE's base**
        //   ([`plane_base`](Run::plane_base)). This is the load-bearing one:
        //   a schedule carved past the fire's lanes is read by a launch whose
        //   `paged_kv` bound is `indptr[lane_offset + num_requests]` and whose
        //   `q_indptr` has to spell that many lanes, and only
        //   `Run::pool_absolute` and `Run::qo_indptr_absolute` hand a launch
        //   the fire's vectors whole. A windowed launch handed tables cut to
        //   its own lanes and a ceiling carve over those is
        //   `kernels_cuda::attn`'s `lanes_carry` refusal. `plane_base` also
        //   carries the gathered and grouped refusals, so neither is spelled
        //   again.
        // * **and the staged vectors actually cover it.** The ceiling is read
        //   OFF the padded sources rather than recomputed from the ladder:
        //   step 4d padded the page CSR, the kv lengths and the fire-wide qo
        //   vector to `min(ladder LANE reach, max_lanes)` on exactly this path, so
        //   the shortest of the three IS that number — and taking it this way
        //   means the carve can never name a lane the staging did not define.
        //   A source that somehow fell short leaves the carve at the window's
        //   own lanes, which reads nothing wrong and costs a reshape — and a
        //   reshape is not cheap any more: under a SEALED map the fire keeps
        //   its eager numbers and is declined, for the life of the load. The
        //   lane reach being capped at the seats (`record::Carve::lanes`) is
        //   what puts that corner out of an ordinary deployment's reach.
        //
        // **AND THE PIN UNDER THE THIRD CLAUSE IS THAT THE CARVE STILL COVERS
        // THE FIRE.** The launch reads lanes `[live.lane_offset,
        // live.lane_offset + live.requests)` — the ids the schedule STAGES —
        // and `fa2_abi`'s protective page bound is `shape.lane_offset +
        // shape.num_requests`, so a carve that reached less far than the fire
        // does would clamp a legitimate page to page zero. Unclamped it
        // cannot: the ceiling origin is a prefix sum of ROW rungs and the live
        // origin a prefix sum of LANE counts, and a lane carries at least one
        // row, so the first dominates the second. Clamped by a short source it
        // could, and the filter is what says so.
        let ceiling: Option<(u32, u32)> = self
            .plane_base()
            .then_some(carve_lanes)
            .flatten()
            .and_then(|(before, own)| {
                let staged = self
                    .windows
                    .qo_absolute()
                    .map_or(0, |bounds| bounds.rows.saturating_sub(1))
                    .min((seat.kv_indptr.len() as u32).saturating_sub(1))
                    .min(seat.kv_len.len() as u32);
                let covered = staged.checked_sub(before)?;
                Some((before, own.min(covered)))
            })
            .filter(|(before, lanes)| {
                *lanes >= span.lanes && before + lanes >= span.lane_offset + span.lanes
            });
        let kind = self.declared(plan);
        // **AND HOW MANY ROWS IT MAY BE CARVED OVER** — the KEY's rows for a
        // ROW-READING schedule the bodies path can serve (fa2 prefill in
        // chunk 4, sm90 and mla since chunk 5), and `None` (this window's own
        // rows) for decode, which reads no row total at all. The ROW axis's
        // half of the same argument the lane ceiling above makes.
        //
        // **WHAT MOVES WITH IT.** `sched_prefill` reads `total_num_rows` in
        // four places and every one of them is hashed: the graph shape's
        // `max_seq_len` picks `cta_tile_q`, which picks the KERNEL SYMBOL; the
        // tile count feeds `padded_batch_size`, which is the prefill grid; the
        // merge indptr is allocated `[rows + 1]`, which moves every offset
        // after it; and the number itself rides `PrefillPlanInfo` and
        // `PrefillPlan::total_tokens`. So a prefill body captured at one row
        // total is demoted by the next, and freezing the four at the bucket is
        // the whole of what this chunk buys.
        //
        // **AND IT IS NOT THE SAME NUMBER AS THE LAUNCH'S GRID, WHICH IS
        // WORTH SAYING SINCE THE GRID GREW A CEILING TOO.**
        // [`carve_rows`](Run::carve_rows) answers "how many rows is the
        // LAUNCH issued over" and this answers "how many rows is the SCHEDULE
        // carved at", and the two are allowed to differ: a whole-fire window's
        // launch takes `pad.bucket` (the number `Ctx::opaque_rows` already
        // pads its GEMMs to) where this takes the ladder's own sum, which on a
        // single decode class can be the lane ceiling below it. What matters
        // is not that they are equal but that both are functions of the
        // `record::BodyKey` — the grid because `record::Body::grids` is
        // compared against it, this one because `Run::schedule_shape` hashes
        // what it produces — and both are.
        //
        // **AND THREE CLAUSES RATHER THAN THE THREE ABOVE, BECAUSE THE ROW
        // CEILING ADDRESSES NOTHING.** The lane ceiling is read by a LAUNCH —
        // `q_indptr` has to spell every lane the schedule names, which is why
        // it takes the plane-base clause and the staged-source clause. The
        // rows above the fire's are never addressed by anybody: no work item
        // is emitted for them (the walk is `live.requests` long), the merge
        // indptr's tail is allocated and never read, and the ONE launch that
        // is gridded off the carved count — the cascade fold, whose
        // `max_seq_len` is `info.total_num_rows` — bounds its loop by
        // `min(win[0], *seq_len_ptr)` and not by the grid
        // (`cascade.cuh`'s `PersistentVariableLengthMergeStatesKernel`). Both
        // of those are this fire's own numbers. So what is left to ask is:
        //
        // * **the plan value declares a schedule that READS a row total** —
        //   which chunk 5 makes three of the four kinds rather than one.
        //   Decode is the exclusion and the only one: `sched_decode` never
        //   sees a row count, so handing it one would be a number nothing
        //   reads. The two chunk 5 added read it in the same shape fa2 does:
        //   `sched_sm90`'s `max_num_works_per_head` is
        //   `ceil(rows / cta_tile_q) + batch - 1`, which sets
        //   `same_schedule_for_all_heads` and, through `max_total_num_works`,
        //   every one of its eight int offsets; `sched_mla`'s carved average
        //   is `rows * heads / batch`, which sets `cluster_size` and with it
        //   `num_blks_x`/`num_blks_y`. Both of those are hashed
        //   (`Run::schedule_shape`), and both are what chunk 5 freezes.
        // * **and this REGION is one a graph holds**
        //   ([`captured`](Run::captured)) — which is what guarantees the
        //   fold's `win` seat is armed (`Run::live_at` arms every region a
        //   body was admitted on, `Windows::admits`), and therefore that the
        //   fold's bound is live rather than the baked ceiling. **AND IT IS
        //   ASKED PER REGION SINCE THE TIER-2 CAMPAIGN**: an island's plan is
        //   rebuilt every fire and read by launches the graph does not hold,
        //   so carving it at the key's rows would hash a schedule nothing
        //   replays and grid a merge indptr past the rows the island brought.
        //   **AND IT IS THE LAST CLAUSE, BECAUSE IT CARRIES THE
        //   PAD.** A third one stood here — `pad.bucket > pad.rows`, "the
        //   shell actually quantized" — written for `PIE_CUDA_PAD=off`, where
        //   a rung is the row count itself and the carve would buy nothing.
        //   That arm cannot reach a body any more (`Shell::prepare`'s gate
        //   requires the pad), and what the clause DID still reach was the
        //   padded fire whose rows land exactly on its bucket: one split of a
        //   lattice point, carving at its own row total while every other
        //   split of the same `record::BodyKey` carved at the point. Two
        //   hashed schedules under one key, which is a reshape per fire and,
        //   under a sealed map, an eager walk for good.
        //
        // **AND THE NUMBER IS THIS WINDOW'S OWN CLASSES' RUNGS, CAPPED AT THE
        // FIRE'S BUCKET** (Option B). `carve` above is the sum of the rungs of
        // the classes this span covers, which is what makes a windowed class's
        // row count a function of the key rather than of the fire's total; the
        // cap is what keeps a window that covers SEVERAL classes from being
        // carved wider than the fire's bucket, which is the number the float
        // grant was sized at (`inputs::reserve`'s `prefill_float_bytes`) and
        // also a key function. Both numbers dominate this span's rows — the
        // sum because a class's rung holds its rows, the bucket because it
        // holds the whole fire's — so the `min` of them still does. For a
        // single-class whole-fire window the two are equal and this is exactly
        // the number chunk 4 carved.
        //
        // **AND THAT FINISHES THE HALF-FROZEN WINDOWED CLASS CHUNKS 4 AND 5
        // LEFT** — the fa2 prefill one whose `shape.num_requests` still
        // followed the fire, and the sm90 and mla ones whose two movers are
        // functions of the row count AND the lane count together and so
        // needed both ceilings. A windowed class now takes all three numbers
        // off the ladder: its rows here, its lanes and its lane origin above.
        // Its hashed image is a function of the `record::BodyKey` and the
        // load, which is what `record::BodyStats::reshapes` being an anomaly
        // counter finally means for every kind rather than for the whole-fire
        // ones only.
        let rows_ceiling: Option<u32> = (self.captured()
            && !matches!(kind, StructKind::AttnDecodePlan))
            .then_some(carve)
            .flatten()
            .map(|(_, own)| own.min(self.pad.bucket))
            .filter(|rows| *rows >= span.rows);
        // **A GATHERED WINDOW'S TWINS ARE THE ONES RE-CUT WITH ITS LANES.**
        // The comment on `Planning` says a slice is the whole adaptation
        // because "the window is contiguous in lanes"; a copy's is not, so
        // the builder gets the union's vectors — the page bounds as a fresh
        // prefix sum over the compacted page-id list, the kv lengths lane by
        // lane in gathered order. `num_requests` is the union's lane count,
        // which is what makes the ONE schedule cover every run.
        let (kv_indptr, kv_len) = match window.gathered.as_ref().and_then(|g| g.spaces.get(*space as usize)) {
            Some(gathered) => (
                gathered.page_indptr_host.as_slice(),
                gathered.kv_len_host.as_slice(),
            ),
            None => {
                let first = span.lane_offset as usize;
                // **AND THE SLICE IS THE CARVE'S AND NOT THE FIRE'S.** The
                // builders walk these to the lane count `shape` states, so a
                // ceiling carve over a slice cut at the window's own lanes
                // would read past the end of one and take its spans from
                // whatever follows.
                //
                // **AND IT STILL BEGINS AT THIS WINDOW'S OWN FIRST LANE**,
                // which is the one place a ceiling and an ORIGIN part company
                // (Option B). The slice is what the builder walks to find
                // THIS FIRE's per-lane spans, so entry `r` of it has to be
                // this window's `r`-th live lane; the ceiling origin beside it
                // is a REACH — how far into the fire's whole vectors the
                // launch may address — and it is spelled on `shape` where the
                // launch reads it. `first + lanes` is inside the padded
                // source by construction: `first` is a prefix sum of LANE
                // counts, the ceiling's origin is the prefix sum of the ROW
                // rungs in front of it and dominates it, and step 4d padded
                // out to the sum of every rung. The fallbacks below are the
                // same belt they were.
                let lanes = ceiling.map_or(span.lanes, |(_, lanes)| lanes) as usize;
                (
                    seat.kv_indptr
                        .get(first..=first + lanes)
                        .unwrap_or(&seat.kv_indptr),
                    seat.kv_len
                        .get(first..first + lanes)
                        .unwrap_or(&seat.kv_len),
                )
            }
        };
        // **THE TWO CHANNELS, BUILT SIDE BY SIDE OUT OF ONE SPAN.** `shape`
        // is what the schedule is CARVED at and rides the plan payload
        // `Run::schedule_shape` hashes; `live` is what this fire brought and
        // reaches the device only through the staged image. `ceiling` above
        // is what parts them, and it parts exactly one field of one plan
        // kind: everywhere else the two are still written twice from the same
        // expression and the asserts under them say so.
        let shape = Shape {
            // **THE CARVED COUNT: THE LADDER'S LANE CEILING WHERE ONE WAS
            // TAKEN, AND THIS WINDOW'S LANES EVERYWHERE ELSE.** This is the
            // one number `Run::schedule_shape` hashes that used to follow the
            // batch, and the whole of what chunk 3 moves.
            num_requests: ceiling.map_or(span.lanes, |(_, lanes)| lanes),
            // **WHERE THIS SCHEDULE'S REQUEST NUMBERS COUNT FROM**, and
            // it is the same predicate the pointers beside them answer
            // to. A launch handed per-lane tables SLICED to its window
            // numbers its requests from that window's zero, which is what
            // every fire but a plane-based one gets; a launch handed the
            // fire's tables WHOLE has to name fire lanes, because nothing
            // in an FA2 params block adds an offset for it
            // (`Run::pool_absolute`, `Run::mask_indptr`).
            //
            // **AND ASKING IT HERE, AT THE PLAN OP, IS ASKING IT AT THE
            // READER.** A schedule may only be read in the window it was
            // built in (`model`'s `no_schedule_straddles_its_readers`),
            // so the plan node and its attention nodes share this span
            // exactly. Their REGIONS may differ, and so may
            // `shifted[region]` — but a region a graph holds is one that is
            // whole-fire or shifting (`Windows::admits`), and a window with a
            // non-zero `lane_offset` is not whole-fire. So wherever this
            // answer is non-zero, both regions are shifting and both read
            // the plane's base; and wherever it could disagree, the span
            // begins at lane zero and both answers are zero. **AND WHERE ONE
            // OF THE TWO IS AN ISLAND, BOTH ARE** — a schedule and its
            // readers share a span, and `Windows::admits` reads the span's
            // shape, so a plan region and its attention regions land on the
            // same side of every cut.
            //
            // **AND WHERE A CEILING WAS TAKEN IT IS THE LADDER'S ORIGIN AND
            // NOT THE FIRE'S** (Option B). This field is a REACH on the
            // structure channel — `fa2_abi`'s protective page bound is
            // `lane_offset + num_requests`, and `sched_prefill` sizes its
            // absolutely-indexed `o_indptr` at `[lane_offset + batch + 1]` —
            // where `Live::lane_offset` beside it is the ORIGIN the staged
            // request ids are written from (`live.lane_offset + r`). Two
            // channels, and only this one is hashed: raising it to the prefix
            // sum of the rungs in front of this class is what stops the reach
            // from following the split, and the staged ids underneath it
            // still name the fire's own lanes. The allocation between the two
            // is a dead prefix, which is exactly what that vector's own note
            // in `sched_prefill::schedule` says it is.
            lane_offset: match ceiling {
                Some((first, _)) => first,
                None if self.plane_base() => span.lane_offset,
                None => 0,
            },
            ..schedule.shape
        };
        let live = Live {
            requests: span.lanes,
            lane_offset: if self.plane_base() { span.lane_offset } else { 0 },
            row_offset: if self.plane_base() { span.row_offset } else { 0 },
            // `Run::total_tokens` IS `span.rows`, and it is what this fire
            // brought. The carved ROW count beside it is
            // [`Planning::rows`] — equal to this everywhere but on the row
            // ceiling, which raises that twin and leaves this one alone
            // (chunk 4, per class since Option B).
            rows: span.rows,
        };
        // **THE PIN, IN THE FORM THE CEILINGS LEFT IT.** The two channels no
        // longer have to be equal — that is the point of having two — but they
        // may part in exactly one direction: a carve is WIDER than the fire or
        // the same, never narrower, on the lane axis and on the row axis
        // alike, and on the lane axis the carve's INTERVAL has to contain the
        // fire's. The row ORIGIN stays this window's on either channel,
        // because nothing here raises one: the row ceiling raises a COUNT that
        // no launch addresses off. Anything else is a plumbing mistake, not a
        // design one.
        debug_assert!(shape.num_requests >= live.requests);
        debug_assert!(shape.lane_offset >= live.lane_offset);
        debug_assert!(
            shape.lane_offset + shape.num_requests >= live.lane_offset + live.requests
        );
        let rows = rows_ceiling.unwrap_or(span.rows);
        // The row axis's half of the same pin: a carve is wider than the fire
        // or the same, never narrower.
        debug_assert!(rows >= live.rows);
        Planning {
            kv_indptr,
            kv_len,
            shape,
            live,
            rows,
            window: schedule.window,
            workspace: schedule.workspace,
        }
    }

    /// The `StructKind` a plan op's output value declares — how the
    /// prefill-building arm tells fa2 from sm90: the trace wrote the choice
    /// into `Trace::values`, the arm only follows it.
    pub(crate) fn declared(&self, id: ValueId) -> StructKind {
        match &self.values[id.0 as usize].ty {
            Ty::Struct(kind) => *kind,
            Ty::Tensor { .. } => panic!(
                "value {} declares a tensor, and a plan op defines a struct",
                id.0
            ),
        }
    }

    /// The dsv4 compressor slabs, for `attention.pool_gather`'s seam.
    pub(crate) fn slabs(&self) -> PoolSlabs {
        self.fire.tables.pool_state.unwrap_or_else(|| {
            panic!(
                "this fire binds no dsv4 compressor slabs, which `attention.pool_gather` reads beside the pool"
            )
        })
    }

    /// A hash of the SHAPE of every plan payload this fire built — every
    /// number off a plan struct that can reach a kernel argument, and none of
    /// the CONTENTS that reach the device through the workspace.
    ///
    /// **THE ONE THING A GRAPH KEY CANNOT SEE.** A recorded fire bakes the
    /// plan's offsets, its padded batch size and its tile width into the
    /// launches it recorded; the prepare phase rebuilds the plan every fire
    /// and the replay keeps reading the captured numbers. Under
    /// [`FireBindings::capture`] the builders carve graph-shaped schedules at
    /// the KEY's ceilings ([`Run::planning`]), so those numbers are a function
    /// of the `record::BodyKey` and the load — but that is a property of
    /// somebody else's arithmetic, and this is the fire path checking it
    /// rather than believing it. A disagreement DEMOTES the body: the fire
    /// walks eagerly, produces its own numbers and re-captures the key at the
    /// shape that arrived (`record::BodyStats::reshapes`), which is a counter
    /// naming a builder rather than a slightly wrong logit.
    ///
    /// The Debug image is the hashed form on purpose: it covers every field
    /// the plan structs have TODAY and every field they grow, where a
    /// hand-listed hash would silently stop covering the one that was added.
    /// It allocates nothing — the formatter writes straight into the hasher.
    pub(crate) fn schedule_shape(&self) -> u64 {
        use core::fmt::Write;
        use std::hash::{DefaultHasher, Hasher};

        struct Sink(DefaultHasher);
        impl Write for Sink {
            fn write_str(&mut self, text: &str) -> core::fmt::Result {
                self.0.write(text.as_bytes());
                Ok(())
            }
        }

        let mut sink = Sink(DefaultHasher::new());
        for (at, slot) in self.structs.iter().enumerate() {
            let Some(slot) = slot else { continue };
            // **AN ISLAND REGION'S PLAN IS NOT IN THE HASH.** Its builder
            // stood down from every ceiling (`Run::captured`), so its numbers
            // follow the fire — and its readers are islands too (the
            // mask-family weld), re-issued eagerly off THIS fire's payload,
            // so no capture holds a stale copy for the hash to catch. Hashing
            // it would reshape a body once per split for a difference no
            // replay reads.
            if self.bodied
                && matches!(
                    self.admits.get(self.struct_region[at] as usize),
                    Some(crate::window::Admit::Island)
                )
            {
                continue;
            }
            let _ = write!(sink, "{at}:");
            // `int_upload` is the one field deliberately left out: it is this
            // fire's schedule CONTENTS, staged into a pointer-stable
            // workspace, and it is supposed to differ every fire.
            let _ = match slot {
                StructSlot::Decode(p) => write!(
                    sink,
                    "d{:?}{:?}{:?}{:?}",
                    p.info, p.workspace, p.shape, p.window
                ),
                // **AND `mask_indptr` IS HERE BECAUSE IT IS A POINTER A
                // CAPTURE BAKES.** The engine binds the span table onto the
                // plan at build (`Run::mask_indptr`), `attention.masked`
                // hands it straight to the params block, and which vector it
                // is depends on the window the plan op fired in. Every other
                // field here is a NUMBER off the payload; this one is an
                // address, and an address that moved between a capture and
                // its replay is exactly the stale read this hash exists to
                // name.
                StructSlot::Prefill(p) => write!(
                    sink,
                    "p{:?}{:?}{:?}{:?}{}{}{}{:?}",
                    p.info, p.workspace, p.shape, p.window, p.total_tokens, p.causal,
                    p.graph_capturable, p.mask_indptr
                ),
                StructSlot::PrefillSm90(p) => write!(
                    sink,
                    "s{:?}{:?}{:?}{}{}",
                    p.info, p.workspace, p.shape, p.total_tokens, p.causal
                ),
                // **AND `causal` IS HERE BECAUSE NOTHING ELSE IN THIS ARM
                // MOVES WITH IT.** The engine derives the latent builder's
                // causal word per fire off the window's own boundaries
                // (`Run::multi_token`), and it steers the schedule: it picks
                // which kv end each work tile runs to, and through that the
                // cluster work split and the merge tables. Those live in
                // `int_upload`, which this hash deliberately does not cover
                // — it is supposed to differ every fire — while `info`,
                // `workspace` and the head count can all agree across a flip.
                // So two fires of one key could disagree about the ONE input
                // the payload never wrote down and the hash would call them
                // the same schedule.
                //
                // The consequence is deliberate: a resident body whose fire
                // flips `causal` is re-captured rather than replayed off the
                // other reading's graph, and `record::BodyStats::reshapes`
                // says it happened. A counted demotion is the answer this hash
                // exists to give.
                StructSlot::Mla(p) => write!(
                    sink,
                    "m{:?}{:?}{}{}",
                    p.info, p.workspace, p.num_heads, p.causal
                ),
            };
        }
        sink.0.finish()
    }

    /// Did every schedule this fire built keep its graph-shaped padding?
    ///
    /// The builders' answer OUT (`PrefillPlan::graph_capturable`): a
    /// graph-shaped prefill schedule that did not fit its workspace grant
    /// falls back to one that fits and is not capturable, and capturing that
    /// would bake this fire's row count into a graph the next fire replays at
    /// another. The shell reads this before it captures, and stays eager when
    /// it is false.
    pub(crate) fn capturable(&self) -> bool {
        self.structs.iter().flatten().all(|slot| match slot {
            StructSlot::Prefill(plan) => plan.graph_capturable,
            _ => true,
        })
    }

    /// Store a plan payload a prepare-phase arm just built.
    pub(crate) fn put(&mut self, id: ValueId, built: StructSlot) {
        let at = self.struct_at(id);
        self.structs[at] = Some(built);
        self.struct_region[at] = self.place.region.get();
    }

    /// One built slot, whichever kind it holds — for the arm that routes on
    /// the kind (prefill's fa2/sm90 fork); the typed accessors below are the
    /// single-kind reads.
    pub(crate) fn slot(&self, id: ValueId) -> &StructSlot {
        let at = self.struct_at(id);
        self.structs[at].as_ref().unwrap_or_else(|| {
            panic!(
                "value {} holds no plan payload for run {} of its window; its plan \
                 op has not fired, and the prepare phase runs first",
                id.0,
                self.place.run.get(),
            )
        })
    }

    /// The decode plan a consuming arm names.
    pub(crate) fn decode_plan(&self, id: ValueId) -> &DecodePlan {
        match self.slot(id) {
            StructSlot::Decode(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes a decode plan",
                id.0
            ),
        }
    }

    /// The fa2 prefill plan a consuming arm names.
    pub(crate) fn prefill_plan(&self, id: ValueId) -> &PrefillPlan {
        match self.slot(id) {
            StructSlot::Prefill(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes an fa2 prefill plan",
                id.0
            ),
        }
    }

    /// The mla plan a consuming arm names.
    pub(crate) fn mla_plan(&self, id: ValueId) -> &MlaPlan {
        match self.slot(id) {
            StructSlot::Mla(plan) => plan,
            _ => panic!(
                "value {} holds another plan kind, and this op consumes an mla plan",
                id.0
            ),
        }
    }
}

/// A row-indexed handle, advanced past `skip` rows and cut to `keep` of them.
///
/// The one arithmetic every windowed table shares: a pointer plus an extent,
/// which is exactly what design §0 says a windowed kernel takes.
fn skip(handle: Tensor, skip: u32, keep: u32) -> Tensor {
    if skip == 0 && keep >= handle.rows {
        return handle;
    }
    let stride = u64::from(handle.width)
        * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or_else(|| {
            panic!(
                "a {:?} table has no element size and so no row to step by",
                handle.dtype
            )
        });
    Tensor::new(
        handle.ptr + u64::from(skip) * stride,
        keep.min(handle.rows.saturating_sub(skip)),
        handle.width,
        handle.dtype,
    )
}

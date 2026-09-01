//! The resident fire inputs: one allocation, carved once, overwritten every
//! fire and never moved.
//!
//! **POINTER-STABLE IS THE WHOLE POINT.** Step 5 records these addresses into
//! a graph that is never re-captured, so a buffer that were reallocated when a
//! fire got bigger would leave the graph reading the old one — which does not
//! fault, because the old allocation is still mapped. So every vector is
//! reserved at the budget's ceiling at load and a smaller fire writes its
//! prefix; the LENGTH rides on the handle and on the geometry, and the
//! address never changes. Eager mode does not need this yet, which is exactly
//! why it is written now: the eager shell is the golden the recorded one is
//! diffed against, and a difference in where the bytes live would be a
//! difference the diff cannot see.
//!
//! # What is here, and what the plan names
//!
//! ```text
//! tokens, positions        RuntimeInput::Tokens / ::Positions
//! per space: indptr,       RuntimeInput::Geometry { space, kind }
//!   indices, last_page_len,
//!   kv_len, write_page,
//!   write_offset
//! window boundaries        ambient — no op names it (design §5); one
//!                          rebased `[lanes + 1]` run per WINDOW, not one
//!                          per fire
//! absolute qo boundaries   ambient — the SAME boundaries read a second
//!                          way: one un-rebased `[lanes + 1]` vector for the
//!                          whole FIRE, sliced by lane rather than rebased
//! live rows                ambient — the staged-geometry seat (bodies
//!                          design); one `u32` per (REGION, run), armed onto
//!                          the context rather than passed as an operand
//! per space: mask bits     RuntimeInput::Mask { space }
//! adapter routes           RuntimeInput::AdapterRoutes
//! mask spans               ambient — `attention.masked`'s op-named bits have
//!                          no seat for their per-request byte offsets, so
//!                          the plan-prefill arm binds one onto the schedule
//! row_valid                the padding mask the kv writers read past the IR
//! slot_ids                 which recurrent bank each lane owns
//! plan workspace           the prepare phase's staging, granted per plan kind
//! ```
//!
//! # A window's boundaries are carved TWICE, and read by two consumers
//!
//! The two qo rows of that table are one fact and two readings of it, and the
//! difference is what the reader's POINTER already is. The per-window blob is
//! REBASED — entry 0 is 0 — because a windowed launch is handed the window's
//! first row and a ragged view's boundaries are offsets INTO the rectangle it
//! was handed; that is what every seated pie kernel, `Cursor::count_of` and
//! both plan builders take, and it is not moving. The fire-wide vector is the
//! same lane bounds UN-SUBTRACTED, and its consumer is the one whose pointer
//! is not the window's: a region on [`crate::SHIFTED`] under a body gets its
//! PLANE's base from `Run::cut`, so the CSR beside it has to count from the
//! plane's zero too. That consumer takes the vector WHOLE — unlike
//! [`Handles::mask_indptr`], which may be cut at `lane_offset` because
//! nothing bakes its address. This one is reached only on the bodies path,
//! where a recorded launch keeps the pointer it was handed and `lane_offset`
//! moves between fires of one key; which entries are a launch's requests is
//! the schedule's business, not the pointer's.
//!
//! Written only for a fire the shell routed to a body, and carved by every
//! load — the live-rows seat's rule, for the live-rows seat's reason.
//!
//! The unseated ones are not oversights: the qo boundaries were deliberately
//! unnamed, and `row_valid`/`slot_ids`/the mask spans/the workspace are
//! engine facts the entries take beside the ops' operands (the `MENLO-SEAM`
//! markers `run.rs` catalogues). The live-rows seat is the same kind of fact
//! one step further out: no entry takes it as an operand at all — the walk
//! stamps its ADDRESS onto the context per region
//! (`kernels_cuda::Ctx::arm_stage`) and the entries that support it pass that
//! address as their `win` argument, which is `0` and therefore absent
//! wherever nothing armed one.
//!
//! # The mask slab is reserved against the CONTEXT, not measured
//!
//! A lane's mask expands to `rows x (held + rows)` bits ([`crate::mask`]), so
//! the fire-wide worst case is every row of the ceiling against a full
//! context — `max_tokens * pages_per_slot * page_size / 8`, plus one byte per
//! lane because each lane's region starts on a byte boundary. Reserved like
//! everything else here, for the same reason: the address is recorded into a
//! graph that is never re-captured. A fire past it is `Fault::Ceiling`
//! naming the mask bits, never a reallocation.
//!
//! # Grants are disjoint carvings, one per plan kind
//!
//! `CachePlanning` wants a separate [`Workspace`] for the decode and prefill
//! builders because their staged int images coexist within a fire — the
//! prepare phase builds both before either is consumed. One pool, cut in two,
//! is what that sentence means in bytes.

use kernels_cuda::Tensor;
use kernels_cuda::attn::plan::{Device, Workspace, prefill_graph_padding};
use model_compiler::Budget;
use model_ir::{Dtype, StructKind};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::{Buffer, Pinned};
use crate::error::Result;
use crate::store::SpaceSeat;
use crate::store::kv::{Facts, Geometry, Paging, SpaceFacts};

/// The int side of one plan grant: where a built schedule's offset table is
/// staged.
///
/// Sized rather than measured. The builders refuse at build time when a
/// schedule does not fit its grant — the refusal names the bytes asked and
/// the bytes left — so an over-grant costs address space and an under-grant
/// costs a typed refusal, never a wrong schedule. These are the numbers a
/// deployment would tune; they are stated here because `kernels-cuda`
/// recommends none.
const GRANT_INT_BYTES: u64 = 8 << 20;

/// The float side's FLOOR: split-kv's partial outputs and their
/// log-sum-exps, for a schedule whose padding is the fire's own.
///
/// A graph-shaped prefill schedule wants more, and how much more is a
/// function of the model's attention rather than of a deployment's taste —
/// so [`graph_float_bytes`] computes it and this is only the floor beneath
/// that answer.
const GRANT_FLOAT_BYTES: u64 = 64 << 20;

/// The float workspace ONE graph-shaped prefill schedule can ask for.
///
/// Asked per PLAN VALUE, because a schedule's grant is a function of the
/// reading it was carved at and a family may carve two out of one page-id
/// space — gemma's global schedules want 149 MiB apiece at head width 512
/// where its sliding ones want 74 (`store::kv::SpaceFacts`). And the KIND
/// beside that reading picks WHICH formula is asked: this one is the paged
/// builders', and a latent schedule's partials are [`latent_float_bytes`].
///
/// **A SHORT GRANT HERE DOES NOT FAIL — IT DECLINES**, which is why it is
/// computed rather than guessed. `plan_prefill` asked to be capturable pads
/// its work items to `2·SMs / kv_heads` regardless of how few rows the fire
/// carries (that padding is the whole point: the schedule's shape must be a
/// function of the KEY, not of this fire's kv lengths), and its partial
/// output buffer is `q_heads × padded × cta_tile_q × head_dim` floats. When
/// that does not fit, the builder quietly falls back to a schedule that does
/// and reports `graph_capturable = false` — and the shell, reading that
/// honestly, never captures a prefill again. Measured on the smoke's SKU: 8
/// query heads over 2 kv heads at width 256 on 142 SMs wants 71 MiB, and the
/// old flat 64 MiB grant is the reason a mixed fire declined every time.
///
/// `cta_tile_q` is bounded rather than predicted: 128 is the widest tile the
/// schedule picks, except at `head_dim >= 256` where `plan_prefill` refuses
/// it outright (no `KernelTraits` exist), so 64 is the bound there.
fn graph_float_bytes(facts: &SpaceFacts, sms: u32) -> u64 {
    let padded = u64::from(2 * sms.max(1)) / u64::from(facts.kv_heads.max(1)).max(1);
    let tile = if facts.head_dim >= 256 { 64 } else { 128 };
    let heads = u64::from(facts.q_heads);
    // `tmp_v` is the partials, `tmp_s` their log-sum-exps; both are f32, and
    // each starts on a 16-byte boundary.
    let v = heads * padded * tile * u64::from(facts.head_dim) * 4;
    let s = heads * padded * tile * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// **THE FLOAT WORKSPACE ONE GRAPH-SHAPED FA2 PREFILL SCHEDULE CAN ASK FOR AT
/// THE BUCKET CEILING** — the half of its padding [`graph_float_bytes`] above
/// cannot see, and the H3 hole the plan-at-bucket-ceiling design's chunk 4
/// walked into.
///
/// **THE BOUND ABOVE IGNORES THE TILE COUNT, AND THAT WAS ALWAYS A HOLE.**
/// `sched_prefill` pads its work items to `max(2·SMs / kv_heads,
/// total_num_tiles_q)`, and [`graph_float_bytes`] sizes only the first term.
/// The second is `ceil(rows · group / cta_tile_q) + lanes - 1`, and it
/// outgrows the first as soon as a fire carries more than about
/// `2·SMs·cta_tile_q / group` rows — so a large prefill has always been able
/// to ask for more than it was granted, retry unshaped, and report
/// `graph_capturable = false`. What chunk 4 changes is that the ask stops
/// being a function of the fire: a prefill plan is now carved at the BUCKET's
/// rows and the lane ceiling every time, so the hole is no longer occasional,
/// and a grant that does not cover it is a prefill body that never captures
/// at all.
///
/// So it is computed, at the top of the lattice (`Budget::buckets`' last rung,
/// which is `max_tokens` under the default ladder) and at `Budget::max_lanes`
/// — the widest carve `Run::planning` can ever take — and out of the
/// PLANNER's own arithmetic rather than a restatement of it
/// (`sched_prefill::graph_padding`, re-exported as
/// [`prefill_graph_padding`]).
fn prefill_float_bytes(facts: &SpaceFacts, rows: u32, lanes: u32, device: &Device) -> u64 {
    let (tile, padded) = prefill_graph_padding(
        rows,
        lanes,
        facts.q_heads,
        facts.kv_heads,
        facts.head_dim,
        device,
    );
    let heads = u64::from(facts.q_heads);
    let tile = u64::from(tile);
    // `sched_prefill::layout`'s `tmp_v` and `tmp_s`, exactly.
    let v = heads * padded * tile * u64::from(facts.head_dim) * 4;
    let s = heads * padded * tile * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// The float workspace one graph-shaped DECODE schedule can ask for at the
/// lane ceiling — the half of its padding the prefill bound above cannot see.
///
/// A graph-shaped decode pads its work items to `max_grid_size / gdy` OR to
/// the fire's lane count, whichever is larger (`sched_decode::schedule`), and
/// its partials are `q_heads × padded × head_dim` floats with no tile factor
/// at all. The occupancy half is covered many times over by the prefill
/// number beside it — same `q_heads`, same `head_dim`, a padding within a
/// factor of the SM count, and a `cta_tile_q` of 128 on top — which is why
/// this shell has always asked the prefill formula for a decode grant. The
/// LANE half is new: it is `Budget::max_lanes`, a deployment word, and no
/// expression built out of SM counts and head widths bounds it. So it is
/// computed here and the caller takes the larger of the two.
///
/// A short grant declines rather than fails, exactly as the prefill one does
/// — but a declined decode capture is a decode that never replays, and the
/// lane ceiling is precisely the shape the bucket design fires at.
fn decode_float_bytes(facts: &SpaceFacts, lanes: u32) -> u64 {
    let padded = u64::from(lanes.max(1));
    let heads = u64::from(facts.q_heads);
    // `tmp_v` and `tmp_s`, on `sched_decode::layout`'s own terms.
    let v = heads * padded * u64::from(facts.head_dim) * 4;
    let s = heads * padded * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// The float workspace ONE latent (mla) schedule can ask for.
///
/// **THE LATENT PLANNER DOES NOT SIZE OFF A QUERY RECTANGLE**, which is why
/// its sibling's formula is wrong here rather than merely generous. `plan_mla`
/// sizes its split-kv partials off the CLUSTER GRID
/// (`kernels-cuda/src/attn/sched_mla.rs`'s `schedule` and its `plan`'s float
/// allocator):
///
/// ```text
/// num_clusters   = SMs / cluster_size
/// cluster_tile_q = cluster_size * CTA_TILE_Q (64)
/// rows           = 2 * num_clusters * cluster_tile_q
/// partial_o      = rows * 2 * head_dim_o bytes (the partials are bf16)
/// partial_lse    = rows * 4              bytes, each 16-byte aligned
/// ```
///
/// `cluster_size` cancels: the planner picks 1 or 2 CTAs per cluster from the
/// CARVED average packed query length — the bucket's rows over the bucket's
/// lanes since chunk 5, this fire's own two before it — and whichever it picks
/// the row count is `2 * SMs * 64`, because `num_clusters` divides by exactly
/// what `cluster_tile_q` multiplies by. The integer division can only round it
/// down, so that is the bound this grant is sized at, and neither a fire's
/// choice nor a ceiling's can exceed it — which is why the plan-at-bucket
/// -ceiling design moved the latent split without moving this number.
/// `head_dim_o` is the RANK, because a latent schedule is carved in the
/// absorbed reading: every query head reads the one shared latent plane and
/// writes `kv_lora_rank` floats (`store::kv`'s carving of `Attention::MlaPlan`).
/// glm-5 at rank 512 on 142 SMs wants 17.8 MiB — under the floor below, where
/// the prefill formula asked 3.3 GiB per plan value for the same schedule.
fn latent_float_bytes(rank: u32, sms: u32) -> u64 {
    let rows = 2 * u64::from(sms.max(1)) * 64;
    let partial_o = (rows * 2 * u64::from(rank)).next_multiple_of(16);
    let partial_lse = (rows * 4).next_multiple_of(16);
    (partial_o + partial_lse).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// The alignment every carved region starts on.
const ALIGN: u64 = 256;

/// How long [`Inputs::claim`] spins before it says the ring is oversubscribed.
///
/// Generous by design: it is not a latency budget, it is the line between
/// "the device is a little behind" and "this caller declared a run-ahead depth
/// it is not keeping to". A correctly-sized ring never reaches it.
const CLAIM_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// One plan value's grant, as offsets — turned into a [`Workspace`] once the
/// store is allocated and its base address is known.
#[derive(Clone, Debug)]
struct Grant {
    /// One int carving per run of the region that builds this schedule, in
    /// run order.
    int_at: Vec<u64>,
    float_at: u64,
    float_bytes: u64,
}

/// One kv space's six vectors, as offsets into the store.
#[derive(Debug, Clone, Copy)]
struct SpaceAt {
    indptr: u64,
    indices: u64,
    last_page_len: u64,
    kv_len: u64,
    write_page: u64,
    write_offset: u64,
}

/// The handles one fire's inputs resolve to.
#[derive(Debug, Clone)]
pub struct Handles {
    /// `RuntimeInput::Tokens`.
    pub tokens: Tensor,
    /// `RuntimeInput::Positions`.
    pub positions: Tensor,
    /// Where the packed per-window boundary vectors landed —
    /// [`Windows::bind`](crate::window::Windows::bind) cuts them apart.
    pub windows: u64,
    /// **THE SECOND READING OF THE SAME BOUNDARIES**: where the FIRE-WIDE,
    /// un-rebased qo prefix sums landed — `[lanes + 1]` `i32`, entry `l` the
    /// fire row lane `l` begins at.
    ///
    /// A base and not a `Tensor` because the shape is the window table's to
    /// state ([`Windows::qo_absolute`](crate::window::Windows::qo_absolute)
    /// dresses it), and because a launch takes the vector WHOLE — a body bakes
    /// that pointer, and a lane-sliced one would move between fires of one
    /// key.
    ///
    /// `None` when this fire staged none, which is every fire the shell did
    /// not route to a body. The CARVE is unconditional and the COPY is not —
    /// [`live_rows`](Handles::live_rows)'s rule, for its reason.
    pub qo_absolute: Option<u64>,
    /// Where the LIVE-ROWS seat landed: one `u32` per (region, run), at the
    /// fixed stride [`Windows::live_at`](crate::window::Windows::live_at)
    /// computes — the staged-geometry seat
    /// [`Ctx::arm_stage`](kernels_cuda::Ctx::arm_stage) reads its address off.
    ///
    /// `None` when this fire staged no live words, and then nothing is bound
    /// and every launch takes the null seat — the mask's rule, for the mask's
    /// reason. The CARVE is unconditional (see [`Inputs::reserve`]) because a
    /// store's layout may not depend on a policy word; the COPY is not,
    /// because an H2D a fire does not need is bytes it should not pay.
    pub live_rows: Option<u64>,
    /// One entry per kv geometry space, in space order.
    pub spaces: Vec<SpaceHandles>,
    /// `i32`, `[lanes]`: which recurrent bank each lane owns.
    pub slot_ids: Tensor,
    /// The padding mask the kv writers read.
    pub row_valid: Tensor,
    /// `RuntimeInput::AdapterRoutes`: `i32`, one adapter id per token row.
    /// `None` when no lane of this fire carried one — the shell then binds no
    /// seat, exactly as it does for the mask, and the correction's window is
    /// empty so nothing reads it.
    pub adapter_routes: Option<Tensor>,
    /// `RuntimeInput::Mask`: the packed `u8` (query, key) bits, fire-wide.
    /// `None` when no lane of this fire carried a mask — the shell then binds
    /// no mask seat at all, so a masked consumer answers `attn::masked`'s own
    /// refusal instead of reading a rectangle of zeros, which is every
    /// position masked OUT.
    pub mask: Option<Tensor>,
    /// `i32`, `[lanes + 1]`: each lane's ABSOLUTE byte offset into
    /// [`mask`](Handles::mask). Absolute, so a windowed consumer takes a
    /// slice of this table and the whole slab.
    pub mask_indptr: Option<Tensor>,
}

/// How many numbers a multimodal position is: `(t, h, w)`. The one place this
/// shell states it, on both axes — `RuntimeInput::PatchPositions` and
/// `RuntimeInput::MropePositions` are the same triple over two rectangles.
const AXES: u64 = 3;

/// **WHAT THE SECOND ROW AXIS RESERVES** (multimodal §5.4), or nothing at
/// all for a load whose plan states no patch row.
///
/// The three numbers are the ladder's and the plan's together: the ceilings
/// come from `PatchLadder` (a deployment statute) and the row width from the
/// plan's own `RuntimeInput::Patches` declaration, because `C·T·P²` is a
/// property of the resize policy a model text bakes against and not of the
/// deployment.
#[derive(Debug, Clone, Copy)]
pub struct PatchSeat {
    /// `PatchLadder::max_patches` — the most patch rows one fire may carry.
    pub rows: u64,
    /// One patch row's width in bytes: `C·T·P²` elements of the plan's
    /// activation dtype.
    pub row_bytes: u64,
    /// `PatchLadder::max_images` — the most images one fire may carry.
    pub images: u64,
    /// The element the patch rows are written in — the plan's own.
    pub dtype: Dtype,
    /// **HOW WIDE THE POSITION GATHER IS** (multimodal §9.2): the `taps` of
    /// the plan's `RuntimeInput::PatchEmbedRows` declaration — 1 on the native
    /// grid, 4 for bilinear, 16 for bicubic — or 0 for a plan that declares no
    /// position gather at all, which reserves nothing.
    pub embed_taps: u64,
    /// Whether the plan also declares `RuntimeInput::PatchEmbedWeights`. A
    /// native-grid tower does not, and then the weight stream costs it not one
    /// byte: the cheap path is the ABSENCE of the stream rather than a vector
    /// of ones.
    pub embed_weights: bool,
}

/// Where the patch vectors sit in the device store.
#[derive(Debug, Clone, Copy)]
struct PatchAt {
    payload: u64,
    segments: u64,
    routes: u64,
    positions: u64,
    embed_rows: u64,
    embed_weights: u64,
    seat: PatchSeat,
}

/// The three seats one fire's images resolve to.
#[derive(Debug, Clone, Copy)]
pub struct PatchHandles {
    /// `RuntimeInput::Patches`: `[patch rows, C·T·P²]`, the plan's element.
    pub patches: Tensor,
    /// `RuntimeInput::PatchSegments`: `i32`, `[images + 1]`.
    pub segments: Tensor,
    /// `RuntimeInput::PatchRoutes`: `i32`, one destination token row per
    /// patch row.
    pub routes: Tensor,
    /// `RuntimeInput::PatchPositions`: `i32`, `[patch rows, 3]` — one
    /// `(t, h, w)` per patch row, the grid coordinates the tower rotates by.
    pub positions: Tensor,
    /// `RuntimeInput::PatchEmbedRows`: `i32`, `[patch rows, taps]` — which
    /// rows of the learned position table each patch reads. `None` for a plan
    /// that declares no position gather.
    pub embed_rows: Option<Tensor>,
    /// `RuntimeInput::PatchEmbedWeights`: `f32`, `[patch rows, taps]`. `None`
    /// for a native-grid plan, which reads one table row per patch and weights
    /// it by nothing.
    pub embed_weights: Option<Tensor>,
}

/// One kv space's device seats.
#[derive(Debug, Clone, Copy)]
pub struct SpaceHandles {
    /// `GeomKind::Indptr`.
    pub indptr: Tensor,
    /// `GeomKind::Indices`.
    pub indices: Tensor,
    /// `GeomKind::LastPageLen`.
    pub last_page_len: Tensor,
    /// `GeomKind::KvLen`.
    pub kv_len: Tensor,
    /// `GeomKind::WritePage`.
    pub write_page: Tensor,
    /// `GeomKind::WriteOffset`.
    pub write_offset: Tensor,
}

/// What one fire wants written, host side.
#[derive(Debug, Clone)]
pub struct Fire<'a> {
    /// Token ids, in fire row order.
    pub tokens: &'a [i32],
    /// Absolute positions, in fire row order.
    pub positions: &'a [i32],
    /// Every window's rebased boundaries, each at its SLOT's offset in the
    /// blob's fixed-width carve
    /// ([`Windows::packed`](crate::window::Windows::packed), and
    /// [`Slots`](crate::window::Slots) for why the tails are padding). The
    /// slice stops after the last written word, so the trailing padding is
    /// device memory the carve holds and not H2D bytes.
    pub windows: &'a [i32],
    /// **THE FIRE'S QO BOUNDARIES, ABSOLUTE AND WHOLE** — `[lanes + 1]`,
    /// entry 0 zero and entry `l` the fire row lane `l` begins at, which is
    /// `model_exec::store::kv::indptr` un-rebased.
    ///
    /// Not a duplicate of [`windows`](Fire::windows) but the other reading of
    /// it: that blob holds one REBASED vector per window, for a consumer whose
    /// pointer is the window's; this one is for a consumer whose pointer is
    /// the PLANE's ([`Run::qo_indptr_absolute`](crate::run::Run)).
    ///
    /// EMPTY for a fire that stages none — the off switch `live` describes
    /// below, in the same words and with the same consequences: no host bytes,
    /// no copy, no handle, and `Windows::qo_absolute` answers `None`.
    ///
    /// **AND `lanes` IS THE BUCKET's LANE CEILING WHERE THE FIRE MEANS IT TO
    /// BE** (`serve::prepare` step 4d): entries past the live lanes repeat the
    /// last bound, which spells a lane of zero rows, so a plan carved past
    /// this fire's lanes finds a defined and monotone bound at every lane it
    /// can name. `Inputs::reserve` carved `max_lanes + 1`, which is the
    /// refusal below.
    pub qo_absolute: &'a [i32],
    /// **THE LIVE-ROWS SEAT'S WORDS**, one per (region, run) at
    /// [`Windows::max_runs`](crate::window::Windows::max_runs) stride
    /// ([`Windows::live`](crate::window::Windows::live)) — or EMPTY for a fire
    /// that stages none, which is every fire this shell serves today.
    ///
    /// Empty is not a degenerate case, it is the off switch: no bytes are
    /// written into the slot, no copy is issued in [`Inputs::commit`], no seat
    /// is bound, and `Ctx::stage` stays the null pointer every launch has
    /// always been handed. A fire whose windows are all identity would stage a
    /// vector of full row counts and change nothing either; what turns the
    /// seat on is a caller that means to serve fewer rows than the graph was
    /// carved at, and there is none yet.
    pub live: &'a [u32],
    /// Which recurrent bank each lane owns, in fire lane order.
    pub slot_ids: &'a [i32],
    /// Which adapter each token ROW routes to, in fire row order, or `None`
    /// when no lane carried one. Per ROW and not per lane, because that is
    /// what the correction kernel indexes with: `routes[row]` beside
    /// `x[row]`, the same shape `tokens` and `positions` have.
    ///
    /// **AND IT REACHES AS FAR AS [`tokens`](Fire::tokens) DOES** (the
    /// grid-at-ceiling wave, `serve::prepare` step 4c-b). `x` is an arena
    /// rectangle carved at the bucket for a bodied fire, and
    /// `linear.lora_correct` DECLARES that rectangle rather than merely
    /// addressing it — `routes.rows == x.rows` is its door — so a vector cut
    /// at the live rows is a refusal and not a stale read. The tail is `-1`,
    /// the base model, which is the same thing an unrouted lane's rows carry.
    pub adapter_routes: Option<&'a [i32]>,
    /// One geometry per kv space, in space order.
    ///
    /// **THE LANE TABLES MAY RUN PAST THE FIRE'S OWN LANES**, and every space
    /// of one fire must run equally far. A bodied fire pads its page CSR flat
    /// and its two per-lane vectors to zero out to the key's LADDER REACH
    /// (`Geometry::pad_to`, `serve::prepare` step 4d), so the lanes a ceiling
    /// plan reads past the live ones are genuinely empty rather than the last
    /// fire's leavings; `Staged::space_lanes` is what that came to and what
    /// [`Inputs::commit`] cuts the copies at. The row tables and the page-id
    /// list are untouched by it — an empty lane brings no row and owns no
    /// page.
    pub spaces: &'a [Geometry],
    /// This fire's expanded lane masks, or `None` when no lane carried one.
    pub mask: Option<&'a crate::mask::Staged>,
    /// **HOW MANY OF [`tokens`](Fire::tokens) ARE THIS FIRE'S OWN** — the rest
    /// are the carve's padding — or `0` for "all of them", which is every fire
    /// off the bodies path.
    ///
    /// **THE GRID-AT-CEILING WAVE IS WHAT SPLIT THE TWO NUMBERS.** A bodied
    /// fire's row vectors reach the BUCKET (`serve::prepare` step 4c-b),
    /// because the entries that declare a rectangle rather than merely
    /// addressing one — `layout.embed`, `elemwise.rope`, `elemwise.rope_mrope`
    /// — have to be handed the rectangle their launch is gridded over
    /// (`Run::carve_rows`). So the vector's LENGTH stopped being the fire's
    /// row count, and one consumer needs the difference: `row_valid`, the mask
    /// the kv writers read to know which of a bucket's rows are real.
    ///
    /// **AND THAT MASK IS THE ONLY THING STOPPING A SEAT-LESS WRITER.** The
    /// seated ones retire a padded block off `win[0]`; `attention.mla_kv_append`,
    /// `attention.index_kv_append` and the `attention.pool_*` writers take no
    /// seat at all and test this and nothing else. A stale `1` in the tail is a
    /// token appended into whatever page a stale write descriptor names, which
    /// is the one failure on this path that would be silent — so the mask is
    /// written ones to here and zeros beyond, and copied at the full length.
    ///
    /// A WINDOWED region needs no more than that: its blocks are retired at
    /// `win[0]` before `win[1]` is added, so the highest row any of them can
    /// name is `row_offset + live rows`, which is inside the fire. Only the
    /// whole-fire arm reaches for the bucket.
    pub live_rows: u32,
}

/// **The free set of a staging ring, as ONE word.**
///
/// A claim is a compare-exchange on the lowest set bit and a release is a
/// `fetch_or`, which is what makes the release legal where it has to happen:
/// on the CUDA driver's host-function thread, where a mutex is a hazard and a
/// CUDA call is forbidden. `engine::runahead::Runahead::MAX_FRAMES` is chosen
/// so that every admissible depth fits here — the bound is this word's, and it
/// says so.
#[derive(Debug)]
pub struct Free {
    /// Bit `i` set means slot `i` is claimable.
    bits: AtomicU64,
    depth: u32,
}

impl Free {
    /// A set of `depth` claimable slots.
    ///
    /// Shared with [`crate::settle::Settlement`], which pools one event per
    /// in-flight step and recycles it from the same callback thread for the
    /// same reason.
    #[must_use]
    pub fn of(depth: usize) -> Arc<Free> {
        debug_assert!(depth <= 64, "the free set is one word");
        let bits = if depth >= 64 {
            u64::MAX
        } else {
            (1u64 << depth) - 1
        };
        Arc::new(Free {
            bits: AtomicU64::new(bits),
            depth: depth as u32,
        })
    }

    /// Take the lowest free slot, or `None` when every one is in flight.
    #[must_use]
    pub fn take(&self) -> Option<u32> {
        let mut seen = self.bits.load(Ordering::Acquire);
        loop {
            if seen == 0 {
                return None;
            }
            let at = seen.trailing_zeros();
            match self.bits.compare_exchange_weak(
                seen,
                seen & !(1u64 << at),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(at),
                Err(now) => seen = now,
            }
        }
    }

    /// Give one back. **Called from the driver's callback thread.**
    pub fn give(&self, at: u32) {
        self.bits.fetch_or(1u64 << at, Ordering::Release);
    }

    /// How many slots are claimed right now — the steps the device may still
    /// be reading staging for.
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.depth - self.bits.load(Ordering::Acquire).count_ones()
    }
}

/// **One claimed staging slot, and the claim IS the value.**
///
/// Held from [`Inputs::claim`] in `prepare` to the instant the step's
/// settlement callback drops it — which is the `+ 1` in
/// `Runahead::staging_depth`'s formula, in a type. A `Prepared` that is
/// dropped without ever being enqueued (a frame whose third step refused)
/// releases its slot the same way, which is what makes the abort path free
/// rather than a leak.
#[derive(Debug)]
pub struct SlotGuard {
    free: Arc<Free>,
    at: u32,
}

impl SlotGuard {
    /// Which slot this is.
    #[must_use]
    pub fn at(&self) -> u32 {
        self.at
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.free.give(self.at);
    }
}

/// **What one fire's host staging came to** — the lengths the copies and the
/// handles are cut at.
///
/// Written by [`Inputs::write_host`] on the host with no stream in reach, read
/// by [`Inputs::commit`] on the stream. Everything here is a COUNT: the bytes
/// themselves are in the claimed slot's pinned memory, at the device carve's
/// own offsets, which is what lets the commit be a copy per region and nothing
/// else.
#[derive(Debug, Clone)]
pub struct Staged {
    /// **HOW MANY ROWS WERE WRITTEN**, which is this fire's own on every path
    /// but the bodies one and the BUCKET its launches are gridded at on that
    /// one (`serve::prepare` step 4c-b).
    ///
    /// [`space_lanes`](Staged::space_lanes)'s twin on the row axis, for its
    /// reason: the copies below are cut at what was WRITTEN, because the whole
    /// point of writing a tail is that the device must not keep the last
    /// fire's bytes there.
    rows: u32,
    lanes: u32,
    windows: usize,
    /// How many absolute qo bounds staged, or `None` for a fire that staged
    /// none — which is every fire the shell did not route to a body.
    qo_absolute: Option<usize>,
    /// How many live-rows words staged, or `None` for a fire that staged
    /// none — which is every fire until a caller fills [`Fire::live`].
    live: Option<usize>,
    /// How many rows the adapter axis staged, or `None` for a fire no lane
    /// routed.
    adapter_rows: Option<u32>,
    /// How many mask bytes staged, or `None` for a fire no lane masked.
    mask_bytes: Option<u32>,
    /// Per kv space, how many page ids its `indices` vector carries.
    space_indices: Vec<u32>,
    /// **HOW MANY LANES THE PER-SPACE LANE TABLES WERE STAGED AT** — the page
    /// CSR's `[n + 1]` bounds, the last-page lengths and the kv lengths.
    ///
    /// [`lanes`](Staged::lanes) for every fire but a bodied one, and the
    /// BUCKET's lane ceiling for that one (`serve::prepare` step 4d): a fire
    /// whose plans will be carved past its own lanes stages genuinely empty
    /// lanes out to that ceiling, so the copy has to be cut at what was
    /// written and not at what the fire brought. Never below `lanes`, so the
    /// live prefix is copied either way.
    space_lanes: u32,
}

/// The resident inputs, carved once.
#[derive(Debug)]
pub struct Inputs {
    store: Buffer,
    /// **The host staging ring** — one pinned mirror of the store's staged
    /// prefix per in-flight step. See [`Inputs::reserve`] for why the device
    /// side has no ring and this one does.
    staging: Vec<Pinned>,
    /// Which of them are claimable. Shared with every live [`SlotGuard`],
    /// because the release runs on the driver's callback thread.
    free: Arc<Free>,
    /// How many bytes of [`Inputs::store`] a fire stages — the prefix one
    /// slot mirrors, everything before the plan grants.
    stage_bytes: u64,
    tokens: u64,
    positions: u64,
    windows: u64,
    window_ints: u64,
    /// **THE CARVE ABOVE, AS THE LAYOUT READS IT.** `window_ints` is how many
    /// `i32` the window blob holds and this is HOW THEY ARE DIVIDED — one
    /// fixed-width slot per distinct window. The two are one arithmetic on
    /// purpose ([`crate::window::Slots`]): the shell hands this to
    /// `Windows::of`, and `Windows::packed` then lays a slot exactly where
    /// this reserve put one.
    window_slots: crate::window::Slots,
    /// **THE FIRE-WIDE QO VECTOR AND ITS CEILING** — `lanes + 1` `i32`, the
    /// boundaries above read absolutely rather than per window. Carved by
    /// every load and written by a bodied fire only; see [`Inputs::reserve`].
    qo_absolute: u64,
    qo_absolute_ints: u64,
    /// The live-geometry seat and how many `u32` it holds: `regions *
    /// max_runs * 4`, the artifact's bound on both axes and a `[rows,
    /// row_offset, lanes, lane_offset]` quad per seat (`window.rs` argues the
    /// order). See [`Inputs::reserve`] for why it is carved by every load.
    live_rows: u64,
    live_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    adapter_routes: u64,
    mask_bits: u64,
    mask_bytes: u64,
    mask_indptr: u64,
    /// **THE SECOND ROW AXIS'S THREE REGIONS, CARVED PAST THE STAGED
    /// PREFIX** (multimodal §5.4) — device bytes and no pinned mirror.
    ///
    /// The ring is depth-multiplied, so a patch rectangle inside it would be
    /// paid `staging_depth` times by every load, including the text-only ones
    /// that never fill it. These sit BELOW `stage_bytes`, exactly where the
    /// schedule grants sit and for exactly the schedule grants' reason: their
    /// source is a `Vec` one `enqueue` makes and drops (pageable, so
    /// `cudaMemcpyAsync` is synchronous in the source) and their destination
    /// is read by kernels that same `enqueue` launched behind them on the
    /// same stream. `None` for a plan that states no patch row, which is
    /// where the whole cost of the axis goes to zero.
    patch: Option<PatchAt>,
    /// **WHERE THE TRUNK'S TRIPLE-WIDE POSITION STREAM SITS**
    /// (`RuntimeInput::MropePositions`, multimodal §6.3), or `None` for the
    /// texts that rotate by a scalar — which is every text this engine served
    /// before the towers.
    ///
    /// Below the staged prefix and on no ring, for [`Inputs::patch`]'s reason
    /// exactly: the source is a `Vec` one `enqueue` makes and drops, and the
    /// destination is read by the rotation that same `enqueue` launched behind
    /// it on the same stream.
    mrope: Option<u64>,
    /// How many bytes [`Inputs::mrope`] holds — the row ceiling tripled, or
    /// zero.
    mrope_bytes: u64,
    spaces: Vec<SpaceAt>,
    /// **THE LANE CEILING EVERY PER-LANE TABLE ABOVE WAS CARVED AT**
    /// (`Budget::max_lanes`), kept as a number rather than re-derived.
    ///
    /// It was implicit while a fire staged exactly its own lanes, which
    /// `fire::compose` had already bounded. A bodied fire now stages lane
    /// tables PAST its own lanes (`Fire::spaces`, `serve::prepare` step 4d),
    /// so [`Inputs::write_host`] has a second thing to refuse: a padded
    /// vector past this ceiling would run off its region and into the one
    /// carved behind it.
    max_lanes: u32,
    /// **THE ROW CEILING EVERY ROW-SHAPED TABLE ABOVE WAS CARVED AT**
    /// (`Budget::max_tokens`), kept beside [`max_lanes`](Inputs::max_lanes)
    /// for exactly its reason and since exactly the same seam.
    ///
    /// It was implicit while a fire staged exactly its own rows, which
    /// `fire::compose` had already bounded. The grid-at-ceiling wave has a
    /// bodied fire stage its token, position and rotation vectors out to the
    /// BUCKET its launches are gridded at ([`Fire::live_rows`]'s note), so a
    /// padded vector past this ceiling would run off its region and into the
    /// one carved behind it — and a table past its carve is a
    /// `Fault::Ceiling`, not a smear.
    max_rows: u32,
    /// One grant per (RUN, PLAN VALUE), flat at `run * plan_values + value`.
    /// Grants are disjoint carvings of the shell's bounded pool because every
    /// schedule a fire builds is staged at once and read at once; what changed
    /// in C1b is the KEY — a family that carves two readings out of one
    /// page-id space (gemma's sliding beside its global, gpt-oss's windowed
    /// beside its full) mints two plan values, and two schedules sharing one
    /// grant would overwrite each other's staged image between the prepare
    /// pass and the launch.
    ///
    /// **THE RUN IS THE SAME ARGUMENT ONE STEP FURTHER IN** (P4's fallback,
    /// design §3). A region whose class set P4 could not seat carves one
    /// schedule per interval of it, all of them staged in the prepare pass and
    /// all of them read in the capture pass, so they are as simultaneous as
    /// two readings of one space are — and they need disjoint INT grants for
    /// exactly the same reason.
    ///
    /// **AND THEY NEED NO RING, WHICH IS A FINDING RATHER THAN AN OMISSION**
    /// (alto F2b). The descriptor staging got one because its SOURCE is host
    /// memory the next fire overwrites; a schedule's grant is different in
    /// both halves. Its source is a `Vec` the builder makes and drops inside
    /// one `enqueue` (`kernels_cuda::attn::plan::upload` copies from PAGEABLE
    /// memory, which makes `cudaMemcpyAsync` synchronous in the source), and
    /// its destination is read by kernels this same `enqueue` launched behind
    /// it on the SAME compute stream. So two frames in flight cannot collide
    /// here: frame W+1's `upload` is enqueued after frame W's attention
    /// kernels, and stream order is the whole of the exclusion — the same
    /// argument the device side of the descriptor region rests on.
    ///
    /// dev carried a 13-slot plan-staging ring (`attention_workspace.hpp:66`)
    /// because its builders ran in a PREPARE phase that was genuinely
    /// concurrent with the previous wave's execution. Here they run inside the
    /// walk, on the stream, which is what makes the ring unnecessary — and
    /// what would make it necessary again is moving the build into `prepare`,
    /// which is where this note should be read.
    ///
    /// The FLOAT side is shared across the runs of one plan value, and that is
    /// not an oversight. The int side holds the schedule's staged image, which
    /// must survive from the build to the launch; the float side is the
    /// split-kv partials, which a launch writes and reads inside itself. The
    /// runs of one region are sequential on that region's stream — the same
    /// sequencing that already lets sixty layers share one plan value's
    /// partials — so sharing costs nothing and multiplying it would cost the
    /// 149 MiB a wide reading asks for, per interval.
    plans: Vec<Option<Workspace>>,
    /// How many plan values one run's slice of [`plans`](Inputs::plans) holds.
    plan_values: usize,
}

impl Inputs {
    /// Reserve the vectors a deployment's ceilings admit.
    ///
    /// # `runahead` is the depth this staging is carved for
    ///
    /// **THE DEVICE SIDE IS ONE REGION AND THE HOST SIDE IS A RING**, and the
    /// asymmetry is not an oversight — it is article 7. Every address a
    /// captured graph reads was fixed at bake, so the DEVICE carve below stays
    /// exactly one pointer-stable region however deep the run-ahead goes; what
    /// F2b added is `runahead.staging_depth()` slots of PINNED HOST memory,
    /// each laid out at the device carve's own offsets, so that
    /// `write_host` (in `prepare`) and `commit` (in `enqueue`) can be two
    /// phases instead of one call.
    ///
    /// What the ring buys is what F1 refused: the host may write frame W+1's
    /// vectors while the copies frame W's launches read are still in flight,
    /// because they are different bytes. Stream order does the rest — W+1's
    /// H2D is enqueued behind W's kernels on the one compute stream — so the
    /// DEVICE region needs no depth at all.
    ///
    /// The number crosses once (article 8: one owner, one spelling —
    /// [`engine::runahead::Runahead`], the reborn `UPLOAD_STAGING_DEPTH`) and
    /// this is the consumer that spends it.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::Fault::Device) for the device allocation or
    /// any of the ring's pinned ones.
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        budget: &Budget,
        paging: Paging,
        spaces: usize,
        facts: &Facts,
        classes: usize,
        regions: usize,
        runs: u32,
        gathered: usize,
        // The device facts the plan builders take. `num_sm` is what the grants
        // below have always been sized off; `cc_major` beside it is what lets
        // the prefill grant ask the PLANNER for its tile rather than bound it
        // by hand (`prefill_float_bytes`).
        device: Device,
        runahead: engine::runahead::Runahead,
        patch: Option<PatchSeat>,
        mrope: bool,
    ) -> Result<Inputs> {
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(budget.max_lanes);
        let pages = u64::from(budget.max_lanes) * u64::from(paging.pages_per_slot);
        // A window is one contiguous run of the fire's class order, so a plan
        // of `k` classes has at most `k(k+1)/2` of them — plus one for the
        // zero window every empty region shares. Reserved rather than
        // measured, because these addresses are recorded into a graph that is
        // never re-captured (the note at the top of this file).
        //
        // **AND A WINDOW CARRIES MORE THAN BOUNDARIES.** Two menu entries add
        // to the `lanes + 1` per window slot, and both are paid whether or not
        // the shell serves them — making the STORE's layout depend on a policy
        // word would make an address depend on it, and addresses go into
        // graphs:
        //
        // - `2 * runs` for the SEGMENT LIST a `Fallback::Grouped` window
        //   carries (`Windows::packed` stages both in the one copy). `runs` is
        //   `model_exec::fire::max_runs`, the artifact's own bound on how many
        //   intervals any mask breaks into, so the ceiling holds for every
        //   fire this load can be handed; an artifact P4 seated whole answers
        //   `1` and pays two ints per slot for a list it never fills.
        // - `Slots::gathered_at`'s stride for what a GATHERED window carries
        //   beside them (`window::Gathered`, `Fallback::Copy`): the row map
        //   the gather and the scatter read, and per kv space the page bounds,
        //   the compacted page-id list and the two per-lane vectors.
        //   `gathered` is `model_exec::fire::fragmentable` — how many distinct
        //   masks this artifact can ever find in pieces, which is 0 for every
        //   artifact P4 seated whole and 1 for today's qwen texts.
        //
        // **AND THE SLOTS ARE FIXED-WIDTH, WHICH IS WHAT THE CARVE ALWAYS
        // SAID AND WHAT THE LAYOUT NOW ALSO SAYS.** This product has always
        // been "slot count x the ceiling one slot can need"; `Windows::packed`
        // used to pack the fire's actual vectors tightly inside it, so a
        // window's device address depended on the LANE COUNTS in front of it
        // and moved between fires of one `record::BodyKey` — under a graph
        // that had baked it (`window.rs`'s header carries the whole argument).
        // It now lays slot `i` at `i * stride`. The padding is bytes this
        // expression was already buying; the arithmetic is shared rather than
        // restated, so neither half can drift.
        //
        // **AND THE GATHERED CEILING IS THAT SAME OBJECT'S NOW TOO.** This
        // function used to spell `rows + spaces * (3 * lanes + 1 + pages)`
        // itself and hand the product over as a total; the tail behind the
        // slots is ADDRESSED at that stride since this wave
        // (`Slots::gathered_at`), so the expression moved to the type that
        // owns the addresses and what is left here are the ceilings it is
        // computed FROM. One expression, one owner, and a carve that cannot
        // become a layout the blob disagrees with.
        //
        // A gathered window is a slot BEYOND the `k(k+1)/2 + 1` runs — `seat`
        // deliberately does not dedupe a gathered window against a plain one
        // of the same span (they mean two different things by that rectangle)
        // — so it takes a slot of its own for its boundary vector, and its
        // payload rides behind every slot at `Slots::gathered_at`.
        let window_slots =
            crate::window::Slots::new(classes, lanes, runs, gathered, rows, spaces, pages);
        let window_ints = window_slots.words();

        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(ALIGN);
            here
        };
        let tokens = take(rows * 4);
        let positions = take(rows * 4);
        let windows = take(window_ints * 4);
        // **THE SAME BOUNDARIES, THE OTHER READING** (bodies design, chunk
        // 2c-a): `[lanes + 1]` `i32` holding the FIRE's qo prefix sums with
        // nothing subtracted, so a windowed consumer whose pointer is the
        // PLANE's base can take THIS vector, whole, instead of the rebased
        // one its slot above holds. One vector per fire and not one per
        // window, because there is nothing per-window about it — that is the
        // whole difference, and it is also what makes the address safe for a
        // body to bake.
        //
        // UNCONDITIONAL, for the reason every carve here is: a store whose
        // layout depended on a policy word would make an address depend on
        // it, and these addresses are recorded into graphs. Four KiB at a
        // thousand lanes; the WRITING is what a fire chooses
        // (`Fire::qo_absolute`), and only a bodied one does.
        let qo_absolute_ints = lanes + 1;
        let qo_absolute = take(qo_absolute_ints * 4);
        // **THE LIVE-ROWS SEAT** (bodies design; the census's option (c)): one
        // `u32` per (region, run), at a FIXED stride, holding how many rows of
        // that launch are the fire's own. `kernels_cuda::Ctx::arm_stage` is
        // handed one of these addresses per region and the entries that
        // support the seat pass it as their `win` argument, so what a REPLAY
        // of a graph carved at a bucket serves is read from memory rather than
        // baked into a node parameter.
        //
        // **FIXED STRIDE, AND CARVED BY EVERY LOAD.** `regions * max_runs` is
        // the artifact's bound on both axes — the template's length and
        // `model_exec::fire::max_runs` — so the seat's arithmetic is a
        // multiplication and not a lookup, and a fire whose regions split
        // fewer ways than the artifact's bound leaves the tail of each row
        // unwritten rather than moving anybody's address. The carve is
        // UNCONDITIONAL for the reason the adapter routes below are: a store
        // whose LAYOUT depended on a policy word would make an address depend
        // on it, and these addresses are recorded into graphs. Eight KiB at a
        // thousand regions P4 seated whole, four words a seat; the WRITING is
        // what a fire chooses (`Fire::live`), and a bodied one does.
        // FOUR words per (region, run) — `[rows, row_offset, lanes,
        // lane_offset]`, and the order is `window.rs`'s contract, argued where
        // the words are filled. The row pair is first because every guard
        // shipped before the lane pair existed reads `win[0]` and every shift
        // `win[1]`; the lane pair is what a request-gridded kernel reads.
        //
        // **AND THE WORD COUNT IS THE SEAT'S OWN ARITHMETIC**
        // (`window::Seat`), for the reason `window::Slots` is one object here:
        // this carve's bytes and `Windows::live_at`'s addresses have to be the
        // same multiplication or an address is a launch reading another
        // region's geometry. Two INSTANCES rather than one shared object —
        // this is the LOAD's rectangle and a fire's is its own, never wider
        // (`window::Seat`'s header states the ceiling) — because a reserve
        // cannot measure a fire it has not seen.
        let live_seat = crate::window::Seat::new(regions as u64, u64::from(runs.max(1)));
        let live_ints = live_seat.words();
        let live_rows = take(live_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * 4);
        // The adapter axis's one vector, reserved at the row ceiling like
        // every other row-shaped table here — 32 KiB at `max_tokens = 8192`,
        // paid by every load whether or not the plan declares a correction,
        // because a conditional carve would make the STORE's layout depend on
        // the plan and the addresses in it are recorded into graphs.
        let adapter_routes = take(rows * 4);
        // The masked axis's two vectors. `context` is what a slot can hold,
        // so `rows * context` bounds every (query, key) cell a fire can
        // present, and the per-lane byte alignment costs one byte a lane.
        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_bytes = (rows * context).div_ceil(8) + lanes;
        let mask_bits = take(mask_bytes);
        let mask_indptr = take((lanes + 1) * 4);
        let spaces: Vec<SpaceAt> = (0..spaces)
            .map(|_| SpaceAt {
                indptr: take((lanes + 1) * 4),
                indices: take(pages * 4),
                last_page_len: take(lanes * 4),
                kv_len: take(lanes * 4),
                write_page: take(rows * 4),
                write_offset: take(rows * 4),
            })
            .collect();
        // **WHERE THE STAGED PREFIX ENDS.** Everything carved above this line
        // is written by a fire (`write_host` fills it, `commit` copies it);
        // everything below is granted to the schedule builders, which stage
        // their own images straight onto the stream inside the walk. So this
        // is the size one host ring slot mirrors — a `take(0)` because the
        // closure owns the cursor and this is how it is read without moving.
        let stage_bytes = take(0);
        // **THE PATCH REGIONS, BELOW THE LINE** — device bytes with no pinned
        // mirror, which is the whole of multimodal §5.4. Nothing above this
        // line moved, so a text-only load's carve is byte-for-byte the one it
        // always was and `PatchSeat`'s `None` costs it not even an offset.
        let patch = patch.map(|seat| PatchAt {
            payload: take(seat.rows * seat.row_bytes),
            segments: take((seat.images + 1) * 4),
            routes: take(seat.rows * 4),
            // The tower's own rotation stream, `[patch rows, 3]` i32 — cut
            // from the same submission, carved beside the three it arrives
            // with, and staged in the same `enqueue`.
            positions: take(seat.rows * AXES * 4),
            // **THE POSITION GATHER'S TWO STREAMS** (multimodal §9.2), sized
            // by the plan's own tap count and by nothing else: `0` taps is a
            // plan with no learned position table and carves nothing, and a
            // native-grid plan carves the ids at one tap and no weights at all.
            embed_rows: take(seat.rows * seat.embed_taps * 4),
            embed_weights: if seat.embed_weights {
                take(seat.rows * seat.embed_taps * 4)
            } else {
                0
            },
            seat,
        });
        // **THE TRUNK'S TRIPLE-WIDE TOKEN STREAM** (multimodal §6.3), below
        // the line for the patch regions' reason and reserved only when the
        // plan names it. `[max_tokens, 3]` i32 is 96 KiB at the row ceiling
        // this shell serves — paid once, not `staging_depth` times, and not
        // at all by a text that rotates by a scalar.
        let mrope = mrope.then(|| take(rows * AXES * 4));
        // One grant per plan value. The float side is the requirement of the
        // builder that will actually run, or the flat floor, whichever is
        // larger — computed rather than guessed, because a short grant here
        // DECLINES to capture instead of failing (build log 13). The KIND is
        // what picks the formula: the paged builders pad their work items out
        // to the SM count and stage a partial per (query head, padded item,
        // tile row), the latent one stages a partial per cluster row and knows
        // nothing about query heads at all.
        //
        // ONE INT GRANT PER RUN, ONE FLOAT GRANT PER VALUE — see
        // [`Inputs::plans`] for which half of a schedule needs which, and why
        // multiplying the float side would be the expensive way to be wrong
        // about it. `runs` is `model_exec::fire::max_runs`, which is `1` for every
        // artifact P4 seated whole, and the carve below is then byte-for-byte
        // the one this shell has always made.
        let runs = runs.max(1);
        let grants: Vec<Option<Grant>> = facts
            .plans
            .iter()
            .map(|seat| {
                seat.map(|seat| {
                    let floats = match seat.kind {
                        // **THE OCCUPANCY TERM AND THE CEILING TERM, AND THE
                        // LARGER WINS.** The first is what this shell has
                        // always granted; the second is the padding the bucket
                        // ceiling carves, which the first cannot see
                        // (`prefill_float_bytes` carries the argument). The
                        // ROW half comes off the lattice's top rung because
                        // that is the widest bucket a fire can land in — a
                        // fire above it is refused at compose
                        // (`Fault::NoBucket`) — and the LANE half off
                        // `max_lanes`, which bounds `Run::planning`'s lane
                        // ceiling from above.
                        StructKind::AttnPrefillPlan => graph_float_bytes(&seat.reading, device.num_sm)
                            .max(prefill_float_bytes(
                                &seat.reading,
                                budget.buckets.last().copied().unwrap_or(budget.max_tokens),
                                budget.max_lanes,
                                &device,
                            )),
                        // The sm90 builder carves at the ceiling too since
                        // chunk 5, and its FLOAT grant is still the one it
                        // had, because `sched_sm90` allocates no floats at all
                        // (`Built::float_bytes` is a literal zero there). What
                        // the ceiling does move is its INT ask —
                        // `4 * max_total_num_works`, whose row and lane inputs
                        // are now the bucket's — and that rides the flat
                        // `GRANT_INT_BYTES` beside every other schedule's: at
                        // the lattice's top rung a 128-row tile over a
                        // hundred-odd heads is low single-digit megabytes
                        // against eight, and an ask that outgrew it would be
                        // the allocator's named refusal rather than a wrong
                        // schedule.
                        StructKind::AttnPrefillPlanSm90 => {
                            graph_float_bytes(&seat.reading, device.num_sm)
                        }
                        // The prefill bound, applied to decode: a decode
                        // schedule stages less than this, and asking the
                        // prefill number for it is what this shell has always
                        // done — pre-existing, and not tightened here. What
                        // IS new is the second term: a graph-shaped decode
                        // pads to the lane count where that outruns the grid,
                        // and `max_lanes` is a deployment's word rather than
                        // anything the prefill formula reads.
                        StructKind::AttnDecodePlan => graph_float_bytes(&seat.reading, device.num_sm)
                            .max(decode_float_bytes(&seat.reading, budget.max_lanes)),
                        StructKind::MlaPlan => {
                            latent_float_bytes(seat.reading.head_dim, device.num_sm)
                        }
                    }
                    .max(GRANT_FLOAT_BYTES);
                    Grant {
                        int_at: (0..runs).map(|_| take(GRANT_INT_BYTES)).collect(),
                        float_at: take(floats),
                        float_bytes: floats,
                    }
                })
            })
            .collect();
        let total = at;

        let store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;
        let base = store.ptr();
        // **THE RING, AND IT IS THE HOST HALF ONLY.** One pinned mirror of the
        // staged prefix per slot; `Runahead::staging_depth` says how many and
        // why (`frames × steps + 1`, the `+ 1` measured). Pinned rather than
        // pageable because that is the entire point: a pageable source makes
        // `cudaMemcpyAsync` synchronous in the source, which is what let one
        // buffer serve depth 1, and page-locking it is what lets the copy be
        // asynchronous and the slot's lifetime be the thing that bounds it.
        let depth = runahead.staging_depth();
        let slot_bytes = usize::try_from(stage_bytes).unwrap_or(usize::MAX);
        let mut staging = Vec::with_capacity(depth);
        for _ in 0..depth {
            staging.push(Pinned::mapped(slot_bytes)?);
        }
        Ok(Inputs {
            staging,
            free: Free::of(depth),
            stage_bytes,
            tokens,
            positions,
            windows,
            window_ints,
            window_slots,
            qo_absolute,
            qo_absolute_ints,
            live_rows,
            live_ints,
            row_valid,
            slot_ids,
            adapter_routes,
            mask_bits,
            mask_bytes,
            mask_indptr,
            patch,
            mrope,
            mrope_bytes: if mrope.is_some() { rows * AXES * 4 } else { 0 },
            spaces,
            max_lanes: budget.max_lanes,
            max_rows: budget.max_tokens,
            plan_values: grants.len(),
            plans: (0..runs as usize)
                .flat_map(|run| {
                    grants.iter().map(move |grant| {
                        grant.as_ref().map(|grant| Workspace {
                            int_ptr: base + grant.int_at[run],
                            int_bytes: GRANT_INT_BYTES as usize,
                            float_ptr: base + grant.float_at,
                            float_bytes: usize::try_from(grant.float_bytes).unwrap_or(usize::MAX),
                        })
                    })
                })
                .collect(),
            store,
        })
    }

    /// One plan value's builder grant, by value id, for one run of the region
    /// that builds it.
    ///
    /// A run past the ones this load reserved answers `None`, which the caller
    /// reads as "no grant" and refuses on — the same answer a value that is
    /// not a plan struct gives. It cannot happen to a load and a fire built
    /// from one artifact: `model_exec::fire::max_runs` bounds every fire's split
    /// and is what `reserve` was handed.
    #[must_use]
    pub fn grant(&self, plan: u32, run: u32) -> Option<Workspace> {
        let at = run as usize * self.plan_values + plan as usize;
        self.plans.get(at).copied().flatten()
    }

    /// Every byte the inputs hold.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }

    /// **Claim a staging slot.** The first half of the fire path's one
    /// resource acquisition, and it happens in `prepare` where a refusal is
    /// still free.
    ///
    /// Spins rather than parks: a slot is released by the settlement callback
    /// of a step the device is finishing right now, so the wait — when there
    /// is one at all — is microseconds, and a condvar here would put a mutex
    /// on the driver's callback thread to buy nothing. The ring is sized so
    /// that a caller obeying its own stated `frames_in_flight` never waits
    /// (that is the `+ 1`), so a spin that runs long is a caller running
    /// deeper than it declared, and the deadline below is what says so out
    /// loud instead of hanging.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::Fault::Ceiling) naming the ring, for a caller
    /// that has been holding every slot for [`CLAIM_DEADLINE`].
    pub fn claim(&self) -> Result<SlotGuard> {
        if let Some(at) = self.free.take() {
            return Ok(SlotGuard {
                free: Arc::clone(&self.free),
                at,
            });
        }
        let until = std::time::Instant::now() + CLAIM_DEADLINE;
        loop {
            std::hint::spin_loop();
            if let Some(at) = self.free.take() {
                return Ok(SlotGuard {
                    free: Arc::clone(&self.free),
                    at,
                });
            }
            if std::time::Instant::now() >= until {
                return Err(crate::error::Fault::Ceiling {
                    what: "staging slots (every one is still in flight; the caller \
                           is running deeper than the frames_in_flight it loaded with)",
                    need: self.staging.len() as u64 + 1,
                    have: self.staging.len() as u64,
                });
            }
        }
    }

    /// How many staging slots are claimed right now.
    ///
    /// The shell's own view of how far ahead of the device it is — read by the
    /// gates that assert saturation, and by nothing on the fire path.
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.free.in_flight()
    }

    /// **THE WINDOW BLOB'S CARVE, FOR THE TABLE THAT LAYS ITSELF OUT IN IT.**
    /// `Windows::of` takes this and `Windows::packed` places every slot from
    /// it, so the offsets a fire writes and the bytes this reserved are one
    /// arithmetic and not two that agree today.
    #[must_use]
    pub fn window_slots(&self) -> crate::window::Slots {
        self.window_slots
    }

    /// **Write one fire's vectors into a claimed slot — host only, no stream**
    /// (alto design §4: `staging.write(slot, ..)`).
    ///
    /// Every ceiling this staging enforces is enforced HERE, in `prepare`,
    /// where a refusal costs nothing: a fire past the reserved window table or
    /// mask slab is refused before a single byte crosses.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved
    /// ceilings.
    pub fn write_host(&self, slot: &SlotGuard, fire: &Fire<'_>) -> Result<Staged> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32;
        let host = &self.staging[slot.at() as usize];

        // **THE ROW CEILING, WHICH A FIRE ONLY NEEDED ONCE IT COULD STAGE MORE
        // ROWS THAN IT BROUGHT.** A bodied fire pads its token, position and
        // rotation vectors out to the bucket its launches are gridded at
        // (`serve::prepare` step 4c-b); what is refused here is a padding past
        // what `reserve` carved, which would write over the region behind it.
        // `fire::compose` bounds a fire's OWN rows and P0 bounds every bucket
        // by the same ceiling, so this is the belt to two braces — and it
        // still refuses a genuinely over-wide vector, padded or not.
        if rows > self.max_rows {
            return Err(crate::error::Fault::Ceiling {
                what: "staged token rows",
                need: u64::from(rows),
                have: u64::from(self.max_rows),
            });
        }

        if fire.windows.len() as u64 > self.window_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: self.window_ints,
            });
        }
        if fire.qo_absolute.len() as u64 > self.qo_absolute_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "absolute qo boundaries",
                need: fire.qo_absolute.len() as u64,
                have: self.qo_absolute_ints,
            });
        }
        if fire.live.len() as u64 > self.live_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "staged live-rows words",
                need: fire.live.len() as u64,
                have: self.live_ints,
            });
        }
        // **AND THE LANE TABLES' OWN CEILING**, which a fire only needed once
        // it could stage more lanes than it brought. A bodied fire pads its
        // page CSR and its two per-lane vectors out to the bucket's lane
        // ceiling (`serve::prepare` step 4d) so that the plans the next chunk
        // carves there read empty lanes rather than the last fire's bytes;
        // what is refused here is a padding past what `reserve` carved, which
        // would write over the region behind it. The caller clamps to this
        // same number, so this is the belt to that braces — and it still
        // refuses a genuinely over-wide vector, padded or not.
        //
        // The lane count is read off the SPACES rather than taken on the
        // caller's word, and every space of one fire must spell the same one:
        // they are built from one `seats` vector and padded by one loop, and
        // a fire that broke that would have `commit` copy one space's tail
        // out of another's unwritten bytes.
        let mut spelled: Option<usize> = None;
        for geometry in fire.spaces {
            let count = geometry.indptr.len().saturating_sub(1);
            if count != geometry.last_page_len.len()
                || count != geometry.kv_len.len()
                || spelled.is_some_and(|first| first != count)
            {
                return Err(crate::error::Fault::program(
                    "inputs::write_host",
                    format!(
                        "this fire's kv spaces state {count} lane(s) of page bounds, {} \
                         last-page length(s) and {} kv length(s), and every space of one fire \
                         is built over one lane vector",
                        geometry.last_page_len.len(),
                        geometry.kv_len.len()
                    ),
                ));
            }
            spelled = Some(count);
        }
        let space_lanes = spelled.map_or(lanes, |count| count as u32).max(lanes);
        if u64::from(space_lanes) > u64::from(self.max_lanes) {
            return Err(crate::error::Fault::Ceiling {
                what: "staged kv lanes",
                need: u64::from(space_lanes),
                have: u64::from(self.max_lanes),
            });
        }
        if let Some(staged) = fire.mask
            && staged.bits.len() as u64 > self.mask_bytes
        {
            return Err(crate::error::Fault::Ceiling {
                what: "mask bits",
                need: staged.bits.len() as u64,
                have: self.mask_bytes,
            });
        }

        // `put` is `Pinned::write` with the slot's own bounds as the refusal:
        // a region past the mirror is the same ceiling the device carve would
        // have refused, caught one phase earlier.
        let put = |offset: u64, bytes: &[u8], what: &'static str| -> Result<()> {
            if host.write(usize::try_from(offset).unwrap_or(usize::MAX), bytes) {
                return Ok(());
            }
            Err(crate::error::Fault::Ceiling {
                what,
                need: offset + bytes.len() as u64,
                have: self.stage_bytes,
            })
        };

        put(self.tokens, bytes_of(fire.tokens), "staged tokens")?;
        put(self.positions, bytes_of(fire.positions), "staged positions")?;
        put(self.windows, bytes_of(fire.windows), "staged window boundaries")?;
        // The padding mask is all-valid over the rows a fire CARRIES and zero
        // over the ones its carve added: every row up to `live` is a row the
        // fire means, and what follows tells the writers which of a bucket's
        // padded rows are real — which is why the buffer exists now rather
        // than at step 5. A fire that padded nothing states `live == rows` and
        // writes the all-ones mask this staging has always written
        // ([`Fire::live_rows`]).
        let live = if fire.live_rows == 0 {
            rows
        } else {
            fire.live_rows.min(rows)
        };
        let mut valid = vec![1u8; live as usize];
        valid.resize(rows as usize, 0);
        put(self.row_valid, &valid, "staged row_valid")?;
        put(self.slot_ids, bytes_of(fire.slot_ids), "staged slot ids")?;

        // THE ABSOLUTE QO VECTOR, STAGED OR NOT STAGED — the live seat's
        // rule below, applied to the reading beside it. An empty
        // `Fire::qo_absolute` writes nothing, copies nothing and publishes no
        // handle, so a fire on any path but the bodied one carries exactly the
        // one reading of its boundaries it always carried.
        let qo_absolute = if fire.qo_absolute.is_empty() {
            None
        } else {
            put(self.qo_absolute, bytes_of(fire.qo_absolute), "staged absolute qo bounds")?;
            Some(fire.qo_absolute.len())
        };

        // THE LIVE-ROWS SEAT, STAGED OR NOT STAGED — the mask's rule again,
        // and here it is what keeps the seat free. An empty `Fire::live`
        // writes nothing into the slot, so a fire that does not mean to serve
        // fewer rows than its graph was carved at pays the seat no host bytes,
        // no H2D and no launch argument; the device carve is there either way,
        // for `reserve`'s reason.
        let live = if fire.live.is_empty() {
            None
        } else {
            put(self.live_rows, u32_bytes_of(fire.live), "staged live rows")?;
            Some(fire.live.len())
        };

        // THE ADAPTER AXIS, STAGED OR NOT STAGED — the mask's rule, for the
        // mask's reason. A fire no lane routed writes nothing here and binds
        // no seat, so a correction that somehow reached a launch would hit
        // `Run::tensor`'s named panic rather than read a slab of zeros, which
        // is every row routed to adapter 0 of a bank nobody registered.
        let adapter_rows = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                put(self.adapter_routes, bytes_of(routes), "staged adapter routes")?;
                Some(routes.len() as u32)
            }
        };

        // THE MASKED AXIS, STAGED OR NOT STAGED. A fire no lane masked writes
        // nothing here and binds no seat, which is what makes
        // `attn::masked`'s "no mask span table rides this plan" refusal
        // reachable — the alternative, a zeroed slab, is every position
        // masked out and a row of `-inf`.
        let mask_bytes = match fire.mask {
            None => None,
            Some(staged) => {
                put(self.mask_bits, &staged.bits, "staged mask bits")?;
                put(self.mask_indptr, bytes_of(&staged.indptr), "staged mask indptr")?;
                Some(u32::try_from(staged.bits.len()).unwrap_or(u32::MAX))
            }
        };

        let mut space_indices = Vec::with_capacity(self.spaces.len());
        for (at, geometry) in self.spaces.iter().zip(fire.spaces) {
            put(at.indptr, bytes_of(&geometry.indptr), "staged kv indptr")?;
            put(at.indices, bytes_of(&geometry.indices), "staged kv page ids")?;
            put(at.last_page_len, bytes_of(&geometry.last_page_len), "staged last page len")?;
            put(at.kv_len, bytes_of(&geometry.kv_len), "staged kv len")?;
            put(at.write_page, bytes_of(&geometry.write_page), "staged write page")?;
            put(at.write_offset, bytes_of(&geometry.write_offset), "staged write offset")?;
            space_indices.push(geometry.indices.len() as u32);
        }

        Ok(Staged {
            rows,
            lanes,
            windows: fire.windows.len(),
            qo_absolute,
            live,
            adapter_rows,
            mask_bytes,
            space_indices,
            space_lanes,
        })
    }

    /// **Commit a written slot to the device region, async on `stream`**
    /// (alto design §4: `staging.commit(s, desc)`).
    ///
    /// **THE COPIES ARE ON THE STREAM.** A synchronous copy would be ordered
    /// against every stream in the process, which is both slower and a lie
    /// about what this fire depends on; an asynchronous one on the fire's own
    /// stream is exactly the dependency the launches behind it have.
    ///
    /// **AND THE DEVICE REGION NEEDS NO RING**, which is the half of the
    /// design worth restating at the call: two in-flight frames ride ONE
    /// compute stream, so frame W+1's copies here are enqueued behind frame
    /// W's launches and stream order is the whole of the mutual exclusion.
    /// What needed the ring was the SOURCE — see [`Inputs::reserve`].
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::Fault::Device) for a copy.
    /// **THE SECOND ROW AXIS'S H2D, INSIDE THE ENQUEUE AND OUTSIDE THE
    /// RING** (multimodal §5.4).
    ///
    /// Three `cudaMemcpyAsync`s from PAGEABLE host memory onto the fire's own
    /// compute stream, in front of the launches that read them. Pageable is
    /// what makes this safe without a slot: the driver copies the source into
    /// its own staging buffer before the call returns, so the caller's `Vec`
    /// may be dropped the instant this returns — which is the argument
    /// [`Inputs::plans`] already makes for the schedule grants, and the reason
    /// neither needs a pinned mirror or a depth.
    ///
    /// `None` for a load whose plan states no patch row, which is a refusal
    /// the caller should never reach: `compose_axes` answers `Fault::Towerless`
    /// for an image against a text with no patch axis, so a fire that gets
    /// here with patch bytes has an artifact that reserved room for them.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved
    /// patch rectangle, and [`Fault::Device`](crate::Fault::Device) for the
    /// copies.
    pub fn stage_patches(
        &mut self,
        stream: *mut core::ffi::c_void,
        payload: &[u8],
        segments: &[i32],
        routes: &[i32],
        positions: &[i32],
        embed_rows: &[i32],
        embed_weights: &[f32],
    ) -> Result<PatchHandles> {
        let Some(at) = self.patch else {
            return Err(crate::error::Fault::Ceiling {
                what: "the patch rectangle, which this load reserved none of",
                need: payload.len() as u64,
                have: 0,
            });
        };
        let owed = [
            (payload.len() as u64, at.seat.rows * at.seat.row_bytes),
            (segments.len() as u64 * 4, (at.seat.images + 1) * 4),
            (routes.len() as u64 * 4, at.seat.rows * 4),
            (positions.len() as u64 * 4, at.seat.rows * AXES * 4),
            (
                embed_rows.len() as u64 * 4,
                at.seat.rows * at.seat.embed_taps * 4,
            ),
            (
                embed_weights.len() as u64 * 4,
                if at.seat.embed_weights {
                    at.seat.rows * at.seat.embed_taps * 4
                } else {
                    0
                },
            ),
        ];
        for (need, have) in owed {
            if need > have {
                return Err(crate::error::Fault::Ceiling {
                    what: "bytes of the patch rectangle this load reserved",
                    need,
                    have,
                });
            }
        }
        let base = self.store.ptr();
        self.store.stage(stream, at.payload, payload)?;
        self.store.stage(stream, at.segments, bytes_of(segments))?;
        self.store.stage(stream, at.routes, bytes_of(routes))?;
        self.store.stage(stream, at.positions, bytes_of(positions))?;
        if !embed_rows.is_empty() {
            self.store.stage(stream, at.embed_rows, bytes_of(embed_rows))?;
        }
        if !embed_weights.is_empty() {
            self.store
                .stage(stream, at.embed_weights, f32_bytes_of(embed_weights))?;
        }
        let rows = if at.seat.row_bytes == 0 {
            0
        } else {
            (payload.len() as u64 / at.seat.row_bytes) as u32
        };
        let width = model_compiler::arena::elem_bytes(at.seat.dtype)
            .filter(|element| *element > 0)
            .map_or(0, |element| (at.seat.row_bytes / element) as u32);
        Ok(PatchHandles {
            patches: Tensor::new(base + at.payload, rows, width, at.seat.dtype),
            segments: i32s(base + at.segments, segments.len() as u32),
            routes: i32s(base + at.routes, routes.len() as u32),
            // `[patch rows, 3]` and not a flat vector: `rope_mrope` reads a
            // rectangle and refuses anything that is not three wide, which is
            // the check that would otherwise be nobody's.
            positions: Tensor::new(
                base + at.positions,
                (positions.len() / AXES as usize) as u32,
                AXES as u32,
                Dtype::I32,
            ),
            // `[patch rows, taps]` and not a flat vector: `embed_weighted`
            // reads the tap count off the operand's width, so the rectangle is
            // where that number lives.
            embed_rows: (!embed_rows.is_empty()).then(|| {
                let taps = at.seat.embed_taps.max(1) as u32;
                Tensor::new(
                    base + at.embed_rows,
                    embed_rows.len() as u32 / taps,
                    taps,
                    Dtype::I32,
                )
            }),
            embed_weights: (!embed_weights.is_empty()).then(|| {
                let taps = at.seat.embed_taps.max(1) as u32;
                Tensor::new(
                    base + at.embed_weights,
                    embed_weights.len() as u32 / taps,
                    taps,
                    Dtype::F32,
                )
            }),
        })
    }

    /// **THE TRUNK'S TRIPLE-WIDE POSITION STREAM, STAGED WHERE THE PATCH
    /// VECTORS ARE** (multimodal §6.3).
    ///
    /// One `cudaMemcpyAsync` from pageable host memory onto the fire's compute
    /// stream, in front of the rotation that reads it; no ring slot, no pinned
    /// reservation, for the reason [`Inputs::stage_patches`] states at length.
    /// `[rows, 3]` `i32`, one `(t, h, w)` per TOKEN row — every row of the
    /// fire, because the trunk is one region over the whole rectangle and a
    /// text lane's `(p, p, p)` is scalar rope rather than an absence.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved
    /// stream (which is the row ceiling's, tripled) or for a plan that
    /// reserved none, and [`Fault::Device`](crate::Fault::Device) for the copy.
    pub fn stage_mrope_positions(
        &mut self,
        stream: *mut core::ffi::c_void,
        positions: &[i32],
    ) -> Result<Tensor> {
        let Some(at) = self.mrope else {
            return Err(crate::error::Fault::Ceiling {
                what: "the triple-wide position stream, which this load reserved none of",
                need: positions.len() as u64 * 4,
                have: 0,
            });
        };
        let have = self.mrope_bytes;
        let need = positions.len() as u64 * 4;
        if need > have {
            return Err(crate::error::Fault::Ceiling {
                what: "bytes of the triple-wide position stream this load reserved",
                need,
                have,
            });
        }
        let base = self.store.ptr();
        self.store.stage(stream, at, bytes_of(positions))?;
        Ok(Tensor::new(
            base + at,
            (positions.len() / AXES as usize) as u32,
            AXES as u32,
            Dtype::I32,
        ))
    }

    pub fn commit(
        &mut self,
        stream: *mut core::ffi::c_void,
        slot: &SlotGuard,
        staged: &Staged,
    ) -> Result<Handles> {
        let Staged {
            rows,
            lanes,
            windows,
            qo_absolute,
            live,
            adapter_rows,
            mask_bytes,
            space_indices,
            space_lanes,
        } = staged;
        let (rows, lanes) = (*rows, *lanes);
        // What the LANE TABLES were written at, which is the fire's own lanes
        // everywhere but the bodies path; see `Staged::space_lanes`.
        let space_lanes = *space_lanes;
        // The offsets, taken as values before the split borrow below: they are
        // the carve's arithmetic and nothing here mutates them.
        let base = self.store.ptr();
        let (at_tokens, at_positions, at_windows) = (self.tokens, self.positions, self.windows);
        let at_qo_absolute = self.qo_absolute;
        let at_live = self.live_rows;
        let (at_row_valid, at_slot_ids) = (self.row_valid, self.slot_ids);
        let (at_routes, at_mask, at_mask_indptr) =
            (self.adapter_routes, self.mask_bits, self.mask_indptr);
        let places: Vec<SpaceAt> = self.spaces.clone();
        // Split borrow: the ring is read, the device store is written.
        let Inputs {
            store, staging, ..
        } = self;
        let host = staging[slot.at() as usize].host();

        // SAFETY (every `copy` below): the source is this slot's pinned
        // allocation, which outlives the copy because the `SlotGuard` is held
        // until the step's settlement callback drops it; the destination span
        // is checked by `Buffer::stage_from`.
        let mut copy = |offset: u64, len: usize| -> Result<()> {
            unsafe { store.stage_from(stream, offset, host.wrapping_add(offset as usize), len) }
        };

        copy(at_tokens, rows as usize * 4)?;
        copy(at_positions, rows as usize * 4)?;
        copy(at_windows, windows * 4)?;
        // And the fire-wide reading of the same boundaries, on the same
        // stream and under the same condition: only a fire that wrote one.
        if let Some(bounds) = qo_absolute {
            copy(at_qo_absolute, *bounds * 4)?;
        }
        // The live-rows seat rides the same stream-ordered H2D as everything
        // above it — and only when the fire wrote one. An all-absent fire is
        // byte-for-byte the copy list it always was.
        if let Some(words) = live {
            copy(at_live, *words * 4)?;
        }
        copy(at_row_valid, rows as usize)?;
        copy(at_slot_ids, lanes as usize * 4)?;
        if let Some(routes) = adapter_rows {
            copy(at_routes, *routes as usize * 4)?;
        }
        if let Some(bytes) = mask_bytes {
            copy(at_mask, *bytes as usize)?;
            copy(at_mask_indptr, (lanes as usize + 1) * 4)?;
        }
        let mut spaces = Vec::with_capacity(places.len());
        for (at, indices) in places.iter().zip(space_indices) {
            // **THE LANE TABLES GO OVER AT WHAT WAS WRITTEN**, which is the
            // fire's own lanes on every path but the bodies one and the
            // key's ladder reach on that one. Copying `lanes` where the
            // host holds more would leave the padded tail on the device as
            // the LAST fire's bytes, which is the exact thing the padding
            // exists to remove.
            copy(at.indptr, (space_lanes as usize + 1) * 4)?;
            copy(at.indices, *indices as usize * 4)?;
            copy(at.last_page_len, space_lanes as usize * 4)?;
            copy(at.kv_len, space_lanes as usize * 4)?;
            copy(at.write_page, rows as usize * 4)?;
            copy(at.write_offset, rows as usize * 4)?;
            // **AND THE HANDLES SAY WHAT WAS WRITTEN**, which is the chunk
            // the comment here used to promise (the plan-at-bucket-ceiling
            // design, chunk 3). A decode schedule carved at the bucket's lane
            // ceiling makes `paged_kv_t`'s `batch_size` that ceiling, and
            // `indptr[batch_size]` is the bound
            // `protective_get_kv_offset` clamps against — so the rectangle
            // has to admit the lane the plan names.
            //
            // **AND NOTHING WINDOWED MOVES A BYTE**, because `space_lanes`
            // IS `lanes` on every path but the bodies one, and even there
            // every windowed reading of these tables is a `Run::cut` at
            // `(lane_offset, lanes)` or `(lane_offset, lanes + 1)` — a slice
            // that answers the same pointer and the same row count whether
            // the handle it was taken from stopped at the fire's lanes or at
            // the ceiling. Only `Run::pool_absolute`, which hands the FIVE
            // FA2 names the fire's vectors whole, sees the difference, and
            // it is the door the ceiling was carved for.
            spaces.push(SpaceHandles {
                indptr: i32s(base + at.indptr, space_lanes + 1),
                indices: i32s(base + at.indices, *indices),
                last_page_len: i32s(base + at.last_page_len, space_lanes),
                kv_len: i32s(base + at.kv_len, space_lanes),
                write_page: i32s(base + at.write_page, rows),
                write_offset: i32s(base + at.write_offset, rows),
            });
        }

        Ok(Handles {
            tokens: i32s(base + at_tokens, rows),
            positions: i32s(base + at_positions, rows),
            windows: base + at_windows,
            qo_absolute: qo_absolute.map(|_| base + at_qo_absolute),
            live_rows: live.map(|_| base + at_live),
            spaces,
            slot_ids: i32s(base + at_slot_ids, lanes),
            adapter_routes: adapter_rows.map(|rows| i32s(base + at_routes, rows)),
            row_valid: Tensor::new(base + at_row_valid, rows, 1, Dtype::U8),
            // The slab is handed over WHOLE: its entries are bits, not fire
            // rows, so `Run::cut` excludes it for the same reason it excludes
            // the page-id list, and the span table beside it is what carries
            // a windowed consumer to the right lane.
            mask: mask_bytes.map(|bytes| Tensor::new(base + at_mask, bytes, 1, Dtype::U8)),
            mask_indptr: mask_bytes.map(|_| i32s(base + at_mask_indptr, lanes + 1)),
        })
    }

    /// The pool seats one fire lends its cache table.
    #[must_use]
    pub fn seats(&self, handles: &Handles, pages: u32, rows: u32, lanes: u32) -> crate::store::Seats {
        crate::store::Seats {
            lanes,
            rows,
            pages,
            spaces: handles
                .spaces
                .iter()
                .map(|space| SpaceSeat {
                    page_indptr: space.indptr,
                    page_indices: space.indices,
                    last_page_lens: space.last_page_len,
                    row_valid: handles.row_valid,
                })
                .collect(),
            slot_ids: handles.slot_ids,
            // **THE PLAIN FOLD IS THE DEFAULT, AND IT IS A DEFAULT AND NOT A
            // FIELD.** The RS seats are not staged inputs: the fold
            // predicate is written on the DEVICE by
            // `channel::mask_from_commit`, and the commit lengths and the
            // buffered-scatter decision are the fire's RS plan. They are
            // seated by `Seats::rs`, which is what a fire that carries a
            // recurrent verb calls and what every other fire does not.
            write_state: true,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::ABSENT,
            begin_at: Tensor::ABSENT,
        }
    }
}

/// One `i32` column, `n` rows tall.
fn i32s(ptr: u64, rows: u32) -> Tensor {
    Tensor::new(ptr, rows, 1, Dtype::I32)
}

/// A vector of `i32` as the bytes a copy takes.
///
/// Little-endian, stated rather than derived: every device this ships on is,
/// and the fire descriptor's own layout says the same thing for the same
/// reason. The one reinterpretation in the shell, and it is the operation
/// `bytemuck::cast_slice` exists to name — pulling a crate in for three lines
/// would add a dependency and say nothing this comment does not.
fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is `Copy`, has no padding and no niche, so all `4 * len`
    // of its bytes are initialized and readable as `u8`. The result borrows
    // the input and is read, never written, for the length of one enqueue.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

/// [`bytes_of`] for the one geometry stream that is not an index: the
/// interpolation weights (multimodal §9.2), which are `f32` because they are
/// the preprocessor's arithmetic.
///
/// A second function rather than a generic one, because its callers are the
/// element types this staging path will ever have and a `T: Pod` bound would
/// be a wider promise than the module keeps.
fn f32_bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `f32` is `Copy`, has no padding and no niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

/// [`bytes_of`] for the live-rows seat, whose words are `u32` because a row
/// COUNT is unsigned and the device guard reads it as one.
fn u32_bytes_of(values: &[u32]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `u32` is `Copy`, has no padding and no niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

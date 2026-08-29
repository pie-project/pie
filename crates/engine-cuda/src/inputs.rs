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
//! The unseated ones are not oversights: the qo boundaries were deliberately
//! unnamed, and `row_valid`/`slot_ids`/the mask spans/the workspace are
//! engine facts the entries take beside the ops' operands (the `MENLO-SEAM`
//! markers `run.rs` catalogues).
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
use kernels_cuda::attn::plan::Workspace;
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

/// The float workspace ONE latent (mla) schedule can ask for.
///
/// **THE LATENT PLANNER DOES NOT SIZE OFF A QUERY RECTANGLE**, which is why
/// its sibling's formula is wrong here rather than merely generous. `plan_mla`
/// sizes its split-kv partials off the CLUSTER GRID
/// (`kernels-cuda/src/attn/sched_mla.rs:127-138` and `:393-397`):
///
/// ```text
/// num_clusters   = SMs / cluster_size
/// cluster_tile_q = cluster_size * CTA_TILE_Q (64)
/// rows           = 2 * num_clusters * cluster_tile_q
/// partial_o      = rows * 2 * head_dim_o bytes (the partials are bf16)
/// partial_lse    = rows * 4              bytes, each 16-byte aligned
/// ```
///
/// `cluster_size` cancels: the planner picks 1 or 2 CTAs per cluster from this
/// fire's average packed query length, and whichever it picks the row count is
/// `2 * SMs * 64` — the integer division can only round it down, so that is the
/// bound this grant is sized at, and no fire's choice can exceed it.
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
    /// Every window's rebased boundaries, end to end
    /// ([`Windows::packed`](crate::window::Windows::packed)).
    pub windows: &'a [i32],
    /// Which recurrent bank each lane owns, in fire lane order.
    pub slot_ids: &'a [i32],
    /// Which adapter each token ROW routes to, in fire row order, or `None`
    /// when no lane carried one. Per ROW and not per lane, because that is
    /// what the correction kernel indexes with: `routes[row]` beside
    /// `x[row]`, the same shape `tokens` and `positions` have.
    pub adapter_routes: Option<&'a [i32]>,
    /// One geometry per kv space, in space order.
    pub spaces: &'a [Geometry],
    /// This fire's expanded lane masks, or `None` when no lane carried one.
    pub mask: Option<&'a crate::mask::Staged>,
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
    rows: u32,
    lanes: u32,
    windows: usize,
    /// How many rows the adapter axis staged, or `None` for a fire no lane
    /// routed.
    adapter_rows: Option<u32>,
    /// How many mask bytes staged, or `None` for a fire no lane masked.
    mask_bytes: Option<u32>,
    /// Per kv space, how many page ids its `indices` vector carries.
    space_indices: Vec<u32>,
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
    row_valid: u64,
    slot_ids: u64,
    adapter_routes: u64,
    mask_bits: u64,
    mask_bytes: u64,
    mask_indptr: u64,
    spaces: Vec<SpaceAt>,
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
        runs: u32,
        gathered: usize,
        sms: u32,
        runahead: engine::runahead::Runahead,
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
        //   `engine::fire::max_runs`, the artifact's own bound on how many
        //   intervals any mask breaks into, so the ceiling holds for every
        //   fire this load can be handed; an artifact P4 seated whole answers
        //   `1` and pays two ints per slot for a list it never fills.
        // - `per_gathered` for what a GATHERED window carries beside them
        //   (`window::Gathered`, `Fallback::Copy`): the row map the gather and
        //   the scatter read, and per kv space the page bounds, the compacted
        //   page-id list and the two per-lane vectors. `gathered` is
        //   `engine::fire::fragmentable` — how many distinct masks this
        //   artifact can ever find in pieces, which is 0 for every artifact P4
        //   seated whole and 1 for today's qwen texts.
        let per_gathered = rows + spaces as u64 * (3 * lanes + 1 + pages);
        let window_slots = (classes * (classes + 1) / 2 + 1) as u64;
        // `+ (lanes + 1)` per gathered window because it is a slot BEYOND the
        // `k(k+1)/2 + 1` runs: `seat` deliberately does not dedupe a gathered
        // window against a plain one of the same span (they mean two different
        // things by that rectangle), so its own boundary vector is an extra.
        let window_ints = window_slots * (lanes + 1 + 2 * u64::from(runs.max(1)))
            + gathered as u64 * (per_gathered + lanes + 1);

        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(ALIGN);
            here
        };
        let tokens = take(rows * 4);
        let positions = take(rows * 4);
        let windows = take(window_ints * 4);
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
        // about it. `runs` is `engine::fire::max_runs`, which is `1` for every
        // artifact P4 seated whole, and the carve below is then byte-for-byte
        // the one this shell has always made.
        let runs = runs.max(1);
        let grants: Vec<Option<Grant>> = facts
            .plans
            .iter()
            .map(|seat| {
                seat.map(|seat| {
                    let floats = match seat.kind {
                        StructKind::AttnPrefillPlan | StructKind::AttnPrefillPlanSm90 => {
                            graph_float_bytes(&seat.reading, sms)
                        }
                        // The prefill bound, applied to decode: a decode
                        // schedule stages less than this, and asking the
                        // prefill number for it is what this shell has always
                        // done — pre-existing, and not tightened here.
                        StructKind::AttnDecodePlan => graph_float_bytes(&seat.reading, sms),
                        StructKind::MlaPlan => latent_float_bytes(seat.reading.head_dim, sms),
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
            row_valid,
            slot_ids,
            adapter_routes,
            mask_bits,
            mask_bytes,
            mask_indptr,
            spaces,
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
    /// from one artifact: `engine::fire::max_runs` bounds every fire's split
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

        if fire.windows.len() as u64 > self.window_ints {
            return Err(crate::error::Fault::Ceiling {
                what: "packed window boundaries",
                need: fire.windows.len() as u64,
                have: self.window_ints,
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
        // The padding mask is all-valid in an eager fire: every row a fire
        // carries is a row it means. Under capture it is what tells the
        // writers which of a bucket's padded rows are real, which is why the
        // buffer exists now rather than at step 5.
        put(self.row_valid, &vec![1u8; rows as usize], "staged row_valid")?;
        put(self.slot_ids, bytes_of(fire.slot_ids), "staged slot ids")?;

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
            adapter_rows,
            mask_bytes,
            space_indices,
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
            adapter_rows,
            mask_bytes,
            space_indices,
        } = staged;
        let (rows, lanes) = (*rows, *lanes);
        // The offsets, taken as values before the split borrow below: they are
        // the carve's arithmetic and nothing here mutates them.
        let base = self.store.ptr();
        let (at_tokens, at_positions, at_windows) = (self.tokens, self.positions, self.windows);
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
            copy(at.indptr, (lanes as usize + 1) * 4)?;
            copy(at.indices, *indices as usize * 4)?;
            copy(at.last_page_len, lanes as usize * 4)?;
            copy(at.kv_len, lanes as usize * 4)?;
            copy(at.write_page, rows as usize * 4)?;
            copy(at.write_offset, rows as usize * 4)?;
            spaces.push(SpaceHandles {
                indptr: i32s(base + at.indptr, lanes + 1),
                indices: i32s(base + at.indices, *indices),
                last_page_len: i32s(base + at.last_page_len, lanes),
                kv_len: i32s(base + at.kv_len, lanes),
                write_page: i32s(base + at.write_page, rows),
                write_offset: i32s(base + at.write_offset, rows),
            });
        }

        Ok(Handles {
            tokens: i32s(base + at_tokens, rows),
            positions: i32s(base + at_positions, rows),
            windows: base + at_windows,
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

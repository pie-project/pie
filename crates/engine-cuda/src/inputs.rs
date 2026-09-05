//! Resident fire inputs: one allocation, carved once at load and overwritten every fire. Addresses are pointer-stable because a captured graph bakes them in and is never re-captured.

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

/// Int side of one plan grant: where a built schedule's offset table is staged. Builders refuse at build time (naming bytes asked vs. left) when a schedule does not fit.
const GRANT_INT_BYTES: u64 = 8 << 20;

/// Floor for the float side: split-kv partial outputs and log-sum-exps. A graph-shaped prefill schedule may want more; see [`graph_float_bytes`].
const GRANT_FLOAT_BYTES: u64 = 64 << 20;

/// Float workspace one graph-shaped prefill schedule can ask for. A short grant does not fail the build — it declines graph capture and falls back to an unshaped schedule.
fn graph_float_bytes(facts: &SpaceFacts, sms: u32) -> u64 {
    let padded = u64::from(2 * sms.max(1)) / u64::from(facts.kv_heads.max(1)).max(1);
    let tile = if facts.head_dim >= 256 { 64 } else { 128 };
    let heads = u64::from(facts.q_heads);
    // `tmp_v`: partials; `tmp_s`: their log-sum-exps. Both f32, 16-byte aligned.
    let v = heads * padded * tile * u64::from(facts.head_dim) * 4;
    let s = heads * padded * tile * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// Float workspace one graph-shaped FA2 prefill schedule can ask for at the bucket ceiling — covers the tile-count term [`graph_float_bytes`] misses, computed at the widest bucket and `Budget::max_lanes`.
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

/// Float workspace one graph-shaped decode schedule can ask for at the lane ceiling; the occupancy term is covered by the prefill formula, but the lane term is not, so the caller takes the max of both.
fn decode_float_bytes(facts: &SpaceFacts, lanes: u32) -> u64 {
    let padded = u64::from(lanes.max(1));
    let heads = u64::from(facts.q_heads);
    // `tmp_v` and `tmp_s`, on `sched_decode::layout`'s own terms.
    let v = heads * padded * u64::from(facts.head_dim) * 4;
    let s = heads * padded * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// Float workspace one latent (mla) schedule can ask for. `plan_mla` sizes split-kv partials off the cluster grid, not a query rectangle; `rows` bounds to `2*SMs*64` regardless of cluster size since it cancels out.
fn latent_float_bytes(rank: u32, sms: u32) -> u64 {
    let rows = 2 * u64::from(sms.max(1)) * 64;
    let partial_o = (rows * 2 * u64::from(rank)).next_multiple_of(16);
    let partial_lse = (rows * 4).next_multiple_of(16);
    (partial_o + partial_lse).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// The alignment every carved region starts on.
const ALIGN: u64 = 256;

/// How long [`Inputs::claim`] spins before it says the ring is oversubscribed. A correctly sized ring never reaches it.
const CLAIM_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// One plan value's grant, as offsets — turned into a [`Workspace`] once the store is allocated and its base address is known.
#[derive(Clone, Debug)]
struct Grant {
    /// One int carving per run of the region that builds this schedule, in run order.
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
    pub tokens: Tensor,
    pub positions: Tensor,
    /// Where the packed per-window boundary vectors landed — [`Windows::bind`](crate::window::Windows::bind) cuts them apart.
    pub windows: u64,
    /// Un-rebased qo prefix sums, `[lanes + 1]` `i32`. A base, not a `Tensor`, since a body bakes the pointer. `None` unless bodied.
    pub qo_absolute: Option<u64>,
    /// `None` when this fire staged no live words.
    pub live_rows: Option<u64>,
    pub spaces: Vec<SpaceHandles>,
    /// `[lanes]`: which recurrent bank each lane owns.
    pub slot_ids: Tensor,
    pub row_valid: Tensor,
    /// One adapter id per token row. `None` when no lane carried one.
    pub adapter_routes: Option<Tensor>,
    /// Packed `u8` (query, key) bits, fire-wide. `None` when no lane carried a mask, so a masked consumer answers `attn::masked`'s own refusal rather than reading zeros.
    pub mask: Option<Tensor>,
    /// Each lane's ABSOLUTE byte offset into [`mask`](Handles::mask).
    pub mask_indptr: Option<Tensor>,
}

/// Numbers per multimodal position: `(t, h, w)`. `RuntimeInput::PatchPositions` and `RuntimeInput::MropePositions` are the same triple over two rectangles.
const AXES: u64 = 3;

/// The second row axis's reservation, or `None` for a load whose plan states no patch row.
#[derive(Debug, Clone, Copy)]
pub struct PatchSeat {
    /// Most patch rows one fire may carry (`PatchLadder::max_patches`).
    pub rows: u64,
    /// One patch row's width in bytes: `C·T·P²` elements of the plan's dtype.
    pub row_bytes: u64,
    /// Most images one fire may carry (`PatchLadder::max_images`).
    pub images: u64,
    pub dtype: Dtype,
    /// Position-gather width: 1 (native grid), 4 (bilinear), 16 (bicubic), or 0 for no position gather.
    pub embed_taps: u64,
    /// Whether the plan also declares `RuntimeInput::PatchEmbedWeights`.
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
    /// `[patch rows, C·T·P²]`, the plan's element.
    pub patches: Tensor,
    pub segments: Tensor,
    /// One destination token row per patch row.
    pub routes: Tensor,
    /// `[patch rows, 3]` — one `(t, h, w)` per patch row.
    pub positions: Tensor,
    /// Rows of the learned position table each patch reads. `None` for a plan that declares no position gather.
    pub embed_rows: Option<Tensor>,
    /// `None` for a native-grid plan.
    pub embed_weights: Option<Tensor>,
}

/// One kv space's device seats.
#[derive(Debug, Clone, Copy)]
pub struct SpaceHandles {
    pub indptr: Tensor,
    pub indices: Tensor,
    pub last_page_len: Tensor,
    pub kv_len: Tensor,
    pub write_page: Tensor,
    pub write_offset: Tensor,
}

/// What one fire wants written, host side.
#[derive(Debug, Clone)]
pub struct Fire<'a> {
    pub tokens: &'a [i32],
    /// Absolute positions, in fire row order.
    pub positions: &'a [i32],
    /// Every window's rebased boundaries, at its slot's offset in the blob's fixed-width carve ([`Windows::packed`](crate::window::Windows::packed)).
    pub windows: &'a [i32],
    /// The other reading of [`windows`](Fire::windows): `[lanes + 1]` absolute qo boundaries. Empty for a fire that stages none.
    pub qo_absolute: &'a [i32],
    /// Live-rows seat words, one per (region, run). Empty for a fire that stages none.
    pub live: &'a [u32],
    /// Which recurrent bank each lane owns, in fire lane order.
    pub slot_ids: &'a [i32],
    /// Which adapter each token row routes to, or `None` if none carried one. Tail value `-1` means the base model.
    pub adapter_routes: Option<&'a [i32]>,
    /// One geometry per kv space. A bodied fire pads page CSR and per-lane vectors to the key's ladder reach so lanes past live ones read empty.
    pub spaces: &'a [Geometry],
    /// This fire's expanded lane masks, or `None` when no lane carried one.
    pub mask: Option<&'a crate::mask::Staged>,
    /// How many of [`tokens`](Fire::tokens) are this fire's own — the rest is carve padding — or `0` for "all of them".
    pub live_rows: u32,
}

/// Free set of a staging ring, as one word. Claim is compare-exchange on the lowest set bit; release is `fetch_or`, legal from the CUDA driver's callback thread (where a mutex or CUDA call would be a hazard).
#[derive(Debug)]
pub struct Free {
    /// Bit `i` set means slot `i` is claimable.
    bits: AtomicU64,
    depth: u32,
}

impl Free {
    /// A set of `depth` claimable slots. Shared with [`crate::settle::Settlement`], which recycles one event per in-flight step from the same callback thread.
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

    /// Give one back. Called from the driver's callback thread.
    pub fn give(&self, at: u32) {
        self.bits.fetch_or(1u64 << at, Ordering::Release);
    }

    /// How many slots are claimed right now — the steps the device may still be reading staging for.
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.depth - self.bits.load(Ordering::Acquire).count_ones()
    }
}

/// One claimed staging slot; dropping it releases the slot. Held from [`Inputs::claim`] in `prepare` until the step's settlement callback drops it, or until an aborted, never-enqueued frame drops it directly.
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

/// What one fire's host staging came to — the lengths the copies and the handles are cut at. Written by [`Inputs::write_host`], read by [`Inputs::commit`]; the bytes live in the claimed slot's pinned memory.
#[derive(Debug, Clone)]
pub struct Staged {
    /// Rows written — the fire's own row count, or the bucket ceiling for a bodied fire.
    rows: u32,
    lanes: u32,
    windows: usize,
    qo_absolute: Option<usize>,
    /// `None` until a caller fills [`Fire::live`].
    live: Option<usize>,
    adapter_rows: Option<u32>,
    mask_bytes: Option<u32>,
    /// Per kv space, how many page ids its `indices` vector carries.
    space_indices: Vec<u32>,
    /// Lanes the per-space lane tables were staged at: [`lanes`](Staged::lanes) normally, or the bucket's lane ceiling for a bodied fire.
    space_lanes: u32,
}

/// The resident inputs, carved once.
#[derive(Debug)]
pub struct Inputs {
    store: Buffer,
    /// Pinned mirror of the store's staged prefix, one per in-flight step.
    staging: Vec<Pinned>,
    /// Claimable slots, shared with every live [`SlotGuard`].
    free: Arc<Free>,
    /// Bytes of [`Inputs::store`] a fire stages, before the plan grants.
    stage_bytes: u64,
    tokens: u64,
    positions: u64,
    windows: u64,
    window_ints: u64,
    /// One fixed-width slot per distinct window ([`crate::window::Slots`]).
    window_slots: crate::window::Slots,
    /// Fire-wide qo vector, `lanes + 1` `i32`. Carved by every load; written only by a bodied fire.
    qo_absolute: u64,
    qo_absolute_ints: u64,
    /// `regions * max_runs * 4`, a `[rows, row_offset, lanes, lane_offset]` quad per seat.
    live_rows: u64,
    live_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    adapter_routes: u64,
    mask_bits: u64,
    mask_bytes: u64,
    mask_indptr: u64,
    /// Carved past the staged prefix: device bytes, no pinned mirror. `None` for a plan that states no patch row.
    patch: Option<PatchAt>,
    /// `RuntimeInput::MropePositions`, or `None` for texts that rotate by a scalar. Below the staged prefix and on no ring, like [`Inputs::patch`].
    mrope: Option<u64>,
    /// Bytes [`Inputs::mrope`] holds — the row ceiling tripled, or zero.
    mrope_bytes: u64,
    /// The self-conditioning taps' seat: ids then weights, each
    /// [`Inputs::self_cond_bytes`] long; `None` for a plan that reads none.
    self_cond: Option<u64>,
    self_cond_taps: u64,
    self_cond_bytes: u64,
    spaces: Vec<SpaceAt>,
    /// Lane ceiling every per-lane table was carved at (`Budget::max_lanes`).
    max_lanes: u32,
    /// Row ceiling every row-shaped table was carved at (`Budget::max_tokens`).
    max_rows: u32,
    /// One grant per (run, plan value), flat at `run * plan_values + value`. Needs no ring: source and destination are on the same stream within one `enqueue`, so in-flight frames cannot collide.
    plans: Vec<Option<Workspace>>,
    /// How many plan values one run's slice of [`plans`](Inputs::plans) holds.
    plan_values: usize,
}

impl Inputs {
    /// Reserve the vectors a deployment's ceilings admit.
    /// The device carve is one pointer-stable region (captured-graph
    /// addresses fixed at bake); the host side is a `runahead.staging_depth()`-deep pinned ring, so `write_host` may write frame W+1 while frame W's copies are in flight.
    /// # Errors: [`Fault::Device`](crate::Fault::Device) for the device allocation or any of the ring's pinned ones.
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
        // Device facts the plan builders take: `num_sm` sizes the grants below; `cc_major` lets the prefill grant ask the planner for its tile rather than bound it by hand (`prefill_float_bytes`).
        device: Device,
        runahead: engine::runahead::Runahead,
        patch: Option<PatchSeat>,
        mrope: bool,
        self_cond_taps: u64,
    ) -> Result<Inputs> {
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(budget.max_lanes);
        let pages = u64::from(budget.max_lanes) * u64::from(paging.pages_per_slot);
        // A window is one contiguous run of classes: at most `k(k+1)/2` for `k` classes, plus one shared empty window. Reserved unconditionally since addresses are recorded into a never-re-captured graph; slots are fixed-width.
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
        // Same boundaries, un-rebased: `[lanes + 1]` i32 fire-wide qo prefix sums, so a consumer can take this vector whole. Carved unconditionally; only a bodied fire writes it.
        let qo_absolute_ints = lanes + 1;
        let qo_absolute = take(qo_absolute_ints * 4);
        // Live-rows seat: `Ctx::arm_stage` reads this per region so a graph replay serves the right row count from memory instead of a baked node parameter. Carved unconditionally; only a bodied fire writes it.
        let live_seat = crate::window::Seat::new(regions as u64, u64::from(runs.max(1)));
        let live_ints = live_seat.words();
        let live_rows = take(live_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * 4);
        // Reserved unconditionally: a conditional carve would make the store's layout depend on the plan.
        let adapter_routes = take(rows * 4);
        // Masked axis's two vectors. `context` is what a slot can hold, so `rows * context` bounds every (query, key) cell a fire can present; `+ lanes` is one byte-alignment pad per lane.
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
        // Where the staged prefix ends: above is written by a fire and copied by `commit`; below is granted to schedule builders staging directly onto the stream. `take(0)` reads the cursor without moving it.
        let stage_bytes = take(0);
        // Patch regions, below the line: device bytes, no pinned mirror. A text-only load's carve is unaffected; `PatchSeat`'s `None` costs not even an offset.
        let patch = patch.map(|seat| PatchAt {
            payload: take(seat.rows * seat.row_bytes),
            segments: take((seat.images + 1) * 4),
            routes: take(seat.rows * 4),
            // Tower's own rotation stream, `[patch rows, 3]` i32.
            positions: take(seat.rows * AXES * 4),
            // Position gather's two streams, sized by the plan's tap count: 0 taps carves nothing; a native-grid plan carves ids at one tap and no weights.
            embed_rows: take(seat.rows * seat.embed_taps * 4),
            embed_weights: if seat.embed_weights {
                take(seat.rows * seat.embed_taps * 4)
            } else {
                0
            },
            seat,
        });
        // Trunk's triple-wide token stream, below the line like patch, reserved only when the plan names it. `[max_tokens, 3]` i32, not paid at all by a scalar-rotate text.
        let mrope = mrope.then(|| take(rows * AXES * 4));
        // The denoiser's taps: `[max_tokens, taps]` i32 ids and f32 weights, reserved only when the plan reads them.
        let self_cond = (self_cond_taps > 0).then(|| take(rows * self_cond_taps * 4 * 2));
        // One grant per plan value: the float side is the requirement of the builder that will actually run, or the flat floor, whichever is larger — computed, not guessed, since a short grant declines to capture instead of failing.
        let runs = runs.max(1);
        let grants: Vec<Option<Grant>> = facts
            .plans
            .iter()
            .map(|seat| {
                seat.map(|seat| {
                    let floats = match seat.kind {
                        // Occupancy term vs. ceiling term; larger wins.
                        StructKind::AttnPrefillPlan => graph_float_bytes(&seat.reading, device.num_sm)
                            .max(prefill_float_bytes(
                                &seat.reading,
                                budget.buckets.last().copied().unwrap_or(budget.max_tokens),
                                budget.max_lanes,
                                &device,
                            )),
                        // `sched_sm90` allocates no floats (`Built::float_bytes` is zero), so only its int ask grows, covered by the flat `GRANT_INT_BYTES`.
                        StructKind::AttnPrefillPlanSm90 => {
                            graph_float_bytes(&seat.reading, device.num_sm)
                        }
                        // Prefill bound covers decode's occupancy term (same q_heads/head_dim); the lane term is separate, since a graph-shaped decode also pads to `max_lanes`.
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
        // Ring is the host half only: one pinned mirror of the staged prefix per slot. Pinned, not pageable, so the H2D copy is asynchronous.
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
            self_cond,
            self_cond_taps,
            self_cond_bytes: if self_cond.is_some() { rows * self_cond_taps * 4 } else { 0 },
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

    /// One plan value's builder grant, by value id and run. `None` for a run past what this load reserved, same as a non-plan value.
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

    /// Claim a staging slot — the fire path's one resource acquisition. Spins rather than parks: a slot releases in microseconds from the settlement callback, and a condvar would put a mutex on the driver's callback thread. # Errors: [`Fault::Ceiling`](crate::Fault::Ceiling) naming the ring, for a caller holding every slot past [`CLAIM_DEADLINE`].
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
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.free.in_flight()
    }

    /// The window blob's carve, for `Windows::of`/`Windows::packed` to lay every slot out where this reserve put it.
    #[must_use]
    pub fn window_slots(&self) -> crate::window::Slots {
        self.window_slots
    }

    /// Write one fire's vectors into a claimed slot — host only, no stream. Every ceiling this staging enforces is enforced here. # Errors: [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved ceilings.
    pub fn write_host(&self, slot: &SlotGuard, fire: &Fire<'_>) -> Result<Staged> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32;
        let host = &self.staging[slot.at() as usize];

        // A bodied fire pads its token, position and rotation vectors out to the bucket its launches are gridded at; refuse a padding past what `reserve` carved, which would write over the region behind it.
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
        // Lane count is read off the spaces since every space of one fire must agree.
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

        // `put` refuses a region past the slot mirror — the same ceiling the device carve would enforce, caught one phase earlier.
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
        // row_valid is all-1 over `live` rows (this fire's own) and 0 over the carve's padding, telling the writers which of a bucket's rows are real. A fire that padded nothing states `live == rows`.
        let live = if fire.live_rows == 0 {
            rows
        } else {
            fire.live_rows.min(rows)
        };
        let mut valid = vec![1u8; live as usize];
        valid.resize(rows as usize, 0);
        put(self.row_valid, &valid, "staged row_valid")?;
        put(self.slot_ids, bytes_of(fire.slot_ids), "staged slot ids")?;
        // Lanes a ceiling plan names but this fire did not bring are padded with `-1`: without it the device tail is whatever the last fire left, and `attn/ssm.cuh`'s `if (slot < 0) return` cannot refuse valid-looking stale ids.
        if space_lanes > lanes {
            let inert = vec![-1i32; (space_lanes - lanes) as usize];
            put(
                self.slot_ids + u64::from(lanes) * 4,
                bytes_of(&inert),
                "staged slot id padding",
            )?;
        }

        // Empty `Fire::qo_absolute` writes/copies/publishes nothing.
        let qo_absolute = if fire.qo_absolute.is_empty() {
            None
        } else {
            put(self.qo_absolute, bytes_of(fire.qo_absolute), "staged absolute qo bounds")?;
            Some(fire.qo_absolute.len())
        };

        // Empty `Fire::live` writes nothing and costs no H2D or launch argument; the device carve exists either way.
        let live = if fire.live.is_empty() {
            None
        } else {
            put(self.live_rows, u32_bytes_of(fire.live), "staged live rows")?;
            Some(fire.live.len())
        };

        // A fire no lane routed writes nothing here and binds no seat.
        let adapter_rows = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                put(self.adapter_routes, bytes_of(routes), "staged adapter routes")?;
                Some(routes.len() as u32)
            }
        };

        // A fire no lane masked writes nothing and binds no seat, so a masked consumer sees `attn::masked`'s refusal rather than reading a zeroed (all-masked-out) slab.
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

    /// Second row axis's H2D, inside the enqueue and outside the ring. Pageable is safe without a slot: the driver copies the source before the call returns. # Errors: [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved patch rectangle, [`Fault::Device`](crate::Fault::Device) for the copies.
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
            // `[patch rows, 3]`: `rope_mrope` refuses anything not three wide.
            positions: Tensor::new(
                base + at.positions,
                (positions.len() / AXES as usize) as u32,
                AXES as u32,
                Dtype::I32,
            ),
            // `[patch rows, taps]`: `embed_weighted` reads the tap count off this width.
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

    /// Trunk's triple-wide position stream, staged where the patch vectors are. `[rows, 3]` i32, one `(t, h, w)` per token row — every row, since a text lane's `(p, p, p)` is scalar rope, not an absence. # Errors: [`Fault::Ceiling`](crate::Fault::Ceiling) for a fire past the reserved stream (the row ceiling's, tripled) or none reserved, [`Fault::Device`](crate::Fault::Device) for the copy.
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

    /// Stage a fire's self-conditioning taps: `rows` ids and `weights`, each
    /// `[token rows, taps]` row major, into the seat the load reserved.
    pub fn stage_self_cond(
        &mut self,
        stream: *mut core::ffi::c_void,
        rows: &[i32],
        weights: &[f32],
    ) -> Result<(Tensor, Tensor)> {
        let Some(at) = self.self_cond else {
            return Err(crate::error::Fault::Ceiling {
                what: "the self-conditioning taps, which this load reserved no seat for",
                need: rows.len() as u64 * 4,
                have: 0,
            });
        };
        let need = rows.len() as u64 * 4;
        if need > self.self_cond_bytes || weights.len() != rows.len() {
            return Err(crate::error::Fault::Ceiling {
                what: "bytes of self-conditioning taps this load reserved",
                need,
                have: self.self_cond_bytes,
            });
        }
        let base = self.store.ptr();
        let weights_at = at + self.self_cond_bytes;
        self.store.stage(stream, at, bytes_of(rows))?;
        self.store.stage(stream, weights_at, f32_bytes_of(weights))?;
        let taps = self.self_cond_taps as u32;
        let token_rows = (rows.len() as u64 / self.self_cond_taps.max(1)) as u32;
        Ok((
            Tensor::new(base + at, token_rows, taps, Dtype::I32),
            Tensor::new(base + weights_at, token_rows, taps, Dtype::F32),
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
        // What the lane tables were written at (see `Staged::space_lanes`).
        let space_lanes = *space_lanes;
        // Offsets taken as values before the split borrow below.
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

        // SAFETY (every `copy` below): source is this slot's pinned allocation, alive until the `SlotGuard` drops; destination span is checked by `Buffer::stage_from`.
        let mut copy = |offset: u64, len: usize| -> Result<()> {
            unsafe { store.stage_from(stream, offset, host.wrapping_add(offset as usize), len) }
        };

        copy(at_tokens, rows as usize * 4)?;
        copy(at_positions, rows as usize * 4)?;
        copy(at_windows, windows * 4)?;
        // Copied only if the fire wrote one.
        if let Some(bounds) = qo_absolute {
            copy(at_qo_absolute, *bounds * 4)?;
        }
        if let Some(words) = live {
            copy(at_live, *words * 4)?;
        }
        copy(at_row_valid, rows as usize)?;
        // Copied at `space_lanes`, with every other per-lane table: the host vector was `-1`-padded to that reach (see `write_host`).
        copy(at_slot_ids, space_lanes as usize * 4)?;
        if let Some(routes) = adapter_rows {
            copy(at_routes, *routes as usize * 4)?;
        }
        if let Some(bytes) = mask_bytes {
            copy(at_mask, *bytes as usize)?;
            copy(at_mask_indptr, (lanes as usize + 1) * 4)?;
        }
        let mut spaces = Vec::with_capacity(places.len());
        for (at, indices) in places.iter().zip(space_indices) {
            // Lane tables copy at `space_lanes`, not `lanes` — leaving the padded tail as a previous fire's bytes would defeat the padding.
            copy(at.indptr, (space_lanes as usize + 1) * 4)?;
            copy(at.indices, *indices as usize * 4)?;
            copy(at.last_page_len, space_lanes as usize * 4)?;
            copy(at.kv_len, space_lanes as usize * 4)?;
            copy(at.write_page, rows as usize * 4)?;
            copy(at.write_offset, rows as usize * 4)?;
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
            // The handle says what was written, at `space_lanes`.
            slot_ids: i32s(base + at_slot_ids, space_lanes),
            adapter_routes: adapter_rows.map(|rows| i32s(base + at_routes, rows)),
            row_valid: Tensor::new(base + at_row_valid, rows, 1, Dtype::U8),
            // Handed over whole: entries are bits, not fire rows, so `Run::cut` excludes it, like the page-id list.
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
            // Plain fold is the default: RS seats aren't staged inputs — the fold predicate lives on the device (`channel::mask_from_commit`); a fire with a recurrent verb calls `Seats::rs` instead.
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

/// A vector of `i32` as the bytes a copy takes. Little-endian — every device this ships on is, and so is the fire descriptor's layout.
fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is `Copy` with no padding/niche, so all `4 * len` bytes are initialized. Result borrows the input, read-only, for one enqueue.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

/// [`bytes_of`] for the one geometry stream that is not an index: the interpolation weights, which are `f32` (preprocessor arithmetic).
fn f32_bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `f32` is `Copy`, no padding or niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

/// [`bytes_of`] for the live-rows seat: `u32` because a row count is unsigned and the device guard reads it as one.
fn u32_bytes_of(values: &[u32]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `u32` is `Copy`, no padding or niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

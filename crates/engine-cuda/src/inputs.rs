#![allow(unsafe_code)]
use kernels_cuda::Tensor;
use kernels_cuda::attn::plan::{Device, Toggles, Workspace, prefill_graph_float_bytes};
use poem_compiler::Budget;
use poem_ir::{Dtype, StructKind};

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::device::{Buffer, Pinned};
use crate::error::Result;
use crate::store::TableSeat;
use crate::store::kv::{Facts, Geometry, Paging, SpaceFacts};

const GRANT_INT_BYTES: u64 = 1 << 20;

const GRANT_FLOAT_BYTES: u64 = 64 << 20;

/// The most float workspace a captured prefill plan reading `facts` asks
/// over `shapes`, each a body's (rows, lanes) carve: each shape's ask is
/// the plan's own layout of it ([`prefill_graph_float_bytes`]). The ask is
/// no function of one corner: the CTA tile narrows as the lanes thin each
/// lane's rows, and a split shows only below half the grid in tiles, so the
/// widest rung over the most lanes can ask less than a middle rung over
/// fewer (on an L40S a 512-row rung of Qwen3-0.6B splits over 256 lanes on
/// 128-row tiles, 132 MiB of partials, half that over 128 lanes, and over
/// 512 lanes its 16-row tiles run unsplit and ask nothing).
fn prefill_float_bytes(
    facts: &SpaceFacts,
    page_size: u32,
    shapes: impl Iterator<Item = (u32, u32)>,
    device: &Device,
    toggles: Toggles,
) -> Result<u64> {
    let mut most = 0usize;
    for (rows, lanes) in shapes {
        let asked = prefill_graph_float_bytes(
            rows,
            lanes,
            facts.q_heads,
            facts.kv_heads,
            facts.head_dim,
            page_size,
            facts.window,
            device,
            toggles,
        )
        .map_err(|refused| crate::error::Fault::Unbound {
            what: format!(
                "the prefill workspace grant: the graph form of {rows} rows over {lanes} \
                 lanes ({} query heads over {} kv heads, {} wide, window {:?}, pages of \
                 {page_size}, {} SMs at {} CTAs an SM) refused: {refused}",
                facts.q_heads,
                facts.kv_heads,
                facts.head_dim,
                facts.window,
                device.num_sm,
                toggles.prefill_ctas_per_sm,
            ),
        })?;
        most = most.max(asked);
    }
    Ok((most as u64).next_multiple_of(ALIGN) + 2 * ALIGN)
}

/// A floor on the decode room, well past what the decode grid uses: the token
/// budget is fitted against it (`serve::load`), and the later state-slot fit
/// does not follow a smaller room.
fn graph_float_bytes(facts: &SpaceFacts, sms: u32) -> u64 {
    let padded = u64::from(2 * sms.max(1)) / u64::from(facts.kv_heads.max(1)).max(1);
    let tile = if facts.head_dim >= 256 { 64 } else { 128 };
    let heads = u64::from(facts.q_heads);
    let v = heads * padded * tile * u64::from(facts.head_dim) * 4;
    let s = heads * padded * tile * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

fn decode_float_bytes(facts: &SpaceFacts, lanes: u32) -> u64 {
    let padded = u64::from(lanes.max(1));
    let heads = u64::from(facts.q_heads);
    let v = heads * padded * u64::from(facts.head_dim) * 4;
    let s = heads * padded * 4;
    (v + s).next_multiple_of(ALIGN) + 2 * ALIGN
}

fn latent_float_bytes(rank: u32, sms: u32) -> u64 {
    let rows = 2 * u64::from(sms.max(1)) * 64;
    let partial_o = (rows * 2 * u64::from(rank)).next_multiple_of(16);
    let partial_lse = (rows * 4).next_multiple_of(16);
    (partial_o + partial_lse).next_multiple_of(ALIGN) + 2 * ALIGN
}

/// The float workspace each attention plan's launches need, sized for the
/// budget's widest fire: a prefill plan for the most any body the lattice
/// arms asks (each bucket's rows over up to `fire_lanes` lanes, the most a
/// body carves: `Shell::lane_ceiling`), a decode plan for `max_lanes` rows,
/// on this device's SMs.
fn plan_floats(
    budget: &Budget,
    fire_lanes: u32,
    page_size: u32,
    facts: &Facts,
    device: &Device,
    toggles: Toggles,
) -> Result<Vec<Option<u64>>> {
    let lanes = fire_lanes.max(1);
    let buckets: Vec<u32> = if budget.buckets.is_empty() {
        vec![budget.max_tokens]
    } else {
        budget.buckets.clone()
    };
    // Plans reading the same space ask the same: each is swept once.
    let mut swept: Vec<((SpaceFacts, bool), u64)> = Vec::new();
    let mut prefill = |reading: SpaceFacts, rows_a_lane: bool| -> Result<u64> {
        if let Some((_, bytes)) = swept.iter().find(|(key, _)| *key == (reading, rows_a_lane)) {
            return Ok(*bytes);
        }
        let bytes = if rows_a_lane {
            // Read only by decoders (a grouped decode on the prefill
            // kernel): a row a lane.
            prefill_float_bytes(
                &reading,
                page_size,
                (1..=lanes).map(|lanes| (lanes, lanes)),
                device,
                toggles,
            )?
        } else {
            prefill_float_bytes(
                &reading,
                page_size,
                buckets
                    .iter()
                    .flat_map(|&rows| (1..=lanes).map(move |lanes| (rows, lanes))),
                device,
                toggles,
            )?
        };
        swept.push(((reading, rows_a_lane), bytes));
        Ok(bytes)
    };
    facts
        .plans
        .iter()
        .map(|seat| {
            seat.map(|seat| {
                Ok(match seat.kind {
                    StructKind::AttnPrefillPlan => prefill(seat.reading, seat.rows_a_lane)?,
                    StructKind::AttnDecodePlan => graph_float_bytes(&seat.reading, device.num_sm)
                        .max(decode_float_bytes(&seat.reading, budget.max_lanes)),
                    StructKind::MlaPlan => latent_float_bytes(seat.reading.head_dim, device.num_sm),
                }
                .max(GRANT_FLOAT_BYTES))
            })
            .transpose()
        })
        .collect()
}

/// What the attention workspaces come to at this budget: plans launched on
/// one stream share a workspace sized for the widest of them, every other
/// plan holds its own. This is the lane- and token-scaled part of the
/// inputs, and on a wide model at a long envelope it is gigabytes, so the
/// boot sizes the token budget against it before anything is allocated.
pub fn attention_workspace_bytes(
    budget: &Budget,
    fire_lanes: u32,
    page_size: u32,
    facts: &Facts,
    device: &Device,
    toggles: Toggles,
    schedule_streams: &[Option<u32>],
) -> Result<u64> {
    let floats_of = plan_floats(budget, fire_lanes, page_size, facts, device, toggles)?;
    let mut per_stream: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    let mut own = 0u64;
    for (plan, floats) in floats_of.iter().enumerate() {
        let Some(floats) = floats else { continue };
        match schedule_streams.get(plan) {
            Some(Some(stream)) => {
                let held = per_stream.entry(*stream).or_insert(0);
                *held = (*held).max(*floats);
            }
            _ => own = own.saturating_add(*floats),
        }
    }
    Ok(per_stream
        .values()
        .fold(own, |sum, floats| sum.saturating_add(*floats)))
}

const ALIGN: u64 = 256;

#[derive(Clone, Debug)]
struct Grant {
    int_at: Vec<u64>,
    float_at: u64,
    float_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct TableAt {
    indptr: u64,
    indices: u64,
    last_page_len: u64,
    kv_len: u64,
    write_page: u64,
    write_offset: u64,
}

#[derive(Debug, Clone)]
pub struct Handles {
    pub tokens: Tensor,
    pub positions: Tensor,
    pub windows: u64,
    pub qo_absolute: Option<u64>,
    pub live_rows: Option<u64>,
    pub page_tables: Vec<TableHandles>,
    pub slot_ids: Tensor,
    pub row_valid: Tensor,
    pub adapter_routes: Option<Tensor>,
    pub readout_rows: Tensor,
    /// How many of `readout_rows` are live (the rest pad a body's ceiling).
    pub readout_live: u64,
    pub mask: Option<Tensor>,
    pub mask_indptr: Option<Tensor>,
    /// Per lane its mask word: its committed boundary and its wide bit
    /// (`mask::Staged::origins`).
    pub mask_origin: Option<Tensor>,
    pub group_of_lane: Option<Tensor>,
    pub packings: Vec<PackingHandles>,
    pub lane_of_row: Tensor,
    /// The fire's rs tables, when it buffers recurrent rows.
    pub rs: Option<RsHandles>,
}

/// Where a fire's staged rs tables lie on the device: a
/// [`kernels_cuda::runtime::channel::RsLane`] a lane (padded lanes inert),
/// the slot address of each lane's rs buffer pages, and each replayed row's
/// buffer token. Fixed for the load, so a recorded body reads whatever slots
/// the fire it serves names.
#[derive(Debug, Clone, Copy)]
pub struct RsHandles {
    pub lanes: u64,
    pub pages: u64,
    pub replay_tok: u64,
}

const RS_LANE_BYTES: u64 = size_of::<kernels_cuda::runtime::channel::RsLane>() as u64;

/// The bytes the staged rs lane table of `lanes` lanes takes: what the slab
/// lays, what a fire writes and what it copies to the device, one figure.
const fn rs_lane_table_bytes(lanes: u64) -> u64 {
    lanes * RS_LANE_BYTES
}

/// A buffering fire's rs tables, as `write_host` stages them.
#[derive(Debug, Clone, Copy)]
pub struct RsFire<'a> {
    pub lanes: &'a [kernels_cuda::runtime::channel::RsLane],
    pub pages: &'a [u64],
    pub replay_tok: &'a [u32],
}

/// The rs section's offsets and capacities.
#[derive(Debug, Clone, Copy)]
struct RsAt {
    lanes: u64,
    pages: u64,
    pages_cap: u64,
    replay_tok: u64,
    replay_cap: u64,
}

const AXES: u64 = 3;

#[derive(Debug, Clone, Copy)]
pub struct PatchSeat {
    pub rows: u64,
    pub row_bytes: u64,
    pub images: u64,
    pub dtype: Dtype,
}

#[derive(Debug, Clone, Copy)]
struct PatchAt {
    payload: u64,
    segments: u64,
    routes: u64,
    positions: u64,
    grids: u64,
    seat: PatchSeat,
}

#[derive(Debug, Clone, Copy)]
pub struct PatchHandles {
    pub patches: Tensor,
    pub segments: Tensor,
    pub routes: Tensor,
    pub positions: Tensor,
    pub grids: Tensor,
}

pub use poem_exec::feeds::Seat;

#[derive(Debug, Clone, Copy)]
struct SeatAt {
    seat: Seat,
    at: u64,
    bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PackingFire<'a> {
    pub group_indptr: &'a [i32],
    pub lane_indptr: &'a [i32],
    pub permutation: &'a [i32],
}

#[derive(Debug, Clone, Copy)]
pub struct PackingHandles {
    pub group_indptr: Tensor,
    pub lane_indptr: Tensor,
    pub permutation: Tensor,
}

#[derive(Debug, Clone, Copy)]
struct PackingAt {
    group_indptr: u64,
    lane_indptr: u64,
    permutation: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TableHandles {
    pub indptr: Tensor,
    pub indices: Tensor,
    pub last_page_len: Tensor,
    pub kv_len: Tensor,
    pub write_page: Tensor,
    pub write_offset: Tensor,
}

#[derive(Debug, Clone)]
pub struct Fire<'a> {
    pub tokens: &'a [i32],
    pub positions: &'a [i32],
    pub windows: &'a [i32],
    pub qo_absolute: &'a [i32],
    pub live: &'a [u32],
    /// The record each lane names (under zero: none), a lane each.
    pub slot_ids: &'a [i32],
    /// The ids the recurrent kernels read the records by: `slot_ids`
    /// rewritten a lane each, or a block of lanes a table of pieces
    /// (`Pools::state_tables`).
    pub state_ids: &'a [i32],
    pub adapter_routes: Option<&'a [i32]>,
    pub readout_rows: &'a [i32],
    pub readout_live: u32,
    pub page_tables: &'a [Geometry],
    pub mask: Option<&'a crate::mask::Staged>,
    pub live_rows: u32,
    pub lane_of_row: &'a [i32],
    pub lane_reach: u32,
    pub group_of_lane: &'a [i32],
    pub packings: &'a [PackingFire<'a>],
    /// The rs tables of a fire whose lanes buffer recurrent rows.
    pub rs: Option<RsFire<'a>>,
}

/// A set of `depth` slots, taken and given back. A slot given back wakes a
/// claim waiting on the set ([`Free::claim`]); a guard handed to its step's
/// landing ([`SlotGuard::landing`]) is counted until it drops.
#[derive(Debug)]
pub struct Free {
    bits: AtomicU64,
    depth: u32,
    /// The taken slots a step's landing holds: each comes back without the
    /// claimant's doing.
    landing: AtomicU32,
    lock: std::sync::Mutex<()>,
    given: std::sync::Condvar,
}

impl Free {
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
            landing: AtomicU32::new(0),
            lock: std::sync::Mutex::new(()),
            given: std::sync::Condvar::new(),
        })
    }

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

    pub fn give(&self, at: u32) {
        self.bits.fetch_or(1u64 << at, Ordering::Release);
        // Under the lock, so a claim between its failed take and its wait
        // does not miss this.
        let _held = self.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.given.notify_all();
    }

    /// A slot of the set. Where every one is taken and a step's landing
    /// holds one, it waits for a slot to come back, however long the device
    /// takes: a landing gives its slot back. Where the claimant's own
    /// unlanded steps hold every one, none can come back until it lands one
    /// itself, and the claim is refused by name at once: the claimant is
    /// running deeper than the `depth` it was laid for.
    pub fn claim(self: &Arc<Self>, what: &'static str) -> Result<SlotGuard> {
        let mut held = self.lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(at) = self.take() {
                return Ok(SlotGuard {
                    free: Arc::clone(self),
                    at,
                    landing: false,
                });
            }
            if self.landing.load(Ordering::Acquire) == 0 {
                return Err(crate::error::Fault::Ceiling {
                    what,
                    need: u64::from(self.depth) + 1,
                    have: u64::from(self.depth),
                });
            }
            held = self
                .given
                .wait(held)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.depth - self.bits.load(Ordering::Acquire).count_ones()
    }
}

#[derive(Debug)]
pub struct SlotGuard {
    free: Arc<Free>,
    at: u32,
    landing: bool,
}

impl SlotGuard {
    #[must_use]
    pub fn at(&self) -> u32 {
        self.at
    }

    /// The guard as its step's landing holds it: it comes back when the step
    /// lands, without its claimant's doing, so a claim on a full set waits
    /// for it.
    #[must_use]
    pub fn landing(mut self) -> SlotGuard {
        if !self.landing {
            self.landing = true;
            self.free.landing.fetch_add(1, Ordering::AcqRel);
        }
        self
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        // Given back before it stops counting as landing: a claim that saw
        // the set full then either takes it or is woken for it.
        self.free.give(self.at);
        if self.landing {
            self.free.landing.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Staged {
    rows: u32,
    lanes: u32,
    windows: usize,
    qo_absolute: Option<usize>,
    live: Option<usize>,
    adapter_rows: Option<u32>,
    readouts: u32,
    mask_bytes: Option<u32>,
    table_indices: Vec<u32>,
    table_lanes: u32,
    /// The ids staged for the recurrent rows.
    state_ids: u32,
    packed: bool,
    /// The rs tables' pages and replayed rows staged, when it buffers.
    rs: Option<(u32, u32)>,
}

impl Staged {
    /// The lanes the fire's lane tables run to, its packing's group bounds
    /// among them.
    #[must_use]
    pub fn table_lanes(&self) -> u32 {
        self.table_lanes
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TokenInject {
    pub dst_off: u64,
    pub src: u64,
    pub bytes: usize,
}

/// A lane's device-carried attention mask: the `bits` bool bytes at `src`
/// packed into its row of the staged mask, `dst_off` bytes in.
#[derive(Clone, Copy, Debug)]
pub struct MaskInject {
    pub dst_off: u64,
    pub src: u64,
    pub bits: u32,
}

/// The fewest device-carried token runs one gather launch takes; under it
/// each is copied on its own.
const GATHERED_INJECTS: usize = 4;

/// Refuses by name, before anything is staged, a fire whose token rows (or
/// any table laid a row each: kv write pages and offsets, readout rows,
/// adapter routes) pass the `regions` rows the slab laid them for: a table
/// longer than its region would run on into the next one.
fn rows_fit(regions: u32, rows: u32, fire: &Fire<'_>) -> Result<()> {
    let tables = fire
        .page_tables
        .iter()
        .flat_map(|geometry| [geometry.write_page.len(), geometry.write_offset.len()])
        .chain([fire.readout_rows.len()])
        .chain(fire.adapter_routes.map(<[i32]>::len));
    let widest = tables.fold(rows as usize, usize::max);
    if widest > regions as usize {
        return Err(crate::error::Fault::program(
            "inputs::write_host",
            format!(
                "token rows a fire stages exceed its rung's regions: {widest} rows (the fire's \
                 {rows}) against the {regions} its regions were laid for"
            ),
        ));
    }
    Ok(())
}

/// Lands each lane's device-carried tokens at its rows of `tokens`: past a few
/// lanes in one gather launch; one copy each inside a stream capture.
fn inject_tokens(
    device: &crate::device::Context,
    runs: (&mut Buffer, u64),
    tokens: u64,
    injects: &[TokenInject],
) -> Result<()> {
    let stream = device.stream();
    let words = injects
        .iter()
        .all(|inject| inject.bytes % 4 == 0 && inject.dst_off % 4 == 0);
    if injects.len() < GATHERED_INJECTS
        || !words
        || crate::device::alloc::is_capturing(stream)
    {
        for inject in injects {
            crate::device::copy_d2d(stream, tokens + inject.dst_off, inject.src, inject.bytes)?;
        }
        return Ok(());
    }
    gather_injects(device, runs, tokens, injects)
}

fn gather_injects(
    device: &crate::device::Context,
    (store, at): (&mut Buffer, u64),
    tokens: u64,
    injects: &[TokenInject],
) -> Result<()> {
    let stream = device.stream();
    let table: Vec<kernels_cuda::runtime::channel::WordRun> = injects
        .iter()
        .map(|inject| kernels_cuda::runtime::channel::WordRun {
            src: inject.src,
            dst_word: u32::try_from(inject.dst_off / 4).unwrap_or(u32::MAX),
            words: u32::try_from(inject.bytes / 4).unwrap_or(u32::MAX),
        })
        .collect();
    let bytes = table.len() * size_of::<kernels_cuda::runtime::channel::WordRun>();
    // SAFETY: `WordRun` is plain `repr(C)` data, read as its bytes.
    let raw = unsafe { core::slice::from_raw_parts(table.as_ptr().cast::<u8>(), bytes) };
    store.stage(stream, at, raw)?;
    kernels_cuda::runtime::channel::gather_words(
        device.ctx(),
        store.ptr() + at,
        tokens,
        u32::try_from(table.len()).unwrap_or(u32::MAX),
    )?;
    Ok(())
}

/// Packs each lane's device-carried mask into its row of the staged mask at
/// `mask`: past a few lanes in one gather launch over a staged run table,
/// one launch each inside a stream capture.
fn inject_masks(
    device: &crate::device::Context,
    (store, at): (&mut Buffer, u64),
    mask: u64,
    injects: &[MaskInject],
) -> Result<()> {
    let stream = device.stream();
    if injects.len() < GATHERED_INJECTS || crate::device::alloc::is_capturing(stream) {
        for inject in injects {
            kernels_cuda::runtime::channel::gather_bits(
                device.ctx(),
                inject.src,
                mask + inject.dst_off,
                inject.bits,
            )?;
        }
        return Ok(());
    }
    let table: Vec<kernels_cuda::runtime::channel::BitRun> = injects
        .iter()
        .map(|inject| kernels_cuda::runtime::channel::BitRun {
            src: inject.src,
            dst_byte: inject.dst_off,
            bits: inject.bits,
            _pad: 0,
        })
        .collect();
    let bytes = table.len() * size_of::<kernels_cuda::runtime::channel::BitRun>();
    // SAFETY: `BitRun` is plain `repr(C)` data, read as its bytes.
    let raw = unsafe { core::slice::from_raw_parts(table.as_ptr().cast::<u8>(), bytes) };
    store.stage(stream, at, raw)?;
    kernels_cuda::runtime::channel::gather_bit_runs(
        device.ctx(),
        store.ptr() + at,
        mask,
        u32::try_from(table.len()).unwrap_or(u32::MAX),
    )?;
    Ok(())
}

#[derive(Debug)]
pub struct Inputs {
    store: Buffer,
    staging: Vec<Pinned>,
    free: Arc<Free>,
    stage_bytes: u64,
    tokens: u64,
    positions: u64,
    windows: u64,
    window_ints: u64,
    window_slots: crate::window::Slots,
    qo_absolute: u64,
    qo_absolute_ints: u64,
    live_rows: u64,
    live_ints: u64,
    row_valid: u64,
    slot_ids: u64,
    adapter_routes: u64,
    readout_rows: u64,
    readout_live: u64,
    mask_bits: u64,
    mask_bytes: u64,
    mask_indptr: u64,
    mask_origin: u64,
    patch: Option<PatchAt>,
    mrope: Option<u64>,
    mrope_bytes: u64,
    seats: Vec<SeatAt>,
    lane_of_row: u64,
    group_of_lane: u64,
    packings: Vec<PackingAt>,
    page_tables: Vec<TableAt>,
    rs: Option<RsAt>,
    max_lanes: u32,
    max_rows: u32,
    plans: Vec<Option<Workspace>>,
    plan_values: usize,
    /// Each plan's workspace grant in the slab, and the runs it is granted
    /// for: where the slab lies decides their addresses ([`Inputs::lay`]).
    grants: Vec<Option<Grant>>,
    runs: u32,
    /// The staged run table of [`inject_tokens`] in the slab: a run a lane.
    inject_runs: u64,
    /// The staged run table of [`inject_masks`] in the slab: a run a lane.
    mask_runs: u64,
    /// Where the slab holds the fires' group masks, one stretch a ragged
    /// attention: a body captured over them reads each fire's at the same
    /// addresses.
    group_masks: Option<GroupMasksAt>,
    /// The window boundaries the device holds ([`restage`]).
    staged_windows: Vec<u8>,
    /// The mask offsets and the mask words the device holds ([`restage`]):
    /// a fire whose lanes' entries are the same (every lane's "none" step
    /// after step) copies neither.
    staged_mask_indptr: Vec<u8>,
    staged_mask_origin: Vec<u8>,
}

/// The slab's group-mask stretches: stretch `i` (the load's `i`th ragged
/// attention) holds its byte offsets a group from `at + i * stride`, then
/// its bits from `indptr_bytes` further on.
#[derive(Debug, Clone, Copy)]
struct GroupMasksAt {
    at: u64,
    feeds: u64,
    stride: u64,
    indptr_bytes: u64,
    bits_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct GroupMaskHandles {
    pub q: poem_ir::Selection,
    pub kv: poem_ir::Selection,
    pub bits: Tensor,
    pub indptr: Tensor,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        budget: &Budget,
        fire_lanes: u32,
        paging: Paging,
        page_tables: usize,
        facts: &Facts,
        classes: usize,
        regions: usize,
        runs: u32,
        gathered: usize,
        device: Device,
        toggles: Toggles,
        runahead: engine::runahead::Runahead,
        patch: Option<PatchSeat>,
        mrope: bool,
        seats: &[Seat],
        selections: usize,
        masked: bool,
        schedule_streams: &[Option<u32>],
        state_ids: u32,
        rs_page_tokens: Option<u32>,
        group_mask_feeds: usize,
    ) -> Result<Inputs> {
        let rows = u64::from(budget.max_tokens);
        let lanes = u64::from(crate::window::lane_capacity(budget.max_lanes));
        let pages = lanes * u64::from(paging.pages_per_slot);
        let window_slots =
            crate::window::Slots::new(classes, lanes, runs, gathered, rows, page_tables, pages, masked);
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
        let qo_absolute_ints = lanes + 1;
        let qo_absolute = take(qo_absolute_ints * 4);
        let live_seat = crate::window::Seat::new(regions as u64, u64::from(runs.max(1)));
        let live_ints = live_seat.words();
        let live_rows = take(live_ints * 4);
        let row_valid = take(rows);
        let slot_ids = take(lanes * 4 * u64::from(state_ids.max(1)));
        let adapter_routes = take(rows * 4);
        let readout_rows = take(rows * 4);
        let readout_live = take(4);
        let context = u64::from(paging.pages_per_slot) * u64::from(paging.page_size);
        let mask_bytes = if masked {
            (rows * context).div_ceil(8) + lanes
        } else {
            0
        };
        let mask_bits = take(mask_bytes);
        // A lane feeds its tokens, and its mask, from the device in one run.
        let inject_runs = take(lanes * size_of::<kernels_cuda::runtime::channel::WordRun>() as u64);
        let mask_runs = take(lanes * size_of::<kernels_cuda::runtime::channel::BitRun>() as u64);
        let mask_indptr = take((lanes + 1) * 8);
        let mask_origin = take(lanes * 4);
        let page_tables: Vec<TableAt> = (0..page_tables)
            .map(|_| TableAt {
                indptr: take((lanes + 1) * 4),
                indices: take(pages * 4),
                last_page_len: take(lanes * 4),
                kv_len: take(lanes * 4),
                write_page: take(rows * 4),
                write_offset: take(rows * 4),
            })
            .collect();
        let lane_of_row = take(rows * 4);
        let group_of_lane = take(lanes * 4);
        let packings: Vec<PackingAt> = (0..selections)
            .map(|_| PackingAt {
                group_indptr: take((lanes + 1) * 4),
                lane_indptr: take((lanes + 1) * 4),
                permutation: take(rows * 4),
            })
            .collect();
        // A fire replays no more rows than the widest bucket's tokens
        // (`engine::fire::rung_rows`): the tables hold those, and the pages
        // they and its own rows lie in.
        let rs = rs_page_tokens.map(|tokens| {
            let replay_cap = rows;
            let pages_cap = (replay_cap + rows).div_ceil(u64::from(tokens.max(1))) + 2 * lanes;
            RsAt {
                lanes: take(rs_lane_table_bytes(lanes)),
                pages: take(pages_cap * 8),
                pages_cap,
                replay_tok: take(replay_cap * 4),
                replay_cap,
            }
        });
        // A group's mask is a bit a (query, key) pair of its rows, from a
        // byte of its own, and only a group some lane states one for has
        // bits (the rest read "none", `-1`, as data). A stated mask speaks of
        // a row's keys as a paged lane's does, at most the context's a row:
        // the stretch holds the fire's rows by the context (as the paged
        // masks' does), a byte apiece more rounding each group's up; a fire
        // stating more is refused by name (`stage_group_masks`).
        let group_masks = (group_mask_feeds > 0).then(|| {
            let indptr_bytes = ((lanes + 1) * 8).next_multiple_of(ALIGN);
            let bits_bytes = ((rows * context).div_ceil(8) + lanes).next_multiple_of(ALIGN);
            let stride = indptr_bytes + bits_bytes;
            GroupMasksAt {
                at: take(stride * group_mask_feeds as u64),
                feeds: group_mask_feeds as u64,
                stride,
                indptr_bytes,
                bits_bytes,
            }
        });
        let stage_bytes = take(0);
        let seats: Vec<SeatAt> = seats
            .iter()
            .map(|seat| {
                let bytes = if seat.per_lane() { lanes } else { rows } * seat.row_bytes();
                SeatAt {
                    seat: *seat,
                    at: take(bytes),
                    bytes,
                }
            })
            .collect();
        let patch = patch.map(|seat| PatchAt {
            payload: take(seat.rows * seat.row_bytes),
            segments: take((seat.images + 1) * 4),
            routes: take(seat.rows * 4),
            positions: take(seat.rows * AXES * 4),
            grids: take(seat.images * AXES * 4),
            seat,
        });
        let mrope = mrope.then(|| take(rows * AXES * 4));
        let runs = runs.max(1);
        let floats_of = plan_floats(
            budget,
            fire_lanes,
            paging.page_size,
            facts,
            &device,
            toggles,
        )?;
        let mut per_stream: std::collections::BTreeMap<u32, u64> =
            std::collections::BTreeMap::new();
        for (plan, floats) in floats_of.iter().enumerate() {
            if let (Some(floats), Some(Some(stream))) = (floats, schedule_streams.get(plan)) {
                let held = per_stream.entry(*stream).or_insert(0);
                *held = (*held).max(*floats);
            }
        }
        let shared: std::collections::BTreeMap<u32, u64> = per_stream
            .iter()
            .map(|(stream, floats)| (*stream, take(*floats)))
            .collect();
        let grants: Vec<Option<Grant>> = floats_of
            .iter()
            .enumerate()
            .map(|(plan, floats)| {
                floats.map(|floats| {
                    let float_at = match schedule_streams.get(plan) {
                        Some(Some(stream)) => shared[stream],
                        _ => take(floats),
                    };
                    Grant {
                        int_at: (0..runs).map(|_| take(GRANT_INT_BYTES)).collect(),
                        float_at,
                        float_bytes: floats,
                    }
                })
            })
            .collect();
        let total = at;

        // The slab is a run of the activation's span, laid once the load
        // knows what else its fires take there ([`Inputs::lay`]); till then
        // it stands in, as every span of a size-only walk does.
        let store = Buffer::lent(usize::try_from(total).unwrap_or(usize::MAX));
        let depth = runahead.staging_depth();
        let slot_bytes = usize::try_from(stage_bytes).unwrap_or(usize::MAX);
        let mut staging = Vec::with_capacity(depth);
        for _ in 0..depth {
            staging.push(Pinned::mapped(slot_bytes)?);
        }
        let mut inputs = Inputs {
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
            readout_rows,
            readout_live,
            mask_bits,
            mask_bytes,
            mask_indptr,
            mask_origin,
            patch,
            mrope,
            mrope_bytes: if mrope.is_some() { rows * AXES * 4 } else { 0 },
            seats,
            lane_of_row,
            group_of_lane,
            packings,
            page_tables,
            rs,
            max_lanes: crate::window::lane_capacity(budget.max_lanes),
            max_rows: budget.max_tokens,
            inject_runs,
            mask_runs,
            group_masks,
            staged_windows: Vec::new(),
            staged_mask_indptr: Vec::new(),
            staged_mask_origin: Vec::new(),
            plan_values: grants.len(),
            plans: Vec::new(),
            grants,
            runs,
            store,
        };
        // Granted where the slab stands in, as a size-only walk reads them;
        // the slab itself is not laid.
        inputs.grant_at(crate::sizing::STAND_IN);
        Ok(inputs)
    }

    #[must_use]
    pub fn grant(&self, plan: u32, run: u32) -> Option<Workspace> {
        let at = run as usize * self.plan_values + plan as usize;
        self.plans.get(at).copied().flatten()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }

    /// The slab lies at `base` from here on: a run of the activation's
    /// span the load holds backed and zeroed, of [`Inputs::bytes`]. It holds
    /// no table a fire staged before (a size-only walk's, or one the zeroing
    /// cleared), so the next fire copies each.
    pub fn lay(&mut self, base: u64) {
        self.store.lay(base);
        self.staged_windows.clear();
        self.staged_mask_indptr.clear();
        self.staged_mask_origin.clear();
        self.grant_at(base);
    }

    /// Each plan's workspace grant, the slab at `base`.
    fn grant_at(&mut self, base: u64) {
        let grants = &self.grants;
        self.plans = (0..self.runs as usize)
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
            .collect();
    }

    pub fn claim(&self) -> Result<SlotGuard> {
        self.free.claim(
            "staging slots (every one held by a step not yet handed to its landing; the caller \
             is running deeper than the frames_in_flight it loaded with)",
        )
    }

    #[must_use]
    pub fn window_slots(&self) -> crate::window::Slots {
        self.window_slots
    }

    pub fn write_host(&self, slot: &SlotGuard, fire: &Fire<'_>) -> Result<Staged> {
        let rows = fire.tokens.len() as u32;
        let lanes = fire.slot_ids.len() as u32;
        let host = &self.staging[slot.at() as usize];

        rows_fit(self.max_rows, rows, fire)?;

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
        let mut spelled: Option<usize> = None;
        for geometry in fire.page_tables {
            let count = geometry.indptr.len().saturating_sub(1);
            if count != geometry.last_page_len.len()
                || count != geometry.kv_len.len()
                || spelled.is_some_and(|first| first != count)
            {
                return Err(crate::error::Fault::program(
                    "inputs::write_host",
                    format!(
                        "this fire's kv page_tables state {count} lane(s) of page bounds, {} \
                         last-page length(s) and {} kv length(s), and every table of one fire \
                         is built over one lane vector",
                        geometry.last_page_len.len(),
                        geometry.kv_len.len()
                    ),
                ));
            }
            spelled = Some(count);
        }
        let table_lanes = spelled
            .map_or(lanes, |count| count as u32)
            .max(lanes)
            .max(fire.lane_reach);
        if u64::from(table_lanes) > u64::from(self.max_lanes) {
            return Err(crate::error::Fault::Ceiling {
                what: "staged kv lanes",
                need: u64::from(table_lanes),
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
        put(
            self.windows,
            bytes_of(fire.windows),
            "staged window boundaries",
        )?;
        let live = if fire.live_rows == 0 {
            rows
        } else {
            fire.live_rows.min(rows)
        };
        let mut valid = vec![1u8; live as usize];
        valid.resize(rows as usize, 0);
        put(self.row_valid, &valid, "staged row_valid")?;
        if fire.lane_of_row.len() != rows as usize {
            return Err(crate::error::Fault::program(
                "inputs::write_host",
                format!(
                    "this fire stages {} lane-of-row entries for {rows} rows; the map is one \
                     lane per staged row, padding included",
                    fire.lane_of_row.len()
                ),
            ));
        }
        put(
            self.lane_of_row,
            bytes_of(fire.lane_of_row),
            "staged lane of row",
        )?;
        put(self.slot_ids, bytes_of(fire.state_ids), "staged slot ids")?;
        let tabled = fire.state_ids.len() != fire.slot_ids.len();
        let state_ids = if tabled {
            fire.state_ids.len() as u32
        } else {
            table_lanes
        };
        if table_lanes > lanes && !tabled {
            let inert = vec![-1i32; (table_lanes - lanes) as usize];
            put(
                self.slot_ids + u64::from(lanes) * 4,
                bytes_of(&inert),
                "staged slot id padding",
            )?;
        }

        let qo_absolute = if fire.qo_absolute.is_empty() {
            None
        } else {
            put(
                self.qo_absolute,
                bytes_of(fire.qo_absolute),
                "staged absolute qo bounds",
            )?;
            Some(fire.qo_absolute.len())
        };

        let live = if fire.live.is_empty() {
            None
        } else {
            put(self.live_rows, u32_bytes_of(fire.live), "staged live rows")?;
            Some(fire.live.len())
        };

        let adapter_rows = match fire.adapter_routes {
            None => None,
            Some(routes) => {
                put(
                    self.adapter_routes,
                    bytes_of(routes),
                    "staged adapter routes",
                )?;
                Some(routes.len() as u32)
            }
        };

        put(
            self.readout_rows,
            bytes_of(fire.readout_rows),
            "staged readout rows",
        )?;
        put(
            self.readout_live,
            &fire.readout_live.to_le_bytes(),
            "staged live readouts",
        )?;

        let mask_bytes = match fire.mask {
            None => None,
            Some(staged) => {
                put(self.mask_bits, &staged.bits, "staged mask bits")?;
                put(
                    self.mask_indptr,
                    i64_bytes_of(&staged.indptr),
                    "staged mask indptr",
                )?;
                put(
                    self.mask_origin,
                    u32_bytes_of(&staged.origins),
                    "staged mask origins",
                )?;
                Some(u32::try_from(staged.bits.len()).map_err(|_| {
                    crate::error::Fault::program(
                        "inputs::write_host",
                        format!(
                            "this fire stages {} bytes of mask bits, past the {} one fire's \
                             mask states",
                            staged.bits.len(),
                            u32::MAX
                        ),
                    )
                })?)
            }
        };

        let packed = !fire.group_of_lane.is_empty();
        if packed {
            if fire.packings.len() != self.packings.len() {
                return Err(crate::error::Fault::program(
                    "inputs::write_host",
                    format!(
                        "this fire stages {} packing(s) and the load carved {}",
                        fire.packings.len(),
                        self.packings.len()
                    ),
                ));
            }
            let mut groups = fire.group_of_lane.to_vec();
            groups.resize(table_lanes as usize, -1);
            put(self.group_of_lane, bytes_of(&groups), "staged group ids")?;
            let mut lane_table: Vec<i32> = Vec::with_capacity(table_lanes as usize + 1);
            let mut row_table: Vec<i32> = Vec::with_capacity(rows as usize);
            for (at, packing) in self.packings.iter().zip(fire.packings) {
                for (offset, table, what) in [
                    (at.group_indptr, packing.group_indptr, "staged group bounds"),
                    (
                        at.lane_indptr,
                        packing.lane_indptr,
                        "staged packed lane bounds",
                    ),
                ] {
                    lane_table.clear();
                    lane_table.extend_from_slice(table);
                    let last = table.last().copied().unwrap_or(0);
                    lane_table.resize(table_lanes as usize + 1, last);
                    put(offset, bytes_of(&lane_table), what)?;
                }
                row_table.clear();
                row_table.extend_from_slice(packing.permutation);
                row_table.resize(rows as usize, -1);
                put(
                    at.permutation,
                    bytes_of(&row_table),
                    "staged row permutation",
                )?;
            }
        }

        let rs = match (fire.rs, self.rs) {
            (None, _) => None,
            (Some(_), None) => {
                return Err(crate::error::Fault::program(
                    "inputs::write_host",
                    "a fire buffers recurrent rows on a load that carved no rs tables",
                ));
            }
            (Some(tables), Some(at)) => {
                if tables.pages.len() as u64 > at.pages_cap
                    || tables.replay_tok.len() as u64 > at.replay_cap
                {
                    return Err(crate::error::Fault::Ceiling {
                        what: "rs buffer pages and replayed rows staged",
                        need: (tables.pages.len() as u64).max(tables.replay_tok.len() as u64),
                        have: at.pages_cap.min(at.replay_cap),
                    });
                }
                let mut lanes_tab = tables.lanes.to_vec();
                lanes_tab.resize(
                    (table_lanes as usize).max(tables.lanes.len()),
                    kernels_cuda::runtime::channel::RsLane::INERT,
                );
                // SAFETY: `RsLane` is plain `repr(C)` data, read as its bytes.
                let raw = unsafe {
                    core::slice::from_raw_parts(
                        lanes_tab.as_ptr().cast::<u8>(),
                        rs_lane_table_bytes(lanes_tab.len() as u64) as usize,
                    )
                };
                put(at.lanes, raw, "staged rs lanes")?;
                // SAFETY: plain integers, read as their bytes.
                let pages = unsafe {
                    core::slice::from_raw_parts(
                        tables.pages.as_ptr().cast::<u8>(),
                        tables.pages.len() * 8,
                    )
                };
                put(at.pages, pages, "staged rs pages")?;
                put(
                    at.replay_tok,
                    u32_bytes_of(tables.replay_tok),
                    "staged rs replay rows",
                )?;
                Some((tables.pages.len() as u32, tables.replay_tok.len() as u32))
            }
        };

        let mut table_indices = Vec::with_capacity(self.page_tables.len());
        for (at, geometry) in self.page_tables.iter().zip(fire.page_tables) {
            put(at.indptr, bytes_of(&geometry.indptr), "staged kv indptr")?;
            put(
                at.indices,
                bytes_of(&geometry.indices),
                "staged kv page ids",
            )?;
            put(
                at.last_page_len,
                bytes_of(&geometry.last_page_len),
                "staged last page len",
            )?;
            put(at.kv_len, bytes_of(&geometry.kv_len), "staged kv len")?;
            put(
                at.write_page,
                bytes_of(&geometry.write_page),
                "staged write page",
            )?;
            put(
                at.write_offset,
                bytes_of(&geometry.write_offset),
                "staged write offset",
            )?;
            table_indices.push(geometry.indices.len() as u32);
        }

        Ok(Staged {
            rows,
            lanes,
            windows: fire.windows.len(),
            qo_absolute,
            live,
            adapter_rows,
            readouts: fire.readout_rows.len() as u32,
            mask_bytes,
            table_indices,
            table_lanes,
            state_ids,
            packed,
            rs,
        })
    }

    #[must_use]
    pub fn tokens_at(&self) -> u64 {
        self.store.ptr() + self.tokens
    }

    #[must_use]
    pub fn seat(&self, id: u32, rows: u32) -> Option<Tensor> {
        let at = self.seats.iter().find(|at| at.seat.id == id)?;
        let rows =
            rows.min(u32::try_from(at.bytes / at.seat.row_bytes().max(1)).unwrap_or(u32::MAX));
        Some(Tensor::new(
            self.store.ptr() + at.at,
            rows,
            at.seat.width,
            at.seat.dtype,
        ))
    }

    #[must_use]
    pub fn guest_seats(&self) -> Vec<Seat> {
        self.seats.iter().map(|at| at.seat).collect()
    }

    pub fn stage_patches(
        &mut self,
        stream: *mut core::ffi::c_void,
        payload: &[u8],
        segments: &[i32],
        routes: &[i32],
        positions: &[i32],
        grids: &[i32],
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
            (grids.len() as u64 * 4, at.seat.images * AXES * 4),
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
        self.store
            .stage(stream, at.positions, bytes_of(positions))?;
        self.store.stage(stream, at.grids, bytes_of(grids))?;
        let rows = (payload.len() as u64)
            .checked_div(at.seat.row_bytes)
            .unwrap_or(0) as u32;
        let width = poem_compiler::arena::elem_bytes(at.seat.dtype)
            .filter(|element| *element > 0)
            .map_or(0, |element| (at.seat.row_bytes / element) as u32);
        Ok(PatchHandles {
            patches: Tensor::new(base + at.payload, rows, width, at.seat.dtype),
            segments: i32s(base + at.segments, segments.len() as u32),
            routes: i32s(base + at.routes, routes.len() as u32),
            positions: Tensor::new(
                base + at.positions,
                (positions.len() / AXES as usize) as u32,
                AXES as u32,
                Dtype::I32,
            ),
            grids: Tensor::new(
                base + at.grids,
                (grids.len() / AXES as usize) as u32,
                AXES as u32,
                Dtype::I32,
            ),
        })
    }

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

    #[allow(
        clippy::not_unsafe_ptr_arg_deref,
        reason = "`stream` is an opaque CUDA handle the host never dereferences"
    )]
    pub fn commit(
        &mut self,
        device: &crate::device::Context,
        slot: &SlotGuard,
        staged: &Staged,
        token_injects: &[TokenInject],
        mask_injects: &[MaskInject],
    ) -> Result<Handles> {
        let Staged {
            rows,
            lanes,
            windows,
            qo_absolute,
            live,
            adapter_rows,
            readouts,
            mask_bytes,
            table_indices,
            table_lanes,
            state_ids,
            packed,
            rs,
        } = staged;
        let readouts = *readouts;
        let (rows, lanes) = (*rows, *lanes);
        let packed = *packed;
        let table_lanes = *table_lanes;
        let base = self.store.ptr();
        let (at_tokens, at_positions, at_windows) = (self.tokens, self.positions, self.windows);
        let at_qo_absolute = self.qo_absolute;
        let at_live = self.live_rows;
        let (at_row_valid, at_slot_ids) = (self.row_valid, self.slot_ids);
        let (at_routes, at_mask, at_mask_indptr, at_mask_origin) = (
            self.adapter_routes,
            self.mask_bits,
            self.mask_indptr,
            self.mask_origin,
        );
        let at_readouts = self.readout_rows;
        let at_readout_live = self.readout_live;
        let places: Vec<TableAt> = self.page_tables.clone();
        let at_lane_of_row = self.lane_of_row;
        let at_group_of_lane = self.group_of_lane;
        let packing_places: Vec<PackingAt> = self.packings.clone();
        let rs_at = self.rs;
        let stream = device.stream();
        let capturing = crate::device::alloc::is_capturing(stream);
        // The run tables hold a run a lane.
        let fed = token_injects.len().max(mask_injects.len()) as u64;
        if fed > u64::from(self.max_lanes) {
            return Err(crate::error::Fault::Ceiling {
                what: "lanes fed from the device in one fire",
                need: fed,
                have: u64::from(self.max_lanes),
            });
        }
        let Inputs {
            store,
            staging,
            inject_runs,
            mask_runs,
            staged_windows,
            staged_mask_indptr,
            staged_mask_origin,
            ..
        } = self;
        let host = staging[slot.at() as usize].host();

        let mut spans: Vec<(u64, *const u8, usize)> = Vec::with_capacity(24);
        let mut copy = |offset: u64, len: usize| -> Result<()> {
            spans.push((offset, host.wrapping_add(offset as usize), len));
            Ok(())
        };

        copy(at_tokens, rows as usize * 4)?;
        copy(at_positions, rows as usize * 4)?;
        // SAFETY: the slot's pinned bytes, written for this fire by write_host.
        let window_bytes =
            unsafe { std::slice::from_raw_parts(host.add(at_windows as usize), windows * 4) };
        if restage(staged_windows, window_bytes, capturing) {
            copy(at_windows, windows * 4)?;
        }
        if let Some(bounds) = qo_absolute {
            copy(at_qo_absolute, *bounds * 4)?;
        }
        if let Some(words) = live {
            copy(at_live, *words * 4)?;
        }
        copy(at_row_valid, rows as usize)?;
        copy(at_lane_of_row, rows as usize * 4)?;
        copy(at_readouts, readouts as usize * 4)?;
        copy(at_readout_live, 4)?;
        copy(at_slot_ids, *state_ids as usize * 4)?;
        if let Some(routes) = adapter_rows {
            copy(at_routes, *routes as usize * 4)?;
        }
        if let Some(bytes) = mask_bytes {
            copy(at_mask, *bytes as usize)?;
            let (indptr_len, origin_len) = ((lanes as usize + 1) * 8, lanes as usize * 4);
            // SAFETY: the slot's pinned bytes, written for this fire by write_host.
            let (indptr, origin) = unsafe {
                (
                    std::slice::from_raw_parts(host.add(at_mask_indptr as usize), indptr_len),
                    std::slice::from_raw_parts(host.add(at_mask_origin as usize), origin_len),
                )
            };
            if restage(staged_mask_indptr, indptr, capturing) {
                copy(at_mask_indptr, indptr_len)?;
            }
            if restage(staged_mask_origin, origin, capturing) {
                copy(at_mask_origin, origin_len)?;
            }
        }
        let mut packings = Vec::with_capacity(packing_places.len());
        if packed {
            copy(at_group_of_lane, table_lanes as usize * 4)?;
            for at in &packing_places {
                copy(at.group_indptr, (table_lanes as usize + 1) * 4)?;
                copy(at.lane_indptr, (table_lanes as usize + 1) * 4)?;
                copy(at.permutation, rows as usize * 4)?;
                packings.push(PackingHandles {
                    group_indptr: i32s(base + at.group_indptr, table_lanes + 1),
                    lane_indptr: i32s(base + at.lane_indptr, table_lanes + 1),
                    permutation: i32s(base + at.permutation, rows),
                });
            }
        }
        let mut page_tables = Vec::with_capacity(places.len());
        for (at, indices) in places.iter().zip(table_indices) {
            copy(at.indptr, (table_lanes as usize + 1) * 4)?;
            copy(at.indices, *indices as usize * 4)?;
            copy(at.last_page_len, table_lanes as usize * 4)?;
            copy(at.kv_len, table_lanes as usize * 4)?;
            copy(at.write_page, rows as usize * 4)?;
            copy(at.write_offset, rows as usize * 4)?;
            page_tables.push(TableHandles {
                indptr: i32s(base + at.indptr, table_lanes + 1),
                indices: i32s(base + at.indices, *indices),
                last_page_len: i32s(base + at.last_page_len, table_lanes),
                kv_len: i32s(base + at.kv_len, table_lanes),
                write_page: i32s(base + at.write_page, rows),
                write_offset: i32s(base + at.write_offset, rows),
            });
        }

        let rs = match (rs, rs_at) {
            (Some((pages, toks)), Some(at)) => {
                // A lane's whole `RsLane`, its extended-run place with it:
                // a shorter copy leaves the last lanes' tails as an earlier
                // fire staged them.
                copy(
                    at.lanes,
                    rs_lane_table_bytes(u64::from(table_lanes.max(lanes))) as usize,
                )?;
                copy(at.pages, *pages as usize * 8)?;
                copy(at.replay_tok, *toks as usize * 4)?;
                Some(RsHandles {
                    lanes: base + at.lanes,
                    pages: base + at.pages,
                    replay_tok: base + at.replay_tok,
                })
            }
            _ => None,
        };

        // SAFETY: the sources are the slot's pinned bytes, held until the `SlotGuard` drops.
        unsafe { store.stage_batch_from(stream, &spans)? };

        inject_tokens(
            device,
            (&mut *store, *inject_runs),
            base + at_tokens,
            token_injects,
        )?;
        if !mask_injects.is_empty() && mask_bytes.is_none() {
            return Err(crate::error::Fault::program(
                "inputs::commit",
                "a lane's attention mask is carried on the device and the fire stages no mask \
                 to pack it into",
            ));
        }
        // Over the cleared rows just staged, in stream order.
        inject_masks(
            device,
            (&mut *store, *mask_runs),
            base + at_mask,
            mask_injects,
        )?;

        Ok(Handles {
            tokens: i32s(base + at_tokens, rows),
            positions: i32s(base + at_positions, rows),
            windows: base + at_windows,
            qo_absolute: qo_absolute.map(|_| base + at_qo_absolute),
            live_rows: live.map(|_| base + at_live),
            page_tables,
            slot_ids: i32s(base + at_slot_ids, table_lanes),
            adapter_routes: adapter_rows.map(|rows| i32s(base + at_routes, rows)),
            readout_rows: i32s(base + at_readouts, readouts),
            readout_live: base + at_readout_live,
            row_valid: Tensor::new(base + at_row_valid, rows, 1, Dtype::U8),
            mask: mask_bytes.map(|bytes| Tensor::new(base + at_mask, bytes, 1, Dtype::U8)),
            mask_indptr: mask_bytes.map(|_| i64s(base + at_mask_indptr, lanes + 1)),
            mask_origin: mask_bytes
                .map(|_| Tensor::new(base + at_mask_origin, lanes, 1, Dtype::U32)),
            group_of_lane: packed.then(|| i32s(base + at_group_of_lane, table_lanes)),
            packings,
            lane_of_row: i32s(base + at_lane_of_row, rows),
            rs,
        })
    }

    /// Stages a fire's group masks, the `i`th ragged attention's into the
    /// slab's `i`th stretch: its byte offsets a group, then its bits, at the
    /// addresses every fire's take (a body reads them as data). The offsets
    /// run to `table_lanes + 1` entries, as the fire's group bounds do
    /// (`Staged::table_lanes`): the groups past the fire's last are empty
    /// and repeat its last offset, so the kernel's table is one entry a
    /// staged group.
    pub fn stage_group_masks(
        &mut self,
        stream: *mut core::ffi::c_void,
        masks: &[(
            poem_ir::Selection,
            poem_ir::Selection,
            poem_exec::fire::GroupMask,
        )],
        table_lanes: u32,
    ) -> Result<Vec<GroupMaskHandles>> {
        if masks.is_empty() {
            return Ok(Vec::new());
        }
        let Some(at) = self.group_masks else {
            return Err(crate::error::Fault::Ceiling {
                what: "group mask stretches, which this load reserved none of (no class it \
                       serves reads a lane's group mask)",
                need: masks.len() as u64,
                have: 0,
            });
        };
        if masks.len() as u64 > at.feeds {
            return Err(crate::error::Fault::Ceiling {
                what: "group mask stretches this load reserved, one a ragged attention",
                need: masks.len() as u64,
                have: at.feeds,
            });
        }
        let offsets = u64::from(table_lanes) + 1;
        let base = self.store.ptr();
        let mut handles = Vec::with_capacity(masks.len());
        for (slot, (q, kv, mask)) in masks.iter().enumerate() {
            // The kernels read a mask's offsets as 64-bit.
            let mut wide: Vec<i64> = mask.indptr.iter().map(|&at| i64::from(at)).collect();
            let last = wide.last().copied().unwrap_or(0);
            wide.resize(wide.len().max(offsets as usize), last);
            let indptr_need = wide.len() as u64 * 8;
            let bits_need = mask.bits.len() as u64;
            if indptr_need > at.indptr_bytes || bits_need > at.bits_bytes {
                return Err(crate::error::Fault::Ceiling {
                    what: "bytes of a group mask stretch this load reserved",
                    need: indptr_need + bits_need,
                    have: at.indptr_bytes + at.bits_bytes,
                });
            }
            let indptr = at.at + slot as u64 * at.stride;
            let bits = indptr + at.indptr_bytes;
            self.store.stage(stream, indptr, i64_bytes_of(&wide))?;
            self.store.stage(stream, bits, &mask.bits)?;
            handles.push(GroupMaskHandles {
                q: *q,
                kv: *kv,
                bits: Tensor::new(
                    base + bits,
                    u32::try_from(at.bits_bytes).unwrap_or(u32::MAX),
                    1,
                    Dtype::U8,
                ),
                indptr: i64s(base + indptr, u32::try_from(wide.len()).unwrap_or(u32::MAX)),
            });
        }
        Ok(handles)
    }

    #[must_use]
    pub fn seats(
        &self,
        handles: &Handles,
        pages: u32,
        rows: u32,
        lanes: u32,
    ) -> crate::store::Seats {
        crate::store::Seats {
            lanes,
            rows,
            pages,
            page_tables: handles
                .page_tables
                .iter()
                .map(|table| TableSeat {
                    page_indptr: table.indptr,
                    page_indices: table.indices,
                    last_page_lens: table.last_page_len,
                    row_valid: handles.row_valid,
                })
                .collect(),
            slot_ids: handles.slot_ids,
            write_state: true,
            write_state_mask: Tensor::ABSENT,
            commit_len: Tensor::ABSENT,
            begin_at: Tensor::ABSENT,
            snapshots: kernels_cuda::Snapshots::NONE,
        }
    }
}

/// Whether a fire copies a table whose bytes are `bytes`, keeping `held` as
/// the bytes the device holds for it. A copy captured into a graph lands
/// only when the graph runs, so it leaves nothing held: the next fire copies
/// the table again.
fn restage(held: &mut Vec<u8>, bytes: &[u8], capturing: bool) -> bool {
    if !capturing && held.as_slice() == bytes {
        return false;
    }
    held.clear();
    if !capturing {
        held.extend_from_slice(bytes);
    }
    true
}

fn i32s(ptr: u64, rows: u32) -> Tensor {
    Tensor::new(ptr, rows, 1, Dtype::I32)
}

fn i64s(ptr: u64, rows: u32) -> Tensor {
    Tensor::new(ptr, rows, 1, Dtype::I64)
}

fn i64_bytes_of(values: &[i64]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `i64` is `Copy`, no padding or niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn bytes_of(values: &[i32]) -> &[u8] {
    // SAFETY: `i32` is `Copy` with no padding/niche, so all `4 * len` bytes are initialized. Result borrows the input, read-only, for one enqueue.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn u32_bytes_of(values: &[u32]) -> &[u8] {
    // SAFETY: as [`bytes_of`] — `u32` is `Copy`, no padding or niche.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod rows_fit_tests {
    use super::{Fire, Geometry, rows_fit};

    fn fire<'a>(tokens: &'a [i32], tables: &'a [Geometry]) -> Fire<'a> {
        Fire {
            tokens,
            positions: &[],
            windows: &[],
            qo_absolute: &[],
            live: &[],
            slot_ids: &[],
            state_ids: &[],
            adapter_routes: None,
            readout_rows: &[],
            readout_live: 0,
            page_tables: tables,
            mask: None,
            live_rows: 0,
            lane_of_row: &[],
            lane_reach: 0,
            group_of_lane: &[],
            packings: &[],
            rs: None,
        }
    }

    /// A fire past its regions' rows, in its tokens or in a kv write table,
    /// is refused by name; one within them passes.
    #[test]
    fn token_rows_past_the_regions_are_refused_by_name() {
        let tables = [Geometry {
            write_page: vec![0; 4],
            write_offset: vec![0; 4],
            ..Geometry::default()
        }];
        assert!(rows_fit(4, 4, &fire(&[0; 4], &tables)).is_ok());
        let wide = rows_fit(4, 5, &fire(&[0; 5], &tables)).unwrap_err();
        assert!(
            wide.to_string()
                .contains("token rows a fire stages exceed its rung's regions"),
            "{wide}"
        );
        let long = [Geometry {
            write_page: vec![0; 5],
            ..Geometry::default()
        }];
        assert!(rows_fit(4, 4, &fire(&[0; 4], &long)).is_err());
    }
}

#[cfg(test)]
mod rs_lane_table_tests {
    use super::rs_lane_table_bytes;
    use kernels_cuda::runtime::channel::RsLane;

    /// The staged lane table is the lanes' `RsLane`s byte for byte: what the
    /// slab lays and a fire copies is what the table's lanes take, every
    /// field to the last lane's `ext`.
    #[test]
    fn the_staged_rs_lane_table_is_its_lanes_whole() {
        let lanes = [RsLane::INERT; 67];
        assert_eq!(
            rs_lane_table_bytes(lanes.len() as u64) as usize,
            std::mem::size_of_val(&lanes)
        );
        let last = std::ptr::addr_of!(lanes[66].ext) as usize - lanes.as_ptr() as usize;
        assert!(last + 4 <= rs_lane_table_bytes(lanes.len() as u64) as usize);
    }
}

#[cfg(test)]
mod prefill_grant_tests {
    use super::{GRANT_INT_BYTES, plan_floats};
    use crate::store::kv::{Facts, ScheduleFacts, SpaceFacts};
    use kernels_cuda::attn::plan::{Device, Live, Shape, Toggles, Workspace, plan_prefill};
    use kernels_cuda::tuning::Tuning;
    use poem_compiler::Budget;
    use poem_ir::StructKind;

    /// Qwen3-0.6B's attention: 16 query heads over 8 kv heads, 128 wide.
    fn qwen3_0_6b() -> Facts {
        Facts {
            rows: Vec::new(),
            plans: vec![Some(ScheduleFacts {
                kind: StructKind::AttnPrefillPlan,
                reading: SpaceFacts {
                    head_dim: 128,
                    kv_heads: 8,
                    q_heads: 16,
                    window: None,
                },
                rows_a_lane: false,
            })],
        }
    }

    /// Every body a lattice arms plans its graph form inside the prefill
    /// grant: each bucket's rows over every lane count a fire carries,
    /// the rows on one long lane or spread a row a lane, through the plan
    /// the capture calls. At a 4 GiB budget of Qwen3-0.6B on an L40S (a
    /// 512-row top rung, 512 lanes seated, 256 a fire) the grant was sized
    /// at the top rung over the 512 seated lanes, whose 16-row tiles run
    /// unsplit (the 64 MiB floor), while the 280..512-row rungs over 256
    /// lanes split on 64- and 128-row tiles and asked up to 132 MiB:
    /// every such prefill, mixed and wide body walked eagerly for the load.
    #[test]
    fn every_armed_prefill_body_plans_its_graph_form_inside_the_grant() {
        let facts = qwen3_0_6b();
        let device = Device::L40S;
        let toggles = Toggles::of(Tuning::DEFAULT);
        for (top, max_lanes) in [(512, 512), (2048, 512), (2048, 128)] {
            let budget = Budget {
                max_lanes,
                max_tokens: top,
                buckets: crate::api::default_lattice(top, 1, 256, 8),
                max_adapters: 0,
                readouts_per_lane: 1,
            };
            let lanes = crate::chip::Engine::of(&crate::chip::profile("")).fire_lanes_of(max_lanes);
            let grant =
                plan_floats(&budget, lanes, 16, &facts, &device, toggles).unwrap()[0].unwrap();
            for &rows in &budget.buckets {
                for lanes in 1..=lanes {
                    let live = lanes.min(rows);
                    for spread in [false, true] {
                        let mut qo = vec![0i32];
                        for lane in 0..live {
                            let len = if spread || lane > 0 {
                                1
                            } else {
                                rows - (live - 1)
                            };
                            qo.push(qo[lane as usize] + len as i32);
                        }
                        if spread {
                            *qo.last_mut().unwrap() = rows as i32;
                        }
                        let kv: Vec<i32> = (0..=live as i32).map(|lane| lane * 128).collect();
                        plan_prefill(
                            &qo,
                            &kv,
                            &[],
                            rows,
                            Shape {
                                num_requests: lanes,
                                lane_offset: 0,
                                num_q_heads: 16,
                                num_kv_heads: 8,
                                head_dim: 128,
                                page_size: 16,
                                hnd_layout: false,
                            },
                            Live {
                                requests: live,
                                lane_offset: 0,
                                row_offset: 0,
                                rows,
                            },
                            None,
                            true,
                            true,
                            None,
                            &device,
                            toggles,
                            Workspace {
                                int_ptr: 0,
                                int_bytes: GRANT_INT_BYTES as usize,
                                float_ptr: 0,
                                float_bytes: grant as usize,
                            },
                        )
                        .unwrap_or_else(|declined| {
                            panic!(
                                "top {top} over {max_lanes} lanes: b{rows} over {lanes} lanes \
                                 declines its graph form in a {grant}-byte grant: {declined}"
                            )
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod claim_tests {
    use super::Free;
    use std::time::{Duration, Instant};

    /// Every slot taken, one held by its step's landing: a claim waits for
    /// the landing to give it back, however long that takes, and then has it.
    #[test]
    fn a_full_set_with_a_landing_pending_waits_and_then_claims() {
        let free = Free::of(2);
        let first = free.claim("slots").expect("a free slot");
        let landing = free.claim("slots").expect("a free slot").landing();
        let at = landing.at();
        let lands = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(landing);
        });
        let began = Instant::now();
        let got = free.claim("slots").expect("the landing's slot comes back");
        assert!(began.elapsed() >= Duration::from_millis(150), "it waited for the landing");
        assert_eq!(got.at(), at, "the landing's slot");
        lands.join().expect("the landing thread");
        drop(first);
    }

    /// Every slot held by the claimant's own steps, none landing: no slot
    /// can come back until the claimant lands one, so the claim is refused
    /// by name at once, never after a timer.
    #[test]
    fn a_claim_deeper_than_the_set_is_refused_at_once_by_name() {
        let free = Free::of(2);
        let held = [
            free.claim("slots").expect("a free slot"),
            free.claim("slots").expect("a free slot"),
        ];
        let began = Instant::now();
        let refused = free
            .claim("staging slots (the caller runs deeper than its frames in flight)")
            .expect_err("a third slot of two, none landing");
        assert!(began.elapsed() < Duration::from_millis(100), "refused at once");
        assert!(refused.to_string().contains("staging slots"), "{refused}");
        drop(held);
    }
}

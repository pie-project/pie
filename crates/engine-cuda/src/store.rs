#![allow(unsafe_code)]
pub mod chunks;
pub mod kv;
pub mod layout;
pub mod rs;

use engine::transfer::KvRun;
use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use poem_ir::{CacheRow, Def, Dtype, Operands, Trace};

use crate::device::elastic::{self, Arena, Commit, PhysicalPool};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable, PoolSlabs};
use crate::settle::Airborne;
use crate::store::kv::{Facts, Paging};
use crate::store::layout::Kind;
pub use engine::fit::{balanced_state_slots, least_state_slots, pages_within, state_slots_within};

impl From<poem_exec::store::Fault> for Fault {
    fn from(fault: poem_exec::store::Fault) -> Fault {
        match fault {
            poem_exec::store::Fault::Ceiling { what, need, have } => {
                Fault::Ceiling { what, need, have }
            }
            poem_exec::store::Fault::Unbound { what } => Fault::Unbound { what },
            poem_exec::store::Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            } => Fault::Straddled {
                value,
                node,
                planned,
                consumed,
            },
        }
    }
}

const NHD: i32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounting {
    pub card: u64,
    pub ceiling: u64,
    pub weights: u64,
    pub floor: u64,
    pub pool: u64,
    pub minimum: u64,
}

impl Accounting {
    #[must_use]
    pub fn of(card: u64, utilization: f64, weights: u64, minimum: u64) -> Accounting {
        let fraction = if utilization.is_finite() {
            utilization.clamp(0.0, 1.0)
        } else {
            1.0
        };
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a byte count of a GPU card is far inside f64's exact integer range, \
                      and the product is floored back into u64 deliberately"
        )]
        let ceiling = (card as f64 * fraction) as u64;
        let floor = elastic::safety_floor_bytes(card);
        Accounting {
            card,
            ceiling,
            weights,
            floor,
            pool: ceiling.saturating_sub(weights).saturating_sub(floor),
            minimum,
        }
    }

    pub fn admit(&self) -> Result<()> {
        if self.pool >= self.minimum {
            return Ok(());
        }
        Err(Fault::Residency(format!(
            "the card does not hold this deployment: {card} bytes on the device, of which \
             `[model] device_memory_share` allows pie {ceiling}; this load's weight tier \
             takes {weights} and the driver's safety floor holds back {floor}, leaving the \
             elastic pool {pool} bytes — and the cache rows need {minimum}: `[model] \
             device_kv_budget`, or one sequence at the declared context when that is \
             omitted. weight tier + elastic pool + safety \
             floor must fit inside the fraction of the card, and here they do not. Lower \
             `[model] max_context` or `[model] slots`, raise `[model] \
             device_memory_share`, or state a `[model] device_weight_budget` that streams \
             the weight tier down.",
            card = self.card,
            ceiling = self.ceiling,
            weights = self.weights,
            floor = self.floor,
            pool = self.pool,
            minimum = self.minimum,
        )))
    }
}

/// What `Pools::fit_the_card` did when the declared pool was larger than
/// the card: the page count asked, the count it was sized to, the bytes the
/// card had for the cache rows, and what was held back from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fitted {
    pub asked: u64,
    pub fit: u64,
    pub room: u64,
    pub held: u64,
    pub bodies: u64,
    pub spare: u64,
    pub slots: u32,
}

/// The least `[engine] bodies_mem` is held down to on a card with little
/// room: enough for the decode bodies, which are what a serve replays every
/// step and cost tens of megabytes each.
pub const BODIES_FLOOR_BYTES: u64 = 512 << 20;

/// What the graph bodies may take on this card: the configured allowance,
/// or a quarter of `room` — what is left for them and the cache rows once
/// one sequence and the guests' programs are seated — whichever is less,
/// but not under the floor while the room affords it. A fixed 4 GiB on a
/// 24 GB card is a sixth of the device before a second sequence seats.
#[must_use]
pub fn bodies_allowance(configured: u64, room: u64) -> u64 {
    configured.min(room).min((room / 4).max(BODIES_FLOOR_BYTES))
}

/// What is held for the bodies and the guests' programs together on a
/// card with `spare` bytes past one sequence, in the order they yield: the
/// bodies' floor first, since a load with no armed body walks every step
/// eagerly and that is no serve, then the programs, then the bodies' share
/// of what is left. Returns `(bodies, programs)`.
#[must_use]
pub fn holds_within(configured: u64, programs: u64, spare: u64) -> (u64, u64) {
    let floor = configured.min(BODIES_FLOOR_BYTES).min(spare);
    let programs = programs.min(spare.saturating_sub(floor));
    let bodies = bodies_allowance(configured, spare.saturating_sub(programs)).max(floor);
    (bodies, programs)
}

/// The largest token budget at or under `max_tokens`, halving toward
/// `floor`, whose working set (`working`, the activation arena and the
/// attention workspaces at that budget) fits `room`. Halving keeps the
/// bucket lattice a prefix of what the operator asked for.
#[must_use]
pub fn tokens_within(max_tokens: u32, floor: u32, room: u64, working: impl Fn(u32) -> u64) -> u32 {
    let mut tokens = max_tokens;
    while tokens > floor && working(tokens) > room {
        tokens = (tokens / 2).max(floor);
    }
    tokens
}

/// The rows a guest program's epilogue is held per lane, each as wide as the
/// readout rows: the logits it reads. The scaled, probability, mask, noise
/// and Gumbel rows a top-p sampling program derives from them (6292224
/// bytes a lane at a 262144 vocabulary, measured) fly it narrower where the
/// card has no room for them.
pub const PROGRAM_ROWS_A_LANE: u64 = 1;

/// What one guest program's scratch takes at the admitted lane count. A
/// program is registered after the load, so its stride is not known when
/// the cache rows are declared; what is known is that its per-lane scratch
/// is laid out over readout rows, that the shell sizes it for the
/// lane count rounded up to a power of two, and that the program contract
/// caps the whole at `eta_exec::SCRATCH_MAX_BYTES`. The pool leaves this
/// much on the card so the first program to arrive is not the frame that
/// finds the pool has taken everything.
#[must_use]
pub fn program_scratch_reserve(max_lanes: u32, out_row_bytes: u64) -> u64 {
    let lanes = u64::from(
        max_lanes
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(u32::MAX),
    );
    let row = out_row_bytes
        .checked_next_multiple_of(eta_exec::SCRATCH_ALIGN)
        .unwrap_or(u64::MAX);
    lanes
        .saturating_mul(row)
        .saturating_mul(PROGRAM_ROWS_A_LANE)
        .min(eta_exec::SCRATCH_MAX_BYTES)
}

/// What the decoded-weight tiles take on this load: a plane stored in codes
/// and scales is decoded to bf16 before a prefill's dense gemm reads it,
/// into one tile a stream the size of the widest plane fired on it, in the
/// 8 MiB grain the scratch grows by. The pool holds this out for them as it
/// does for the guests' programs, since the tiles are taken at fire time.
#[must_use]
pub fn decoded_weight_reserve(widest_plane_bytes: u64, streams: u32) -> u64 {
    const GRAIN: u64 = 8 << 20;
    widest_plane_bytes
        .checked_next_multiple_of(GRAIN)
        .unwrap_or(u64::MAX)
        .saturating_mul(u64::from(streams.max(1)))
}

/// What a host pool of `budget` bytes seats: one rs row per device slot
/// first (`rows` of `row_bytes`), then kv pages (`pages` of `page_bytes`),
/// then windowed pages (`window_pages` of `window_page_bytes`), then more
/// kv pages with the rest, which can be more than the device holds. Each is
/// what the budget holds of it, so zero seats nothing.
#[must_use]
pub fn host_pool(
    budget: u64,
    rows: u32,
    row_bytes: u64,
    pages: u64,
    page_bytes: u64,
    window_pages: u64,
    window_page_bytes: u64,
) -> (u32, u32, u32) {
    let clamp = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
    let rows = u64::from(rows).min(budget.checked_div(row_bytes).unwrap_or(0));
    let left = budget - rows * row_bytes;
    let pairs = pages.min(window_pages).min(
        left.checked_div(page_bytes + window_page_bytes)
            .unwrap_or(0),
    );
    let left = left - pairs * (page_bytes + window_page_bytes);
    let pages = pairs + (pages - pairs).min(left.checked_div(page_bytes).unwrap_or(0));
    let left = left - (pages - pairs) * page_bytes;
    let window_pages =
        pairs + (window_pages - pairs).min(left.checked_div(window_page_bytes).unwrap_or(0));
    let left = left - (window_pages - pairs) * window_page_bytes;
    let pages = pages + left.checked_div(page_bytes).unwrap_or(0);
    (clamp(rows), clamp(pages), clamp(window_pages))
}
/// `slots` of `shelf`, one after another, each its planes in order.
fn read_shelf(shelf: &Shelf, slots: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &slot in slots {
        for (plane, &bytes) in shelf.planes.iter().enumerate() {
            let (pinned, at) = shelf.place(slot, plane).expect("backed by the caller");
            out.extend_from_slice(pinned.view(at, bytes).expect("inside its chunk"));
        }
    }
    out
}

/// Writes `bytes`, as [`read_shelf`] lays them out, into `slots` of `shelf`.
fn write_shelf(shelf: &Shelf, slots: &[u32], bytes: &[u8], what: &'static str) -> Result<()> {
    let slot_bytes: u64 = shelf.planes.iter().sum();
    let need = slots.len() as u64 * slot_bytes;
    if bytes.len() as u64 != need {
        return Err(Fault::Ceiling {
            what,
            need,
            have: bytes.len() as u64,
        });
    }
    let mut from = 0;
    for &slot in slots {
        for (plane, &len) in shelf.planes.iter().enumerate() {
            let (pinned, at) = shelf.place(slot, plane).expect("backed by the caller");
            let len = usize::try_from(len).unwrap_or(0);
            let at = usize::try_from(at).unwrap_or(usize::MAX);
            assert!(
                pinned.write(at, &bytes[from..from + len]),
                "inside its chunk"
            );
            from += len;
        }
    }
    Ok(())
}

fn slot_file(error: std::io::Error) -> Fault {
    Fault::Integrity {
        at: "kv slot file",
        why: error.to_string(),
    }
}

/// The most pages `bytes` of cache rows could hold, the ceiling the pool's
/// address space is reserved for: `bytes` over the least a page adds to the
/// rows. A full row adds the page; a windowed one at most half of it, since
/// its window grows by a page for every two past the first sequence.
pub fn page_ceiling(trace: &Trace, page_size: u32, context: u32, bytes: u64) -> Result<u64> {
    let mut page: u64 = 0;
    for row in &trace.caches {
        if let CacheRow::Kv {
            name,
            planes,
            dtype,
            window,
            ..
        } = row
        {
            let cells = u64::from(page_size) / if window.is_some() { 2 } else { 1 };
            let element = elem_bytes(name, *dtype)?;
            for width in planes {
                page = page.saturating_add(cells * width * element);
            }
        }
    }
    let one_sequence = u64::from(context.div_ceil(page_size.max(1)));
    Ok(match page {
        0 => one_sequence,
        page => (bytes / page).clamp(one_sequence, u64::from(u32::MAX)),
    })
}

/// A model's declared context shortened to what the card holds: the
/// longest whole number of pages, at most `declared`, at which one sequence
/// takes no more than half of what `utilization` of the card, less what others
/// already hold, leaves past the weight tier: the activations and a second
/// sequence have the other half.
/// Never under one page.
pub fn declared_context_that_fits(
    trace: &Trace,
    page_size: u32,
    declared: u32,
    max_tokens: u32,
    utilization: f64,
    weights: u64,
) -> Result<u32> {
    let page_size = page_size.max(1);
    let window = window_of(trace)?;
    let one = |pages: u32| -> Result<u64> {
        let paging = Paging::of(page_size, pages.saturating_mul(page_size), 1, 1)?
            .windowed(window, max_tokens);
        one_slot_bytes(trace, paging)
    };
    let (free, total) = device_memory()?;
    let room = elastic::budget_bytes(free, total, utilization)
        .saturating_sub(weight_tier_bytes(trace, weights, 0)?)
        / 2;
    let most = declared.div_ceil(page_size).max(1);
    if one(most)? <= room {
        return Ok(declared);
    }
    let (mut fits, mut over) = (1u32, most);
    while over - fits > 1 {
        let mid = fits + (over - fits) / 2;
        if one(mid)? <= room {
            fits = mid;
        } else {
            over = mid;
        }
    }
    Ok(fits.saturating_mul(page_size))
}

pub fn one_slot_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut bytes: u64 = 0;
    for row in &trace.caches {
        match row {
            CacheRow::Kv {
                name,
                planes,
                dtype,
                window,
                ..
            } => {
                let element = elem_bytes(name, *dtype)?;
                let pages = match (window, paging.window) {
                    (Some(_), Some(_)) => paging.window_ring(),
                    _ => paging.pages_per_slot,
                };
                let cells = u64::from(pages) * u64::from(paging.page_size);
                for width in planes {
                    bytes = bytes.saturating_add(cells * width * element);
                }
            }
            CacheRow::State {
                name, slab, dtype, ..
            } => {
                let stride: u64 = slab.iter().product();
                bytes = bytes.saturating_add(stride * elem_bytes(name, *dtype)?);
            }
        }
    }
    Ok(bytes)
}

/// What one fire's rows take in the windowed kv rows past the windows they
/// land in: every row a fire writes is resident there at once, so this
/// grows with `max_forward_tokens` as the activation arena does.
pub fn window_fire_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let Some(window) = paging.window else {
        return Ok(0);
    };
    let cells = u64::from(window.fire_pages) * u64::from(paging.page_size);
    let mut bytes: u64 = 0;
    for row in &trace.caches {
        if let CacheRow::Kv {
            name,
            planes,
            dtype,
            window: Some(_),
            ..
        } = row
        {
            let element = elem_bytes(name, *dtype)?;
            for width in planes {
                bytes = bytes.saturating_add(cells * width * element);
            }
        }
    }
    Ok(bytes)
}

/// One sequence's cache rows: its kv pages at the declared context and the
/// recurrent slabs of the least state slots, the least `Pools::fit_the_card`
/// seats before refusing.
pub fn least_sequence_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    let mut state: u64 = 0;
    for row in &trace.caches {
        if let CacheRow::State {
            name, slab, dtype, ..
        } = row
        {
            let stride: u64 = slab.iter().product();
            state = state.saturating_add(stride * elem_bytes(name, *dtype)?);
        }
    }
    let slots = u64::from(engine::fit::least_state_slots(1));
    Ok(one_slot_bytes(trace, paging)?.saturating_add(state.saturating_mul(slots)))
}

pub fn admit_the_card(
    utilization: f64,
    weights: u64,
    extra_resident: u64,
    trace: &Trace,
    paging: Paging,
    kv_budget: Option<u64>,
) -> Result<Accounting> {
    let accounting = Accounting::of(
        device_memory()?.1,
        utilization,
        weight_tier_bytes(trace, weights, extra_resident)?,
        match kv_budget {
            Some(budget) => budget,
            None => one_slot_bytes(trace, paging)?,
        },
    );
    accounting.admit()?;
    Ok(accounting)
}

/// The bytes this load's weight tier puts on the device: every plane the
/// trace declares, or the `[model] device_weight_budget` stated under it,
/// plus what else rides along resident.
pub fn weight_tier_bytes(trace: &Trace, stated: u64, extra_resident: u64) -> Result<u64> {
    let full: u64 = crate::weights::plane_bytes(trace)?
        .iter()
        .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
        .sum();
    Ok(match stated {
        0 => full,
        stated => stated.min(full),
    }
    .saturating_add(extra_resident))
}

/// The device's free and total bytes, as the driver reports them now.
pub fn device_memory() -> Result<(u64, u64)> {
    #[cfg(feature = "device")]
    {
        use crate::device::rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        crate::device::ctx::check("cudaMemGetInfo", asked)?;
        Ok((free as u64, total as u64))
    }
    #[cfg(not(feature = "device"))]
    {
        Err(Fault::Runtimeless)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Kv {
        space: u32,
        windowed: bool,
        dtype: Dtype,
        keys_width: u64,
        values_width: u64,
        values_plane: usize,
        head_stride: u64,
        live: Option<[u32; 2]>,
        /// Where the key plane and the value plane start in the row's slot
        /// of the state pool.
        offsets: [u64; 2],
    },
    State {
        stride: u64,
        dtype: Dtype,
        /// The row's place in a working set's record: its index among the
        /// fixed shape's rows.
        run: usize,
    },
}

impl Shape {
    /// The bytes one page holds in a kv row's plane `plane` (0 keys, else
    /// values).
    fn page_bytes(&self, plane: usize, page_size: u32) -> u64 {
        match *self {
            Shape::Kv {
                dtype,
                keys_width,
                values_width,
                ..
            } => {
                let width = if plane == 0 { keys_width } else { values_width };
                u64::from(page_size) * width * u64::from(elem_size(dtype))
            }
            Shape::State { .. } => 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpaceSeat {
    pub page_indptr: Tensor,
    pub page_indices: Tensor,
    pub last_page_lens: Tensor,
    pub row_valid: Tensor,
}

#[derive(Debug, Clone)]
pub struct Seats {
    pub lanes: u32,
    pub rows: u32,
    pub pages: u32,
    pub spaces: Vec<SpaceSeat>,
    pub slot_ids: Tensor,
    pub write_state: bool,
    pub write_state_mask: Tensor,
    pub commit_len: Tensor,
    pub begin_at: Tensor,
    pub snapshots: kernels_cuda::Snapshots,
}

impl Seats {
    #[must_use]
    pub fn rs(mut self, write_state: bool, mask: Tensor, commit_len: Tensor) -> Seats {
        self.write_state = write_state;
        self.write_state_mask = mask;
        self.commit_len = commit_len;
        self
    }

    #[must_use]
    pub fn splitting(mut self, boundary: Tensor) -> Seats {
        self.begin_at = boundary;
        self
    }

    #[must_use]
    pub fn snapshotting(mut self, snapshots: kernels_cuda::Snapshots) -> Seats {
        self.snapshots = snapshots;
        self
    }
}

#[derive(Debug)]
pub struct Pools {
    pool: PhysicalPool,
    /// The state pool: the kv rows' pages and the working sets' recurrent
    /// records, a slot each.
    chunks: chunks::Chunks,
    /// Each record a lane named by slot refs, by its first piece's ref: the
    /// refs of all its pieces.
    records: std::collections::HashMap<u32, Vec<u32>>,
    /// The lanes a fire's staged id tables are laid for, and the entries its
    /// snapshot table is ([`Pools::seat_tables`]).
    id_lanes: u32,
    snap_entries: u32,
    shapes: Vec<Shape>,
    pooled: Vec<PoolRow>,
    paging: Paging,
    asked: u64,
    ceiling: u64,
    /// `[model] device_kv_budget`: the bytes the cache rows take, never cut
    /// to fit; `None` fills what the card hands out.
    budget: Option<u64>,
    /// The state slots `fit_the_card` keeps whatever the declared context
    /// seats (`keep_slots`).
    kept_slots: u32,
    airborne: Option<Airborne>,
    committed_kv_pages: u32,
    committed_state_slots: u32,
    host: Option<HostKv>,
    disk: Option<DiskKv>,
    /// Every declared page and slot is backed ([`Pools::back_whole`]), and
    /// never trimmed since.
    whole: bool,
    /// The moves to or from the host tier still on the swap stream, and
    /// the read-outs on the tap stream, each with the device pages and the
    /// host slots it holds.
    swaps: Vec<Swap>,
}

/// A move between the device and the host tier, on the swap stream, or a
/// tap's read-out: its paged and windowed device pages, the paged host
/// slots it reads or writes, and the event it lands at.
#[derive(Debug)]
struct Swap {
    landed: crate::device::graph::Event,
    paged: Vec<u32>,
    windowed: Vec<u32>,
    host: Vec<u32>,
}

/// The slot file kv is suspended into below host memory, and the pinned
/// page a copy to or from it goes through, in slot layout.
#[derive(Debug)]
struct DiskKv {
    file: engine::disk::SlotFile,
    stage: crate::device::Pinned,
    /// An h2d out of the stage may still be in flight on the stream.
    landing: bool,
}
/// The pinned host pages kv is suspended into. Each paged plane keeps its
/// host pages side by side, as its device arena does, so a run of pages
/// moves in one copy; the host rs rows follow them.
#[derive(Debug)]
struct HostKv {
    paged: Shelf,
    windowed: Shelf,
    rows: Shelf,
}

/// The bytes a host shelf pins at a time.
const SHELF_CHUNK: u64 = 256 << 20;

/// One kind of host slot, pinned a chunk of slots at a time as a copy first
/// touches it. A chunk lays its slots out plane by plane, so a run of slots
/// within one chunk is one span a plane.
#[derive(Debug, Default)]
struct Shelf {
    planes: Vec<u64>,
    slots: u32,
    per_chunk: u32,
    chunks: Vec<Option<crate::device::Pinned>>,
}

impl Shelf {
    fn new(planes: Vec<u64>, slots: u32) -> Shelf {
        let slot: u64 = planes.iter().sum();
        let per_chunk = (SHELF_CHUNK / slot.max(1)).clamp(1, u64::from(slots.max(1)));
        let per_chunk = u32::try_from(per_chunk).unwrap_or(u32::MAX);
        Shelf {
            planes,
            slots,
            per_chunk,
            chunks: (0..slots.div_ceil(per_chunk)).map(|_| None).collect(),
        }
    }

    /// Pins the chunks that hold `slots`.
    fn back(&mut self, slots: impl IntoIterator<Item = u32>, what: &'static str) -> Result<()> {
        let slot_bytes: u64 = self.planes.iter().sum();
        for slot in slots {
            if slot >= self.slots {
                return Err(Fault::Ceiling {
                    what,
                    need: u64::from(slot) + 1,
                    have: u64::from(self.slots),
                });
            }
            let chunk = slot / self.per_chunk;
            if self.chunks[chunk as usize].is_none() {
                let held = (self.slots - chunk * self.per_chunk).min(self.per_chunk);
                let bytes = usize::try_from(u64::from(held) * slot_bytes).unwrap_or(usize::MAX);
                let pinned = huge_tier(bytes)
                    .map_or_else(|| crate::device::Pinned::mapped_uninit(bytes), Ok)?;
                self.chunks[chunk as usize] = Some(pinned);
            }
        }
        Ok(())
    }

    /// The pinned chunk holding `slot`, and where its `plane` starts in it;
    /// `None` until [`Self::back`] pins it.
    fn place(&self, slot: u32, plane: usize) -> Option<(&crate::device::Pinned, u64)> {
        let chunk = slot / self.per_chunk;
        let pinned = self.chunks.get(chunk as usize)?.as_ref()?;
        let held = (self.slots - chunk * self.per_chunk).min(self.per_chunk);
        let before: u64 = self.planes[..plane].iter().sum();
        Some((
            pinned,
            before * u64::from(held) + u64::from(slot % self.per_chunk) * self.planes[plane],
        ))
    }

    fn at(&self, slot: u32, plane: usize) -> Result<u64> {
        self.place(slot, plane)
            .map(|(pinned, offset)| pinned.host() as u64 + offset)
            .ok_or(Fault::Ceiling {
                what: "pinned host slots",
                need: u64::from(slot) + 1,
                have: 0,
            })
    }
}

/// `bytes` pinned on reserved huge pages, where the host's pool holds them.
#[cfg(target_os = "linux")]
fn huge_tier(bytes: usize) -> Option<crate::device::Pinned> {
    crate::device::Pinning::huge(bytes)?.lock().ok()
}

#[cfg(not(target_os = "linux"))]
fn huge_tier(_bytes: usize) -> Option<crate::device::Pinned> {
    None
}

/// A group a tap reads a host slot's planes in: its layer, its planes (of
/// the paged planes, in a slot's order), and their spans `(offset, bytes)`
/// of a slot.
#[derive(Debug, Clone, Default)]
pub struct TapGroup {
    pub layer: u32,
    pub planes: Vec<usize>,
    pub spans: Vec<(u64, u64)>,
}

/// A tap's pages of a group: the copies `(dst, src, bytes)` out to the host
/// tier, and where each page's planes land there, page after page,
/// `(address, bytes)`.
#[derive(Debug, Clone, Default)]
pub struct TapSpans {
    pub copies: Vec<(u64, u64, usize)>,
    pub lands: Vec<(u64, usize)>,
}

/// What [`Pools::export_kv`] hands out: the planes' regions and the shape
/// their pages have.
#[derive(Debug, Clone)]
pub struct ExportedKv {
    pub regions: Vec<engine::KvRegion>,
    pub layers: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub dtype: Dtype,
    pub page_size: u32,
}

impl Pools {
    #[allow(clippy::too_many_arguments)]
    pub fn reserve(
        device: i32,
        utilization: f64,
        trace: &Trace,
        paging: Paging,
        budget: Option<u64>,
        facts: &Facts,
        budgets: &poem_compiler::Budgets,
        activation: u64,
    ) -> Result<Pools> {
        let pool = PhysicalPool::open(device, utilization)?;
        let layout = lay(trace, &paging, budgets)?;
        let mut shapes = Vec::with_capacity(trace.caches.len());

        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    space,
                    window,
                    live,
                    rows: quantized,
                    ..
                } => {
                    let element = elem_bytes(name, *dtype)?;
                    let windowed = window.is_some() && paging.window.is_some();
                    let (keys_width, values_width, values_plane) = match planes.as_slice() {
                        [plane] => (*plane, *plane, 0),
                        [keys, values] => (*keys, *values, 1),
                        [] => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares no planes at all — one \
                                     token's entry is written as at least one plane"
                                ),
                            });
                        }
                        many => {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares {} planes — this shell \
                                     binds a key plane and a value plane, and knows no third",
                                    many.len()
                                ),
                            });
                        }
                    };
                    let restated = facts.row(index);
                    if let Some(seat) = restated.filter(|seat| seat.kv_heads != 0) {
                        let heads = u64::from(seat.kv_heads) * u64::from(seat.head_dim);
                        // DeepSeek's quantized rows hold a 512-wide head in
                        // a u8 plane of the format the trace declares: its
                        // width (a compact pool's share of it) the format's.
                        let ds_row = *dtype == Dtype::U8
                            && heads == 512
                            && quantized.is_some_and(|f| {
                                matches!(
                                    f,
                                    poem_ir::KvRows::V4
                                        | poem_ir::KvRows::Mx8
                                        | poem_ir::KvRows::Nv4
                                ) && keys_width != 0
                                    && f.bytes() % keys_width == 0
                            })
                            && values_width == keys_width;
                        if !ds_row && (heads != keys_width || heads != values_width) {
                            return Err(Fault::Unbound {
                                what: format!(
                                    "cache `{name}`, which declares the planes {planes:?} while \
                                     the paged launches that read it restate {} heads of {} — a \
                                     {heads}-wide row",
                                    seat.kv_heads, seat.head_dim
                                ),
                            });
                        }
                    }
                    let at = layout
                        .shape(Kind::paged(windowed))
                        .and_then(|shape| shape.offset_of(index as u32))
                        .ok_or_else(|| {
                            Fault::program(
                                "store::reserve",
                                format!("cache `{name}` has no run in the chunk recipe's slot"),
                            )
                        })?;
                    let keys = u64::from(paging.page_size) * keys_width * element;
                    shapes.push(Shape::Kv {
                        offsets: [at, at + if values_plane == 0 { 0 } else { keys }],
                        space: *space,
                        windowed,
                        dtype: *dtype,
                        keys_width,
                        values_width,
                        values_plane,
                        head_stride: restated.map_or(keys_width, |seat| u64::from(seat.head_dim)),
                        live: *live,
                    });
                }
                CacheRow::State {
                    name, slab, dtype, ..
                } => {
                    let stride: u64 = slab.iter().product();
                    let run = layout
                        .fixed
                        .as_ref()
                        .and_then(|fixed| fixed.rows.iter().position(|run| run.row == index as u32))
                        .ok_or_else(|| {
                            Fault::program(
                                "store::reserve",
                                format!("state `{name}` has no place in the chunk recipe's record"),
                            )
                        })?;
                    shapes.push(Shape::State {
                        stride,
                        dtype: *dtype,
                        run,
                    });
                }
            }
            debug_assert_eq!(shapes.len(), index + 1, "one shape per cache row");
        }
        let mut pooled = Vec::new();
        for (space, width) in poem_compiler::chunk::census::pooled(trace) {
            // A state keyed by a windowed row's pages (`space` is the cache
            // row its gather reads) lies in the windowed pages' slots.
            let windowed = paging.window.is_some()
                && matches!(
                    shapes.get(space as usize),
                    Some(Shape::Kv { windowed: true, .. })
                );
            let offset = layout
                .shape(Kind::paged(windowed))
                .and_then(|shape| shape.pool_of(space))
                .ok_or_else(|| {
                    Fault::program(
                        "store::reserve",
                        format!(
                            "the compressor state keyed by cache row {space} has no run in \
                             the chunk recipe's slot"
                        ),
                    )
                })?;
            pooled.push(PoolRow {
                space,
                width,
                windowed,
                offset,
            });
        }
        let total = chunks_at(&layout, &paging, paging.pages(), paging.slots);
        let chunks = chunks::Chunks::reserve(&pool, layout, total, activation)?;
        Ok(Pools {
            pool,
            chunks,
            records: std::collections::HashMap::new(),
            id_lanes: 1,
            snap_entries: 1,
            shapes,
            pooled,
            paging,
            asked: paging.pages(),
            ceiling: paging.pages(),
            budget,
            kept_slots: 0,
            airborne: None,
            committed_kv_pages: 0,
            committed_state_slots: 0,
            host: None,
            disk: None,
            whole: false,
            swaps: Vec::new(),
        })
    }

    /// The state slots the fit keeps while the pages beside them seat one
    /// sequence at the declared context, rather than cut them to the
    /// sequences the pages seat there: a slot for each lane a fire batches.
    pub fn keep_slots(&mut self, slots: u32) {
        self.kept_slots = slots;
    }

    pub fn watch(&mut self, airborne: Airborne) {
        self.airborne = Some(airborne);
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    #[must_use]
    pub fn pool_slabs(&self) -> Vec<(u32, PoolSlabs)> {
        let layout = self.chunks.layout();
        self.pooled
            .iter()
            .map(|row| {
                let grain = layout
                    .shape(Kind::paged(row.windowed))
                    .map_or(0, |shape| shape.grain);
                let rows = u32::try_from(self.chunks.bytes() / row.plane_bytes(1).max(1))
                    .unwrap_or(u32::MAX);
                let plane = |at: u64| {
                    Tensor::new(
                        self.chunks.base()
                            + row.offset
                            + at * row.plane_bytes(self.paging.page_size),
                        rows,
                        narrow(row.width) as u32,
                        POOL_STATE,
                    )
                };
                (
                    row.space,
                    PoolSlabs {
                        state_kv: plane(0),
                        state_score: plane(1),
                        ape: Tensor::ABSENT,
                        stride: wide(grain / u64::from(elem_size(POOL_STATE))),
                    },
                )
            })
            .collect()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.chunks.bytes()
    }

    #[must_use]
    pub fn declared_bytes(&self) -> u64 {
        self.declared_at(self.paging.pages())
    }

    /// The watermark the cache rows reach with `pages` kv pages declared and
    /// the paging's state slots, as the map units their arenas back.
    #[must_use]
    pub fn declared_at(&self, pages: u64) -> u64 {
        self.declared_reshaped(pages, self.paging.slots)
    }

    /// The watermark with `pages` kv pages and `slots` state slots: the
    /// chunks they take.
    fn declared_reshaped(&self, pages: u64, slots: u32) -> u64 {
        let layout = self.chunks.layout();
        u64::from(chunks_at(layout, &self.paging, pages, slots)).saturating_mul(layout.chunk_bytes)
    }

    /// Re-reserves the kv planes and compressor slabs for a pool of `pages`,
    /// so each maps in the unit its size earns rather than the one the asked
    /// count gave it. Only before anything is committed to them: a mapped
    /// unit has a base a table may already hold.
    fn reshape(&mut self, pages: u64, slots: u32) -> Result<()> {
        self.ceiling = pages;
        self.paging.pages = pages;
        self.paging.slots = slots;
        let total = chunks_at(self.chunks.layout(), &self.paging, pages, slots);
        self.chunks.reshape(&mut self.pool, total)
    }

    #[must_use]
    pub fn map_unit_bytes(&self) -> u64 {
        self.pool.map_unit_bytes()
    }

    pub fn spare_bytes(&self) -> Result<u64> {
        let reserve = self
            .declared_bytes()
            .max(self.high_water_bytes())
            .saturating_sub(self.committed_bytes());
        self.pool.spare_bytes(reserve)
    }

    pub fn spare_bytes_for_bodies(&self, room: u64) -> Result<u64> {
        let live = self.pool.spare_bytes(0)?;
        let reserve = self
            .declared_bytes()
            .max(self.high_water_bytes())
            .saturating_sub(self.committed_bytes());
        Ok(live.saturating_sub(reserve).max(live.min(room)))
    }

    /// Sizes the declared pool to the card. The page count arrives from the
    /// operator's config (or its 256-slot default) knowing nothing of the
    /// device, and the watermark it declares is what `spare_bytes` holds back
    /// and what the runtime admits frames against, so a pool declared past
    /// what the card can back turns every growth into a refusal the runtime's
    /// static admission has already forbidden. The fit is against the elastic
    /// budget as it stands now, so it is called once everything else this
    /// load puts on the device is resident, and again after arming, when the
    /// bodies have taken what they take and the rest is the cache rows' to
    /// declare. `programs` is held for the guests' programs throughout;
    /// `bodies` is the configured allowance, which `bodies_allowance` holds
    /// down to what the room affords, and both yield before one sequence at
    /// the declared context does, since a pool under one sequence is no
    /// deployment. While nothing is committed to the planes, a pool the
    /// asked count's map unit would leave under a sequence is re-reserved
    /// in the finer unit its own size earns. `resident` spells what is on
    /// the device already, for the refusal when not even that sequence fits.
    pub fn fit_the_card(
        &mut self,
        bodies: u64,
        programs: u64,
        resident: &dyn core::fmt::Display,
    ) -> Result<Fitted> {
        let live = self
            .pool
            .spare_bytes(0)?
            .saturating_add(self.committed_bytes());
        let fresh = self.chunks.fresh();
        let one_slot = u64::from(self.paging.pages_per_slot).min(self.ceiling);
        let mut need = if fresh {
            self.declared_at(one_slot)
                .min(self.declared_reshaped(one_slot, self.paging.slots))
        } else {
            self.declared_at(one_slot)
        };
        // A stated budget is the rows' before the bodies and programs are
        // held anything; with none, one sequence is.
        let claim = self.budget.unwrap_or(need);
        let configured = bodies;
        let asked_programs = programs;
        let (mut bodies, mut programs) =
            holds_within(configured, asked_programs, live.saturating_sub(claim));
        let mut held = programs.saturating_add(bodies);
        let mut room = live.saturating_sub(held);
        if let Some(budget) = self.budget.filter(|&budget| budget > room) {
            return Err(Fault::Residency(
                engine::fit::Short::Budget { budget, room }.to_string(),
            ));
        }
        let limit = self.budget.unwrap_or(room);
        let mut fit = pages_within(self.ceiling, limit, |pages| self.declared_at(pages));
        if fresh && fit < self.ceiling {
            let slots = self.paging.slots;
            let fine = pages_within(self.ceiling, limit, |pages| {
                self.declared_reshaped(pages, slots)
            });
            if fine > fit {
                // The planes are reserved for what the rows reach once the
                // bodies' allowance comes back after arming (the second fit
                // sizes the pages to the room then), not only for this
                // pass's: reserved at this pass's pages, the later fit could
                // never pass them.
                let reach = match self.budget {
                    Some(_) => fine,
                    None => pages_within(self.ceiling, live.saturating_sub(programs), |pages| {
                        self.declared_reshaped(pages, slots)
                    })
                    .max(fine),
                };
                self.reshape(reach, slots)?;
                fit = pages_within(self.ceiling, limit, |pages| self.declared_at(pages));
            }
        }
        // A hybrid declares a recurrent slab for every state slot, whatever
        // the pool seats: a heavy slab can take the room its lanes' pages
        // needed, and on a short card the slab alone can be the refusal.
        // Where the pages that fit seat fewer sequences at the declared
        // context than there are slots, the slots are cut to where the two
        // seat alike, never under what leaves the pages their half of the
        // room past one sequence.
        let one = one_slot.max(1);
        let seats = u64::from(self.paging.slots);
        if fit / one < seats && fresh && self.has_state() && self.budget.is_none() {
            let one_sequence = self.declared_reshaped(one_slot, 0);
            // What the fires were asked and the uncut slab left no room to
            // hold is held after the cut, so the slots leave it too.
            let cut = room.saturating_sub(asked_programs.saturating_sub(programs));
            let floor = state_slots_within(self.paging.slots, cut, one_sequence, |slots| {
                self.declared_reshaped(0, slots)
            });
            // The kept slots (`keep_slots`) stay while the pages beside them
            // still seat one sequence, as TensorRT-LLM's state pool holds a
            // slot for each of `max_batch_size` and its kv takes the rest.
            let kept = self.kept_slots.min(self.paging.slots);
            let floor = if kept > floor
                && pages_within(self.ceiling, cut, |pages| {
                    self.declared_reshaped(pages, kept)
                }) >= one_slot
            {
                kept
            } else {
                floor
            };
            let slots = balanced_state_slots(self.paging.slots, floor, one_slot, |slots| {
                pages_within(self.ceiling, cut, |pages| {
                    self.declared_reshaped(pages, slots)
                })
            });
            let pages = pages_within(self.ceiling, cut, |pages| {
                self.declared_reshaped(pages, slots)
            });
            if pages >= one_slot && slots < self.paging.slots {
                self.reshape(pages, slots)?;
                need = self.declared_at(one_slot);
                // The slab the slots gave back is room the bodies were
                // refused when the allowance was struck; strike it again.
                (bodies, programs) =
                    holds_within(configured, asked_programs, live.saturating_sub(need));
                held = programs.saturating_add(bodies);
                room = live.saturating_sub(held);
                fit = pages_within(self.ceiling, room, |pages| self.declared_at(pages));
            }
        }
        if let Some(budget) = self.budget.filter(|_| fit < one_slot) {
            return Err(Fault::Residency(
                engine::fit::Short::Sequence {
                    budget,
                    least: need,
                }
                .to_string(),
            ));
        }
        if fit < one_slot {
            return Err(Fault::Residency(format!(
                "the card does not hold this deployment: {resident}; under `[model] \
                 device_memory_share` that leaves {room} bytes for the cache rows, and one \
                 sequence at the declared context, beside the declared state slots, needs \
                 {need} across them. Lower `[engine] max_model_len` or `[engine] \
                 max_state_slots`, raise `[model] device_memory_share`, state a `[model] \
                 device_weight_budget` that streams the weight tier down, or free what \
                 else holds the device."
            )));
        }
        self.paging.pages = fit;
        // A flat arena allocates its whole range on its first commit, so its
        // range is the fitted one, not the one the asked count reserved.
        if !elastic::mapped_on_demand() && self.committed_bytes() == 0 {
            self.reshape(fit, self.paging.slots)?;
        }
        // The runtime admits frames against this watermark, so the pool is
        // pledged it beside what its other arenas hold now: a frame inside
        // it is a frame the fit already sized the card for, and what a fire
        // allocates beside the pool later is not its refusal to answer.
        let others = self
            .pool
            .committed_pages()
            .saturating_sub(elastic::pages_for_bytes(self.committed_bytes()));
        self.pool
            .pledge(elastic::pages_for_bytes(self.declared_at(fit)).saturating_add(others));
        Ok(Fitted {
            asked: self.asked,
            fit,
            room,
            held,
            bodies,
            spare: live.saturating_sub(need).saturating_sub(programs),
            slots: self.paging.slots,
        })
    }

    /// Where a fire's activation starts: under the state pool's end.
    #[must_use]
    pub fn activation_base(&self) -> u64 {
        self.chunks.activation_base()
    }

    /// Backs the first `bytes` of the activation's span, beside the pool's
    /// budget where `beside` and the budget refuses them.
    pub(crate) fn hold_activation(&mut self, bytes: u64, beside: bool) -> Result<()> {
        match self.chunks.hold_activation(&mut self.pool, bytes, beside)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    pub(crate) fn reserve_arena(&self, max_bytes: u64, label: &'static str) -> Result<Arena> {
        Arena::reserve(&self.pool, max_bytes, label)
    }

    /// Backs a prefix of `bytes` of an arena beside the pool's budget
    /// (`elastic::commit_beside`).
    pub(crate) fn commit_beside(&self, arena: &mut Arena, bytes: u64) -> Result<()> {
        elastic::commit_beside(&self.pool, arena, &elastic::Want::Prefix(bytes))
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.chunks.committed_bytes()
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.chunks.high_water_bytes()
    }

    #[must_use]
    pub fn elastic_page_bytes(&self) -> u64 {
        self.pool.page_bytes()
    }

    #[must_use]
    pub fn elastic_budget_pages(&self) -> u64 {
        self.pool.hard_pages()
    }

    #[must_use]
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match *shape {
                Shape::State { stride, dtype, .. } => stride * u64::from(elem_size(dtype)),
                Shape::Kv { .. } => 0,
            })
            .sum::<u64>()
            + self
                .chunks
                .layout()
                .fixed
                .as_ref()
                .map_or(0, |fixed| fixed.rs.iter().map(|rs| rs.bytes).sum())
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::State { .. }))
    }

    #[must_use]
    pub fn windowed_space(&self, space: u32) -> bool {
        self.shapes.iter().any(
            |shape| matches!(shape, Shape::Kv { space: at, windowed: true, .. } if *at == space),
        )
    }

    #[must_use]
    pub fn has_windowed(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::Kv { windowed: true, .. }))
    }

    /// Seats the kv pages `pages` (the windowed ones, `ring`) on slots of
    /// the state pool and backs the segments their chunks lie in.
    fn seat_pages(&mut self, kind: Kind, pages: impl IntoIterator<Item = u64>) -> Result<()> {
        let mut highest = None;
        for page in pages {
            let page = u32::try_from(page).map_err(|_| Fault::Ceiling {
                what: kind.name(),
                need: page,
                have: u64::from(u32::MAX),
            })?;
            let chunk = self.chunks.seat(kind, page)?;
            highest = highest.max(Some(chunk));
        }
        let Some(highest) = highest else {
            return Ok(());
        };
        let segments = self.chunks.segments_for(u64::from(highest) + 1);
        if segments <= self.chunks.committed_segments() {
            return Ok(());
        }
        if self.chunks.by_refs() {
            // Nothing is backed on a fire's path: the runtime commits a
            // segment before it hands out a chunk of it.
            return Err(Fault::program(
                "store::chunks",
                format!(
                    "a {} lies in chunk {highest}, in a segment the runtime has not committed \
                     ({} are)",
                    kind.name(),
                    self.chunks.committed_segments()
                ),
            ));
        }
        match self.chunks.commit_segments(&mut self.pool, segments)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// Backs the windowed pages `pages` (and the null page behind a
    /// window): their slots in the state pool, and their rows of the
    /// compressor slabs their pages key.
    pub fn ensure_windowed(&mut self, pages: impl IntoIterator<Item = u64>) -> Result<()> {
        if !self.has_windowed() {
            return Ok(());
        }
        let capacity = if self.chunks.by_refs() {
            u64::MAX
        } else {
            self.paging.window_pages()
        };
        let mut fresh: Vec<u64> = core::iter::once(0).chain(pages).collect();
        fresh.sort_unstable();
        fresh.dedup();
        if let Some(&page) = fresh.iter().find(|&&page| page >= capacity) {
            return Err(Fault::Ceiling {
                what: "windowed kv pages",
                need: page + 1,
                have: capacity,
            });
        }
        self.seat_pages(Kind::Ring, fresh.iter().copied())
    }

    #[must_use]
    pub fn bases(&self) -> Vec<u64> {
        vec![self.chunks.base()]
    }

    pub fn table(&self, seats: &Seats) -> Result<CacheTable> {
        let mut table = Vec::with_capacity(self.shapes.len());
        for shape in &self.shapes {
            table.push(match *shape {
                Shape::Kv {
                    space,
                    windowed,
                    dtype,
                    keys_width,
                    values_width,
                    values_plane,
                    head_stride,
                    live,
                    offsets,
                } => {
                    let seat = seats
                        .spaces
                        .get(space as usize)
                        .ok_or_else(|| Fault::Unbound {
                            what: format!(
                                "cache space {space}, for which this fire wrote no \
                                     geometry"
                            ),
                        })?;
                    let pages = if windowed {
                        self.paging.window_pages()
                    } else {
                        self.paging.pages()
                    };
                    let cells = pages * u64::from(self.paging.page_size);
                    // A plane's pointer carries its offset in a page's
                    // slot; the pages step by the shape's grain.
                    let grain = self
                        .chunks
                        .layout()
                        .shape(Kind::paged(windowed))
                        .map_or(0, |shape| shape.grain);
                    let _ = values_plane;
                    let plane = |at: usize, width: u64| {
                        Tensor::new(
                            self.chunks.base() + offsets[at],
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    CachePool::Kv {
                        space,
                        pool: KvPool {
                            keys: plane(0, keys_width),
                            values: plane(1, values_width),
                            bf16_keys: Tensor::new(0, 0, 0, dtype),
                            bf16_values: Tensor::new(0, 0, 0, dtype),
                            key_scales: Tensor::new(0, 0, 0, Dtype::U8),
                            value_scales: Tensor::new(0, 0, 0, Dtype::U8),
                            page_indices: seat.page_indices,
                            page_indptr: seat.page_indptr,
                            last_page_lens: seat.last_page_lens,
                            row_valid: seat.row_valid,
                            env_min: Tensor::new(0, 0, 0, dtype),
                            env_max: Tensor::new(0, 0, 0, dtype),
                            has_envelopes: false,
                            page_size: narrow(u64::from(self.paging.page_size)),
                            seq_stride: wide(keys_width),
                            head_stride: wide(head_stride),
                            layout: NHD,
                            scheme_byte: 0,
                            block_size: 0,
                            max_pages_per_request: narrow(u64::from(self.paging.pages_per_slot)),
                            pages_in_batch: narrow(u64::from(seats.pages)),
                            live: live.unwrap_or([0, 0]),
                            page_stride: [grain, grain],
                        },
                    }
                }
                Shape::State { stride, dtype, run } => {
                    // A lane names each piece's slot a grain at a time, in
                    // the id table the row's pieces share; a piece lies
                    // `offset` into its slot.
                    let fixed = self.chunks.layout().fixed.as_ref().ok_or_else(|| {
                        Fault::program("store::table", "a recurrent row and no fixed shape")
                    })?;
                    let row = &fixed.rows[run];
                    let element = u64::from(elem_size(dtype)).max(1);
                    let step = wide(fixed.slot.grain / element);
                    let slab = Tensor::new(
                        self.chunks.base(),
                        u32::try_from(self.chunks.bytes() / fixed.slot.grain.max(1))
                            .unwrap_or(u32::MAX),
                        narrow(stride) as u32,
                        dtype,
                    );
                    let column = u64::from(row.column);
                    let mut snapshots = seats.snapshots;
                    if !snapshots.slots.is_absent() {
                        snapshots.slots.ptr += column * u64::from(self.snap_entries) * 4;
                    }
                    CachePool::Recurrent(RecurrentPool {
                        write_state: seats.write_state,
                        write_state_mask: seats.write_state_mask,
                        commit_len: seats.commit_len,
                        begin_at: seats.begin_at,
                        fused_decay: false,
                        snapshots,
                        slab,
                        slot_ids: Tensor::new(
                            seats.slot_ids.ptr + column * u64::from(self.id_lanes) * 4,
                            seats.slot_ids.rows,
                            row.pieces,
                            Dtype::I32,
                        ),
                        slot_stride_elems: step,
                        conv_slab: slab,
                        conv_stride: step,
                        segments: kernels_cuda::Segments {
                            table: Tensor::ABSENT,
                            seg_chunks: 0,
                            pieces: row.pieces,
                            piece_off: wide(row.offset / element),
                            piece_atoms: if row.pieces > 1 {
                                u32::try_from(row.atoms_per_piece).unwrap_or(u32::MAX)
                            } else {
                                0
                            },
                        },
                    })
                }
            });
        }
        Ok(CacheTable(table))
    }

    pub fn clear(&mut self, slot: u32) -> Result<()> {
        self.zero_slot(None, slot)
    }

    pub fn clear_on(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<()> {
        self.zero_slot(Some(stream), slot)
    }

    /// Copies the record `src` names onto the one `dst` names, slot for
    /// slot: its recurrent rows and the rs buffer's rows with them.
    pub fn copy_slot(&mut self, stream: *mut core::ffi::c_void, src: u32, dst: u32) -> Result<()> {
        for slot in [src, dst] {
            if !self.state_row(slot)? {
                return Ok(());
            }
        }
        if src == dst {
            return Ok(());
        }
        let (from, to) = (self.ensure_record(src)?, self.ensure_record(dst)?);
        let bytes = self.fixed()?.slot.slot_bytes;
        for (&from, &to) in from.iter().zip(&to) {
            crate::device::copy_d2d(
                stream,
                self.chunks.span(Kind::Fixed, to, 0, bytes)?,
                self.chunks.span(Kind::Fixed, from, 0, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
        }
        Ok(())
    }

    fn fixed(&self) -> Result<&layout::FixedShape> {
        self.chunks
            .layout()
            .fixed
            .as_ref()
            .ok_or_else(|| Fault::program("store::state", "a record named and no fixed shape"))
    }

    /// The lanes a fire's id tables are staged for and the entries of its
    /// snapshot table: where each row's ids start in them.
    pub fn seat_tables(&mut self, id_lanes: u32, snap_entries: u32) {
        self.id_lanes = id_lanes.max(1);
        self.snap_entries = snap_entries.max(1);
    }

    /// The ids a fire stages a lane for the recurrent rows.
    #[must_use]
    pub fn ids_a_lane(&self) -> u32 {
        self.chunks
            .layout()
            .fixed
            .as_ref()
            .map_or(1, |fixed| fixed.ids_a_lane().max(1))
    }

    /// Names the record whose pieces lie in the slot refs `pieces`, for the
    /// fires and copies that name it by its first.
    pub fn name_record(&mut self, pieces: &[u32]) -> Result<u32> {
        let want = self.fixed()?.pieces as usize;
        let Some(&key) = pieces.first().filter(|_| pieces.len() == want) else {
            return Err(Fault::program(
                "store::state",
                format!(
                    "a record named by {} slot refs, and this load's takes {want}",
                    pieces.len()
                ),
            ));
        };
        if want > 1 {
            self.records.insert(key, pieces.to_vec());
        }
        Ok(key)
    }

    /// The slots of the record `key` names, in the record's order: the
    /// refs a lane named it by, or, where lanes name state slots, the
    /// slot's own run.
    fn record(&self, key: u32) -> Result<Vec<u32>> {
        let pieces = self.fixed()?.pieces;
        if !self.chunks.by_refs() {
            return Ok((0..pieces).map(|piece| key * pieces + piece).collect());
        }
        if pieces == 1 {
            return Ok(vec![key]);
        }
        self.records.get(&key).cloned().ok_or_else(|| {
            Fault::program(
                "store::state",
                format!("the record at slot ref {key} is read before a lane names its pieces"),
            )
        })
    }

    /// The record `key` names, its slots seated and backed.
    fn ensure_record(&mut self, key: u32) -> Result<Vec<u32>> {
        let slots = self.record(key)?;
        self.seat_pages(Kind::Fixed, slots.iter().map(|&slot| u64::from(slot)))?;
        Ok(slots)
    }

    /// The record's bytes as the host holds them, `(slot of the record,
    /// offset, bytes)` in order: each recurrent row piece after piece, then
    /// the rs buffer's rows.
    fn record_spans(&self) -> Vec<(u32, u64, u64)> {
        let Some(fixed) = self.chunks.layout().fixed.as_ref() else {
            return Vec::new();
        };
        let mut spans = Vec::new();
        for shape in &self.shapes {
            let Shape::State { run, .. } = *shape else {
                continue;
            };
            let row = &fixed.rows[run];
            for (piece, &slot) in row.slots.iter().enumerate() {
                spans.push((slot, row.offset, row.piece_bytes(piece as u32)));
            }
        }
        spans.extend(fixed.rs.iter().map(|rs| (rs.slot, rs.offset, rs.bytes)));
        spans
    }

    /// The staged id tables of a fire whose lanes name the records `keys`
    /// (under zero: none): each table's block `id_lanes` lanes long, a
    /// lane's ids its pieces' slots as the recurrent kernels read them.
    /// Seats and backs each record.
    pub fn state_tables(&mut self, keys: &[i32]) -> Result<Vec<i32>> {
        if !self.has_state() {
            return Ok(keys.to_vec());
        }
        let fixed = self.fixed()?.clone();
        let lanes = self.id_lanes as usize;
        self.expand(&fixed, keys, lanes)
    }

    /// A fire's snapshot slots, as [`Pools::state_tables`] lays a fire's
    /// lanes: an entry a record a snapshot lands in.
    pub fn snapshot_tables(&mut self, keys: &[i32]) -> Result<Vec<i32>> {
        if !self.has_state() || keys.is_empty() {
            return Ok(keys.to_vec());
        }
        let fixed = self.fixed()?.clone();
        let entries = self.snap_entries as usize;
        self.expand(&fixed, keys, entries)
    }

    fn expand(
        &mut self,
        fixed: &layout::FixedShape,
        keys: &[i32],
        rows: usize,
    ) -> Result<Vec<i32>> {
        if keys.len() > rows {
            return Err(Fault::Ceiling {
                what: "records a fire's id tables name",
                need: keys.len() as u64,
                have: rows as u64,
            });
        }
        // One table of one piece: the ids as the keys lie, no padding.
        let single = fixed.tables.len() == 1 && fixed.tables[0].len() == 1;
        let mut out = if single {
            vec![-1i32; keys.len()]
        } else {
            vec![-1i32; rows * fixed.ids_a_lane() as usize]
        };
        // A table's block is `rows` lanes long: where it starts.
        let block = if single { 0 } else { rows };
        for (lane, &key) in keys.iter().enumerate() {
            if key < 0 {
                continue;
            }
            self.state_row(key as u32)?;
            let record = self.ensure_record(key as u32)?;
            let mut column = 0usize;
            for table in &fixed.tables {
                for (piece, &slot) in table.iter().enumerate() {
                    out[column * block + lane * table.len() + piece] =
                        self.chunks.page_id(Kind::Fixed, record[slot as usize])?;
                }
                column += table.len();
            }
        }
        Ok(out)
    }

    /// The bytes of each recurrent layer's rs rows in a record.
    #[must_use]
    pub fn rs_rows(&self) -> Vec<u64> {
        self.chunks
            .layout()
            .fixed
            .as_ref()
            .map_or_else(Vec::new, |fixed| {
                fixed.rs.iter().map(|rs| rs.bytes).collect()
            })
    }

    /// The address of `bytes` at `offset` of recurrent layer `layer`'s rs
    /// buffer rows in the record `key` names.
    pub fn rs_span(&self, key: u32, layer: u32, offset: u64, bytes: u64) -> Result<u64> {
        let fixed = self.fixed()?;
        let rows = fixed.rs.get(layer as usize).ok_or(Fault::Ceiling {
            what: "recurrent layers whose rs rows the record holds",
            need: u64::from(layer) + 1,
            have: fixed.rs.len() as u64,
        })?;
        if offset + bytes > rows.bytes {
            return Err(Fault::Ceiling {
                what: "bytes of a layer's rs rows in the record",
                need: offset + bytes,
                have: rows.bytes,
            });
        }
        let record = self.record(key)?;
        self.chunks.span(
            Kind::Fixed,
            record[rows.slot as usize],
            rows.offset + offset,
            bytes,
        )
    }

    /// The state rows of `slot`, backed first; none when the load keeps no state.
    pub fn state_spans(&mut self, slot: u32) -> Result<Vec<(u64, u64)>> {
        if !self.state_row(slot)? {
            return Ok(Vec::new());
        }
        let record = self.ensure_record(slot)?;
        self.record_spans()
            .into_iter()
            .map(|(piece, offset, bytes)| {
                Ok((
                    self.chunks
                        .span(Kind::Fixed, record[piece as usize], offset, bytes)?,
                    bytes,
                ))
            })
            .collect()
    }

    /// Copies `moves` in the full rows and `windowed` (whole pages) in the
    /// windowed ones, whose page ids are their own.
    pub fn copy_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        moves: &[KvRun],
        windowed: &[KvRun],
    ) -> Result<()> {
        if moves.is_empty() && windowed.is_empty() {
            return Ok(());
        }
        let page_size = u64::from(self.paging.page_size);
        let mut highest = 0u64;
        for span in moves.iter().chain(windowed) {
            for token in [span.src_token, span.dst_token] {
                let end = u64::from(token) + u64::from(span.tokens);
                if end > page_size {
                    return Err(Fault::Ceiling {
                        what: "token slots in one kv page",
                        need: end,
                        have: page_size,
                    });
                }
            }
        }
        for span in moves {
            highest = highest.max(u64::from(span.src_page.max(span.dst_page)) + 1);
        }
        self.kv_within(highest)?;
        self.seat_pages(
            Kind::List,
            moves
                .iter()
                .flat_map(|span| [u64::from(span.src_page), u64::from(span.dst_page)]),
        )?;
        self.ensure_windowed(
            windowed
                .iter()
                .flat_map(|span| [u64::from(span.src_page), u64::from(span.dst_page)]),
        )?;
        for shape in &self.shapes {
            let Shape::Kv {
                windowed: in_window,
                dtype,
                keys_width,
                values_width,
                values_plane,
                offsets,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            let moves = if in_window { windowed } else { moves };
            for (at, &offset) in offsets.iter().enumerate().take(values_plane + 1) {
                let width = if at == 0 { keys_width } else { values_width };
                let cell = width * element;
                for span in moves {
                    if span.tokens == 0 {
                        continue;
                    }
                    let bytes = u64::from(span.tokens) * cell;
                    let src = offset + u64::from(span.src_token) * cell;
                    let dst = offset + u64::from(span.dst_token) * cell;
                    if span.src_page == span.dst_page && src == dst {
                        continue;
                    }
                    crate::device::copy_d2d(
                        stream,
                        self.chunks
                            .span(Kind::paged(in_window), span.dst_page, dst, bytes)?,
                        self.chunks
                            .span(Kind::paged(in_window), span.src_page, src, bytes)?,
                        usize::try_from(bytes).unwrap_or(0),
                    )?;
                }
            }
        }
        Ok(())
    }

    /// The paged planes a kv page spans and the bytes it holds in each: the
    /// full-context rows' key and value planes and the compressor slabs
    /// indexed by the same pages. Windowed rows name their own pages.
    fn paged_planes(&self) -> Vec<(PlaneAt, u64)> {
        let mut planes = self.kv_planes(false);
        planes.extend(self.pooled_planes(false));
        planes
    }

    /// The compressor slabs keyed by a windowed space's pages (or not), and
    /// the bytes a page holds in each.
    fn pooled_planes(&self, windowed: bool) -> Vec<(PlaneAt, u64)> {
        let page_size = self.paging.page_size;
        self.pooled
            .iter()
            .filter(|row| row.windowed == windowed)
            .flat_map(|row| {
                let bytes = row.plane_bytes(page_size);
                (0..2).map(move |at| {
                    (
                        PlaneAt {
                            kind: Kind::paged(windowed),
                            offset: row.offset + at * bytes,
                        },
                        bytes,
                    )
                })
            })
            .collect()
    }

    /// The windowed planes a windowed page spans: the windowed kv rows' and
    /// the compressor slabs their pages key.
    fn windowed_planes(&self) -> Vec<(PlaneAt, u64)> {
        let mut planes = self.kv_planes(true);
        planes.extend(self.pooled_planes(true));
        planes
    }

    /// The planes of the kv rows that are (or are not) `windowed`, and the
    /// bytes a page holds in each.
    fn kv_planes(&self, windowed: bool) -> Vec<(PlaneAt, u64)> {
        let mut planes = Vec::new();
        for shape in &self.shapes {
            let Shape::Kv {
                windowed: in_window,
                values_plane,
                offsets,
                ..
            } = *shape
            else {
                continue;
            };
            if in_window != windowed {
                continue;
            }
            for (at, &offset) in offsets.iter().enumerate().take(values_plane + 1) {
                planes.push((
                    PlaneAt {
                        kind: Kind::paged(windowed),
                        offset,
                    },
                    shape.page_bytes(at, self.paging.page_size),
                ));
            }
        }
        planes
    }

    /// The address of page `page`'s `bytes` in `plane`; the page is seated
    /// and backed by the caller.
    fn plane_span(&self, plane: PlaneAt, page: u32, bytes: u64) -> Result<u64> {
        self.chunks.span(plane.kind, page, plane.offset, bytes)
    }

    /// Rewrites the page ids of a fire's table (the windowed rows', `ring`)
    /// as the ids the paged kernels read their slots by; the fire's commit
    /// has seated them.
    pub fn page_ids(&self, ring: bool, ids: &mut [i32]) -> Result<()> {
        self.chunks.page_ids(Kind::paged(ring), ids)
    }

    /// Whether the load keeps a recurrent record for state slot `slot`;
    /// refused past the slots it seats, where lanes do not name slot refs.
    fn state_row(&self, slot: u32) -> Result<bool> {
        if self.chunks.by_refs() {
            return Ok(self.has_state());
        }
        slab_row(self.has_state(), self.paging.slots, slot)
    }

    /// From here on lanes name their slots by slot ref
    /// ([`chunks::Chunks::take_refs`]).
    pub fn take_refs(&mut self) {
        self.chunks.take_refs();
    }

    /// Whether lanes name slot refs.
    #[must_use]
    pub fn by_refs(&self) -> bool {
        self.chunks.by_refs()
    }

    /// See `Engine::commit_segments`.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        match self.chunks.commit_segments(&mut self.pool, segments)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// See `Engine::release_segments`: the activation's span stays.
    pub fn release_segments(&mut self, segments: u32) -> Result<()> {
        self.chunks.release_segments(&mut self.pool, segments)
    }

    /// See `Engine::evacuate`.
    pub fn evacuate(
        &mut self,
        stream: *mut core::ffi::c_void,
        evacuation: &engine::chunk::Evacuation,
    ) -> Result<()> {
        self.fence_all_swaps(stream)?;
        self.chunks.evacuate(stream, evacuation)
    }

    /// The state pool as the runtime counts it, for fires of up to
    /// `max_tokens`: the chunks the fit left the cache rows, the rest of the
    /// range stated with the activation's.
    #[must_use]
    pub fn chunk_facts(&self, max_tokens: u32) -> engine::chunk::ChunkFacts {
        let fitted = chunks_at(
            self.chunks.layout(),
            &self.paging,
            self.paging.pages(),
            self.paging.slots,
        );
        let spare = self.chunks.total().saturating_sub(fitted);
        self.chunks
            .facts(max_tokens, spare, !elastic::mapped_on_demand())
    }

    /// Refuses a page count past the pool's.
    fn kv_within(&self, pages: u64) -> Result<()> {
        let capacity = self.paging.pages();
        if pages > capacity && !self.chunks.by_refs() {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: pages,
                have: capacity,
            });
        }
        Ok(())
    }

    /// Backs every page and slot the pools declare, at load, and keeps them
    /// so: backing a page as a fire first touches it takes
    /// the driver's lock, which a dying process on the host holds for
    /// seconds, and a worker taking a dead peer's turns needs new pages then
    /// above all.
    pub fn back_whole(&mut self) -> Result<()> {
        let pages = u32::try_from(self.paging.pages()).unwrap_or(u32::MAX);
        let slots = if self.has_state() {
            self.paging.slots
        } else {
            0
        };
        match self.commit_to(pages, slots)? {
            Commit::Committed => {}
            refusal => return Err(refuse(&self.pool, refusal)),
        }
        match self
            .chunks
            .commit_segments(&mut self.pool, self.chunks.segments())?
        {
            Commit::Committed => {}
            refusal => return Err(refuse(&self.pool, refusal)),
        }
        self.ensure_windowed(0..self.paging.window_pages())?;
        self.whole = true;
        Ok(())
    }

    /// The kv planes as another worker reads them: every declared page, one
    /// region per paged plane, a page `p` at `base + p * page_stride`. A
    /// windowed space is refused: its pages are named by its own ring, not by
    /// the kv pages a peer reads.
    pub fn export_kv(&mut self, ordinal: u32) -> Result<ExportedKv> {
        if !self.windowed_planes().is_empty() {
            return Err(Fault::Residency(
                "a windowed kv space names its pages by its own ring, so it cannot be \
                 handed to another worker by the kv pages it reads"
                    .to_string(),
            ));
        }
        if !self.whole {
            self.back_whole()?;
        }
        let mut regions = Vec::new();
        // A kv plane's region is the pool from the plane's offset in a
        // slot on: the page a slot ref `r` names starts `r` strides in,
        // while its chunk lies where it was opened.
        let grain = self
            .chunks
            .layout()
            .shape(Kind::List)
            .map_or(0, |shape| shape.grain);
        for (plane, _) in self.paged_planes() {
            regions.push(engine::KvRegion {
                base: self.chunks.base() + plane.offset,
                len: self.chunks.bytes().saturating_sub(plane.offset),
                page_stride: grain,
                domain: engine::MemoryDomain::CudaDevice(ordinal),
            });
        }
        let (layers, heads, head_dim, dtype) = self
            .shapes
            .iter()
            .find_map(|shape| match *shape {
                Shape::Kv {
                    windowed: false,
                    dtype,
                    keys_width,
                    head_stride,
                    ..
                } => Some((dtype, keys_width, head_stride)),
                _ => None,
            })
            .map(|(dtype, keys_width, head_stride)| {
                let layers = self
                    .shapes
                    .iter()
                    .filter(|shape| matches!(shape, Shape::Kv { .. }))
                    .count();
                (
                    u32::try_from(layers).unwrap_or(u32::MAX),
                    u32::try_from(keys_width / head_stride.max(1)).unwrap_or(u32::MAX),
                    u32::try_from(head_stride).unwrap_or(u32::MAX),
                    dtype,
                )
            })
            .ok_or_else(|| {
                Fault::Residency("this load pages no full-attention kv row to export".to_string())
            })?;
        Ok(ExportedKv {
            regions,
            layers,
            heads,
            head_dim,
            dtype,
            page_size: self.paging.page_size,
        })
    }

    /// The bytes one windowed kv page holds over every windowed plane.
    #[must_use]
    pub fn window_page_bytes(&self) -> u64 {
        self.windowed_planes().iter().map(|&(_, bytes)| bytes).sum()
    }

    /// The bytes one kv page holds over every paged plane.
    #[must_use]
    pub fn paged_page_bytes(&self) -> u64 {
        self.paged_planes().iter().map(|&(_, bytes)| bytes).sum()
    }

    /// Seats `pages` host kv pages, `window_pages` host windowed pages and
    /// `rows` host rs rows of `row_bytes`, each pinned a chunk at a time as a
    /// copy first touches it ([`Shelf`]), on reserved huge pages where the
    /// host keeps a pool that holds them. Pinning at once would hold boot for
    /// as long as the driver takes to pin it all, and when a process dies the
    /// driver unpins its memory under a lock every process on the host waits
    /// on for any allocation, about 0.17 s a GiB of 4 KiB pages and a fifth
    /// of that of huge ones.
    pub fn seat_host(
        &mut self,
        pages: u32,
        window_pages: u32,
        rows: u32,
        row_bytes: u64,
    ) -> Result<()> {
        let rows = if row_bytes == 0 { 0 } else { rows };
        let planes = |planes: Vec<(PlaneAt, u64)>| planes.into_iter().map(|(_, b)| b).collect();
        let host = HostKv {
            paged: Shelf::new(planes(self.paged_planes()), pages),
            windowed: Shelf::new(planes(self.windowed_planes()), window_pages),
            rows: Shelf::new(vec![row_bytes], rows),
        };
        self.host = (pages > 0 || window_pages > 0 || rows > 0).then_some(host);
        Ok(())
    }

    #[must_use]
    pub fn host_window_pages(&self) -> u32 {
        self.host.as_ref().map_or(0, |host| host.windowed.slots)
    }

    /// The host rs rows seated and the bytes of one.
    #[must_use]
    pub fn host_rows(&self) -> (u32, u64) {
        self.host
            .as_ref()
            .map_or((0, 0), |host| (host.rows.slots, host.rows.planes[0]))
    }

    /// The host address of rs row `row`.
    pub fn host_row(&mut self, row: u32) -> Result<u64> {
        let Some(host) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what: "host rs rows",
                need: u64::from(row) + 1,
                have: 0,
            });
        };
        host.rows.back([row], "host rs rows")?;
        host.rows.at(row, 0)
    }

    #[must_use]
    pub fn host_pages(&self) -> u32 {
        self.host.as_ref().map_or(0, |host| host.paged.slots)
    }

    /// Opens `dir`'s slot file for this rank with `budget` bytes of kv pages;
    /// a budget short of one page seats none. `keep` names the load when the
    /// file is kept for the next run ([`engine::disk::SlotFile::open_kept`]):
    /// with the page's planes, the layout its slots are read back by.
    pub fn seat_disk(
        &mut self,
        dir: &std::path::Path,
        rank: u32,
        budget: u64,
        keep: Option<&str>,
    ) -> Result<()> {
        let name = format!("r{rank}.kv");
        let file = match keep {
            None => engine::disk::SlotFile::open(dir, &name, self.paged_page_bytes(), budget),
            Some(load) => {
                let planes: Vec<u64> = self.paged_planes().iter().map(|&(_, b)| b).collect();
                let layout = format!(
                    "cuda {load} page {} planes {planes:?}",
                    self.paging.page_size
                );
                engine::disk::SlotFile::open_kept(
                    dir,
                    &name,
                    self.paged_page_bytes(),
                    budget,
                    &layout,
                )
            }
        }
        .map_err(slot_file)?;
        self.disk = file
            .map(|file| {
                crate::device::Pinned::mapped_uninit(file.stride()).map(|stage| DiskKv {
                    file,
                    stage,
                    landing: false,
                })
            })
            .transpose()?;
        Ok(())
    }

    pub fn unseat_disk(&mut self) {
        self.disk = None;
    }

    #[must_use]
    pub fn disk_pages(&self) -> u32 {
        self.disk.as_ref().map_or(0, |disk| disk.file.slots())
    }

    /// The slot file is kept, and holds what the last run left.
    #[must_use]
    pub fn disk_kept(&self) -> bool {
        self.disk.as_ref().is_some_and(|disk| disk.file.kept())
    }

    /// Copies whole kv pages between the device pool and the slot file,
    /// `device[i]` with `disk[i]`, one page through the stage at a time. A
    /// write lands behind the fires that wrote the page; a read returns once
    /// its last h2d copy is enqueued.
    pub fn spill_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        to_disk: bool,
        device: &[u32],
        disk: &[u32],
    ) -> Result<()> {
        let have = u64::from(self.disk_pages());
        if let Some(&slot) = disk.iter().find(|&&slot| u64::from(slot) >= have) {
            return Err(Fault::Ceiling {
                what: "disk kv pages",
                need: u64::from(slot) + 1,
                have,
            });
        }
        self.ensure_kv_pages(device)?;
        let Some(mut spill) = self.disk.take() else {
            return Ok(());
        };
        let copied = self.spill_through(&mut spill, stream, to_disk, device, disk);
        self.disk = Some(spill);
        copied
    }

    fn spill_through(
        &self,
        spill: &mut DiskKv,
        stream: *mut core::ffi::c_void,
        to_disk: bool,
        device: &[u32],
        disk: &[u32],
    ) -> Result<()> {
        let planes = self.paged_planes();
        let (stage, stride) = (spill.stage.host(), spill.file.stride());
        // SAFETY: the stage is this load's pinned page, and every copy on the
        // stream that touches it is waited for before the host reads or
        // writes it.
        let staged = || unsafe { std::slice::from_raw_parts_mut(stage, stride) };
        for (&page, &slot) in device.iter().zip(disk) {
            if spill.landing {
                crate::device::ctx::sync(stream)?;
                spill.landing = false;
            }
            if !to_disk {
                spill.file.read(slot, staged()).map_err(slot_file)?;
            }
            let mut offset = 0u64;
            for &(plane, page_bytes) in &planes {
                let on_device = self.plane_span(plane, page, page_bytes)?;
                let on_host = stage as u64 + offset;
                let (dst, src) = if to_disk {
                    (on_host, on_device)
                } else {
                    (on_device, on_host)
                };
                crate::device::copy_any(stream, dst, src, page_bytes as usize)?;
                offset += page_bytes;
            }
            if to_disk {
                crate::device::ctx::sync(stream)?;
                spill.file.write(slot, staged()).map_err(slot_file)?;
            } else {
                spill.landing = true;
            }
        }
        Ok(())
    }
    /// Copies whole kv pages between the device pool and the host pages,
    /// `device[i]` with `host[i]`, and the windowed pages `(device, host)`
    /// between the windowed rows and theirs: on `swap`, behind what `stream`
    /// has enqueued (the fires that wrote them), so the fires after run
    /// beside it; one that holds its pages waits for it
    /// ([`Pools::fence_swaps`]).
    pub fn swap_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        swap: *mut core::ffi::c_void,
        to_host: bool,
        device: &[u32],
        host: &[u32],
        windowed: &[(u32, u32)],
    ) -> Result<()> {
        let paged: Vec<(u32, u32)> = device.iter().copied().zip(host.iter().copied()).collect();
        self.ensure_kv_pages(device)?;
        self.ensure_windowed(windowed.iter().map(|&(page, _)| u64::from(page)))?;
        let Some(tier) = self.host.as_mut() else {
            return match (host.first(), windowed.first()) {
                (None, None) => Ok(()),
                (paged, _) => Err(Fault::Ceiling {
                    what: if paged.is_some() {
                        "host kv pages"
                    } else {
                        "host windowed pages"
                    },
                    need: 1,
                    have: 0,
                }),
            };
        };
        tier.paged.back(host.iter().copied(), "host kv pages")?;
        tier.windowed.back(
            windowed.iter().map(|&(_, slot)| slot),
            "host windowed pages",
        )?;
        let tier = self.host.as_ref().expect("seated above");
        let mut spans: Vec<(u64, u64, usize)> = Vec::new();
        for (planes, pairs, shelf) in [
            (self.paged_planes(), &paged[..], &tier.paged),
            (self.windowed_planes(), windowed, &tier.windowed),
        ] {
            // A page's planes lie apart in its slot, so each is a span.
            for (at, (plane, page_bytes)) in planes.into_iter().enumerate() {
                for &(d, h) in pairs {
                    let on_device = self.plane_span(plane, d, page_bytes)?;
                    let on_host = shelf.at(h, at)?;
                    let (dst, src) = if to_host {
                        (on_host, on_device)
                    } else {
                        (on_device, on_host)
                    };
                    spans.push((dst, src, usize::try_from(page_bytes).unwrap_or(0)));
                }
            }
        }
        if spans.is_empty() {
            return Ok(());
        }
        let written = crate::device::graph::Event::new()?;
        written.record(stream)?;
        written.wait(swap)?;
        crate::device::copy_batch(swap, to_host, &spans)?;
        let landed = crate::device::graph::Event::new()?;
        landed.record(swap)?;
        self.swaps.push(Swap {
            landed,
            paged: device.to_vec(),
            windowed: windowed.iter().map(|&(page, _)| page).collect(),
            host: host.to_vec(),
        });
        Ok(())
    }

    /// `paged` device pages are read, and `host` slots written, on `stream`
    /// by what it has enqueued: a fire that names one of the pages waits for
    /// it, as for a move, and so does the host before it touches a slot.
    pub fn hold_until(
        &mut self,
        stream: *mut core::ffi::c_void,
        paged: Vec<u32>,
        host: Vec<u32>,
    ) -> Result<()> {
        let landed = crate::device::graph::Event::new()?;
        landed.record(stream)?;
        self.swaps.push(Swap {
            landed,
            paged,
            windowed: Vec::new(),
            host,
        });
        Ok(())
    }

    /// Waits, on the host, for the moves and read-outs still landing that
    /// read or write any of the paged host `slots`: the host reads or
    /// writes them next, and nothing else on the device touches them.
    pub fn settle_host_slots(&mut self, slots: &[u32]) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            if swap.host.iter().any(|slot| slots.contains(slot)) {
                swap.landed.settle()?;
            }
        }
        Ok(())
    }

    /// Orders `stream` behind the moves still landing that hold any of the
    /// device pages a fire names: its paged runs `(first, count)` and its
    /// windowed pages.
    pub fn fence_swaps(
        &mut self,
        stream: *mut core::ffi::c_void,
        paged: &[(u64, u64)],
        windowed: &[u64],
    ) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            let holds = swap.paged.iter().any(|&page| {
                let page = u64::from(page);
                paged
                    .iter()
                    .any(|&(first, count)| first <= page && page < first + count)
            }) || swap
                .windowed
                .iter()
                .any(|&page| windowed.contains(&u64::from(page)));
            if holds {
                swap.landed.wait(stream)?;
            }
        }
        Ok(())
    }

    /// Orders `stream` behind every move still landing: for work that reads
    /// or writes pages without naming them.
    pub fn fence_all_swaps(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            swap.landed.wait(stream)?;
        }
        Ok(())
    }

    fn prune_swaps(&mut self) -> Result<()> {
        let mut kept = Vec::with_capacity(self.swaps.len());
        for swap in self.swaps.drain(..) {
            if !swap.landed.done()? {
                kept.push(swap);
            }
        }
        self.swaps = kept;
        Ok(())
    }

    /// How a host kv slot lays out its pages, rank by rank: each paged plane
    /// `[token][head][dim]`, a row's keys then its values. `None` with the
    /// compressor planes some models page beside their kv.
    #[must_use]
    pub fn host_kv_layout(&self, ranks: u32) -> Option<state::PageLayout> {
        if !self.pooled.is_empty() {
            return None;
        }
        let mut planes = Vec::new();
        for shape in &self.shapes {
            let Shape::Kv {
                windowed: false,
                dtype,
                keys_width,
                values_width,
                head_stride,
                ..
            } = *shape
            else {
                continue;
            };
            for width in [keys_width, values_width] {
                if head_stride == 0 || width % head_stride != 0 {
                    return None;
                }
                planes.push(state::Plane {
                    heads: u32::try_from(width / head_stride).ok()? * ranks,
                    head_dim: u32::try_from(head_stride).ok()?,
                    elem_bytes: elem_size(dtype),
                });
            }
        }
        Some(state::PageLayout {
            ranks,
            page_tokens: self.paging.page_size,
            planes,
        })
    }

    /// Bytes one host kv slot holds across the paged planes.
    #[must_use]
    pub fn host_kv_slot_bytes(&self) -> u64 {
        self.paged_planes().iter().map(|&(_, bytes)| bytes).sum()
    }

    /// The cache row of each paged plane, in a host slot's order.
    fn paged_plane_rows(&self) -> Vec<usize> {
        let mut rows = Vec::new();
        for (row, shape) in self.shapes.iter().enumerate() {
            if let Shape::Kv {
                windowed: false,
                values_plane,
                ..
            } = *shape
            {
                rows.extend(std::iter::repeat_n(row, values_plane + 1));
            }
        }
        rows
    }

    /// How a tap reads a host slot's planes out: a group a layer, each the
    /// planes of the kv rows `row_layer` puts in it, as `(planes, spans of a
    /// slot)`, in the layers' order. `None` where a tap cannot read a layer
    /// whole: windowed rows (named by their own pages), compressor slabs, or
    /// a kv row no layer writes.
    #[must_use]
    pub fn tap_groups(&self, row_layer: &[Option<u32>]) -> Option<Vec<TapGroup>> {
        let windowed = self
            .shapes
            .iter()
            .any(|shape| matches!(shape, Shape::Kv { windowed: true, .. }));
        if windowed || !self.pooled.is_empty() {
            return None;
        }
        let mut groups: std::collections::BTreeMap<u32, TapGroup> =
            std::collections::BTreeMap::new();
        let mut offset = 0u64;
        for (plane, (row, (_, bytes))) in self
            .paged_plane_rows()
            .into_iter()
            .zip(self.paged_planes())
            .enumerate()
        {
            let layer = (*row_layer.get(row)?)?;
            let group = groups.entry(layer).or_insert_with(|| TapGroup {
                layer,
                ..TapGroup::default()
            });
            group.planes.push(plane);
            group.spans.push((offset, bytes));
            offset += bytes;
        }
        Some(groups.into_values().collect())
    }

    /// For planes `planes`, device pages `device[i]` to host slots
    /// `host[i]`.
    pub fn tap_spans(
        &mut self,
        planes: &[usize],
        device: &[u32],
        host: &[u32],
    ) -> Result<TapSpans> {
        self.ensure_kv_pages(device)?;
        let Some(tier) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what: "host kv pages",
                need: 1,
                have: 0,
            });
        };
        tier.paged.back(host.iter().copied(), "host kv pages")?;
        let shelf = &self.host.as_ref().expect("seated above").paged;
        let all = self.paged_planes();
        let mut copies = Vec::with_capacity(planes.len() * device.len());
        let mut lands = Vec::with_capacity(planes.len() * device.len());
        for (&d, &h) in device.iter().zip(host) {
            for &plane in planes {
                let (at, bytes) = all[plane];
                let on_host = shelf.at(h, plane)?;
                let on_device = self.plane_span(at, d, bytes)?;
                let len = usize::try_from(bytes).unwrap_or(0);
                copies.push((on_host, on_device, len));
                lands.push((on_host, len));
            }
        }
        Ok(TapSpans { copies, lands })
    }

    /// Host kv slots `slots`, one after another, each its planes in order.
    /// The caller has synchronized every copy into them.
    pub fn read_host_kv(&mut self, slots: &[u32]) -> Result<Vec<u8>> {
        Ok(read_shelf(self.host_shelf(false, slots)?, slots))
    }

    /// Writes `bytes`, as [`Self::read_host_kv`] lays them out, into host kv
    /// slots `slots`.
    pub fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        write_shelf(
            self.host_shelf(false, slots)?,
            slots,
            bytes,
            "bytes of host kv slots",
        )
    }

    /// Host windowed pages `pages`, a ring's on the host, laid out as
    /// [`Self::read_host_kv`] lays out kv slots.
    pub fn read_host_window(&mut self, pages: &[u32]) -> Result<Vec<u8>> {
        Ok(read_shelf(self.host_shelf(true, pages)?, pages))
    }

    /// Writes `bytes`, as [`Self::read_host_window`] lays them out.
    pub fn write_host_window(&mut self, pages: &[u32], bytes: &[u8]) -> Result<()> {
        write_shelf(
            self.host_shelf(true, pages)?,
            pages,
            bytes,
            "bytes of host windowed pages",
        )
    }

    /// Reads `(row, from, bytes)` spans of the host rs rows, in order.
    pub fn read_host_rows(&mut self, spans: &[(u32, u64, u64)]) -> Result<Vec<u8>> {
        let rows = self.backed_rows(spans)?;
        let mut out = Vec::new();
        for &(row, from, bytes) in spans {
            let (pinned, at) = rows.place(row, 0).expect("backed above");
            out.extend_from_slice(pinned.view(at + from, bytes).expect("inside its chunk"));
        }
        Ok(out)
    }

    /// Writes `bytes`, laid out as [`Self::read_host_rows`] returns them.
    pub fn write_host_rows(&mut self, spans: &[(u32, u64, u64)], bytes: &[u8]) -> Result<()> {
        let need: u64 = spans.iter().map(|&(.., bytes)| bytes).sum();
        if bytes.len() as u64 != need {
            return Err(Fault::Ceiling {
                what: "bytes of host rs rows",
                need,
                have: bytes.len() as u64,
            });
        }
        let rows = self.backed_rows(spans)?;
        let mut from = 0;
        for &(row, offset, len) in spans {
            let (pinned, at) = rows.place(row, 0).expect("backed above");
            let at = usize::try_from(at + offset).unwrap_or(usize::MAX);
            let len = usize::try_from(len).unwrap_or(0);
            assert!(
                pinned.write(at, &bytes[from..from + len]),
                "inside its chunk"
            );
            from += len;
        }
        Ok(())
    }

    fn backed_rows(&mut self, spans: &[(u32, u64, u64)]) -> Result<&Shelf> {
        let Some(host) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what: "host rs rows",
                need: spans.len() as u64,
                have: 0,
            });
        };
        host.rows
            .back(spans.iter().map(|&(row, ..)| row), "host rs rows")?;
        Ok(&host.rows)
    }

    /// The host kv (or, `windowed`, windowed) shelf, with the chunks
    /// holding `slots` pinned.
    fn host_shelf(&mut self, windowed: bool, slots: &[u32]) -> Result<&Shelf> {
        let what = if windowed {
            "host windowed pages"
        } else {
            "host kv pages"
        };
        let Some(host) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what,
                need: 1,
                have: 0,
            });
        };
        let shelf = if windowed {
            &mut host.windowed
        } else {
            &mut host.paged
        };
        shelf.back(slots.iter().copied(), what)?;
        Ok(shelf)
    }

    /// `slot`'s recurrent state, read on `stream` behind the fires that
    /// wrote it.
    pub fn state_bytes(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<Vec<u8>> {
        if !self.state_row(slot)? {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for (at, bytes) in self.state_spans(slot)? {
            let from = out.len();
            out.resize(from + usize::try_from(bytes).unwrap_or(0), 0);
            crate::device::copy_d2h_on(stream, at, &mut out[from..])?;
        }
        Ok(out)
    }

    fn zero_slot(&mut self, stream: Option<*mut core::ffi::c_void>, slot: u32) -> Result<()> {
        if !self.state_row(slot)? {
            return Ok(());
        }
        let record = self.ensure_record(slot)?;
        let bytes = self.fixed()?.slot.slot_bytes;
        for piece in record {
            let at = self.chunks.span(Kind::Fixed, piece, 0, bytes)?;
            let len = usize::try_from(bytes).unwrap_or(0);
            match stream {
                Some(stream) => crate::device::zero_span_on(stream, at, len)?,
                None => crate::device::zero_span(at, len)?,
            }
        }
        Ok(())
    }

    /// Seats and backs the kv pages `pages`, and the compressor slabs' rows
    /// under them.
    fn ensure_kv_pages(&mut self, pages: &[u32]) -> Result<()> {
        let highest = pages.iter().map(|&page| u64::from(page) + 1).max();
        let Some(highest) = highest else {
            return Ok(());
        };
        self.kv_within(highest)?;
        self.seat_pages(Kind::List, pages.iter().map(|&page| u64::from(page)))
    }

    fn commit_to(&mut self, kv_pages: u32, state_slots: u32) -> Result<Commit> {
        self.commit_ranges(kv_pages, state_slots, None)
    }

    pub fn commit_frame(
        &mut self,
        demand: engine::frame::Demand,
        kv_ranges: &[(u64, u64)],
        window_pages: &[u64],
    ) -> Result<()> {
        self.seat_pages(
            Kind::List,
            kv_ranges
                .iter()
                .flat_map(|&(first, count)| first..first.saturating_add(count)),
        )?;
        let _ = demand;
        self.ensure_windowed(window_pages.iter().copied())
    }

    fn commit_ranges(
        &mut self,
        kv_pages: u32,
        state_slots: u32,
        kv_ranges: Option<&[(u64, u64)]>,
    ) -> Result<Commit> {
        let kv_pages = kv_pages.max(self.committed_kv_pages);
        let state_slots = state_slots.max(self.committed_state_slots);
        let _ = kv_ranges;
        // The pool's chunks these counts reach, backed a segment at a time.
        let layout = self.chunks.layout();
        let chunks = (u64::from(chunks_at(
            layout,
            &self.paging,
            u64::from(kv_pages),
            state_slots,
        )))
        .max(u64::from(self.chunks.opened()))
        .min(u64::from(self.chunks.total()));
        let segments = self.chunks.segments_for(chunks);
        if (kv_pages > 0 || state_slots > 0) && segments > self.chunks.committed_segments() {
            match self.chunks.commit_segments(&mut self.pool, segments)? {
                Commit::Committed => {}
                refusal => return Ok(refusal),
            }
        }
        self.committed_kv_pages = kv_pages;
        self.committed_state_slots = state_slots;
        Ok(Commit::Committed)
    }
}

/// How `trace`'s kv rows sit in the state pool's chunks: the compiler's
/// recipe for a server card, its token slots this paging's pages. Where the
/// paging keeps no window (the widest spans the context) every row is a list
/// row.
fn lay(trace: &Trace, paging: &Paging, budgets: &poem_compiler::Budgets) -> Result<layout::Layout> {
    let context = if paging.window.is_some() {
        paging.context()
    } else {
        0
    };
    let profile = poem_compiler::DeviceProfile {
        chunk: poem_compiler::chunk::ChunkProfile::SERVER_CUDA.paged(paging.page_size, context),
        ..poem_compiler::DeviceProfile::default()
    };
    let recipe =
        poem_compiler::chunk::plan(trace, budgets, &profile).map_err(|why| Fault::Unbound {
            what: format!("a chunk recipe for this load's cache rows: {why}"),
        })?;
    layout::Layout::of(&recipe)
}

/// The chunks a pool of `pages` kv pages takes: the null chunk, the pages'
/// and the windowed pages beside them.
fn chunks_at(layout: &layout::Layout, paging: &Paging, pages: u64, slots: u32) -> u32 {
    let chunks = 1
        + layout.chunks_for(Kind::List, pages)
        + layout.chunks_for(Kind::Ring, paging.window_pages_at(pages))
        + layout.chunks_for(
            Kind::Fixed,
            u64::from(slots)
                * layout
                    .fixed
                    .as_ref()
                    .map_or(1, |fixed| u64::from(fixed.pieces)),
        );
    u32::try_from(chunks).unwrap_or(u32::MAX)
}

/// One plane a page holds bytes in: `offset` into the page's slot of the
/// state pool.
#[derive(Debug, Clone, Copy)]
struct PlaneAt {
    kind: Kind,
    offset: u64,
}

const POOL_STATE: Dtype = Dtype::Bf16;

/// A compressor's state for the kv row `space` its gather keys it by: two
/// planes (the projection, then its score) of `width` a token, `offset`
/// into the row's pages' slots.
#[derive(Debug)]
struct PoolRow {
    space: u32,
    width: u64,
    /// Keyed by a windowed row's pages.
    windowed: bool,
    offset: u64,
}

impl PoolRow {
    /// The bytes one of its planes holds a page.
    fn plane_bytes(&self, page_size: u32) -> u64 {
        u64::from(page_size) * self.width * u64::from(elem_size(POOL_STATE))
    }
}

/// Whether a slab holds a row for the seat. The runtime seats a kv-only
/// model with no ceiling, so the slot count only binds a model with a slab.
fn slab_row(has_state: bool, slots: u32, slot: u32) -> Result<bool> {
    if !has_state {
        return Ok(false);
    }
    if slot >= slots {
        return Err(Fault::Ceiling {
            what: "recurrent slots",
            need: u64::from(slot) + 1,
            have: u64::from(slots),
        });
    }
    Ok(true)
}

fn refuse(pool: &PhysicalPool, outcome: Commit) -> Fault {
    let page = pool.page_bytes();
    match outcome {
        Commit::Committed => Fault::program(
            "store::commit",
            "a committed outcome reached the refusal path".to_string(),
        ),
        Commit::Exhausted { growth, spare } => Fault::OutOfMemory {
            need: growth.saturating_mul(page),
            have: spare.saturating_mul(page),
        },
        Commit::Impossible { required, ceiling } => Fault::Ceiling {
            what: "bytes of elastic device memory",
            need: required.saturating_mul(page),
            have: ceiling.saturating_mul(page),
        },
    }
}

fn elem_bytes(name: &str, dtype: Dtype) -> Result<u64> {
    poem_compiler::arena::elem_bytes(dtype).ok_or_else(|| Fault::Unbound {
        what: format!("cache `{name}`, stored as {dtype:?}, which has no element size"),
    })
}

fn elem_size(dtype: Dtype) -> u32 {
    poem_compiler::arena::elem_bytes(dtype).unwrap_or(1) as u32
}

/// The widest window a windowed kv row is declared with. Every read of such
/// a row must look through a window no wider, since the pages behind it
/// read the null page.
pub fn window_of(trace: &Trace) -> Result<Option<u32>> {
    for node in &trace.nodes {
        let Some(read) = kv::reads(&node.op) else {
            continue;
        };
        let Some(Def::Cache(row)) = trace.values.get(read.cache.0 as usize).map(|v| &v.def) else {
            continue;
        };
        if let Some(CacheRow::Kv {
            name,
            window: Some(held),
            ..
        }) = trace.caches.get(*row as usize)
            && read.window.is_none_or(|read| read > *held)
        {
            return Err(Fault::Unbound {
                what: format!(
                    "cache `{name}`, which holds a window of {held} tokens while `{}` reads \
                     it through {:?}",
                    node.op.name(),
                    read.window
                ),
            });
        }
    }
    Ok(trace
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { window, .. } => *window,
            CacheRow::State { .. } => None,
        })
        .max())
}

fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

fn wide(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

impl Pools {
    /// See `Engine::commit_pool`.
    pub fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        if let Some(highest) = demand.state_slots.checked_sub(1) {
            self.state_row(highest)?;
        }
        let capacity = self.paging.pages();
        if u64::from(demand.kv_pages) > capacity {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(demand.kv_pages),
                have: capacity,
            });
        }
        match self.commit_to(demand.kv_pages, demand.state_slots)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_every_case() {
        a_seat_past_the_slot_count_is_a_row_only_a_slab_can_refuse();
        a_host_pool_seats_its_rows_first_and_pages_with_the_rest();
    }

    fn a_host_pool_seats_its_rows_first_and_pages_with_the_rest() {
        let mib = 1u64 << 20;
        assert_eq!(
            host_pool(100 * mib, 8, mib, 1000, mib / 16, 0, 0),
            (8, 1472, 0)
        );
        assert_eq!(
            host_pool(20 * mib, 8, mib, 1000, mib / 16, 100, mib / 16),
            (8, 96, 96)
        );
        assert_eq!(
            host_pool(30 * mib, 8, mib, 100, mib / 16, 100, mib / 16),
            (8, 252, 100)
        );
        assert_eq!(host_pool(4 * mib, 8, mib, 1000, mib / 16, 0, 0), (4, 0, 0));
        assert_eq!(host_pool(0, 8, mib, 1000, mib / 16, 0, 0), (0, 0, 0));
        assert_eq!(host_pool(mib, 8, 0, 4, mib / 4, 0, 0), (0, 4, 0));
    }
    // gemma-4 (no slab) with 256 slots reserved seated a lane at 312.
    fn a_seat_past_the_slot_count_is_a_row_only_a_slab_can_refuse() {
        assert!(!slab_row(false, 256, 312).expect("no slab, no ceiling"));
        assert!(slab_row(true, 256, 255).expect("the last row"));
        assert!(matches!(
            slab_row(true, 256, 312),
            Err(Fault::Ceiling {
                what: "recurrent slots",
                need: 313,
                have: 256
            })
        ));
    }
}

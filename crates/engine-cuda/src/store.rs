#![allow(unsafe_code)]
pub mod kv;
pub mod rs;

use engine::transfer::KvRun;
use kernels_cuda::{KvPool, RecurrentPool, Tensor};
use poem_ir::{Attn, CacheRow, Def, Dtype, Operands, Operation, Trace};

use crate::device::elastic::{self, Arena, Commit, PhysicalPool};
use crate::error::{Fault, Result};
use crate::run::{CachePool, CacheTable, PoolSlabs};
use crate::settle::Airborne;
use crate::store::kv::{Facts, Paging};
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

/// The rows a guest program's epilogue holds per lane, each as wide as the
/// readout rows: the logits it reads, and the scaled, probability, mask, noise
/// and Gumbel rows a top-p sampling program derives from them (6292224
/// bytes a lane at a 262144 vocabulary, measured).
pub const PROGRAM_ROWS_A_LANE: u64 = 6;

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
    let pages = pages.min(left.checked_div(page_bytes).unwrap_or(0));
    let left = left - pages * page_bytes;
    let window_pages = window_pages.min(left.checked_div(window_page_bytes).unwrap_or(0));
    let left = left - window_pages * window_page_bytes;
    let pages = pages + left.checked_div(page_bytes).unwrap_or(0);
    (clamp(rows), clamp(pages), clamp(window_pages))
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
    },
    State {
        stride: u64,
        dtype: Dtype,
    },
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
    rows: Vec<Vec<Arena>>,
    shapes: Vec<Shape>,
    pooled: Vec<PoolRow>,
    paging: Paging,
    asked: u64,
    ceiling: u64,
    /// `[model] device_kv_budget`: the bytes the cache rows take, never cut
    /// to fit; `None` fills what the card hands out.
    budget: Option<u64>,
    airborne: Option<Airborne>,
    committed_kv_pages: u32,
    committed_state_slots: u32,
    backed_pages: Vec<u64>,
    /// The windowed pages backed so far, a bit each: the windowed rows and
    /// the compressor slabs their pages key are backed a page at a time, as
    /// the fires' windowed tables (and the windowed moves) first touch it.
    backed_windowed: Vec<u64>,
    windowed_backed: bool,
    buffered: (u64, u64),
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

    fn same_chunk(&self, a: u32, b: u32) -> bool {
        a / self.per_chunk == b / self.per_chunk
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
                self.chunks[chunk as usize] = Some(pin_chunk(bytes)?);
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
/// A shelf chunk, pinned as a copy first touches it, on the request's
/// path: huge pages where the host reserves them, else an anonymous
/// mapping (transparent huge pages) registered with the driver, which
/// pins a chunk several times faster than `cudaHostAlloc` faults and
/// locks its small pages, so a demotion waits on it that much less.
fn pin_chunk(bytes: usize) -> Result<crate::device::Pinned> {
    huge_tier(bytes).map_or_else(|| crate::device::Pinning::uninit(bytes)?.lock(), Ok)
}

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
    pub fn reserve(
        device: i32,
        utilization: f64,
        trace: &Trace,
        paging: Paging,
        budget: Option<u64>,
        facts: &Facts,
    ) -> Result<Pools> {
        let pool = PhysicalPool::open(device, utilization)?;
        let mut rows: Vec<Vec<Arena>> = Vec::with_capacity(trace.caches.len());
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
                    let pages = if windowed {
                        paging.window_pages()
                    } else {
                        paging.pages()
                    };
                    let cells = pages * u64::from(paging.page_size);
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
                    let mut planes_of_row = Vec::with_capacity(planes.len());
                    for width in planes {
                        planes_of_row.push(Arena::reserve(
                            &pool,
                            cells * width * element,
                            "bytes of a kv plane",
                        )?);
                    }
                    rows.push(planes_of_row);
                    shapes.push(Shape::Kv {
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
                    let bytes = stride * u64::from(paging.slots) * elem_bytes(name, *dtype)?;
                    rows.push(vec![Arena::reserve(
                        &pool,
                        bytes,
                        "bytes of a recurrent slab",
                    )?]);
                    shapes.push(Shape::State {
                        stride,
                        dtype: *dtype,
                    });
                }
            }
            debug_assert_eq!(rows.len(), index + 1, "one arena set per cache row");
        }
        let mut pooled = Vec::new();
        for (space, width) in pooled_spaces(trace) {
            // A slab keyed by a windowed row's pages (`space` is the cache row
            // its gather reads) is as long as its windowed pool.
            let windowed = paging.window.is_some()
                && matches!(
                    shapes.get(space as usize),
                    Some(Shape::Kv { windowed: true, .. })
                );
            let row = PoolRow {
                space,
                width,
                windowed,
                planes: Vec::new(),
            };
            let bytes = row.watermark_bytes(
                narrow_pages(paging.pages()),
                narrow_pages(paging.window_pages()),
                paging.page_size,
            );
            let planes = (0..2)
                .map(|_| Arena::reserve(&pool, bytes, "bytes of a compressor state slab"))
                .collect::<Result<Vec<Arena>>>()?;
            pooled.push(PoolRow { planes, ..row });
        }
        Ok(Pools {
            pool,
            rows,
            shapes,
            pooled,
            paging,
            asked: paging.pages(),
            ceiling: paging.pages(),
            budget,
            airborne: None,
            committed_kv_pages: 0,
            committed_state_slots: 0,
            backed_pages: Vec::new(),
            backed_windowed: Vec::new(),
            windowed_backed: false,
            buffered: (0, 1),
            host: None,
            disk: None,
            whole: false,
            swaps: Vec::new(),
        })
    }

    /// A fire commits rs buffered pages by slot, so the fit seats them with
    /// the slots rather than leave them growth past what it sized.
    pub fn seat_buffered(&mut self, slot_bytes: u64, unit: u64) {
        self.buffered = (slot_bytes, unit.max(1));
    }

    pub fn watch(&mut self, airborne: Airborne) {
        self.airborne = Some(airborne);
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.paging
    }

    fn arenas(&self) -> impl Iterator<Item = &Arena> {
        self.rows
            .iter()
            .flatten()
            .chain(self.pooled.iter().flat_map(|row| row.planes.iter()))
    }

    #[must_use]
    pub fn pool_slabs(&self) -> Vec<(u32, PoolSlabs)> {
        self.pooled
            .iter()
            .map(|row| {
                let pages = if row.windowed {
                    self.paging.window_pages()
                } else {
                    self.paging.pages()
                };
                let cells = pages * u64::from(self.paging.page_size);
                let plane = |at: usize| {
                    Tensor::new(
                        row.planes.get(at).map_or(0, elastic::Arena::base),
                        narrow(cells) as u32,
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
                    },
                )
            })
            .collect()
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.arenas().map(elastic::Arena::max_bytes).sum()
    }

    #[must_use]
    pub fn declared_bytes(&self) -> u64 {
        self.declared_at(self.paging.pages())
    }

    /// The watermark the cache rows reach with `pages` kv pages declared and
    /// the paging's state slots, as the map units their arenas back.
    #[must_use]
    pub fn declared_at(&self, pages: u64) -> u64 {
        self.declared_by(pages, self.paging.slots, |arena, _| arena.map_unit())
    }

    /// The same watermark were the planes re-reserved for `pages`: each in
    /// the unit an arena of that size earns, which for a small pool is the
    /// driver's granularity rather than the handle a huge one maps in.
    fn declared_reshaped(&self, pages: u64, slots: u32) -> u64 {
        let (granularity, handle) = (self.pool.granularity(), self.pool.map_unit_bytes());
        self.declared_by(pages, slots, |_, bytes| {
            elastic::map_unit_for(bytes, granularity, handle)
        })
    }

    fn declared_by(&self, pages: u64, slots: u32, unit_of: impl Fn(&Arena, u64) -> u64) -> u64 {
        let windowed = narrow_pages(self.paging.window_pages_at(pages));
        let pages = u32::try_from(pages).unwrap_or(u32::MAX);
        let page_size = self.paging.page_size;
        let mapped = |bytes: u64, arena: &Arena| {
            let unit = unit_of(arena, bytes).max(1);
            bytes.div_ceil(unit).saturating_mul(unit)
        };
        let rows: u64 = self
            .rows
            .iter()
            .zip(self.shapes.iter())
            .map(|(planes, shape)| {
                planes
                    .iter()
                    .enumerate()
                    .map(|(at, arena)| {
                        mapped(
                            watermark_bytes(shape, at, pages, windowed, slots, page_size),
                            arena,
                        )
                    })
                    .sum::<u64>()
            })
            .sum();
        let pooled: u64 = self
            .pooled
            .iter()
            .map(|row| {
                let bytes = row.watermark_bytes(pages, windowed, page_size);
                row.planes
                    .iter()
                    .map(|arena| mapped(bytes, arena))
                    .sum::<u64>()
            })
            .sum();
        let (slot_bytes, unit) = self.buffered;
        let buffered = u64::from(slots)
            .saturating_mul(slot_bytes)
            .div_ceil(unit)
            .saturating_mul(unit);
        rows.saturating_add(pooled).saturating_add(buffered)
    }

    /// Re-reserves the kv planes and compressor slabs for a pool of `pages`,
    /// so each maps in the unit its size earns rather than the one the asked
    /// count gave it. Only before anything is committed to them: a mapped
    /// unit has a base a table may already hold.
    fn reshape(&mut self, pages: u64, slots: u32) -> Result<()> {
        let narrow = u32::try_from(pages).unwrap_or(u32::MAX);
        let windowed = narrow_pages(self.paging.window_pages_at(pages));
        let page_size = self.paging.page_size;
        let fresh = |arena: &Arena| {
            if arena.committed_bytes() == 0 {
                Ok(())
            } else {
                Err(Fault::program(
                    "store::reshape",
                    "a cache plane reshaped under committed pages",
                ))
            }
        };
        for (planes, shape) in self.rows.iter_mut().zip(self.shapes.iter()) {
            let (label, changed) = match shape {
                Shape::Kv { .. } => ("bytes of a kv plane", true),
                Shape::State { .. } => ("bytes of a recurrent slab", slots != self.paging.slots),
            };
            if !changed {
                continue;
            }
            for (at, arena) in planes.iter_mut().enumerate() {
                fresh(arena)?;
                *arena = Arena::reserve(
                    &self.pool,
                    watermark_bytes(shape, at, narrow, windowed, slots, page_size),
                    label,
                )?;
            }
        }
        for row in &mut self.pooled {
            let bytes = row.watermark_bytes(narrow, windowed, page_size);
            for arena in &mut row.planes {
                fresh(arena)?;
                *arena = Arena::reserve(&self.pool, bytes, "bytes of a compressor state slab")?;
            }
        }
        self.ceiling = pages;
        self.paging.pages = pages;
        self.paging.slots = slots;
        self.windowed_backed = false;
        self.backed_windowed.clear();
        Ok(())
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
        let fresh = self
            .rows
            .iter()
            .flatten()
            .chain(self.pooled.iter().flat_map(|row| row.planes.iter()))
            .all(|arena| arena.committed_bytes() == 0);
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

    pub(crate) fn reserve_arena(&self, max_bytes: u64, label: &'static str) -> Result<Arena> {
        Arena::reserve(&self.pool, max_bytes, label)
    }

    pub(crate) fn commit_arena(&mut self, arena: &mut Arena, bytes: u64) -> Result<()> {
        let mut targets = [elastic::Target {
            arena,
            want: elastic::Want::Prefix(bytes),
        }];
        match elastic::commit_atomically(&mut self.pool, &mut targets)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    /// Backs a prefix of `bytes` of an arena beside the pool's budget
    /// (`elastic::commit_beside`).
    pub(crate) fn commit_beside(&self, arena: &mut Arena, bytes: u64) -> Result<()> {
        elastic::commit_beside(&self.pool, arena, &elastic::Want::Prefix(bytes))
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.arenas().map(elastic::Arena::committed_bytes).sum()
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.arenas().map(elastic::Arena::high_water_bytes).sum()
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
                Shape::State { stride, dtype } => stride * u64::from(elem_size(dtype)),
                Shape::Kv { .. } => 0,
            })
            .sum()
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

    /// Zeroes page 0 of the windowed rows once they are backed: every page
    /// behind a window reads it, and a masked column still multiplies its
    /// value, so it must hold no NaN.
    fn zero_null_page(&mut self) -> Result<()> {
        if self.windowed_backed || !self.has_windowed() {
            return Ok(());
        }
        for (arena, bytes) in self.windowed_planes() {
            crate::device::zero_span(arena.span(0, bytes)?, usize::try_from(bytes).unwrap_or(0))?;
        }
        self.windowed_backed = true;
        Ok(())
    }

    /// Backs the not-yet-backed windowed pages `pages` (and the null page behind a
    /// window) in the windowed rows and compressor slabs: the pool grows on first
    /// use, costing only the windows the seated sequences hold.
    pub fn ensure_windowed(&mut self, pages: impl IntoIterator<Item = u64>) -> Result<()> {
        if !self.has_windowed() {
            return Ok(());
        }
        let capacity = self.paging.window_pages();
        let mut fresh: Vec<u64> = core::iter::once(0)
            .chain(pages)
            .filter(|&page| {
                !self
                    .backed_windowed
                    .get((page / 64) as usize)
                    .is_some_and(|word| word & (1u64 << (page % 64)) != 0)
            })
            .collect();
        fresh.sort_unstable();
        fresh.dedup();
        if fresh.is_empty() {
            return Ok(());
        }
        if let Some(&page) = fresh.iter().find(|&&page| page >= capacity) {
            return Err(Fault::Ceiling {
                what: "windowed kv pages",
                need: page + 1,
                have: capacity,
            });
        }
        // Runs of consecutive pages, a range each.
        let mut runs: Vec<(u64, u64)> = Vec::new();
        for &page in &fresh {
            match runs.last_mut() {
                Some((first, count)) if *first + *count == page => *count += 1,
                _ => runs.push((page, 1)),
            }
        }
        let page_size = u64::from(self.paging.page_size);
        let Pools {
            pool,
            rows,
            shapes,
            pooled,
            ..
        } = self;
        let mut targets = Vec::new();
        let ranges = |page_bytes: u64| {
            elastic::Want::Ranges(
                runs.iter()
                    .map(|&(first, count)| (first * page_bytes, count * page_bytes))
                    .collect(),
            )
        };
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            if !matches!(shape, Shape::Kv { windowed: true, .. }) {
                continue;
            }
            for (at, arena) in planes.iter_mut().enumerate() {
                let page_bytes = watermark_bytes(shape, at, 0, 1, 0, self.paging.page_size);
                targets.push(elastic::Target {
                    arena,
                    want: ranges(page_bytes),
                });
            }
        }
        for row in pooled.iter_mut().filter(|row| row.windowed) {
            let page_bytes = page_size * row.width * u64::from(elem_size(POOL_STATE));
            for arena in row.planes.iter_mut() {
                targets.push(elastic::Target {
                    arena,
                    want: ranges(page_bytes),
                });
            }
        }
        match elastic::commit_atomically(pool, &mut targets)? {
            Commit::Committed => {}
            refusal => return Err(refuse(&self.pool, refusal)),
        }
        for &page in &fresh {
            let (word, bit) = ((page / 64) as usize, page % 64);
            if self.backed_windowed.len() <= word {
                self.backed_windowed.resize(word + 1, 0);
            }
            self.backed_windowed[word] |= 1u64 << bit;
        }
        self.zero_null_page()
    }

    #[must_use]
    pub fn bases(&self) -> Vec<u64> {
        self.arenas().map(elastic::Arena::base).collect()
    }

    pub fn table(&self, seats: &Seats) -> Result<CacheTable> {
        let mut table = Vec::with_capacity(self.shapes.len());
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
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
                    let plane = |at: usize, width: u64| {
                        Tensor::new(
                            planes.get(at).map_or(0, elastic::Arena::base),
                            u32::try_from(cells).unwrap_or(u32::MAX),
                            u32::try_from(width).unwrap_or(u32::MAX),
                            dtype,
                        )
                    };
                    CachePool::Kv {
                        space,
                        pool: KvPool {
                            keys: plane(0, keys_width),
                            values: plane(values_plane, values_width),
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
                        },
                    }
                }
                Shape::State { stride, dtype } => CachePool::Recurrent(RecurrentPool {
                    write_state: seats.write_state,
                    write_state_mask: seats.write_state_mask,
                    commit_len: seats.commit_len,
                    begin_at: seats.begin_at,
                    fused_decay: false,
                    snapshots: seats.snapshots,
                    slab: Tensor::new(
                        planes.first().map_or(0, elastic::Arena::base),
                        self.paging.slots,
                        narrow(stride) as u32,
                        dtype,
                    ),
                    slot_ids: seats.slot_ids,
                    slot_stride_elems: stride as i64,
                    conv_slab: Tensor::new(
                        planes.first().map_or(0, elastic::Arena::base),
                        self.paging.slots,
                        narrow(stride) as u32,
                        dtype,
                    ),
                    conv_stride: stride as i64,
                }),
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

    pub fn copy_slot(&mut self, stream: *mut core::ffi::c_void, src: u32, dst: u32) -> Result<()> {
        for slot in [src, dst] {
            if !slab_row(self.has_state(), self.paging.slots, slot)? {
                return Ok(());
            }
        }
        if src == dst {
            return Ok(());
        }
        self.ensure_state(src.max(dst) + 1)?;
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            crate::device::copy_d2d(
                stream,
                arena.span(u64::from(dst) * bytes, bytes)?,
                arena.span(u64::from(src) * bytes, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
        }
        Ok(())
    }

    /// The state rows of `slot`, backed first; none when the load keeps no state.
    pub fn state_spans(&mut self, slot: u32) -> Result<Vec<(u64, u64)>> {
        if !slab_row(self.has_state(), self.paging.slots, slot)? {
            return Ok(Vec::new());
        }
        self.ensure_state(slot + 1)?;
        let mut spans = Vec::new();
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            spans.push((arena.span(u64::from(slot) * bytes, bytes)?, bytes));
        }
        Ok(spans)
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
        self.ensure_kv(u32::try_from(highest).unwrap_or(u32::MAX))?;
        self.ensure_windowed(
            windowed
                .iter()
                .flat_map(|span| [u64::from(span.src_page), u64::from(span.dst_page)]),
        )?;
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::Kv {
                windowed: in_window,
                dtype,
                keys_width,
                values_width,
                ..
            } = *shape
            else {
                continue;
            };
            let element = u64::from(elem_size(dtype));
            let moves = if in_window { windowed } else { moves };
            for (at, arena) in planes.iter().enumerate() {
                let width = if at == 0 { keys_width } else { values_width };
                let cell = width * element;
                for span in moves {
                    if span.tokens == 0 {
                        continue;
                    }
                    let bytes = u64::from(span.tokens) * cell;
                    let src =
                        (u64::from(span.src_page) * page_size + u64::from(span.src_token)) * cell;
                    let dst =
                        (u64::from(span.dst_page) * page_size + u64::from(span.dst_token)) * cell;
                    if src == dst {
                        continue;
                    }
                    crate::device::copy_d2d(
                        stream,
                        arena.span(dst, bytes)?,
                        arena.span(src, bytes)?,
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
    fn paged_planes(&self) -> Vec<(&Arena, u64)> {
        let mut planes = self.kv_planes(false);
        planes.extend(self.pooled_planes(false));
        planes
    }

    /// The compressor slabs keyed by a windowed space's pages (or not), and
    /// the bytes a page holds in each.
    fn pooled_planes(&self, windowed: bool) -> Vec<(&Arena, u64)> {
        let page_size = u64::from(self.paging.page_size);
        self.pooled
            .iter()
            .filter(|row| row.windowed == windowed)
            .flat_map(|row| {
                let bytes = page_size * row.width * u64::from(elem_size(POOL_STATE));
                row.planes.iter().map(move |arena| (arena, bytes))
            })
            .collect()
    }

    /// The windowed planes a windowed page spans: the windowed kv rows' and
    /// the compressor slabs their pages key.
    fn windowed_planes(&self) -> Vec<(&Arena, u64)> {
        let mut planes = self.kv_planes(true);
        planes.extend(self.pooled_planes(true));
        planes
    }

    /// The planes of the kv rows that are (or are not) `windowed`, and the
    /// bytes a page holds in each.
    fn kv_planes(&self, windowed: bool) -> Vec<(&Arena, u64)> {
        let page_size = u64::from(self.paging.page_size);
        let mut planes = Vec::new();
        for (arenas, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::Kv {
                windowed: in_window,
                dtype,
                keys_width,
                values_width,
                ..
            } = *shape
            else {
                continue;
            };
            if in_window != windowed {
                continue;
            }
            for (at, arena) in arenas.iter().enumerate() {
                let width = if at == 0 { keys_width } else { values_width };
                planes.push((arena, page_size * width * u64::from(elem_size(dtype))));
            }
        }
        planes
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
        self.note_backed(&[(0, u64::from(pages))]);
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
        let pages = u32::try_from(self.paging.pages()).unwrap_or(u32::MAX);
        let mut regions = Vec::new();
        for (arena, page_bytes) in self.paged_planes() {
            let len = u64::from(pages) * page_bytes;
            regions.push(engine::KvRegion {
                base: arena.span(0, len)?,
                len,
                page_stride: page_bytes,
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
        let planes = |planes: Vec<(&Arena, u64)>| planes.into_iter().map(|(_, b)| b).collect();
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
        let highest = device.iter().map(|&page| page + 1).max().unwrap_or(0);
        self.ensure_kv(highest)?;
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
            for &(arena, page_bytes) in &planes {
                let on_device = arena.span(u64::from(page) * page_bytes, page_bytes)?;
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
        let highest = device.iter().map(|&page| page + 1).max().unwrap_or(0);
        self.ensure_kv(highest)?;
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
            let mut runs: Vec<(u32, u32, u32)> = Vec::new();
            for &(d, h) in pairs {
                match runs.last_mut() {
                    Some((d0, h0, n))
                        if *d0 + *n == d && *h0 + *n == h && shelf.same_chunk(*h0, h) =>
                    {
                        *n += 1;
                    }
                    _ => runs.push((d, h, 1)),
                }
            }
            for (plane, (arena, page_bytes)) in planes.into_iter().enumerate() {
                for &(d, h, n) in &runs {
                    let bytes = u64::from(n) * page_bytes;
                    let on_device = arena.span(u64::from(d) * page_bytes, bytes)?;
                    let on_host = shelf.at(h, plane)?;
                    let (dst, src) = if to_host {
                        (on_host, on_device)
                    } else {
                        (on_device, on_host)
                    };
                    spans.push((dst, src, usize::try_from(bytes).unwrap_or(0)));
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
        for (row, (arenas, shape)) in self.rows.iter().zip(&self.shapes).enumerate() {
            if matches!(
                shape,
                Shape::Kv {
                    windowed: false,
                    ..
                }
            ) {
                rows.extend(std::iter::repeat_n(row, arenas.len()));
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
        self.ensure_kv(device.iter().map(|&page| page + 1).max().unwrap_or(0))?;
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
                let (arena, bytes) = all[plane];
                let on_host = shelf.at(h, plane)?;
                let on_device = arena.span(u64::from(d) * bytes, bytes)?;
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
        let shelf = self.paged_shelf(slots)?;
        let mut out = Vec::new();
        for &slot in slots {
            for (plane, &bytes) in shelf.planes.iter().enumerate() {
                let (pinned, at) = shelf.place(slot, plane).expect("backed above");
                out.extend_from_slice(pinned.view(at, bytes).expect("inside its chunk"));
            }
        }
        Ok(out)
    }

    /// Writes `bytes`, as [`Self::read_host_kv`] lays them out, into host kv
    /// slots `slots`.
    pub fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        let slot_bytes = usize::try_from(self.host_kv_slot_bytes()).unwrap_or(0);
        if bytes.len() != slots.len() * slot_bytes {
            return Err(Fault::Ceiling {
                what: "bytes of host kv slots",
                need: (slots.len() * slot_bytes) as u64,
                have: bytes.len() as u64,
            });
        }
        let shelf = self.paged_shelf(slots)?;
        let mut from = 0;
        for &slot in slots {
            for (plane, &len) in shelf.planes.iter().enumerate() {
                let (pinned, at) = shelf.place(slot, plane).expect("backed above");
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

    /// The host kv shelf, with the chunks holding `slots` pinned.
    fn paged_shelf(&mut self, slots: &[u32]) -> Result<&Shelf> {
        let Some(host) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what: "host kv pages",
                need: 1,
                have: 0,
            });
        };
        host.paged.back(slots.iter().copied(), "host kv pages")?;
        Ok(&host.paged)
    }

    /// `slot`'s recurrent state, read on `stream` behind the fires that
    /// wrote it.
    pub fn state_bytes(&mut self, stream: *mut core::ffi::c_void, slot: u32) -> Result<Vec<u8>> {
        if !slab_row(self.has_state(), self.paging.slots, slot)? {
            return Ok(Vec::new());
        }
        self.ensure_state(slot + 1)?;
        let mut out = Vec::new();
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let at = out.len();
            out.resize(at + usize::try_from(bytes).unwrap_or(0), 0);
            crate::device::copy_d2h_on(
                stream,
                arena.span(u64::from(slot) * bytes, bytes)?,
                &mut out[at..],
            )?;
        }
        Ok(out)
    }

    fn zero_slot(&mut self, stream: Option<*mut core::ffi::c_void>, slot: u32) -> Result<()> {
        if !slab_row(self.has_state(), self.paging.slots, slot)? {
            return Ok(());
        }
        self.ensure_state(slot + 1)?;
        for (planes, shape) in self.rows.iter().zip(&self.shapes) {
            let Shape::State { stride, dtype } = *shape else {
                continue;
            };
            let Some(arena) = planes.first() else {
                continue;
            };
            let bytes = stride * u64::from(elem_size(dtype));
            let at = arena.span(u64::from(slot) * bytes, bytes)?;
            let len = usize::try_from(bytes).unwrap_or(0);
            match stream {
                Some(stream) => crate::device::zero_span_on(stream, at, len)?,
                None => crate::device::zero_span(at, len)?,
            }
        }
        Ok(())
    }

    fn ensure_kv(&mut self, pages: u32) -> Result<()> {
        let capacity = self.paging.pages();
        if u64::from(pages) > capacity {
            return Err(Fault::Ceiling {
                what: "kv pages",
                need: u64::from(pages),
                have: capacity,
            });
        }
        match self.commit_to(pages, 0)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
    }

    fn ensure_state(&mut self, slots: u32) -> Result<()> {
        match self.commit_to(0, slots)? {
            Commit::Committed => Ok(()),
            refusal => Err(refuse(&self.pool, refusal)),
        }
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
        if !(demand.kv_pages <= self.committed_kv_pages
            && demand.state_slots <= self.committed_state_slots
            && kv_ranges
                .iter()
                .all(|&(first, count)| self.pages_backed(first, count)))
        {
            match self.commit_ranges(demand.kv_pages, demand.state_slots, Some(kv_ranges))? {
                Commit::Committed => self.note_backed(kv_ranges),
                refusal => return Err(refuse(&self.pool, refusal)),
            }
        }
        self.ensure_windowed(window_pages.iter().copied())
    }

    fn pages_backed(&self, first: u64, count: u64) -> bool {
        (first..first.saturating_add(count)).all(|page| {
            self.backed_pages
                .get((page / 64) as usize)
                .is_some_and(|word| word & (1u64 << (page % 64)) != 0)
        })
    }

    fn note_backed(&mut self, ranges: &[(u64, u64)]) {
        for &(first, count) in ranges {
            for page in first..first.saturating_add(count) {
                let (word, bit) = ((page / 64) as usize, page % 64);
                if self.backed_pages.len() <= word {
                    self.backed_pages.resize(word + 1, 0);
                }
                self.backed_pages[word] |= 1u64 << bit;
            }
        }
    }

    fn commit_ranges(
        &mut self,
        kv_pages: u32,
        state_slots: u32,
        kv_ranges: Option<&[(u64, u64)]>,
    ) -> Result<Commit> {
        let kv_pages = kv_pages.max(self.committed_kv_pages);
        let state_slots = state_slots.max(self.committed_state_slots);
        let page_size = self.paging.page_size;
        let windowed = narrow_pages(self.paging.window_pages());
        let Pools {
            pool,
            rows,
            shapes,
            pooled,
            ..
        } = self;
        let mut targets = Vec::new();
        for (planes, shape) in rows.iter_mut().zip(shapes.iter()) {
            for (at, arena) in planes.iter_mut().enumerate() {
                let want = match (shape, kv_ranges) {
                    // The windowed pool backs its pages as they are first
                    // used (`ensure_windowed`).
                    (Shape::Kv { windowed: true, .. }, _) => continue,
                    (Shape::Kv { .. }, Some(ranges)) => {
                        let page_bytes = watermark_bytes(shape, at, 1, 0, 0, page_size);
                        elastic::Want::Ranges(
                            ranges
                                .iter()
                                .map(|&(first, count)| (first * page_bytes, count * page_bytes))
                                .collect(),
                        )
                    }
                    _ => elastic::Want::Prefix(watermark_bytes(
                        shape,
                        at,
                        kv_pages,
                        0,
                        state_slots,
                        page_size,
                    )),
                };
                targets.push(elastic::Target { arena, want });
            }
        }
        for row in pooled.iter_mut().filter(|row| !row.windowed) {
            let bytes = row.watermark_bytes(kv_pages, windowed, page_size);
            for arena in row.planes.iter_mut() {
                targets.push(elastic::Target {
                    arena,
                    want: elastic::Want::Prefix(bytes),
                });
            }
        }
        let outcome = elastic::commit_atomically(pool, &mut targets)?;
        if outcome == Commit::Committed {
            self.committed_kv_pages = kv_pages;
            self.committed_state_slots = state_slots;
        }
        Ok(outcome)
    }
}

const POOL_STATE: Dtype = Dtype::Bf16;

#[derive(Debug)]
struct PoolRow {
    space: u32,
    width: u64,
    /// Keyed by a windowed space's pages: as long as its windowed pool.
    windowed: bool,
    planes: Vec<Arena>,
}

impl PoolRow {
    fn watermark_bytes(&self, kv_pages: u32, window_pages: u32, page_size: u32) -> u64 {
        let pages = if self.windowed {
            window_pages
        } else {
            kv_pages
        };
        u64::from(pages) * u64::from(page_size) * self.width * u64::from(elem_size(POOL_STATE))
    }
}

const fn compressor_coff(ratio: u32) -> u64 {
    if ratio == 4 { 2 } else { 1 }
}

fn pooled_spaces(trace: &Trace) -> Vec<(u32, u64)> {
    let mut spaces: Vec<(u32, u64)> = Vec::new();
    for node in &trace.nodes {
        // A gather, or one the compressor's fused op holds.
        let gather = match &node.op {
            Operation::Attn(a) => a,
            Operation::Fused(poem_ir::Fused::PoolCompress { gather, .. }) => &**gather,
            _ => continue,
        };
        let Attn::PoolGather {
            pages,
            head_dim,
            ratio,
            ..
        } = gather
        else {
            continue;
        };
        let Some(Def::Cache(space)) = trace.values.get(pages.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let width = compressor_coff(*ratio) * u64::from(*head_dim);
        if width == 0 {
            continue;
        }
        match spaces.iter_mut().find(|(held, _)| *held == *space) {
            Some((_, held)) => *held = (*held).max(width),
            None => spaces.push((*space, width)),
        }
    }
    spaces
}

fn watermark_bytes(
    shape: &Shape,
    plane: usize,
    kv_pages: u32,
    window_pages: u32,
    state_slots: u32,
    page_size: u32,
) -> u64 {
    match *shape {
        Shape::Kv {
            windowed,
            dtype,
            keys_width,
            values_width,
            ..
        } => {
            let width = if plane == 0 { keys_width } else { values_width };
            let pages = if windowed { window_pages } else { kv_pages };
            u64::from(pages) * u64::from(page_size) * width * u64::from(elem_size(dtype))
        }
        Shape::State { stride, dtype } => {
            u64::from(state_slots) * stride * u64::from(elem_size(dtype))
        }
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

fn narrow_pages(pages: u64) -> u32 {
    u32::try_from(pages).unwrap_or(u32::MAX)
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

impl engine::frame::Supply for Pools {
    type Error = Fault;

    fn commit(&mut self, demand: engine::frame::Demand) -> Result<()> {
        if let Some(highest) = demand.state_slots.checked_sub(1) {
            slab_row(self.has_state(), self.paging.slots, highest)?;
        }
        let capacity =
            u64::from(self.paging.slots).saturating_mul(u64::from(self.paging.pages_per_slot));
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

    /// A chunk is the driver's registration of a mapping, not a
    /// `cudaHostAlloc`: a chunk a demotion first touches pinned there held
    /// the next prefill, which waits on the demotion, for the whole pin.
    #[cfg(feature = "device")]
    #[test]
    fn a_shelf_chunk_is_a_registered_mapping() {
        let mut shelf = Shelf::new(vec![1 << 20, 1 << 20], 256);
        shelf.back([0], "host kv pages").expect("the chunk pins");
        let chunk = shelf.chunks[0].as_ref().expect("backed");
        assert!(chunk.registered(), "a shelf chunk registers its mapping");
        assert_ne!(chunk.device(), 0, "and maps it for the device");
        let at = shelf
            .at(127, 1)
            .expect("the chunk's last slot's last plane");
        // SAFETY: inside the chunk just pinned, 1 MiB a plane a slot.
        unsafe { core::ptr::write_bytes(at as *mut u8, 7, 1 << 20) };
    }

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
            (8, 192, 0)
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

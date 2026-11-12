#![allow(unsafe_code)]
pub mod chunks;
mod host;
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
        // Dense weights always live on the device: past the share, no other
        // knob makes room for them.
        let room = self.ceiling.saturating_sub(self.floor);
        if self.weights > room {
            return Err(Fault::Residency(format!(
                "the card does not hold this deployment's weights: they take {weights} bytes \
                 on the device, and of its {card} bytes `[model] device_memory_share` allows \
                 pie {ceiling}, {room} past the driver's safety floor. Dense weights always \
                 live in device memory (only routed experts and token-id tables leave it, \
                 through the weight cache), so serve a quantized deployment of this model, \
                 split it across devices with tensor parallelism, or raise `[model] \
                 device_memory_share`.",
                weights = self.weights,
                card = self.card,
                ceiling = self.ceiling,
            )));
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
             device_memory_share`, or serve a quantized deployment of the model.",
            card = self.card,
            ceiling = self.ceiling,
            weights = self.weights,
            floor = self.floor,
            pool = self.pool,
            minimum = self.minimum,
        )))
    }
}

/// What `Pools::fit_the_card` sized the pool to: the kv pages asked, the
/// pages its chunks would hold were every one the kv rows', the chunks it
/// has, the bytes the card had for them, and what was held back from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fitted {
    pub asked: u64,
    pub fit: u64,
    pub room: u64,
    pub held: u64,
    pub bodies: u64,
    pub spare: u64,
    pub chunks: u32,
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

/// One working set's cache rows: its kv pages at the declared context and
/// its recurrent record, the least `Pools::fit_the_card` seats before
/// refusing.
pub fn least_sequence_bytes(trace: &Trace, paging: Paging) -> Result<u64> {
    one_slot_bytes(trace, paging)
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
/// trace declares, or (`stated`, nonzero) what the load's plan keeps there
/// once the weight cache's planes leave, plus what else rides along
/// resident.
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
        windowed: bool,
        /// The list its pages are slots of: 0 the kv list, `n` window list
        /// `n - 1`.
        list: u8,
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
pub struct TableSeat {
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
    /// One a page table: the kv list's, then the window list's.
    pub page_tables: Vec<TableSeat>,
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
    /// The row-streamed tables' caches, a shape each ([`Pools::seat_rows`]).
    row_caches: Vec<RowCache>,
    /// The routed experts' caches, a shape each ([`Pools::seat_experts`]).
    expert_caches: Vec<ExpertCache>,
    /// Each window a windowed kv row is declared with, and the list its
    /// pages are slots of.
    windows_of: Vec<(u32, u8)>,
    /// Whether the runtime holds the weights' resident set for the load's
    /// life (`WeightResidency::held`).
    weights_held: bool,
    /// Whether the host copies the caches fill from are gone
    /// ([`Pools::drop_weight_hosts`]).
    hosts_dropped: bool,
    /// The caches' host copies are the serving file's mapping, which no
    /// fire may read in place ([`Pools::file_weights`]).
    weights_filed: bool,
    /// The pinned copy misses are read from, where the load left the
    /// weights in the file and the runtime streams them after all.
    miss_copy: Option<crate::device::Pinned>,
    /// The spans of the serving file's mapping copied out of it since the
    /// pages were last let go ([`Pools::release_file_pages`]).
    file_spans: Vec<(u64, usize)>,
    /// The load seated every weight atom in slots of its own choosing
    /// ([`Pools::preseat_weights`]), until the runtime's first residency.
    preseated: bool,
    /// Arming over a streamed set: an atom on the host copy is read
    /// through a slot of its shape that the pool holds instead
    /// ([`Pools::stand_in`]).
    stand_in: bool,
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
    /// The chunks the fit left the cache rows, of the range's.
    fitted: u32,
    /// The range is reserved anew to the fit even where it maps on demand:
    /// a smaller activation rung hands state the chunks under the
    /// activation, which must be the fit's, not past it.
    whole_fit: bool,
    airborne: Option<Airborne>,
    host: Option<HostKv>,
    disk: Option<DiskKv>,
    /// The shared pool's host tier (CONTRACT-BC 22): chunks of the state
    /// pool's size, a slot of any shape at the place its host ref names, in
    /// pageable memory behind a page-locked staging ring ([`host`]).
    tier: Option<host::HostTier>,
    /// Every declared page and slot is backed ([`Pools::back_whole`]), and
    /// never trimmed since.
    whole: bool,
    /// The moves to or from the host tier still on the swap stream, and
    /// the read-outs on the tap stream, each with the device pages and the
    /// host slots it holds.
    swaps: Vec<Swap>,
}

/// A tier a slot copy reads or writes (CONTRACT-BC 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Device,
    Host,
    Disk,
}

/// A move of kv pages between the device and their host slots, on the
/// swap stream, or a tap's read-out: its device pages, the host slots it
/// reads or writes, and the event it lands at.
#[derive(Debug)]
struct Swap {
    landed: crate::device::graph::Event,
    paged: Vec<u32>,
    host: Vec<u32>,
    /// It holds slots a fire names by no kv run (a window list's, a
    /// record's, the rs list's): every fire waits for it.
    wide: bool,
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
/// The pinned host kv slots a page's bytes are read out to and written in
/// from by plane ([`Pools::read_host_kv`], a tap's read-out). Each paged
/// plane keeps its host pages side by side, as its device arena does, so a
/// run of pages moves in one copy. What a runtime parks moves through the
/// host tier instead ([`Pools::copy_tier_slots`]).
#[derive(Debug)]
struct HostKv {
    paged: Shelf,
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

/// The spare chunks under the activation a smaller activation rung may
/// still hand state, backed beside the budget (the few the fit takes back
/// once the bodies are armed); more, and no smaller rung is stated.
const SPARE_UNDER_RUNGS: u32 = 4;

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
        chunk_bytes: Option<u64>,
        experts: &[usize],
        rows: bool,
        one_window: bool,
    ) -> Result<Pools> {
        let pool = PhysicalPool::open(device, utilization)?;
        let layout = lay(
            trace,
            &paging,
            budgets,
            chunk_bytes,
            experts,
            rows,
            one_window,
        )?;
        let mut shapes = Vec::with_capacity(trace.caches.len());
        let mut windows_of: Vec<(u32, u8)> = Vec::new();

        for (index, row) in trace.caches.iter().enumerate() {
            match row {
                CacheRow::Kv {
                    name,
                    planes,
                    dtype,
                    window,
                    live,
                    rows: quantized,
                    ..
                } => {
                    let element = elem_bytes(name, *dtype)?;
                    let windowed = window.is_some() && paging.window.is_some();
                    // A windowed row pages in the window list the recipe
                    // lays it in: its own window's, or a wider one's where
                    // the windows are merged.
                    let list = match window.filter(|_| windowed) {
                        None => 0u8,
                        Some(held) => layout
                            .windows
                            .iter()
                            .position(|shape| shape.offset_of(index as u32).is_some())
                            .map(|at| at as u8 + 1)
                            .ok_or_else(|| {
                                Fault::program(
                                    "store::reserve",
                                    format!(
                                        "cache `{name}` is read through a window of {held} \
                                         tokens, and the chunk recipe lays it in no window \
                                         list"
                                    ),
                                )
                            })?,
                    };
                    if let Some(held) = window.filter(|_| list != 0) {
                        windows_of.push((held, list));
                    }
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
                        .shape(Kind::paged(list))
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
                        windowed,
                        list,
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
        for (row, width) in poem_compiler::chunk::census::pooled(trace) {
            // A state keyed by a windowed row's pages (`row` is the cache
            // row its gather reads) lies in the windowed pages' slots.
            let list = match shapes.get(row as usize) {
                Some(Shape::Kv { list, .. }) if paging.window.is_some() => *list,
                _ => 0,
            };
            let windowed = list != 0;
            let offset = layout
                .shape(Kind::paged(list))
                .and_then(|shape| shape.pool_of(row))
                .ok_or_else(|| {
                    Fault::program(
                        "store::reserve",
                        format!(
                            "the compressor state keyed by cache row {row} has no run in \
                             the chunk recipe's slot"
                        ),
                    )
                })?;
            pooled.push(PoolRow {
                row,
                width,
                windowed,
                list,
                offset,
            });
        }
        let total = chunks_at(&layout, &paging, paging.pages(), paging.slots);
        let chunks = chunks::Chunks::reserve(&pool, layout, total, activation)?;
        Ok(Pools {
            pool,
            chunks,
            row_caches: Vec::new(),
            expert_caches: Vec::new(),
            windows_of,
            weights_held: false,
            hosts_dropped: false,
            weights_filed: false,
            miss_copy: None,
            preseated: false,
            stand_in: false,
            file_spans: Vec::new(),
            records: std::collections::HashMap::new(),
            id_lanes: 1,
            snap_entries: 1,
            shapes,
            pooled,
            paging,
            asked: paging.pages(),
            ceiling: paging.pages(),
            budget,
            fitted: total,
            whole_fit: false,
            airborne: None,
            host: None,
            disk: None,
            tier: None,
            whole: false,
            swaps: Vec::new(),
        })
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
                    .shape(Kind::paged(row.list))
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
                    row.row,
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

    /// The watermark the pool reaches: the chunks the fit left it, as the
    /// map units the range backs them in.
    #[must_use]
    pub fn declared_bytes(&self) -> u64 {
        self.chunk_bytes(self.fitted)
    }

    /// The bytes `chunks` chunks take backed: the last map unit whole.
    fn chunk_bytes(&self, chunks: u32) -> u64 {
        let unit = self.chunks.map_unit().max(1);
        u64::from(chunks)
            .saturating_mul(self.chunks.layout().chunk_bytes)
            .div_ceil(unit)
            .saturating_mul(unit)
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

    /// What a load's synthetic fires may name, inside the chunks the fit
    /// left the pool: of `records` recurrent records and `pages` kv pages
    /// wanted, the most records beside every page, or one record and the
    /// most pages; and the windowed pages beside those pages, the null page
    /// among them. A synthetic lane's rows are never read back, so lanes
    /// may share what they name; the fit holds one working set, so one
    /// record and its pages always fit.
    #[must_use]
    pub fn synthetic_room(&self, records: u32, pages: u64) -> (u32, u64, u64) {
        let fitted = self.fitted_chunks();
        let room = fitted.saturating_sub(self.chunks.weight_chunks_under(fitted));
        let layout = self.chunks.layout();
        let state = self.has_state();
        let fits = |records: u32, pages: u64| {
            chunks_at(layout, &self.paging, pages, if state { records } else { 0 }) <= room
        };
        let largest = |lo: u64, hi: u64, ok: &dyn Fn(u64) -> bool| {
            let (mut lo, mut hi) = (lo, hi);
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if ok(mid) { lo = mid } else { hi = mid - 1 }
            }
            lo
        };
        let records = records.max(1);
        let pages = pages.max(1);
        let (records, pages) = if fits(records, pages) {
            (records, pages)
        } else if fits(1, pages) {
            let most = largest(1, u64::from(records), &|r| fits(r as u32, pages));
            (u32::try_from(most).unwrap_or(1), pages)
        } else {
            (1, largest(1, pages, &|p| fits(1, p)))
        };
        (records, pages, self.paging.window_pages_at(pages))
    }

    /// Sizes the pool to the card: every whole chunk the card has room for,
    /// each any shape's to take, provided they seat one working set at the
    /// declared context ([`engine::chunk::fit`]). The room is the elastic
    /// budget as it stands now, so the fit is called once everything else
    /// this load puts on the device is resident, and again after arming,
    /// when the bodies have taken what they take and the rest is the pool's.
    /// `programs` is held for the guests' programs throughout; `bodies` is
    /// the configured allowance, which `bodies_allowance` holds down to what
    /// the room affords, and both yield before that one working set does,
    /// since a pool under it is no deployment. `resident` spells what is on
    /// the device already, for the refusal when not even that fits.
    /// `settled` says the activation has reached its high water: every
    /// composition fired while the bodies armed.
    pub fn fit_the_card(
        &mut self,
        bodies: u64,
        programs: u64,
        settled: bool,
        resident: &dyn core::fmt::Display,
    ) -> Result<Fitted> {
        // The room is what the card has free plus every byte the range has
        // backed (its committed segments and held spans; a flat range, once
        // allocated, the whole of it: the chunks the kv rows have not yet
        // taken are backed too, and so not free), less the activation the
        // cache rows never take. Until the activation is settled that is its
        // whole span, held or still to be (the first fire's activation, and
        // the room held for the bodies, must find it); once settled (every
        // composition armed, the activation at its high water), only the
        // span a fire backed: what no fire backed is no fire's, and the cache
        // rows take it.
        let activation = if settled {
            self.chunks
                .activation_bytes()
                .saturating_sub(self.chunks.activation_unheld_bytes())
        } else {
            self.chunks.activation_bytes()
        };
        let live = self
            .pool
            .spare_bytes(0)?
            .saturating_add(self.chunks.backed_bytes())
            .saturating_sub(activation);
        let layout = self.chunks.layout();
        let chunk = layout.chunk_bytes;
        // One working set at the declared context beside the null chunk:
        // its kv pages, its windowed pages and its recurrent record.
        let one_slot = u64::from(self.paging.pages_per_slot).min(self.ceiling);
        let slots = u32::from(self.has_state());
        let least = chunks_at(layout, &self.paging, one_slot, slots);
        let need = self.chunk_bytes(least);
        // A stated budget is the pool's before the bodies and programs are
        // held anything; with none, one working set is, and beside it every
        // weight the pool would hold wherever the card holds them with the
        // programs (the activation's span already out of `live`): the
        // bodies are held out of what is left after the weights, so their
        // allowance never turns weights that fit into weights that stream.
        // A card short of that streams them (the runtime's call).
        let whole = need.saturating_add(self.weight_pool_bytes());
        let claim = self
            .budget
            .unwrap_or(if live >= whole.saturating_add(programs) {
                whole
            } else {
                need
            });
        let (bodies, programs) = holds_within(bodies, programs, live.saturating_sub(claim));
        let held = programs.saturating_add(bodies);
        let room = live.saturating_sub(held);
        if let Some(budget) = self.budget.filter(|&budget| budget > room) {
            return Err(Fault::Residency(
                engine::fit::Short::Budget { budget, room }.to_string(),
            ));
        }
        // Whole map units of the room, in whole chunks, of the range's.
        let unit = self.chunks.map_unit().max(1);
        let limit = self.budget.unwrap_or(room) / unit * unit;
        let fitted = match engine::chunk::fit(limit, chunk, 0, least) {
            Ok(chunks) => chunks.min(self.chunks.total()),
            Err(_) if self.budget.is_some() => {
                return Err(Fault::Residency(
                    engine::fit::Short::Sequence {
                        budget: self.budget.unwrap_or(0),
                        least: need,
                    }
                    .to_string(),
                ));
            }
            Err(_) => {
                return Err(Fault::Residency(format!(
                    "the card does not hold this deployment: {resident}; under `[model] \
                     device_memory_share` that leaves {room} bytes for the state pool, and one \
                     working set at the declared context needs {need}. Lower `[engine] \
                     max_model_len`, raise `[model] device_memory_share`, serve a quantized \
                     deployment of the model, or free what else holds the device."
                )));
            }
        };
        self.fitted = fitted;
        // The kv pages the shell's own seats may name: what the chunks hold
        // were every one the kv rows', but those a held load's weights take.
        // The fit claimed every weight the pool holds (the card holds them
        // with one working set and the programs), so the runtime holds them
        // for the load's life and no kv row ever takes their chunks; counted
        // as the rows', they are pages admission hands out that no chunk
        // backs, and a load filling them starves with nothing to discard.
        let weight_chunks = if self.budget.is_none() && claim == whole {
            u32::try_from(self.weight_pool_bytes() / chunk).unwrap_or(u32::MAX)
        } else {
            0
        };
        let kv_chunks = fitted.saturating_sub(weight_chunks).max(least);
        let layout = self.chunks.layout();
        let (mut fits, mut over) = (one_slot, self.ceiling.saturating_add(1));
        while over - fits > 1 {
            let mid = fits + (over - fits) / 2;
            if chunks_at(layout, &self.paging, mid, 0) <= kv_chunks {
                fits = mid;
            } else {
                over = mid;
            }
        }
        self.paging.pages = fits;
        // A flat range is one allocation made on its first commit: of the
        // fitted chunks, not of every chunk the asked count reserved.
        // A range mapped on demand reserved anew before the activation is
        // settled keeps room for what the settled fit may yet give the cache
        // rows: the activation's span no fire backs, and the bodies'
        // allowance past what arming holds. Reserved to this fit alone, the
        // settled one could never grow past it.
        let flat = !elastic::mapped_on_demand();
        let reshape = flat || (self.whole_fit && fitted < self.chunks.total());
        if reshape && self.chunks.fresh() {
            let reserved = if flat || settled {
                fitted
            } else {
                let more = self
                    .chunks
                    .activation_unheld_bytes()
                    .saturating_add(bodies)
                    / chunk;
                fitted
                    .saturating_add(u32::try_from(more).unwrap_or(u32::MAX))
                    .min(self.chunks.total())
            };
            self.chunks.reshape(&mut self.pool, reserved)?;
        }
        self.chunks.limit(fitted);
        // The runtime admits frames against this watermark, so the pool is
        // pledged it beside what its other arenas hold now: a frame inside
        // it is a frame the fit already sized the card for, and what a fire
        // allocates beside the pool later is not its refusal to answer.
        let others = self
            .pool
            .committed_pages()
            .saturating_sub(elastic::pages_for_bytes(self.committed_bytes()));
        self.pool
            .pledge(elastic::pages_for_bytes(self.chunk_bytes(fitted)).saturating_add(others));
        Ok(Fitted {
            asked: self.asked,
            fit: fits,
            room,
            held,
            bodies,
            spare: live.saturating_sub(need).saturating_sub(programs),
            chunks: fitted,
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

    /// Reserves the range anew to the fit even where it maps on demand, so
    /// no chunk past the fit lies under the activation (smaller activation
    /// rungs are stated only then).
    pub fn fit_range_whole(&mut self) {
        self.whole_fit = true;
    }

    /// Backs `bytes` of the activation's span from `offset` into it.
    pub(crate) fn hold_activation_at(
        &mut self,
        offset: u64,
        bytes: u64,
        beside: bool,
    ) -> Result<()> {
        match self
            .chunks
            .hold_activation_at(&mut self.pool, offset, bytes, beside)?
        {
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
    pub fn state_slot_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|shape| match *shape {
                Shape::State { stride, dtype, .. } => stride * u64::from(elem_size(dtype)),
                Shape::Kv { .. } => 0,
            })
            .sum::<u64>()
    }

    #[must_use]
    pub fn has_state(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::State { .. }))
    }

    /// The page tables a fire stages: one a kv shape, the list's and then
    /// each window list's, narrowest first. Every kv row of a shape reads
    /// its shape's.
    #[must_use]
    pub fn geometries(&self) -> usize {
        let layout = self.chunks.layout();
        usize::from(layout.list.is_some()) + layout.windows.len()
    }

    /// Whether page table `at` of a fire's is a window list's.
    #[must_use]
    pub fn windowed_geometry(&self, at: usize) -> bool {
        self.geometry_window(at).is_some()
    }

    /// The window list page table `at` of a fire's is of, if it is one's.
    #[must_use]
    pub fn geometry_window(&self, at: usize) -> Option<usize> {
        let layout = self.chunks.layout();
        at.checked_sub(usize::from(layout.list.is_some()))
            .filter(|&window| window < layout.windows.len())
    }

    /// The tokens window list `window` retains.
    #[must_use]
    pub fn window_tokens(&self, window: usize) -> Option<u32> {
        self.chunks
            .layout()
            .windows
            .get(window)
            .and_then(|shape| shape.window)
    }

    /// Each window a kv row is declared with and the page table of a
    /// fire's that row's list reads: its own window's list's, or the wider
    /// list's its rows are merged into.
    #[must_use]
    pub fn window_tables(&self) -> Vec<(u32, u32)> {
        let mut tables: Vec<(u32, u32)> = Vec::new();
        for &(window, list) in &self.windows_of {
            let table = self.table_of(list) as u32;
            if !tables.contains(&(window, table)) {
                tables.push((window, table));
            }
        }
        tables
    }

    /// The page table of a fire's that the kv rows paged in the window
    /// list (`windowed`) or in the kv list read. The kv list's is the
    /// first, and the All axis's: a model keeping no full-retention row
    /// pages its All axis in its window list, then the only one.
    #[must_use]
    pub fn table_of(&self, list: u8) -> usize {
        match list {
            0 => 0,
            window => usize::from(self.chunks.layout().list.is_some()) + usize::from(window) - 1,
        }
    }

    #[must_use]
    pub fn has_windowed(&self) -> bool {
        self.shapes
            .iter()
            .any(|shape| matches!(shape, Shape::Kv { windowed: true, .. }))
    }

    /// Seats the kv pages `pages` (the windowed ones, a window list's) on slots of
    /// the state pool and backs the segments their chunks lie in.
    fn seat_pages(&mut self, kind: Kind, pages: impl IntoIterator<Item = u64>) -> Result<()> {
        // The physical chunks the pages lie in. A lane's pages run through
        // a few chunks, so a chunk just checked is not checked again.
        let mut highest = None;
        let mut unbacked = None;
        let mut checked = None;
        for page in pages {
            let page = u32::try_from(page).map_err(|_| Fault::Ceiling {
                what: kind.name(),
                need: page,
                have: u64::from(u32::MAX),
            })?;
            let chunk = self.chunks.seat(kind, page)?;
            highest = highest.max(Some(chunk));
            if unbacked.is_none() && checked != Some(chunk) {
                checked = Some(chunk);
                if !self.chunks.backed(chunk) {
                    unbacked = Some(chunk);
                }
            }
        }
        let (Some(highest), Some(unbacked)) = (highest, unbacked) else {
            return Ok(());
        };
        if self.chunks.by_refs() {
            // Nothing is backed on a fire's path: the runtime commits a
            // segment before it hands out a chunk of it.
            return Err(Fault::program(
                "store::chunks",
                format!(
                    "a {} lies in physical chunk {unbacked}, which the runtime has not \
                     committed ({} segments are)",
                    kind.name(),
                    self.chunks.committed_segments()
                ),
            ));
        }
        let segments = self.chunks.segments_for(u64::from(highest) + 1);
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
        // A windowed page is a slot of every window list.
        for window in 0..self.chunks.layout().windows.len() {
            self.seat_pages(Kind::Window(window as u8), fresh.iter().copied())?;
        }
        Ok(())
    }

    pub fn table(&self, seats: &Seats) -> Result<CacheTable> {
        let mut table = Vec::with_capacity(self.shapes.len());
        for shape in &self.shapes {
            table.push(match *shape {
                Shape::Kv {
                    windowed,
                    list,
                    dtype,
                    keys_width,
                    values_width,
                    values_plane,
                    head_stride,
                    live,
                    offsets,
                } => {
                    let table = self.table_of(list);
                    let seat = seats.page_tables.get(table).ok_or_else(|| Fault::Unbound {
                        what: format!("page table {table}, which this fire staged none of"),
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
                        .shape(Kind::paged(list))
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
                        table: table as u32,
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
        // A record moving to or from the host tier lands first.
        self.fence_all_swaps(stream)?;
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
        let mut slots = Vec::new();
        self.record_into(key, &mut slots)?;
        Ok(slots)
    }

    /// [`Pools::record`] into `slots`, cleared first: a fire's lanes reuse
    /// one buffer.
    fn record_into(&self, key: u32, slots: &mut Vec<u32>) -> Result<()> {
        slots.clear();
        let pieces = self.fixed()?.pieces;
        if !self.chunks.by_refs() {
            slots.extend((0..pieces).map(|piece| key * pieces + piece));
            return Ok(());
        }
        if pieces == 1 {
            slots.push(key);
            return Ok(());
        }
        let held = self.records.get(&key).ok_or_else(|| {
            Fault::program(
                "store::state",
                format!("the record at slot ref {key} is read before a lane names its pieces"),
            )
        })?;
        slots.extend_from_slice(held);
        Ok(())
    }

    /// The record `key` names, its slots seated and backed.
    fn ensure_record(&mut self, key: u32) -> Result<Vec<u32>> {
        let slots = self.record(key)?;
        self.seat_pages(Kind::Fixed, slots.iter().map(|&slot| u64::from(slot)))?;
        Ok(slots)
    }

    /// The record's bytes as the host holds them, `(slot of the record,
    /// offset, bytes)` in order: each recurrent row piece after piece.
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
        let lanes = self.id_lanes as usize;
        self.expand(keys, lanes)
    }

    /// A fire's snapshot slots, as [`Pools::state_tables`] lays a fire's
    /// lanes: an entry a record a snapshot lands in.
    pub fn snapshot_tables(&mut self, keys: &[i32]) -> Result<Vec<i32>> {
        if !self.has_state() || keys.is_empty() {
            return Ok(keys.to_vec());
        }
        let entries = self.snap_entries as usize;
        self.expand(keys, entries)
    }

    fn expand(&mut self, keys: &[i32], rows: usize) -> Result<Vec<i32>> {
        if keys.len() > rows {
            return Err(Fault::Ceiling {
                what: "records a fire's id tables name",
                need: keys.len() as u64,
                have: rows as u64,
            });
        }
        // Each record seated and backed first; the ids are read after, the
        // shape borrowed, a buffer of slots reused lane to lane.
        let mut record = Vec::new();
        for &key in keys.iter().filter(|&&key| key >= 0) {
            self.state_row(key as u32)?;
            self.record_into(key as u32, &mut record)?;
            self.seat_pages(Kind::Fixed, record.iter().map(|&slot| u64::from(slot)))?;
        }
        let fixed = self.fixed()?;
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
            self.record_into(key as u32, &mut record)?;
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

    /// The rs list as the chunk recipe lays it: the tokens a page holds, and
    /// each rs row's run in a page, `(offset, bytes a token)`.
    #[must_use]
    pub fn rs_layout(&self) -> Option<(u32, Vec<(u64, u64)>)> {
        self.chunks
            .layout()
            .rs
            .as_ref()
            .map(|rs| (rs.slot.tokens, rs.layers.clone()))
    }

    /// Seats the rs buffer pages `pages` on slots of the state pool and
    /// backs the segments their chunks lie in.
    pub fn seat_rs(&mut self, pages: &[u32]) -> Result<()> {
        self.seat_pages(Kind::Rs, pages.iter().map(|&page| u64::from(page)))
    }

    /// The address of `bytes` at `offset` of rs buffer page `page`'s slot.
    pub fn rs_span(&self, page: u32, offset: u64, bytes: u64) -> Result<u64> {
        self.chunks.span(Kind::Rs, page, offset, bytes)
    }

    /// Copies whole slots of the shape `shape` names, `(src, dst)`, on
    /// `stream`: a shared tail slot's rows onto the slot a fire then writes.
    pub fn copy_whole_slots(
        &mut self,
        stream: *mut core::ffi::c_void,
        shape: engine::chunk::ShapeId,
        pairs: &[[u32; 2]],
    ) -> Result<()> {
        let kind = Self::kind_of(shape)?;
        let bytes = self
            .chunks
            .layout()
            .shape(kind)
            .map(|shape| shape.slot_bytes)
            .ok_or_else(|| {
                Fault::program(
                    "store::chunks",
                    format!("a slot copy of a {} in a load that lays none", kind.name()),
                )
            })?;
        self.seat_pages(kind, pairs.iter().flatten().map(|&slot| u64::from(slot)))?;
        self.fence_all_swaps(stream)?;
        for &[src, dst] in pairs {
            if src == dst {
                continue;
            }
            crate::device::copy_d2d(
                stream,
                self.chunks.span(kind, dst, 0, bytes)?,
                self.chunks.span(kind, src, 0, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
        }
        Ok(())
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
                list,
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
                            .span(Kind::paged(list), span.dst_page, dst, bytes)?,
                        self.chunks
                            .span(Kind::paged(list), span.src_page, src, bytes)?,
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

    /// The compressor slabs keyed by a windowed row's pages (or not), and
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
                            kind: Kind::paged(row.list),
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
                list,
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
                        kind: Kind::paged(list),
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

    /// Rewrites the page ids of a fire's table (list `list`'s: 0 the kv
    /// list, `n` window list `n - 1`)
    /// as the ids the paged kernels read their slots by; the fire's commit
    /// has seated them.
    pub fn page_ids(&self, list: u8, ids: &mut [i32]) -> Result<()> {
        self.chunks.page_ids(Kind::paged(list), ids)
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
        // No move still landing reads or writes a chunk that goes.
        self.settle_swaps()?;
        self.chunks.release_segments(&mut self.pool, segments)
    }

    /// The chunks the fit left the cache rows, of the range's.
    fn fitted_chunks(&self) -> u32 {
        self.fitted.min(self.chunks.total())
    }

    /// See `Engine::evacuate`.
    pub fn evacuate(
        &mut self,
        stream: *mut core::ffi::c_void,
        evacuation: &engine::chunk::Evacuation,
    ) -> Result<()> {
        self.fence_all_swaps(stream)?;
        self.chunks.evacuate(stream, evacuation)?;
        // A cached atom's chunk may have moved: its table entry with it.
        self.stage_row_tables(stream)?;
        self.stage_expert_tables(stream)
    }

    /// The params of the routed banks streamed an expert at a time.
    #[must_use]
    pub fn expert_params(&self) -> Vec<u32> {
        self.chunks
            .layout()
            .experts
            .iter()
            .flat_map(|shape| shape.banks.iter().flatten().map(|plane| plane.param))
            .collect()
    }

    /// Opens the cache of each expert shape: `hosts` gives each bank's
    /// mapped host copy by its param. No expert is cached yet; each is read
    /// from the host copy.
    ///
    /// `groups` names each packed bank by its codes' param with the params
    /// of its planes in the order its select's table lists them (the codes
    /// first; `None` a plane the bank lacks): one table for the bank, an
    /// expert's entry a plane. A plane in no group is a dense bank, a table
    /// of its own.
    pub fn seat_experts(
        &mut self,
        hosts: &[(u32, u64)],
        groups: &[(u32, Vec<Option<u32>>)],
    ) -> Result<()> {
        self.expert_caches.clear();
        for shape in &self.chunks.layout().experts {
            let experts = shape.experts as usize;
            let mut planes: Vec<ExpertBank> = Vec::new();
            for (layer, bank) in shape.banks.iter().enumerate() {
                let first = planes.len();
                // A companion plane is read through its bank's table alone.
                let companion = |param: u32| {
                    groups.iter().any(|(codes, members)| {
                        *codes != param && members.iter().any(|member| *member == Some(param))
                    })
                };
                let mut counted = false;
                for plane in bank {
                    let Some(&(_, host)) = hosts.iter().find(|(param, _)| *param == plane.param)
                    else {
                        return Err(Fault::program(
                            "store::experts",
                            format!(
                                "param {} streams by its experts and has no host copy",
                                plane.param
                            ),
                        ));
                    };
                    let head = !companion(plane.param);
                    // The planes its table lists, as places among the
                    // layer's (resolved below, once all are known).
                    let members = match groups.iter().find(|(codes, _)| *codes == plane.param) {
                        Some((_, members)) => members.clone(),
                        None => vec![Some(plane.param)],
                    };
                    planes.push(ExpertBank {
                        plane: *plane,
                        layer: layer as u32,
                        host,
                        head,
                        members: members
                            .iter()
                            .map(|member| member.map(|param| param as usize))
                            .collect(),
                        table: crate::device::Buffer::zeroed(experts * members.len().max(1) * 8)?,
                        // A layer's reads are counted at its first bank.
                        hits: if head && !std::mem::replace(&mut counted, true) {
                            Some(crate::device::Buffer::zeroed(experts * 4)?)
                        } else {
                            None
                        },
                    });
                }
                // Each member's param to its place among the cache's planes.
                let places: Vec<(usize, usize)> = planes[first..]
                    .iter()
                    .enumerate()
                    .map(|(at, held)| (held.plane.param as usize, first + at))
                    .collect();
                for held in &mut planes[first..] {
                    for member in &mut held.members {
                        if let Some(param) = *member {
                            let place = places
                                .iter()
                                .find(|(held, _)| *held == param)
                                .map(|&(_, place)| place)
                                .ok_or_else(|| {
                                    Fault::program(
                                        "store::experts",
                                        format!(
                                            "param {param} is a plane of a bank whose experts \
                                             the recipe does not lay it with"
                                        ),
                                    )
                                })?;
                            *member = Some(place);
                        }
                    }
                }
            }
            self.expert_caches.push(ExpertCache {
                shape: shape.clone(),
                planes,
                slots: vec![None; shape.count() as usize],
            });
        }
        self.stage_expert_tables(core::ptr::null_mut())
    }

    /// Each bank's table as its experts now lie, written behind what
    /// `stream` holds (a null stream: at once, before any fire).
    fn stage_expert_tables(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        for at in 0..self.expert_caches.len() {
            for bank in 0..self.expert_caches[at].planes.len() {
                let cache = &self.expert_caches[at];
                let held = &cache.planes[bank];
                if !held.head {
                    continue;
                }
                let first = (held.layer * cache.shape.experts) as usize;
                // Arming's stand-in: the shape's first slot the pool holds.
                let stand_in = self
                    .stand_in
                    .then(|| cache.slots.iter().flatten().next().copied())
                    .flatten();
                let mut entries =
                    Vec::with_capacity(cache.shape.experts as usize * held.members.len() * 8);
                for expert in 0..cache.shape.experts as usize {
                    let slot = match cache.slots[first + expert].or(stand_in) {
                        Some(slot) => Some(self.chunks.weight_slot(
                            cache.shape.per_chunk,
                            cache.shape.slot_bytes,
                            slot,
                        )?),
                        None => None,
                    };
                    // An expert's entry a plane of the bank, in the order
                    // its select lists them; 0 a plane the bank lacks.
                    for member in &held.members {
                        let address = match member.map(|place| &cache.planes[place]) {
                            None => 0,
                            Some(plane) => match slot {
                                Some(slot) => slot + plane.plane.offset,
                                None => plane.host + expert as u64 * plane.plane.bytes,
                            },
                        };
                        entries.extend_from_slice(&address.to_le_bytes());
                    }
                }
                let held = &mut self.expert_caches[at].planes[bank];
                if stream.is_null() {
                    held.table.write(0, &entries)?;
                } else {
                    held.table.stage(stream, 0, &entries)?;
                }
            }
        }
        Ok(())
    }

    /// The expert table of each streamed bank, by its param: what its
    /// select reads its experts through. A bank whose experts all lie in
    /// pool slots the runtime holds for the load's life is located, any
    /// select's; any other may name the host copy by the next fire, and is
    /// the plain-load selects' (a recorded body keeps the select it chose).
    #[must_use]
    pub fn expert_tables(&self) -> Vec<(u32, kernels_cuda::linear::moe::ExpertTable)> {
        let window = (self.chunks.base(), self.chunks.bytes());
        self.expert_caches
            .iter()
            .flat_map(|cache| {
                cache
                    .planes
                    .iter()
                    .filter(|bank| bank.head)
                    .map(move |bank| {
                        let first = (bank.layer * cache.shape.experts) as usize;
                        let located = (self.weights_held || self.preseated)
                            && cache.slots[first..first + cache.shape.experts as usize]
                                .iter()
                                .all(Option::is_some);
                        (
                            bank.plane.param,
                            kernels_cuda::linear::moe::ExpertTable {
                                table: bank.table.ptr(),
                                hits: bank.hits.as_ref().map_or(0, crate::device::Buffer::ptr),
                                located,
                                window: if located { window } else { (0, 0) },
                            },
                        )
                    })
            })
            .collect()
    }

    /// Opens the cache of each row-streamed table: `hosts` gives each
    /// table's mapped host copy by its param. Every block starts where that
    /// copy has it, read in place until the runtime names a slot for it.
    pub fn seat_rows(&mut self, hosts: &[(u32, u64)]) -> Result<()> {
        self.row_caches.clear();
        for shape in &self.chunks.layout().rows {
            let Some(&(_, host)) = hosts.iter().find(|(param, _)| *param == shape.param) else {
                return Err(Fault::program(
                    "store::rows",
                    format!(
                        "param {} streams by its rows and has no host copy",
                        shape.param
                    ),
                ));
            };
            let blocks = shape.blocks as usize;
            let cache = RowCache {
                shape: shape.clone(),
                host,
                table: crate::device::Buffer::zeroed(blocks * 8)?,
                hits: crate::device::Buffer::zeroed(blocks * 4)?,
                slots: vec![None; blocks],
            };
            self.row_caches.push(cache);
        }
        self.stage_row_tables(core::ptr::null_mut())
    }

    /// Each cache's table as its blocks now lie, written behind what
    /// `stream` holds (a null stream: at once, before any fire).
    fn stage_row_tables(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        for at in 0..self.row_caches.len() {
            let cache = &self.row_caches[at];
            let block = u64::from(cache.shape.block_rows) * cache.shape.row_bytes;
            let mut entries = Vec::with_capacity(cache.slots.len() * 8);
            let stand_in = self
                .stand_in
                .then(|| cache.slots.iter().flatten().next().copied())
                .flatten();
            for (atom, slot) in cache.slots.iter().enumerate() {
                let address = match slot.as_ref().or(stand_in.as_ref()) {
                    Some(slot) => self.chunks.row_slot(&cache.shape, *slot)?,
                    None => cache.host + atom as u64 * block,
                };
                entries.extend_from_slice(&address.to_le_bytes());
            }
            let cache = &mut self.row_caches[at];
            if stream.is_null() {
                cache.table.write(0, &entries)?;
            } else {
                cache.table.stage(stream, 0, &entries)?;
            }
        }
        Ok(())
    }

    /// The params of the tables streamed by their rows.
    #[must_use]
    pub fn row_params(&self) -> Vec<u32> {
        self.chunks
            .layout()
            .rows
            .iter()
            .map(|rows| rows.param)
            .collect()
    }

    /// The row table of each streamed table, by its param: what its gather
    /// reads its rows through, counting each block's reads.
    #[must_use]
    pub fn row_tables(&self) -> Vec<(u32, kernels_cuda::linear::moe::RowTable)> {
        self.row_caches
            .iter()
            .map(|cache| {
                (
                    cache.shape.param,
                    kernels_cuda::linear::moe::RowTable::located(
                        cache.table.ptr(),
                        cache.shape.block_rows,
                    )
                    .counting(cache.hits.ptr()),
                )
            })
            .collect()
    }

    /// See `Engine::weight_residency`: the blocks newly named a slot are
    /// filled from the host copy and every table rewritten, on `stream`
    /// behind the fires it holds and before the next. A block no longer
    /// named is read from the host copy again. Returns whether a bank's
    /// table became located or stopped being so: the selects a recorded
    /// body chose for it are then not the ones a fire would choose.
    pub fn weight_residency(
        &mut self,
        stream: *mut core::ffi::c_void,
        residency: &engine::chunk::WeightResidency,
    ) -> Result<bool> {
        if self.weights_held && !residency.held {
            return Err(Fault::Unbound {
                what: "a weight residency that is not held after one that was: a held set \
                       is the load's for life"
                    .to_string(),
            });
        }
        let located = |pools: &Pools| -> Vec<bool> {
            pools
                .expert_tables()
                .iter()
                .map(|(_, table)| table.located)
                .collect()
        };
        if self.weights_filed {
            let named = |shape: engine::chunk::ShapeId| {
                residency
                    .resident
                    .iter()
                    .filter(|resident| resident.atom.shape == shape)
                    .count()
            };
            let whole = self
                .row_caches
                .iter()
                .map(|cache| (cache.shape.id, cache.slots.len()))
                .chain(
                    self.expert_caches
                        .iter()
                        .map(|cache| (cache.shape.id, cache.slots.len())),
                )
                .all(|(shape, atoms)| named(shape) >= atoms);
            if !residency.held || !whole {
                // The load was told its weights would be held, and the
                // runtime streams them after all (its pool does not seat
                // every atom): the copy a miss is read from is made now.
                self.pin_weight_hosts()?;
            }
        }
        let before = located(self);
        // An atom the load seated that the runtime names another slot for
        // is filled there again: said, since a runtime that adopted the
        // load's seating moves none.
        if self.preseated {
            let moved = residency
                .resident
                .iter()
                .filter(|resident| {
                    self.row_caches
                        .iter()
                        .map(|cache| (cache.shape.id, &cache.slots))
                        .chain(
                            self.expert_caches
                                .iter()
                                .map(|cache| (cache.shape.id, &cache.slots)),
                        )
                        .find(|(id, _)| *id == resident.atom.shape)
                        .and_then(|(_, slots)| slots.get(resident.atom.atom as usize))
                        .is_some_and(|held| *held != Some(resident.slot))
                })
                .count();
            if moved > 0 {
                eprintln!(
                    "engine-cuda: the first weight residency names {moved} of {} atoms a slot \
                     other than the one the load seated them in (`WeightShape::seated`); each \
                     is read from the serving file again",
                    residency.resident.len()
                );
            }
        }
        // The runtime's set stands from here on, the load's own no longer.
        self.preseated = false;
        self.weights_held = residency.held;
        for at in 0..self.row_caches.len() {
            let shape = self.row_caches[at].shape.clone();
            let mut named: Vec<Option<u32>> = vec![None; shape.blocks as usize];
            for resident in residency
                .resident
                .iter()
                .filter(|resident| resident.atom.shape == shape.id)
            {
                let Some(seat) = named.get_mut(resident.atom.atom as usize) else {
                    return Err(Fault::Ceiling {
                        what: "row blocks of a streamed table",
                        need: u64::from(resident.atom.atom) + 1,
                        have: u64::from(shape.blocks),
                    });
                };
                *seat = Some(resident.slot);
            }
            let block = u64::from(shape.block_rows) * shape.row_bytes;
            let table_bytes = shape.rows * shape.row_bytes;
            let host = self.row_caches[at].host;
            let mut fills = Vec::new();
            for (atom, slot) in named.iter().enumerate() {
                let Some(slot) = *slot else { continue };
                if self.row_caches[at].slots[atom] == Some(slot) {
                    continue;
                }
                let from = atom as u64 * block;
                let bytes = block.min(table_bytes.saturating_sub(from));
                fills.push((
                    self.chunks.row_slot(&shape, slot)?,
                    host + from,
                    usize::try_from(bytes).unwrap_or(0),
                ));
            }
            self.fill_weights(stream, &fills)?;
            self.row_caches[at].slots = named;
        }
        for at in 0..self.expert_caches.len() {
            let shape = self.expert_caches[at].shape.clone();
            let mut named: Vec<Option<u32>> = vec![None; shape.count() as usize];
            for resident in residency
                .resident
                .iter()
                .filter(|resident| resident.atom.shape == shape.id)
            {
                let Some(seat) = named.get_mut(resident.atom.atom as usize) else {
                    return Err(Fault::Ceiling {
                        what: "experts of a streamed shape",
                        need: u64::from(resident.atom.atom) + 1,
                        have: u64::from(shape.count()),
                    });
                };
                *seat = Some(resident.slot);
            }
            let mut fills = Vec::new();
            for (atom, slot) in named.iter().enumerate() {
                let Some(slot) = *slot else { continue };
                if self.expert_caches[at].slots[atom] == Some(slot) {
                    continue;
                }
                let (layer, expert) = (
                    atom as u32 / shape.experts,
                    u64::from(atom as u32 % shape.experts),
                );
                let to = self
                    .chunks
                    .weight_slot(shape.per_chunk, shape.slot_bytes, slot)?;
                for bank in self.expert_caches[at]
                    .planes
                    .iter()
                    .filter(|bank| bank.layer == layer)
                {
                    fills.push((
                        to + bank.plane.offset,
                        bank.host + expert * bank.plane.bytes,
                        usize::try_from(bank.plane.bytes).unwrap_or(0),
                    ));
                }
            }
            self.fill_weights(stream, &fills)?;
            self.expert_caches[at].slots = named;
        }
        self.stage_row_tables(stream)?;
        self.stage_expert_tables(stream)?;
        Ok(before != located(self))
    }

    /// The chunks `residency` names weight slots in.
    #[must_use]
    pub fn weight_chunks(
        &self,
        residency: &engine::chunk::WeightResidency,
    ) -> std::collections::BTreeSet<u32> {
        let per_chunk = |shape: engine::chunk::ShapeId| {
            self.row_caches
                .iter()
                .map(|cache| (cache.shape.id, cache.shape.per_chunk))
                .chain(
                    self.expert_caches
                        .iter()
                        .map(|cache| (cache.shape.id, cache.shape.per_chunk)),
                )
                .find(|(id, _)| *id == shape)
                .map(|(_, per_chunk)| per_chunk.max(1))
        };
        residency
            .resident
            .iter()
            .filter_map(|resident| Some(resident.slot / per_chunk(resident.atom.shape)?))
            .collect()
    }

    /// Whether a page this pool seated for a lane lies in one of `chunks`.
    #[must_use]
    pub fn seated_in(&self, chunks: &std::collections::BTreeSet<u32>) -> bool {
        self.chunks.seated_in(chunks)
    }

    /// See [`chunks::Chunks::keep_clear`].
    pub fn keep_clear(&mut self, chunks: std::collections::BTreeSet<u32>) {
        self.chunks.keep_clear(chunks);
    }

    /// Fills weight slots from the host copies, `(slot, host, bytes)` each.
    fn fill_weights(
        &mut self,
        stream: *mut core::ffi::c_void,
        fills: &[(u64, u64, usize)],
    ) -> Result<()> {
        if self.hosts_dropped && !fills.is_empty() {
            return Err(Fault::Unbound {
                what: "a weight atom named a new slot after the held set's host copy was \
                       dropped: a held atom stays where it was named"
                    .to_string(),
            });
        }
        if self.weights_filed {
            // Pageable memory: a copy a span, staged by the runtime.
            for &(slot, file, bytes) in fills {
                crate::device::copy_any(stream, slot, file, bytes)?;
                self.file_spans.push((file, bytes));
            }
            return Ok(());
        }
        crate::device::copy_batch(stream, false, fills)
    }

    /// The caches' host copies are the serving file's mapping: a slot is
    /// filled from it, and no atom is read where it lies. A residency that
    /// does not hold every atom makes the copy a miss is read from first
    /// ([`Pools::pin_weight_hosts`]).
    pub fn file_weights(&mut self) {
        self.weights_filed = true;
    }

    /// Whether a span of the serving file's mapping was copied out since
    /// its pages were last let go.
    #[must_use]
    pub fn copied_from_file(&self) -> bool {
        !self.file_spans.is_empty()
    }

    /// Lets go of the pages of the serving file's mapping that were copied
    /// out of it (into pool slots, or the pinned copy): they are read again
    /// only to fill a slot, from the page cache. The copies have landed
    /// (the caller synchronized the stream that carried them).
    pub fn release_file_pages(&mut self) {
        const PAGE: u64 = 4096;
        for (at, bytes) in std::mem::take(&mut self.file_spans) {
            let first = at.next_multiple_of(PAGE);
            let end = (at + bytes as u64) / PAGE * PAGE;
            if end > first {
                // SAFETY: whole pages inside a read-only file mapping the
                // weight store holds for the load's life; the advice drops
                // their resident copies, which a later read faults back in.
                unsafe {
                    libc::madvise(
                        first as *mut core::ffi::c_void,
                        (end - first) as usize,
                        libc::MADV_DONTNEED,
                    );
                }
            }
        }
    }

    /// The bytes every weight the caches state takes in the pool: each
    /// shape's atoms in whole chunks of its own, as
    /// [`Pools::preseat_plan`] seats them.
    #[must_use]
    pub fn weight_pool_bytes(&self) -> u64 {
        let chunks: u64 = self
            .row_caches
            .iter()
            .map(|cache| (cache.shape.per_chunk, cache.slots.len()))
            .chain(
                self.expert_caches
                    .iter()
                    .map(|cache| (cache.shape.per_chunk, cache.slots.len())),
            )
            .map(|(per_chunk, atoms)| (atoms as u64).div_ceil(u64::from(per_chunk.max(1))))
            .sum();
        chunks.saturating_mul(self.chunks.layout().chunk_bytes)
    }

    /// The residency a load seats its weights by before the runtime names
    /// any slot: every atom of every shape, each shape's in chunks of its
    /// own from the pool's first on. `None` where the pool cannot hold them
    /// all with a chunk to spare: the load then keeps a copy its fires read
    /// in place ([`Pools::pin_weight_hosts`]), and the runtime decides what
    /// a pool that small may serve.
    #[must_use]
    pub fn preseat_plan(&self) -> Option<engine::chunk::WeightResidency> {
        let shapes = self
            .row_caches
            .iter()
            .map(|cache| (cache.shape.id, cache.shape.per_chunk, cache.slots.len()))
            .chain(
                self.expert_caches
                    .iter()
                    .map(|cache| (cache.shape.id, cache.shape.per_chunk, cache.slots.len())),
            );
        let mut first = 1u32;
        let mut resident = Vec::new();
        for (shape, per_chunk, atoms) in shapes {
            let per_chunk = per_chunk.max(1);
            resident.extend((0..atoms as u32).map(|atom| engine::chunk::ResidentWeight {
                atom: engine::chunk::WeightAtom { shape, atom },
                slot: first * per_chunk + atom,
            }));
            first = first.checked_add((atoms as u32).div_ceil(per_chunk))?;
        }
        // No more chunks than the fit left the pool (a stated
        // `device_kv_budget` is all of it): the range is limited to them,
        // under whatever was seated past it.
        let most = self.fitted_chunks();
        (first < most).then_some(engine::chunk::WeightResidency {
            staging: Vec::new(),
            resident,
            held: true,
        })
    }

    /// Seats the load's weights by `plan` ([`Pools::preseat_plan`]): the
    /// slots' segments backed, each filled from the serving file, the
    /// tables located. The load arms over them; the runtime's first
    /// residency then names the slots it chose, and an atom it puts
    /// elsewhere is filled there again.
    pub fn preseat_weights(
        &mut self,
        stream: *mut core::ffi::c_void,
        plan: &engine::chunk::WeightResidency,
    ) -> Result<()> {
        let segment = self.chunks.layout().segment_chunks.max(1);
        let last = self.weight_chunks(plan).into_iter().max().unwrap_or(0);
        self.commit_segments((last + 1).div_ceil(segment))?;
        self.weight_residency(stream, plan)?;
        self.weights_held = false;
        self.preseated = true;
        Ok(())
    }

    /// Gives the caches a copy a fire can read a miss from, out of the
    /// serving file's mapping they were left in: the mapping itself where
    /// the device faults mapped pages in, else a pinned copy of every
    /// streamed table and bank, refused where the machine has not the
    /// memory for it.
    pub fn pin_weight_hosts(&mut self) -> Result<()> {
        self.weights_filed = false;
        if crate::experts::pageable_access() {
            return Ok(());
        }
        let align = |bytes: u64| bytes.next_multiple_of(crate::weights::ALIGN);
        // Each copy's place in the pinned span, its source and its bytes.
        let mut spans: Vec<(u64, u64, u64)> = Vec::new();
        let mut total = 0u64;
        let mut lay = |file: u64, bytes: u64, total: &mut u64| {
            spans.push((*total, file, bytes));
            *total += align(bytes);
        };
        for cache in &self.row_caches {
            lay(
                cache.host,
                cache.shape.rows * cache.shape.row_bytes,
                &mut total,
            );
        }
        for cache in &self.expert_caches {
            for bank in &cache.planes {
                lay(
                    bank.host,
                    u64::from(cache.shape.experts) * bank.plane.bytes,
                    &mut total,
                );
            }
        }
        let have = crate::weights::available_memory();
        if total > have {
            return Err(Fault::Residency(format!(
                "the runtime streams this load's weights, which needs a pinned host copy of \
                 them, {:.1} GiB, and this machine has {:.1} GiB of memory left for it. Give \
                 the pool room to hold them all (`weight_streaming = \"off\"` with a larger \
                 device share), or use a device with pageable memory access (HMM).",
                total as f64 / f64::from(1u32 << 30),
                have as f64 / f64::from(1u32 << 30),
            )));
        }
        let pinned =
            crate::device::Pinned::mapped_uninit(usize::try_from(total).unwrap_or(usize::MAX))?;
        for &(at, file, bytes) in &spans {
            let len = usize::try_from(bytes).unwrap_or(usize::MAX);
            // SAFETY: `file` and `bytes` are a plane of the serving file's
            // mapping the weight store bound and holds for the load's life.
            let plane = unsafe { core::slice::from_raw_parts(file as *const u8, len) };
            if !pinned.write(usize::try_from(at).unwrap_or(usize::MAX), plane) {
                return Err(Fault::program(
                    "store::weights",
                    "a streamed weight does not fit the pinned copy sized for it",
                ));
            }
            self.file_spans.push((file, len));
        }
        let base = pinned.device();
        let mut copies = spans.iter().map(|&(at, _, _)| base + at);
        for cache in &mut self.row_caches {
            cache.host = copies.next().unwrap_or(cache.host);
        }
        for cache in &mut self.expert_caches {
            for bank in &mut cache.planes {
                bank.host = copies.next().unwrap_or(bank.host);
            }
        }
        eprintln!(
            "engine-cuda: the runtime streams weights this load was told it would hold; {} \
             MiB of them pinned on the host from the serving file, for misses",
            total >> 20
        );
        self.miss_copy = Some(pinned);
        // The tables name the copy from here on: at once, where no fire has
        // read them yet (a load that seats none of its weights arms over
        // the copy); a residency restages them behind its stream after.
        self.stage_row_tables(core::ptr::null_mut())?;
        self.stage_expert_tables(core::ptr::null_mut())?;
        Ok(())
    }

    /// The params whose host copies no fire reads and no fill needs again:
    /// every streamed table and bank, where the runtime holds the set for
    /// the load's life and it names every atom. Empty otherwise.
    #[must_use]
    pub fn held_weights(&self) -> Vec<usize> {
        let whole = self.weights_held
            && self
                .row_caches
                .iter()
                .map(|cache| &cache.slots)
                .chain(self.expert_caches.iter().map(|cache| &cache.slots))
                .all(|slots| slots.iter().all(Option::is_some));
        if !whole {
            return Vec::new();
        }
        self.row_params()
            .into_iter()
            .chain(self.expert_params())
            .map(|param| param as usize)
            .collect()
    }

    /// The host copies are gone: nothing is filled from them again.
    pub fn drop_weight_hosts(&mut self) {
        self.hosts_dropped = true;
    }

    /// Arming over a streamed set reads every atom on the host copy through
    /// a slot of its shape the pool holds (`on`), and the tables are staged
    /// so at once; off, as they lie. A synthetic fire's rows are never read
    /// back, so what the slot holds does not matter, and nothing it records
    /// names an entry: a body reads the table as it stands when it replays.
    /// The tables stay unlocated either way, so a body takes the select
    /// that reads the host copy.
    pub fn stand_in(&mut self, on: bool) -> Result<()> {
        self.stand_in = on;
        self.stage_row_tables(core::ptr::null_mut())?;
        self.stage_expert_tables(core::ptr::null_mut())
    }

    /// Whether every atom of every weight shape lies in a pool slot (true
    /// where the load states none).
    #[must_use]
    pub fn weights_seated_whole(&self) -> bool {
        self.row_caches
            .iter()
            .all(|cache| cache.slots.iter().all(Option::is_some))
            && self
                .expert_caches
                .iter()
                .all(|cache| cache.slots.iter().all(Option::is_some))
    }

    /// Whether every streamed bank's table is located: each expert in a
    /// slot of a set that is held.
    #[must_use]
    pub fn weights_located(&self) -> bool {
        self.expert_tables().iter().all(|(_, table)| table.located)
    }

    /// The row-streamed tables' blocks named a slot, and all of them.
    #[must_use]
    pub fn row_blocks(&self) -> (u64, u64) {
        self.row_caches.iter().fold((0, 0), |(named, all), cache| {
            (
                named + cache.slots.iter().flatten().count() as u64,
                all + cache.slots.len() as u64,
            )
        })
    }

    /// Zeroes the use record on `stream`, behind the fires that counted:
    /// what the shell's own synthetic fires read (arming the lattice) is
    /// no use the runtime orders its weight cache by.
    pub fn forget_weight_uses(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        for cache in &mut self.row_caches {
            let blocks = cache.slots.len();
            cache.hits.zero_span_on(stream, 0, blocks * 4)?;
        }
        for cache in &mut self.expert_caches {
            let experts = cache.shape.experts as usize;
            for hits in cache.planes.iter_mut().filter_map(|bank| bank.hits.as_mut()) {
                hits.zero_span_on(stream, 0, experts * 4)?;
            }
        }
        Ok(())
    }

    /// See `Engine::weight_uses`: each block's reads since the last call,
    /// read on `stream` behind the fires that counted them, and zeroed.
    pub fn weight_uses(
        &mut self,
        stream: *mut core::ffi::c_void,
    ) -> Result<engine::chunk::WeightUses> {
        let mut uses = Vec::new();
        for cache in &mut self.row_caches {
            let blocks = cache.slots.len();
            let mut counted = vec![0u8; blocks * 4];
            cache.hits.read_on(stream, 0, &mut counted)?;
            cache.hits.zero_span_on(stream, 0, blocks * 4)?;
            for (atom, reads) in counted.as_chunks::<4>().0.iter().enumerate() {
                let reads = u32::from_le_bytes(*reads);
                if reads > 0 {
                    uses.push((
                        engine::chunk::WeightAtom {
                            shape: cache.shape.id,
                            atom: atom as u32,
                        },
                        reads,
                    ));
                }
            }
        }
        for cache in &mut self.expert_caches {
            let experts = cache.shape.experts as usize;
            for bank in &mut cache.planes {
                let Some(hits) = bank.hits.as_mut() else {
                    continue;
                };
                let mut counted = vec![0u8; experts * 4];
                hits.read_on(stream, 0, &mut counted)?;
                hits.zero_span_on(stream, 0, experts * 4)?;
                for (expert, reads) in counted.as_chunks::<4>().0.iter().enumerate() {
                    let reads = u32::from_le_bytes(*reads);
                    if reads > 0 {
                        uses.push((
                            engine::chunk::WeightAtom {
                                shape: cache.shape.id,
                                atom: bank.layer * cache.shape.experts + expert as u32,
                            },
                            reads,
                        ));
                    }
                }
            }
        }
        let held = self
            .row_caches
            .iter()
            .map(|cache| (cache.shape.id, &cache.slots))
            .chain(
                self.expert_caches
                    .iter()
                    .map(|cache| (cache.shape.id, &cache.slots)),
            )
            .map(|(shape, slots)| {
                let resident = slots.iter().flatten().count() as u32;
                engine::chunk::WeightHeld {
                    shape,
                    resident,
                    host: slots.len() as u32 - resident,
                }
            })
            .collect();
        Ok(engine::chunk::WeightUses { uses, held })
    }

    /// The state pool as the runtime counts it, for fires of up to
    /// `max_tokens`: the chunks the fit left the cache rows, the rest of the
    /// range stated with the activation's.
    ///
    /// `rungs`, each smaller rung's tokens and the bytes at the bottom of
    /// the activation's span state may take at it, are stated where no
    /// spare chunk lies under the activation (a smaller rung hands state
    /// the span's bottom, which a spare chunk would put past the fit).
    pub fn chunk_facts(
        &mut self,
        max_tokens: u32,
        rungs: &[(u32, u64)],
    ) -> Result<engine::chunk::ChunkFacts> {
        let fitted = self.fitted_chunks();
        self.chunks.limit(fitted);
        let spare = self.chunks.total().saturating_sub(fitted);
        // The state's chunks of the segment the stated activation starts in
        // are backed from here on, as the runtime counts them.
        match self.chunks.hold_boundary(&mut self.pool, spare)? {
            Commit::Committed => {}
            refusal => return Err(refuse(&self.pool, refusal)),
        }
        let chunk = self.chunks.layout().chunk_bytes.max(1);
        let act = self.chunks.act_chunks();
        // A smaller rung hands state the chunks under the activation, the
        // spare ones first: stated only where none lies there, or so few
        // (what the fit took back after the bodies were armed) that they are
        // backed beside the budget.
        let usable = spare == 0
            || (spare <= SPARE_UNDER_RUNGS && !rungs.is_empty() && {
                match self.chunks.hold_spare(&mut self.pool, spare)? {
                    Commit::Committed => true,
                    _ => false,
                }
            });
        let rungs: Vec<(u32, u32)> = if usable {
            rungs
                .iter()
                .map(|&(tokens, under)| {
                    let under = u32::try_from(under / chunk).unwrap_or(u32::MAX);
                    // `facts` states each rung with the spare chunks; at a
                    // smaller rung they are state's too.
                    (
                        tokens,
                        act.saturating_sub(under).saturating_sub(spare).max(1),
                    )
                })
                .collect()
        } else {
            if !rungs.is_empty() {
                eprintln!(
                    "engine-cuda: no smaller activation rung is stated: {spare} chunk(s) under \
                     the activation lie past the fit"
                );
            }
            Vec::new()
        };
        let mut facts = self
            .chunks
            .facts(max_tokens, spare, !elastic::mapped_on_demand(), &rungs);
        // Where the load seated the weights itself, for a runtime to adopt.
        if self.preseated {
            for shape in &mut facts.weights {
                shape.seated = self
                    .row_caches
                    .iter()
                    .map(|cache| (cache.shape.id, &cache.slots))
                    .chain(
                        self.expert_caches
                            .iter()
                            .map(|cache| (cache.shape.id, &cache.slots)),
                    )
                    .find(|(id, _)| *id == shape.id)
                    .and_then(|(_, slots)| slots.first().copied().flatten());
            }
        }
        facts.host_chunks = self.tier_chunks();
        facts.disk_chunks = self.disk_chunks();
        facts.disk_kept = self.disk_kept();
        Ok(facts)
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
        // Every segment the fitted pool's chunks lie in, the last as far
        // as they reach: the range is reserved for more than the fit left
        // the cache rows.
        let fitted = self.fitted_chunks();
        self.chunks.limit(fitted);
        match self
            .chunks
            .commit_segments(&mut self.pool, self.chunks.segments_for(u64::from(fitted)))?
        {
            Commit::Committed => {}
            refusal => return Err(refuse(&self.pool, refusal)),
        }
        self.whole = true;
        Ok(())
    }

    /// The kv planes as another worker reads them: every declared page, one
    /// region per paged plane, a page `p` at `base + p * page_stride`. A
    /// windowed row is refused: its pages are named by its own window list, not
    /// by the kv pages a peer reads.
    pub fn export_kv(&mut self, ordinal: u32) -> Result<ExportedKv> {
        if !self.windowed_planes().is_empty() {
            return Err(Fault::Residency(
                "a windowed kv row names its pages by its own window list, so it cannot be \
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

    /// Seats the host tier of `budget` bytes: as many whole chunks of the
    /// state pool's size, in pageable memory the tier commits as moves
    /// first write it, behind a page-locked staging ring sized for a device
    /// pool of `device` bytes ([`host::staging_ring`]).
    pub fn seat_tier(&mut self, budget: u64, device: u64) -> Result<()> {
        // No more than the machine has free, known now: a tier stated larger
        // would fail mid-serve, at the first move past what the host holds.
        let free = crate::weights::available_memory();
        let budget = if budget > free {
            eprintln!(
                "engine-cuda: WARNING: the host tier's budget of {} MiB is more than this \
                 machine has free ({} MiB); the tier is seated with {} MiB",
                budget >> 20,
                free >> 20,
                free >> 20
            );
            free
        } else {
            budget
        };
        let chunk = self.chunks.layout().chunk_bytes;
        let chunks = u32::try_from(budget.checked_div(chunk).unwrap_or(0)).unwrap_or(u32::MAX);
        self.tier = match chunks {
            0 => None,
            chunks => Some(host::HostTier::new(
                chunk,
                chunks,
                host::staging_ring(device, budget),
            )?),
        };
        Ok(())
    }

    /// The host tier's chunks.
    #[must_use]
    pub fn tier_chunks(&self) -> u32 {
        self.tier.as_ref().map_or(0, host::HostTier::chunks)
    }

    /// The kv list's slots the slot file holds, in whole chunks of them.
    #[must_use]
    pub fn disk_chunks(&self) -> u32 {
        let per_chunk = self
            .chunks
            .layout()
            .shape(Kind::List)
            .map_or(0, |list| list.per_chunk);
        self.disk_pages().checked_div(per_chunk).unwrap_or(0)
    }

    /// The load's shape `shape` names.
    fn kind_of(shape: engine::chunk::ShapeId) -> Result<Kind> {
        Ok(match shape {
            layout::LIST => Kind::List,
            window if layout::window_of(window).is_some() => {
                Kind::Window(layout::window_of(window).unwrap_or(0) as u8)
            }
            layout::FIXED => Kind::Fixed,
            layout::RS => Kind::Rs,
            other => {
                return Err(Fault::program(
                    "store::chunks",
                    format!("a slot copy names shape {other}, which this load does not state"),
                ));
            }
        })
    }

    /// Where host ref `slot` of `kind` lies in the host tier, from its
    /// start: its chunk, and its place among the chunk's slots.
    fn tier_offset(&self, kind: Kind, slot: u32) -> Result<u64> {
        let shape = self.chunks.layout().shape(kind).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!("a host slot of a {} in a load that lays none", kind.name()),
            )
        })?;
        let (per_chunk, bytes) = (shape.per_chunk.max(1), shape.slot_bytes);
        let chunk = slot / per_chunk;
        let have = self.tier_chunks();
        if chunk >= have {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: u64::from(chunk) + 1,
                have: u64::from(have),
            });
        }
        Ok(u64::from(chunk) * self.chunks.layout().chunk_bytes
            + u64::from(slot % per_chunk) * bytes)
    }

    /// Where each of `refs`, host tier slots of `shape`, lies: its chunk of
    /// the tier (backed) and its byte offset in that chunk; and a slot's
    /// bytes.
    fn tier_places(
        &mut self,
        shape: engine::chunk::ShapeId,
        refs: &[u32],
    ) -> Result<(u64, Vec<(u32, u64)>)> {
        let kind = Self::kind_of(shape)?;
        let facts = self.chunks.layout().shape(kind).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!("a host slot of a {} in a load that lays none", kind.name()),
            )
        })?;
        let (per_chunk, bytes) = (facts.per_chunk.max(1), facts.slot_bytes);
        let Some(tier) = self.tier.as_mut() else {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: 1,
                have: 0,
            });
        };
        if let Some(&slot) = refs.iter().find(|&&slot| slot / per_chunk >= tier.chunks()) {
            return Err(Fault::Ceiling {
                what: "chunks of the host tier",
                need: u64::from(slot / per_chunk) + 1,
                have: u64::from(tier.chunks()),
            });
        }
        Ok((
            bytes,
            refs.iter()
                .map(|&slot| (slot / per_chunk, u64::from(slot % per_chunk) * bytes))
                .collect(),
        ))
    }

    /// See `Engine::read_tier_slots`: the bytes of host tier slots, shape
    /// by shape, each laid out as its device slot. The caller has settled
    /// every copy into them.
    pub fn read_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
    ) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for (shape, refs) in slots {
            let (bytes, places) = self.tier_places(*shape, refs)?;
            let tier = self.tier.as_ref().expect("backed above");
            let chunk_bytes = tier.chunk_bytes();
            for (chunk, offset) in places {
                let len = usize::try_from(bytes).unwrap_or(usize::MAX);
                out.extend_from_slice(tier.slice(u64::from(chunk) * chunk_bytes + offset, len)?);
            }
        }
        Ok(out)
    }

    /// See `Engine::write_tier_slots`: `bytes`, as [`Self::read_tier_slots`]
    /// lays them out, into host tier slots. The caller has settled every
    /// copy that touches them.
    pub fn write_tier_slots(
        &mut self,
        slots: &[(engine::chunk::ShapeId, Vec<u32>)],
        bytes: &[u8],
    ) -> Result<()> {
        let mut spans = Vec::new();
        for (shape, refs) in slots {
            let (len, places) = self.tier_places(*shape, refs)?;
            spans.extend(places.into_iter().map(|place| (place, len)));
        }
        let need: u64 = spans.iter().map(|&(_, len)| len).sum();
        if bytes.len() as u64 != need {
            return Err(Fault::Ceiling {
                what: "bytes of host tier slots",
                need,
                have: bytes.len() as u64,
            });
        }
        let tier = self.tier.as_ref().expect("backed above");
        let chunk_bytes = tier.chunk_bytes();
        let mut from = 0usize;
        for ((chunk, offset), len) in spans {
            let len = usize::try_from(len).unwrap_or(usize::MAX);
            tier.slice(u64::from(chunk) * chunk_bytes + offset, len)?
                .copy_from_slice(&bytes[from..from + len]);
            from += len;
        }
        Ok(())
    }

    /// Waits out every move between the device and the host tier still
    /// landing: what a byte read or write of host tier slots stands behind.
    pub fn settle_all_swaps(&mut self) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            swap.landed.settle()?;
        }
        Ok(())
    }

    /// Copies whole slots of `shape`, `(src, dst)`, from tier `from` to tier
    /// `to` (CONTRACT-BC 22), on `stream` behind every fire enqueued on it
    /// and ahead of every one after: a device ref is the runtime's (through
    /// the chunk table), a host ref a place in the host tier, a disk ref a
    /// kv list slot of the slot file. A slot is laid the same on every
    /// tier, so each moves as one span. What goes to disk is written once
    /// the stream has it; what comes from disk is read first.
    ///
    /// Between the device and the host tier the copy rides `swap` where one
    /// is given: behind what `stream` has enqueued (the fires that wrote
    /// the slots), beside the fires after, of which one that names a slot it
    /// holds waits for it ([`Pools::fence_swaps`]).
    pub fn copy_tier_slots(
        &mut self,
        stream: *mut core::ffi::c_void,
        swap: Option<*mut core::ffi::c_void>,
        shape: engine::chunk::ShapeId,
        from: Tier,
        to: Tier,
        pairs: &[(u32, u32)],
    ) -> Result<()> {
        if pairs.is_empty() || from == to {
            return Ok(());
        }
        let kind = Self::kind_of(shape)?;
        let bytes = self
            .chunks
            .layout()
            .shape(kind)
            .map(|shape| shape.slot_bytes)
            .ok_or_else(|| {
                Fault::program(
                    "store::chunks",
                    format!("a slot copy of a {} in a load that lays none", kind.name()),
                )
            })?;
        let len = usize::try_from(bytes).unwrap_or(usize::MAX);
        if (from == Tier::Disk || to == Tier::Disk) && kind != Kind::List {
            return Err(Fault::Unbound {
                what: format!(
                    "a {} to or from disk: the slot file holds kv list slots only",
                    kind.name()
                ),
            });
        }
        let device: Vec<u64> = pairs
            .iter()
            .filter_map(|&(src, dst)| match (from, to) {
                (Tier::Device, _) => Some(u64::from(src)),
                (_, Tier::Device) => Some(u64::from(dst)),
                _ => None,
            })
            .collect();
        self.seat_pages(kind, device.iter().copied())?;
        match (from, to) {
            (Tier::Device, Tier::Host) | (Tier::Host, Tier::Device) => {
                let to_host = to == Tier::Host;
                let mut slots = Vec::with_capacity(pairs.len());
                for &(src, dst) in pairs {
                    let (on_device, on_host) = if to_host { (src, dst) } else { (dst, src) };
                    let on_host = self.tier_offset(kind, on_host)?;
                    let on_device = self.chunks.span(kind, on_device, 0, bytes)?;
                    slots.push(if to_host {
                        (on_device, on_host, len)
                    } else {
                        (on_host, on_device, len)
                    });
                }
                // Behind the fires that wrote the slots, and behind every
                // move before it (the swap stream's own order); without a
                // swap stream, on `stream` behind every move still landing.
                let lane = match swap {
                    Some(swap) => {
                        let written = crate::device::graph::Event::new()?;
                        written.record(stream)?;
                        written.wait(swap)?;
                        swap
                    }
                    None => {
                        self.fence_all_swaps(stream)?;
                        stream
                    }
                };
                let tier = self.tier.as_mut().ok_or(Fault::Ceiling {
                    what: "chunks of the host tier",
                    need: 1,
                    have: 0,
                })?;
                let landed = if to_host {
                    tier.to_host(lane, &slots)?
                } else {
                    tier.to_device(lane, &slots)?
                };
                if let (Some(landed), Some(_)) = (landed, swap) {
                    self.swaps.push(Swap {
                        landed,
                        paged: if kind == Kind::List {
                            device.iter().map(|&slot| slot as u32).collect()
                        } else {
                            Vec::new()
                        },
                        host: Vec::new(),
                        wide: kind != Kind::List,
                    });
                }
                Ok(())
            }
            (Tier::Device, Tier::Disk) | (Tier::Disk, Tier::Device) => {
                self.fence_all_swaps(stream)?;
                let to_disk = to == Tier::Disk;
                let (device, disk): (Vec<u32>, Vec<u32>) = pairs
                    .iter()
                    .map(|&(src, dst)| if to_disk { (src, dst) } else { (dst, src) })
                    .unzip();
                self.through_disk(stream, to_disk, &disk, |pools, at| {
                    pools.chunks.span(kind, device[at], 0, bytes)
                })
            }
            (Tier::Host, Tier::Disk) | (Tier::Disk, Tier::Host) => {
                let to_disk = to == Tier::Disk;
                // A copy still on a stream may read or write a host slot.
                self.settle_swaps()?;
                crate::device::ctx::sync(stream)?;
                let (host, disk): (Vec<u32>, Vec<u32>) = pairs
                    .iter()
                    .map(|&(src, dst)| if to_disk { (src, dst) } else { (dst, src) })
                    .unzip();
                let host = host
                    .into_iter()
                    .map(|slot| self.tier_offset(kind, slot))
                    .collect::<Result<Vec<u64>>>()?;
                self.host_through_disk(to_disk, &disk, &host, len)
            }
            _ => Ok(()),
        }
    }

    /// Moves kv list slots between the slot file's slots `disk` and the
    /// host tier's slots at `host` (offsets), `len` bytes each, through the
    /// slot file's stage, each once every move into it has landed.
    fn host_through_disk(
        &mut self,
        to_disk: bool,
        disk: &[u32],
        host: &[u64],
        len: usize,
    ) -> Result<()> {
        let have = u64::from(self.disk_pages());
        if let Some(&slot) = disk.iter().find(|&&slot| u64::from(slot) >= have) {
            return Err(Fault::Ceiling {
                what: "disk kv pages",
                need: u64::from(slot) + 1,
                have,
            });
        }
        let (Some(spill), Some(tier)) = (self.disk.as_mut(), self.tier.as_ref()) else {
            return Ok(());
        };
        let (stage, stride) = (spill.stage.host(), spill.file.stride());
        // SAFETY: the stage is this load's pinned page; no copy on a stream
        // reads or writes it while the host does (the device moves that go
        // through it wait for each other, `landing`).
        let staged = unsafe { std::slice::from_raw_parts_mut(stage, stride) };
        for (&slot, &offset) in disk.iter().zip(host) {
            let place = tier.slice(offset, len)?;
            if to_disk {
                staged[..len].copy_from_slice(place);
                spill.file.write(slot, staged).map_err(slot_file)?;
            } else {
                spill.file.read(slot, staged).map_err(slot_file)?;
                place.copy_from_slice(&staged[..len]);
            }
        }
        Ok(())
    }

    /// Moves kv list slots between the slot file's slots `disk` and the
    /// device addresses `at(i)` through the stage, one slot at a time.
    fn through_disk(
        &mut self,
        stream: *mut core::ffi::c_void,
        to_disk: bool,
        disk: &[u32],
        at: impl Fn(&Self, usize) -> Result<u64>,
    ) -> Result<()> {
        let have = u64::from(self.disk_pages());
        if let Some(&slot) = disk.iter().find(|&&slot| u64::from(slot) >= have) {
            return Err(Fault::Ceiling {
                what: "disk kv pages",
                need: u64::from(slot) + 1,
                have,
            });
        }
        let bytes = self
            .chunks
            .layout()
            .shape(Kind::List)
            .map_or(0, |list| list.slot_bytes);
        let Some(mut spill) = self.disk.take() else {
            return Ok(());
        };
        let moved = (|| {
            let (stage, stride) = (spill.stage.host(), spill.file.stride());
            // SAFETY: the stage is this load's pinned page, and every copy on
            // the stream that touches it is waited for before the host reads
            // or writes it.
            let staged = || unsafe { std::slice::from_raw_parts_mut(stage, stride) };
            for (i, &slot) in disk.iter().enumerate() {
                if spill.landing {
                    crate::device::ctx::sync(stream)?;
                    spill.landing = false;
                }
                let place = at(self, i)?;
                if to_disk {
                    crate::device::copy_any(stream, stage as u64, place, bytes as usize)?;
                    crate::device::ctx::sync(stream)?;
                    spill.file.write(slot, staged()).map_err(slot_file)?;
                } else {
                    spill.file.read(slot, staged()).map_err(slot_file)?;
                    crate::device::copy_any(stream, place, stage as u64, bytes as usize)?;
                    spill.landing = true;
                }
            }
            Ok(())
        })();
        self.disk = Some(spill);
        moved
    }

    /// Seats `pages` host kv pages, each pinned a chunk at a time as a copy
    /// first touches it ([`Shelf`]), on reserved huge pages where the host
    /// keeps a pool that holds them. Pinning at once would hold boot for as
    /// long as the driver takes to pin it all, and when a process dies the
    /// driver unpins its memory under a lock every process on the host
    /// waits on for any allocation, about 0.17 s a GiB of 4 KiB pages and a
    /// fifth of that of huge ones.
    pub fn seat_host(&mut self, pages: u32) {
        let planes = self.paged_planes().into_iter().map(|(_, b)| b).collect();
        self.host = (pages > 0).then(|| HostKv {
            paged: Shelf::new(planes, pages),
        });
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
        // A slot holds a kv page as the planes lay it, or a kv list slot of
        // the shared pool (CONTRACT-BC 22), whichever is larger.
        let list = self
            .chunks
            .layout()
            .shape(Kind::List)
            .map_or(0, |list| list.slot_bytes);
        let slot_bytes = self.paged_page_bytes().max(list);
        let file = match keep {
            None => engine::disk::SlotFile::open(dir, &name, slot_bytes, budget),
            Some(load) => {
                let planes: Vec<u64> = self.paged_planes().iter().map(|&(_, b)| b).collect();
                let layout = format!(
                    "cuda {load} page {} planes {planes:?} list slot {list}",
                    self.paging.page_size
                );
                engine::disk::SlotFile::open_kept(dir, &name, slot_bytes, budget, &layout)
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

    /// Copies whole kv pages between the device pool and the host pages,
    /// `device[i]` with `host[i]`: on `swap`, behind what `stream` has
    /// enqueued (the fires that wrote them), so the fires after run beside
    /// it; one that holds its pages waits for it ([`Pools::fence_swaps`]).
    pub fn swap_kv(
        &mut self,
        stream: *mut core::ffi::c_void,
        swap: *mut core::ffi::c_void,
        to_host: bool,
        device: &[u32],
        host: &[u32],
    ) -> Result<()> {
        let paged: Vec<(u32, u32)> = device.iter().copied().zip(host.iter().copied()).collect();
        self.ensure_kv_pages(device)?;
        let Some(tier) = self.host.as_mut() else {
            return match host.first() {
                None => Ok(()),
                Some(_) => Err(Fault::Ceiling {
                    what: "host kv pages",
                    need: 1,
                    have: 0,
                }),
            };
        };
        tier.paged.back(host.iter().copied(), "host kv pages")?;
        let shelf = &self.host.as_ref().expect("seated above").paged;
        let mut spans: Vec<(u64, u64, usize)> = Vec::new();
        // A page's planes lie apart in its slot, so each is a span.
        for (at, (plane, page_bytes)) in self.paged_planes().into_iter().enumerate() {
            for &(d, h) in &paged {
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
            host: host.to_vec(),
            wide: false,
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
            host,
            wide: false,
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
    /// device pages a fire names: its paged runs `(first, count)`.
    pub fn fence_swaps(
        &mut self,
        stream: *mut core::ffi::c_void,
        paged: &[(u64, u64)],
    ) -> Result<()> {
        self.prune_swaps()?;
        for swap in &self.swaps {
            let holds = swap.paged.iter().any(|&page| {
                let page = u64::from(page);
                paged
                    .iter()
                    .any(|&(first, count)| first <= page && page < first + count)
            });
            if holds || swap.wide {
                swap.landed.wait(stream)?;
            }
        }
        Ok(())
    }

    /// Waits, on the host, for every move still landing.
    pub fn settle_swaps(&mut self) -> Result<()> {
        for swap in self.swaps.drain(..) {
            swap.landed.settle()?;
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
        Ok(read_shelf(self.host_shelf(slots)?, slots))
    }

    /// Writes `bytes`, as [`Self::read_host_kv`] lays them out, into host kv
    /// slots `slots`.
    pub fn write_host_kv(&mut self, slots: &[u32], bytes: &[u8]) -> Result<()> {
        write_shelf(
            self.host_shelf(slots)?,
            slots,
            bytes,
            "bytes of host kv slots",
        )
    }

    /// The host kv shelf, with the chunks holding `slots` pinned.
    fn host_shelf(&mut self, slots: &[u32]) -> Result<&Shelf> {
        let what = "host kv pages";
        let Some(host) = self.host.as_mut() else {
            return Err(Fault::Ceiling {
                what,
                need: 1,
                have: 0,
            });
        };
        host.paged.back(slots.iter().copied(), what)?;
        Ok(&host.paged)
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

    pub fn commit_frame(
        &mut self,
        demand: engine::frame::Demand,
        kv_ranges: &[(u64, u64)],
        window_pages: &[Vec<u64>],
    ) -> Result<()> {
        self.seat_pages(
            Kind::List,
            kv_ranges
                .iter()
                .flat_map(|&(first, count)| first..first.saturating_add(count)),
        )?;
        let _ = demand;
        if self.chunks.by_refs() {
            // Each window list's slot refs are its own.
            for (window, pages) in window_pages.iter().enumerate() {
                self.seat_pages(Kind::Window(window as u8), pages.iter().copied())?;
            }
            Ok(())
        } else {
            self.ensure_windowed(window_pages.iter().flatten().copied())
        }
    }
}

/// How `trace`'s kv rows sit in the state pool's chunks: the compiler's
/// recipe for a server card, its token slots this paging's pages. Where the
/// paging keeps no window (the widest spans the context) every row is a list
/// row.
fn lay(
    trace: &Trace,
    paging: &Paging,
    budgets: &poem_compiler::Budgets,
    chunk_bytes: Option<u64>,
    experts: &[usize],
    rows: bool,
    one_window: bool,
) -> Result<layout::Layout> {
    let context = if paging.window.is_some() {
        paging.context()
    } else {
        0
    };
    let mut server = poem_compiler::chunk::ChunkProfile::SERVER_CUDA;
    if let Some(bytes) = chunk_bytes {
        server.target_chunk_bytes = bytes;
    }
    let paged = server.paged(paging.page_size, context);
    let profile = poem_compiler::DeviceProfile {
        chunk: if one_window { paged.merged() } else { paged },
        ..poem_compiler::DeviceProfile::default()
    };
    let recipe =
        poem_compiler::chunk::plan(trace, budgets, &profile).map_err(|why| Fault::Unbound {
            what: format!("a chunk recipe for this load's cache rows: {why}"),
        })?;
    let banks: Vec<(usize, u64)> = if experts.is_empty() {
        Vec::new()
    } else {
        let bytes = crate::weights::plane_bytes(trace)?;
        experts.iter().map(|&param| (param, bytes[param])).collect()
    };
    let streamed = if rows {
        layout::streamed_rows(trace)
    } else {
        Vec::new()
    };
    layout::Layout::of(&recipe, &streamed, &banks)
}

/// The chunks a pool of `pages` kv pages takes: the null chunk, the pages'
/// and the windowed pages beside them.
fn chunks_at(layout: &layout::Layout, paging: &Paging, pages: u64, slots: u32) -> u32 {
    let chunks = 1
        + layout.chunks_for(Kind::List, pages)
        + (0..layout.windows.len())
            .map(|window| {
                layout.chunks_for(Kind::Window(window as u8), paging.window_pages_at(pages))
            })
            .sum::<u64>()
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

/// A row-streamed table's cache: where each block of its rows lies (a pool
/// slot the runtime named, or the mapped host copy), the device table its
/// gather reads them through, and each block's reads since last asked.
#[derive(Debug)]
struct RowCache {
    shape: layout::RowsShape,
    /// The table's mapped host copy.
    host: u64,
    /// A u64 address a block.
    table: crate::device::Buffer,
    /// A u32 counter a block.
    hits: crate::device::Buffer,
    /// The slot ref each block is cached in.
    slots: Vec<Option<u32>>,
}

/// One bank of an expert shape's cache: its plane in an expert's slot, the
/// layer it is of, its mapped host copy, the device table its select reads
/// its experts through, and (a layer's first bank) each expert's reads.
#[derive(Debug)]
struct ExpertBank {
    plane: layout::ExpertPlane,
    layer: u32,
    host: u64,
    /// Whether a select reads this plane as its bank (a dense bank, or a
    /// packed bank's codes): it has the table. A packed bank's other planes
    /// are read through that table.
    head: bool,
    /// The planes an expert's table entry lists, as places among the
    /// cache's planes; `None` one the bank lacks.
    members: Vec<Option<usize>>,
    /// A u64 address an expert.
    table: crate::device::Buffer,
    /// A u32 counter an expert.
    hits: Option<crate::device::Buffer>,
}

/// An expert shape's cache: where each expert of each layer lies (a pool
/// slot the runtime named, or the mapped host copy).
#[derive(Debug)]
struct ExpertCache {
    shape: layout::ExpertShape,
    planes: Vec<ExpertBank>,
    /// The slot ref each atom is cached in.
    slots: Vec<Option<u32>>,
}

/// One plane a page holds bytes in: `offset` into the page's slot of the
/// state pool.
#[derive(Debug, Clone, Copy)]
struct PlaneAt {
    kind: Kind,
    offset: u64,
}

const POOL_STATE: Dtype = Dtype::Bf16;

/// A compressor's state for the kv row `row` its gather keys it by: two
/// planes (the projection, then its score) of `width` a token, `offset`
/// into the row's pages' slots.
#[derive(Debug)]
struct PoolRow {
    row: u32,
    width: u64,
    /// Keyed by a windowed row's pages.
    windowed: bool,
    /// The list those pages are slots of ([`Shape::Kv`]'s).
    list: u8,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_every_case() {
        a_seat_past_the_slot_count_is_a_row_only_a_slab_can_refuse();
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

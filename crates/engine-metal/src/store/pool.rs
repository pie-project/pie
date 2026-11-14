//! The state pool: one buffer of `total_chunks` chunks of `chunk_bytes`,
//! physical chunk `p` at `p * chunk_bytes` from its base, every kind of held
//! state (kv pages, recurrent records, rs buffer pages) in chunks of it.
//!
//! The buffer is placement sparse (Metal 4, which the context requires): its address range is
//! the whole pool from the load on, and memory comes and goes beneath it a
//! segment (`segment_chunks` chunks) at a time, always the first `n`
//! segments ([`Pool::commit_segments`], [`Pool::release_segments`]). A
//! segment edge inside a 256 KiB tile keeps that tile mapped while either
//! side is committed.
//!
//! The activation span is the top `act` chunks (`act_chunks` of the
//! largest rung), from the segment it starts in up: the top of the sparse
//! buffer ([`elastic::Elastic::reserve_top`]), and the first `n` segments
//! grow and shrink only below it. The top is backed on demand, from the end
//! down: as deep as the widest carve a fire has taken ([`Pool::cover`]) and
//! the span segments the runtime committed to lend state
//! ([`Pool::commit_span`]), whichever reaches lower. A mapped placement heap
//! is resident whether or not a kernel touches it, so nothing is backed that
//! neither needs.
//!
//! The runtime names chunks logically; the pool places each on a physical
//! chunk through its [`ChunkTable`], so emptying the top segment
//! ([`Pool::evacuate`]) moves bytes and never renames a held slot.
#![allow(unsafe_code)]

use engine::chunk::{ChunkTable, Evacuation, Growth};

use crate::device::ctx::Frame;
#[cfg(target_vendor = "apple")]
use crate::device::elastic;
use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

/// Bytes a segment aims at: four 16 MiB placement heaps.
pub const SEGMENT_BYTES: u64 = 64 << 20;

/// The bits of a page id under its buffer index where the range spans
/// `parts` buffers: the buffer takes the bits that count them under the
/// sign bit, so every id stays a non-negative `int`, as the page tables
/// kernels read hold it (`kv_page.h`'s `pie_page_at` decodes the same split
/// with this as a load constant).
#[must_use]
pub const fn part_shift(parts: u32) -> u32 {
    31 - bits_for(parts)
}

thread_local! {
    /// A cap on each buffer's bytes below the device's own, for tests that
    /// spread a small range over several buffers; 0 for none. Per thread:
    /// a load runs on the thread that asks for it, and tests run beside
    /// each other on threads of their own.
    static PART_CAP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Caps each buffer of a pool this thread loads from now on at `bytes` (0
/// lifts it): a test spreads a small model's range over several buffers.
pub fn cap_parts(bytes: u64) {
    PART_CAP.with(|cap| cap.set(bytes));
}

/// The bytes one buffer of a pool holds at most on `device`: its
/// `maxBufferLength`, or less where a test caps it ([`cap_parts`]).
#[must_use]
pub fn part_cap(device: &Context) -> u64 {
    match PART_CAP.with(std::cell::Cell::get) {
        0 => device.max_buffer(),
        cap => cap.min(device.max_buffer()),
    }
}

/// One placement sparse buffer of the range, its chunks
/// `[first, first + chunks)`.
#[derive(Debug)]
struct Part {
    first: u32,
    chunks: u32,
    /// The binding view over the sparse buffer.
    view: Buffer,
    #[cfg(target_vendor = "apple")]
    store: elastic::Elastic,
}

#[derive(Debug)]
pub struct Pool {
    /// The range's buffers, lowest chunks first; the activation span is
    /// the top of the last.
    parts: Vec<Part>,
    /// The one budget every part draws on.
    #[cfg(target_vendor = "apple")]
    arena: elastic::Arena,
    chunk_bytes: u64,
    total_chunks: u32,
    segment_chunks: u32,
    /// Chunks each part but the last holds: whole segments, as many as one
    /// buffer binds.
    part_chunks: u32,
    /// Segments backed, the first ones.
    committed: u32,
    /// The activation span's chunks, the top of the range.
    act: u32,
    /// The bytes from the range's end the current rung's carve takes.
    carve: u64,
    /// The span's segments from the top the runtime committed for state.
    span: u32,
    table: ChunkTable,
    /// How a page id splits its slot ([`Pool::split_slots`]): `None` where
    /// every paged shape's slots tile a chunk, so an id is its slot in its
    /// buffer; else the bits of a slot's index in its chunk, the chunk in
    /// its buffer over them.
    slot_shift: Option<u32>,
    /// The bits of a page id under its buffer index ([`part_shift`]).
    part_shift: u32,
}

/// The low `bits` bits of a word.
const fn low_bits(bits: u32) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    }
}

/// The bits that count `n` values `0 .. n`.
const fn bits_for(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// How a page id names a slot, as kernels resolve it (`kv_page.h`'s
/// `pie_page_at`) and the host does ([`PageIds::place`]): its buffer in the
/// bits over `part_shift` ([`part_shift`]); under them its slot in that
/// buffer, or, where a
/// paged shape's slots do not tile a chunk, the chunk in the buffer over
/// `shift` bits and the slot's index in that chunk under them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageIds {
    pub shift: Option<u32>,
    pub chunk_bytes: u64,
    pub part_shift: u32,
}

impl PageIds {
    /// The buffer page id `id` of a shape of `per_chunk` slots of
    /// `slot_bytes` a chunk lies in, and its first byte in that buffer.
    #[must_use]
    pub fn place(&self, id: u32, per_chunk: u32, slot_bytes: u64) -> (usize, u64) {
        let part = id.checked_shr(self.part_shift).unwrap_or(0) as usize;
        let local = id & low_bits(self.part_shift);
        let (chunk, index) = match self.shift {
            None => (local / per_chunk.max(1), local % per_chunk.max(1)),
            Some(shift) => (local >> shift, local & ((1 << shift) - 1)),
        };
        (
            part,
            u64::from(chunk) * self.chunk_bytes + u64::from(index) * slot_bytes,
        )
    }
}

/// The chunks a segment of `chunk_bytes` chunks holds: as many as fill
/// [`SEGMENT_BYTES`], one at least.
#[must_use]
pub fn segment_chunks(chunk_bytes: u64) -> u32 {
    u32::try_from((SEGMENT_BYTES / chunk_bytes.max(1)).max(1)).unwrap_or(u32::MAX)
}

/// Where the backed top starts in bytes in the buffer it lies in, that
/// buffer starting at chunk `start`: chunk `first`'s start, down to a
/// placement heap's seam (a pool chunk need not be one heap).
#[must_use]
pub fn pinned_from(first: u32, start: u32, chunk_bytes: u64) -> u64 {
    #[cfg(target_vendor = "apple")]
    let heap = elastic::CHUNK;
    #[cfg(not(target_vendor = "apple"))]
    let heap = 16 << 20;
    u64::from(first - start) * chunk_bytes / heap * heap
}

/// The first chunk of the segment the top `act` chunks start in, or `total`
/// where `act` is none.
#[must_use]
pub fn top_first(total: u32, segment_chunks: u32, act: u32) -> u32 {
    if act == 0 {
        return total;
    }
    let start = total.saturating_sub(act);
    start / segment_chunks.max(1) * segment_chunks.max(1)
}

/// The chunks each part but the last holds when one buffer binds at most
/// `cap` bytes: whole segments, one at least.
#[must_use]
pub fn part_chunks(cap: u64, chunk_bytes: u64, segment_chunks: u32) -> u32 {
    let seg = segment_chunks.max(1);
    let fits = u32::try_from(cap / chunk_bytes.max(1)).unwrap_or(u32::MAX);
    (fits / seg * seg).max(seg)
}

impl Pool {
    /// A pool of `total_chunks` chunks of `chunk_bytes`, segments of
    /// `segment_chunks`, the top `act` of them the activation span: as many
    /// placement sparse buffers of at most `part_cap` bytes as the range
    /// takes, under one budget, nothing backed yet.
    pub fn reserve(
        device: &Context,
        chunk_bytes: u64,
        total_chunks: u32,
        segment_chunks: u32,
        act: u32,
        part_cap: u64,
    ) -> Result<Pool> {
        if act >= total_chunks {
            return Err(Fault::Ceiling {
                what: "chunks of the state pool beside the activation span",
                need: u64::from(act) + 1,
                have: u64::from(total_chunks),
            });
        }
        if chunk_bytes == 0 || total_chunks == 0 || segment_chunks == 0 {
            return Err(Fault::Ceiling {
                what: "chunks of the state pool",
                need: 1,
                have: 0,
            });
        }
        let per = part_chunks(part_cap, chunk_bytes, segment_chunks);
        let parts = total_chunks.div_ceil(per);
        let last = (parts - 1) * per;
        if act > 0 && top_first(total_chunks, segment_chunks, act) < last {
            return Err(Fault::Residency(format!(
                "the activation span's {act} chunks reach under the last of the pool's \
                 {parts} buffers, which holds {} chunks; one fire's activation lies in one \
                 buffer",
                total_chunks - last
            )));
        }
        #[cfg(target_vendor = "apple")]
        {
            let sparse = device.sparse()?;
            let bytes = chunk_bytes.saturating_mul(u64::from(total_chunks));
            let arena = elastic::Arena::new(elastic::pages_up(bytes));
            let mut built = Vec::with_capacity(parts as usize);
            for at in 0..parts {
                let first = at * per;
                let chunks = per.min(total_chunks - first);
                let mut store = elastic::create(&sparse, &arena, u64::from(chunks) * chunk_bytes)?;
                if at + 1 == parts && act > 0 {
                    let top = top_first(total_chunks, segment_chunks, act);
                    store.reserve_top(pinned_from(top, first, chunk_bytes))?;
                }
                built.push(Part {
                    first,
                    chunks,
                    view: store.view(),
                    store,
                });
            }
            Ok(Pool {
                parts: built,
                arena,
                chunk_bytes,
                total_chunks,
                segment_chunks,
                part_chunks: per,
                committed: 0,
                act,
                carve: 0,
                span: 0,
                table: ChunkTable::identity(total_chunks),
                slot_shift: None,
                part_shift: part_shift(parts),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            Err(Fault::Deviceless)
        }
    }

    #[must_use]
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    #[must_use]
    pub fn total_chunks(&self) -> u32 {
        self.total_chunks
    }

    #[must_use]
    pub fn segment_chunks(&self) -> u32 {
        self.segment_chunks
    }

    /// The segments the pool's range holds.
    #[must_use]
    pub fn segments(&self) -> u32 {
        self.total_chunks.div_ceil(self.segment_chunks)
    }

    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.committed
    }

    /// The buffers the range spans.
    #[must_use]
    pub fn parts(&self) -> usize {
        self.parts.len()
    }

    /// The chunks each buffer but the last holds.
    #[must_use]
    pub fn part_chunks(&self) -> u32 {
        self.part_chunks
    }

    /// The buffer that holds byte `offset` of the range, and where in it.
    #[must_use]
    pub fn locate(&self, offset: u64) -> (usize, u64) {
        let chunk = u32::try_from(offset / self.chunk_bytes.max(1)).unwrap_or(u32::MAX);
        let at = ((chunk / self.part_chunks.max(1)) as usize).min(self.parts.len() - 1);
        (
            at,
            offset - u64::from(self.parts[at].first) * self.chunk_bytes,
        )
    }

    /// The view of the buffer that holds byte `offset` of the range, and the
    /// offset in it: what an encoder binds or a blit names.
    #[must_use]
    pub fn view_at(&self, offset: u64) -> (&Buffer, u64) {
        let (at, local) = self.locate(offset);
        (&self.parts[at].view, local)
    }

    /// Buffer `at`'s view.
    #[must_use]
    pub fn part_view(&self, at: usize) -> &Buffer {
        &self.parts[at].view
    }

    /// Splits page ids for the paged shapes `paged` (each's slots a chunk
    /// and bytes a slot): where every one's slots tile a chunk, an id is its
    /// slot in its buffer; else its chunk in the buffer over the bits of the
    /// widest shape's index in a chunk ([`PageIds`]). Refused by name where
    /// the buffer's chunks and a chunk's slots pass the bits under the
    /// buffer index.
    pub fn split_slots(&mut self, paged: &[(u32, u64)]) -> Result<()> {
        let chunks = self.parts.iter().map(|part| part.chunks).max().unwrap_or(0);
        let room = 1u64 << self.part_shift;
        if paged
            .iter()
            .all(|&(_, bytes)| bytes > 0 && self.chunk_bytes.is_multiple_of(bytes))
        {
            if let Some(&(per, bytes)) = paged
                .iter()
                .find(|&&(per, _)| u64::from(per) * u64::from(chunks) >= room)
            {
                return Err(Fault::Residency(format!(
                    "a buffer of {chunks} chunks holds {} slots of {bytes} bytes, and a page \
                     id names {room} slots a buffer",
                    u64::from(per) * u64::from(chunks)
                )));
            }
            self.slot_shift = None;
            return Ok(());
        }
        let widest = paged.iter().map(|&(per, _)| per).max().unwrap_or(1);
        let shift = bits_for(widest);
        if shift + bits_for(chunks) > self.part_shift {
            return Err(Fault::Residency(format!(
                "a page id names a slot by its chunk in its buffer ({chunks} chunks, {} bits) \
                 and its index in the chunk ({widest} slots, {shift} bits), and holds \
                 {} bits under its buffer ({} buffers)",
                bits_for(chunks),
                self.part_shift,
                self.parts.len()
            )));
        }
        self.slot_shift = Some(shift);
        Ok(())
    }

    /// How page ids name slots ([`PageIds`]).
    #[must_use]
    pub fn page_ids(&self) -> PageIds {
        PageIds {
            shift: self.slot_shift,
            chunk_bytes: self.chunk_bytes,
            part_shift: self.part_shift,
        }
    }

    /// The id a kernel addresses physical slot `slot` of a shape with
    /// `per_chunk` slots a chunk by, where a page is one slot
    /// ([`PageIds`]): its buffer in the top bits, its slot in that buffer
    /// under them, or its chunk in the buffer and index in the chunk.
    #[must_use]
    pub fn slot_page(&self, slot: u32, per_chunk: u32) -> u32 {
        let per = per_chunk.max(1);
        let chunk = slot / per;
        let at = ((chunk / self.part_chunks.max(1)) as usize).min(self.parts.len() - 1);
        let local_chunk = chunk - self.parts[at].first;
        let id = match self.slot_shift {
            None => slot - self.parts[at].first * per,
            Some(shift) => (local_chunk << shift) | (slot % per),
        };
        if self.parts.len() == 1 {
            return id;
        }
        ((at as u32) << self.part_shift) | id
    }

    /// The bits of a page id under its buffer index ([`part_shift`]).
    #[must_use]
    pub fn part_shift(&self) -> u32 {
        self.part_shift
    }

    /// The physical slot a kernel's page id ([`Pool::slot_page`]) names.
    #[must_use]
    pub fn page_slot(&self, id: u32, per_chunk: u32) -> u32 {
        let per = per_chunk.max(1);
        let at = id.checked_shr(self.part_shift).unwrap_or(0) as usize;
        let local = id & low_bits(self.part_shift);
        let (chunk, index) = match self.slot_shift {
            None => (local / per, local % per),
            Some(shift) => (local >> shift, local & ((1 << shift) - 1)),
        };
        self.parts
            .get(at)
            .map_or(u32::MAX, |part| (part.first + chunk) * per + index)
    }

    /// Each buffer's address less the first's, in bytes (wrapping): what a
    /// kernel adds to an address in the first to reach the same place in
    /// another.
    #[must_use]
    pub fn deltas(&self) -> Vec<u64> {
        let base = crate::device::alloc::slab_address(self.parts[0].view.slab());
        self.parts
            .iter()
            .map(|part| crate::device::alloc::slab_address(part.view.slab()).wrapping_sub(base))
            .collect()
    }

    /// The GPU address of byte `offset` of the range.
    #[must_use]
    pub fn address(&self, offset: u64) -> u64 {
        let (view, local) = self.view_at(offset);
        crate::device::alloc::slab_address(view.slab()) + local
    }

    fn last(&self) -> &Part {
        self.parts.last().expect("a pool has one buffer at least")
    }

    /// The first byte of the last buffer in the range.
    fn last_start(&self) -> u64 {
        u64::from(self.last().first) * self.chunk_bytes
    }

    /// The bottom's chunks the runtime committed: the first ones, a whole
    /// number of segments, up to the span's backed segments.
    #[must_use]
    pub fn committed_chunks(&self) -> u32 {
        self.committed
            .saturating_mul(self.segment_chunks)
            .min(self.top_first())
    }

    /// Where the backed top starts in bytes ([`pinned_from`]); the range's
    /// end where there is no span.
    fn pinned_from(&self) -> u64 {
        if self.act == 0 {
            return self.bytes();
        }
        self.last_start() + pinned_from(self.top_first(), self.last().first, self.chunk_bytes)
    }

    /// The activation span's chunks.
    #[must_use]
    pub fn act_chunks(&self) -> u32 {
        self.act
    }

    /// Where the activation span starts, in bytes from the range's start.
    #[must_use]
    pub fn span_offset(&self) -> u64 {
        u64::from(self.total_chunks - self.act) * self.chunk_bytes
    }

    /// The first chunk of the segment the span starts in: the top, in the
    /// last buffer, from it on. The range's end where there is no span.
    #[must_use]
    pub fn top_first(&self) -> u32 {
        top_first(self.total_chunks, self.segment_chunks, self.act)
    }

    /// The span's segments, from the one it starts in to the range's end.
    #[must_use]
    pub fn span_segments(&self) -> u32 {
        self.segments() - self.top_first() / self.segment_chunks.max(1)
    }

    /// Where the top is to be backed from: the widest carve's start or the
    /// lowest committed span segment's, whichever is lower.
    #[cfg(target_vendor = "apple")]
    fn top_want(&self) -> u64 {
        let carve = self.bytes().saturating_sub(self.carve);
        let span = u64::from(self.segments() - self.span.min(self.span_segments()))
            * u64::from(self.segment_chunks)
            * self.chunk_bytes;
        carve.min(span).max(self.pinned_from())
    }

    /// Backs the top `bytes` of the range for the rung's carve, on a raise;
    /// a narrower `bytes` is no change ([`Pool::narrow`] gives back).
    pub fn cover(&mut self, bytes: u64) -> Result<()> {
        if bytes <= self.carve {
            return Ok(());
        }
        let was = self.carve;
        self.carve = bytes.min(self.bytes() - self.pinned_from());
        let backed = self.back_top();
        if backed.is_err() {
            self.carve = was;
        }
        backed
    }

    /// The bytes from the range's end the rung's carve backs.
    #[must_use]
    pub fn covered(&self) -> u64 {
        self.carve
    }

    /// Whether the top `bytes` of the range are backed for a fire's carve.
    #[must_use]
    pub fn covers(&self, bytes: u64) -> bool {
        bytes <= self.carve
    }

    /// Gives back the carve past the top `bytes`, on a lower, keeping what
    /// the span segments committed for state take.
    ///
    /// # Safety
    ///
    /// No frame reads the carve given back: the caller has drained every
    /// frame wider than the rung it lowers to.
    pub unsafe fn narrow(&mut self, bytes: u64) -> Result<()> {
        if bytes >= self.carve {
            return Ok(());
        }
        self.carve = bytes;
        #[cfg(target_vendor = "apple")]
        {
            let want = self.top_want() - self.last_start();
            let part = self
                .parts
                .last_mut()
                .expect("a pool has one buffer at least");
            // SAFETY: the caller's contract above.
            unsafe { part.store.unmap_top(want)? };
        }
        Ok(())
    }

    /// Backs the top `segments` of the span for state the runtime opens
    /// there; fewer than are backed is no change.
    pub fn commit_span(&mut self, segments: u32) -> Result<()> {
        if segments > self.span_segments() {
            return Err(Fault::Ceiling {
                what: "segments of the activation span",
                need: u64::from(segments),
                have: u64::from(self.span_segments()),
            });
        }
        if segments <= self.span {
            return Ok(());
        }
        self.span = segments;
        self.back_top()
    }

    /// Gives back the span's segments past the top `segments`, down to what
    /// the widest carve still takes.
    ///
    /// # Safety
    ///
    /// No frame reads the chunks given back: the caller has drained every
    /// frame, and the runtime holds no state in them.
    pub unsafe fn release_span(&mut self, segments: u32) -> Result<()> {
        if segments >= self.span {
            return Ok(());
        }
        self.span = segments;
        #[cfg(target_vendor = "apple")]
        {
            let want = self.top_want() - self.last_start();
            let part = self
                .parts
                .last_mut()
                .expect("a pool has one buffer at least");
            // SAFETY: the caller's contract above.
            unsafe { part.store.unmap_top(want)? };
        }
        Ok(())
    }

    /// Backs the top down to [`Pool::top_want`]. Its bytes are reserved
    /// from the load (the ledger's fire class), so the budget never refuses
    /// it; a heap the system refuses is [`Fault::Refused`], which the
    /// runtime takes as a refused raise or lend, never a fault mid-fire.
    fn back_top(&mut self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            let want = self.top_want() - self.last_start();
            let part = self
                .parts
                .last_mut()
                .expect("a pool has one buffer at least");
            let need = part.store.top_mapped_from().saturating_sub(want);
            part.store.map_top(want).map_err(|fault| match fault {
                Fault::Ceiling { .. } => Fault::Refused { bytes: need },
                other => other,
            })?;
        }
        Ok(())
    }

    /// The span's committed segments from the top.
    #[must_use]
    pub fn committed_span(&self) -> u32 {
        self.span
    }

    /// The top's bytes, backed or not: the span from the heap seam under
    /// the segment it starts in, which the ledger holds reserved.
    #[must_use]
    pub fn top_bytes(&self) -> u64 {
        self.bytes() - self.pinned_from()
    }

    /// Where the top's backed bytes start in the range: its end where none
    /// are.
    fn top_mapped_from(&self) -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            self.last_start() + self.last().store.top_mapped_from()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            self.pinned_from()
        }
    }

    /// The device bytes the span's backed segments hold.
    #[must_use]
    pub fn top_held_bytes(&self) -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            self.last().store.top_held()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    #[must_use]
    pub fn growth(&self) -> Growth {
        if self.act > 0 {
            Growth::TwoEnded
        } else {
            Growth::Bottom
        }
    }

    /// The first buffer's view: what a kernel binds a plane in, reaching the
    /// other buffers by [`Pool::deltas`].
    #[must_use]
    pub fn view(&self) -> &Buffer {
        &self.parts[0].view
    }

    /// The pool's whole range.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.chunk_bytes * u64::from(self.total_chunks)
    }

    /// The bytes with memory behind them now.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            self.parts.iter().map(|part| part.store.committed()).sum()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    /// The device bytes behind the pool now: the placement heaps its
    /// committed segments and its top hold.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            self.parts.iter().map(|part| part.store.held()).sum()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    /// The most that has had memory behind it at once.
    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        #[cfg(target_vendor = "apple")]
        {
            self.arena.budget().high_water
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    /// The physical chunk logical chunk `logical` sits on.
    pub fn physical(&self, logical: u32) -> Result<u32> {
        self.table.physical(logical).ok_or_else(|| Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total_chunks),
        })
    }

    /// Where logical chunk `logical` starts, in bytes from the range's start.
    pub fn offset(&self, logical: u32) -> Result<u64> {
        Ok(u64::from(self.physical(logical)?) * self.chunk_bytes)
    }

    /// The logical-to-physical table, by logical chunk.
    #[must_use]
    pub fn table(&self) -> &ChunkTable {
        &self.table
    }

    /// Refuses a physical chunk with no memory behind it: a kernel that
    /// reads an unmapped tile faults the whole queue.
    pub fn admit(&self, physical: u32) -> Result<()> {
        // The bottom maps its committed chunks up to the heap seam the top
        // is backed from, and the top every chunk from the span's segment
        // on, which lies wholly past that seam.
        let have = self.committed_chunks();
        let backed = physical < self.total_chunks
            && (physical < have
                || (physical >= self.top_first()
                    && u64::from(physical) * self.chunk_bytes >= self.top_mapped_from()));
        if !backed {
            return Err(Fault::Ceiling {
                what: "committed chunks of the state pool",
                need: u64::from(physical) + 1,
                have: u64::from(have),
            });
        }
        Ok(())
    }

    /// Each buffer's share of the range's first `bytes`.
    #[cfg(target_vendor = "apple")]
    fn shares(&self, bytes: u64) -> Vec<u64> {
        self.parts
            .iter()
            .map(|part| {
                let start = u64::from(part.first) * self.chunk_bytes;
                bytes
                    .saturating_sub(start)
                    .min(u64::from(part.chunks) * self.chunk_bytes)
            })
            .collect()
    }

    /// Backs the first `segments` segments; what is backed already stands,
    /// and fewer than are backed is no change. Newly backed memory is zeroed
    /// before this returns, since a placement heap holds what it last held.
    pub fn commit_segments(&mut self, segments: u32) -> Result<()> {
        let max = self.segments();
        if segments > max {
            return Err(Fault::Ceiling {
                what: "segments of the state pool",
                need: u64::from(segments),
                have: u64::from(max),
            });
        }
        if segments <= self.committed {
            return Ok(());
        }
        #[cfg(target_vendor = "apple")]
        {
            let chunks = segments
                .saturating_mul(self.segment_chunks)
                .min(self.top_first());
            let bytes = (u64::from(chunks) * self.chunk_bytes).min(self.pinned_from());
            let shares = self.shares(bytes);
            let mut targets: Vec<elastic::Target<'_>> = self
                .parts
                .iter_mut()
                .zip(shares)
                .map(|(part, bytes)| elastic::Target {
                    buffer: &mut part.store,
                    bytes,
                })
                .collect();
            elastic::grow_all(&mut targets)?;
        }
        self.committed = segments;
        Ok(())
    }

    /// Gives back every segment at and past `segments`.
    ///
    /// # Safety
    ///
    /// No frame reads the chunks given back: the caller has drained every
    /// frame, and the runtime evacuated them first.
    pub unsafe fn release_segments(&mut self, segments: u32) -> Result<()> {
        if segments >= self.committed {
            return Ok(());
        }
        #[cfg(target_vendor = "apple")]
        {
            let chunks = segments
                .saturating_mul(self.segment_chunks)
                .min(self.top_first());
            let bytes = (u64::from(chunks) * self.chunk_bytes).min(self.pinned_from());
            let shares = self.shares(bytes);
            let mut targets: Vec<elastic::Target<'_>> = self
                .parts
                .iter_mut()
                .zip(shares)
                .map(|(part, bytes)| elastic::Target {
                    buffer: &mut part.store,
                    bytes,
                })
                .collect();
            // SAFETY: the caller's contract above.
            unsafe { elastic::shrink_all(&mut targets)? };
        }
        self.committed = segments;
        Ok(())
    }

    /// Copies `bytes` from byte `from` of the range to byte `to` in
    /// `frame`; neither side crosses a chunk, so neither crosses a buffer.
    pub fn copy(&self, frame: &mut Frame, from: u64, to: u64, bytes: u64) -> Result<()> {
        let (src, src_at) = self.view_at(from);
        let (dst, dst_at) = self.view_at(to);
        frame.copy_span(src, src_at, dst, dst_at, bytes)
    }

    /// Copies each moved chunk onto its new physical chunk in `frame`, then
    /// takes the moves into the table: the caller commits `frame` before any
    /// later fire is encoded, so the copies land first.
    pub fn evacuate(&mut self, frame: &mut Frame, evacuation: &Evacuation) -> Result<()> {
        let mut table = self.table.clone();
        table.apply(evacuation).map_err(|why| Fault::Program {
            at: "store::pool::evacuate",
            why: why.to_string(),
        })?;
        for &(logical, physical) in &evacuation.moves {
            let from = self.physical(logical)?;
            self.admit(from)?;
            self.admit(physical)?;
            if from == physical {
                continue;
            }
            self.copy(
                frame,
                u64::from(from) * self.chunk_bytes,
                u64::from(physical) * self.chunk_bytes,
                self.chunk_bytes,
            )?;
        }
        self.table = table;
        Ok(())
    }

    /// The host address of physical chunk `physical`'s bytes, valid until
    /// the segment holding it is given back; the caller writes through it
    /// only where no frame in flight reads them.
    pub fn chunk_host(&self, physical: u32) -> Result<std::ptr::NonNull<u8>> {
        self.admit(physical)?;
        let (at, local) = self.locate(u64::from(physical) * self.chunk_bytes);
        #[cfg(target_vendor = "apple")]
        {
            self.parts[at].store.host_span(local, self.chunk_bytes)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            self.parts[at].view.host_span(local, self.chunk_bytes)
        }
    }

    /// Walks `len` bytes at byte `offset` of the range a buffer at a time,
    /// handing `piece` the buffer, the offset in it, how far into the walk
    /// and how many bytes.
    fn walk(
        &self,
        offset: u64,
        len: u64,
        mut piece: impl FnMut(&Part, u64, u64, u64) -> Result<()>,
    ) -> Result<()> {
        let mut done = 0;
        while done < len {
            let (at, local) = self.locate(offset + done);
            let part = &self.parts[at];
            let take = (len - done).min(u64::from(part.chunks) * self.chunk_bytes - local);
            piece(part, local, done, take)?;
            done += take;
        }
        Ok(())
    }

    /// `len` bytes from `offset` of the range as host runs `(from, len,
    /// address)`, `from` counted from `offset`: one per heap they cross,
    /// valid while those heaps stay mapped; the caller writes through them
    /// only where no frame in flight reads them.
    pub fn host_runs(
        &self,
        offset: u64,
        len: u64,
    ) -> Result<Vec<(u64, u64, std::ptr::NonNull<u8>)>> {
        let mut runs = Vec::new();
        self.walk(offset, len, |part, local, done, take| {
            #[cfg(target_vendor = "apple")]
            {
                let mut piece = 0;
                while piece < take {
                    let here = local + piece;
                    let run = (take - piece).min(elastic::CHUNK - here % elastic::CHUNK);
                    runs.push((done + piece, run, part.store.host_span(here, run)?));
                    piece += run;
                }
                Ok(())
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                let _ = (part, local, done, take, &mut runs);
                Err(Fault::Deviceless)
            }
        })?;
        Ok(runs)
    }

    /// `into.len()` bytes from `offset`, read on the host.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.walk(offset, into.len() as u64, |part, local, done, take| {
            let into = &mut into[done as usize..(done + take) as usize];
            #[cfg(target_vendor = "apple")]
            {
                part.store.read_into(local, into)
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                part.view.read(local, into)
            }
        })
    }

    /// Writes `from` at `offset` on the host. The caller has drained every
    /// frame that touches these bytes.
    pub fn write(&mut self, offset: u64, from: &[u8]) -> Result<()> {
        self.walk(offset, from.len() as u64, |part, local, done, take| {
            let from = &from[done as usize..(done + take) as usize];
            #[cfg(target_vendor = "apple")]
            {
                // SAFETY: the caller's contract, restated on every host write.
                unsafe { part.store.write_from(local, from) }
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                let _ = (part, local, from);
                Err(Fault::Deviceless)
            }
        })
    }

    /// Zeroes `len` bytes at `offset` on the host; as [`Pool::write`].
    pub fn zero(&mut self, offset: u64, len: u64) -> Result<()> {
        self.walk(offset, len, |part, local, _, take| {
            #[cfg(target_vendor = "apple")]
            {
                // SAFETY: as `write`.
                unsafe { part.store.zero(local, take) }
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                let _ = (part, local, take);
                Err(Fault::Deviceless)
            }
        })
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::{Pool, part_shift, top_first};
    use crate::device::Context;

    const CHUNK: u64 = 16 << 20;

    #[test]
    fn the_span_is_backed_on_demand_from_the_top_and_the_bottom_grows_under_it() {
        assert_eq!(top_first(12, 4, 3), 8, "the segment the span starts in");
        assert_eq!(top_first(12, 4, 4), 8);
        assert_eq!(top_first(12, 4, 5), 4);
        assert_eq!(top_first(12, 4, 0), 12, "no span, no top");
        let device = Context::bind().expect("a device");
        let mut pool = Pool::reserve(&device, CHUNK, 12, 4, 3, u64::MAX).expect("a pool");
        assert_eq!(pool.growth(), engine::chunk::Growth::TwoEnded);
        assert_eq!(pool.span_offset(), 9 * CHUNK);
        assert_eq!(pool.span_segments(), 1);
        assert_eq!(pool.top_held_bytes(), 0, "nothing is backed at the load");
        assert!(pool.admit(11).is_err());
        // A fire's carve backs the top down to itself, and no further.
        pool.cover(CHUNK + 1).expect("a carve");
        assert_eq!(pool.top_held_bytes(), 2 * CHUNK);
        pool.admit(11).expect("under the carve");
        pool.admit(10).expect("under the carve");
        assert!(pool.admit(9).is_err());
        pool.cover(CHUNK)
            .expect("a narrower carve keeps the widest");
        assert_eq!(pool.top_held_bytes(), 2 * CHUNK);
        // The span's segment committed for state backs all of it.
        pool.commit_span(1).expect("the span segment");
        assert_eq!(pool.top_held_bytes(), 4 * CHUNK);
        for chunk in 8..12 {
            pool.admit(chunk).expect("the span's segment is backed");
        }
        assert!(pool.commit_span(2).is_err(), "the span has one segment");
        let mut seen = [7u8; 64];
        pool.write(10 * CHUNK + 4096, &[9u8; 64]).expect("a write");
        // The bottom grows to the span and no further.
        pool.commit_segments(3).expect("every segment");
        assert_eq!(pool.committed_chunks(), 8);
        assert_eq!(pool.held_bytes(), 12 * CHUNK);
        // SAFETY: nothing ran on this pool.
        unsafe { pool.release_segments(1) }.expect("the bottom shrinks");
        assert_eq!(pool.held_bytes(), 8 * CHUNK);
        pool.read(10 * CHUNK + 4096, &mut seen).expect("a read");
        assert_eq!(seen, [9u8; 64], "the span outlives the bottom's shrink");
        // Releasing the span gives back what the carve does not take.
        // SAFETY: nothing ran on this pool.
        unsafe { pool.release_span(0) }.expect("the span goes back");
        assert_eq!(pool.top_held_bytes(), 2 * CHUNK);
        assert!(pool.admit(9).is_err());
        pool.admit(10).expect("still under the carve");
        // A lowered rung gives back the carve above it.
        assert!(pool.covers(CHUNK + 1));
        // SAFETY: nothing ran on this pool.
        unsafe { pool.narrow(1) }.expect("the rung lowers");
        assert_eq!(pool.top_held_bytes(), CHUNK);
        assert!(!pool.covers(CHUNK + 1));
        assert!(pool.admit(10).is_err());
        pool.admit(11).expect("the narrow carve's heap stays");
        assert!(
            Pool::reserve(&device, CHUNK, 4, 4, 4, u64::MAX).is_err(),
            "a span that leaves no chunk under it"
        );
        // Chunks of 12 MiB: the span's segment starts mid-heap, and the top
        // is backed from the heap seam under it.
        let small = 12 << 20;
        let mut pool = Pool::reserve(&device, small, 10, 3, 2, u64::MAX).expect("a pool");
        assert_eq!(pool.top_first(), 6);
        pool.commit_span(pool.span_segments()).expect("the span");
        for chunk in 6..10 {
            pool.admit(chunk).expect("the span's segment is backed");
        }
        assert!(
            pool.admit(5).is_err(),
            "under the span's segment, uncommitted"
        );
        pool.write(9 * small, &[5u8; 64]).expect("a write");
        pool.write(5 * small, &[5u8; 64])
            .expect_err("chunk 5's bottom half is unmapped");
        pool.commit_segments(4).expect("every segment");
        for chunk in 0..10 {
            pool.admit(chunk).expect("all of it is backed");
        }
        // Chunk 5 straddles the seam at 64 MiB: the bottom maps its first
        // 4 MiB, the top the rest.
        pool.write(5 * small + (4 << 20) - 32, &[6u8; 64])
            .expect("across the seam");
        pool.read(5 * small + (4 << 20) - 32, &mut seen)
            .expect("a read");
        assert_eq!(seen, [6u8; 64]);
    }

    /// The buffer index takes the bits that count the buffers under the
    /// sign bit: as many buffers as the range takes, none fixed.
    #[test]
    fn a_page_ids_buffer_bits_count_the_buffers() {
        assert_eq!(part_shift(1), 31);
        assert_eq!(part_shift(2), 30);
        assert_eq!(part_shift(4), 29);
        assert_eq!(part_shift(5), 28);
        assert_eq!(part_shift(300), 22);
        let device = Context::bind().expect("a device");
        // 40 chunks, buffers of 4: ten, past any fixed four.
        let pool = Pool::reserve(&device, CHUNK, 40, 4, 3, 4 * CHUNK).expect("a pool");
        assert_eq!(pool.parts(), 10);
        assert_eq!(pool.part_shift(), 27);
        assert_eq!(pool.slot_page(37 * 8 + 5, 8), (9 << 27) | 13);
        assert_eq!(pool.page_slot((9 << 27) | 13, 8), 37 * 8 + 5);
        assert_eq!(pool.deltas().len(), 10);
    }

    #[test]
    fn a_range_past_one_buffer_spans_several_under_one_budget() {
        let device = Context::bind().expect("a device");
        // 12 chunks, segments of 4, buffers of at most 4 chunks: three.
        let mut pool = Pool::reserve(&device, CHUNK, 12, 4, 3, 4 * CHUNK).expect("a pool");
        assert_eq!(pool.parts(), 3);
        assert_eq!(pool.part_chunks(), 4);
        assert_eq!(pool.locate(5 * CHUNK + 7), (1, CHUNK + 7));
        assert_eq!(pool.locate(11 * CHUNK), (2, 3 * CHUNK));
        // Three buffers: their index in the two bits under the sign bit.
        let ps = pool.part_shift();
        assert_eq!(ps, 29);
        // Slots that tile a chunk: an id is the slot in its buffer.
        pool.split_slots(&[(8, CHUNK / 8)]).expect("whole slots");
        assert_eq!(pool.slot_page(5 * 8 + 3, 8), (1 << ps) | 11);
        assert_eq!(pool.page_slot((1 << ps) | 11, 8), 5 * 8 + 3);
        // Slots that do not (3 of CHUNK / 3 + 64 bytes a chunk): an id is
        // the chunk in its buffer over two bits of index.
        let odd = CHUNK / 3 + 64;
        pool.split_slots(&[(8, CHUNK / 8), (3, odd)])
            .expect("split slots");
        assert_eq!(pool.page_ids().shift, Some(3));
        let id = pool.slot_page(9 * 3 + 2, 3);
        assert_eq!(id, (2 << ps) | (1 << 3) | 2);
        assert_eq!(pool.page_slot(id, 3), 9 * 3 + 2);
        assert_eq!(pool.page_ids().place(id, 3, odd), (2, CHUNK + 2 * odd));
        let id = pool.slot_page(5 * 8 + 3, 8);
        assert_eq!(id, (1 << ps) | (1 << 3) | 3);
        assert_eq!(
            pool.page_ids().place(id, 8, CHUNK / 8),
            (1, CHUNK + 3 * (CHUNK / 8))
        );
        assert_eq!(pool.deltas().len(), 3);
        assert_eq!(pool.deltas()[0], 0);
        // The bottom grows across buffers under the one budget.
        pool.commit_segments(2).expect("two segments");
        for chunk in 0..8 {
            pool.admit(chunk).expect("committed");
        }
        assert_eq!(pool.held_bytes(), 8 * CHUNK);
        pool.write(5 * CHUNK + 64, &[3u8; 32])
            .expect("a write in the second buffer");
        // A copy between buffers lands.
        let mut frame = device.frame().expect("a frame");
        pool.copy(&mut frame, 5 * CHUNK + 64, CHUNK + 64, 32)
            .expect("the copy");
        frame.commit().expect("it lands");
        let mut seen = [0u8; 32];
        pool.read(CHUNK + 64, &mut seen)
            .expect("a read in the first buffer");
        assert_eq!(seen, [3u8; 32]);
        // The span is the last buffer's top.
        pool.commit_span(1).expect("the span");
        for chunk in 8..12 {
            pool.admit(chunk).expect("the span is backed");
        }
        assert_eq!(pool.held_bytes(), 12 * CHUNK);
        // A span that reaches under the last buffer is refused.
        assert!(Pool::reserve(&device, CHUNK, 9, 4, 3, 4 * CHUNK).is_err());
        // Past four buffers: as many as the range takes.
        assert_eq!(
            Pool::reserve(&device, CHUNK, 20, 4, 0, 4 * CHUNK)
                .expect("five buffers")
                .parts(),
            5
        );
    }
}

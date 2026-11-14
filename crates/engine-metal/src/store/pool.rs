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

#[derive(Debug)]
enum Backing {
    #[cfg(not(target_vendor = "apple"))]
    Flat(Buffer),
    #[cfg(target_vendor = "apple")]
    Sparse {
        /// The binding view over the sparse buffer.
        view: Buffer,
        store: elastic::Elastic,
        arena: elastic::Arena,
    },
}

#[derive(Debug)]
pub struct Pool {
    backing: Backing,
    chunk_bytes: u64,
    total_chunks: u32,
    segment_chunks: u32,
    /// Segments backed, the first ones.
    committed: u32,
    /// The activation span's chunks, the top of the range.
    act: u32,
    /// The bytes from the range's end the current rung's carve takes.
    carve: u64,
    /// The span's segments from the top the runtime committed for state.
    span: u32,
    table: ChunkTable,
}

/// The chunks a segment of `chunk_bytes` chunks holds: as many as fill
/// [`SEGMENT_BYTES`], one at least.
#[must_use]
pub fn segment_chunks(chunk_bytes: u64) -> u32 {
    u32::try_from((SEGMENT_BYTES / chunk_bytes.max(1)).max(1)).unwrap_or(u32::MAX)
}

/// Where the backed top starts in bytes: chunk `first`'s start, down to a
/// placement heap's seam (a pool chunk need not be one heap).
#[must_use]
pub fn pinned_from(first: u32, chunk_bytes: u64) -> u64 {
    #[cfg(target_vendor = "apple")]
    let heap = elastic::CHUNK;
    #[cfg(not(target_vendor = "apple"))]
    let heap = 16 << 20;
    u64::from(first) * chunk_bytes / heap * heap
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

impl Pool {
    /// A pool of `total_chunks` chunks of `chunk_bytes`, segments of
    /// `segment_chunks`, the top `act` of them the activation span:
    /// placement sparse with only the span's segments backed where `device`
    /// maps sparse buffers, else flat and whole.
    pub fn reserve(
        device: &Context,
        chunk_bytes: u64,
        total_chunks: u32,
        segment_chunks: u32,
        act: u32,
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
        let bytes = chunk_bytes.saturating_mul(u64::from(total_chunks));
        #[cfg(target_vendor = "apple")]
        {
            let sparse = device.sparse()?;
            let arena = elastic::Arena::new(elastic::pages_up(bytes));
            let mut store = elastic::create(&sparse, &arena, bytes)?;
            if act > 0 {
                let first = top_first(total_chunks, segment_chunks, act);
                store.reserve_top(pinned_from(first, chunk_bytes))?;
            }
            Ok(Pool {
                backing: Backing::Sparse {
                    view: store.view(),
                    store,
                    arena,
                },
                chunk_bytes,
                total_chunks,
                segment_chunks,
                committed: 0,
                act,
                carve: 0,
                span: 0,
                table: ChunkTable::identity(total_chunks),
            })
        }
        // No device off Apple: a flat pool stands in for the tests that
        // build one, and `Buffer::zeroed` refuses there.
        #[cfg(not(target_vendor = "apple"))]
        let mut pool = Pool {
            backing: Backing::Flat(Buffer::zeroed(device, bytes)?),
            chunk_bytes,
            total_chunks,
            segment_chunks,
            committed: 0,
            act,
            carve: 0,
            span: 0,
            table: ChunkTable::identity(total_chunks),
        };
        #[cfg(not(target_vendor = "apple"))]
        {
            pool.committed = pool.segments();
            Ok(pool)
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
        pinned_from(self.top_first(), self.chunk_bytes)
    }

    /// The activation span's chunks.
    #[must_use]
    pub fn act_chunks(&self) -> u32 {
        self.act
    }

    /// Where the activation span starts, in bytes from the pool's base.
    #[must_use]
    pub fn span_offset(&self) -> u64 {
        u64::from(self.total_chunks - self.act) * self.chunk_bytes
    }

    /// The first chunk of the segment the span starts in: it and every one
    /// above are backed from the load (the runtime counts that segment
    /// committed and opens state below the span inside it). The range's end
    /// where there is no span.
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
            let want = self.top_want();
            let Backing::Sparse { store, .. } = &mut self.backing;
            // SAFETY: the caller's contract above.
            unsafe { store.unmap_top(want)? };
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
            let want = self.top_want();
            let Backing::Sparse { store, .. } = &mut self.backing;
            // SAFETY: the caller's contract above.
            unsafe { store.unmap_top(want)? };
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
            let want = self.top_want();
            let Backing::Sparse { store, .. } = &mut self.backing;
            let need = store.top_mapped_from().saturating_sub(want);
            store.map_top(want).map_err(|fault| match fault {
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

    /// Where the top's backed bytes start: the range's end where none are.
    fn top_mapped_from(&self) -> u64 {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(_) => self.pinned_from(),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.top_mapped_from(),
        }
    }

    /// The device bytes the span's backed segments hold.
    #[must_use]
    pub fn top_held_bytes(&self) -> u64 {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(_) => u64::from(self.total_chunks - self.top_first()) * self.chunk_bytes,
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.top_held(),
        }
    }

    #[must_use]
    pub fn growth(&self) -> Growth {
        if !self.is_sparse() {
            Growth::Fixed
        } else if self.act > 0 {
            Growth::TwoEnded
        } else {
            Growth::Bottom
        }
    }

    #[must_use]
    pub fn is_sparse(&self) -> bool {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(_) => false,
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { .. } => true,
        }
    }

    /// The buffer kernels and blits bind: the whole pool.
    #[must_use]
    pub fn view(&self) -> &Buffer {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer,
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { view, .. } => view,
        }
    }

    /// The pool's whole range.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.chunk_bytes * u64::from(self.total_chunks)
    }

    /// The bytes with memory behind them now.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer.bytes(),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.committed(),
        }
    }

    /// The device bytes behind the pool now: the placement heaps its
    /// committed segments hold, or the whole buffer where it is flat.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer.bytes(),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.held(),
        }
    }

    /// The most that has had memory behind it at once.
    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer.bytes(),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { arena, .. } => arena.budget().high_water,
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

    /// Where logical chunk `logical` starts, in bytes from the pool's base.
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
        let (top, pinned) = (self.top_first(), self.pinned_from());
        #[cfg(target_vendor = "apple")]
        {
            let Backing::Sparse { store, .. } = &mut self.backing;
            let chunks = segments.saturating_mul(self.segment_chunks).min(top);
            let bytes = (u64::from(chunks) * self.chunk_bytes).min(pinned);
            elastic::grow_all(&mut [elastic::Target {
                buffer: store,
                bytes,
            }])?;
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
        let (top, pinned) = (self.top_first(), self.pinned_from());
        #[cfg(target_vendor = "apple")]
        {
            let Backing::Sparse { store, .. } = &mut self.backing;
            let bytes = (u64::from(segments.saturating_mul(self.segment_chunks).min(top))
                * self.chunk_bytes)
                .min(pinned);
            // SAFETY: the caller's contract above.
            unsafe {
                elastic::shrink_all(&mut [elastic::Target {
                    buffer: store,
                    bytes,
                }])?;
            }
        }
        #[cfg(target_vendor = "apple")]
        {
            self.committed = segments;
        }
        Ok(())
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
            let view = self.view();
            frame.copy_span(
                view,
                u64::from(from) * self.chunk_bytes,
                view,
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
        let offset = u64::from(physical) * self.chunk_bytes;
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer.host_span(offset, self.chunk_bytes),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.host_span(offset, self.chunk_bytes),
        }
    }

    /// `len` bytes from `offset` of the buffer as host runs `(from, len,
    /// address)`, `from` counted from `offset`: one per heap they cross,
    /// valid while those heaps stay mapped; the caller writes through them
    /// only where no frame in flight reads them.
    pub fn host_runs(
        &self,
        offset: u64,
        len: u64,
    ) -> Result<Vec<(u64, u64, std::ptr::NonNull<u8>)>> {
        let mut runs = Vec::new();
        let mut done = 0;
        while done < len {
            let here = offset + done;
            #[cfg(target_vendor = "apple")]
            let seam = elastic::CHUNK;
            #[cfg(not(target_vendor = "apple"))]
            let seam = u64::MAX;
            let take = (len - done).min(seam - here % seam);
            let address = match &self.backing {
                #[cfg(not(target_vendor = "apple"))]
                Backing::Flat(buffer) => buffer.host_span(here, take)?,
                #[cfg(target_vendor = "apple")]
                Backing::Sparse { store, .. } => store.host_span(here, take)?,
            };
            runs.push((done, take, address));
            done += take;
        }
        Ok(runs)
    }

    /// `into.len()` bytes from `offset`, read on the host.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        match &self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer.read(offset, into),
            #[cfg(target_vendor = "apple")]
            Backing::Sparse { store, .. } => store.read_into(offset, into),
        }
    }

    /// Writes `from` at `offset` on the host. The caller has drained every
    /// frame that touches these bytes.
    pub fn write(&mut self, offset: u64, from: &[u8]) -> Result<()> {
        match &mut self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer.write(offset, from),
            #[cfg(target_vendor = "apple")]
            // SAFETY: the caller's contract, restated on every host write.
            Backing::Sparse { store, .. } => unsafe { store.write_from(offset, from) },
        }
    }

    /// Zeroes `len` bytes at `offset` on the host; as [`Pool::write`].
    pub fn zero(&mut self, offset: u64, len: u64) -> Result<()> {
        match &mut self.backing {
            #[cfg(not(target_vendor = "apple"))]
            Backing::Flat(buffer) => buffer.zero_span(offset, len),
            #[cfg(target_vendor = "apple")]
            // SAFETY: as `write`.
            Backing::Sparse { store, .. } => unsafe { store.zero(offset, len) },
        }
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::{Pool, top_first};
    use crate::device::Context;

    const CHUNK: u64 = 16 << 20;

    #[test]
    fn the_span_is_backed_on_demand_from_the_top_and_the_bottom_grows_under_it() {
        assert_eq!(top_first(12, 4, 3), 8, "the segment the span starts in");
        assert_eq!(top_first(12, 4, 4), 8);
        assert_eq!(top_first(12, 4, 5), 4);
        assert_eq!(top_first(12, 4, 0), 12, "no span, no top");
        let device = Context::bind().expect("a device");
        let mut pool = Pool::reserve(&device, CHUNK, 12, 4, 3).expect("a pool");
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
            Pool::reserve(&device, CHUNK, 4, 4, 4).is_err(),
            "a span that leaves no chunk under it"
        );
        // Chunks of 12 MiB: the span's segment starts mid-heap, and the top
        // is backed from the heap seam under it.
        let small = 12 << 20;
        let mut pool = Pool::reserve(&device, small, 10, 3, 2).expect("a pool");
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
}

//! The state pool's chunks on the device: one address range
//! ([`Range`]), the table that places each logical chunk on a physical one,
//! and the ids the paged kernels read slots by.
//!
//! A lane names slots by slot ref (`logical chunk * slots a chunk + index`).
//! Until the runtime hands those out, a lane names today's page ids, and
//! [`Seating`] seats each on a slot the first time it is named: the lowest
//! free chunk opened for its shape, as the runtime's pool will.

use engine::chunk::{ChunkTable, Evacuation};

use crate::device::elastic::{Commit, PhysicalPool, Range};
use crate::error::{Fault, Result};
use crate::store::layout::{Layout, TokenShape};

const UNSEATED: u32 = u32::MAX;

/// Today's page ids of one shape, each seated on a slot ref as it is first
/// named.
#[derive(Debug, Default)]
struct Seating {
    refs: Vec<u32>,
    /// The chunk being filled and the next free index in it.
    open: Option<(u32, u32)>,
}

#[derive(Debug)]
pub struct Chunks {
    range: Range,
    layout: Layout,
    total: u32,
    table: ChunkTable,
    seated: [Seating; 2],
    /// The lowest chunk no shape has opened; chunk 0 is the null chunk.
    next: u32,
    null_zeroed: bool,
}

impl Chunks {
    /// Reserves `total` chunks of addresses; nothing is backed yet.
    pub fn reserve(pool: &PhysicalPool, layout: Layout, total: u32) -> Result<Chunks> {
        let total = total.max(1);
        let bytes = u64::from(total) * layout.chunk_bytes;
        let segment = u64::from(layout.segment_chunks) * layout.chunk_bytes;
        Ok(Chunks {
            range: Range::reserve(pool, bytes, segment, "bytes of the state pool")?,
            layout,
            total,
            table: ChunkTable::identity(total),
            seated: [Seating::default(), Seating::default()],
            next: 1,
            null_zeroed: false,
        })
    }

    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[must_use]
    pub fn base(&self) -> u64 {
        self.range.base()
    }

    #[must_use]
    pub fn total(&self) -> u32 {
        self.total
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.range.bytes()
    }

    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.range.committed_bytes()
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.range.high_water_bytes()
    }

    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.range.committed_segments()
    }

    #[must_use]
    pub fn segments(&self) -> u32 {
        self.range.segments()
    }

    /// Whether anything of the range is backed.
    #[must_use]
    pub fn fresh(&self) -> bool {
        self.range.committed_bytes() == 0
    }

    /// The segments that hold the first `chunks` chunks.
    #[must_use]
    pub fn segments_for(&self, chunks: u64) -> u32 {
        u32::try_from(chunks.div_ceil(u64::from(self.layout.segment_chunks.max(1))))
            .unwrap_or(u32::MAX)
            .min(self.range.segments())
    }

    /// Backs the first `segments` segments, and zeroes the null chunk once
    /// it is backed: every page behind a window reads it, and a masked
    /// column still multiplies its value, so it must hold no NaN.
    pub fn commit_segments(&mut self, pool: &mut PhysicalPool, segments: u32) -> Result<Commit> {
        let outcome = self.range.commit_segments(pool, segments)?;
        if outcome == Commit::Committed && !self.null_zeroed && self.range.committed_segments() > 0
        {
            let bytes = self.layout.chunk_bytes;
            crate::device::zero_span(
                self.range.span(0, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
            self.null_zeroed = true;
        }
        Ok(outcome)
    }

    /// Gives back the segments past the first `segments`.
    pub fn release_segments(&mut self, pool: &mut PhysicalPool, segments: u32) -> Result<()> {
        if segments == 0 {
            self.null_zeroed = false;
        }
        self.range.release_segments(pool, segments)
    }

    fn shape(&self, ring: bool) -> Result<&TokenShape> {
        self.layout.shape(ring).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!(
                    "a {} page named in a load that lays none",
                    if ring { "windowed" } else { "kv" }
                ),
            )
        })
    }

    /// Seats page `id` of today's ids on a slot, opening the lowest free
    /// chunk for its shape where the one being filled is full. The windowed
    /// id 0 is the null slot. Returns the logical chunk the slot lies in.
    pub fn seat(&mut self, ring: bool, id: u32) -> Result<u32> {
        if ring && id == 0 {
            return Ok(0);
        }
        let per_chunk = self.shape(ring)?.per_chunk;
        let seating = &mut self.seated[usize::from(ring)];
        if let Some(&slot) = seating.refs.get(id as usize)
            && slot != UNSEATED
        {
            return Ok(slot / per_chunk);
        }
        let (chunk, index) = match seating.open {
            Some((chunk, index)) if index < per_chunk => (chunk, index),
            _ => {
                if self.next >= self.total {
                    return Err(Fault::Ceiling {
                        what: "chunks of the state pool",
                        need: u64::from(self.next) + 1,
                        have: u64::from(self.total),
                    });
                }
                self.next += 1;
                (self.next - 1, 0)
            }
        };
        seating.open = Some((chunk, index + 1));
        if seating.refs.len() <= id as usize {
            seating.refs.resize(id as usize + 1, UNSEATED);
        }
        seating.refs[id as usize] = chunk * per_chunk + index;
        Ok(chunk)
    }

    /// The chunks opened so far, the null chunk among them.
    #[must_use]
    pub fn opened(&self) -> u32 {
        self.next
    }

    /// The slot ref page `id` is seated on.
    pub fn slot_ref(&self, ring: bool, id: u32) -> Result<u32> {
        if ring && id == 0 {
            return Ok(0);
        }
        self.seated[usize::from(ring)]
            .refs
            .get(id as usize)
            .copied()
            .filter(|&slot| slot != UNSEATED)
            .ok_or_else(|| {
                Fault::program(
                    "store::chunks",
                    format!(
                        "{} page {id} is read before it is seated",
                        if ring { "windowed" } else { "kv" }
                    ),
                )
            })
    }

    /// The physical chunk and the index in it of page `id`'s slot.
    fn place(&self, ring: bool, id: u32) -> Result<(u32, u32)> {
        let shape = self.shape(ring)?;
        let slot = self.slot_ref(ring, id)?;
        let logical = slot / shape.per_chunk;
        let physical = self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total),
        })?;
        Ok((physical, slot % shape.per_chunk))
    }

    /// Bytes from the pool's base to page `id`'s slot.
    pub fn slot_at(&self, ring: bool, id: u32) -> Result<u64> {
        let (chunk, index) = self.place(ring, id)?;
        Ok(self
            .shape(ring)?
            .slot_at(self.layout.chunk_bytes, chunk, index))
    }

    /// The address of `len` bytes at `offset` of page `id`'s slot.
    pub fn span(&self, ring: bool, id: u32, offset: u64, len: u64) -> Result<u64> {
        self.range.span(self.slot_at(ring, id)? + offset, len)
    }

    /// The id a paged kernel reads page `id`'s slot by.
    pub fn page_id(&self, ring: bool, id: u32) -> Result<i32> {
        let (chunk, index) = self.place(ring, id)?;
        let q = self
            .shape(ring)?
            .page_id(self.layout.chunk_bytes, chunk, index);
        i32::try_from(q).map_err(|_| Fault::Ceiling {
            what: "page ids of the state pool",
            need: q,
            have: i32::MAX as u64,
        })
    }

    /// Rewrites today's page ids in `ids` as the ids the kernels read; an id
    /// under zero (a row that writes nothing) stays.
    pub fn page_ids(&self, ring: bool, ids: &mut [i32]) -> Result<()> {
        for id in ids {
            if *id >= 0 {
                *id = self.page_id(ring, *id as u32)?;
            }
        }
        Ok(())
    }

    /// Moves each logical chunk of `evacuation` onto its new physical chunk:
    /// the bytes in one batched copy on `stream`, then the table, so every
    /// fire enqueued after reads the chunk where it now lies. The runtime
    /// names only free physical chunks as destinations.
    pub fn evacuate(
        &mut self,
        stream: *mut core::ffi::c_void,
        evacuation: &Evacuation,
    ) -> Result<()> {
        let chunk = self.layout.chunk_bytes;
        let mut spans = Vec::with_capacity(evacuation.moves.len());
        for &(logical, to) in &evacuation.moves {
            let from = self.table.physical(logical).ok_or(Fault::Ceiling {
                what: "chunks of the state pool",
                need: u64::from(logical) + 1,
                have: u64::from(self.total),
            })?;
            if from == to {
                continue;
            }
            spans.push((
                self.range.span(u64::from(to) * chunk, chunk)?,
                self.range.span(u64::from(from) * chunk, chunk)?,
                usize::try_from(chunk).unwrap_or(0),
            ));
        }
        let mut table = self.table.clone();
        table
            .apply(evacuation)
            .map_err(|why| Fault::program("store::chunks", why.to_string()))?;
        crate::device::copy_batch_d2d(stream, &spans)?;
        self.table = table;
        Ok(())
    }

    /// The physical chunk logical chunk `logical` lies on.
    #[must_use]
    pub fn physical(&self, logical: u32) -> Option<u32> {
        self.table.physical(logical)
    }
}

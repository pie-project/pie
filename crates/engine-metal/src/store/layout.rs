//! How a load's held state lays into the pool's slot shapes, and the chunk
//! facts the runtime counts them by.
//!
//! Three shapes, each a run of bytes the engine lays out itself (the slot's
//! inside is the engine's, the runtime only counts slots):
//!
//! - [`LIST`]: a kv page, `page_size` tokens of every kv row layer-major,
//!   each row its key run then its value run, every run 256-aligned. A kernel
//!   reads row `r` of physical page `q` at `q * slot_bytes + offset(r)`
//!   ([`kernels_metal::KvPool`]'s page stride and plane offset), so a chunk
//!   is a whole number of list slots and `q = chunk * per_chunk + index`.
//! - [`FIXED`]: a working set's recurrent record, every state row whole,
//!   256-aligned; a state kernel finds it through its row's segment table
//!   ([`kernels_metal::StateMap`]) by the physical slot id.
//! - [`BUFFER`]: one rs buffer page (`page_size` tokens of every replayed
//!   plane). Its kernels step pages by a stride; the stride is
//!   `gcd(chunk, slot)` and a page's id counts such units from the base.
//!
//! The chunk size comes from the compiler's on-device Metal recipe, raised to
//! the largest slot and rounded to a whole number of list slots. A record
//! larger than that chunk is not cut yet: the chunk grows to hold it.
//!
//! A lane the runtime does not address by slot refs (a direct
//! [`crate::Shell::fire`]) is *seat-addressed*: seat `s` takes the list slots
//! `s * pages_per_slot ..` of a run kept for seats, the fixed slot `s` of
//! another, and so on; those runs sit in the lowest chunks after the null
//! chunk, in the order list, fixed, buffer.

use engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotShape,
};

use crate::error::{Fault, Result};

pub const LIST: ShapeId = ShapeId(0);
pub const FIXED: ShapeId = ShapeId(1);
pub const BUFFER: ShapeId = ShapeId(2);

/// Where every run inside a slot starts, and every slot's size.
pub const ALIGN: u64 = 256;

/// The chunk the on-device Metal recipe aims at when the trace packs into
/// none.
pub const TARGET_CHUNK: u64 = 2 << 20;

#[must_use]
pub fn align(bytes: u64) -> u64 {
    bytes.div_ceil(ALIGN) * ALIGN
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// One slot shape's place in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slots {
    pub id: ShapeId,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    /// The run seat-addressed lanes take: its first chunk and its slots.
    pub seat_chunk: u32,
    pub seat_slots: u32,
}

impl Slots {
    fn of(id: ShapeId, slot_bytes: u64, chunk_bytes: u64) -> Option<Slots> {
        (slot_bytes > 0).then(|| Slots {
            id,
            slot_bytes,
            per_chunk: u32::try_from(chunk_bytes / slot_bytes).unwrap_or(u32::MAX),
            seat_chunk: 0,
            seat_slots: 0,
        })
    }

    /// The logical chunk slot ref `slot` lies in, and its index there.
    #[must_use]
    pub fn split(&self, slot: u32) -> (u32, u32) {
        (slot / self.per_chunk, slot % self.per_chunk)
    }

    /// The slot ref seat-addressed id `id` stands for.
    pub fn seat_ref(&self, id: u32) -> Result<u32> {
        if id >= self.seat_slots {
            return Err(Fault::Ceiling {
                what: "seat-addressed slots of a pool shape",
                need: u64::from(id) + 1,
                have: u64::from(self.seat_slots),
            });
        }
        Ok(self.seat_chunk * self.per_chunk + id)
    }

    fn seat_chunks(&self) -> u32 {
        self.seat_slots.div_ceil(self.per_chunk.max(1))
    }
}

/// The pool's chunk size, its shapes and its extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub chunk_bytes: u64,
    pub total_chunks: u32,
    pub segment_chunks: u32,
    pub page_size: u32,
    pub list: Option<Slots>,
    pub fixed: Option<Slots>,
    pub buffer: Option<Slots>,
}

impl Layout {
    /// The chunk for slots of `list`, `fixed` and `buffer` bytes (zero:
    /// no such shape) aiming at `target`: no smaller than any slot, and a
    /// whole number of list slots.
    #[must_use]
    pub fn chunk_bytes(list: u64, fixed: u64, buffer: u64, target: u64) -> u64 {
        let unit = if list > 0 { list } else { ALIGN };
        let least = list.max(fixed).max(buffer).max(target).max(ALIGN);
        least.div_ceil(unit) * unit
    }

    /// The layout of `total_chunks` chunks of `chunk_bytes` for slots of
    /// these bytes, with seat runs for `seats` seats of `pages_per_seat`
    /// kv pages each.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_bytes: u64,
        total_chunks: u32,
        segment_chunks: u32,
        page_size: u32,
        slot_bytes: [u64; 3],
        seats: u32,
        pages_per_seat: u32,
    ) -> Result<Layout> {
        let [list, fixed, buffer] = slot_bytes;
        let mut layout = Layout {
            chunk_bytes,
            total_chunks,
            segment_chunks,
            page_size,
            list: Slots::of(LIST, list, chunk_bytes),
            fixed: Slots::of(FIXED, fixed, chunk_bytes),
            buffer: Slots::of(BUFFER, buffer, chunk_bytes),
        };
        if let Some(shape) = layout.shapes().find(|s| s.per_chunk == 0) {
            return Err(Fault::Ceiling {
                what: "bytes of a pool chunk",
                need: shape.slot_bytes,
                have: chunk_bytes,
            });
        }
        let mut at = 1u32;
        let runs = [
            (&mut layout.list, seats.saturating_mul(pages_per_seat)),
            (&mut layout.fixed, seats),
            (&mut layout.buffer, seats),
        ];
        for (shape, slots) in runs {
            if let Some(shape) = shape {
                shape.seat_chunk = at;
                shape.seat_slots = slots;
                at = at.saturating_add(shape.seat_chunks());
            }
        }
        Ok(layout)
    }

    pub fn shapes(&self) -> impl Iterator<Item = &Slots> {
        self.list.iter().chain(&self.fixed).chain(&self.buffer)
    }

    #[must_use]
    pub fn shape(&self, id: ShapeId) -> Option<&Slots> {
        self.shapes().find(|s| s.id == id)
    }

    /// The chunks the seat runs take past the null chunk.
    #[must_use]
    pub fn seat_chunks(&self) -> u32 {
        self.shapes().map(Slots::seat_chunks).sum()
    }

    /// The unit an rs buffer page's id counts: `gcd(chunk, page)`.
    #[must_use]
    pub fn buffer_stride(&self) -> u64 {
        self.buffer
            .map_or(ALIGN, |b| gcd(self.chunk_bytes, b.slot_bytes))
    }

    /// The chunks one working set at `pages` kv pages takes beside the null
    /// chunk: its pages, its record and the one a fire writes beside it,
    /// its buffer pages.
    #[must_use]
    pub fn least_chunks(chunk_bytes: u64, slot_bytes: [u64; 3], pages: u32) -> u32 {
        let [list, fixed, buffer] = slot_bytes;
        let chunks = |slots: u32, bytes: u64| -> u32 {
            if bytes == 0 {
                return 0;
            }
            let per = u32::try_from((chunk_bytes / bytes).max(1)).unwrap_or(u32::MAX);
            slots.div_ceil(per)
        };
        1 + chunks(pages, list) + chunks(2, fixed) + chunks(pages, buffer)
    }

    /// The pool as the runtime counts it: one shared chunk space, chunk 0
    /// null, segments of `segment_chunks` growing from the bottom where the
    /// pool is sparse; activation lives in the arena apart.
    #[must_use]
    pub fn facts(&self, growth: Growth, committed: u32, rows: &[Vec<u32>; 3]) -> ChunkFacts {
        let segments = self.total_chunks.div_ceil(self.segment_chunks.max(1));
        let shape = |slots: &Slots, kind: SlotKind, rows: &Vec<u32>| SlotShape {
            id: slots.id,
            kind,
            slot_bytes: slots.slot_bytes,
            slots_per_chunk: slots.per_chunk,
            sets_per_working_set: kind.sets_per_working_set(),
            rows: rows.clone(),
            pinned: None,
        };
        let mut shapes = Vec::new();
        if let Some(list) = &self.list {
            shapes.push(shape(
                list,
                SlotKind::List {
                    tokens: self.page_size,
                    rate: 1,
                },
                &rows[0],
            ));
        }
        if let Some(fixed) = &self.fixed {
            shapes.push(shape(fixed, SlotKind::Fixed { pieces: 1 }, &rows[1]));
        }
        if let Some(buffer) = &self.buffer {
            shapes.push(shape(buffer, SlotKind::Fixed { pieces: 1 }, &rows[2]));
        }
        ChunkFacts {
            chunk_bytes: self.chunk_bytes,
            total_chunks: self.total_chunks,
            segment_chunks: self.segment_chunks,
            segments: SegmentFacts {
                committed: if growth == Growth::Fixed {
                    segments
                } else {
                    committed.min(segments)
                },
                max: segments,
                growth,
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            }],
            shapes,
            act_chunks: Vec::new(),
            host_chunks: 0,
            disk_chunks: 0,
            disk_persistent: false,
            disk_kept: false,
            weights: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_holds_whole_list_slots_and_the_largest_slot() {
        // A recipe chunk of 2 MiB over 48 KiB pages: 43 pages.
        assert_eq!(
            Layout::chunk_bytes(48 << 10, 1 << 20, 64 << 10, 2 << 20),
            43 * (48 << 10)
        );
        // A record past the recipe's chunk grows the chunk to hold it.
        let chunk = Layout::chunk_bytes(48 << 10, 10_338_560, 0, 2 << 20);
        assert!(chunk >= 10_338_560 && chunk.is_multiple_of(48 << 10));
        assert!(chunk - 10_338_560 < 48 << 10);
        // No kv: the largest slot, 256-aligned.
        assert_eq!(Layout::chunk_bytes(0, 3 << 20, 0, 2 << 20), 3 << 20);
    }

    #[test]
    fn seats_take_runs_from_chunk_one_and_the_facts_hang_together() {
        let chunk = Layout::chunk_bytes(48 << 10, 256 << 10, 80 << 10, 2 << 20);
        let layout = Layout::new(
            chunk,
            100,
            crate::store::pool::segment_chunks(chunk),
            16,
            [48 << 10, 256 << 10, 80 << 10],
            4,
            8,
        )
        .unwrap();
        let list = layout.list.unwrap();
        let fixed = layout.fixed.unwrap();
        let buffer = layout.buffer.unwrap();
        assert_eq!((list.seat_chunk, list.seat_slots), (1, 32));
        assert_eq!(fixed.seat_chunk, 2, "32 pages fit the first chunk");
        assert_eq!(buffer.seat_chunk, 3, "four records fit a chunk");
        assert_eq!(list.seat_ref(9).unwrap(), list.per_chunk + 9);
        assert!(list.seat_ref(32).is_err());
        assert_eq!(layout.seat_chunks(), 3);
        assert_eq!(layout.buffer_stride(), gcd(chunk, 80 << 10));
        let facts = layout.facts(Growth::Bottom, 1, &Default::default());
        facts.validate().unwrap();
        assert_eq!(facts.state_chunks(), 99);
        assert_eq!(facts.segments.committed, 1);
        assert!(facts.shapes.iter().all(|s| s.pinned.is_none()));
        let flat = layout.facts(Growth::Fixed, 0, &Default::default());
        flat.validate().unwrap();
        assert_eq!(flat.segments.committed, flat.segments.max);
    }

    #[test]
    fn a_working_set_takes_its_pages_two_records_and_its_buffer() {
        let bytes = [64 << 10, 1 << 20, 0];
        let chunk = Layout::chunk_bytes(bytes[0], bytes[1], bytes[2], 2 << 20);
        // 32 pages a chunk: 128 pages are four; two records share one.
        assert_eq!(Layout::least_chunks(chunk, bytes, 128), 1 + 4 + 1);
    }
}

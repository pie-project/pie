//! The pool's slot shapes and extent, and the chunk facts the runtime counts
//! them by. The compiler's on-device Metal recipe (`poem_compiler::chunk`)
//! is the one source of the chunk size and of what a slot holds where
//! ([`crate::store::Rows`] reads it); this module only places the shapes:
//!
//! - [`LIST`]: a kv page, the recipe's list slot: `page_size` tokens of every
//!   kv row, layer-major. A kernel reads row `r` of physical page `q` at
//!   `q * slot_bytes + offset(r)` ([`kernels_metal::KvPool`]'s page stride
//!   and plane offset); a chunk is a whole number of list slots, so
//!   `q = chunk * per_chunk + index`.
//! - [`FIXED`]: one piece of a working set's record, `pieces` of them a
//!   record: every state row (whole, or cut into equal pieces at one offset)
//!   and the rs buffer's `[16, width]` rows. A state kernel finds a row
//!   through its segment table ([`kernels_metal::StateMap`]) by the physical
//!   ids of the pieces it lies in.
//!
//! A lane the runtime does not address by slot refs (a direct
//! [`crate::Shell::fire`]) is *seat-addressed*: seat `s` takes the list slots
//! `s * pages_per_slot ..` of a run kept for seats, and the record of fixed
//! slots `s * pieces ..` of another; those runs sit in the lowest chunks
//! after the null chunk.

use engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole, SlotShape,
};

use crate::error::{Fault, Result};

pub const LIST: ShapeId = ShapeId(0);
pub const FIXED: ShapeId = ShapeId(1);
/// The first window list's shape.
pub const WINDOW: ShapeId = ShapeId(2);
pub const RS: ShapeId = ShapeId(3);

/// The most window lists a load pages (CONTRACT-BC 21).
pub const MAX_WINDOWS: usize = 4;

/// Each window list's shape, in the recipe's order: the first is
/// [`WINDOW`], the others follow the rs list.
pub const WINDOWS: [ShapeId; MAX_WINDOWS] = [WINDOW, ShapeId(4), ShapeId(5), ShapeId(6)];

/// One window list's place in the pool: its slots, the rows its window
/// retains and its tokens (`T_w`) a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowList {
    pub slots: Slots,
    pub rows: u32,
    pub tokens: u32,
}

/// The chunk an on-device Metal recipe aims at.
pub const TARGET_CHUNK: u64 = 2 << 20;

#[must_use]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
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
    /// The list slot's tokens: the kv page size.
    pub page_size: u32,
    /// The fixed slots one record takes.
    pub pieces: u32,
    pub list: Option<Slots>,
    pub fixed: Option<Slots>,
    /// The window lists (CONTRACT-BC 19, 21), one a distinct window: each
    /// `T_w` tokens a slot of the rows it windows, a table in its own token
    /// order a lane; slots trimmed behind the window are the null chunk.
    pub windows: [Option<WindowList>; MAX_WINDOWS],
    /// The rs list (CONTRACT-BC 18, 20): `rs_tokens` unfolded rows of every
    /// recurrent layer's rs planes a slot.
    pub rs: Option<Slots>,
    pub rs_tokens: u32,
    /// The seats seat-addressed lanes take.
    pub seats: u32,
}

impl Layout {
    /// The layout of `total_chunks` chunks of `chunk_bytes` for list slots
    /// of `list` bytes and fixed slots of `fixed` bytes (`pieces` a record),
    /// with seat runs for `seats` seats of `pages_per_seat` kv pages.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_bytes: u64,
        total_chunks: u32,
        segment_chunks: u32,
        page_size: u32,
        [list, fixed]: [u64; 2],
        pieces: u32,
        seats: u32,
        pages_per_seat: u32,
    ) -> Result<Layout> {
        let mut layout = Layout {
            chunk_bytes,
            total_chunks,
            segment_chunks,
            page_size,
            pieces: pieces.max(1),
            list: Slots::of(LIST, list, chunk_bytes),
            fixed: Slots::of(FIXED, fixed, chunk_bytes),
            windows: [None; MAX_WINDOWS],
            rs: None,
            rs_tokens: 0,
            seats,
        };
        if let Some(shape) = layout.shapes().find(|s| s.per_chunk == 0) {
            return Err(Fault::Ceiling {
                what: "bytes of a pool chunk",
                need: shape.slot_bytes,
                have: chunk_bytes,
            });
        }
        let mut at = 1u32;
        let records = seats.saturating_mul(layout.pieces);
        for (shape, slots) in [
            (&mut layout.list, seats.saturating_mul(pages_per_seat)),
            (&mut layout.fixed, records),
        ] {
            if let Some(shape) = shape {
                shape.seat_chunk = at;
                shape.seat_slots = slots;
                at = at.saturating_add(shape.seat_chunks());
            }
        }
        Ok(layout)
    }

    /// This layout with one more window list, of `bytes` slots of `tokens`
    /// rows retaining `rows` rows; its seat run, `pages_per_seat` slots a
    /// seat, after the others.
    pub fn with_window(
        mut self,
        bytes: u64,
        rows: u32,
        tokens: u32,
        pages_per_seat: u32,
    ) -> Result<Layout> {
        let at = self.windows.iter().take_while(|w| w.is_some()).count();
        let Some(&id) = WINDOWS.get(at) else {
            return Err(Fault::Ceiling {
                what: "window lists of a load",
                need: at as u64 + 1,
                have: MAX_WINDOWS as u64,
            });
        };
        let Some(mut window) = Slots::of(id, bytes, self.chunk_bytes) else {
            return Ok(self);
        };
        if window.per_chunk == 0 {
            return Err(Fault::Ceiling {
                what: "bytes of a pool chunk",
                need: bytes,
                have: self.chunk_bytes,
            });
        }
        let seats = self.seats;
        window.seat_chunk = 1 + self.seat_chunks();
        window.seat_slots = seats.saturating_mul(pages_per_seat);
        self.windows[at] = Some(WindowList {
            slots: window,
            rows,
            tokens,
        });
        Ok(self)
    }

    /// The window lists, in order.
    pub fn window_lists(&self) -> impl Iterator<Item = &WindowList> {
        self.windows.iter().flatten()
    }

    /// Window list `k`.
    #[must_use]
    pub fn window(&self, k: usize) -> Option<&WindowList> {
        self.windows.get(k).and_then(Option::as_ref)
    }

    /// This layout with an rs list of `bytes` slots of `tokens` rows; its
    /// seat run, `pages_per_seat` slots a seat, after the others.
    pub fn with_rs(mut self, bytes: u64, tokens: u32, pages_per_seat: u32) -> Result<Layout> {
        let Some(mut rs) = Slots::of(RS, bytes, self.chunk_bytes) else {
            return Ok(self);
        };
        if rs.per_chunk == 0 {
            return Err(Fault::Ceiling {
                what: "bytes of a pool chunk",
                need: bytes,
                have: self.chunk_bytes,
            });
        }
        rs.seat_chunk = 1 + self.seat_chunks();
        rs.seat_slots = self.seats.saturating_mul(pages_per_seat);
        self.rs = Some(rs);
        self.rs_tokens = tokens;
        Ok(self)
    }

    pub fn shapes(&self) -> impl Iterator<Item = &Slots> {
        self.list
            .iter()
            .chain(&self.fixed)
            .chain(self.window_lists().map(|window| &window.slots))
            .chain(&self.rs)
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

    /// What an rs-list page id counts: `gcd(chunk, rs slot)`.
    #[must_use]
    pub fn rs_stride(&self) -> u64 {
        self.rs
            .map_or(256, |rs| gcd(self.chunk_bytes, rs.slot_bytes))
    }

    /// The chunks one working set at `pages` kv pages takes beside the null
    /// chunk: its pages, its record and one a fire writes copy-on-write.
    #[must_use]
    pub fn least_chunks(chunk_bytes: u64, [list, fixed]: [u64; 2], pieces: u32, pages: u32) -> u32 {
        let chunks = |slots: u32, bytes: u64| -> u32 {
            if bytes == 0 {
                return 0;
            }
            let per = u32::try_from((chunk_bytes / bytes).max(1)).unwrap_or(u32::MAX);
            slots.div_ceil(per)
        };
        1 + chunks(pages, list) + chunks(2 * pieces.max(1), fixed)
    }

    /// The pool as the runtime counts it: one shared chunk space, chunk 0
    /// null, segments of `segment_chunks` growing from the bottom where the
    /// pool is sparse; activation lives in the arena apart.
    #[must_use]
    pub fn facts(&self, growth: Growth, committed: u32, rows: &[Vec<u32>; 2]) -> ChunkFacts {
        let segments = self.total_chunks.div_ceil(self.segment_chunks.max(1));
        let shape = |slots: &Slots, kind: SlotKind, rows: &Vec<u32>| SlotShape {
            id: slots.id,
            kind,
            slot_bytes: slots.slot_bytes,
            slots_per_chunk: slots.per_chunk,
            sets_per_working_set: kind.sets_per_working_set(),
            rows: rows.clone(),
            pinned: None,
            role: SlotRole::State,
        };
        let mut shapes = Vec::new();
        if let Some(list) = &self.list {
            shapes.push(shape(
                list,
                SlotKind::List {
                    tokens: self.page_size,
                    rate: 1,
                    window: None,
                },
                &rows[0],
            ));
        }
        // The window lists: lists of their own slots, each with its window
        // (CONTRACT-BC 19, 21).
        for window in self.window_lists() {
            shapes.push(shape(
                &window.slots,
                SlotKind::List {
                    tokens: window.tokens,
                    rate: 1,
                    window: Some(window.rows),
                },
                &Vec::new(),
            ));
        }
        if let Some(rs) = &self.rs {
            shapes.push(SlotShape {
                role: SlotRole::RsBuffer,
                ..shape(
                    rs,
                    SlotKind::List {
                        tokens: self.rs_tokens,
                        rate: 1,
                        window: None,
                    },
                    &Vec::new(),
                )
            });
        }
        if let Some(fixed) = &self.fixed {
            shapes.push(shape(
                fixed,
                SlotKind::Fixed {
                    pieces: self.pieces,
                },
                &rows[1],
            ));
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
    fn seats_take_runs_from_chunk_one_and_the_facts_hang_together() {
        // 1 MiB chunks of 64 KiB pages and 256 KiB pieces, two a record.
        let chunk = 1 << 20;
        let layout = Layout::new(
            chunk,
            100,
            crate::store::pool::segment_chunks(chunk),
            64,
            [64 << 10, 256 << 10],
            2,
            4,
            8,
        )
        .unwrap();
        let list = layout.list.unwrap();
        let fixed = layout.fixed.unwrap();
        assert_eq!((list.seat_chunk, list.seat_slots), (1, 32));
        assert_eq!(fixed.seat_chunk, 3, "32 pages fill two chunks");
        assert_eq!(fixed.seat_slots, 8, "four records of two pieces");
        assert_eq!(list.seat_ref(9).unwrap(), list.per_chunk + 9);
        assert!(list.seat_ref(32).is_err());
        assert_eq!(layout.seat_chunks(), 4);
        let facts = layout.facts(Growth::Bottom, 1, &Default::default());
        facts.validate().unwrap();
        assert_eq!(facts.state_chunks(), 99);
        assert_eq!(facts.segments.committed, 1);
        assert!(facts.shapes.iter().all(|s| s.pinned.is_none()));
        assert_eq!(
            facts.shape(FIXED).unwrap().kind,
            SlotKind::Fixed { pieces: 2 }
        );
        let flat = layout.facts(Growth::Fixed, 0, &Default::default());
        flat.validate().unwrap();
        assert_eq!(flat.segments.committed, flat.segments.max);
    }

    #[test]
    fn a_working_set_takes_its_pages_and_two_records() {
        // 32 pages a chunk: 128 pages are four; two records of one piece
        // share one chunk, of three pieces take two.
        let bytes = [64 << 10, 1 << 20];
        assert_eq!(Layout::least_chunks(2 << 20, bytes, 1, 128), 1 + 4 + 1);
        assert_eq!(Layout::least_chunks(2 << 20, bytes, 3, 128), 1 + 4 + 3);
    }
}

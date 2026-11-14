//! The pool's slot shapes and extent, and the chunk facts the runtime counts
//! them by. The compiler's GPU recipe (`poem_compiler::chunk`, see
//! [`crate::chunks::recipe`]) is the one source of the chunk size and of
//! what a slot holds where ([`crate::store::Rows`] reads it); this module
//! only places the shapes:
//!
//! - [`LIST`]: a kv page, the recipe's list slot: `page_size` tokens of every
//!   kv row (windowed rows too: this backend pages them in full), layer-major.
//!   A kernel reads row `r` of physical page `q` at `q * slot_bytes +
//!   offset(r)` of the pool's virtual space ([`kernels_slang::KvPool`]'s page
//!   stride and origin); a chunk is a whole number of list slots, so `q =
//!   chunk * per_chunk + index`.
//! - [`FIXED`]: one piece of a working set's record, `pieces` of them a
//!   record: every state row at its offset. A state kernel finds a lane's
//!   row at `unit * gcd(chunk, slot) + offset` ([`kernels_slang::ssm::StatePool`]).
//! - [`RS`]: the rs list (CONTRACT-BC 18, 20): `rs_tokens` unfolded rows of
//!   every recurrent layer's rs planes a slot.
//! - [`window_shape`]`(k)`: window list `k` (CONTRACT-BC 19, 21), narrowest
//!   window first: `T_w` tokens of every row of that window a slot,
//!   layer-major, a page table of its own; a slot trimmed behind the window
//!   is ref 0 (the null chunk), which the window kernels never read.
//!
//! - [`INSTANCE`], [`CHANNEL`]: a guest instance's bind-time launch tables
//!   and the cells a channel keeps on the device (`SlotRole::Instance`,
//!   `SlotRole::Channel`; REDESIGN step 4, `new-memory/STATE-ROLES.md`),
//!   `Fixed { pieces: 1 }` shapes no working set holds: the runtime takes
//!   them per instance and channel. An item past a slot lies in a run of
//!   adjacent slots of one chunk.
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
/// The first (narrowest) window list.
pub const WINDOW: ShapeId = ShapeId(2);
pub const RS: ShapeId = ShapeId(3);

/// A guest instance's bind-time launch tables (`SlotRole::Instance`). The
/// weight shapes follow the guest shapes (`store::rows::WEIGHTS`).
pub const INSTANCE: ShapeId = ShapeId(7);
/// The cells a channel keeps on the device (`SlotRole::Channel`).
pub const CHANNEL: ShapeId = ShapeId(8);

/// An `Instance` slot's bytes: a measured instance's tables in one
/// (`new-memory/STATE-ROLES.md` section 3). Structural, not config.
pub const INSTANCE_SLOT: u64 = 4 << 10;
/// A `Channel` slot's bytes.
pub const CHANNEL_SLOT: u64 = 1 << 10;

/// The most window lists a load holds (`ChunkFacts::validate`'s bound).
pub const WINDOWS: usize = 4;

/// Window list `k`'s shape: [`WINDOW`], then 4, 5, 6.
#[must_use]
pub fn window_shape(k: usize) -> ShapeId {
    match k {
        0 => WINDOW,
        k => ShapeId(3 + u32::try_from(k).unwrap_or(u32::MAX)),
    }
}

/// Which window list shape `id` is, if one.
#[must_use]
pub fn window_of(id: ShapeId) -> Option<usize> {
    (0..WINDOWS).find(|&k| window_shape(k) == id)
}

/// The chunk a GPU recipe aims at.
pub const TARGET_CHUNK: u64 = 4 << 20;

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

/// A window list's place: its slots, the rows its window retains and the
/// tokens (`T_w`) a slot holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowList {
    pub slots: Slots,
    pub window: u32,
    pub tokens: u32,
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
    /// The window lists (CONTRACT-BC 19, 21), narrowest window first, list
    /// `k` shape [`window_shape`]`(k)`: a table in its own token order a
    /// lane; slots trimmed behind the window are the null chunk.
    pub windows: [Option<WindowList>; WINDOWS],
    /// The rs list (CONTRACT-BC 18, 20): `rs_tokens` unfolded rows of every
    /// recurrent layer's rs planes a slot.
    pub rs: Option<Slots>,
    pub rs_tokens: u32,
    /// The seats seat-addressed lanes take.
    pub seats: u32,
    /// Guest instances' tables and the cells channels keep on the device
    /// (REDESIGN step 4): shapes no working set holds, taken per instance
    /// and channel. No channel shape where the backend keeps every cell on
    /// the host.
    pub instance: Option<Slots>,
    pub channel: Option<Slots>,
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
            windows: [None; WINDOWS],
            rs: None,
            rs_tokens: 0,
            seats,
            instance: None,
            channel: None,
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

    /// This layout with window list `k` of `bytes` slots of `tokens` rows,
    /// retaining `window` rows; its seat run, `pages_per_seat` slots a seat,
    /// after the others.
    pub fn with_window(
        mut self,
        k: usize,
        bytes: u64,
        window: u32,
        tokens: u32,
        pages_per_seat: u32,
    ) -> Result<Layout> {
        if k >= WINDOWS {
            return Err(Fault::Ceiling {
                what: "window lists of a load",
                need: k as u64 + 1,
                have: WINDOWS as u64,
            });
        }
        let Some(mut slots) = Slots::of(window_shape(k), bytes, self.chunk_bytes) else {
            return Ok(self);
        };
        if slots.per_chunk == 0 {
            return Err(Fault::Ceiling {
                what: "bytes of a pool chunk",
                need: bytes,
                have: self.chunk_bytes,
            });
        }
        slots.seat_chunk = 1 + self.seat_chunks();
        slots.seat_slots = self.seats.saturating_mul(pages_per_seat);
        self.windows[k] = Some(WindowList {
            slots,
            window,
            tokens,
        });
        Ok(self)
    }

    /// This layout with the guest shapes: [`INSTANCE`], and [`CHANNEL`]
    /// where `channel_cells` (the backend keeps channel cells on the
    /// device).
    pub fn with_guests(mut self, channel_cells: bool) -> Result<Layout> {
        for (id, bytes) in [(INSTANCE, INSTANCE_SLOT), (CHANNEL, CHANNEL_SLOT)] {
            if id == CHANNEL && !channel_cells {
                continue;
            }
            let slots = Slots::of(id, bytes, self.chunk_bytes).filter(|s| s.per_chunk > 0);
            let Some(slots) = slots else {
                return Err(Fault::Ceiling {
                    what: "bytes of a pool chunk",
                    need: bytes,
                    have: self.chunk_bytes,
                });
            };
            if id == INSTANCE {
                self.instance = Some(slots);
            } else {
                self.channel = Some(slots);
            }
        }
        Ok(self)
    }

    /// Window list `k`, where the load holds one.
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
            .chain(self.windows.iter().flatten().map(|w| &w.slots))
            .chain(&self.rs)
            .chain(&self.instance)
            .chain(&self.channel)
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
    /// pool is sparse; `act_chunks` the activation rungs on its top chunks.
    #[must_use]
    pub fn facts(
        &self,
        growth: Growth,
        committed: u32,
        rows: &[Vec<u32>],
        [host_chunks, disk_chunks]: [u32; 2],
        act_chunks: &[(u32, u32)],
    ) -> ChunkFacts {
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
                rows.first().unwrap_or(&Vec::new()),
            ));
        }
        // The window lists: lists of their own slot with their window
        // (CONTRACT-BC 19, 21).
        for (k, list) in self.windows.iter().enumerate() {
            if let Some(list) = list {
                shapes.push(shape(
                    &list.slots,
                    SlotKind::List {
                        tokens: list.tokens,
                        rate: 1,
                        window: Some(list.window),
                    },
                    rows.get(2 + k).unwrap_or(&Vec::new()),
                ));
            }
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
                rows.get(1).unwrap_or(&Vec::new()),
            ));
        }
        for (slots, role) in [
            (&self.instance, SlotRole::Instance),
            (&self.channel, SlotRole::Channel),
        ] {
            if let Some(slots) = slots {
                // The role, not the count, keeps these out of a working
                // set: the runtime takes them per instance and channel.
                shapes.push(SlotShape {
                    role,
                    ..shape(slots, SlotKind::Fixed { pieces: 1 }, &Vec::new())
                });
            }
        }
        ChunkFacts {
            span_backed_on_demand: false,
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
            act_chunks: act_chunks.to_vec(),
            host_chunks,
            disk_chunks,
            disk_persistent: false,
            disk_kept: false,
            weights: Vec::new(),
        }
    }
}

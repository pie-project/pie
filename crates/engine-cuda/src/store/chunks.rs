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
use crate::store::layout::{Kind, Layout, SlotShape};

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
    /// The chunks state may take: the pool's but the activation's.
    total: u32,
    /// The chunks under the range's end a fire's activation takes.
    act: u32,
    table: ChunkTable,
    seated: [Seating; Kind::COUNT],
    /// The lowest chunk no shape has opened; chunk 0 is the null chunk.
    next: u32,
    null_zeroed: bool,
    /// Lanes name slot refs: nothing is seated here any more.
    refs: bool,
    /// The chunks the runtime named weight slots in: no page is seated in
    /// one ([`Chunks::keep_clear`]).
    weights: std::collections::BTreeSet<u32>,
}

impl Chunks {
    /// Reserves `total` chunks of addresses for state and, above them, the
    /// chunks `act_bytes` of activation take; nothing is backed yet.
    pub fn reserve(
        pool: &PhysicalPool,
        layout: Layout,
        total: u32,
        act_bytes: u64,
    ) -> Result<Chunks> {
        let act = u32::try_from(act_bytes.div_ceil(layout.chunk_bytes.max(1))).unwrap_or(u32::MAX);
        // A flat range (no device memory mapped on demand: ROCm) is one
        // allocation of every chunk at once, so it can hold no more chunks
        // than the pool's ceiling backs; a mapped range only reserves
        // addresses, and the fit decides how many chunks are ever backed.
        let total = if crate::device::elastic::mapped_on_demand() {
            total
        } else {
            let backed =
                pool.hard_pages().saturating_mul(pool.page_bytes()) / layout.chunk_bytes.max(1);
            total.min(
                u32::try_from(backed)
                    .unwrap_or(u32::MAX)
                    .saturating_sub(act),
            )
        };
        let total = total.max(1);
        let bytes = (u64::from(total) + u64::from(act)) * layout.chunk_bytes;
        let segment = u64::from(layout.segment_chunks) * layout.chunk_bytes;
        Ok(Chunks {
            range: Range::reserve(pool, bytes, segment, "bytes of the state pool")?,
            layout,
            total,
            act,
            table: ChunkTable::identity(total.saturating_add(act)),
            seated: Default::default(),
            next: 1,
            null_zeroed: false,
            refs: false,
            weights: std::collections::BTreeSet::new(),
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

    /// The bytes the committed segments back; the activation's are apart.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.range.segment_bytes_committed()
    }

    /// Every byte the range has backed: its committed segments and held
    /// spans, or, a flat range once allocated, all of it.
    #[must_use]
    pub fn backed_bytes(&self) -> u64 {
        self.range.committed_bytes()
    }

    /// The unit the range is backed in.
    #[must_use]
    pub fn map_unit(&self) -> u64 {
        self.range.map_unit()
    }

    /// The chunks a fire's activation takes, under the range's end.
    #[must_use]
    pub fn act_chunks(&self) -> u32 {
        self.act
    }

    /// The bytes of the activation's span under the range's end.
    #[must_use]
    pub fn activation_bytes(&self) -> u64 {
        u64::from(self.act) * self.layout.chunk_bytes
    }

    /// The bytes of the activation's span not yet held backed: what a
    /// fire's activation will still take beside the state's chunks. None on
    /// a flat range once allocated: its one allocation backs the whole span,
    /// held or not, so no byte of it is free for the cache rows.
    #[must_use]
    pub fn activation_unheld_bytes(&self) -> u64 {
        if !crate::device::elastic::mapped_on_demand() && self.range.committed_bytes() > 0 {
            return 0;
        }
        let chunk = self.layout.chunk_bytes;
        (u64::from(self.act) * chunk).saturating_sub(self.range.held(u64::from(self.total) * chunk))
    }

    /// Where a fire's activation starts: the range's end less its chunks.
    #[must_use]
    pub fn activation_base(&self) -> u64 {
        self.range.base() + u64::from(self.total) * self.layout.chunk_bytes
    }

    /// Backs the first `bytes` of the activation's span, and keeps them
    /// backed; `beside` the pool's budget where it refuses them.
    pub fn hold_activation(
        &mut self,
        pool: &mut PhysicalPool,
        bytes: u64,
        beside: bool,
    ) -> Result<Commit> {
        let from = u64::from(self.total) * self.layout.chunk_bytes;
        let room = u64::from(self.act) * self.layout.chunk_bytes;
        if bytes > room {
            return Err(Fault::Ceiling {
                what: "bytes of activation under the state pool's end",
                need: bytes,
                have: room,
            });
        }
        self.range.hold(pool, from, bytes, beside)
    }

    /// Backs the `spare` chunks just under the activation, beside the
    /// pool's budget: what a smaller activation rung hands state first.
    pub fn hold_spare(&mut self, pool: &mut PhysicalPool, spare: u32) -> Result<Commit> {
        let chunk = self.layout.chunk_bytes;
        let from = u64::from(self.total.saturating_sub(spare)) * chunk;
        self.range.hold(pool, from, u64::from(spare) * chunk, true)
    }

    /// Backs `bytes` of the activation's span from `offset` into it, and
    /// keeps them backed: where a carve laid at the span's top starts.
    pub fn hold_activation_at(
        &mut self,
        pool: &mut PhysicalPool,
        offset: u64,
        bytes: u64,
        beside: bool,
    ) -> Result<Commit> {
        let from = u64::from(self.total) * self.layout.chunk_bytes;
        let room = u64::from(self.act) * self.layout.chunk_bytes;
        if offset.saturating_add(bytes) > room {
            return Err(Fault::Ceiling {
                what: "bytes of activation under the state pool's end",
                need: offset.saturating_add(bytes),
                have: room,
            });
        }
        self.range.hold(pool, from + offset, bytes, beside)
    }

    /// Backs the state's chunks of the segment the runtime's activation
    /// span starts in, and keeps them backed: the runtime counts that
    /// segment committed from the load and opens state below the activation
    /// inside it. The span starts `spare` chunks under the activation's own
    /// ([`Chunks::facts`]).
    pub fn hold_boundary(&mut self, pool: &mut PhysicalPool, spare: u32) -> Result<Commit> {
        let chunk = self.layout.chunk_bytes;
        let start = self.total.saturating_sub(spare);
        let segment = self.layout.segment_chunks.max(1);
        let first = start / segment * segment;
        if first >= start {
            return Ok(Commit::Committed);
        }
        let outcome = self.range.hold(
            pool,
            u64::from(first) * chunk,
            u64::from(start - first) * chunk,
            false,
        )?;
        if outcome == Commit::Committed && first == 0 && !self.null_zeroed {
            crate::device::zero_span(
                self.range.span(0, chunk)?,
                usize::try_from(chunk).unwrap_or(0),
            )?;
            self.null_zeroed = true;
        }
        Ok(outcome)
    }

    /// Holds the state's segments to the pool's first `chunks` chunks: what
    /// the fit left the cache rows of a range reserved for more.
    pub fn limit(&mut self, chunks: u32) {
        self.range
            .limit(u64::from(chunks) * self.layout.chunk_bytes);
    }

    /// Whether physical chunk `chunk` is backed: in a committed segment, or
    /// in a span held backed.
    #[must_use]
    pub fn backed(&self, chunk: u32) -> bool {
        self.range
            .span(
                u64::from(chunk) * self.layout.chunk_bytes,
                self.layout.chunk_bytes,
            )
            .is_ok()
    }

    /// Reserves the range anew for `total` chunks of state, before any
    /// segment is backed: the activation's span is backed again where it
    /// then lies.
    pub fn reshape(&mut self, pool: &mut PhysicalPool, total: u32) -> Result<()> {
        if !self.fresh() {
            return Err(Fault::program(
                "store::chunks",
                "the state pool reshaped under committed chunks",
            ));
        }
        let held = self
            .range
            .held(u64::from(self.total) * self.layout.chunk_bytes);
        self.range.release_all(pool);
        let act_bytes = u64::from(self.act) * self.layout.chunk_bytes;
        let mut fresh = Chunks::reserve(pool, self.layout.clone(), total, act_bytes)?;
        if held > 0 {
            match fresh.hold_activation(pool, held, true)? {
                Commit::Committed => {}
                _ => {
                    return Err(Fault::program(
                        "store::chunks",
                        "the activation's span was refused the room it held",
                    ));
                }
            }
        }
        *self = fresh;
        Ok(())
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.range.high_water_bytes()
    }

    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.range.committed_segments()
    }

    /// The segments the state's chunks lie in.
    #[must_use]
    pub fn segments(&self) -> u32 {
        self.segments_for(u64::from(self.total))
    }

    /// Whether no segment is backed yet.
    #[must_use]
    pub fn fresh(&self) -> bool {
        self.range.committed_segments() == 0
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

    fn shape(&self, kind: Kind) -> Result<&SlotShape> {
        self.layout.shape(kind).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!("a {} named in a load that lays none", kind.name()),
            )
        })
    }

    /// From here on a lane names its slots by slot ref, as the runtime's
    /// pool hands them out: what this pool seated for the load's own fires
    /// is let go, and nothing is seated again.
    pub fn take_refs(&mut self) {
        if !self.refs {
            self.refs = true;
            self.seated = Default::default();
        }
    }

    /// Whether a page this pool seated itself lies in one of `chunks`.
    #[must_use]
    pub fn seated_in(&self, chunks: &std::collections::BTreeSet<u32>) -> bool {
        !self.refs
            && Kind::ALL.iter().any(|kind| {
                let Ok(shape) = self.shape(*kind) else {
                    return false;
                };
                self.seated[kind.index()]
                    .refs
                    .iter()
                    .any(|&slot| slot != UNSEATED && chunks.contains(&(slot / shape.per_chunk)))
            })
    }

    /// Keeps the pages this pool seats itself clear of `chunks`, the ones
    /// the runtime named weight slots in. Where a seated page already lies
    /// in one, every page is let go and seated again as it is next named:
    /// what the pages held is lost, so the caller holds no lane's rows.
    pub fn keep_clear(&mut self, chunks: std::collections::BTreeSet<u32>) {
        if self.seated_in(&chunks) {
            self.seated = Default::default();
            self.next = 1;
        }
        self.weights = chunks;
    }

    /// The chunks under `chunks` the runtime named weight slots in.
    #[must_use]
    pub fn weight_chunks_under(&self, chunks: u32) -> u32 {
        u32::try_from(self.weights.range(..chunks).count()).unwrap_or(u32::MAX)
    }

    /// Whether lanes name slot refs.
    #[must_use]
    pub fn by_refs(&self) -> bool {
        self.refs
    }

    /// Seats page `id` of today's ids on a slot, opening the lowest free
    /// chunk for its shape where the one being filled is full. The windowed
    /// id 0 is the null slot. Returns the physical chunk the slot lies in.
    pub fn seat(&mut self, kind: Kind, id: u32) -> Result<u32> {
        if matches!(kind, Kind::Window(_)) && id == 0 {
            return Ok(0);
        }
        let per_chunk = self.shape(kind)?.per_chunk;
        if self.refs {
            // A slot ref: its chunk is one the runtime opened, in a segment
            // it had backed ahead of this fire.
            let chunk = id / per_chunk;
            if chunk == 0 || chunk >= self.total {
                return Err(Fault::Ceiling {
                    what: "chunks of the state pool a slot ref may name",
                    need: u64::from(chunk) + 1,
                    have: u64::from(self.total),
                });
            }
            return Ok(self.table.physical(chunk).unwrap_or(chunk));
        }
        let seating = &mut self.seated[kind.index()];
        if let Some(&slot) = seating.refs.get(id as usize)
            && slot != UNSEATED
        {
            let chunk = slot / per_chunk;
            return Ok(self.table.physical(chunk).unwrap_or(chunk));
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
                // A chunk the runtime laid weights in is not this pool's
                // to open.
                while self.weights.contains(&self.next) {
                    self.next += 1;
                }
                if self.next >= self.total {
                    return Err(Fault::Ceiling {
                        what: "chunks of the state pool clear of the weights' slots",
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
        Ok(self.table.physical(chunk).unwrap_or(chunk))
    }

    /// The slot ref page `id` is seated on.
    pub fn slot_ref(&self, kind: Kind, id: u32) -> Result<u32> {
        if (matches!(kind, Kind::Window(_)) && id == 0) || self.refs {
            return Ok(id);
        }
        self.seated[kind.index()]
            .refs
            .get(id as usize)
            .copied()
            .filter(|&slot| slot != UNSEATED)
            .ok_or_else(|| {
                Fault::program(
                    "store::chunks",
                    format!("{} {id} is read before it is seated", kind.name()),
                )
            })
    }

    /// The physical chunk and the index in it of page `id`'s slot.
    fn place(&self, kind: Kind, id: u32) -> Result<(u32, u32)> {
        let shape = self.shape(kind)?;
        let slot = self.slot_ref(kind, id)?;
        let logical = slot / shape.per_chunk;
        let physical = self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total),
        })?;
        Ok((physical, slot % shape.per_chunk))
    }

    /// Bytes from the pool's base to page `id`'s slot.
    pub fn slot_at(&self, kind: Kind, id: u32) -> Result<u64> {
        let (chunk, index) = self.place(kind, id)?;
        Ok(self
            .shape(kind)?
            .slot_at(self.layout.chunk_bytes, chunk, index))
    }

    /// The address of `len` bytes at `offset` of page `id`'s slot.
    pub fn span(&self, kind: Kind, id: u32, offset: u64, len: u64) -> Result<u64> {
        self.range.span(self.slot_at(kind, id)? + offset, len)
    }

    /// The address of weight slot `slot` (a slot ref) of the row-streamed
    /// table `rows`, where its chunk is backed.
    pub fn row_slot(&self, rows: &crate::store::layout::RowsShape, slot: u32) -> Result<u64> {
        self.weight_slot(rows.per_chunk, rows.slot_bytes, slot)
    }

    /// The address of weight slot `slot` (a slot ref) of a weight shape of
    /// `per_chunk` slots of `slot_bytes` a chunk, where its chunk is backed.
    pub fn weight_slot(&self, per_chunk: u32, slot_bytes: u64, slot: u32) -> Result<u64> {
        let logical = slot / per_chunk;
        let physical = self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total),
        })?;
        self.range.span(
            u64::from(physical) * self.layout.chunk_bytes
                + u64::from(slot % per_chunk) * slot_bytes,
            slot_bytes,
        )
    }

    /// The id a paged kernel reads page `id`'s slot by.
    pub fn page_id(&self, kind: Kind, id: u32) -> Result<i32> {
        let (chunk, index) = self.place(kind, id)?;
        let q = self
            .shape(kind)?
            .page_id(self.layout.chunk_bytes, chunk, index);
        i32::try_from(q).map_err(|_| Fault::Ceiling {
            what: "page ids of the state pool",
            need: q,
            have: i32::MAX as u64,
        })
    }

    /// Rewrites today's page ids in `ids` as the ids the kernels read; an id
    /// under zero (a row that writes nothing) stays.
    pub fn page_ids(&self, kind: Kind, ids: &mut [i32]) -> Result<()> {
        for id in ids {
            if *id >= 0 {
                *id = self.page_id(kind, *id as u32)?;
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

impl Chunks {
    /// The pool as the runtime counts it: every chunk of the range, the
    /// null chunk reserved, each shape free to take any chunk, and the
    /// activation's chunks under the range's end. `spare` chunks under the
    /// activation's hold nothing the device can back (what the load's own
    /// allocations took after the range was reserved): they are stated with
    /// the activation's, so state stays below them.
    #[must_use]
    pub fn facts(
        &self,
        max_tokens: u32,
        spare: u32,
        flat: bool,
        rungs: &[(u32, u32)],
    ) -> engine::chunk::ChunkFacts {
        use engine::chunk::{
            ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, SlotKind, SlotShape,
        };
        let layout = &self.layout;
        let shape = |shape: &crate::store::layout::SlotShape, kind: SlotKind| SlotShape {
            id: shape.id,
            kind,
            slot_bytes: shape.slot_bytes,
            slots_per_chunk: shape.per_chunk,
            sets_per_working_set: kind.sets_per_working_set(),
            rows: shape.rows.iter().map(|&(row, _)| row).collect(),
            pinned: None,
            role: engine::chunk::SlotRole::State,
        };
        let mut shapes = Vec::new();
        if let Some(list) = &layout.list {
            shapes.push(shape(
                list,
                SlotKind::List {
                    tokens: list.tokens,
                    rate: 1,
                    window: None,
                },
            ));
        }
        for window in &layout.windows {
            shapes.push(shape(
                window,
                SlotKind::List {
                    tokens: window.tokens,
                    rate: 1,
                    window: window.window,
                },
            ));
        }
        if let Some(rs) = &layout.rs {
            let mut stated = shape(
                &rs.slot,
                SlotKind::List {
                    tokens: rs.slot.tokens,
                    rate: 1,
                    window: None,
                },
            );
            stated.role = engine::chunk::SlotRole::RsBuffer;
            shapes.push(stated);
        }
        if let Some(fixed) = &layout.fixed {
            shapes.push(shape(
                &fixed.slot,
                SlotKind::Fixed {
                    pieces: fixed.pieces,
                },
            ));
        }
        let total = self.total.saturating_add(self.act);
        let max = u32::try_from(u64::from(total).div_ceil(u64::from(layout.segment_chunks.max(1))))
            .unwrap_or(u32::MAX);
        ChunkFacts {
            chunk_bytes: layout.chunk_bytes,
            total_chunks: total,
            segment_chunks: layout.segment_chunks,
            segments: SegmentFacts {
                committed: if flat { max } else { self.committed_segments() },
                max,
                growth: if flat {
                    Growth::Fixed
                } else {
                    Growth::TwoEnded
                },
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            }],
            shapes,
            act_chunks: rungs
                .iter()
                .filter(|&&(tokens, chunks)| tokens < max_tokens && chunks < self.act)
                .map(|&(tokens, chunks)| (tokens, chunks.saturating_add(spare)))
                .chain(std::iter::once((
                    max_tokens,
                    self.act.saturating_add(spare),
                )))
                .collect(),
            host_chunks: 0,
            disk_chunks: 0,
            disk_persistent: false,
            disk_kept: false,
            weights: layout
                .rows
                .iter()
                .map(|rows| engine::chunk::WeightShape {
                    id: rows.id,
                    kind: engine::chunk::WeightKind::Rows,
                    slot_bytes: rows.slot_bytes,
                    slots_per_chunk: rows.per_chunk,
                    count: rows.blocks,
                    floor_slots: rows.floor.min(rows.blocks),
                    name: format!("param {}", rows.param),
                    seated: None,
                })
                .chain(
                    layout
                        .experts
                        .iter()
                        .map(|experts| engine::chunk::WeightShape {
                            id: experts.id,
                            kind: engine::chunk::WeightKind::Expert,
                            slot_bytes: experts.slot_bytes,
                            slots_per_chunk: experts.per_chunk,
                            count: experts.count(),
                            floor_slots: experts.floor.min(experts.count()),
                            name: format!(
                                "experts of param {}",
                                experts
                                    .banks
                                    .first()
                                    .and_then(|l| l.first())
                                    .map_or(0, |p| p.param)
                            ),
                            seated: None,
                        }),
                )
                .collect(),
        }
    }
}

//! The chunk pool on the host (SPEC 11: Cerebras keeps state on the host).
//!
//! Every working set's state lives in one host arena of `total_chunks`
//! chunks, the shapes' slots drawing from it as the runtime hands them out
//! (a shared pool: no shape is pinned):
//!
//! - the **kv list**: a slot is one kv page (`T` = the kv page size) of
//!   every full-attention kv plane and every compressor plane, plane after
//!   plane (layer-major), `T` rows of each;
//! - the **window list**: the same over the windowed kv planes, one list
//!   for the widest window (CONTRACT-BC 21 lets the engine keep one; its
//!   `T_w` is the kv page);
//! - the **fixed record**: one row of every state slab, slab after slab,
//!   cut into `pieces` equal slots where it is larger than a chunk (the
//!   cut is a word range: the record is reassembled on the host before a
//!   fire reads it, so no op's atom constrains it here);
//! - the **rs list**: `T_rs` (= the kv page size) buffered rows of every
//!   recurrent-verb plane.
//!
//! The kernels never address the arena: a fire reads and writes typed
//! planes, so `super::stage` gathers the slots a fire names into its planes
//! under fire-local ids and scatters what it wrote back. Logical chunks
//! (the ids in slot refs) sit on physical ones through the engine's
//! [`ChunkTable`]; segments are host allocations that commit and release
//! as the runtime asks, an evacuation copying chunks out of a segment
//! first. Every fire runs to completion inside `submit`, so each of these
//! verbs is ordered after every fire before it (the per-fire epoch of
//! CONTRACT-BC 17 holds trivially).

use engine::chunk::{
    ChunkFacts, ChunkTable, Evacuation, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId,
    SlotKind, SlotRole, SlotShape,
};

use crate::error::{Fault, Result};

/// The shapes' ids, each stated only where the model holds it.
pub const LIST: ShapeId = ShapeId(0);
pub const WINDOW: ShapeId = ShapeId(1);
pub const FIXED: ShapeId = ShapeId(2);
pub const RS: ShapeId = ShapeId(3);

/// What a chunk takes at most, unless one list slot is larger.
pub const TARGET_CHUNK_BYTES: u64 = 16 << 20;
/// What a segment (one host allocation) takes, at least one chunk.
const SEGMENT_BYTES: u64 = 256 << 20;
/// The kv list spans at least this many chunks where its slots allow, so a
/// small pool is not one chunk of mostly nothing.
const LIST_CHUNKS_AT_LEAST: u64 = 16;
/// Every element is one 32-bit word on the host (`crate::device::Buffer`).
const WORD: u64 = 4;

/// One plane's part of a slot: `width` words a row of plane `plane` of
/// pool row `row` (a cache row, a compressor plane or an rs plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section {
    pub row: u32,
    pub plane: u8,
    pub width: u32,
}

/// One slot shape as the arena lays it out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub id: ShapeId,
    pub kind: SlotKind,
    pub role: SlotRole,
    pub sections: Vec<Section>,
    /// Rows a list slot holds of each section; 1 for a record.
    pub tokens: u32,
    pub slot_words: u64,
    pub per_chunk: u32,
    /// A record's words (its pieces hold at least these).
    pub record_words: u64,
    /// The trace's cache rows the slot lays out, in order.
    pub rows: Vec<u32>,
}

impl Shape {
    /// Each section with its first word in the slot (a list's) or the
    /// record (a fixed shape's) and the words it takes there.
    pub fn sections(&self) -> impl Iterator<Item = (Section, u64, u64)> + '_ {
        let tokens = u64::from(self.tokens);
        self.sections.iter().scan(0u64, move |at, &section| {
            let words = tokens * u64::from(section.width);
            let first = *at;
            *at += words;
            Some((section, first, words))
        })
    }

    #[must_use]
    pub fn pieces(&self) -> u32 {
        match self.kind {
            SlotKind::Fixed { pieces } => pieces,
            SlotKind::List { .. } => 1,
        }
    }
}

/// A list shape before the pool is sized: its rows a slot and sections.
#[derive(Debug, Clone, Default)]
pub struct Draft {
    pub tokens: u32,
    pub sections: Vec<Section>,
    pub rows: Vec<u32>,
    /// Slots the pool holds of it, at the least.
    pub slots: u64,
}

impl Draft {
    fn words(&self) -> u64 {
        u64::from(self.tokens)
            * self
                .sections
                .iter()
                .map(|s| u64::from(s.width))
                .sum::<u64>()
    }
}

/// The pool's shapes and size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub chunk_words: u64,
    pub total_chunks: u32,
    pub segment_chunks: u32,
    pub shapes: Vec<Shape>,
}

/// What [`Layout::plan`] lays out: each shape's draft, where the model
/// holds it, and the window list's window.
#[derive(Debug, Clone, Default)]
pub struct Drafts {
    pub list: Option<Draft>,
    pub window: Option<(Draft, u32)>,
    /// The record's sections (`tokens` 1) and the records the pool holds.
    pub fixed: Option<Draft>,
    pub rs: Option<Draft>,
}

impl Layout {
    /// Sizes the pool: a chunk is a whole number of the largest list slot,
    /// up to `target` bytes ([`TARGET_CHUNK_BYTES`] at load), and no more
    /// than the kv list's slots spread over [`LIST_CHUNKS_AT_LEAST`]
    /// chunks; a record larger than a chunk is cut into equal pieces; the
    /// pool holds every draft's slots past the null chunk 0.
    pub fn plan(drafts: &Drafts, target: u64) -> Result<Layout> {
        let lists = [
            drafts.list.as_ref(),
            drafts.window.as_ref().map(|(d, _)| d),
            drafts.rs.as_ref(),
        ];
        let unit = lists.iter().flatten().map(|d| d.words()).max().unwrap_or(0);
        let target = (target / WORD).max(1);
        let record_words = drafts.fixed.as_ref().map_or(0, Draft::words);
        let chunk_words = if unit > 0 {
            let spread = drafts
                .list
                .as_ref()
                .map_or(u64::MAX, |list| (list.slots / LIST_CHUNKS_AT_LEAST).max(1));
            unit * (target / unit).clamp(1, spread)
        } else {
            record_words.clamp(1, target)
        };
        let mut shapes = Vec::new();
        let mut chunks = 1u64;
        let mut list = |id, kind, role, draft: &Draft| -> Result<()> {
            let words = draft.words().max(1);
            let per_chunk = narrow(chunk_words / words, "slots a chunk")?;
            chunks += draft.slots.div_ceil(u64::from(per_chunk));
            shapes.push(Shape {
                id,
                kind,
                role,
                sections: draft.sections.clone(),
                tokens: draft.tokens,
                slot_words: words,
                per_chunk,
                record_words: 0,
                rows: draft.rows.clone(),
            });
            Ok(())
        };
        if let Some(draft) = &drafts.list {
            let kind = SlotKind::List {
                tokens: draft.tokens,
                rate: 1,
                window: None,
            };
            list(LIST, kind, SlotRole::State, draft)?;
        }
        if let Some((draft, window)) = &drafts.window {
            let kind = SlotKind::List {
                tokens: draft.tokens,
                rate: 1,
                window: Some(*window),
            };
            list(WINDOW, kind, SlotRole::State, draft)?;
        }
        if let Some(draft) = &drafts.rs {
            let kind = SlotKind::List {
                tokens: draft.tokens,
                rate: 1,
                window: None,
            };
            list(RS, kind, SlotRole::RsBuffer, draft)?;
        }
        if let Some(draft) = &drafts.fixed {
            let pieces = record_words.div_ceil(chunk_words).max(1);
            let piece_words = record_words.div_ceil(pieces).max(1);
            let per_chunk = narrow(chunk_words / piece_words, "pieces a chunk")?;
            chunks += (draft.slots * pieces).div_ceil(u64::from(per_chunk));
            shapes.push(Shape {
                id: FIXED,
                kind: SlotKind::Fixed {
                    pieces: narrow(pieces, "pieces of a record")?,
                },
                role: SlotRole::State,
                sections: draft.sections.clone(),
                tokens: 1,
                slot_words: piece_words,
                per_chunk,
                record_words,
                rows: draft.rows.clone(),
            });
        }
        shapes.sort_by_key(|shape| shape.id);
        let total_chunks = narrow(chunks, "chunks of the pool")?;
        let segment_chunks = narrow(
            (SEGMENT_BYTES / WORD / chunk_words).clamp(1, u64::from(total_chunks)),
            "chunks a segment",
        )?;
        Ok(Layout {
            chunk_words,
            total_chunks,
            segment_chunks,
            shapes,
        })
    }

    #[must_use]
    pub fn shape(&self, id: ShapeId) -> Option<&Shape> {
        self.shapes.iter().find(|shape| shape.id == id)
    }

    #[must_use]
    pub fn max_segments(&self) -> u32 {
        self.total_chunks.div_ceil(self.segment_chunks.max(1))
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::from(self.total_chunks) * self.chunk_words * WORD
    }
}

/// The arena: the layout's chunks in segments the host allocates, the
/// logical -> physical chunk table, and what is committed. A dry pool
/// allocates nothing and moves nothing.
#[derive(Debug)]
pub struct HostPool {
    layout: Layout,
    segments: Vec<Option<Box<[u32]>>>,
    committed: u32,
    table: ChunkTable,
    dry: bool,
    /// The host chunks and the kv slot file below the pool.
    pub(super) tiers: super::tier::Tiers,
}

impl HostPool {
    /// The pool of `layout`, every segment committed (as today's planes
    /// stand from the load); a dry one holds no words.
    #[must_use]
    pub fn new(layout: Layout, dry: bool) -> HostPool {
        let max = layout.max_segments();
        let mut pool = HostPool {
            segments: (0..max).map(|_| None).collect(),
            committed: 0,
            table: ChunkTable::identity(layout.total_chunks),
            layout,
            dry,
            tiers: super::tier::Tiers::default(),
        };
        pool.commit(max)
            .expect("a pool commits the segments it lays out");
        pool
    }

    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[must_use]
    pub fn is_dry(&self) -> bool {
        self.dry
    }

    #[must_use]
    pub fn committed(&self) -> u32 {
        self.committed
    }

    /// Bytes the committed segments take.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        (0..self.committed)
            .map(|seg| u64::try_from(self.segment_len(seg)).unwrap_or(0) * WORD)
            .sum()
    }

    /// The pool as the runtime counts it: one shared pool, its shapes
    /// unpinned, chunk 0 the null chunk, state growing from the bottom and
    /// activation (the fire's staging planes) apart.
    #[must_use]
    pub fn facts(&self) -> ChunkFacts {
        let layout = &self.layout;
        ChunkFacts {
            chunk_bytes: layout.chunk_words * WORD,
            total_chunks: layout.total_chunks,
            segment_chunks: layout.segment_chunks,
            segments: SegmentFacts {
                committed: self.committed,
                max: layout.max_segments(),
                growth: Growth::Bottom,
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            }],
            shapes: layout
                .shapes
                .iter()
                .map(|shape| SlotShape {
                    id: shape.id,
                    kind: shape.kind,
                    slot_bytes: shape.slot_words * WORD,
                    slots_per_chunk: shape.per_chunk,
                    sets_per_working_set: shape.kind.sets_per_working_set(),
                    rows: shape.rows.clone(),
                    pinned: None,
                    role: shape.role,
                })
                .collect(),
            host_chunks: self.host_chunks(),
            disk_chunks: self.disk_chunks(),
            disk_persistent: self.disk_persistent(),
            disk_kept: self.disk_kept(),
            ..ChunkFacts::default()
        }
    }

    pub fn shape(&self, id: ShapeId) -> Result<&Shape> {
        self.layout.shape(id).ok_or_else(|| Fault::Unbound {
            what: format!("slot shape {id}, which this pool does not hold"),
        })
    }

    fn segment_len(&self, seg: u32) -> usize {
        let first = u64::from(seg) * u64::from(self.layout.segment_chunks);
        let chunks = (u64::from(self.layout.total_chunks) - first)
            .min(u64::from(self.layout.segment_chunks));
        usize::try_from(chunks * self.layout.chunk_words).unwrap_or(usize::MAX)
    }

    /// Where slot ref `slot` of shape `id` lies: its segment and the
    /// words it takes there. Refused in the null chunk, past the pool, or
    /// in a segment not committed.
    pub fn span(&self, id: ShapeId, slot: u32) -> Result<(usize, std::ops::Range<usize>)> {
        let shape = self.shape(id)?;
        let chunk = slot / shape.per_chunk.max(1);
        let index = slot % shape.per_chunk.max(1);
        if chunk == 0 || chunk >= self.layout.total_chunks {
            return Err(Fault::Unbound {
                what: format!(
                    "slot {slot} of shape {id}, in chunk {chunk} of the {} this pool holds \
                     (chunk 0 is the null chunk)",
                    self.layout.total_chunks
                ),
            });
        }
        let physical = self.table.physical(chunk).unwrap_or(u32::MAX);
        let seg = physical / self.layout.segment_chunks.max(1);
        if seg >= self.committed {
            return Err(Fault::Unbound {
                what: format!(
                    "slot {slot} of shape {id}, on physical chunk {physical} in segment {seg}, \
                     past the {} committed",
                    self.committed
                ),
            });
        }
        let within = u64::from(physical % self.layout.segment_chunks.max(1));
        let first = within * self.layout.chunk_words + u64::from(index) * shape.slot_words;
        let first = usize::try_from(first).unwrap_or(usize::MAX);
        let words = usize::try_from(shape.slot_words).unwrap_or(usize::MAX);
        Ok((seg as usize, first..first + words))
    }

    /// Slot `slot`'s words; `None` on a dry pool.
    pub fn slot(&self, id: ShapeId, slot: u32) -> Result<Option<&[u32]>> {
        let (seg, range) = self.span(id, slot)?;
        Ok(self.segments[seg].as_deref().map(|words| &words[range]))
    }

    pub fn slot_mut(&mut self, id: ShapeId, slot: u32) -> Result<Option<&mut [u32]>> {
        let (seg, range) = self.span(id, slot)?;
        Ok(self.segments[seg]
            .as_deref_mut()
            .map(|words| &mut words[range]))
    }

    /// Copies each whole slot `(src, dst)` of shape `id`, in order.
    pub fn copy_slots(&mut self, id: ShapeId, pairs: &[(u32, u32)]) -> Result<()> {
        for &(src, dst) in pairs {
            let Some(words) = self.slot(id, src)?.map(<[u32]>::to_vec) else {
                self.span(id, dst)?;
                continue;
            };
            if let Some(to) = self.slot_mut(id, dst)? {
                to.copy_from_slice(&words);
            }
        }
        Ok(())
    }

    /// Copies `tokens` rows of list shape `id` from row `src_token` of slot
    /// `src` to row `dst_token` of slot `dst`, in every section.
    pub fn copy_rows(
        &mut self,
        id: ShapeId,
        (src, src_token): (u32, u32),
        (dst, dst_token): (u32, u32),
        tokens: u32,
    ) -> Result<()> {
        let shape = self.shape(id)?;
        let per = shape.tokens;
        if src_token.saturating_add(tokens) > per || dst_token.saturating_add(tokens) > per {
            return Err(Fault::Ceiling {
                what: "rows of a slot a copy reaches",
                need: u64::from(src_token.max(dst_token)) + u64::from(tokens),
                have: u64::from(per),
            });
        }
        let runs: Vec<(usize, usize, usize)> = shape
            .sections()
            .map(|(section, first, _)| {
                let w = u64::from(section.width);
                let at = |token: u32| usize::try_from(first + u64::from(token) * w).unwrap_or(0);
                (
                    at(src_token),
                    at(dst_token),
                    usize::try_from(u64::from(tokens) * w).unwrap_or(0),
                )
            })
            .collect();
        let Some(from) = self.slot(id, src)?.map(<[u32]>::to_vec) else {
            self.span(id, dst)?;
            return Ok(());
        };
        if let Some(to) = self.slot_mut(id, dst)? {
            for (s, d, n) in runs {
                to[d..d + n].copy_from_slice(&from[s..s + n]);
            }
        }
        Ok(())
    }

    /// Zeros each piece of a record.
    pub fn clear(&mut self, id: ShapeId, slots: &[u32]) -> Result<()> {
        for &slot in slots {
            if let Some(words) = self.slot_mut(id, slot)? {
                words.fill(0);
            }
        }
        Ok(())
    }

    /// Backs segments `0..segments`; what is committed stays.
    pub fn commit(&mut self, segments: u32) -> Result<()> {
        let max = self.layout.max_segments();
        if segments > max {
            return Err(Fault::Ceiling {
                what: "segments of the pool",
                need: u64::from(segments),
                have: u64::from(max),
            });
        }
        for seg in 0..segments {
            if !self.dry && self.segments[seg as usize].is_none() {
                self.segments[seg as usize] =
                    Some(vec![0u32; self.segment_len(seg)].into_boxed_slice());
            }
        }
        self.committed = self.committed.max(segments);
        Ok(())
    }

    /// Gives back every segment at and past `segments`: the runtime has
    /// evacuated what they held.
    pub fn release(&mut self, segments: u32) {
        for seg in self.segments.iter_mut().skip(segments as usize) {
            *seg = None;
        }
        self.committed = self.committed.min(segments);
    }

    /// Copies each moved chunk onto its new physical chunk and updates the
    /// table, in order; all or nothing.
    pub fn evacuate(&mut self, evacuation: &Evacuation) -> Result<()> {
        let invalid = |why: String| Fault::Unbound {
            what: format!("an evacuation: {why}"),
        };
        let mut checked = self.table.clone();
        checked
            .apply(evacuation)
            .map_err(|e| invalid(e.to_string()))?;
        let seg_chunks = self.layout.segment_chunks.max(1);
        for &(_, physical) in &evacuation.moves {
            if physical / seg_chunks >= self.committed {
                return Err(invalid(format!(
                    "physical chunk {physical} lies past the {} committed segments",
                    self.committed
                )));
            }
        }
        let chunk = usize::try_from(self.layout.chunk_words).unwrap_or(usize::MAX);
        for &(logical, physical) in &evacuation.moves {
            let from = self.table.physical(logical).unwrap_or(u32::MAX);
            if from / seg_chunks >= self.committed {
                return Err(invalid(format!(
                    "chunk {logical} sits on physical chunk {from}, past the committed segments"
                )));
            }
            let at = |p: u32| ((p / seg_chunks) as usize, (p % seg_chunks) as usize * chunk);
            let ((fs, fo), (ts, to)) = (at(from), at(physical));
            if let Some(words) = self.segments[fs]
                .as_deref()
                .map(|w| w[fo..fo + chunk].to_vec())
                && let Some(dst) = self.segments[ts].as_deref_mut()
            {
                dst[to..to + chunk].copy_from_slice(&words);
            }
            self.table
                .apply(&Evacuation {
                    moves: vec![(logical, physical)],
                })
                .map_err(|e| invalid(e.to_string()))?;
        }
        Ok(())
    }
}

fn narrow(n: u64, what: &'static str) -> Result<u32> {
    u32::try_from(n).map_err(|_| Fault::Ceiling {
        what,
        need: n,
        have: u64::from(u32::MAX),
    })
}

#[cfg(test)]
mod tests;

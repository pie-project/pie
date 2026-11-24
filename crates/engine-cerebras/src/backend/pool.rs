//! The pool as the load states it: one shared chunk space on the host, every
//! shape's slots drawn from it. A kv page holds its rows' tokens (the wafer
//! program rebuilds a lane's cache from them), a record holds the recurrent
//! state's bytes and each windowed layer's ring, and a window-buffer slot
//! holds its unfolded rows' tokens. Nothing here is device memory the
//! runtime could evict or stream: no host or disk tier, no activation span,
//! one segment whole from the load.

use engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId, SlotKind, SlotRole,
    SlotShape, WindowedLayer,
};

/// The kv list: a lane's tokens, `page_size` a slot.
pub(super) const KV: ShapeId = ShapeId(0);
/// The record: the recurrent state's bytes, then each windowed layer's ring.
pub(super) const RECORD: ShapeId = ShapeId(1);
/// The window buffer: `page_size` unfolded rows' tokens a slot.
pub(super) const BUFFER: ShapeId = ShapeId(2);

/// The bytes a token takes in a page, a ring or a buffer slot.
pub(super) const TOKEN_BYTES: u64 = 4;

/// What the pool holds.
pub(super) struct Pool {
    pub page_size: u32,
    pub kv_pages: u64,
    /// Records the pool holds.
    pub records: u32,
    /// The recurrent state's bytes of one record (0: none).
    pub state_bytes: u64,
    /// The windowed layers, ascending, each with its window.
    pub windowed: Vec<(u32, u32)>,
    /// Window-buffer slots the pool holds (0 where no layer is windowed).
    pub buffer_pages: u64,
}

impl Pool {
    /// Each windowed layer's ring as the record states it: `window` tokens.
    #[must_use]
    pub fn rings(&self) -> Vec<WindowedLayer> {
        self.windowed
            .iter()
            .map(|&(layer, window)| WindowedLayer {
                layer,
                window,
                bytes: u64::from(window) * TOKEN_BYTES,
            })
            .collect()
    }

    /// The pool as chunks: chunk 0 null, then a chunk a slot of the widest
    /// shape and as many of the narrower ones as fill it.
    #[must_use]
    pub fn facts(&self) -> ChunkFacts {
        let page_bytes = u64::from(self.page_size) * TOKEN_BYTES;
        let rings = self.rings();
        let record_bytes = self.state_bytes + rings.iter().map(|ring| ring.bytes).sum::<u64>();
        let has_record = record_bytes > 0;
        let chunk_bytes = page_bytes.max(record_bytes);
        let per_chunk = |bytes: u64| u32::try_from(chunk_bytes / bytes).unwrap_or(u32::MAX);
        let chunks = |slots: u64, bytes: u64| -> u32 {
            u32::try_from(slots.div_ceil(u64::from(per_chunk(bytes)))).unwrap_or(u32::MAX)
        };
        let list = |tokens: u32| SlotKind::List {
            tokens,
            rate: 1,
            window: None,
        };
        let shape = |id: ShapeId, kind: SlotKind, slot_bytes: u64, role: SlotRole| SlotShape {
            id,
            kind,
            slot_bytes,
            slots_per_chunk: per_chunk(slot_bytes),
            sets_per_working_set: kind.sets_per_working_set(),
            rows: Vec::new(),
            pinned: None,
            role,
        };
        let mut shapes = vec![shape(KV, list(self.page_size), page_bytes, SlotRole::State)];
        let mut total = 1u32.saturating_add(chunks(self.kv_pages, page_bytes));
        if has_record {
            shapes.push(shape(
                RECORD,
                SlotKind::Fixed { pieces: 1 },
                record_bytes,
                SlotRole::State,
            ));
            total = total.saturating_add(chunks(u64::from(self.records), record_bytes));
        }
        if !rings.is_empty() {
            shapes.push(shape(
                BUFFER,
                list(self.page_size),
                page_bytes,
                SlotRole::WindowBuffer,
            ));
            total = total.saturating_add(chunks(self.buffer_pages, page_bytes));
        }
        ChunkFacts {
            chunk_bytes,
            total_chunks: total,
            segment_chunks: total,
            segments: SegmentFacts {
                committed: 1,
                max: 1,
                growth: Growth::Fixed,
            },
            reserved: vec![ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            }],
            shapes,
            record_bytes,
            windowed: rings,
            ..ChunkFacts::default()
        }
    }
}

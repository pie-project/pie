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
    /// A unit program's own ids, where its pages and state slots are the
    /// wafer's: every slot ref the pool states must name one of them.
    pub wafer: Option<WaferIds>,
}

/// What a unit program addresses on the wafer: its K / V columns' pages (the
/// scratch page not among them) and its recurrences' state slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WaferIds {
    pub pages: u32,
    pub state_slots: u32,
}

/// How a unit program's kv list slot refs name its wafer pages: the pool's
/// chunks after the null one, `per_chunk` pages a chunk, page `ref -
/// per_chunk`. The pool states no more chunks than leave every ref a page
/// ([`Pool::facts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WaferPages {
    pub per_chunk: u32,
    pub pages: u32,
}

impl WaferPages {
    /// The wafer page kv list slot ref `slot` names, refused by name past
    /// the program's pages.
    pub fn page(&self, slot: u32) -> Result<u32, String> {
        slot.checked_sub(self.per_chunk)
            .filter(|&page| page < self.pages)
            .ok_or_else(|| {
                format!(
                    "kv slot ref {slot} names no page of the unit program's {} (refs {}..{})",
                    self.pages,
                    self.per_chunk,
                    u64::from(self.per_chunk) + u64::from(self.pages)
                )
            })
    }
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

    /// How a unit program's kv refs name its pages; `None` where its pages
    /// are not the wafer's.
    #[must_use]
    pub fn wafer_pages(&self) -> Option<WaferPages> {
        let ids = self.wafer?;
        let chunk_bytes = self.facts().chunk_bytes;
        let page_bytes = u64::from(self.page_size) * TOKEN_BYTES;
        Some(WaferPages {
            per_chunk: u32::try_from(chunk_bytes / page_bytes.max(1)).unwrap_or(u32::MAX),
            pages: ids.pages,
        })
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
        // A unit program's ids: the shapes share every chunk, so a kv ref
        // or a record may lie in any of them. No more chunks than leave each
        // kv ref a wafer page and each record (the seats the runtime numbers
        // by them) a state slot.
        if let Some(ids) = self.wafer {
            let mut most = u64::from(ids.pages) / u64::from(per_chunk(page_bytes).max(1));
            if has_record {
                most = most
                    .min(u64::from(ids.state_slots) / u64::from(per_chunk(record_bytes).max(1)));
            }
            total = total.min(1u32.saturating_add(u32::try_from(most).unwrap_or(u32::MAX)));
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

#[cfg(test)]
mod tests {
    use super::{Pool, WaferIds};

    fn unit_pool(state_bytes: u64, wafer: WaferIds) -> Pool {
        Pool {
            page_size: 16,
            kv_pages: 1000,
            records: 40,
            state_bytes,
            windowed: Vec::new(),
            buffer_pages: 0,
            wafer: Some(wafer),
        }
    }

    /// A unit program's pages and state slots are the wafer's: every kv ref
    /// the shared pool can hand out names one of its pages, and no more
    /// records (the runtime's seats) than its state slots, whichever shape
    /// takes which chunk.
    #[test]
    fn every_ref_a_unit_pool_states_is_a_wafer_id() {
        let wafer = WaferIds {
            pages: 30,
            state_slots: 4,
        };
        for state_bytes in [0, 64, 4096] {
            let pool = unit_pool(state_bytes, wafer);
            let facts = pool.facts();
            facts.validate().expect("a unit pool's facts hang together");
            let pages = pool.wafer_pages().expect("a unit pool's pages");
            for shape in &facts.shapes {
                let refs = shape.slots_per_chunk..facts.total_chunks * shape.slots_per_chunk;
                if shape.id == super::KV {
                    for slot in refs {
                        let page = pages.page(slot).expect("a kv ref names a wafer page");
                        assert!(page < wafer.pages);
                    }
                } else if shape.id == super::RECORD {
                    assert!(
                        refs.len() as u32 <= wafer.state_slots,
                        "{state_bytes}: {refs:?}"
                    );
                }
            }
        }
        let pages = unit_pool(0, wafer).wafer_pages().unwrap();
        let refused = pages.page(0).expect_err("the null chunk names no page");
        assert!(refused.contains("names no page"), "{refused}");
    }
}

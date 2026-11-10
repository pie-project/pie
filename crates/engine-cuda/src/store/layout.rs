//! How the cache rows sit in the state pool's chunks: the compiler's chunk
//! recipe ([`poem_compiler::chunk::plan`]) read for this engine's paging.
//!
//! The pool is `total` chunks of `chunk_bytes`, chunk `p` at
//! `base + p * chunk_bytes`. A chunk holds slots of one shape. A **list**
//! slot is one kv page of every full-retention row, layer after layer, and a
//! **ring** slot one page of every windowed row; a row's key plane and then
//! its value plane run where the recipe puts the row.
//!
//! A paged kernel reads page `q` of a plane at `plane base + q * grain`
//! ([`kernels_cuda::KvPool::page_stride`]). `grain` is the largest step both
//! a chunk and the shape's slot are whole numbers of, so every slot of every
//! chunk has a `q`: slot `i` of physical chunk `p` is
//! `(p * chunk_bytes + i * slot_bytes) / grain`. The recipe makes a chunk a
//! whole number of list slots, so a list slot's `q` is its slot ref.

use engine::chunk::ShapeId;
use poem_compiler::chunk::{ChunkRecipe, RowId, SlotLayout};

use crate::error::{Fault, Result};

/// The list shape's id in the facts this engine states.
pub const LIST: ShapeId = ShapeId(0);
/// The ring shape's.
pub const RING: ShapeId = ShapeId(1);
/// The fixed shape's.
pub const FIXED: ShapeId = ShapeId(2);

/// One per-token shape: the pages of the rows that share a page table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenShape {
    pub id: ShapeId,
    /// Tokens a slot holds: the page size.
    pub tokens: u32,
    pub slot_bytes: u64,
    pub per_chunk: u32,
    /// The step page ids count in.
    pub grain: u64,
    /// The window its rows are read through; `None` for the list.
    pub window: Option<u32>,
    /// Each cache row laid in the slot, and where its run starts.
    pub rows: Vec<(u32, u64)>,
}

impl TokenShape {
    fn of(layout: &SlotLayout, chunk_bytes: u64, id: ShapeId) -> Result<TokenShape> {
        let (tokens, bytes, window, runs) = match layout {
            SlotLayout::List {
                tokens,
                bytes,
                rows,
            } => (*tokens, *bytes, None, rows),
            SlotLayout::Ring {
                tokens,
                bytes,
                cap,
                rows,
                ..
            } => (*tokens, *bytes, Some(cap.window), rows),
            SlotLayout::Fixed { .. } => {
                return Err(Fault::program(
                    "store::layout",
                    "a fixed slot read as a token slot",
                ));
            }
        };
        let per_chunk = u32::try_from(chunk_bytes / bytes.max(1)).unwrap_or(u32::MAX);
        if bytes == 0 || per_chunk == 0 {
            return Err(Fault::program(
                "store::layout",
                format!("a token slot of {bytes} bytes fits no chunk of {chunk_bytes}"),
            ));
        }
        Ok(TokenShape {
            id,
            tokens,
            slot_bytes: bytes,
            per_chunk,
            grain: gcd(chunk_bytes, bytes),
            window,
            rows: runs
                .iter()
                .filter_map(|run| match run.row {
                    RowId::Cache(row) => Some((row, run.offset)),
                    RowId::Rs(_) => None,
                })
                .collect(),
        })
    }

    /// Where cache row `row`'s run starts in a slot.
    #[must_use]
    pub fn offset_of(&self, row: u32) -> Option<u64> {
        self.rows
            .iter()
            .find(|(held, _)| *held == row)
            .map(|&(_, offset)| offset)
    }

    /// Bytes from the pool's base to slot `index` of physical chunk `chunk`.
    #[must_use]
    pub fn slot_at(&self, chunk_bytes: u64, chunk: u32, index: u32) -> u64 {
        u64::from(chunk) * chunk_bytes + u64::from(index) * self.slot_bytes
    }

    /// The page id a kernel reads that slot by.
    #[must_use]
    pub fn page_id(&self, chunk_bytes: u64, chunk: u32, index: u32) -> u64 {
        self.slot_at(chunk_bytes, chunk, index) / self.grain
    }

    #[must_use]
    pub fn slot_ref(&self, chunk: u32, index: u32) -> u32 {
        chunk.saturating_mul(self.per_chunk).saturating_add(index)
    }
}

/// The token shapes of a load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub chunk_bytes: u64,
    /// Chunks a segment, by the recipe's device profile.
    pub segment_chunks: u32,
    pub list: Option<TokenShape>,
    pub ring: Option<TokenShape>,
}

impl Layout {
    /// The recipe's token shapes. This engine pages every windowed row by one
    /// table, so a recipe with rings of two windows is refused.
    pub fn of(recipe: &ChunkRecipe) -> Result<Layout> {
        let chunk_bytes = recipe.chunk_bytes;
        let list = recipe
            .list()
            .map(|layout| TokenShape::of(layout, chunk_bytes, LIST))
            .transpose()?;
        let mut rings = recipe.rings();
        let ring = rings
            .next()
            .map(|layout| TokenShape::of(layout, chunk_bytes, RING))
            .transpose()?;
        if rings.next().is_some() {
            return Err(Fault::Unbound {
                what: "kv rows read through windows of two widths, which this engine pages \
                       by one windowed table"
                    .to_string(),
            });
        }
        let (_, segment_chunks) = recipe.profile.segment(chunk_bytes);
        Ok(Layout {
            chunk_bytes,
            segment_chunks: u32::try_from(segment_chunks).unwrap_or(u32::MAX).max(1),
            list,
            ring,
        })
    }

    #[must_use]
    pub fn shape(&self, ring: bool) -> Option<&TokenShape> {
        if ring {
            self.ring.as_ref()
        } else {
            self.list.as_ref()
        }
    }

    /// The chunks `slots` slots of a shape take.
    #[must_use]
    pub fn chunks_for(&self, ring: bool, slots: u64) -> u64 {
        self.shape(ring)
            .map_or(0, |shape| slots.div_ceil(u64::from(shape.per_chunk.max(1))))
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_slot_is_read_by_its_slot_ref_and_a_ring_slot_by_its_grain() {
        let chunk = 48 * 256;
        let list = TokenShape {
            id: LIST,
            tokens: 16,
            slot_bytes: 12 * 256,
            per_chunk: 4,
            grain: gcd(chunk, 12 * 256),
            window: None,
            rows: vec![(0, 0), (3, 1024)],
        };
        assert_eq!(list.grain, list.slot_bytes);
        assert_eq!(list.page_id(chunk, 5, 3), u64::from(list.slot_ref(5, 3)));
        assert_eq!(list.offset_of(3), Some(1024));
        assert_eq!(list.offset_of(1), None);

        // A ring slot of 20 units in a chunk of 48: two a chunk, eight spare.
        let ring = TokenShape {
            id: RING,
            tokens: 16,
            slot_bytes: 20 * 256,
            per_chunk: 2,
            grain: gcd(chunk, 20 * 256),
            window: Some(64),
            rows: vec![(1, 0)],
        };
        assert_eq!(ring.grain, 4 * 256);
        for (chunk_id, index) in [(0u32, 0u32), (1, 1), (7, 0), (7, 1)] {
            let q = ring.page_id(chunk, chunk_id, index);
            assert_eq!(q * ring.grain, ring.slot_at(chunk, chunk_id, index));
        }
    }
}

//! The paged KV pool, sized by the fire's geometry rather than by a model.
//!
//! # What was already here
//!
//! `metal::stage_decode_storage` allocates `KvSlots { k_pages, v_pages }` per
//! full-attention layer, and has since the port. The pool is not missing — what
//! was missing is a pool whose SIZE comes from numbers a text states rather
//! than from `batch::DecodeGeometry`, which is a model definition inside the
//! driver and is on the retirement list.
//!
//! So this is not a re-port. It is the same allocation with its arguments
//! taken from the frame instead: layers, kv heads, head dim, page size, pages.
//! Every one of those is either the fire's geometry (which the caller already
//! hands the executor) or the ABI's own (`DeviceFacts::page_size`).
//!
//! # Page-major, one stride everywhere
//!
//! `store::kv_move`'s plan already assumes it, and says so: *"one plan serves
//! every buffer — the same `(src, dst, bytes)` offsets apply to the K pages
//! and the V pages of every full-attention layer, because the pool is
//! page-major `[page, row]` at one stride everywhere."* This lays the pool out
//! that way, which is what makes that plan true rather than hopeful.

use crate::error::Result;
use crate::metal::{Context, Handle, allocate};
use crate::region::Region as _;
use crate::store::{CellMovePlan, PoolGrid};

/// One full-attention layer's pages.
#[derive(Debug)]
pub struct Layer {
    /// The key pages, `[pages, page_size, kv_heads * head_dim]`.
    pub k: Handle,
    /// The value pages, same shape.
    pub v: Handle,
}

/// The shape a pool is allocated at.
///
/// Named rather than positional: five `u32`s in a row is the defect
/// `PARITY-LOADER.md` records in `plan_heap`, where two adjacent counts could
/// be swapped without any type noticing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Full-attention layers, each of which gets its own pages.
    pub layers: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Token rows per page — the ABI's `DeviceFacts::page_size`, and the unit
    /// every `kv_translation` index is in.
    pub page_size: u32,
    /// Physical pages the pool holds.
    pub pages: u32,
    /// Bytes per activation element (2 for bf16).
    pub element_bytes: u32,
}

impl Shape {
    /// Bytes one token row occupies: every head's channels, contiguously.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.kv_heads) * u64::from(self.head_dim) * u64::from(self.element_bytes)
    }

    /// Bytes one page occupies.
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        u64::from(self.page_size) * self.row_bytes()
    }

    /// Bytes one layer's K (or V) region occupies.
    #[must_use]
    pub fn layer_bytes(&self) -> u64 {
        u64::from(self.pages) * self.page_bytes()
    }

    /// The grid `store::kv_move` plans against.
    ///
    /// Handed over rather than restated at the call site: a move plan built
    /// from a grid that disagrees with the allocation is a move that lands
    /// somewhere else, and the two answers would be a page apart rather than
    /// obviously wrong.
    #[must_use]
    pub fn grid(&self) -> PoolGrid {
        PoolGrid {
            total_pages: self.pages,
            page_size: self.page_size,
            row_bytes: self.row_bytes(),
        }
    }
}

/// A paged KV pool: one K and one V region per full-attention layer.
#[derive(Debug)]
pub struct Pool {
    shape: Shape,
    layers: Vec<Layer>,
}

impl Pool {
    /// Allocate a pool at `shape`.
    ///
    /// # Errors
    ///
    /// Any layer's allocation. Nothing partial survives: the vector is local
    /// until every layer is in it, so a pool that half-allocated releases the
    /// half it got rather than being bound against.
    pub fn allocate(context: &Context, shape: Shape) -> Result<Self> {
        let bytes = shape.layer_bytes().max(1);
        let mut layers = Vec::with_capacity(shape.layers as usize);
        for _ in 0..shape.layers {
            layers.push(Layer {
                k: allocate(context, bytes, "kv k pages")?,
                v: allocate(context, bytes, "kv v pages")?,
            });
        }
        Ok(Self { shape, layers })
    }

    /// The shape this pool was allocated at.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Layer `l`'s pages, or `None` past the end.
    #[must_use]
    pub fn layer(&self, l: u32) -> Option<&Layer> {
        self.layers.get(l as usize)
    }

    /// Physical pages the pool holds — what a frame's `required_kv_pages` is
    /// admitted against.
    #[must_use]
    pub fn pages(&self) -> u32 {
        self.shape.pages
    }

    /// Whether `required` pages fit.
    ///
    /// The admission question, and it is a method rather than a comparison at
    /// the call site because "fits" is the pool's own word: a caller comparing
    /// against `pages()` would have to know the demand is exclusive.
    #[must_use]
    pub fn admits(&self, required: u32) -> bool {
        required <= self.shape.pages
    }

    /// Total bytes this pool holds, across every layer and both tensors.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.shape.layer_bytes() * 2 * u64::from(self.shape.layers)
    }

    /// Run a move plan over every layer's K and V pages.
    ///
    /// # One plan, every buffer, on the host
    ///
    /// `store::kv_move` says why one plan serves all of them: the pool is
    /// page-major at one stride everywhere, so the same `(src, dst, bytes)`
    /// offsets apply to the K and V pages of every layer. This walks them.
    ///
    /// **No encoder.** The pages are `StorageModeShared`, so they are host
    /// addressable and a move is a `memmove` — which is what `Region::copy`
    /// is, and its memmove semantics are not incidental here: a compaction
    /// slides rows toward the front of the pool, so source and destination
    /// overlap.
    ///
    /// What makes that safe is the fire's shape rather than a lock: `run`
    /// blocks until the stepper's command buffer completes, so between fires
    /// the host owns these bytes outright. A driver that overlapped fires
    /// would need a barrier here, and would know it because this comment
    /// stopped being true.
    ///
    /// # Errors
    ///
    /// A copy whose span leaves a layer's region — which the plan's own
    /// validation should have refused first, so reaching it is drift between
    /// the grid the plan was built from and the pool it is run against.
    pub fn apply(&self, plan: &CellMovePlan) -> Result<()> {
        for layer in &self.layers {
            for side in [&layer.k, &layer.v] {
                for copy in &plan.copies {
                    // SAFETY: both spans are checked against the region's own
                    // length by `Region::copy`, and overlap is permitted
                    // because the operation is a memmove.
                    unsafe { side.copy(copy.dst_off, side, copy.src_off, copy.bytes)? };
                }
            }
        }
        Ok(())
    }
}

/// Why a frame's page translation could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Untranslatable {
    /// The CSR does not describe the roster it came with.
    RaggedPartition {
        /// Segments the CSR describes.
        segments: usize,
        /// Entries the roster holds.
        roster: usize,
    },
    /// A lane's segment runs past the translation table.
    SegmentOutOfRange {
        /// Which lane.
        lane: usize,
        /// Where its segment ends.
        end: u32,
        /// How long the table is.
        len: usize,
    },
    /// The translation names a physical page the pool does not hold.
    ///
    /// This is the one that must not be tolerated: a page index past the pool
    /// addresses another layer's memory, or the allocator's, and attention
    /// would read it without complaint.
    PageOutOfRange {
        /// Which lane named it.
        lane: usize,
        /// The page it named.
        page: u32,
        /// How many the pool holds.
        pages: u32,
    },
}

/// The physical pages lane `lane` may address, checked against the pool.
///
/// `translation` and `indptr` are the frame's own
/// (`FrameSubmission::kv_translation` and its CSR): a WorkingSet-relative page
/// becomes a physical slot, and the segment for a lane is its run.
///
/// # Errors
///
/// [`Untranslatable`], naming the lane in every case, because a frame is
/// diagnosed per lane and a bare index says nothing about whose it was.
pub fn translate<'a>(
    pool: &Pool,
    translation: &'a [u32],
    indptr: &[u32],
    lane: usize,
) -> core::result::Result<&'a [u32], Untranslatable> {
    let segments = indptr.len().saturating_sub(1);
    if lane >= segments {
        return Err(Untranslatable::RaggedPartition {
            segments,
            roster: lane + 1,
        });
    }
    let (start, end) = (indptr[lane], indptr[lane + 1]);
    if end as usize > translation.len() || start > end {
        return Err(Untranslatable::SegmentOutOfRange {
            lane,
            end,
            len: translation.len(),
        });
    }
    let pages = &translation[start as usize..end as usize];
    // `0xFFFF_FFFF` marks a reserved-but-unmaterialized page, which is not a
    // page this lane will address — the ABI says so for the RS translation and
    // the same sentinel is used here.
    for &page in pages {
        if page != u32::MAX && page >= pool.pages() {
            return Err(Untranslatable::PageOutOfRange {
                lane,
                page,
                pages: pool.pages(),
            });
        }
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> Shape {
        Shape {
            layers: 2,
            kv_heads: 8,
            head_dim: 128,
            page_size: 16,
            pages: 64,
            element_bytes: 2,
        }
    }

    #[test]
    fn a_pages_bytes_are_its_rows_times_every_heads_channels() {
        let s = shape();
        assert_eq!(s.row_bytes(), 8 * 128 * 2);
        assert_eq!(s.page_bytes(), 16 * 8 * 128 * 2);
        assert_eq!(s.layer_bytes(), 64 * 16 * 8 * 128 * 2);
    }

    #[test]
    fn the_move_grid_is_the_pools_own_and_not_a_restatement() {
        // A move plan built from a grid that disagrees with the allocation
        // lands a page away, which is not obviously wrong from either side.
        let s = shape();
        let grid = s.grid();
        assert_eq!(grid.total_pages, s.pages);
        assert_eq!(grid.page_size, s.page_size);
        assert_eq!(grid.row_bytes, s.row_bytes());
    }

    /// A pool with no device behind it, for the translation checks.
    fn paperless(pages: u32) -> Pool {
        Pool {
            shape: Shape { pages, ..shape() },
            layers: Vec::new(),
        }
    }

    #[test]
    fn a_lane_gets_the_run_the_csr_gives_it() {
        let pool = paperless(64);
        let table = [3u32, 4, 9, 1];
        assert_eq!(
            translate(&pool, &table, &[0, 2, 4], 0).expect("lane 0"),
            &[3, 4]
        );
        assert_eq!(
            translate(&pool, &table, &[0, 2, 4], 1).expect("lane 1"),
            &[9, 1]
        );
    }

    #[test]
    fn a_page_past_the_pool_is_refused_by_lane() {
        // The one that must not be tolerated: a page index past the pool
        // addresses another layer's memory, and attention would read it
        // without complaint.
        let pool = paperless(8);
        assert_eq!(
            translate(&pool, &[0, 99], &[0, 2], 0),
            Err(Untranslatable::PageOutOfRange {
                lane: 0,
                page: 99,
                pages: 8
            })
        );
    }

    #[test]
    fn the_unmaterialized_sentinel_is_not_a_page_out_of_range() {
        // `0xFFFF_FFFF` marks reserved-but-unmaterialized. Reading it as an
        // index would refuse every frame that reserves ahead.
        let pool = paperless(8);
        assert_eq!(
            translate(&pool, &[0, u32::MAX], &[0, 2], 0).expect("a reserved page is not a fault"),
            &[0, u32::MAX]
        );
    }

    #[test]
    fn a_segment_past_the_table_names_the_lane_that_wanted_it() {
        let pool = paperless(8);
        assert_eq!(
            translate(&pool, &[0, 1], &[0, 9], 0),
            Err(Untranslatable::SegmentOutOfRange {
                lane: 0,
                end: 9,
                len: 2
            })
        );
    }

    #[test]
    fn admission_is_the_pools_own_word() {
        let pool = paperless(8);
        assert!(pool.admits(8), "an exact fit fits");
        assert!(!pool.admits(9));
    }
}

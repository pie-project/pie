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
    /// The per-head width the FULL-attention layers use, or zero for a stack
    /// whose layers all share [`Self::head_dim`].
    ///
    /// gemma-4 states two. Its full layers are twice as wide per head as its
    /// sliding ones and carry a quarter the KV heads, so their pages are a
    /// different size — which is why this reaches the POOL and not only the
    /// text.
    pub global_head_dim: u32,
    /// The key/value head count the FULL-attention layers use, or zero for
    /// one shape everywhere. See [`Self::global_head_dim`].
    pub global_kv_heads: u32,
    /// One full-attention layer every `full_attn_every`, or zero for a stack
    /// that does not alternate.
    ///
    /// The same rule `model::text` derives `window_left` from, so the pool
    /// and the text agree about which layers are full without a second list
    /// to keep in step.
    pub full_attn_every: u32,
}

impl Shape {
    /// Whether layer `l` attends the whole context.
    #[must_use]
    pub fn is_full_attention(&self, l: u32) -> bool {
        self.full_attn_every > 1 && (l + 1).is_multiple_of(self.full_attn_every)
    }

    /// Whether every layer has the same page size.
    ///
    /// True for every deployment but gemma-4, and the question anything that
    /// wants ONE stride for the whole pool has to ask first.
    #[must_use]
    pub const fn is_uniform(&self) -> bool {
        self.global_head_dim == 0 && self.global_kv_heads == 0
    }

    /// This layer's key/value head count and per-head width.
    #[must_use]
    pub fn heads_at(&self, l: u32) -> (u32, u32) {
        if self.is_full_attention(l) {
            (
                if self.global_kv_heads > 0 { self.global_kv_heads } else { self.kv_heads },
                if self.global_head_dim > 0 { self.global_head_dim } else { self.head_dim },
            )
        } else {
            (self.kv_heads, self.head_dim)
        }
    }

    /// Bytes one token row of layer `l` occupies: every head's channels,
    /// contiguously.
    ///
    /// Per layer because `kv_append_paged` derives its own row stride from
    /// the `n_kv_heads` and `head_dim` the statement hands it — so a layer's
    /// pages are packed at that layer's shape, and its allocation has to be
    /// sized the same way.
    #[must_use]
    pub fn row_bytes_at(&self, l: u32) -> u64 {
        let (heads, dim) = self.heads_at(l);
        heads as u64 * dim as u64 * self.element_bytes as u64
    }

    /// Bytes one page of layer `l` occupies.
    #[must_use]
    pub fn page_bytes_at(&self, l: u32) -> u64 {
        self.page_size as u64 * self.row_bytes_at(l)
    }

    /// Bytes layer `l`'s K (or V) region occupies.
    #[must_use]
    pub fn layer_bytes_at(&self, l: u32) -> u64 {
        self.pages as u64 * self.page_bytes_at(l)
    }

    /// Bytes one token row occupies, on a pool where every layer agrees.
    ///
    /// `None` on a stack with two attention shapes, which is the whole point:
    /// a caller that wants one stride for the pool has to find out that there
    /// is not one, rather than being handed the first layer's and applying it
    /// to all of them.
    #[must_use]
    pub fn row_bytes(&self) -> Option<u64> {
        if self.is_uniform() { Some(self.row_bytes_at(0)) } else { None }
    }

    /// Bytes one page occupies, where every layer agrees. See
    /// [`Self::row_bytes`].
    #[must_use]
    pub fn page_bytes(&self) -> Option<u64> {
        if self.is_uniform() { Some(self.page_bytes_at(0)) } else { None }
    }

    /// The grid `store::kv_move` plans against, where every layer agrees.
    ///
    /// Handed over rather than restated at the call site: a move plan built
    /// from a grid that disagrees with the allocation is a move that lands
    /// somewhere else, and the two answers would be a page apart rather than
    /// obviously wrong.
    ///
    /// `None` on a heterogeneous pool. A move planned at one stride and
    /// applied to every layer is exactly the "page apart rather than
    /// obviously wrong" failure this exists to prevent, one axis over.
    #[must_use]
    pub fn grid(&self) -> Option<PoolGrid> {
        self.row_bytes().map(|row_bytes| PoolGrid {
            total_pages: self.pages,
            page_size: self.page_size,
            row_bytes,
        })
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
        let mut layers = Vec::with_capacity(shape.layers as usize);
        for l in 0..shape.layers {
            // THIS layer's size. gemma-4's full-attention layers pack their
            // pages at a different shape from its sliding ones, and
            // `kv_append_paged` derives its row stride from the statement's
            // own `n_kv_heads`/`head_dim` -- so each layer's pages are
            // self-consistent at its own shape, and each allocation has to
            // match. Uniform stacks answer the same number for every layer.
            let bytes = shape.layer_bytes_at(l).max(1);
            let (k, v) = (
                allocate(context, bytes, "kv k pages")?,
                allocate(context, bytes, "kv v pages")?,
            );
            // ZEROED, and it is the difference between a pool and a hazard.
            //
            // A fresh Metal buffer is usually zero and nothing promises it. An
            // attention reads every row its length says it has, and a row the
            // fire has not written yet is one the allocator last handed to
            // somebody else -- so a decode over a half-filled pool attends to
            // whatever was there. Measured: three runs of one fire over one
            // checkpoint gave widest activations of 29.75, 222208 and 53.75.
            // Same input, same weights, three answers.
            //
            // Zero is also the semantically right filler: an all-zero key
            // contributes a uniform logit rather than a wild one, so a pool
            // read too far degrades toward the mean instead of exploding.
            //
            // SAFETY: freshly allocated; nothing is encoded against either.
            unsafe {
                k.zero(0, k.len())?;
                v.zero(0, v.len())?;
            }
            layers.push(Layer { k, v });
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
        // Summed rather than multiplied: a stack with two attention shapes
        // has two page sizes, and one layer's times the count is neither.
        (0..self.shape.layers)
            .map(|l| self.shape.layer_bytes_at(l) * 2)
            .sum()
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
    ///
    /// # A heterogeneous pool is refused
    ///
    /// The offsets in a plan are BYTES, computed by the caller from one page
    /// size. On a stack with two attention shapes there is no such number:
    /// applying the full layers' offsets to the sliding ones lands a page
    /// apart rather than obviously wrong, which is the failure `Shape::grid`
    /// exists to prevent one axis over. `Shape::grid` returns `None` there,
    /// so a caller cannot build the plan in the first place -- this is the
    /// second door, for a plan built some other way.
    pub fn apply(&self, plan: &CellMovePlan) -> Result<()> {
        if !self.shape.is_uniform() {
            return Err(crate::Error::Create {
                what: "kv move",
                message: "this pool's layers have two page sizes (gemma-4's \
                          full-attention layers are not its sliding ones), and \
                          a move plan states one set of byte offsets for every \
                          layer"
                    .to_owned(),
            });
        }
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
            global_head_dim: 0,
            global_kv_heads: 0,
            full_attn_every: 0,
        }
    }

    #[test]
    fn a_pages_bytes_are_its_rows_times_every_heads_channels() {
        let s = shape();
        assert_eq!(s.row_bytes(), Some(8 * 128 * 2));
        assert_eq!(s.page_bytes(), Some(16 * 8 * 128 * 2));
        assert_eq!(s.layer_bytes_at(0), 64 * 16 * 8 * 128 * 2);
        // A uniform stack answers the same number for every layer, which is
        // what makes the per-layer form a strict generalisation.
        assert_eq!(s.layer_bytes_at(1), s.layer_bytes_at(0));
    }

    /// gemma-4's stack has TWO page sizes, and the pool has to allocate each
    /// layer at its own.
    ///
    /// Measured on `gemma-4-31b-it-4bit`: its sliding layers are 16 kv heads
    /// x 256, its full ones 4 x 512, and one layer in six is full. The pool
    /// used to size every layer from the sliding shape, so a full layer's
    /// pages were allocated for 4096 elements a row and written at 2048 --
    /// which does not fault, because the write is SMALLER. It reads back
    /// whatever the allocator left in the other half.
    #[test]
    fn a_stack_with_two_attention_shapes_sizes_each_layer_at_its_own() {
        let s = Shape {
            layers: 12,
            kv_heads: 16,
            head_dim: 256,
            page_size: 16,
            pages: 64,
            element_bytes: 2,
            global_head_dim: 512,
            global_kv_heads: 4,
            full_attn_every: 6,
        };
        assert!(!s.is_uniform());
        // Layers 5 and 11 are full; the rest slide.
        assert!(s.is_full_attention(5) && s.is_full_attention(11));
        assert!(!s.is_full_attention(0) && !s.is_full_attention(4));
        assert_eq!(s.heads_at(0), (16, 256));
        assert_eq!(s.heads_at(5), (4, 512));
        assert_eq!(s.row_bytes_at(0), 16 * 256 * 2);
        assert_eq!(s.row_bytes_at(5), 4 * 512 * 2);
        assert_ne!(
            s.layer_bytes_at(0),
            s.layer_bytes_at(5),
            "if these agreed the whole distinction would be decorative"
        );

        // And nothing may take ONE stride for this pool. A move planned at
        // either shape and applied to both lands a page apart rather than
        // obviously wrong.
        assert_eq!(s.row_bytes(), None);
        assert_eq!(s.page_bytes(), None);
        assert!(s.grid().is_none());
    }

    #[test]
    fn the_move_grid_is_the_pools_own_and_not_a_restatement() {
        // A move plan built from a grid that disagrees with the allocation
        // lands a page away, which is not obviously wrong from either side.
        let s = shape();
        let grid = s.grid().expect("a uniform pool has one grid");
        assert_eq!(grid.total_pages, s.pages);
        assert_eq!(grid.page_size, s.page_size);
        assert_eq!(Some(grid.row_bytes), s.row_bytes());
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

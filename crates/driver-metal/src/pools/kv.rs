//! The allocated pool: the half of a KV store that needs a device.
//!
//! [`crate::layout::kv::Shape`] is the arithmetic and is portable; this is what turns it
//! into memory. The split is `.wiki/driver/north-star.md` rule 2 — *cut by "is
//! this correct without a GPU?", not by subsystem* — and it is load-bearing
//! rather than tidy: `lowering::resolve` reads a pool's strides to answer
//! `KvHeadStride`/`KvSeqStride`, and it is host logic that has to stay
//! testable on a box with no card. While `Shape` lived beside `Pool` behind
//! one Apple gate, `resolve` named a module that a non-Apple build had
//! removed, and no job in the tree compiled the configuration that would have
//! said so.

use crate::device::{Allocation, Arena, Context, Elastic, Regions, create_elastic};
use crate::error::Result;
use crate::layout::kv_move::CellMovePlan;
use crate::layout::region::Region as _;

use std::ptr::NonNull;

use crate::layout::kv::Shape;

/// Where one side of a layer's pages lives.
///
/// Two storage kinds behind one set of operations. [`Fixed`] is a single
/// allocation sized once at load; [`Elastic`] is a sparse address space with
/// memory attached under it as the pool grows, which is what lets a pool be
/// resized without every bound address moving.
///
/// The operations are the same for both because both are host addressable --
/// [`Fixed`] because [`Allocation`] makes `Shared` buffers, [`Elastic`] because
/// its placement heaps are `Shared` even though the sparse buffer over them
/// is `Private`. That is not a coincidence: it is why `make_chunk` picks the
/// mode it picks. Without it [`Pool::apply`] could not stay a `memmove` and
/// prefix sharing would need an encoder.
///
/// [`Fixed`]: Self::Fixed
/// [`Elastic`]: Self::Elastic
#[derive(Debug)]
pub enum Pages {
    /// One allocation, sized once, never resized.
    Fixed(Allocation),
    /// A sparse address space, sized once, with memory attached as needed.
    Elastic(Elastic),
}

impl Pages {
    /// The address a kernel binds. Stable across a resize for either kind.
    #[must_use]
    pub fn gpu_address(&self) -> u64 {
        match self {
            Self::Fixed(h) => h.gpu_address(),
            Self::Elastic(e) => e.gpu_address(),
        }
    }

    /// How many bytes the pages span.
    ///
    /// Address space for [`Elastic`](Self::Elastic), which is not the same as
    /// memory: a span inside this length can still be unmapped, and asking
    /// for it is refused rather than served. A pool commits its whole length
    /// at load, so the two agree unless something has trimmed it.
    #[must_use]
    pub fn len(&self) -> u64 {
        match self {
            Self::Fixed(h) => h.len(),
            Self::Elastic(e) => e.len(),
        }
    }

    /// Whether it holds no bytes at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Name these pages to the residency set, so a fire may read them.
    pub fn register(&self, regions: &mut Regions) {
        match self {
            Self::Fixed(h) => regions.add(h),
            // An elastic buffer and its heaps are named in the context's own
            // residency set when they are created and when they grow, which
            // a `Regions` cannot do for it: the heaps are not knowable from
            // out here and they change under it.
            Self::Elastic(_) => {}
        }
    }

    /// A host address for the `bytes` at `offset`.
    ///
    /// # Errors
    ///
    /// A span that leaves the pages, or -- for [`Elastic`](Self::Elastic) --
    /// one that reaches past what is mapped or across a chunk seam.
    pub fn host_span(&self, offset: u64, bytes: u64) -> Result<NonNull<u8>> {
        match self {
            Self::Fixed(h) => {
                h.check("kv pages", offset, bytes)?;
                let base = h.contents().as_ptr().cast::<u8>();
                // SAFETY: `check` put the span inside the allocation.
                let at = unsafe { base.add(usize::try_from(offset).unwrap_or(usize::MAX)) };
                NonNull::new(at).ok_or(crate::Error::Create {
                    what: "kv pages",
                    message: "the allocation has no host address".to_owned(),
                })
            }
            Self::Elastic(e) => e.host_span(offset, bytes),
        }
    }

    /// Clear `bytes` at `offset`.
    ///
    /// # Errors
    ///
    /// As [`host_span`](Self::host_span).
    ///
    /// # Safety
    ///
    /// Nothing may be reading these bytes on the GPU.
    pub unsafe fn zero(&self, offset: u64, bytes: u64) -> Result<()> {
        match self {
            // SAFETY: the caller's.
            Self::Fixed(h) => unsafe { h.zero(offset, bytes) },
            // SAFETY: the caller's.
            Self::Elastic(e) => unsafe { e.zero(offset, bytes) },
        }
    }

    /// Move `bytes` from `src` to `dst` within these pages. A `memmove`: the
    /// spans may overlap, and in a compaction they do.
    ///
    /// # Errors
    ///
    /// As [`host_span`](Self::host_span), for either side.
    ///
    /// # Safety
    ///
    /// As [`zero`](Self::zero).
    pub unsafe fn copy_within(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
        match self {
            // SAFETY: the caller's. Both spans are checked by `Region::copy`.
            Self::Fixed(h) => unsafe { h.copy(dst, h.handle(), src, bytes) },
            // SAFETY: the caller's.
            Self::Elastic(e) => unsafe { e.copy_within(dst, src, bytes) },
        }
    }

    /// Write `src` at `offset`.
    ///
    /// # Errors
    ///
    /// As [`host_span`](Self::host_span).
    pub fn write(&self, offset: u64, src: &[u8]) -> Result<()> {
        let bytes = src.len() as u64;
        if bytes == 0 {
            return Ok(());
        }
        let at = self.host_span(offset, bytes)?;
        // SAFETY: `host_span` returned `bytes` writable bytes there, and the
        // source is a slice the caller owns, so they cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), at.as_ptr(), src.len()) };
        Ok(())
    }
}

/// One full-attention layer's pages.
#[derive(Debug)]
pub struct Layer {
    /// The key pages, `[pages, page_size, kv_heads * head_dim]`.
    pub k: Pages,
    /// The value pages, same shape.
    pub v: Pages,
}

/// A paged KV pool: one K and one V region per full-attention layer.
#[derive(Debug)]
pub struct Pool {
    shape: Shape,
    /// The page count the pages were RESERVED at, which a resize may not
    /// exceed.
    ///
    /// Distinct from `shape.pages`, which is what the pool can serve right
    /// now: a shrink lowers the second so the scheduler stops admitting onto
    /// memory that has been given back, but the address space is still
    /// reserved and the pool can take it again. Folding the two together
    /// makes a shrink permanent -- measured, by a resize down and back up
    /// that was refused on the way up.
    reserved: u32,
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
        let pool = Self::build(shape, |bytes| {
            Ok(Pages::Fixed(Allocation::new(context, bytes, "kv pages")?))
        })?;
        pool.zero_all()?;
        Ok(pool)
    }

    /// Allocate a pool at `shape` whose pages can be resized.
    ///
    /// Same pool, different storage: every layer is a sparse buffer in
    /// `arena`, committed to its full size here so that behaviour is
    /// identical to [`allocate`](Self::allocate) until somebody asks for it
    /// not to be. The point is not to start smaller -- it is that the
    /// addresses bound into every argument table survive a later resize,
    /// which a fixed allocation cannot offer at any size.
    ///
    /// # Errors
    ///
    /// Any layer's allocation, or an arena without room for the whole pool.
    /// Nothing partial survives, as [`allocate`](Self::allocate).
    pub fn allocate_elastic(
        context: &Context,
        stepper: &mut crate::device::Stepper,
        arena: &Arena,
        shape: Shape,
    ) -> Result<Self> {
        let mut pool = Self::build(shape, |bytes| {
            Ok(Pages::Elastic(create_elastic(context, arena, bytes)?))
        })?;
        // Address space costs nothing; the memory under it is what a fire
        // reads. Committed in one ask so that a pool which does not fit is
        // refused whole rather than half-mapped -- and, having been asked
        // for, declared mandatory, because a pool that pressure can unmap
        // underneath a bound address is not a pool.
        let mut targets: Vec<(&mut Elastic, u64)> = Vec::new();
        for layer in &mut pool.layers {
            for side in [&mut layer.k, &mut layer.v] {
                if let Pages::Elastic(e) = side {
                    let want = e.len();
                    targets.push((e, want));
                }
            }
        }
        stepper.ensure_all(
            &mut targets,
            crate::device::Pressure::Normal,
            // Not growth. [`Need`] spells out why: the pool is what makes an
            // admitted model exist, and admission already weighed the machine
            // against the whole budget -- clamping here does not hand a page
            // back, it turns a model that just loaded into one that cannot
            // take a step.
            crate::device::Need::Step,
        )?;
        for (buffer, bytes) in targets {
            stepper.declare_mandatory(buffer, bytes);
        }
        pool.zero_all()?;
        Ok(pool)
    }

    /// The layer walk both allocators share.
    ///
    /// # Errors
    ///
    /// Whatever `make` refuses. Nothing partial survives: the vector is local
    /// until every layer is in it, so a pool that half-allocated releases the
    /// half it got rather than being bound against.
    fn build(shape: Shape, mut make: impl FnMut(u64) -> Result<Pages>) -> Result<Self> {
        let mut layers = Vec::with_capacity(shape.layers as usize);
        for l in 0..shape.layers {
            // THIS layer's size. gemma-4's full-attention layers pack their
            // pages at a different shape from its sliding ones, and
            // `kv_append_paged` derives its row stride from the statement's
            // own `n_kv_heads`/`head_dim` -- so each layer's pages are
            // self-consistent at its own shape, and each allocation has to
            // match. Uniform stacks answer the same number for every layer.
            let bytes = shape.layer_bytes_at(l).max(1);
            layers.push(Layer {
                k: make(bytes)?,
                v: make(bytes)?,
            });
        }
        Ok(Self {
            shape,
            reserved: shape.pages,
            layers,
        })
    }

    /// Clear every page.
    ///
    /// ZEROED, and it is the difference between a pool and a hazard.
    ///
    /// A fresh Metal buffer is usually zero and nothing promises it. An
    /// attention reads every row its length says it has, and a row the fire
    /// has not written yet is one the allocator last handed to somebody else
    /// -- so a decode over a half-filled pool attends to whatever was there.
    /// Measured: three runs of one fire over one checkpoint gave widest
    /// activations of 29.75, 222208 and 53.75. Same input, same weights,
    /// three answers.
    ///
    /// Zero is also the semantically right filler: an all-zero key
    /// contributes a uniform logit rather than a wild one, so a pool read too
    /// far degrades toward the mean instead of exploding.
    ///
    /// # Errors
    ///
    /// A span that is not addressable, which for an elastic pool means one
    /// the commit did not reach.
    fn zero_all(&self) -> Result<()> {
        for layer in &self.layers {
            for side in [&layer.k, &layer.v] {
                // SAFETY: freshly allocated; nothing is encoded against it.
                unsafe { side.zero(0, side.len())? };
            }
        }
        Ok(())
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

    /// Commit or release memory so the pool holds `target` pages.
    ///
    /// # What a resize is, and is not
    ///
    /// It is not a reallocation. Every address bound into an argument table
    /// stays exactly where it was -- that is the whole reason the pages are
    /// sparse. What changes is how much memory is attached under those
    /// addresses, and therefore how many pages a frame may name.
    ///
    /// `pages()` follows the target, because the pool's page count IS what it
    /// can serve: a pool that reported the old count after giving memory back
    /// would have the scheduler admitting frames onto pages that are no
    /// longer there, and a sparse read of an unmapped page returns zeros
    /// rather than faulting -- a wrong answer, not a crash.
    ///
    /// Growing past the count the pool was allocated at is refused. That is
    /// the address space it reserved, and past it there is nothing to attach
    /// memory to; it is also the count admission was granted against, so a
    /// pool that grew past it would be serving pages nobody weighed.
    ///
    /// Newly committed pages are cleared, for the reason
    /// `zero_all` gives: a page handed to a frame with
    /// somebody else's bytes in it is attended to as if it were keys.
    ///
    /// # Errors
    ///
    /// A fixed pool, which has one allocation and cannot give part of it
    /// back; a target past what was allocated; or an arena without room to
    /// grow into.
    pub fn resize(&mut self, stepper: &mut crate::device::Stepper, target: u32) -> Result<()> {
        if !matches!(self.layers.first().map(|l| &l.k), Some(Pages::Elastic(_))) {
            return Err(crate::Error::Create {
                what: "kv resize",
                message: "this pool's pages are one fixed allocation per \
                          layer; resizing them would move every address \
                          already bound into an argument table"
                    .to_owned(),
            });
        }
        if target > self.reserved {
            return Err(crate::Error::OutOfRange {
                what: "kv resize",
                offset: u64::from(target),
                bytes: 1,
                len: u64::from(self.reserved),
            });
        }
        let was = self.shape.pages;
        if target == was {
            return Ok(());
        }
        // The size each layer's pages want, at the new count. Per layer,
        // because a stack with two attention shapes has two page sizes and
        // one of them times the count is neither.
        let want: Vec<u64> = (0..self.shape.layers)
            .map(|l| (self.shape.page_bytes_at(l) * u64::from(target)).max(1))
            .collect();

        if target > was {
            // Priced and mapped in one ask, so a growth that does not fit
            // leaves the pool at the size it was rather than half-grown.
            let mut targets: Vec<(&mut Elastic, u64)> = Vec::new();
            for (l, layer) in self.layers.iter_mut().enumerate() {
                let bytes = want[l];
                for side in [&mut layer.k, &mut layer.v] {
                    if let Pages::Elastic(e) = side {
                        targets.push((e, bytes));
                    }
                }
            }
            stepper.ensure_all(
                &mut targets,
                crate::device::Pressure::Normal,
                crate::device::Need::Step,
            )?;
            for (buffer, bytes) in targets {
                stepper.declare_mandatory(buffer, bytes);
            }
        }
        // Only after the memory is in hand: a frame admitted against a count
        // the pool does not hold yet is the failure this ordering avoids.
        self.shape.pages = target;

        if target > was {
            for (l, layer) in self.layers.iter().enumerate() {
                let from = self.shape.page_bytes_at(l as u32) * u64::from(was);
                for side in [&layer.k, &layer.v] {
                    // SAFETY: the pages past `was` were not in the pool a
                    // frame could name until the line above, so nothing has
                    // encoded against them.
                    unsafe { side.zero(from, want[l] - from)? };
                }
            }
        } else {
            // Shrinking last, and only after the count is down: the trim
            // waits for the GPU to pass the unmap, but the count is what
            // stops a new frame from naming these pages in the first place.
            for (l, layer) in self.layers.iter_mut().enumerate() {
                let bytes = want[l];
                for side in [&mut layer.k, &mut layer.v] {
                    if let Pages::Elastic(e) = side {
                        // No `declare_mandatory` beside this one: the trim
                        // lowers the claim itself, and the declaration only
                        // ever raises.
                        stepper.trim(e, bytes)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The most pages this pool can ever hold.
    ///
    /// What it RESERVED address space for, which is not what it currently
    /// holds: a trimmed pool can be grown back to this without any address
    /// moving. Admission is against this rather than [`pages`](Self::pages),
    /// because a frame that needs more than is mapped needs a resize, not a
    /// refusal.
    #[must_use]
    pub fn reserved_pages(&self) -> u32 {
        self.reserved
    }

    /// Whether `required` pages fit.
    ///
    /// The admission question, and it is a method rather than a comparison at
    /// the call site because "fits" is the pool's own word: a caller comparing
    /// against `pages()` would have to know the demand is exclusive.
    ///
    /// Against the RESERVED count. A pool that has given memory back can take
    /// it again, so a frame within the reservation is one the pool can serve
    /// -- after a resize, which is the caller's next move and not a reason to
    /// call the frame impossible.
    #[must_use]
    pub fn admits(&self, required: u32) -> bool {
        required <= self.reserved
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
    /// `layout::kv_move` says why one plan serves all of them: the pool is
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
                    // SAFETY: both spans are checked against the pages' own
                    // length, and overlap is permitted because the operation
                    // is a memmove.
                    unsafe { side.copy_within(copy.dst_off, copy.src_off, copy.bytes)? };
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

    /// A pool with no device behind it, for the translation checks.
    fn paperless(pages: u32) -> Pool {
        Pool {
            shape: Shape { pages, ..shape() },
            reserved: pages,
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

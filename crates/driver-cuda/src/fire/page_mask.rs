//! The page mask: `FirePageMask` and the hook-graph prepare pass that must predict it
//! exactly (port of `csrc/src/model/attn_page_mask.{hpp,cu}`).
//! Both paths must agree to the byte or the replayed graph misreads the compacted
//! table; both call one layout, [`MaskSlotLayout::plan`].

use core::ffi::c_void;

use super::sideband_arena::{Refusal, Region, SidebandArena};

/// Sub-buffer alignment inside the arena's mask slot: 256 is also `cudaMalloc`'s
/// guarantee, so any multiple aligns all four element types.
pub const SIDEBAND_ALIGN: usize = 256;

const fn align_up(n: usize) -> usize {
    n.next_multiple_of(SIDEBAND_ALIGN)
}

/// Why a fire cannot carry a page mask. Split out so a caller can tell "no pages"
/// (routine — a first-token fire has none) from a malformed CSR (a bug).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskError {
    /// The observation is missing geometry the carve needs.
    NoGeometry,
    /// The fire has no KV pages at all, so there is nothing to mask.
    NoPages,
    /// A request's page range runs backwards or past the end of the table.
    MalformedCsr,
    /// A compaction asked for a different request count than the fire carries —
    /// distinct from [`Self::MalformedCsr`]: the CSR is fine, the caller is out of
    /// step.
    RequestCountMismatch,
    /// The fire carries no `sideband_arena` to carve from.
    NoArena,
    /// The arena refused the acquire.
    ArenaRefused(Refusal),
}

impl MaskError {
    /// The C++ `what()` string this corresponds to, or `None` where C++ returned null.
    #[must_use]
    pub const fn cpp_message(self) -> Option<&'static str> {
        match self {
            Self::NoGeometry => Some("attn_page_mask needs a fire with kv geometry"),
            Self::NoPages | Self::MalformedCsr => {
                Some("attn_page_mask fire has no kv pages or a malformed page CSR")
            }
            Self::RequestCountMismatch => {
                Some("attn_page_mask compaction and fire disagree on request count")
            }
            Self::NoArena => Some("attn_page_mask fire carries no hook sideband arena"),
            Self::ArenaRefused(_) => Some("attn_page_mask could not acquire its page buffers"),
        }
    }
}

/// The fire geometry the carve reads: the host page CSR of `AttentionObservation`,
/// nothing else — the device CSR is deliberately absent, since the host copy only sizes
/// rows and never addresses them.
#[derive(Debug, Clone, Copy)]
pub struct FireGeometry<'a> {
    /// Host page CSR: `num_requests + 1` entries, a conservative *bound* on the real
    /// per-request page counts (only ever over-allocates).
    pub kv_page_indptr_h: &'a [u32],
}

impl<'a> FireGeometry<'a> {
    /// Wraps a host page CSR, rejecting one too short to describe a fire: fewer than
    /// two entries is zero requests. Device-pointer checks are `usable()`'s job, not
    /// this one's.
    pub const fn new(kv_page_indptr_h: &'a [u32]) -> Result<Self, MaskError> {
        if kv_page_indptr_h.len() < 2 {
            return Err(MaskError::NoGeometry);
        }
        Ok(Self { kv_page_indptr_h })
    }

    /// Requests in the fire.
    #[must_use]
    pub const fn num_requests(&self) -> u32 {
        (self.kv_page_indptr_h.len() - 1) as u32
    }
}

/// The mask slot's size and the offsets of its five buffers — the single definition of
/// the carve, used by both `FirePageMask::new` and `prepare_page_mask_capture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskSlotLayout {
    /// Requests in the fire; the keep rows and both length arrays are this long.
    pub num_requests: u32,
    /// Entries per keep row: the widest request's page count, so every row is at least
    /// as long as its page list.
    pub stride: u32,
    /// Bytes to acquire from the arena's [`Region::Mask`] slot.
    pub total: usize,
    /// Compacted page ids. Always 0 — it is the slot base.
    pub indices_offset: usize,
    /// Compacted per-request page offsets, `num_requests + 1` entries.
    pub indptr_offset: usize,
    /// Scratch for the compaction's per-request survivor counts.
    pub counts_offset: usize,
    /// Compacted per-request last-page lengths.
    pub last_lens_offset: usize,
    /// The `[num_requests, stride]` u8 keep rows. Last: u8 is the one element type here
    /// with no alignment requirement.
    pub keep_offset: usize,
}

impl MaskSlotLayout {
    /// Sizes and carves the mask slot for a fire, or says why it cannot. u32 outputs
    /// come first, u8 keep rows last (no alignment needed), so the total is
    /// `keep_offset + keep_bytes` with no pad.
    pub fn plan(geometry: FireGeometry<'_>) -> Result<Self, MaskError> {
        let requests = geometry.num_requests();
        let indptr = geometry.kv_page_indptr_h;
        let total_pages = indptr[requests as usize];
        if total_pages == 0 {
            return Err(MaskError::NoPages);
        }

        let mut stride = 0u32;
        for r in 0..requests as usize {
            let begin = indptr[r];
            let end = indptr[r + 1];
            if end < begin || end > total_pages {
                return Err(MaskError::MalformedCsr);
            }
            stride = stride.max(end - begin);
        }
        if stride == 0 {
            return Err(MaskError::NoPages);
        }

        let keep_bytes = requests as usize * stride as usize;
        let idx_bytes = total_pages as usize * size_of::<u32>();
        let indptr_bytes = (requests as usize + 1) * size_of::<u32>();
        let lens_bytes = requests as usize * size_of::<u32>();

        let indices_offset = 0;
        let indptr_offset = align_up(idx_bytes);
        let counts_offset = indptr_offset + align_up(indptr_bytes);
        let last_lens_offset = counts_offset + align_up(lens_bytes);
        let keep_offset = last_lens_offset + align_up(lens_bytes);

        Ok(Self {
            num_requests: requests,
            stride,
            total: keep_offset + keep_bytes,
            indices_offset,
            indptr_offset,
            counts_offset,
            last_lens_offset,
            keep_offset,
        })
    }

    /// Bytes `begin_layer` must seed — every keep row in full; a seed short by one row
    /// leaves a stale mask evicting pages the program never asked to evict.
    #[must_use]
    pub const fn keep_bytes(&self) -> usize {
        self.num_requests as usize * self.stride as usize
    }
}

/// The write side of the attention hook: `[num_requests, stride]` u8 row-major; 1 keeps
/// the page, entry `[r, p]` governing request `r`'s slot `p`. Pre-filled with 1 each
/// hook, so a layer with no sink stays unrestricted.
#[derive(Debug)]
pub struct AttentionMaskSink {
    /// The `[num_requests, stride]` u8 rows, row-major.
    pub keep: *mut u8,
    /// Rows in `keep`.
    pub num_requests: u32,
    /// Entries per row: an upper bound on any request's page count.
    pub stride: u32,
    /// Layer whose sink last wrote `keep`, or `None` for "nothing written" — stops a
    /// mask for layer L silently governing L+1 when the sink stops.
    pub written_layer: Option<u32>,
}

impl AttentionMaskSink {
    /// Whether the sink actually points at rows a program could write.
    #[must_use]
    pub fn usable(&self) -> bool {
        !self.keep.is_null() && self.num_requests > 0 && self.stride > 0
    }
}

/// What a captured hook body needs before it exists. Every pointer here is baked into
/// the graph, so all must equal what the fire-time carve produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageMaskCapturePlan {
    /// Where the sink kernel's destination rows will be.
    pub keep: *mut u8,
    /// Rows in `keep`.
    pub num_requests: u32,
    /// Entries per row.
    pub stride: u32,
    /// Where the compaction will write the gathered page ids.
    pub out_indices: *const u32,
    /// Where the compaction will write the gathered CSR offsets.
    pub out_indptr: *const u32,
    /// Where the compaction will write the gathered last-page lengths.
    pub out_last_lens: *const u32,
}

/// The prepare pass: pre-grow the mask slot and report where it carves. Growth is a
/// stream-synchronised free-and-realloc that must not happen inside a captured region,
/// so it happens here; capture-time [`FirePageMask::new`] then just finds capacity.
pub fn prepare_page_mask_capture<M: super::sideband_arena::DeviceMemory>(
    arena: &mut SidebandArena,
    mem: &mut M,
    geometry: FireGeometry<'_>,
) -> Result<PageMaskCapturePlan, MaskError> {
    let layout = MaskSlotLayout::plan(geometry)?;
    let base = arena
        .acquire(mem, Region::Mask, layout.total)
        .map_err(MaskError::ArenaRefused)?
        .cast::<u8>();
    arena.release(Region::Mask);
    Ok(unsafe { plan_from_base(base, &layout) })
}

/// The carve itself, applied to a base pointer.
/// # Safety: `base` must point at `layout.total` writable bytes.
unsafe fn plan_from_base(base: *mut u8, layout: &MaskSlotLayout) -> PageMaskCapturePlan {
    unsafe {
        PageMaskCapturePlan {
            keep: base.add(layout.keep_offset),
            num_requests: layout.num_requests,
            stride: layout.stride,
            out_indices: base.add(layout.indices_offset).cast::<u32>(),
            out_indptr: base.add(layout.indptr_offset).cast::<u32>(),
            out_last_lens: base.add(layout.last_lens_offset).cast::<u32>(),
        }
    }
}

/// The stream operations `FirePageMask` needs — a trait, not direct CUDA calls, so the
/// parity oracle can observe the layer loop's memset and kernel-launch calls.
pub trait MaskOps {
    /// `cudaMemsetAsync(dst, value, bytes, stream)`.
    fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize);

    /// `kernels::attn::compact_page_csr`. Gathers the fire's page table down to the
    /// kept pages; inputs are never modified, since the fire's CSR stays source of
    /// truth for the KV append and `kv_len`.
    #[allow(clippy::too_many_arguments)]
    fn compact_page_csr(
        &mut self,
        page_indices_in: *const u32,
        page_indptr_in: *const u32,
        last_page_lens_in: *const u32,
        keep: *const u8,
        scratch_counts: *mut u32,
        keep_stride: u32,
        num_requests: i32,
        page_indices_out: *mut u32,
        page_indptr_out: *mut u32,
        last_page_lens_out: *mut u32,
    );
}

/// Fire-scoped owner of the page mask and the compacted CSR it produces. An inactive
/// mask (fire wants none) answers every call with a no-op, since `begin_layer` runs
/// unconditionally and a conditional per site is where a missed one hides.
#[derive(Debug)]
pub struct FirePageMask {
    sink: Option<AttentionMaskSink>,
    out_indices: *mut u32,
    out_indptr: *mut u32,
    out_last_lens: *mut u32,
    /// Per-request survivor counts for the compaction. Acquired once per fire and
    /// reused by every layer — cheaper than an alloc/free pair at decode batch sizes.
    counts: *mut u32,
    /// Set when the [`Region::Mask`] slot is held, so [`Self::release`] knows whether
    /// it owes the arena a hand-back.
    holds_slot: bool,
}

impl FirePageMask {
    /// The inactive mask: what a fire that wants no page mask gets.
    #[must_use]
    pub const fn inactive() -> Self {
        Self {
            sink: None,
            out_indices: core::ptr::null_mut(),
            out_indptr: core::ptr::null_mut(),
            out_last_lens: core::ptr::null_mut(),
            counts: core::ptr::null_mut(),
            holds_slot: false,
        }
    }

    /// Carves the fire's five buffers out of one arena slot. A `false`
    /// `wants_page_mask` is the common case, not an error — returns [`Self::inactive`]
    /// untouched. Reuse across fires is safe since every buffer is rewritten before
    /// read, so the steady state is alloc-free.
    pub fn new<M: super::sideband_arena::DeviceMemory>(
        wants_page_mask: bool,
        geometry: Option<FireGeometry<'_>>,
        arena: Option<&mut SidebandArena>,
        mem: &mut M,
    ) -> Result<Self, MaskError> {
        if !wants_page_mask {
            return Ok(Self::inactive());
        }
        let geometry = geometry.ok_or(MaskError::NoGeometry)?;
        let layout = MaskSlotLayout::plan(geometry)?;
        let arena = arena.ok_or(MaskError::NoArena)?;
        let base = arena
            .acquire(mem, Region::Mask, layout.total)
            .map_err(MaskError::ArenaRefused)?
            .cast::<u8>();

        let plan = unsafe { plan_from_base(base, &layout) };
        Ok(Self {
            sink: Some(AttentionMaskSink {
                keep: plan.keep,
                num_requests: plan.num_requests,
                stride: plan.stride,
                written_layer: None,
            }),
            out_indices: plan.out_indices.cast_mut(),
            out_indptr: plan.out_indptr.cast_mut(),
            out_last_lens: plan.out_last_lens.cast_mut(),
            counts: unsafe { base.add(layout.counts_offset) }.cast::<u32>(),
            holds_slot: true,
        })
    }

    /// Whether this fire carries a mask at all.
    #[must_use]
    pub const fn active(&self) -> bool {
        self.sink.is_some()
    }

    /// The write destination for a layer's `OnAttnProj` sideband; `None` when the fire
    /// wants no mask.
    pub const fn sink(&mut self) -> Option<&mut AttentionMaskSink> {
        self.sink.as_mut()
    }

    /// Re-seed to "keep everything" and clear the layer tag: all-zero would evict the
    /// whole cache for a layer that chose not to score.
    pub fn begin_layer<O: MaskOps>(&mut self, ops: &mut O) {
        let Some(sink) = self.sink.as_mut() else {
            return;
        };
        ops.memset_async(
            sink.keep,
            1,
            sink.num_requests as usize * sink.stride as usize,
        );
        sink.written_layer = None;
    }

    /// Whether the sink was written *for this layer*.
    #[must_use]
    pub fn written_for(&self, layer: u32) -> bool {
        self.sink
            .as_ref()
            .is_some_and(|s| s.written_layer == Some(layer))
    }

    /// Gather the fire's page table down to the kept pages. `num_requests` must match
    /// the fire's, or the kernel (which walks `keep` by row) reads past the last row.
    pub fn compact<O: MaskOps>(
        &mut self,
        ops: &mut O,
        page_indices_d: *const u32,
        page_indptr_d: *const u32,
        last_page_lens_d: *const u32,
        num_requests: u32,
    ) -> Result<(), MaskError> {
        let Some(sink) = self.sink.as_ref() else {
            return Ok(());
        };
        if num_requests != sink.num_requests {
            return Err(MaskError::RequestCountMismatch);
        }
        ops.compact_page_csr(
            page_indices_d,
            page_indptr_d,
            last_page_lens_d,
            sink.keep,
            self.counts,
            sink.stride,
            sink.num_requests as i32,
            self.out_indices,
            self.out_indptr,
            self.out_last_lens,
        );
        Ok(())
    }

    /// The compacted page ids, valid after a [`Self::compact`].
    #[must_use]
    pub const fn page_indices(&self) -> *const u32 {
        self.out_indices
    }

    /// The compacted CSR offsets, valid after a [`Self::compact`].
    #[must_use]
    pub const fn page_indptr(&self) -> *const u32 {
        self.out_indptr
    }

    /// The compacted last-page lengths, valid after a [`Self::compact`].
    #[must_use]
    pub const fn last_page_lens(&self) -> *const u32 {
        self.out_last_lens
    }

    /// Hand the slot back for the next fire's mask to reuse; nothing is freed, the
    /// bytes belong to the arena. Not done in `Drop`, since that would need to borrow
    /// `&mut SidebandArena` for the mask's whole life; [`Self::still_holds_slot`] lets
    /// a test check the caller met the obligation.
    pub fn release(&mut self, arena: &mut SidebandArena) {
        if self.holds_slot {
            arena.release(Region::Mask);
            self.holds_slot = false;
        }
        self.sink = None;
        self.out_indices = core::ptr::null_mut();
        self.out_indptr = core::ptr::null_mut();
        self.out_last_lens = core::ptr::null_mut();
        self.counts = core::ptr::null_mut();
    }

    /// Whether this mask still owes the arena a [`Self::release`]. A leaked hold turns
    /// every later acquire into a busy refusal, far from its cause.
    #[must_use]
    pub const fn still_holds_slot(&self) -> bool {
        self.holds_slot
    }
}

impl Drop for FirePageMask {
    /// Cannot release (the arena is not reachable here); exists only to make a
    /// forgotten [`Self::release`] loud in a debug build.
    fn drop(&mut self) {
        debug_assert!(
            !self.holds_slot,
            "FirePageMask dropped while still holding the arena's mask slot; \
             call release(&mut arena) before dropping"
        );
    }
}

/// The raw base of a carve, for callers that hold the plan rather than the mask.
#[must_use]
pub const fn plan_base(plan: &PageMaskCapturePlan) -> *const c_void {
    plan.out_indices.cast::<c_void>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::sideband_arena::DeviceMemory;

    struct Slab {
        bytes: Vec<u8>,
        next: usize,
        allocs: u32,
    }

    impl Slab {
        fn new() -> Self {
            Self {
                bytes: vec![0u8; 8 << 20],
                next: 0,
                allocs: 0,
            }
        }
    }

    impl DeviceMemory for Slab {
        fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
            let start = self.next.next_multiple_of(256);
            if start + bytes > self.bytes.len() {
                return None;
            }
            self.next = start + bytes;
            self.allocs += 1;
            Some(unsafe { self.bytes.as_mut_ptr().add(start) }.cast::<c_void>())
        }
        fn free(&mut self, _ptr: *mut c_void) {}
        fn synchronize(&mut self) -> bool {
            true
        }
    }

    #[derive(Default)]
    struct Ops {
        memsets: Vec<(usize, u8, usize)>,
        compactions: u32,
        last_stride: u32,
    }

    impl MaskOps for Ops {
        fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize) {
            self.memsets.push((dst as usize, value, bytes));
        }
        fn compact_page_csr(
            &mut self,
            _a: *const u32,
            _b: *const u32,
            _c: *const u32,
            _keep: *const u8,
            _counts: *mut u32,
            keep_stride: u32,
            _n: i32,
            _d: *mut u32,
            _e: *mut u32,
            _f: *mut u32,
        ) {
            self.compactions += 1;
            self.last_stride = keep_stride;
        }
    }

    fn geometry(csr: &[u32]) -> FireGeometry<'_> {
        FireGeometry::new(csr).unwrap()
    }

    #[test]
    fn the_stride_is_the_widest_request_so_every_row_covers_its_page_list() {
        let csr = [0, 3, 3, 40, 41, 97];
        let layout = MaskSlotLayout::plan(geometry(&csr)).unwrap();
        assert_eq!(layout.num_requests, 5);
        assert_eq!(layout.stride, 56, "request 4 spans 97-41 pages");
        for r in 0..5usize {
            let pages = csr[r + 1] - csr[r];
            assert!(
                pages <= layout.stride,
                "request {r} has {pages} pages but the row holds {}",
                layout.stride
            );
        }
    }

    #[test]
    fn every_sub_buffer_is_aligned_and_none_overlaps_the_next() {
        let csr = [0, 3, 3, 40, 41, 97];
        let layout = MaskSlotLayout::plan(geometry(&csr)).unwrap();
        let requests = layout.num_requests as usize;
        let spans = [
            (layout.indices_offset, 97 * 4, "indices"),
            (layout.indptr_offset, (requests + 1) * 4, "indptr"),
            (layout.counts_offset, requests * 4, "counts"),
            (layout.last_lens_offset, requests * 4, "last_lens"),
            (layout.keep_offset, layout.keep_bytes(), "keep"),
        ];
        for (i, &(offset, size, name)) in spans.iter().enumerate() {
            assert_eq!(offset % SIDEBAND_ALIGN, 0, "{name} is misaligned");
            let end = offset + size;
            assert!(end <= layout.total, "{name} runs past the slot");
            if let Some(&(next_offset, _, next_name)) = spans.get(i + 1) {
                assert!(end <= next_offset, "{name} overlaps {next_name}");
            }
        }
    }

    #[test]
    fn a_malformed_csr_is_rejected_rather_than_producing_a_short_stride() {
        assert_eq!(
            MaskSlotLayout::plan(geometry(&[0, 9, 4, 12])),
            Err(MaskError::MalformedCsr)
        );
        assert_eq!(
            MaskSlotLayout::plan(geometry(&[0, 5, 99, 12])),
            Err(MaskError::MalformedCsr)
        );
        assert_eq!(
            MaskSlotLayout::plan(geometry(&[0, 0, 0])),
            Err(MaskError::NoPages)
        );
        assert_eq!(FireGeometry::new(&[0]).unwrap_err(), MaskError::NoGeometry);
    }

    #[test]
    fn the_prepare_pass_and_the_fire_carve_land_on_the_same_addresses() {
        // A captured graph bakes what prepare returns and new must reproduce.
        for csr in [
            vec![0u32, 1],
            vec![0, 129],
            vec![0, 8, 16, 24, 32],
            vec![0, 3, 3, 40, 41, 97],
            vec![0, 0, 7],
            vec![
                0, 2, 5, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135,
            ],
        ] {
            let mut mem = Slab::new();
            let mut arena = SidebandArena::new();
            let plan = prepare_page_mask_capture(&mut arena, &mut mem, geometry(&csr)).unwrap();
            let mut mask =
                FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem).unwrap();
            let sink = mask.sink().unwrap();
            assert_eq!(plan.keep, sink.keep, "keep moved for {csr:?}");
            assert_eq!(plan.num_requests, sink.num_requests);
            assert_eq!(plan.stride, sink.stride);
            assert_eq!(plan.out_indices, mask.page_indices(), "indices moved");
            assert_eq!(plan.out_indptr, mask.page_indptr(), "indptr moved");
            assert_eq!(plan.out_last_lens, mask.last_page_lens(), "lens moved");
            mask.release(&mut arena);
            arena.destroy(&mut mem);
        }
    }

    #[test]
    fn begin_layer_seeds_every_keep_row_in_full_and_clears_the_tag() {
        let csr = [0, 3, 3, 40, 41, 97];
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let mut ops = Ops::default();
        let mut mask =
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem).unwrap();

        mask.begin_layer(&mut ops);
        let sink = mask.sink().unwrap();
        assert_eq!(ops.memsets.len(), 1);
        assert_eq!(ops.memsets[0].0, sink.keep as usize);
        assert_eq!(ops.memsets[0].1, 1, "the safe default is keep-everything");
        assert_eq!(ops.memsets[0].2, 5 * 56, "the seed must cover every row");

        sink.written_layer = Some(7);
        assert!(mask.written_for(7));
        mask.begin_layer(&mut ops);
        assert!(
            !mask.written_for(7),
            "a new layer must not inherit the previous layer's mask"
        );
        mask.release(&mut arena);
        arena.destroy(&mut mem);
    }

    #[test]
    fn only_the_layer_the_sink_wrote_for_may_compact() {
        let csr = [0, 8, 16];
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let mut ops = Ops::default();
        let mut mask =
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem).unwrap();

        mask.begin_layer(&mut ops);
        mask.sink().unwrap().written_layer = Some(1);
        for layer in 0..3u32 {
            assert_eq!(mask.written_for(layer), layer == 1);
        }

        mask.compact(
            &mut ops,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            2,
        )
        .unwrap();
        assert_eq!(ops.compactions, 1);
        assert_eq!(ops.last_stride, 8);

        assert_eq!(
            mask.compact(
                &mut ops,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                3
            ),
            Err(MaskError::RequestCountMismatch),
            "a request-count disagreement must refuse, not launch"
        );
        assert_eq!(ops.compactions, 1, "the refused compaction must not launch");
        mask.release(&mut arena);
        arena.destroy(&mut mem);
    }

    #[test]
    fn an_inactive_mask_answers_every_call_without_touching_anything() {
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let mut ops = Ops::default();
        let mut mask = FirePageMask::new(false, None, None, &mut mem).unwrap();

        assert!(!mask.active());
        assert!(mask.sink().is_none());
        mask.begin_layer(&mut ops);
        mask.compact(
            &mut ops,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            99,
        )
        .unwrap();
        assert!(mask.page_indices().is_null());
        assert!(!mask.written_for(0));
        assert_eq!(ops.memsets.len(), 0);
        assert_eq!(ops.compactions, 0, "an inactive mask must launch nothing");
        assert_eq!(mem.allocs, 0, "and must not touch the arena");
        assert!(!mask.still_holds_slot());
        arena.destroy(&mut mem);
    }

    #[test]
    fn a_fire_that_wants_a_mask_but_has_no_arena_is_told_which_thing_is_missing() {
        let csr = [0, 8, 16];
        let mut mem = Slab::new();
        assert_eq!(
            FirePageMask::new(true, Some(geometry(&csr)), None, &mut mem).unwrap_err(),
            MaskError::NoArena
        );
        assert_eq!(
            FirePageMask::new(true, None, None, &mut mem).unwrap_err(),
            MaskError::NoGeometry,
            "geometry is checked before the arena, as in the C++"
        );
        assert_eq!(mem.allocs, 0);
    }

    #[test]
    fn the_steady_state_across_fires_allocates_nothing_and_never_moves() {
        // Once the arena has grown to the widest fire, every narrower one reuses the base.
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let widest = vec![0u32, 8000, 16000, 24000];
        let mut base = None;

        let mut warm =
            FirePageMask::new(true, Some(geometry(&widest)), Some(&mut arena), &mut mem).unwrap();
        base = base.or_else(|| Some(warm.page_indices()));
        warm.release(&mut arena);
        let allocs_after_warm = mem.allocs;

        for csr in [
            vec![0u32, 8, 16, 24, 32],
            vec![0, 4, 8],
            vec![0, 1],
            vec![0, 3, 3, 40, 41, 97],
            widest.clone(),
        ] {
            let mut mask =
                FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem).unwrap();
            assert_eq!(mask.page_indices(), base.unwrap(), "the carve moved");
            mask.release(&mut arena);
            assert!(!mask.still_holds_slot());
        }
        assert_eq!(
            mem.allocs, allocs_after_warm,
            "the steady state must allocate nothing"
        );
        arena.destroy(&mut mem);
    }

    #[test]
    fn a_leaked_hold_jams_the_next_fire_and_release_unjams_it() {
        let csr = [0, 8, 16];
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let mut first =
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem).unwrap();
        assert!(first.still_holds_slot());
        assert_eq!(
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem).unwrap_err(),
            MaskError::ArenaRefused(Refusal::Busy),
            "an overlapping fire must be refused, not handed the same buffers"
        );
        first.release(&mut arena);
        let mut second =
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem).unwrap();
        second.release(&mut arena);
        arena.destroy(&mut mem);
    }
}

/// The element mask the custom-mask attention dispatch reads — element-, not
/// page-granularity like everything above.
///
/// # The layout is BIT-packed, and this file used to think otherwise
///
/// Both CUDA kernels that read a custom mask index it the same way, and both
/// index it by BIT:
///
/// ```text
///   // kernels/flashinfer/attention/variants.cuh
///   mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
///   // kernels/attn/attention_naive_paged.cuh
///   const long long bit  = qo_off * kv_total + kv_idx;
///   const long long byte = mask_indptr[request_idx] + (bit >> 3);
///   return ((mask[byte] >> (bit & 7)) & 1) != 0;
/// ```
///
/// So `mask_d` holds `(q, kv)` as one BIT, `mask_indptr_d` counts BYTES, and a
/// request's mask begins on a byte boundary. This module published one byte per
/// pair instead, with the CSR counting pairs. The kernel then read pair `8i` as
/// the whole byte for pairs `8i..8i+8` and took bit `k % 8` of a byte that is
/// only ever `0` or `1`, so seven of every eight positions were forced closed
/// and the eighth answered for its neighbours.
///
/// Nothing caught it, because nothing READS the mask on the arm that is
/// exercised. The causal plan below is published unconditionally so the
/// unmasked arm can still be RECORDED under `GuardMode::Union` — which captures
/// both arms and aborts if either's mask was never built — but that arm's
/// kernel is compiled without custom-mask support and never dereferences it.
/// Only a fire that actually supplies a mask reaches the reading form, and the
/// one curated fixture that does (`tart-masked`) was wedged on an unrelated
/// channel-cursor defect for as long as this was here. Its answer with a mask
/// whose numerics are exactly causal was `" wore of of of.. the."`; without it,
/// `"<think>\nOkay, the user is asking"`.
pub mod element_mask {
    /// A mask this large is refused rather than published — the extent (`sum_r
    /// qo_len[r] * kv_len[r]`) grows with the context.
    const MAX_MASK_BYTES: u64 = 1 << 30;

    /// Bytes a request of `cells` `(q, kv)` pairs occupies, one bit each.
    const fn packed_len(cells: u64) -> usize {
        cells.div_ceil(8) as usize
    }

    /// Set pair `index` of the request whose mask starts at byte `base`.
    fn set_bit(mask: &mut [u8], base: usize, index: usize) {
        mask[base + (index >> 3)] |= 1 << (index & 7);
    }

    /// One fire's element mask, planned but not allocated.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ElementMaskPlan {
        /// Bytes of `mask_d`.
        pub mask_bytes: usize,
        /// Byte offset of the `num_requests + 1` i32 CSR.
        pub indptr_offset: usize,
        /// Total bytes to allocate.
        pub bytes: usize,
        /// The CSR, in mask BYTES — what the kernels add `bit >> 3` to.
        pub indptr: Vec<i32>,
        /// The mask bits themselves, one `(q, kv)` pair each, request by
        /// request, every request starting on a byte boundary.
        pub mask: Vec<u8>,
    }

    /// Plan and fill a causal element mask for the fire's geometry. `None` when there
    /// is nothing to mask or it would exceed [`MAX_MASK_BYTES`]; the custom arm
    /// declines in that case.
    #[must_use]
    pub fn plan_causal(
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        page_size: i32,
    ) -> Option<ElementMaskPlan> {
        let requests = qo_indptr_h.len().checked_sub(1)?;
        if requests == 0 || kv_page_indptr_h.len() < requests + 1 {
            return None;
        }
        let page = u32::try_from(page_size.max(0)).unwrap_or(0);
        let mut indptr = vec![0i32; requests + 1];
        let mut extents = Vec::with_capacity(requests);
        let mut total: u64 = 0;
        let mut pairs: u64 = 0;
        for r in 0..requests {
            let qo = qo_indptr_h[r + 1].saturating_sub(qo_indptr_h[r]);
            let pages = kv_page_indptr_h[r + 1].saturating_sub(kv_page_indptr_h[r]);
            let kv = if pages == 0 {
                0
            } else {
                (pages - 1) * page + kv_last_page_lens_h.get(r).copied().unwrap_or(0)
            };
            indptr[r] = i32::try_from(total).ok()?;
            extents.push((qo, kv));
            let cells = u64::from(qo) * u64::from(kv);
            pairs += cells;
            // Byte-aligned per request: the kernel adds `bit >> 3` to
            // `mask_indptr[r]`, so a request that began mid-byte would read its
            // neighbour's tail.
            total += packed_len(cells) as u64;
        }
        indptr[requests] = i32::try_from(total).ok()?;
        if pairs == 0 || total > MAX_MASK_BYTES {
            return None;
        }
        let mask_bytes = usize::try_from(total).ok()?;
        let mut mask = vec![0u8; mask_bytes];
        let mut at = 0usize;
        for &(qo, kv) in &extents {
            // Local row `qi` sits at absolute position `kv - qo + qi`, so it attends
            // every key at or before that.
            for qi in 0..qo {
                let last = kv.saturating_sub(qo) + qi;
                for ki in 0..kv {
                    if ki <= last {
                        set_bit(&mut mask, at, (qi * kv + ki) as usize);
                    }
                }
            }
            at += packed_len(u64::from(qo) * u64::from(kv));
        }
        let indptr_offset = mask_bytes.next_multiple_of(4);
        Some(ElementMaskPlan {
            mask_bytes,
            indptr_offset,
            bytes: indptr_offset + (requests + 1) * 4,
            indptr,
            mask,
        })
    }

    /// The engine's mask, repacked into the bits the launcher reads: the engine
    /// gives one bitset per query row, bit `i` whether that row attends KV
    /// position `i`, each row starting at bit 0 of its own words; the kernel
    /// wants one contiguous `qo * kv` bitset per REQUEST, byte-aligned, with
    /// the same CSR as [`plan_causal`].
    /// `None` when the fire's and table's shapes disagree — a REFUSAL, not a fallback,
    /// since serving causally would look exactly right.
    #[must_use]
    pub fn from_words(
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        page_size: i32,
        request_indptr: &[u32],
        word_indptr: &[u32],
        words: &[u32],
    ) -> Option<ElementMaskPlan> {
        let requests = qo_indptr_h.len().checked_sub(1)?;
        if requests == 0 || request_indptr.len() < requests + 1 {
            return None;
        }
        let page = u32::try_from(page_size.max(0)).unwrap_or(0);
        let mut indptr = vec![0i32; requests + 1];
        let mut mask: Vec<u8> = Vec::new();
        let mut total: u64 = 0;
        let mut pairs: u64 = 0;
        for r in 0..requests {
            let qo = qo_indptr_h[r + 1].saturating_sub(qo_indptr_h[r]) as usize;
            let pages = kv_page_indptr_h[r + 1].saturating_sub(kv_page_indptr_h[r]);
            let kv = if pages == 0 {
                0
            } else {
                (pages - 1) * page + kv_last_page_lens_h.get(r).copied().unwrap_or(0)
            } as usize;
            indptr[r] = i32::try_from(total).ok()?;
            // One mask per query row; the count must match or the table describes a different fire.
            let (lo, hi) = (request_indptr[r] as usize, request_indptr[r + 1] as usize);
            if hi.saturating_sub(lo) != qo || hi > word_indptr.len().saturating_sub(1) {
                return None;
            }
            let base = mask.len();
            mask.resize(base + packed_len((qo * kv) as u64), 0);
            for (qi, m) in (lo..hi).enumerate() {
                let (wlo, whi) = (word_indptr[m] as usize, word_indptr[m + 1] as usize);
                let row = words.get(wlo..whi)?;
                // A mask shorter than the row's KV extent can't say what the tail
                // attends, and guessing is what this refuses.
                if row.len() * 32 < kv {
                    return None;
                }
                for k in 0..kv {
                    if row[k / 32] >> (k % 32) & 1 == 1 {
                        set_bit(&mut mask, base, qi * kv + k);
                    }
                }
            }
            pairs += (qo * kv) as u64;
            total += packed_len((qo * kv) as u64) as u64;
        }
        indptr[requests] = i32::try_from(total).ok()?;
        if pairs == 0 || total > MAX_MASK_BYTES {
            return None;
        }
        let mask_bytes = usize::try_from(total).ok()?;
        debug_assert_eq!(mask.len(), mask_bytes);
        let indptr_offset = mask_bytes.next_multiple_of(4);
        Some(ElementMaskPlan {
            mask_bytes,
            indptr_offset,
            bytes: indptr_offset + (requests + 1) * 4,
            indptr,
            mask,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The kernels' own read, spelled out once so every expectation below is
        /// checked against the thing that consumes it rather than against a
        /// restatement of the packer.
        ///
        /// `attention_naive_paged.cuh`:
        /// ```text
        ///   bit  = qo_off * kv_total + kv_idx;
        ///   byte = mask_indptr[request_idx] + (bit >> 3);
        ///   ((mask[byte] >> (bit & 7)) & 1) != 0
        /// ```
        fn kernel_reads(
            p: &ElementMaskPlan,
            request: usize,
            qi: usize,
            ki: usize,
            kv: usize,
        ) -> bool {
            let bit = qi * kv + ki;
            let byte = p.indptr[request] as usize + (bit >> 3);
            (p.mask[byte] >> (bit & 7)) & 1 != 0
        }

        /// Every pair of a request, as the kernel would read them.
        fn read_all(p: &ElementMaskPlan, request: usize, qo: usize, kv: usize) -> Vec<bool> {
            (0..qo)
                .flat_map(|qi| (0..kv).map(move |ki| (qi, ki)))
                .map(|(qi, ki)| kernel_reads(p, request, qi, ki, kv))
                .collect()
        }

        #[test]
        fn a_decode_row_attends_its_whole_context() {
            let p = plan_causal(&[0, 1], &[0, 1], &[3], 16).expect("planned");
            // Three pairs is three BITS, so one byte, and the CSR counts bytes.
            assert_eq!(p.indptr, vec![0, 1]);
            assert_eq!(p.mask, vec![0b111]);
            assert_eq!(read_all(&p, 0, 1, 3), vec![true; 3]);
        }

        #[test]
        fn a_prefill_row_attends_no_further_than_itself() {
            // 3 query rows against a 3-long context: the plain lower triangle.
            let p = plan_causal(&[0, 3], &[0, 1], &[3], 16).expect("planned");
            assert_eq!(
                read_all(&p, 0, 3, 3),
                vec![true, false, false, true, true, false, true, true, true]
            );
            // Nine pairs is two bytes, and the ninth is the low bit of the second.
            assert_eq!(p.mask, vec![0b1101_1001, 0b1]);
        }

        #[test]
        fn a_continuation_attends_the_prefix_it_did_not_write() {
            // 2 new rows onto a 5-long context: rows 3 and 4 are the new ones.
            let p = plan_causal(&[0, 2], &[0, 1], &[5], 16).expect("planned");
            assert_eq!(
                read_all(&p, 0, 2, 5),
                vec![true, true, true, true, false, true, true, true, true, true]
            );
        }

        #[test]
        fn two_requests_get_their_own_bases() {
            // Two pairs and four pairs: one byte each, because a request has to
            // START on a byte boundary or the kernel's `bit >> 3` walks into its
            // neighbour's tail.
            let p = plan_causal(&[0, 1, 3], &[0, 1, 2], &[2, 2], 16).expect("planned");
            assert_eq!(p.indptr, vec![0, 1, 2]);
            assert_eq!(read_all(&p, 0, 1, 2), vec![true, true]);
            assert_eq!(read_all(&p, 1, 2, 2), vec![true, false, true, true]);
        }

        /// The engine's bitset, repacked: one decode row attending 3 KV positions
        /// is three set bits in one byte.
        #[test]
        fn a_set_bit_survives_the_repack() {
            let p = from_words(&[0, 1], &[0, 1], &[3], 16, &[0, 1], &[0, 1], &[0b111])
                .expect("decoded");
            assert_eq!(p.mask, vec![0b111]);
            assert_eq!(p.indptr, vec![0, 1]);
            assert_eq!(read_all(&p, 0, 1, 3), vec![true; 3]);
        }

        /// A CLEARED bit is a position the kernel skips — the whole point of a
        /// caller's mask, and what a causal fallback would silently undo.
        #[test]
        fn a_cleared_bit_survives_the_repack() {
            let p = from_words(&[0, 1], &[0, 1], &[4], 16, &[0, 1], &[0, 1], &[0b1011])
                .expect("decoded");
            assert_eq!(read_all(&p, 0, 1, 4), vec![true, true, false, true]);
        }

        /// A prefill's rows are its own masks; the repack CONCATENATES them into
        /// one `qo * kv` bitset, because the engine's rows each start at bit 0 of
        /// their own words and the kernel's do not.
        #[test]
        fn each_query_row_brings_its_own_mask() {
            let p = from_words(
                &[0, 2],
                &[0, 1],
                &[3],
                16,
                &[0, 2],
                &[0, 1, 2],
                &[0b001, 0b011],
            )
            .expect("decoded");
            assert_eq!(
                read_all(&p, 0, 2, 3),
                vec![true, false, false, true, true, false]
            );
            let causal = plan_causal(&[0, 2], &[0, 1], &[3], 16).expect("causal");
            assert_eq!(p.indptr, causal.indptr, "same geometry, same CSR");
        }

        /// The defect this layout was changed for: a mask whose numerics ARE
        /// causal has to read back identical to the causal plan, pair for pair.
        /// It did not — the packer wrote a byte per pair while both kernels read
        /// a bit per pair, so seven of every eight positions were forced closed.
        #[test]
        fn a_causal_custom_mask_reads_back_as_the_causal_plan() {
            // 24 query rows over 24 keys, which is `tart-masked`'s prefill: two
            // bytes' worth of row and a row length that is not a multiple of 8,
            // so every misalignment this could have shows up.
            let (qo, kv) = (24usize, 24usize);
            let words: Vec<u32> = (0..qo)
                .flat_map(|qi| {
                    let row: u32 = (0..kv).filter(|&ki| ki <= qi).map(|ki| 1u32 << ki).sum();
                    [row]
                })
                .collect();
            let word_indptr: Vec<u32> = (0..=qo as u32).collect();
            let user = from_words(
                &[0, qo as u32],
                &[0, 2],
                &[8],
                16,
                &[0, qo as u32],
                &word_indptr,
                &words,
            )
            .expect("decoded");
            let causal = plan_causal(&[0, qo as u32], &[0, 2], &[8], 16).expect("causal");
            assert_eq!(user.mask, causal.mask);
            assert_eq!(user.indptr, causal.indptr);
            assert_eq!(user.mask.len(), (qo * kv).div_ceil(8));
        }

        /// A `set` never spills into the neighbouring pair, which one byte per
        /// pair could not get wrong and one bit per pair can.
        #[test]
        fn a_single_open_position_opens_exactly_one() {
            let p = from_words(&[0, 1], &[0, 1], &[9], 16, &[0, 1], &[0, 1], &[1 << 8])
                .expect("decoded");
            assert_eq!(
                read_all(&p, 0, 1, 9),
                vec![false, false, false, false, false, false, false, false, true]
            );
        }

        #[test]
        fn an_empty_fire_publishes_nothing() {
            assert!(plan_causal(&[0], &[0], &[], 16).is_none());
            assert!(plan_causal(&[0, 1], &[0, 0], &[0], 16).is_none());
        }
    }
}

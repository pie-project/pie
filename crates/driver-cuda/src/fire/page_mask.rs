//! The page mask: `FirePageMask` and the hook-graph prepare pass that has to
//! predict it exactly.
//!
//! Port of `driver-cuda/csrc/src/model/attn_page_mask.{hpp,cu}`.
//!
//! # What the object is
//!
//! `attn_page_mask` is the one PTIR sideband that flows *into* the backend: a
//! program hands the model a per-page keep mask and the layer's attention is
//! expected to honour it. Honouring it is a page-table gather, not a kernel
//! change — FlashInfer already takes the page list as a launch argument — so
//! this type owns the mask, owns the gathered CSR the gather produces, and
//! owns nothing else.
//!
//! One instance brackets a whole layer loop. The page geometry belongs to the
//! fire, not to the layer, so the five buffers are carved once and every layer
//! reuses them.
//!
//! # The invariant this module is built around
//!
//! Two separate call paths compute the carve:
//!
//! - [`FirePageMask::new`], at fire time; and
//! - [`prepare_page_mask_capture`], during the hook-graph prepare pass, which
//!   **bakes the addresses it computes into a captured CUDA graph**.
//!
//! If those two disagree by one byte, the replayed graph writes its compacted
//! page table where the attention does not read, and the model attends over
//! stale pages with no error anywhere. So — exactly as the C++ does — there is
//! **one** layout function, [`MaskSlotLayout::plan`], and both paths call it.
//! The parity oracle records both carves for all eleven fire geometries and
//! compares them row by row; see `tests/page_mask_parity.rs`.
//!
//! This is the same shape as the `workspace_bytes` bug (`.wiki/kernel-refactor`
//! §8), where two hand-written walks of one layout drifted. The C++ here got it
//! right by construction, and the port keeps that property rather than
//! re-deriving it.
//!
//! # Why the keep rows have a fixed stride
//!
//! The obvious layout — one entry per page, sliced by the fire's page CSR —
//! forces the writer (host) and the reader (a device kernel walking the real
//! page table) to agree on that CSR, and they do not. On the decode-envelope
//! path the host holds a conservative *bound* while the device resolves the
//! real geometry itself. A per-request stride removes the page CSR from the
//! mask's addressing entirely: the only shared fact left is "slot `p` of
//! request `r`", which is exactly what a program means when it writes
//! `mask[p]`.
//!
//! A conservative host CSR therefore only ever over-allocates. It cannot
//! mis-address anything.

use core::ffi::c_void;

use super::sideband_arena::{Refusal, Region, SidebandArena};

/// Sub-buffer alignment inside the arena's mask slot.
///
/// Mirrors `attn_score.cu`. 256 also happens to be `cudaMalloc`'s guarantee,
/// so an offset that is a multiple of it lands aligned for any of the four
/// element types carved here.
pub const SIDEBAND_ALIGN: usize = 256;

const fn align_up(n: usize) -> usize {
    n.next_multiple_of(SIDEBAND_ALIGN)
}

/// Why a fire cannot carry a page mask.
///
/// The C++ throws `std::runtime_error` with a message for the first two and
/// returns a default-constructed plan for all of them; splitting them out
/// lets a caller tell "this fire has no pages" (routine — a first-token fire
/// has none) from "this fire's CSR is malformed" (a bug upstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskError {
    /// The observation is missing geometry the carve needs. C++ message:
    /// `attn_page_mask needs a fire with kv geometry`.
    NoGeometry,
    /// The fire has no KV pages at all, so there is nothing to mask.
    NoPages,
    /// A request's page range runs backwards or past the end of the table.
    MalformedCsr,
    /// A compaction was asked for a different request count than the fire
    /// carries. Distinct from [`Self::MalformedCsr`] because the CSR is fine —
    /// the *caller* is out of step, which is a different bug in a different
    /// place.
    RequestCountMismatch,
    /// The fire carries no `sideband_arena` to carve from.
    NoArena,
    /// The arena refused the acquire.
    ArenaRefused(Refusal),
}

impl MaskError {
    /// The C++ `what()` string this corresponds to, or `None` where the C++
    /// reports the failure by returning a null pointer instead of throwing.
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

/// The fire geometry the carve reads.
///
/// This is the sliver of `AttentionObservation` that `attn_page_mask` touches:
/// the host page CSR, and nothing else. The device CSR is deliberately absent
/// — the host copy *sizes* the rows and never addresses them, and a type that
/// cannot see the device pointers cannot accidentally address with the host
/// ones.
#[derive(Debug, Clone, Copy)]
pub struct FireGeometry<'a> {
    /// Host page CSR: `num_requests + 1` entries, a *bound* on the real
    /// per-request page counts.
    pub kv_page_indptr_h: &'a [u32],
}

impl<'a> FireGeometry<'a> {
    /// Wraps a host page CSR, rejecting one too short to describe a fire.
    ///
    /// The C++ gates on `AttentionObservation::usable()`, which additionally
    /// requires six device pointers to be non-null. Those are not the carve's
    /// business, so the port checks the one thing that is: a CSR with fewer
    /// than two entries describes zero requests, which is `usable()`'s
    /// `num_requests > 0`.
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

/// The mask slot's size and the offsets of the five buffers inside it.
///
/// **The single definition of the carve.** Both `FirePageMask::new` and
/// `prepare_page_mask_capture` go through here; see the module docs for why
/// there must not be a second one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskSlotLayout {
    /// Requests in the fire; the keep rows and both length arrays are this
    /// long.
    pub num_requests: u32,
    /// Entries per keep row: the widest request's page count, so every row is
    /// at least as long as the page list it governs.
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
    /// The `[num_requests, stride]` u8 keep rows. Last, because u8 is the one
    /// element type here with no alignment requirement.
    pub keep_offset: usize,
}

impl MaskSlotLayout {
    /// Sizes and carves the mask slot for a fire, or says why it cannot.
    ///
    /// The order — u32 outputs first, u8 keep rows last, each aligned — is not
    /// arbitrary: `keep` is the only sub-buffer whose element type has no
    /// alignment requirement, so putting it last is what lets the total be
    /// `keep_offset + keep_bytes` with no trailing pad.
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

    /// Bytes `begin_layer` must seed. Every keep row in full — a seed short by
    /// one row leaves a stale mask governing a request, which evicts pages the
    /// program never asked to evict.
    #[must_use]
    pub const fn keep_bytes(&self) -> usize {
        self.num_requests as usize * self.stride as usize
    }
}

/// The write side of the attention hook: `[num_requests, stride]` u8,
/// row-major, 1 keeps the page.
///
/// Entry `[r, p]` governs slot `p` of request `r`'s page list. Pre-filled with
/// 1 before every hook, so a program that emits no sink for a layer leaves
/// that layer's attention unrestricted rather than evicting everything.
#[derive(Debug)]
pub struct AttentionMaskSink {
    /// The `[num_requests, stride]` u8 rows, row-major.
    pub keep: *mut u8,
    /// Rows in `keep`.
    pub num_requests: u32,
    /// Entries per row: an upper bound on any request's page count.
    pub stride: u32,
    /// Layer whose sink last wrote `keep`, or `None` for "nothing written".
    ///
    /// The C++ spells this `int written_layer = -1`. The tag is what stops a
    /// mask computed for layer L from silently governing layer L+1 when the
    /// program stops emitting the sink — the same stale-view guard the layer
    /// tag on `AttentionScores` provides.
    pub written_layer: Option<u32>,
}

impl AttentionMaskSink {
    /// Whether the sink actually points at rows a program could write.
    #[must_use]
    pub fn usable(&self) -> bool {
        !self.keep.is_null() && self.num_requests > 0 && self.stride > 0
    }
}

/// What a captured hook body needs to know before it exists.
///
/// Returned by [`prepare_page_mask_capture`]. Every pointer here is baked into
/// the graph, so all of them must equal what the fire-time carve produces.
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

/// The prepare pass: pre-grow the mask slot and report where it carves.
///
/// **Acquire-and-release.** Growth is a stream-synchronised free-and-realloc
/// and must not happen inside a captured region, so it is pulled forward to
/// here; the capture-time [`FirePageMask::new`] then finds sufficient capacity
/// and its own acquire is a host-side pointer return with no stream work at
/// all. The slot is released immediately because this pass holds nothing.
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
///
/// # Safety
/// `base` must point at `layout.total` writable bytes.
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

/// The stream operations `FirePageMask` needs.
///
/// A trait rather than direct CUDA calls for the same reason
/// [`super::sideband_arena::DeviceMemory`] is one: it makes the layer loop's
/// device traffic — which is exactly two calls, a memset and a kernel launch —
/// observable, so the parity oracle can check the *extent* of the seed and
/// which carved buffer landed in which kernel parameter.
pub trait MaskOps {
    /// `cudaMemsetAsync(dst, value, bytes, stream)`.
    fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize);

    /// `kernels::attn::compact_page_csr`.
    ///
    /// Gathers the fire's page table down to the kept pages. The inputs are
    /// never modified: the fire's CSR remains the source of truth for the KV
    /// append and for `kv_len`, and compacting it in place would corrupt the
    /// cache.
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

/// Fire-scoped owner of the page mask and of the compacted CSR it produces.
///
/// Per layer:
///
/// ```text
/// mask.begin_layer(ops);                 // re-seed to "keep everything"
/// invoke_stage_hook(.., mask.sink());    // the program's sink may write
/// if mask.written_for(layer) { mask.compact(..); }
/// // ... attention, over mask.page_indices() when compacted ...
/// ```
///
/// An **inactive** mask — the fire wants none — answers every call with a
/// no-op, because the layer loop calls `begin_layer` unconditionally and a
/// conditional at every site is where a missed one hides.
#[derive(Debug)]
pub struct FirePageMask {
    sink: Option<AttentionMaskSink>,
    out_indices: *mut u32,
    out_indptr: *mut u32,
    out_last_lens: *mut u32,
    /// Per-request survivor counts for the compaction. Acquired once per fire
    /// and reused by all ~28 layers: this runs once per layer, and an
    /// alloc/free pair costs more than both compaction kernels put together at
    /// decode batch sizes.
    counts: *mut u32,
    /// Set when the [`Region::Mask`] slot is held, so [`Self::release`] knows
    /// whether it owes the arena a hand-back.
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

    /// Carves the fire's five buffers out of one arena slot.
    ///
    /// `wants_page_mask` is the launch's own answer to "does any program write
    /// the sink"; a `false` here is not an error, it is the common case, and
    /// it returns [`Self::inactive`] without touching the arena.
    ///
    /// Reuse across fires is safe because every buffer is rewritten before it
    /// is read — `begin_layer` re-seeds `keep` each layer, and `compact`
    /// writes the outputs before attention reads them. Nothing here needs a
    /// fresh-allocation guarantee, which is what makes the whole steady state
    /// allocation-free.
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

    /// The write destination for a layer's `OnAttnProj` sideband; `None` when
    /// the fire wants no mask.
    pub const fn sink(&mut self) -> Option<&mut AttentionMaskSink> {
        self.sink.as_mut()
    }

    /// Re-seed to "keep everything" and clear the layer tag.
    ///
    /// An all-ones seed is the only safe default: an all-zero one would evict
    /// the entire cache for any layer whose policy chose not to score.
    pub fn begin_layer<O: MaskOps>(&mut self, ops: &mut O) {
        let Some(sink) = self.sink.as_mut() else {
            return;
        };
        ops.memset_async(sink.keep, 1, sink.num_requests as usize * sink.stride as usize);
        sink.written_layer = None;
    }

    /// Whether the sink was written *for this layer*.
    #[must_use]
    pub fn written_for(&self, layer: u32) -> bool {
        self.sink
            .as_ref()
            .is_some_and(|s| s.written_layer == Some(layer))
    }

    /// Gather the fire's page table down to the kept pages.
    ///
    /// `num_requests` must match the fire's: the kernel walks `keep` by row,
    /// and a disagreement would read past the last row.
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

    /// Hand the slot back for the next fire's mask to reuse.
    ///
    /// Nothing is freed: the bytes belong to the arena.
    ///
    /// The C++ does this in `~FirePageMask`. The port cannot, for the same
    /// reason [`SidebandArena`] cannot free in `Drop` — the arena is not owned
    /// here, and a `Drop` that needed `&mut SidebandArena` would have to hold
    /// a borrow for the mask's whole life, which is exactly the borrow the
    /// layer loop needs for everything else. Calling this is therefore the
    /// caller's obligation; [`Self::still_holds_slot`] exists so a test can
    /// prove the caller met it.
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

    /// Whether this mask still owes the arena a [`Self::release`].
    ///
    /// A leaked hold is not a leak of memory — it turns every subsequent
    /// fire's acquire into a busy refusal, which surfaces as a fire that
    /// cannot mask rather than as anything resembling its cause.
    #[must_use]
    pub const fn still_holds_slot(&self) -> bool {
        self.holds_slot
    }
}

impl Drop for FirePageMask {
    /// Cannot release — the arena is not reachable from here. This exists only
    /// to make a forgotten [`Self::release`] loud in a debug build instead of
    /// silently jamming the next fire.
    fn drop(&mut self) {
        debug_assert!(
            !self.holds_slot,
            "FirePageMask dropped while still holding the arena's mask slot; \
             call release(&mut arena) before dropping"
        );
    }
}

/// The raw base of a carve, for callers that hold the plan rather than the
/// mask.
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
        // The whole reason both functions exist. A captured graph bakes what
        // the first one returns and the second one must reproduce it.
        for csr in [
            vec![0u32, 1],
            vec![0, 129],
            vec![0, 8, 16, 24, 32],
            vec![0, 3, 3, 40, 41, 97],
            vec![0, 0, 7],
            vec![0, 2, 5, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135],
        ] {
            let mut mem = Slab::new();
            let mut arena = SidebandArena::new();
            let plan =
                prepare_page_mask_capture(&mut arena, &mut mem, geometry(&csr)).unwrap();
            let mut mask =
                FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem)
                    .unwrap();
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
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem)
                .unwrap();

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
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem)
                .unwrap();

        mask.begin_layer(&mut ops);
        mask.sink().unwrap().written_layer = Some(1);
        for layer in 0..3u32 {
            assert_eq!(mask.written_for(layer), layer == 1);
        }

        mask.compact(&mut ops, core::ptr::null(), core::ptr::null(), core::ptr::null(), 2)
            .unwrap();
        assert_eq!(ops.compactions, 1);
        assert_eq!(ops.last_stride, 8);

        assert_eq!(
            mask.compact(&mut ops, core::ptr::null(), core::ptr::null(), core::ptr::null(), 3),
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
        mask.compact(&mut ops, core::ptr::null(), core::ptr::null(), core::ptr::null(), 99)
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
        // The graph-capture precondition, end to end. Once the arena has grown
        // to the widest fire, every narrower one reuses the same base.
        let mut mem = Slab::new();
        let mut arena = SidebandArena::new();
        let widest = vec![0u32, 8000, 16000, 24000];
        let mut base = None;

        let mut warm = FirePageMask::new(
            true,
            Some(geometry(&widest)),
            Some(&mut arena),
            &mut mem,
        )
        .unwrap();
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
                FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem)
                    .unwrap();
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
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem)
                .unwrap();
        assert!(first.still_holds_slot());
        assert_eq!(
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem)
                .unwrap_err(),
            MaskError::ArenaRefused(Refusal::Busy),
            "an overlapping fire must be refused, not handed the same buffers"
        );
        first.release(&mut arena);
        let mut second =
            FirePageMask::new(true, Some(geometry(&csr)), Some(&mut arena), &mut mem)
                .unwrap();
        second.release(&mut arena);
        arena.destroy(&mut mem);
    }
}

/// The ELEMENT mask the custom-mask attention dispatch reads — a different
/// object from everything above, which is page-granularity.
///
/// FlashInfer's custom dispatch takes `mask_d` as one byte per `(q, kv)`
/// pair and `mask_indptr_d` as the per-request base into it. This is the
/// resident, always-published form: a plain causal mask, which is what the
/// unmasked arm computes anyway, so a fire that takes the custom arm with
/// nothing else staged gets the same answer as the fire that does not.
///
/// It exists so the arm can be RECORDED. Under `GuardMode::Union` both
/// arms of `HasCustomMask` are captured whether this fire takes either,
/// and an arm whose mask was never built aborts the whole recording —
/// which is why the union used to decline every lowering mentioning
/// `_custom`. See `.wiki/driver/graph.md` §5 ①.
pub mod element_mask {
    /// A mask this large is refused rather than published: the extent is
    /// `sum_r qo_len[r] * kv_len[r]`, which grows with the context.
    const MAX_MASK_BYTES: u64 = 1 << 30;

    /// One fire's element mask, planned but not allocated.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ElementMaskPlan {
        /// Bytes of `mask_d`.
        pub mask_bytes: usize,
        /// Byte offset of the `num_requests + 1` i32 CSR.
        pub indptr_offset: usize,
        /// Total bytes to allocate.
        pub bytes: usize,
        /// The CSR, in mask ELEMENTS.
        pub indptr: Vec<i32>,
        /// The mask bytes themselves — causal, request by request.
        pub mask: Vec<u8>,
    }

    /// Plan and fill a causal element mask for the fire's geometry.
    ///
    /// `None` when there is nothing to mask or the mask would exceed
    /// [`MAX_MASK_BYTES`], in which case the pointers stay null and the
    /// custom arm declines as it always did.
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
            total += u64::from(qo) * u64::from(kv);
        }
        indptr[requests] = i32::try_from(total).ok()?;
        if total == 0 || total > MAX_MASK_BYTES {
            return None;
        }
        let mask_bytes = usize::try_from(total).ok()?;
        let mut mask = vec![0u8; mask_bytes];
        let mut at = 0usize;
        for &(qo, kv) in &extents {
            // The query at local row `qi` sits at absolute position
            // `kv - qo + qi`, so it attends every key at or before that.
            for qi in 0..qo {
                let last = kv.saturating_sub(qo) + qi;
                for ki in 0..kv {
                    mask[at + (qi * kv + ki) as usize] = u8::from(ki <= last);
                }
            }
            at += (qo * kv) as usize;
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

    /// The ENGINE'S mask, unpacked into the bytes the launcher reads.
    ///
    /// `brle` never needed porting: the engine decodes its own BRLE runs
    /// host-side (`MaskWordsStorage::from_plan`) and ships a plain packed
    /// bitset. One mask per QUERY ROW -- `request_indptr` partitions the
    /// masks by request, `word_indptr` the `u32` words by mask, and bit
    /// `i` of a mask says whether that row attends KV position `i`.
    ///
    /// FlashInfer wants one BYTE per `(q, kv)` pair, so this is a widen
    /// and a relayout, and the CSR it produces is the same one
    /// [`plan_causal`] produces for the same geometry -- which is what
    /// lets the two be published through one path.
    ///
    /// `None` when the fire's shape and the table's disagree, which is a
    /// REFUSAL rather than a fallback: a fire asked to attend over a
    /// caller's mask and served causally instead returns an answer that
    /// looks exactly like a correct one.
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
        for r in 0..requests {
            let qo = qo_indptr_h[r + 1].saturating_sub(qo_indptr_h[r]) as usize;
            let pages = kv_page_indptr_h[r + 1].saturating_sub(kv_page_indptr_h[r]);
            let kv = if pages == 0 {
                0
            } else {
                (pages - 1) * page + kv_last_page_lens_h.get(r).copied().unwrap_or(0)
            } as usize;
            indptr[r] = i32::try_from(total).ok()?;
            // One mask per query row, and the count has to match or the
            // table is describing a different fire.
            let (lo, hi) = (request_indptr[r] as usize, request_indptr[r + 1] as usize);
            if hi.saturating_sub(lo) != qo || hi > word_indptr.len().saturating_sub(1) {
                return None;
            }
            for m in lo..hi {
                let (wlo, whi) = (word_indptr[m] as usize, word_indptr[m + 1] as usize);
                let row = words.get(wlo..whi)?;
                // A mask shorter than the row's KV extent cannot say what
                // the tail attends, and guessing is the thing this refuses.
                if row.len() * 32 < kv {
                    return None;
                }
                mask.extend((0..kv).map(|k| u8::from(row[k / 32] >> (k % 32) & 1 == 1)));
            }
            total += (qo * kv) as u64;
        }
        indptr[requests] = i32::try_from(total).ok()?;
        if total == 0 || total > MAX_MASK_BYTES {
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

        #[test]
        fn a_decode_row_attends_its_whole_context() {
            let p = plan_causal(&[0, 1], &[0, 1], &[3], 16).expect("planned");
            assert_eq!(p.indptr, vec![0, 3]);
            assert_eq!(p.mask, vec![1, 1, 1]);
        }

        #[test]
        fn a_prefill_row_attends_no_further_than_itself() {
            // 3 query rows against a 3-long context: the plain lower triangle.
            let p = plan_causal(&[0, 3], &[0, 1], &[3], 16).expect("planned");
            assert_eq!(p.mask, vec![1, 0, 0, 1, 1, 0, 1, 1, 1]);
        }

        #[test]
        fn a_continuation_attends_the_prefix_it_did_not_write() {
            // 2 new rows onto a 5-long context: rows 3 and 4 are the new ones.
            let p = plan_causal(&[0, 2], &[0, 1], &[5], 16).expect("planned");
            assert_eq!(p.mask, vec![1, 1, 1, 1, 0, 1, 1, 1, 1, 1]);
        }

        #[test]
        fn two_requests_get_their_own_bases() {
            let p = plan_causal(&[0, 1, 3], &[0, 1, 2], &[2, 2], 16).expect("planned");
            assert_eq!(p.indptr, vec![0, 2, 6]);
        }

        /// The engine's bitset, widened. One decode row attending its
        /// whole 3-long context is three set bits and three bytes.
        #[test]
        fn a_set_bit_becomes_a_kept_byte() {
            let p = from_words(&[0, 1], &[0, 1], &[3], 16, &[0, 1], &[0, 1], &[0b111])
                .expect("decoded");
            assert_eq!(p.mask, vec![1, 1, 1]);
            assert_eq!(p.indptr, vec![0, 3]);
        }

        /// And a CLEARED bit is a byte the kernel skips -- which is the
        /// whole point of a caller's mask, and the thing a causal
        /// fallback would silently undo.
        #[test]
        fn a_cleared_bit_becomes_a_dropped_byte() {
            let p = from_words(&[0, 1], &[0, 1], &[4], 16, &[0, 1], &[0, 1], &[0b1011])
                .expect("decoded");
            assert_eq!(p.mask, vec![1, 1, 0, 1]);
        }

        /// A prefill's rows are its own masks, laid out row-major -- the
        /// same extent `plan_causal` produces, so the two publish through
        /// one path.
        #[test]
        fn each_query_row_brings_its_own_mask() {
            let p = from_words(
                &[0, 2], &[0, 1], &[3], 16,
                &[0, 2], &[0, 1, 2], &[0b001, 0b011],
            )
            .expect("decoded");
            assert_eq!(p.mask, vec![1, 0, 0, 1, 1, 0]);
            let causal = plan_causal(&[0, 2], &[0, 1], &[3], 16).expect("causal");
            assert_eq!(p.indptr, causal.indptr, "same geometry, same CSR");
        }

        /// A table describing a different fire is REFUSED. Serving it
        /// causally returns an answer that looks exactly like a correct
        /// one, which is the failure this whole path exists to avoid.
        #[test]
        fn a_table_that_does_not_describe_this_fire_is_refused() {
            // Two query rows, one mask.
            assert!(
                from_words(&[0, 2], &[0, 1], &[3], 16, &[0, 1], &[0, 1], &[0b111]).is_none(),
                "one mask cannot serve two rows"
            );
            // A mask too short to say what the row's tail attends.
            assert!(
                from_words(&[0, 1], &[0, 3], &[16], 16, &[0, 1], &[0, 1], &[0b1]).is_none(),
                "32 bits cannot describe 48 KV positions"
            );
        }

        #[test]
        fn an_empty_fire_publishes_nothing() {
            assert!(plan_causal(&[0], &[0], &[], 16).is_none());
            assert!(plan_causal(&[0, 1], &[0, 0], &[0], 16).is_none());
        }
    }
}

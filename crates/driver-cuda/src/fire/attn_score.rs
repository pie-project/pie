//! One layer's attention scores, published for the duration of that layer's
//! `OnAttn` hook. Ports `model/attn_score.cu`. Capture and prepare share one
//! caller-owned [`ScoreScratch`] so replay reads the same bytes.

// stderr is the C++'s refusal channel for these messages.
#![allow(clippy::print_stderr)]


use crate::fire::sideband_arena::{DeviceMemory, Region, SidebandArena};

/// One layer's attention scores — the read side of `AttnScore`. `values` is
/// ragged and head-folded: request `r` occupies `[offsets_h[r], offsets_h[r+1])`.
/// `layer` guards a consumer against silently scoring the wrong layer.
#[derive(Debug, Clone, Copy)]
pub struct AttentionScores {
    /// The folded rows, device side.
    pub values: *const f32,
    /// Host offsets, `num_requests + 1` entries, in ELEMENTS of `values`.
    pub offsets_h: *const u32,
    /// Requests in the fire.
    pub num_requests: u32,
    /// The layer the rows describe.
    pub layer: u32,
}

impl AttentionScores {
    /// The C++ `usable()`.
    #[must_use]
    pub fn usable(&self) -> bool {
        !self.values.is_null() && !self.offsets_h.is_null() && self.num_requests > 0
    }
}

/// What a PTIR attention-stage program observes about the fire — the subset the
/// score captures read, with the same seven-pointer `usable()` gate.
/// `kv_page_size` is `None` for a null cache, not `kv->page_size()` directly.
#[derive(Debug, Clone, Copy)]
pub struct AttentionObservation<'a> {
    /// `kv->page_size()`, or `None` for a null cache pointer.
    pub kv_page_size: Option<i32>,
    /// Fire CSR page ids, device side.
    pub kv_page_indices_d: *const u32,
    /// Fire CSR, device side, EXACT.
    pub kv_page_indptr_d: *const u32,
    /// Last-page lengths, device side.
    pub kv_last_page_lens_d: *const u32,
    /// Query offsets, host side.
    pub qo_indptr_h: Option<&'a [u32]>,
    /// Page CSR, host side — a BOUND, not the truth.
    pub kv_page_indptr_h: Option<&'a [u32]>,
    /// Last-page lengths, host side.
    pub kv_last_page_lens_h: Option<&'a [u32]>,
    /// Requests in the fire.
    pub num_requests: i32,
    /// Token rows in the fire.
    pub total_tokens: i32,
}

impl AttentionObservation<'_> {
    /// The C++ `usable()`: all seven pointers plus a positive request count.
    #[must_use]
    pub fn usable(&self) -> bool {
        self.kv_page_size.is_some()
            && !self.kv_page_indices_d.is_null()
            && !self.kv_page_indptr_d.is_null()
            && !self.kv_last_page_lens_d.is_null()
            && self.qo_indptr_h.is_some()
            && self.kv_page_indptr_h.is_some()
            && self.kv_last_page_lens_h.is_some()
            && self.num_requests > 0
    }
}

/// The three `StageHooks` fields the score captures read. The arena travels
/// separately since Rust needs it `&mut` for the slot acquire.
#[derive(Debug, Clone, Copy)]
pub struct ScoreHookView<'a> {
    /// Does any program in the launch read `AttnScore`?
    pub wants_attn_score: bool,
    /// The fire's observation, while a body runs.
    pub observation: Option<&'a AttentionObservation<'a>>,
}

/// The stream ops a capture issues — recorders in the parity test, CUDA in the
/// real driver. Methods, not free functions, so tests can swap them at once.
pub trait ScoreOps {
    /// `cudaMemsetAsync` over the folded rows.
    fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize);
    /// The CSR upload (`cudaMemcpyAsync`, host to device).
    fn upload_csr(&mut self, dst: *mut i32, src: &[i32]);
    /// `kernels::attn::attn_score_normalize`, the decode divide-by-total.
    /// In place: `scores` is read and written by the same block.
    #[allow(clippy::too_many_arguments)]
    fn normalize_decode(
        &mut self,
        scores: *mut f32,
        score_indptr_d: *const i32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
    );
    /// `kernels::attn::attn_prefill_score_normalize` — the PREFILL
    /// capture's, which additionally strides the observation window.
    #[allow(clippy::too_many_arguments)]
    fn normalize_prefill(
        &mut self,
        scores: *mut f32,
        score_indptr_d: *const i32,
        qo_indptr_d: *const u32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    );
    /// `attn_prefill_score_fold` — the prefill fold, collapsing heads AND window
    /// rows into the published row. Not in place: it reads `scores`, writes `folded`.
    #[allow(clippy::too_many_arguments)]
    fn fold_prefill(
        &mut self,
        scores: *const f32,
        folded: *mut f32,
        score_indptr_d: *const i32,
        qo_indptr_d: *const u32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    );
    /// `kernels::attn::attn_score_fold_heads`.
    #[allow(clippy::too_many_arguments)]
    fn fold_heads(
        &mut self,
        raw: *const f32,
        score_indptr_d: *const i32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        folded: *mut f32,
    );
}

/// The live [`ScoreOps`]. Memset and CSR upload are stream-ordered on the fire's
/// stream; the CSR source is pageable host memory the runtime staging-copies.
#[cfg(feature = "_cuda")]
#[derive(Debug, Clone, Copy)]
pub struct LiveScoreOps {
    stream: *mut std::ffi::c_void,
}

#[cfg(feature = "_cuda")]
impl LiveScoreOps {
    /// Ops ordered on the fire's stream.
    #[must_use]
    pub const fn new(stream: *mut std::ffi::c_void) -> Self {
        Self { stream }
    }
}

/// Refuse loudly for any of the four launches: one that silently didn't happen
/// leaves the memset pattern behind, read as a plausible zero row not a fault.
#[cfg(feature = "_cuda")]
fn or_panic(what: &str, fired: Result<(), kernels::Refusal>) {
    if let Err(why) = fired {
        panic!("{what}: {why}");
    }
}

/// A context on the fire's stream, for one launch.
///
/// # Safety
///
/// The caller of `publish` must hold the fire's stream live across the launch.
#[cfg(feature = "_cuda")]
fn ctx_on<'a>(stream: *mut std::ffi::c_void) -> kernels_cuda::jit::Ctx<'a> {
    // SAFETY: as stated above.
    unsafe { kernels_cuda::jit::Ctx::on(stream) }
}

#[cfg(feature = "_cuda")]
impl ScoreOps for LiveScoreOps {
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize) {
        use cudarc::runtime::sys::{cudaError, cudaMemsetAsync};
        let code =
            unsafe { cudaMemsetAsync(dst.cast(), i32::from(value), bytes, self.stream.cast()) };
        assert!(code == cudaError::cudaSuccess, "cudaMemsetAsync: {code:?}");
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn upload_csr(&mut self, dst: *mut i32, src: &[i32]) {
        use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
        let code = unsafe {
            cudaMemcpyAsync(
                dst.cast(),
                src.as_ptr().cast(),
                std::mem::size_of_val(src),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream.cast(),
            )
        };
        assert!(code == cudaError::cudaSuccess, "cudaMemcpyAsync: {code:?}");
    }

    /// The decode normalize (divide-by-total). `kv_len` comes from the page
    /// CSR in the body, not a parameter. Returns early for `num_requests` or
    /// `num_q_heads` of 0: an empty fire is a legal no-op, not a panic.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn normalize_decode(
        &mut self,
        scores: *mut f32,
        score_indptr_d: *const i32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
    ) {
        use kernels_cuda::attn::attention_score_post::attn_score_normalize;

        if num_requests <= 0 || num_q_heads <= 0 {
            return;
        }
        assert!(
            !scores.is_null() && !score_indptr_d.is_null(),
            "attn_score_normalize: scores and score_indptr must be device pointers \
             (scores={scores:?}, indptr={score_indptr_d:?})"
        );

        or_panic(
            "attn_score_normalize",
            attn_score_normalize(
                &ctx_on(self.stream),
                scores,
                score_indptr_d,
                kv_page_indptr_d,
                kv_last_page_lens_d,
                page_size,
                num_requests,
                num_q_heads,
            ),
        );
    }

    /// The prefill normalize. `window` is BOTH the third grid extent
    /// (`blockIdx.z` selects the window row) and an operand bounding
    /// `rows = min(window, qo_len)`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn normalize_prefill(
        &mut self,
        scores: *mut f32,
        score_indptr_d: *const i32,
        qo_indptr_d: *const u32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    ) {
        use kernels_cuda::attn::attention_score_post::attn_prefill_score_normalize;

        if num_requests <= 0 || num_q_heads <= 0 || window <= 0 {
            return;
        }
        assert!(
            !scores.is_null() && !score_indptr_d.is_null() && !qo_indptr_d.is_null(),
            "attn_prefill_score_normalize: scores, score_indptr and qo_indptr must be \
             device pointers (scores={scores:?}, indptr={score_indptr_d:?}, \
             qo={qo_indptr_d:?})"
        );

        or_panic(
            "attn_prefill_score_normalize",
            attn_prefill_score_normalize(
                &ctx_on(self.stream),
                scores,
                score_indptr_d,
                qo_indptr_d,
                kv_page_indptr_d,
                kv_last_page_lens_d,
                page_size,
                num_requests,
                num_q_heads,
                window,
            ),
        );
    }

    /// The prefill fold. `num_q_heads` is an OPERAND here but a grid extent in the
    /// normalize above: it collapses the head axis rather than indexing it.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn fold_prefill(
        &mut self,
        scores: *const f32,
        folded: *mut f32,
        score_indptr_d: *const i32,
        qo_indptr_d: *const u32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    ) {
        use kernels_cuda::attn::attention_score_post::attn_prefill_score_fold;

        if num_requests <= 0 {
            return;
        }
        assert!(
            !scores.is_null()
                && !folded.is_null()
                && !score_indptr_d.is_null()
                && !qo_indptr_d.is_null(),
            "attn_prefill_score_fold: scores, folded, score_indptr and qo_indptr must all \
             be device pointers (scores={scores:?}, folded={folded:?}, \
             indptr={score_indptr_d:?}, qo={qo_indptr_d:?})"
        );

        or_panic(
            "attn_prefill_score_fold",
            attn_prefill_score_fold(
                &ctx_on(self.stream),
                scores,
                folded,
                score_indptr_d,
                qo_indptr_d,
                kv_page_indptr_d,
                kv_last_page_lens_d,
                page_size,
                num_requests,
                num_q_heads,
                window,
            ),
        );
    }

    /// The fold. `num_requests <= 0` returns early — an empty fire is a legal
    /// no-op, not a refusal. A null buffer PANICS: a fold that did not run
    /// leaves `folded` holding the memset pattern, read as a plausible zero row.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn fold_heads(
        &mut self,
        raw: *const f32,
        score_indptr_d: *const i32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        folded: *mut f32,
    ) {
        use kernels_cuda::attn::attention_flashinfer::attn_score_fold_heads;

        if num_requests <= 0 {
            return;
        }
        // The launcher's throw, as a refusal that cannot be mistaken for a fold.
        assert!(
            !raw.is_null() && !folded.is_null() && !score_indptr_d.is_null(),
            "attn_score_fold_heads: scores, folded and score_indptr must all be device \
             pointers (raw={raw:?}, folded={folded:?}, indptr={score_indptr_d:?})"
        );

        or_panic(
            "attn_score_fold_heads_dev",
            attn_score_fold_heads(
                &ctx_on(self.stream),
                kernels::routine::In { ptr: raw, rows: 0, width: 0 },
                kernels::routine::In { ptr: score_indptr_d, rows: 0, width: 0 },
                kernels::routine::In { ptr: kv_page_indptr_d, rows: 0, width: 0 },
                kernels::routine::In { ptr: kv_last_page_lens_d, rows: 0, width: 0 },
                page_size,
                num_requests,
                num_q_heads,
                kernels::routine::Out { ptr: folded, rows: 0, width: 0 },
            ),
        );
    }
}

/// Per-worker scratch for the host CSRs — the C++ thread-locals as a value.
#[derive(Debug, Default)]
pub struct ScoreScratch {
    raw_offsets: Vec<u32>,
    folded_offsets: Vec<u32>,
    raw_offsets_i32: Vec<i32>,
    capture_depth: i32,
    pf_folded_offsets: Vec<u32>,
    pf_raw_offsets_i32: Vec<i32>,
    pf_capture_depth: i32,
}

/// A capture whose raw rows would cost more than this is refused rather
/// than served — the prefill row set grows with the context.
const MAX_SCORE_BYTES: u64 = 1 << 30;

/// Sub-buffer alignment inside an arena slot.
const SIDEBAND_ALIGN: usize = 256;

const fn align_up(n: usize) -> usize {
    (n + SIDEBAND_ALIGN - 1) & !(SIDEBAND_ALIGN - 1)
}

/// The score slot's internal carve — raw, then folded, then CSR, each aligned.
/// ONE definition for the acquire and hook-graph prepare paths, or replay miscomputes.
struct ScoreSlotLayout {
    total: usize,
    folded_offset: usize,
    indptr_offset: usize,
}

const fn score_slot_layout(
    raw_bytes: usize,
    folded_bytes: usize,
    indptr_bytes: usize,
) -> ScoreSlotLayout {
    ScoreSlotLayout {
        total: align_up(raw_bytes) + align_up(folded_bytes) + align_up(indptr_bytes),
        folded_offset: align_up(raw_bytes),
        indptr_offset: align_up(raw_bytes) + align_up(folded_bytes),
    }
}

struct DecodeScoreCsrTotals {
    raw_total: u64,
    folded_total: u64,
}

/// One fire's score sink, carved but not yet allocated — the shape a fire
/// publishes UNCONDITIONALLY so a score-capturing arm can be RECORDED whether
/// or not this fire takes it. Under `Union` it must cover what ANY arm needs.
#[derive(Debug, Clone)]
pub struct ScoreSinkPlan {
    /// Bytes the whole slot needs — raw, folded and CSR, each aligned.
    pub bytes: usize,
    /// Byte offset of the folded rows within the slot.
    pub folded_offset: usize,
    /// Byte offset of the device CSR within the slot.
    pub indptr_offset: usize,
    /// The RAW-offset CSR, in elements, as the kernels index it.
    pub indptr: Vec<i32>,
}

/// Plan a fire's score sink from its KV geometry. `window` is the observation
/// window: 1 for decode, or the prefill window — one planner serves both.
/// `None` when the sink would be empty or exceeds [`MAX_SCORE_BYTES`]: the sink
/// stays null and the capturing arm declines.
#[must_use]
pub fn plan_score_sink(
    kv_page_indptr_h: &[u32],
    kv_last_page_lens_h: &[u32],
    page_size: i32,
    num_q_heads: u32,
    window: u32,
) -> Option<ScoreSinkPlan> {
    let requests = kv_page_indptr_h.len().checked_sub(1)?;
    if requests == 0 || window == 0 {
        return None;
    }
    let mut indptr = vec![0i32; requests + 1];
    let mut raw_total: u64 = 0;
    let mut folded_total: u64 = 0;
    for r in 0..requests {
        let pages = kv_page_indptr_h[r + 1].saturating_sub(kv_page_indptr_h[r]);
        let kv_len = if pages == 0 {
            0
        } else {
            (pages - 1) * u32::try_from(page_size.max(0)).unwrap_or(0)
                + kv_last_page_lens_h.get(r).copied().unwrap_or(0)
        };
        indptr[r] = i32::try_from(raw_total).ok()?;
        raw_total += u64::from(kv_len) * u64::from(num_q_heads) * u64::from(window);
        folded_total += u64::from(kv_len);
    }
    indptr[requests] = i32::try_from(raw_total).ok()?;
    if raw_total == 0 || raw_total > 0x7fff_ffff || raw_total * 4 > MAX_SCORE_BYTES {
        return None;
    }
    let carve = score_slot_layout(
        usize::try_from(raw_total).ok()? * 4,
        usize::try_from(folded_total).ok()? * 4,
        (requests + 1) * 4,
    );
    Some(ScoreSinkPlan {
        bytes: carve.total,
        folded_offset: carve.folded_offset,
        indptr_offset: carve.indptr_offset,
        indptr,
    })
}

/// Fill the decode capture's scratch CSRs from the fire's KV geometry. Shared
/// by the capture constructor and [`prepare_decode_score_capture`]: both must
/// compute byte-identical contents, or a replayed fire scores a stale view.
fn compute_decode_score_csr(
    obs: &AttentionObservation<'_>,
    num_q_heads: u32,
    scratch: &mut ScoreScratch,
) -> DecodeScoreCsrTotals {
    let requests = usize::try_from(obs.num_requests.max(0)).unwrap_or(0);
    let page_size = obs.kv_page_size.unwrap_or(0);
    let kvpp = obs.kv_page_indptr_h.unwrap_or(&[]);
    let lens = obs.kv_last_page_lens_h.unwrap_or(&[]);
    scratch.raw_offsets.clear();
    scratch.raw_offsets.resize(requests + 1, 0);
    scratch.folded_offsets.clear();
    scratch.folded_offsets.resize(requests + 1, 0);
    let mut totals = DecodeScoreCsrTotals { raw_total: 0, folded_total: 0 };
    for r in 0..requests {
        let pages = kvpp[r + 1] - kvpp[r];
        let kv_len = if pages == 0 {
            0
        } else {
            (pages - 1) * u32::try_from(page_size.max(0)).unwrap_or(0) + lens[r]
        };
        scratch.raw_offsets[r] = totals.raw_total as u32;
        scratch.folded_offsets[r] = totals.folded_total as u32;
        totals.raw_total += u64::from(kv_len) * u64::from(num_q_heads);
        totals.folded_total += u64::from(kv_len);
    }
    scratch.raw_offsets[requests] = totals.raw_total as u32;
    scratch.folded_offsets[requests] = totals.folded_total as u32;
    totals
}

/// The three device buffers every score capture needs, carved out of the
/// arena's score slot. Ports `detail::ScoreBuffers`.
#[derive(Debug, Default)]
struct ScoreBuffers {
    raw: *mut f32,
    folded: *mut f32,
    indptr_d: *mut i32,
    held: bool,
}

impl ScoreBuffers {
    #[allow(clippy::too_many_arguments)]
    fn acquire<O: DeviceMemory + ScoreOps>(
        &mut self,
        ops: &mut O,
        arena: &mut SidebandArena,
        raw_elems: u64,
        folded_elems: u64,
        indptr_h: &[i32],
        num_requests: u32,
    ) -> bool {
        let raw_bytes = usize::try_from(raw_elems).unwrap_or(usize::MAX) * 4;
        let folded_bytes = usize::try_from(folded_elems).unwrap_or(usize::MAX) * 4;
        let indptr_bytes = (usize::try_from(num_requests).unwrap_or(0) + 1) * 4;
        let layout = score_slot_layout(raw_bytes, folded_bytes, indptr_bytes);
        let Ok(base) = arena.acquire(ops, Region::Score, layout.total) else {
            return false;
        };
        let base = base.cast::<u8>();
        self.held = true;
        self.raw = base.cast();
        // SAFETY: the offsets are inside the slot the arena just handed out.
        unsafe {
            self.folded = base.add(layout.folded_offset).cast();
            self.indptr_d = base.add(layout.indptr_offset).cast();
        }
        // The host CSR is an upper bound; the kernels write only the true
        // kv_len, so slack and a reused slot's stale row must read as zero.
        ops.memset_async(self.folded.cast(), 0, folded_bytes);
        ops.upload_csr(self.indptr_d, indptr_h);
        true
    }

    fn release(&mut self, arena: &mut SidebandArena) {
        if self.held {
            arena.release(Region::Score);
        }
        self.held = false;
        self.raw = std::ptr::null_mut();
        self.folded = std::ptr::null_mut();
        self.indptr_d = std::ptr::null_mut();
    }
}

/// The default prefill observation window — SnapKV's 32, overridable with
/// `PIE_ATTN_SCORE_WINDOW`. Parses like `strtol` (leading space, optional sign,
/// then digits); anything outside `1..=4096` falls back to 32.
#[must_use]
pub fn default_attn_score_window_from(value: Option<&std::ffi::OsStr>) -> u32 {
    let Some(v) = value else { return 32 };
    let bytes = v.as_encoded_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i].is_ascii_whitespace()) {
        i += 1;
    }
    let mut sign: i64 = 1;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        if bytes[i] == b'-' {
            sign = -1;
        }
        i += 1;
    }
    let mut parsed: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        parsed = parsed.saturating_mul(10).saturating_add(i64::from(bytes[i] - b'0'));
        i += 1;
    }
    let parsed = sign * parsed;
    if parsed <= 0 || parsed > 4096 {
        return 32;
    }
    u32::try_from(parsed).unwrap_or(32)
}

#[must_use]

/// The hook-graph prepare pass's fire-level view of the decode capture.
/// Ports `DecodeScoreCapturePlan`.
#[derive(Debug, Clone, Copy)]
pub struct DecodeScoreCapturePlan {
    /// Whether the fire is replayable at all.
    pub ok: bool,
    /// Arena-stable folded-row base.
    pub folded: *const f32,
    /// Arena-stable device CSR base.
    pub indptr_d: *const i32,
    /// The host storage the captured upload reads at replay time — its
    /// ADDRESS is fingerprinted.
    pub indptr_h_data: *const i32,
    /// Folded offsets, host, `num_requests + 1` entries.
    pub folded_offsets_h: *const u32,
    /// Requests in the fire.
    pub num_requests: u32,
}

impl DecodeScoreCapturePlan {
    const fn refused() -> Self {
        Self {
            ok: false,
            folded: std::ptr::null(),
            indptr_d: std::ptr::null(),
            indptr_h_data: std::ptr::null(),
            folded_offsets_h: std::ptr::null(),
            num_requests: 0,
        }
    }
}

/// Refresh the host CSR and pre-grow the arena score slot before a hook
/// replay. Enqueues NO stream work and holds NO slot. Ports
/// `prepare_decode_score_capture`.
pub fn prepare_decode_score_capture<M: DeviceMemory>(
    mem: &mut M,
    arena: Option<&mut SidebandArena>,
    scratch: &mut ScoreScratch,
    observation: &AttentionObservation<'_>,
    num_q_heads: u32,
) -> DecodeScoreCapturePlan {
    let Some(arena) = arena else {
        return DecodeScoreCapturePlan::refused();
    };
    if !observation.usable() || num_q_heads == 0 {
        return DecodeScoreCapturePlan::refused();
    }
    let totals = compute_decode_score_csr(observation, num_q_heads, scratch);
    // Same validity bounds as the constructor: a fire the constructor would
    // refuse must not be declared replayable.
    if totals.raw_total == 0 || totals.raw_total > u64::from(u32::MAX) {
        return DecodeScoreCapturePlan::refused();
    }
    scratch.raw_offsets_i32.clear();
    scratch.raw_offsets_i32.extend(scratch.raw_offsets.iter().map(|&v| v as i32));

    let raw_bytes = usize::try_from(totals.raw_total).unwrap_or(usize::MAX) * 4;
    let folded_bytes = usize::try_from(totals.folded_total).unwrap_or(usize::MAX) * 4;
    let indptr_bytes = (usize::try_from(observation.num_requests.max(0)).unwrap_or(0) + 1) * 4;
    let layout = score_slot_layout(raw_bytes, folded_bytes, indptr_bytes);
    // Acquire-and-release: growth is pulled to HERE, outside any captured
    // region; the capture-time constructor then finds sufficient capacity.
    let Ok(base) = arena.acquire(mem, Region::Score, layout.total) else {
        return DecodeScoreCapturePlan::refused();
    };
    arena.release(Region::Score);
    let base = base.cast::<u8>();
    // SAFETY: the offsets are inside the slot the arena just handed out.
    let (folded, indptr_d) = unsafe {
        (
            base.add(layout.folded_offset).cast::<f32>().cast_const(),
            base.add(layout.indptr_offset).cast::<i32>().cast_const(),
        )
    };
    DecodeScoreCapturePlan {
        ok: true,
        folded,
        indptr_d,
        indptr_h_data: scratch.raw_offsets_i32.as_ptr(),
        folded_offsets_h: scratch.folded_offsets.as_ptr(),
        num_requests: u32::try_from(observation.num_requests).unwrap_or(0),
    }
}

/// RAII capture of one layer's DECODE scores. Ports `LayerScoreCapture`.
#[derive(Debug)]
pub struct LayerScoreCapture {
    active: bool,
    published: bool,
    layer: u32,
    num_q_heads: u32,
    buf: ScoreBuffers,
    folded_offsets_h: *const u32,
    payload: Option<AttentionScores>,
}

impl LayerScoreCapture {
    /// Construct the capture; a no-op unless the fire's hooks asked for
    /// scores and `capturable` holds (a windowed layer passes false — its
    /// row would describe a truncated context while claiming all of it).
    #[allow(clippy::too_many_arguments)]
    pub fn new<O: DeviceMemory + ScoreOps>(
        ops: &mut O,
        arena: Option<&mut SidebandArena>,
        scratch: &mut ScoreScratch,
        hooks: Option<&ScoreHookView<'_>>,
        layer: u32,
        num_q_heads: u32,
        capturable: bool,
    ) -> Self {
        let mut me = Self {
            active: false,
            published: false,
            layer,
            num_q_heads,
            buf: ScoreBuffers::default(),
            folded_offsets_h: std::ptr::null(),
            payload: None,
        };
        let Some(hooks) = hooks else { return me };
        if !hooks.wants_attn_score || !capturable || num_q_heads == 0 {
            return me;
        }
        // Exactly one capture may be live at a time: the shared scratch CSR
        // means a nested use would hand the outer capture the inner one's
        // offsets.
        if scratch.capture_depth != 0 {
            eprintln!(
                "[pie-driver-cuda] nested attention score capture is not \
                 supported; the inner capture is disabled"
            );
            return me;
        }
        let Some(obs) = hooks.observation else {
            return me;
        };
        if !obs.usable() {
            return me;
        }
        let totals = compute_decode_score_csr(obs, num_q_heads, scratch);
        if totals.raw_total == 0 || totals.raw_total > u64::from(u32::MAX) {
            return me;
        }
        scratch.raw_offsets_i32.clear();
        scratch.raw_offsets_i32.extend(scratch.raw_offsets.iter().map(|&v| v as i32));
        let Some(arena) = arena else {
            eprintln!(
                "[pie-driver-cuda] score capture has no hook sideband arena; \
                 refusing the capture"
            );
            return me;
        };
        let requests = u32::try_from(obs.num_requests).unwrap_or(0);
        let indptr = std::mem::take(&mut scratch.raw_offsets_i32);
        let acquired =
            me.buf.acquire(ops, arena, totals.raw_total, totals.folded_total, &indptr, requests);
        scratch.raw_offsets_i32 = indptr;
        if !acquired {
            return me;
        }
        me.folded_offsets_h = scratch.folded_offsets.as_ptr();
        me.active = true;
        scratch.capture_depth += 1;
        me
    }

    /// Whether the capture is live.
    #[must_use]
    pub const fn active(&self) -> bool {
        self.active
    }

    /// The raw per-head rows the capture kernel writes.
    #[must_use]
    pub const fn raw(&self) -> *mut f32 {
        self.buf.raw
    }

    /// The device CSR both kernels address rows with.
    #[must_use]
    pub const fn indptr_d(&self) -> *const i32 {
        self.buf.indptr_d
    }

    /// Normalize, fold heads and finalise the payload. The observation is re-read
    /// at publish time — a vanished mid-layer geometry is an error, not a fold
    /// against a stale view. Order matters: `normalize_decode` divides in place
    /// before `fold_heads` averages; reversed folds un-normalised scores silently.
    pub fn publish<O: ScoreOps>(
        &mut self,
        ops: &mut O,
        hooks: Option<&ScoreHookView<'_>>,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
    ) -> Result<(), ScoreError> {
        if !self.active || self.published {
            return Ok(());
        }
        let obs = hooks.and_then(|h| h.observation);
        let Some(obs) = obs else {
            return Err(ScoreError::LostGeometry);
        };
        if !obs.usable() {
            return Err(ScoreError::LostGeometry);
        }
        let num_q_heads = i32::try_from(self.num_q_heads).unwrap_or(0);
        ops.normalize_decode(
            self.buf.raw,
            self.buf.indptr_d,
            kv_page_indptr_d,
            kv_last_page_lens_d,
            page_size,
            obs.num_requests,
            num_q_heads,
        );
        ops.fold_heads(
            self.buf.raw,
            self.buf.indptr_d,
            kv_page_indptr_d,
            kv_last_page_lens_d,
            page_size,
            obs.num_requests,
            num_q_heads,
            self.buf.folded,
        );
        self.payload = Some(AttentionScores {
            values: self.buf.folded,
            offsets_h: self.folded_offsets_h,
            num_requests: u32::try_from(obs.num_requests).unwrap_or(0),
            layer: self.layer,
        });
        self.published = true;
        Ok(())
    }

    /// The published payload, for the layer's `OnAttn` sideband; `None`
    /// until [`Self::publish`] ran.
    #[must_use]
    pub const fn scores(&self) -> Option<&AttentionScores> {
        if self.published { self.payload.as_ref() } else { None }
    }

    /// The C++ destructor: hand the slot back and drop the depth.
    pub fn release(&mut self, arena: &mut SidebandArena, scratch: &mut ScoreScratch) {
        if self.active {
            scratch.capture_depth -= 1;
        }
        self.buf.release(arena);
        self.active = false;
        self.published = false;
    }
}

impl Drop for LayerScoreCapture {
    fn drop(&mut self) {
        debug_assert!(!self.buf.held, "LayerScoreCapture dropped without release()");
    }
}

/// Why a publish refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreError {
    /// The fire's observation vanished between capture and publish (the C++
    /// throws `"...lost its fire geometry mid-layer"`).
    LostGeometry,
}

/// RAII capture of one layer's PREFILL scores — SnapKV's observation
/// window. Ports `LayerPrefillScoreCapture`; raw rows carry the window factor.
#[derive(Debug)]
pub struct LayerPrefillScoreCapture {
    active: bool,
    published: bool,
    layer: u32,
    window: u32,
    num_q_heads: u32,
    num_requests: u32,
    buf: ScoreBuffers,
    folded_offsets_h: *const u32,
    payload: Option<AttentionScores>,
}

impl LayerPrefillScoreCapture {
    /// Construct the capture. Same gate chain as decode, plus the window and
    /// two ceilings: the int32 element bound, and the 1 GiB byte cap.
    #[allow(clippy::too_many_arguments)]
    pub fn new<O: DeviceMemory + ScoreOps>(
        ops: &mut O,
        arena: Option<&mut SidebandArena>,
        scratch: &mut ScoreScratch,
        hooks: Option<&ScoreHookView<'_>>,
        layer: u32,
        num_q_heads: u32,
        window: u32,
        capturable: bool,
    ) -> Self {
        let mut me = Self {
            active: false,
            published: false,
            layer,
            window,
            num_q_heads,
            num_requests: 0,
            buf: ScoreBuffers::default(),
            folded_offsets_h: std::ptr::null(),
            payload: None,
        };
        let Some(hooks) = hooks else { return me };
        if !hooks.wants_attn_score || !capturable || num_q_heads == 0 || window == 0 {
            return me;
        }
        if scratch.pf_capture_depth != 0 {
            eprintln!(
                "[pie-driver-cuda] nested prefill score capture is not \
                 supported; the inner capture is disabled"
            );
            return me;
        }
        let Some(obs) = hooks.observation else {
            return me;
        };
        if !obs.usable() {
            return me;
        }
        let requests = usize::try_from(obs.num_requests.max(0)).unwrap_or(0);
        let page_size = obs.kv_page_size.unwrap_or(0);
        let kvpp = obs.kv_page_indptr_h.unwrap_or(&[]);
        let lens = obs.kv_last_page_lens_h.unwrap_or(&[]);
        scratch.pf_folded_offsets.clear();
        scratch.pf_folded_offsets.resize(requests + 1, 0);
        scratch.pf_raw_offsets_i32.clear();
        scratch.pf_raw_offsets_i32.resize(requests + 1, 0);
        let mut raw_total: u64 = 0;
        let mut folded_total: u64 = 0;
        for r in 0..requests {
            let pages = kvpp[r + 1] - kvpp[r];
            let kv_len = if pages == 0 {
                0
            } else {
                (pages - 1) * u32::try_from(page_size.max(0)).unwrap_or(0) + lens[r]
            };
            scratch.pf_raw_offsets_i32[r] = raw_total as i32;
            scratch.pf_folded_offsets[r] = folded_total as u32;
            raw_total += u64::from(kv_len) * u64::from(num_q_heads) * u64::from(window);
            folded_total += u64::from(kv_len);
        }
        scratch.pf_raw_offsets_i32[requests] = raw_total as i32;
        scratch.pf_folded_offsets[requests] = folded_total as u32;
        // The int32 CSR is what the kernels index with, so the total must fit
        // a signed 32-bit offset (the byte ceiling bites first anyway).
        if raw_total == 0 || raw_total > 0x7fff_ffff || raw_total * 4 > MAX_SCORE_BYTES {
            if raw_total != 0 {
                eprintln!(
                    "[pie-driver-cuda] prefill score capture needs {} MiB \
                     ({num_q_heads} heads x {window} window rows); refusing",
                    (raw_total * 4) >> 20
                );
            }
            return me;
        }
        let Some(arena) = arena else {
            eprintln!(
                "[pie-driver-cuda] score capture has no hook sideband arena; \
                 refusing the capture"
            );
            return me;
        };
        let indptr = std::mem::take(&mut scratch.pf_raw_offsets_i32);
        let acquired = me.buf.acquire(
            ops,
            arena,
            raw_total,
            folded_total,
            &indptr,
            u32::try_from(requests).unwrap_or(0),
        );
        scratch.pf_raw_offsets_i32 = indptr;
        if !acquired {
            return me;
        }
        me.folded_offsets_h = scratch.pf_folded_offsets.as_ptr();
        me.num_requests = u32::try_from(requests).unwrap_or(0);
        me.active = true;
        scratch.pf_capture_depth += 1;
        me
    }

    /// Whether the capture is live.
    #[must_use]
    pub const fn active(&self) -> bool {
        self.active
    }

    /// The raw per-(head, window-row) rows.
    #[must_use]
    pub const fn raw(&self) -> *mut f32 {
        self.buf.raw
    }

    /// The folded row the payload publishes.
    #[must_use]
    pub const fn folded(&self) -> *mut f32 {
        self.buf.folded
    }

    /// The device CSR.
    #[must_use]
    pub const fn indptr_d(&self) -> *const i32 {
        self.buf.indptr_d
    }

    /// The observation window, as the C++ `int` accessor.
    #[must_use]
    pub const fn window(&self) -> i32 {
        self.window as i32
    }

    /// Normalize, fold and finalise the folded row. Order is load-bearing:
    /// `normalize_prefill` divides in place before `fold_prefill` collapses
    /// heads and window rows; reversed averages un-normalised scores.
    pub fn publish<O: ScoreOps>(
        &mut self,
        ops: &mut O,
        qo_indptr_d: *const u32,
        kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32,
        page_size: i32,
    ) {
        if !self.active || self.published {
            return;
        }
        let requests = i32::try_from(self.num_requests).unwrap_or(0);
        let heads = i32::try_from(self.num_q_heads).unwrap_or(0);
        let window = self.window();
        ops.normalize_prefill(
            self.buf.raw,
            self.buf.indptr_d,
            qo_indptr_d,
            kv_page_indptr_d,
            kv_last_page_lens_d,
            page_size,
            requests,
            heads,
            window,
        );
        ops.fold_prefill(
            self.buf.raw,
            self.buf.folded,
            self.buf.indptr_d,
            qo_indptr_d,
            kv_page_indptr_d,
            kv_last_page_lens_d,
            page_size,
            requests,
            heads,
            window,
        );
        self.payload = Some(AttentionScores {
            values: self.buf.folded,
            offsets_h: self.folded_offsets_h,
            num_requests: self.num_requests,
            layer: self.layer,
        });
        self.published = true;
    }

    /// The published payload; `None` until [`Self::publish`] ran.
    #[must_use]
    pub const fn scores(&self) -> Option<&AttentionScores> {
        if self.published { self.payload.as_ref() } else { None }
    }

    /// The C++ destructor.
    pub fn release(&mut self, arena: &mut SidebandArena, scratch: &mut ScoreScratch) {
        if self.active {
            scratch.pf_capture_depth -= 1;
        }
        self.buf.release(arena);
        self.active = false;
        self.published = false;
    }
}

impl Drop for LayerPrefillScoreCapture {
    fn drop(&mut self) {
        debug_assert!(!self.buf.held, "LayerPrefillScoreCapture dropped without release()");
    }
}

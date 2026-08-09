//! One layer's attention scores, published for the duration of that
//! layer's `OnAttn` hook — gate-score-capture.
//!
//! Ports `model/attn_score.cu`: the decode capture (one query row per
//! request), the prefill capture (SnapKV's observation window), the shared
//! score-slot carve in the fire's [`SidebandArena`], and the hook-graph
//! prepare helper that refreshes the replay-read host CSR.
//!
//! # The scratch, made explicit
//!
//! The C++ keeps the host CSRs in `thread_local` vectors for two reasons:
//! allocation reuse across the layers of a fire, and ADDRESS IDENTITY —
//! a captured hook body's upload node reads that storage at replay time,
//! so the prepare pass must refresh the same bytes at the same address.
//! Rust makes the identity a value: the caller owns a [`ScoreScratch`] and
//! passes it to every capture and prepare of a worker thread. The nested-
//! capture depth guard lives in the scratch for the same reason it guarded
//! the thread-locals: a second live capture would clobber the first one's
//! published offsets.
//!
//! # Explicit release
//!
//! As with [`super::page_mask::FirePageMask`], the arena is not owned by
//! the capture, so `Drop` cannot release the slot; `release(&mut arena)`
//! is the destructor and `Drop` only `debug_assert!`s it ran.
//!
//! # The capture pipeline, now that three launches have crossed
//!
//! [`ScoreOps`] is the seam every kernel this module fires goes through:
//! `unit_of` -> `cache::module` -> `Args::bind` -> a hand-built `Launch`,
//! fired on the fire's own stream. `fold_heads` crossed first. As of this
//! change **three more have**, and the capture pipeline reads:
//!
//! ```text
//! decode   FA2 kernel   C++    dispatch_attention_flashinfer_decode_capture_bf16
//!          normalize    RUST   ScoreOps::normalize_decode
//!          fold_heads   RUST   ScoreOps::fold_heads
//!
//! prefill  FA2 kernel   C++    dispatch_attention_flashinfer_prefill_capture_bf16
//!          normalize    RUST   ScoreOps::normalize_prefill
//!          fold         RUST   ScoreOps::fold_prefill
//! ```
//!
//! Only the FA2 kernel itself is still C++, and it is C++ for a reason that
//! is written down rather than assumed: it is a template cross-product with
//! hundreds of instantiations and no table rows (`new-horizon.md` §53).
//! Everything downstream of it in the capture is Rust firing NVRTC'd device
//! text out of `kernels-cuda-new/csrc/src/attn/attention_score_post.cuh`,
//! whose rows are `families::attn::ATTN_SCORE_POST`.
//!
//! **Stream order is why `publish` is the home.** The launches were the tail
//! of the capture dispatch, after `CUDA_CHECK(status)` and before it
//! returned. `publish` runs on the fire's stream and nothing between the
//! dispatch and it touches the score buffer, so issuing them here puts the
//! same four kernels on the same stream in the same order. What changed is
//! which language enqueues them.
//!
//! # What this cost, and what is owed
//!
//! The three launches carry three geometries the C++ stated literally, and
//! this module states them as named constants beside the launches that use
//! them — `NORMALIZE_BLOCK`, `PREFILL_FOLD_GRID_Y`, `FOLD_GRID_Y`,
//! `FOLD_BLOCK`. None of them is a `kernels::LaunchRule` and none should
//! become one; the rows are `LaunchRule::Unstated` and carry the argument.
//!
//! `tests/attn_score_parity.rs` hashes a recorder transcript of these ops
//! against a golden produced from `model/attn_score.cu`. **That file was
//! deleted in `4569b9e4b`**, so `tests/oracle/attn_score/run.sh` cannot run
//! and the golden cannot be regenerated. The transcript therefore keeps its
//! old shape: the parity `Recorder` implements the three new methods as
//! silent no-ops, documented at the point of omission, because the C++
//! program that golden describes never contained these launches — they were
//! in a different translation unit the oracle never compiled. Their geometry
//! is pinned instead by `tests/attn_score_post_geometry.rs`, which is a
//! transcription check and needs no oracle.

// stderr is the C++'s own refusal channel for these messages; routing them
// anywhere else would change what the operator sees.
#![allow(clippy::print_stderr)]

use crate::fire::sideband_arena::{DeviceMemory, Region, SidebandArena};

/// One layer's attention scores — the read side of `AttnScore`.
///
/// `values` is ragged and head-folded: request `r` occupies
/// `[offsets_h[r], offsets_h[r + 1])`, one float per live KV position.
/// `layer` is carried so a consumer can refuse a payload from a different
/// layer rather than silently scoring the wrong one.
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

/// What a PTIR attention-stage program observes about the fire — the
/// subset of the C++ `AttentionObservation` the score captures read, with
/// the same seven-pointer `usable()` gate. The C++ reaches the page size
/// through `kv->page_size()`; the cache itself is proven by its own gate,
/// so here the observation carries the answer (`None` is the null cache).
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
    /// The C++ `usable()` — all seven pointers and a positive request
    /// count.
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

/// The three `StageHooks` fields the score captures read. Gate-stage-hooks
/// owns the full struct; the arena travels separately because Rust needs
/// it `&mut` for the slot acquire.
#[derive(Debug, Clone, Copy)]
pub struct ScoreHookView<'a> {
    /// Does any program in the launch read `AttnScore`?
    pub wants_attn_score: bool,
    /// The fire's observation, while a body runs.
    pub observation: Option<&'a AttentionObservation<'a>>,
}

/// The stream ops a capture issues — recorders in the parity test, CUDA in
/// the real driver.
///
/// Four of the six are kernel launches out of `ATTN_SCORE_POST` and
/// `ATTN_SCORE_FOLD`. They are methods rather than free functions because
/// the parity test substitutes a recorder for all of them at once; see the
/// module header for which C++ launcher each replaced.
pub trait ScoreOps {
    /// `cudaMemsetAsync` over the folded rows.
    fn memset_async(&mut self, dst: *mut u8, value: u8, bytes: usize);
    /// The CSR upload (`cudaMemcpyAsync`, host to device).
    fn upload_csr(&mut self, dst: *mut i32, src: &[i32]);
    /// `kernels::attn::attn_score_normalize` — the DECODE capture's
    /// divide-by-total, which ran as the tail of
    /// `dispatch_attention_flashinfer_decode_capture_bf16`.
    ///
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
    /// `kernels::attn::attn_prefill_score_fold` — the prefill fold, which
    /// collapses heads AND window rows into the published row.
    ///
    /// Not in place, unlike its two siblings: it reads `scores` and writes
    /// `folded`, the same split [`Self::fold_heads`] makes.
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

/// The live [`ScoreOps`] (retirement plan phase B). The memset and the CSR
/// upload are stream-ordered like the C++'s (`cudaMemsetAsync` /
/// `cudaMemcpyAsync` on the fire's stream); the CSR source is pageable host
/// memory in both drivers, which the runtime staging-copies — same behaviour,
/// stated rather than assumed.
///
/// # Why this is no longer `#[cfg(feature = "bridge")]`
///
/// It was, and only for `fold_heads`: that method called
/// `bind::abi::ffi::pie_k_attn_attn_score_fold_heads`, a generated shim entry
/// into `attention_flashinfer.cu`'s launcher, which exists only when the
/// kernels archive is linked. **It no longer calls it.** The fold's device
/// text is `kernels-cuda-new`'s `attn/attention_flashinfer` unit, NVRTC
/// compiles it, and this method builds its own [`Launch`]. Nothing on this
/// path needs the archive, so nothing on this path is gated on it — which is
/// the whole claim of the migration made checkable: `_cuda` without `bridge`
/// now reaches a real fold.
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

/// The fold's symbol, in the JIT table.
///
/// `attn::attn_score_fold_heads` is `families::attn::ATTN_SCORE_FOLD_SIGS`'
/// only row and `table::attn`'s row of the same name — one string, resolved
/// through `unit_of` rather than declared here, so a rename in that crate is
/// a refusal here and not a silent miss.
#[cfg(feature = "_cuda")]
const FOLD_SYMBOL: &str = "attn::attn_score_fold_heads";

/// The decode normalize's symbol — `ATTN_SCORE_POST_SIGS[0]`.
#[cfg(feature = "_cuda")]
const NORMALIZE_SYMBOL: &str = "attn::attn_score_normalize";

/// The prefill normalize's symbol — `ATTN_SCORE_POST_SIGS[1]`.
#[cfg(feature = "_cuda")]
const PREFILL_NORMALIZE_SYMBOL: &str = "attn::attn_prefill_score_normalize";

/// The prefill fold's symbol — `ATTN_SCORE_POST_SIGS[2]`.
#[cfg(feature = "_cuda")]
const PREFILL_FOLD_SYMBOL: &str = "attn::attn_prefill_score_fold";

/// Fire one `ATTN_SCORE_POST`/`ATTN_SCORE_FOLD` row at a stated geometry.
///
/// Every launch in this module goes through here, so the resolution chain is
/// written once: `unit_of` finds the unit that hosts the symbol,
/// `unit.row()` gets the signature, `cache::module` compiles-or-reuses,
/// `Args::bind` checks the values against that signature, and `fire` takes
/// the `Launch` the caller transcribed from the C++ `<<<>>>`.
///
/// **Every failure here is a panic and never a skip.** These kernels write
/// the score rows a policy will read; a launch that silently did not happen
/// leaves the memset pattern behind, and a payload published over it is a
/// plausible row of zeros rather than a fault. That is the module header's
/// `LostGeometry` argument applied to the launch itself.
#[cfg(feature = "_cuda")]
fn fire_score_row(
    symbol: &str,
    launch: kernels_cuda_new::runtime::Launch,
    values: &[kernels_cuda_new::runtime::ArgValue],
    stream_raw: *mut std::ffi::c_void,
) {
    use kernels_cuda_new::runtime::{Args, Stream, cache};

    let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
    };
    let module = match cache::module(index, unit) {
        Ok(module) => module,
        Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
    };
    let mut args = match Args::bind(sig, values) {
        Ok(args) => args,
        Err(why) => panic!("{symbol}: {why}"),
    };
    // SAFETY: the caller of `publish` holds the fire's stream live across the
    // launch — the same assertion the `pie_k_*` call made when it handed
    // `self.stream` to a C++ launcher that put it in a `<<<>>>`.
    let stream = unsafe { Stream::from_runtime(stream_raw) };
    if let Err(why) = module.fire(sig, launch, &mut args, stream) {
        panic!("{symbol}: {why}");
    }
}

/// The fold's grid fanout: `attention_flashinfer.cu:828`'s literal `64u`.
///
/// **This constant is the reason the fold is fired by hand.** It is not an
/// extent. The kernel's inner loop is
/// `for (int i = blockIdx.y; i < n; i += gridDim.y)` over the request's KV
/// positions, so `gridDim.y` is an OCCUPANCY FANOUT: every value of it
/// computes the same floats, and `1` computes them correctly in a
/// sixty-fourth of the blocks. That is why no [`kernels::LaunchRule`] states
/// it and why one must not be added to make it look stated —
/// `families::attn::ATTN_SCORE_FOLD` carries the argument at length, and the
/// short form is that a rule is a function of the fire's rectangle and `64`
/// is not in the rectangle. It is a property of a kernel's grid-stride loop,
/// and the only other one in the whole of `csrc/src` is a *different* literal
/// (`:1138`'s `32u`), so there is no shared rule to write.
///
/// Naming it here puts it beside the launch that uses it and one `git grep`
/// from the `<<<>>>` it was copied from.
#[cfg(feature = "_cuda")]
const FOLD_GRID_Y: u32 = 64;

/// The fold's block width: `attention_flashinfer.cu:829`'s `256`.
///
/// Also load-bearing and also not tuning: the kernel folds warp partials
/// through `__shared__ float red[256 / 32]`, so a launch at another width
/// would read reduction slots nothing wrote — a plausible score row rather
/// than a fault, which is the same hazard `PAGE_COMPACT_ROWS` records for its
/// own `BLOCK`.
#[cfg(feature = "_cuda")]
const FOLD_BLOCK: u32 = 256;

/// The three post-kernels' block width: the literal `256` in all three
/// `<<<grid, 256, 0, stream>>>` at `attention_flashinfer.cu:591`, `:923` and
/// `:929`.
///
/// Load-bearing for the same reason `FOLD_BLOCK` is: every one of them
/// reduces through `__shared__ float red[256 / 32]`, one slot per warp, so a
/// launch at another width reads reduction slots nothing wrote. That is a
/// plausible score row rather than a fault — the hazard
/// `PAGE_COMPACT_ROWS` records for its own `BLOCK`.
#[cfg(feature = "_cuda")]
const NORMALIZE_BLOCK: u32 = 256;

/// The prefill fold's grid fanout: `attention_flashinfer.cu:928`'s literal
/// `32u`.
///
/// The sibling of `FOLD_GRID_Y` and the same kind of thing — an OCCUPANCY
/// FANOUT, not an extent. `attn_prefill_score_fold`'s inner loop strides
/// `blockIdx.y` by `gridDim.y` over the request's KV positions, so every
/// value of it computes the same floats and `1` computes them correctly in a
/// thirty-second of the blocks. No [`kernels::LaunchRule`] states it, none
/// should be added to make it look stated, and the fact that the two folds
/// in this file chose *different* literals (`64` and `32`) for the identical
/// construct is the clearest evidence available that neither is a rule.
#[cfg(feature = "_cuda")]
const PREFILL_FOLD_GRID_Y: u32 = 32;

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

    /// The decode normalize, at the geometry this driver states.
    ///
    /// # What this replaced, line for line
    ///
    /// The tail of `dispatch_attention_flashinfer_decode_capture_bf16` in
    /// `driver-cuda/csrc/attn/attention_flashinfer.cu`, immediately after
    /// its `CUDA_CHECK(status)`:
    ///
    /// ```text
    /// const dim3 grid(static_cast<unsigned>(cache.num_requests),
    ///                 static_cast<unsigned>(cache.num_q_heads));
    /// device::attn_score_normalize<<<grid, 256, 0, stream>>>(
    ///     score_out, score_indptr_d, kv_page_indptr_d, kv_last_page_lens_d,
    ///     cache.page_size);
    /// ```
    ///
    /// Five operands, two grid extents, one block width. `kv_len` is derived
    /// from the page CSR inside the body rather than passed, which is why no
    /// length appears here — `attention_score_post.cuh` argues that beside
    /// the body and this must not "helpfully" add one.
    ///
    /// # The guard
    ///
    /// `num_requests <= 0` returns. The C++ had no such guard because the
    /// dispatch above it could not be reached with an empty fire; here the
    /// guard is required, because a zero `grid.x` reaching
    /// [`kernels_cuda_new::runtime::KernelModule::fire`] is `Error::Geometry`
    /// and would turn a legal no-op into a refusal. `num_q_heads == 0` is
    /// the same case on the other axis.
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
        use kernels_cuda_new::runtime::{ArgValue, Launch};

        if num_requests <= 0 || num_q_heads <= 0 {
            return;
        }
        assert!(
            !scores.is_null() && !score_indptr_d.is_null(),
            "attn_score_normalize: scores and score_indptr must be device pointers \
             (scores={scores:?}, indptr={score_indptr_d:?})"
        );

        // `ATTN_SCORE_POST_SIGS[0]`'s operands, in the row's order — which is
        // the launcher's argument order above, unchanged.
        let values = [
            ArgValue::Ptr(scores.cast()),
            ArgValue::Ptr(score_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_page_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_last_page_lens_d.cast_mut().cast()),
            ArgValue::I32(page_size),
        ];
        let launch = Launch {
            grid: [num_requests.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
            block: [NORMALIZE_BLOCK, 1, 1],
            smem: 0,
        };
        fire_score_row(NORMALIZE_SYMBOL, launch, &values, self.stream);
    }

    /// The prefill normalize.
    ///
    /// # What this replaced, line for line
    ///
    /// The tail of `dispatch_attention_flashinfer_prefill_capture_bf16`:
    ///
    /// ```text
    /// const dim3 norm_grid(static_cast<unsigned>(cache.num_requests),
    ///                      static_cast<unsigned>(cache.num_q_heads),
    ///                      static_cast<unsigned>(window));
    /// device::attn_prefill_score_normalize<<<norm_grid, 256, 0, stream>>>(
    ///     score_out, score_indptr_d, qo_indptr_d, kv_page_indptr_d,
    ///     kv_last_page_lens_d, cache.page_size, window);
    /// ```
    ///
    /// `window` is BOTH the third grid extent and the last operand, and that
    /// duplication is the launcher's, not an oversight to tidy: `blockIdx.z`
    /// selects the window row and the operand bounds `rows = min(window,
    /// qo_len)` inside the body. Passing one and deriving the other would be
    /// a different kernel.
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
        use kernels_cuda_new::runtime::{ArgValue, Launch};

        if num_requests <= 0 || num_q_heads <= 0 || window <= 0 {
            return;
        }
        assert!(
            !scores.is_null() && !score_indptr_d.is_null() && !qo_indptr_d.is_null(),
            "attn_prefill_score_normalize: scores, score_indptr and qo_indptr must be \
             device pointers (scores={scores:?}, indptr={score_indptr_d:?}, \
             qo={qo_indptr_d:?})"
        );

        let values = [
            ArgValue::Ptr(scores.cast()),
            ArgValue::Ptr(score_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(qo_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_page_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_last_page_lens_d.cast_mut().cast()),
            ArgValue::I32(page_size),
            ArgValue::I32(window),
        ];
        let launch = Launch {
            grid: [
                num_requests.unsigned_abs(),
                num_q_heads.unsigned_abs(),
                window.unsigned_abs(),
            ],
            block: [NORMALIZE_BLOCK, 1, 1],
            smem: 0,
        };
        fire_score_row(PREFILL_NORMALIZE_SYMBOL, launch, &values, self.stream);
    }

    /// The prefill fold.
    ///
    /// # What this replaced, line for line
    ///
    /// The last two statements of
    /// `dispatch_attention_flashinfer_prefill_capture_bf16`:
    ///
    /// ```text
    /// const dim3 fold_grid(static_cast<unsigned>(cache.num_requests), 32u);
    /// device::attn_prefill_score_fold<<<fold_grid, 256, 0, stream>>>(
    ///     score_out, folded_out, score_indptr_d, qo_indptr_d,
    ///     kv_page_indptr_d, kv_last_page_lens_d, cache.page_size,
    ///     cache.num_q_heads, window);
    /// ```
    ///
    /// `num_q_heads` is an OPERAND here and a grid extent in the normalize
    /// above — the fold collapses the head axis rather than indexing it, so
    /// it must know the count without having a block per head. See
    /// `PREFILL_FOLD_GRID_Y` for why the second grid axis is `32` and why
    /// that is not a rule.
    ///
    /// The null guard panics, matching [`Self::fold_heads`]: a fold that did
    /// not run leaves `folded` holding the memset pattern, and the payload
    /// published over it is a score row of zeros every downstream policy
    /// will happily read.
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
        use kernels_cuda_new::runtime::{ArgValue, Launch};

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

        let values = [
            ArgValue::Ptr(scores.cast_mut().cast()),
            ArgValue::Ptr(folded.cast()),
            ArgValue::Ptr(score_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(qo_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_page_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_last_page_lens_d.cast_mut().cast()),
            ArgValue::I32(page_size),
            ArgValue::I32(num_q_heads),
            ArgValue::I32(window),
        ];
        let launch = Launch {
            grid: [num_requests.unsigned_abs(), PREFILL_FOLD_GRID_Y, 1],
            block: [NORMALIZE_BLOCK, 1, 1],
            smem: 0,
        };
        fire_score_row(PREFILL_FOLD_SYMBOL, launch, &values, self.stream);
    }

    /// The fold, fired at a geometry this driver states and no rule does.
    ///
    /// # What this replaced, line for line
    ///
    /// `attn::attn_score_fold_heads` in
    /// `kernels-cuda/csrc/src/attn/attention_flashinfer.cu:812-832` — a
    /// nine-argument host launcher whose whole body is two guards, a `dim3`
    /// and a `<<<>>>`. The seven-argument kernel it launched is now
    /// `kernels-cuda-new`'s `attn/attention_flashinfer` unit; the two guards
    /// and the `dim3` are here. The launcher's remaining two arguments were
    /// `num_requests`, which was only ever `grid.x`, and `stream`, which was
    /// only ever the launch's — neither is a kernel operand, and this is
    /// where that stops being invisible.
    ///
    /// # The guards, and why they are two different things
    ///
    /// `num_requests <= 0` returns, exactly as the C++ did. An empty fire is
    /// not an error — the capture publishes an empty payload — and it must be
    /// caught HERE, because a zero `grid.x` reaching
    /// [`kernels_cuda_new::runtime::KernelModule::fire`] is `Error::Geometry`
    /// and would turn a legal no-op into a refusal.
    ///
    /// A null buffer PANICS, because the C++ threw. This crate's C++ threw
    /// through a shim that caught, and the catch is gone with the shim, so
    /// the refusal has to be spelled in Rust or it is not spelled at all. It
    /// is a panic and not a log-and-return for the reason the module header
    /// gives about `LostGeometry`: a fold that does not run leaves `folded`
    /// holding the memset pattern, and the payload published over it is a
    /// score row of zeros that every downstream policy will happily read.
    /// Silence here is a wrong answer, not a missing one.
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
        use kernels_cuda_new::runtime::{ArgValue, Launch};

        // `attention_flashinfer.cu:817` — `if (num_requests <= 0) return;`
        if num_requests <= 0 {
            return;
        }
        // `attention_flashinfer.cu:818-822` — the launcher's throw, as a
        // refusal that cannot be mistaken for a fold.
        assert!(
            !raw.is_null() && !folded.is_null() && !score_indptr_d.is_null(),
            "attn_score_fold_heads: scores, folded and score_indptr must all be device \
             pointers (raw={raw:?}, folded={folded:?}, indptr={score_indptr_d:?})"
        );

        // The row's operands, in the row's order. `Args::bind` checks them
        // against the signature, so a drift between this list and
        // `ATTN_SCORE_FOLD_SIGS` is a refusal and not a shifted argument.
        let values = [
            ArgValue::Ptr(raw.cast_mut().cast()),
            ArgValue::Ptr(score_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_page_indptr_d.cast_mut().cast()),
            ArgValue::Ptr(kv_last_page_lens_d.cast_mut().cast()),
            ArgValue::I32(page_size),
            ArgValue::I32(num_q_heads),
            ArgValue::Ptr(folded.cast()),
        ];

        // `attention_flashinfer.cu:828-829`, transcribed. `num_requests` is
        // `grid.x` because the kernel indexes the request by `blockIdx.x`;
        // see `FOLD_GRID_Y` for why the second axis is a constant here and
        // not a rule there.
        let launch = Launch {
            grid: [num_requests.unsigned_abs(), FOLD_GRID_Y, 1],
            block: [FOLD_BLOCK, 1, 1],
            smem: 0,
        };
        fire_score_row(FOLD_SYMBOL, launch, &values, self.stream);
    }
}

/// Per-worker scratch for the host CSRs — the C++ thread-locals as a
/// value. See the module docs for why identity matters.
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

/// The score slot's internal carve — raw, then folded, then the CSR, each
/// aligned. ONE definition shared by the captures' acquire and the
/// hook-graph prepare helper, exactly as in the C++: the helper derives
/// arena-stable addresses a captured graph bakes, so the two going out of
/// step would be a silent replay miscompute.
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
/// publishes UNCONDITIONALLY so that a score-capturing arm can be
/// RECORDED whether or not this fire takes it.
///
/// The old shell published a null sink on purpose, and the reasoning was
/// sound for the world it was written in: the capturing dispatch refuses
/// without a sink, and refusing before the launcher is reached beats
/// throwing across the C ABI. But it makes the union decline every
/// lowering that so much as mentions `_capture` — which is the case the
/// union exists for, since `WantsAttnScore` is a folded predicate and one
/// exec is meant to serve both answers. Under `GuardMode::Union` every
/// arm is recorded, so "the state this fire happens to need" is the wrong
/// question; the right one is "the state ANY arm could need."
///
/// See `.wiki/driver/graph.md` §5 ①. The cost is resident memory and
/// plan-raise time, not runtime: the arm a fire does not take is skipped
/// by the conditional rather than executed.
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

/// Plan a fire's score sink from its KV geometry.
///
/// `window` is the observation window — 1 for a decode capture (one query
/// row per request), [`default_attn_score_window`] for a prefill one. The
/// two forms differ only in that factor, so one planner serves both.
///
/// `None` when the sink would be empty (no rows to score) or larger than
/// [`MAX_SCORE_BYTES`], which is a refusal to publish rather than a
/// refusal to fire: the sink stays null and the capturing arm declines as
/// it always did.
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

/// Fill the decode capture's scratch CSRs from the fire's KV geometry.
/// Shared by the capture constructor and [`prepare_decode_score_capture`],
/// as in the C++ — both sites must compute byte-identical contents into
/// the same storage or a replayed fire scores against a stale channel
/// view of its KV lengths.
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
    let mut totals = DecodeScoreCsrTotals {
        raw_total: 0,
        folded_total: 0,
    };
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
        // The host CSR that sized these is an UPPER BOUND; the kernels write
        // only the true kv_len of each request, so the slack — and, with a
        // reused slot, the PREVIOUS layer's folded row — must read as "this
        // position received no attention" on every acquire.
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
/// `PIE_ATTN_SCORE_WINDOW`, as a pure function of the variable's value.
///
/// The parse is `strtol`'s: leading whitespace, an optional sign, then
/// digits — `"1e3"` reads as 1, `"abc"` as 0 — and anything outside
/// `1..=4096` falls back to 32.
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
        parsed = parsed
            .saturating_mul(10)
            .saturating_add(i64::from(bytes[i] - b'0'));
        i += 1;
    }
    let parsed = sign * parsed;
    if parsed <= 0 || parsed > 4096 {
        return 32;
    }
    u32::try_from(parsed).unwrap_or(32)
}

/// `PIE_ATTN_SCORE_WINDOW`, read once and cached — ports
/// `default_attn_score_window`.
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
    scratch
        .raw_offsets_i32
        .extend(scratch.raw_offsets.iter().map(|&v| v as i32));

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

/// RAII capture of one layer's DECODE scores. Ports `LayerScoreCapture`;
/// see the class comment in `attn_score.hpp` for the call-site shape.
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
        // Exactly one capture may be live at a time: the host CSR lives in
        // the shared scratch, and a nested use would hand the outer capture
        // the inner one's offsets.
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
        scratch
            .raw_offsets_i32
            .extend(scratch.raw_offsets.iter().map(|&v| v as i32));
        let Some(arena) = arena else {
            eprintln!(
                "[pie-driver-cuda] score capture has no hook sideband arena; \
                 refusing the capture"
            );
            return me;
        };
        let requests = u32::try_from(obs.num_requests).unwrap_or(0);
        let indptr = std::mem::take(&mut scratch.raw_offsets_i32);
        let acquired = me.buf.acquire(
            ops,
            arena,
            totals.raw_total,
            totals.folded_total,
            &indptr,
            requests,
        );
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

    /// Normalize, fold heads and finalise the payload. The observation is
    /// re-read at publish time exactly as the C++ re-reads
    /// `hooks_->observation`; a fire whose geometry vanished mid-layer is an
    /// error, not a fold against a stale view.
    ///
    /// # Two kernels, in the capture dispatch's order
    ///
    /// [`ScoreOps::normalize_decode`] runs first and
    /// [`ScoreOps::fold_heads`] second, which is the order
    /// `dispatch_attention_flashinfer_decode_capture_bf16` issued them in:
    /// the normalize divides each `(request, head)` row by its total in
    /// place, and the fold averages the normalised heads into the published
    /// row. Reversing them would fold un-normalised scores and then divide
    /// nothing — a plausible row, not a crash, which is why the order is
    /// stated here rather than left to the reader.
    ///
    /// The normalize used to be the C++ dispatch's tail. It is here because
    /// this is where the same stream reaches the same buffer at the same
    /// point; see the module header.
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
        if self.published {
            self.payload.as_ref()
        } else {
            None
        }
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
        debug_assert!(
            !self.buf.held,
            "LayerScoreCapture dropped without release()"
        );
    }
}

/// Why a publish refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreError {
    /// The fire's observation vanished between capture and publish — the
    /// C++ throws `"attention score capture lost its fire geometry
    /// mid-layer"`.
    LostGeometry,
}

/// RAII capture of one layer's PREFILL scores — SnapKV's observation
/// window. Ports `LayerPrefillScoreCapture`: the raw rows carry the
/// window factor, and [`Self::publish`] fires the window-aware normalize and
/// the two-axis fold that were the C++ capture dispatch's tail.
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
    /// Construct the capture. Same gate chain as decode, plus the window,
    /// plus two ceilings: the int32 element bound the kernels index with,
    /// and the 1 GiB byte cap.
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
        // The int32 CSR is what the kernels index with, so the total has to
        // fit a signed 32-bit element offset — and the byte ceiling bites
        // first anyway.
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

    /// Normalize, fold and finalise the folded row.
    ///
    /// # It used to launch nothing, and that is what changed
    ///
    /// The doc that stood here said "folding is part of the capture
    /// dispatch, whose causal limits only it can derive". The second half is
    /// still true — `qo_indptr` and the causal window are what bound
    /// `rows = min(window, qo_len)`, and the kernels read them from the CSRs
    /// rather than being told. The first half was an artefact of the fold
    /// being C++: the dispatch derived nothing the caller could not pass,
    /// and the four pointers below are exactly what it passed.
    ///
    /// So both launches are here now, in the dispatch's order:
    /// [`ScoreOps::normalize_prefill`] divides each `(request, head,
    /// window-row)` by its total in place, then
    /// [`ScoreOps::fold_prefill`] collapses heads AND window rows into
    /// `folded`. The order is load-bearing for the same reason the decode's
    /// is — folding first would average un-normalised scores and produce a
    /// plausible row rather than a fault.
    ///
    /// `qo_indptr_d` is the fire's query CSR, device side; the other three
    /// arguments are the KV page CSR, the last-page lengths and the page
    /// size, matching [`LayerScoreCapture::publish`].
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
        if self.published {
            self.payload.as_ref()
        } else {
            None
        }
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
        debug_assert!(
            !self.buf.held,
            "LayerPrefillScoreCapture dropped without release()"
        );
    }
}

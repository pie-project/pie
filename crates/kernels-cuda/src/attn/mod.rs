#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;

use crate::jit::abi::{Elem, bf16, f16};
use crate::jit::{Abi, Ctx, Family, Launch, Routine};
// The YaRN quartet is `rope`'s, and attention names it because the fused
// prepare rotates on the way in. One import across a family boundary, which
// is the shape `x::cx` existed to avoid and was not worth a module for.
use crate::rope::Yarn;
use crate::{driver_bound, routine};
use kernels::Refusal;

// `routine!` names a `fn` by IDENTIFIER, so the host programs that live
// beside the root whose kernels they launch are brought into scope here to
// be named in `ROUTINES`. The routine's name is the identifier and not the
// path, which is what keeps it equal to its contract symbol's tail, and
// `driver_bound!` names its `fn` the same way.
//
// Two come from further off than a submodule of this file: the score fold
// from the root module that compiles it, and `split_qkv_bf16` from
// `driver_internal`, whose header says why that body stays there while the
// declaration is here.
use attention_flashinfer::attn_score_fold_heads;
use crate::driver_internal::split_qkv_bf16;
use dsv4_compress::{dsv4_compress_gather_paged_bf16, dsv4_store_comp_entries_bf16};
use kv_paged::{
    dequant_kv_cache_layer_to_bf16_active, write_kv_explicit_bf16, write_kv_explicit_bf16_devwin,
    write_kv_to_pages,
};
use qkv_fused::{
    qkv_decode_fused_dispatch, qkv_decode_qk_norm_rope_write_kv_bf16,
    qkv_packed_qk_norm_rope_vnorm_write_kv_bf16,
};

/// The FlashInfer FA2 lattice, its host arithmetic and its param structs.
pub mod fa2;
/// The scheduler: `flashinfer/attention/scheduler.cuh` as host Rust.
pub mod plan;
/// XQA's decode kernel and the five members its host program picks between.
pub mod xqa;

// No `cudarc::cublas` import: this family launches `__global__`s and nothing
// else now that MLA's absorb pair is `gemm::absorb`'s.

/// `pie::attn::KvScheme` — how a paged KV bank is quantised, as the device
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_scheme(pub u8);

impl kv_scheme {
    /// The device spelling of a [`KvLayer::scheme`], for the one operand that
    #[must_use]
    pub const fn of(scheme: crate::attn::KvScheme) -> Self {
        Self(scheme as i32 as u8)
    }
}

impl crate::jit::Abi for kv_scheme {
    const CPP: &'static str = "::pie::attn::KvScheme";
    const TY: kernels::Ty = kernels::Ty::KvScheme;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::U8(self.0)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::U8(v) => Ok(Self(*v)),
            _ => Err(kernels::Refusal::Kind { at, want: kernels::Ty::KvScheme }),
        }
    }
}

crate::arg_via_abi!(kv_scheme);

/// `pie::attn::KvDType` — what a page element actually is.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_dtype(pub u8);

impl kv_dtype {
    /// The device spelling of a [`KvLayer::storage_dtype`], for the one operand
    #[must_use]
    pub const fn of(dtype: crate::attn::KvDType) -> Self {
        Self(dtype as i32 as u8)
    }
}

impl crate::jit::Abi for kv_dtype {
    const CPP: &'static str = "::pie::attn::KvDType";
    const TY: kernels::Ty = kernels::Ty::KvDType;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::U8(self.0)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::U8(v) => Ok(Self(*v)),
            _ => Err(kernels::Refusal::Kind { at, want: kernels::Ty::KvDType }),
        }
    }
}

crate::arg_via_abi!(kv_dtype);

/// `attn`'s `#[repr(C)]` mirrors of C++ aggregates, and their measured
pub mod params {
    use core::ffi::c_void;

    use kernels::Ty;

    /// One lane's structured-mask descriptor, as
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct StructuredMaskParams {
        /// `kind` — 1 causal, 2 sliding window, 3 sink. See the type's doc
        pub kind: u32,
        /// `window` — the sliding window's extent in keys, for kinds 2 and 3.
        pub window: u32,
        /// `sink` — the attention-sink width in keys, for kind 3: every key
        pub sink: u32,
    }

    /// How C++ spells the struct itself, for the `static_assert`s
    const STRUCTURED_MASK_PARAMS: &str =
        "::pie::attn::StructuredMaskParams";

    /// The array of descriptors, one per lane, as `pack_structured_mask`
    impl crate::jit::Abi for *const StructuredMaskParams {
        const CPP: &'static str =
            "const ::pie::attn::StructuredMaskParams*";
        const TY: Ty = Ty::StructuredMasks;
        fn arg(&self) -> crate::jit::ArgValue {
            crate::jit::ArgValue::Ptr(*self as *mut c_void)
        }
        fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
            match value {
                crate::jit::ArgValue::Ptr(p) => Ok(p.cast::<StructuredMaskParams>().cast_const()),
                _ => Err(kernels::Refusal::Kind { at, want: Ty::StructuredMasks }),
            }
        }
    }

    crate::arg_via_abi!(*const StructuredMaskParams);

    const _: () = assert!(
        ::core::mem::size_of::<StructuredMaskParams>() == 12,
        "StructuredMaskParams: sizeof disagrees with the measured \
         pie::attn::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::align_of::<StructuredMaskParams>() == 4,
        "StructuredMaskParams: alignof disagrees with the measured \
         pie::attn::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, kind) == 0,
        "StructuredMaskParams.kind: offset disagrees with the measured \
         pie::attn::StructuredMaskParams::kind",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, window) == 4,
        "StructuredMaskParams.window: offset disagrees with the measured \
         pie::attn::StructuredMaskParams::window",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, sink) == 8,
        "StructuredMaskParams.sink: offset disagrees with the measured \
         pie::attn::StructuredMaskParams::sink",
    );

    /// The measured layout, as C++ `static_assert`s.
    pub static LAYOUTS: &[crate::jit::Layout] = &[crate::jit::Layout {
        cpp: STRUCTURED_MASK_PARAMS,
        size: 12,
        align: 4,
        fields: &[("kind", 0), ("window", 4), ("sink", 8)],
        probe: "nvrtc-probes/attn_structured_mask.py",
    }];
}

/// `attn/attention_flashinfer.cuh` — the per-head → per-request score fold.
pub mod attention_flashinfer {
    use crate::jit::{Ctx, Launch};
    use crate::jit::Abi;
    use kernels::Refusal;

    /// `attn::attn_score_fold_heads_dev` — the per-head rows of one request
    /// averaged into the one row a policy reads.
    ///
    /// The `_dev` suffix is the device symbol's, not the trace symbol's:
    /// `attn::attn_score_fold_heads` is a contract on a STATEMENT and a
    /// contract symbol may never also be a kernel's, which is why the two are
    /// spelled apart.
    ///
    /// What the caller must guarantee, as `call()` states it: `scores`
    /// addresses the raw per-head rows the capture kernel wrote, `folded` one
    /// writable float per KV position per request, `score_indptr` the capture
    /// CSR, and the two KV arrays `num_requests + 1` and `num_requests`
    /// entries.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] for an empty fire — a legal no-op that must not
    /// reach a launch, since a zero `grid.x` is itself a refusal — and
    /// whatever the compile, the load or the launch refuses.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_score_fold_heads(
        ctx: &Ctx,
        scores: *const f32,
        score_indptr: *const i32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        folded: *mut f32,
    ) -> Result<(), Refusal> {
        /// The fold's block width: `attention_flashinfer.cu:829`'s `256`.
        ///
        /// Load-bearing and not tuning: the kernel folds warp partials through
        /// `__shared__ float red[256 / 32]`, so a launch at another width would
        /// read reduction slots nothing wrote — a plausible score row rather than
        /// a fault.
        const FOLD_BLOCK: u32 = 256;

        /// `attn/attention_flashinfer.cuh` — the root the fold compiles out of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
        /// The fold's grid fanout: `attention_flashinfer.cu:828`'s literal `64u`.
        ///
        /// **Not an extent.** The kernel's inner loop is
        /// `for (int i = blockIdx.y; i < n; i += gridDim.y)` over the request's KV
        /// positions, so `gridDim.y` is an OCCUPANCY FANOUT: every value of it
        /// computes the same floats, and `1` computes them correctly in a
        /// sixty-fourth of the blocks. That is why it is a citation here and not a
        /// rule anywhere — a rule is a function of the fire's rectangle and `64`
        /// is not in the rectangle. The only other grid-stride literal in
        /// `kernels/` is a *different* number
        /// ([`attention_score_post::PREFILL_FOLD_GRID_Y`]'s `32`), which is the
        /// clearest evidence available that neither is a rule.
        ///
        /// [`attention_score_post::PREFILL_FOLD_GRID_Y`]:
        ///     super::attention_score_post::PREFILL_FOLD_GRID_Y
        const FOLD_GRID_Y: u32 = 64;

        // `attention_flashinfer.cu:817` — `if (num_requests <= 0) return;`
        // `attention_flashinfer.cu:828-829`, transcribed. `num_requests` is
        // `grid.x` because the kernel indexes the request by `blockIdx.x`; the
        // second axis is `FOLD_GRID_Y` and is not an extent.
        //
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/attention_flashinfer.cuh",
                "::pie::attn::attn_score_fold_heads",
                Launch::grid([num_requests.unsigned_abs(), FOLD_GRID_Y, 1], [FOLD_BLOCK, 1, 1]),
                &[
                    scores.arg(),
                    score_indptr.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    page_size.arg(),
                    num_q_heads.arg(),
                    folded.arg(),
                ],
            )
        }
    }
}

/// `attn/attention_score_post.cuh` — the arithmetic that turns a captured
/// FlashInfer score buffer into the row a gate-score policy reads.
///
/// Three `__global__`s and their host programs. The FA2 kernel that fills
/// `scores` is still C++ — it is a template cross-product with hundreds of
/// instantiations — but everything downstream of it in the capture is here,
/// and the three launches were the TAIL of the two capture dispatches rather
/// than separate calls. The driver issues them at the point on its own stream
/// where those dispatches used to.
pub mod attention_score_post {
    use crate::jit::{Ctx, Launch};
    use crate::jit::Abi;
    use kernels::Refusal;

    /// `attn/attention_score_post.cuh` — the root all three compile out of.
    ///
    /// ONE root for the three, and that is a cost, not a tidiness: a compile
    /// is per instantiation, so nothing is shared between them at run time —
    /// but the three are one file because they are one program, and a split
    /// would be three files that must agree about a score row's layout.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
        /// The three post-kernels' block width: the literal `256` in all three
    /// `<<<grid, 256, 0, stream>>>` at `attention_flashinfer.cu:591`, `:923`
    /// and `:929`.
    ///
    /// Load-bearing: every one of them reduces through
    /// `__shared__ float red[256 / 32]`, one slot per warp, so a launch at
    /// another width reads reduction slots nothing wrote. That is a plausible
    /// score row rather than a fault.
    const NORMALIZE_BLOCK: u32 = 256;

    /// The prefill fold's grid fanout: `attention_flashinfer.cu:928`'s literal
    /// `32u`.
    ///
    /// The sibling of [`attention_flashinfer`]'s `FOLD_GRID_Y` and the same
    /// kind of thing — an OCCUPANCY FANOUT, not an extent.
    /// `attn_prefill_score_fold`'s inner loop strides `blockIdx.y` by
    /// `gridDim.y` over the request's KV positions, so every value of it
    /// computes the same floats.
    ///
    /// [`attention_flashinfer`]: super::attention_flashinfer
    pub const PREFILL_FOLD_GRID_Y: u32 = 32;

    /// `attn::attn_score_normalize` — the decode capture's divide-by-total,
    /// in place.
    ///
    /// `dispatch_attention_flashinfer_decode_capture_bf16`'s tail, immediately
    /// after its `CUDA_CHECK(status)`:
    ///
    /// ```text
    /// const dim3 grid(cache.num_requests, cache.num_q_heads);
    /// device::attn_score_normalize<<<grid, 256, 0, stream>>>(
    ///     score_out, score_indptr_d, kv_page_indptr_d, kv_last_page_lens_d,
    ///     cache.page_size);
    /// ```
    ///
    /// Five operands, two grid extents, one block width. `kv_len` is DERIVED
    /// from the page CSR inside the body rather than passed, which is why no
    /// length appears here — `attention_score_post.cuh` argues that beside the
    /// body, and a caller must not "helpfully" add one.
    ///
    /// What the caller must guarantee, as `call()` states it: `scores`
    /// addresses the capture's raw rows and is writable, `score_indptr` the
    /// capture CSR, and the two KV arrays `num_requests + 1` and
    /// `num_requests` entries.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] for an empty fire on either grid axis, and whatever
    /// the compile, the load or the launch refuses.
    pub fn attn_score_normalize(
        ctx: &Ctx,
        scores: *mut f32,
        score_indptr: *const i32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
    ) -> Result<(), Refusal> {
        // The C++ had no such guard, because the dispatch above it could not
        // be reached with an empty fire. It is required here: a zero grid axis
        // reaching a launch is a refusal, which would turn a legal no-op into
        // one.
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/attention_score_post.cuh",
                "::pie::attn::attn_score_normalize",
                Launch::grid(
                    [num_requests.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
                    [NORMALIZE_BLOCK, 1, 1],
                ),
                &[
                    scores.arg(),
                    score_indptr.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    page_size.arg(),
                ],
            )
        }
    }

    /// `attn::attn_prefill_score_normalize` — the same, over SnapKV's
    /// observation window.
    ///
    /// `dispatch_attention_flashinfer_prefill_capture_bf16`'s tail:
    ///
    /// ```text
    /// const dim3 norm_grid(cache.num_requests, cache.num_q_heads, window);
    /// device::attn_prefill_score_normalize<<<norm_grid, 256, 0, stream>>>(
    ///     score_out, score_indptr_d, qo_indptr_d, kv_page_indptr_d,
    ///     kv_last_page_lens_d, cache.page_size, window);
    /// ```
    ///
    /// `window` is BOTH the third grid extent and the last operand, and the
    /// duplication is the launcher's rather than an oversight to tidy:
    /// `blockIdx.z` selects the window row and the operand bounds
    /// `rows = min(window, qo_len)` inside the body. Passing one and deriving
    /// the other would be a different kernel.
    ///
    /// [`attn_score_normalize`]'s obligation, plus `qo_indptr` addressing the
    /// fire's query CSR.
    ///
    /// # Errors
    ///
    /// As [`attn_score_normalize`], with `window` a third empty axis.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_score_normalize(
        ctx: &Ctx,
        scores: *mut f32,
        score_indptr: *const i32,
        qo_indptr: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    ) -> Result<(), Refusal> {
        // SAFETY: as [`attn_score_normalize`]'s.
        unsafe {
            ctx.launch(
                "attn/attention_score_post.cuh",
                "::pie::attn::attn_prefill_score_normalize",
                Launch::grid(
                    [
                        num_requests.unsigned_abs(),
                        num_q_heads.unsigned_abs(),
                        window.unsigned_abs(),
                    ],
                    [NORMALIZE_BLOCK, 1, 1],
                ),
                &[
                    scores.arg(),
                    score_indptr.arg(),
                    qo_indptr.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    page_size.arg(),
                    window.arg(),
                ],
            )
        }
    }

    /// `attn::attn_prefill_score_fold` — heads AND window rows collapsed into
    /// the published row.
    ///
    /// The last two statements of
    /// `dispatch_attention_flashinfer_prefill_capture_bf16`:
    ///
    /// ```text
    /// const dim3 fold_grid(cache.num_requests, 32u);
    /// device::attn_prefill_score_fold<<<fold_grid, 256, 0, stream>>>(
    ///     score_out, folded_out, score_indptr_d, qo_indptr_d,
    ///     kv_page_indptr_d, kv_last_page_lens_d, cache.page_size,
    ///     cache.num_q_heads, window);
    /// ```
    ///
    /// `num_q_heads` is an OPERAND here and a grid extent in the normalize
    /// above — the fold collapses the head axis rather than indexing it, so it
    /// must know the count without having a block per head.
    ///
    /// Not in place, unlike its two siblings: it reads `scores` and writes
    /// `folded`.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] for an empty fire, and whatever the compile, the
    /// load or the launch refuses.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_score_fold(
        ctx: &Ctx,
        scores: *const f32,
        folded: *mut f32,
        score_indptr: *const i32,
        qo_indptr: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    ) -> Result<(), Refusal> {
        // SAFETY: as [`attn_score_normalize`]'s, and `folded` is written
        // rather than read.
        unsafe {
            ctx.launch(
                "attn/attention_score_post.cuh",
                "::pie::attn::attn_prefill_score_fold",
                Launch::grid(
                    [num_requests.unsigned_abs(), PREFILL_FOLD_GRID_Y, 1],
                    [NORMALIZE_BLOCK, 1, 1],
                ),
                &[
                    scores.arg(),
                    folded.arg(),
                    score_indptr.arg(),
                    qo_indptr.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    page_size.arg(),
                    num_q_heads.arg(),
                    window.arg(),
                ],
            )
        }
    }

}

/// `attn/dsa_indexer.cuh` — glm5's sparse-attention index network, three
pub mod dsa_indexer {
    

    /// `attn/dsa_indexer.cuh` — the root these routines compile a symbol out
    /// of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
        /// `dsa_indexer.cuh`'s `kBlock`, which `knorm_rope` and `topk_mask` both
    pub const K_BLOCK: u32 = 256;

    /// `dsa_indexer.cu:34-35` — `index_q_rope`'s block width.
    #[must_use]
    pub fn q_rope_block(n_heads: i32) -> u32 {
        let rounded = ((n_heads.max(0) + 31) / 32) * 32;
        #[allow(clippy::cast_sign_loss)]
        let block = rounded as u32;
        if block < 32 { 32 } else { block }
    }
}

/// `attn/page_compact.cuh` — dropping the pages a keep-mask rejects and
pub mod page_compact {
    

    /// `attn/page_compact.cuh` — the root both halves compile out of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
        /// `page_compact.cuh:114` — `constexpr int kBlock = 256`.
    pub const K_BLOCK: u32 = 256;
}

/// `attn::compact_page_csr` — the page compactor's host program.
///
/// Two launches, ordered by the stream: the count writes one survivor count
/// per request into `scratch_counts` and the scan reads it back.
///
/// What the caller must guarantee, as `call()` states it: every pointer is a
/// device address live across BOTH launches — `scratch_counts` especially,
/// which carries the dependency between them.
pub fn compact_page_csr(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    if scratch_counts.is_null() {
        return Err(Refusal::Absent { what: "the compaction scratch buffer" });
    }
    let launch = Launch::per_row(num_requests.unsigned_abs(), page_compact::K_BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as. The two launches are
    // ordered by the stream, and the second reads what the first wrote.
    unsafe {
        ctx.launch(
            "attn/page_compact.cuh",
            "::pie::attn::count_kept<::pie::i32(256)>",
            launch,
            &[
                page_indptr_in.arg(),
                keep.arg(),
                keep_stride.arg(),
                num_requests.arg(),
                scratch_counts.arg(),
            ],
        )?;
        ctx.launch(
            "attn/page_compact.cuh",
            "::pie::attn::scan_and_scatter<::pie::i32(256)>",
            launch,
            &[
                page_indices_in.arg(),
                page_indptr_in.arg(),
                last_page_lens_in.arg(),
                keep.arg(),
                scratch_counts.cast_const().arg(),
                keep_stride.arg(),
                num_requests.arg(),
                page_indptr_out.arg(),
                last_page_lens_out.arg(),
                page_indices_out.arg(),
            ],
        )
    }
}

/// `attn/attention_naive.cuh` — the MTP pair and the reference attention.
pub mod attention_naive {
    

    /// `attn/attention_naive.cuh` — the root these routines compile a symbol
    /// out of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The reference attention itself — `attn::attention_naive_bf16` — has no
    /// routine in this file, so no constant here names it.
        /// `attention_naive.cu:57` — `constexpr int BLOCK = device::BLOCK;`,
    pub const BLOCK: u32 = 256;
}

/// `attn::mtp_shift_hidden_bf16` — one block per TOKEN.
///
/// What the caller must guarantee, as `call()` states it: every pointer is a
/// device address live across the launch.
pub fn mtp_shift_hidden<T>(
    ctx: &Ctx,
    target_hidden: *const T,
    pending_hidden: *const T,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    out: *mut T,
    total_tokens: i32,
    num_requests: i32,
    hidden_size: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    if pending_hidden.is_null() {
        return Err(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attention_naive.cuh",
            &format!("::pie::attn::mtp_shift_hidden<{}>", T::CPP),
            Launch::per_row(total_tokens.unsigned_abs(), attention_naive::BLOCK),
            &[
                target_hidden.arg(),
                pending_hidden.arg(),
                qo_indptr.arg(),
                slot_ids.arg(),
                out.arg(),
                num_requests.arg(),
                hidden_size.arg(),
            ],
        )
    }
}

/// `attn::mtp_update_pending_hidden_bf16` — one block per REQUEST.
///
/// [`mtp_shift_hidden`]'s obligation.
pub fn mtp_update_pending_hidden<T>(
    ctx: &Ctx,
    target_hidden: *const T,
    pending_hidden: *mut T,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    num_requests: i32,
    hidden_size: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    if pending_hidden.is_null() {
        return Err(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attention_naive.cuh",
            &format!("::pie::attn::mtp_update_pending_hidden<{}>", T::CPP),
            Launch::per_row(num_requests.unsigned_abs(), attention_naive::BLOCK),
            &[
                target_hidden.arg(),
                pending_hidden.arg(),
                qo_indptr.arg(),
                slot_ids.arg(),
                num_requests.arg(),
                hidden_size.arg(),
            ],
        )
    }
}

/// `attn::mla_prepare_bf16` — the whole MLA prologue in one kernel.
///
/// What the caller must guarantee, as `call()` states it: every pointer is a
/// device address live across the launch, `layer`'s two page pointers
/// included.
#[allow(clippy::similar_names)]
pub fn mla_prepare_bf16(
    ctx: &Ctx,
    layer: MlaLayer,
    kv_a: *const bf16,
    kv_a_norm_weight: *const bf16,
    q_b: *const bf16,
    kv_c: *mut bf16,
    k_pe: *mut bf16,
    q_nope: *mut bf16,
    q_pe: *mut bf16,
    positions: *const i32,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    row_valid: *const u8,
    total_tokens: i32,
    num_requests: i32,
    heads: i32,
    qk_nope_head_dim: i32,
    eps: f32,
    theta: f32,
    interleaved: bool,
    kv_a_row_stride: i32,
    yarn: Option<Yarn>,
) -> Result<(), Refusal> {
    /// `mla_paged.cu:65` — the grid's second axis, less its KV lane.
    #[must_use]
    pub fn mla_q_blocks(heads: i32, heads_per_block: i32) -> i32 {
    if heads_per_block <= 0 {
    return 0;
    }
    heads.saturating_add(heads_per_block - 1) / heads_per_block
    }

    /// `mla_paged.cu:64` — the query lane's head packing.
    #[must_use]
    pub fn mla_heads_per_block(rope: i32) -> i32 {
    let half = rope / 2;
    if half >= MLA_PREPARE_BLOCK {
    1
    } else if half > 0 {
    MLA_PREPARE_BLOCK / half
    } else {
    1
    }
    }

    /// `mla_paged.cu:52` — `constexpr int BS = 256;`, the prepare block.
    pub const MLA_PREPARE_BLOCK: i32 = 256;

    let kv_lora = layer.kv_lora_rank;
    let rope = layer.qk_rope_head_dim;
    let stride = if kv_a_row_stride > 0 { kv_a_row_stride } else { kv_lora + rope };
    let per_block = mla_heads_per_block(rope);
    let blocks = mla_q_blocks(heads, per_block);

    let (low_dim, high_dim) = match yarn {
        Some(y) => crate::rope::ramp_bounds(
            rope,
            theta,
            y.beta_fast,
            y.beta_slow,
            y.original_max_position,
        ),
        None => (0.0, 0.0),
    };
    let yarn_factor = yarn.map_or(-1.0_f32, |y| y.factor);
    let yarn_mscale = yarn.map_or(1.0_f32, |y| y.attention_factor);

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/mla_paged.cuh",
            "::pie::attn::mla_prepare<::pie::i32(256)>",
            Launch::grid(
                [total_tokens.unsigned_abs(), blocks.saturating_add(1).max(1).unsigned_abs(), 1],
                [MLA_PREPARE_BLOCK.unsigned_abs(), 1, 1],
            ),
            &[
                kv_a.arg(),
                kv_a_norm_weight.arg(),
                q_b.arg(),
                kv_c.arg(),
                k_pe.arg(),
                q_nope.arg(),
                q_pe.arg(),
                layer.ckv_pages.cast::<bf16>().arg(),
                layer.kpe_pages.cast::<bf16>().arg(),
                positions.arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                row_valid.arg(),
                num_requests.arg(),
                layer.page_size.arg(),
                heads.arg(),
                kv_lora.arg(),
                qk_nope_head_dim.arg(),
                rope.arg(),
                stride.arg(),
                eps.arg(),
                theta.arg(),
                interleaved.arg(),
                per_block.arg(),
                yarn_factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
                yarn_mscale.arg(),
            ],
        )
    }
}

/// `attn::write_mla_to_pages` — appends one step's compressed latent and rope
///
/// [`mla_prepare_bf16`]'s obligation.
pub fn write_mla_to_pages(
    ctx: &Ctx,
    layer: MlaLayer,
    ckv_curr: *const bf16,
    kpe_curr: *const bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    row_valid: *const u8,
    total_tokens: i32,
    num_requests: i32,
) -> Result<(), Refusal> {
    /// `mla_paged.cu:105` — `write_mla`'s block, one per token row.
    pub const MLA_WRITE_BLOCK: u32 = 256;

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/mla_paged.cuh",
            "::pie::attn::write_mla",
            Launch::per_row(total_tokens.unsigned_abs(), MLA_WRITE_BLOCK),
            &[
                ckv_curr.arg(),
                kpe_curr.arg(),
                layer.ckv_pages.cast::<bf16>().arg(),
                layer.kpe_pages.cast::<bf16>().arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                row_valid.arg(),
                num_requests.arg(),
                layer.page_size.arg(),
                layer.kv_lora_rank.arg(),
                layer.qk_rope_head_dim.arg(),
            ],
        )
    }
}

/// `dsv4_compress.cu:139` and `:161` — the boundary-meta block.
const DSV4_META_BLOCK: u32 = 128;

/// `attn::dsv4_boundary_meta_decode` — each decode row's compressed-block
///
/// What the caller must guarantee, as `call()` states it: every pointer is a
/// device address live across the launch.
pub fn dsv4_boundary_meta_decode(
    ctx: &Ctx,
    positions: *const i32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    ratio: i32,
    row_valid: *const u8,
) -> Result<(), Refusal> {
    if ratio <= 0 {
        return Err(Refusal::Narrow { what: "ratio", at: i64::from(ratio) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsv4_compress.cuh",
            "::pie::attn::dsv4_boundary_meta_decode<::pie::i32>",
            Launch::flat(n.unsigned_abs(), DSV4_META_BLOCK),
            &[
                positions.arg(),
                out_pos.arg(),
                out_req.arg(),
                out_rope.arg(),
                n.arg(),
                ratio.arg(),
                row_valid.arg(),
            ],
        )
    }
}

/// `attn::dsv4_boundary_meta_paged` — the prefill form of
///
/// [`dsv4_boundary_meta_decode`]'s obligation.
pub fn dsv4_boundary_meta_paged(
    ctx: &Ctx,
    positions: *const i32,
    qo_indptr: *const u32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    num_requests: i32,
    ratio: i32,
    row_valid: *const u8,
) -> Result<(), Refusal> {
    if ratio <= 0 {
        return Err(Refusal::Narrow { what: "ratio", at: i64::from(ratio) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsv4_compress.cuh",
            "::pie::attn::dsv4_boundary_meta_paged<::pie::i32>",
            Launch::flat(n.unsigned_abs(), DSV4_META_BLOCK),
            &[
                positions.arg(),
                qo_indptr.arg(),
                out_pos.arg(),
                out_req.arg(),
                out_rope.arg(),
                n.arg(),
                num_requests.arg(),
                ratio.arg(),
                row_valid.arg(),
            ],
        )
    }
}

/// `attn::attention_compressed_paged_bf16` — attention against the COMPRESSED
///
/// [`dsv4_boundary_meta_decode`]'s obligation.
pub fn attention_compressed_paged_bf16(
    ctx: &Ctx,
    q: *const bf16,
    comp_kv_pages: *const bf16,
    o: *mut bf16,
    lse_out: *mut f32,
    positions: *const i32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    req_of_token: *const i32,
    total_tokens: i32,
    num_q_heads: i32,
    head_dim: i32,
    ratio: i32,
    page_size: i32,
    sm_scale: f32,
) -> Result<(), Refusal> {
    /// `dsv4_compress.cu:37` — `constexpr int ATTN_BLOCK = 128;`.
    const DSV4_ATTN_BLOCK: u32 = 128;

    let smem = head_dim
        .max(0)
        .unsigned_abs()
        .saturating_add(DSV4_ATTN_BLOCK)
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsv4_compress.cuh",
            "::pie::attn::compressed_attn_paged",
            Launch::grid(
                [total_tokens.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
                [DSV4_ATTN_BLOCK, 1, 1],
            )
            .smem(smem),
            &[
                q.arg(),
                comp_kv_pages.arg(),
                o.arg(),
                lse_out.arg(),
                positions.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                req_of_token.arg(),
                num_q_heads.arg(),
                head_dim.arg(),
                ratio.arg(),
                page_size.arg(),
                sm_scale.arg(),
            ],
        )
    }
}

/// `attn::dsa_index_knorm_rope_bf16` — LayerNorm then interleaved RoPE on the
///
/// What the caller must guarantee, as `call()` states it: every pointer is a
/// device address live across the launch.
pub fn dsa_index_knorm_rope<T>(
    ctx: &Ctx,
    idx_k: *mut T,
    k_norm_weight: *const T,
    k_norm_bias: *const T,
    positions: *const i32,
    tokens: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    eps: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsa_indexer.cuh",
            &format!("::pie::attn::index_knorm_rope<{}>", T::CPP),
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::K_BLOCK),
            &[
                idx_k.arg(),
                k_norm_weight.arg(),
                k_norm_bias.arg(),
                positions.arg(),
                head_dim.arg(),
                rope_dim.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `attn::dsa_index_q_rope_bf16` — interleaved RoPE on the indexer's QUERY
///
/// [`dsa_index_knorm_rope`]'s obligation.
pub fn dsa_index_q_rope<T>(
    ctx: &Ctx,
    idx_q: *mut T,
    positions: *const i32,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsa_indexer.cuh",
            &format!("::pie::attn::index_q_rope<{}>", T::CPP),
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::q_rope_block(n_heads)),
            &[
                idx_q.arg(),
                positions.arg(),
                n_heads.arg(),
                head_dim.arg(),
                rope_dim.arg(),
                theta.arg(),
            ],
        )
    }
}

/// `attn::dsa_index_topk_mask` — score every causal (query, key) pair and
///
/// [`dsa_index_knorm_rope`]'s obligation.
pub fn dsa_index_topk_mask(
    ctx: &Ctx,
    idx_q: *const bf16,
    idx_k: *const bf16,
    idx_w: *const bf16,
    mask: *mut u8,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    topk: i32,
) -> Result<(), Refusal> {
    let smem = tokens
        .unsigned_abs()
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsa_indexer.cuh",
            "::pie::attn::index_topk_mask<::pie::bf16>",
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::K_BLOCK).smem(smem),
            &[
                idx_q.arg(),
                idx_k.arg(),
                idx_w.arg(),
                mask.arg(),
                tokens.arg(),
                n_heads.arg(),
                head_dim.arg(),
                topk.arg(),
            ],
        )
    }
}

/// `flashinfer::MLAParams` — measured, mirrored, and pinned with
pub mod mla_params {
    use super::bf16;
    use crate::by_value;
    use crate::jit::{ByValue, Layout};

    /// `flashinfer::uint_fastdiv` — twenty-four bytes, and that is the whole
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct UintFastdiv {
        /// `impl_` and `d_` together, unreachable individually.
        pub opaque: [u64; 3],
    }

    impl UintFastdiv {
        /// Build the pair the device halves read, by the shim's algorithm.
        #[must_use]
        pub const fn new(d: u32) -> Self {
            let d64 = d as u64;
            let magic = if d == 0 {
                0
            } else {
                let q = u64::MAX / d64;
                let r = u64::MAX % d64;
                q + if r + 1 == d64 { 1 } else { 0 } + 1
            };
            Self { opaque: [d64, magic, d64] }
        }
    }

    /// `flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>`, at
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct MlaParams {
        /// @0 — the non-positional half of Q.
        pub q_nope: *mut bf16,
        /// @8 — the rotary half of Q.
        pub q_pe: *mut bf16,
        /// @16 — the compressed KV cache.
        pub ckv: *mut bf16,
        /// @24 — the K positional cache.
        pub kpe: *mut bf16,
        /// @32 — split-K partial output.
        pub partial_o: *mut bf16,
        /// @40 — split-K partial log-sum-exp, always f32.
        pub partial_lse: *mut f32,
        /// @48 — the merged output.
        pub final_o: *mut bf16,
        /// @56 — the merged log-sum-exp, always f32.
        pub final_lse: *mut f32,
        /// @64
        pub q_indptr: *mut i32,
        /// @72
        pub kv_indptr: *mut i32,
        /// @80
        pub partial_indptr: *mut i32,
        /// @88
        pub merge_packed_offset_start: *mut i32,
        /// @96
        pub merge_packed_offset_end: *mut i32,
        /// @104
        pub merge_partial_packed_offset_start: *mut i32,
        /// @112
        pub merge_partial_packed_offset_end: *mut i32,
        /// @120
        pub merge_partial_stride: *mut i32,
        /// @128
        pub kv_indices: *mut i32,
        /// @136
        pub q_len: *mut i32,
        /// @144
        pub kv_len: *mut i32,
        /// @152
        pub q_start: *mut i32,
        /// @160
        pub kv_start: *mut i32,
        /// @168
        pub kv_end: *mut i32,
        /// @176 — the persistent-kernel work queue.
        pub work_indptr: *mut i32,
        /// @184 — **twenty-four bytes**, not four. See the module doc.
        pub block_size: UintFastdiv,
        /// @208 — twenty-four bytes.
        pub num_heads: UintFastdiv,
        /// @232
        pub q_nope_stride_n: u32,
        /// @236
        pub q_nope_stride_h: u32,
        /// @240
        pub q_pe_stride_n: u32,
        /// @244
        pub q_pe_stride_h: u32,
        /// @248
        pub ckv_stride_page: u32,
        /// @252
        pub ckv_stride_n: u32,
        /// @256
        pub kpe_stride_page: u32,
        /// @260
        pub kpe_stride_n: u32,
        /// @264
        pub o_stride_n: u32,
        /// @268
        pub o_stride_h: u32,
        /// @272
        pub sm_scale: f32,
        /// @276 — per-tensor symmetric dequant scale for an fp8 `ckv`.
        pub ckv_scale: f32,
        /// @280 — the same for `kpe`, and the same warning.
        pub kpe_scale: f32,
        /// @284 — one byte, then three of tail padding to reach 288.
        pub return_lse_base_on_e: bool,
    }

    by_value! {
        MlaParams as "::flashinfer::MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>",
        untagged,
        probe = "nvrtc-probes/attn_mla_params.py",
        size = 288, align = 8,
        {
            q_nope               @ 0   as "q_nope",
            work_indptr          @ 176 as "work_indptr",
            block_size           @ 184 as "block_size",
            num_heads            @ 208 as "num_heads",
            q_nope_stride_n      @ 232 as "q_nope_stride_n",
            o_stride_h           @ 268 as "o_stride_h",
            sm_scale             @ 272 as "sm_scale",
            ckv_scale            @ 276 as "ckv_scale",
            kpe_scale            @ 280 as "kpe_scale",
            return_lse_base_on_e @ 284 as "return_lse_base_on_e",
        }
    }

    /// The layouts this module pins.
    pub static LAYOUTS: &[Layout] = &[<MlaParams as ByValue>::LAYOUT];

    const _: () = assert!(
        ::core::mem::size_of::<UintFastdiv>() == 24,
        "UintFastdiv: sizeof disagrees with the measured ::flashinfer::uint_fastdiv \
         (24, NOT 4 — see nvrtc-probes/attn_mla_params.py)",
    );
    const _: () = assert!(
        ::core::mem::align_of::<UintFastdiv>() == 8,
        "UintFastdiv: alignof disagrees with the measured ::flashinfer::uint_fastdiv",
    );
}

/// `attn/attention_mla_naive.cuh` — the Blackwell MLA fallback pair,
///
/// A scalar flash-softmax kernel and a tensor-core one, the shape predicate
/// that chooses between them, and the shared-memory arithmetic each needs.
/// **Exactly ONE of the two fires per call**: they are alternatives and not a
/// sequence, [`mla_naive::plan`] returns whichever the shape admits, and
/// there is no intermediate buffer between them: both read the caller's paged
/// cache and write the caller's output, and the only scratch either has is
/// dynamic shared memory.
///
/// # The two grids are transposed and it is not a typo
///
/// ```text
/// attention_mla_naive.cuh:265   dim3 grid(total_tokens, num_heads / G);
/// attention_mla_naive.cuh:266   mla_naive_paged_kernel<grid, kMlaNaiveBlock, smem, stream>(...)
/// attention_mla_naive.cuh:725   dim3 grid(num_heads / kBM, total_tokens);
/// attention_mla_naive.cuh:726   mla_mma_paged_kernel<grid, kThreads, smem, stream>(...)
/// ```
///
/// The scalar arm puts tokens on x and head groups on y; the tensor-core arm
/// puts head blocks on x and tokens on y. Bringing them into agreement would
/// silently cap one: `grid.y` and `grid.z` are 16-bit-limited to 65 535 and
/// `grid.x` is not, so the scalar arm supports 2^31 tokens and 65 535 head
/// groups while the mma arm supports 65 535 tokens and 2^31 head blocks. Both
/// are stated as the C++ stated them. No [`kernels::LaunchRule`] opens either
/// rectangle, which is why [`mla_naive::plan`] builds the [`Launch`] itself.
///
/// # What `std::call_once` became
///
/// Both C++ launchers guarded their `cudaFuncSetAttribute` with a function
/// `static std::once_flag` (`:259-264` and `:717-723`). **It became once per
/// `(CUdevice, CUfunction)`, not once per process and not once per module**,
/// and it is not written here at all:
/// `jit/launch.rs`'s `raise_dynamic_smem_cap` does it on the way into every
/// launch, keyed on `(device, function)` with a high-water mark, whenever
/// `launch.smem` exceeds the 48 KiB default. This module sets
/// [`Launch::smem`] and nothing else.
///
/// **Once per process was a latent bug and the port does not carry it.** A
/// `std::once_flag` fires on the first call in the process, on whatever device
/// that call's context belonged to; `cuFuncSetAttribute` is per (device,
/// function). On a two-GPU box the second device never receives the opt-in and
/// its first launch fails with `CUDA_ERROR_INVALID_VALUE` — a diagnostic that
/// names the launch and not the missing attribute.
pub mod mla_naive {
    use super::bf16;
    use super::{Ctx, Launch, Refusal};
    
    use crate::jit::Abi;

    /// `attn/attention_mla_naive.cuh` — the root the fallback pair compiles
    /// out of.
    ///
        /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Neither is a template: both are plain `__global__`s, and the second
    /// sits a namespace deeper.
        /// `attention_mla_naive.cuh:45` — `constexpr int kMlaNaiveBlock = 256;`.
    ///
    /// A block width AND, through [`NAIVE_WARPS`], the divisor of the shared
    /// allocation: named once and used three times, which is why it is a
    /// constant here rather than a literal in [`plan`].
    pub const NAIVE_BLOCK: u32 = 256;

    /// `attention_mla_naive.cuh:46` — `kMlaNaiveBlock / 32`, the warps per
    /// block.
    ///
    /// It is the head-group ceiling as well as the warp count: the C++ starts
    /// `G` at `kMlaNaiveWarps` and halves, so 8 is the largest group any shape
    /// gets.
    pub const NAIVE_WARPS: i32 = NAIVE_BLOCK as i32 / 32;

    /// `attention_mla_naive.cuh:47` — `kv_lora_rank / 32` must not exceed this.
    ///
    /// The comment beside it is the measurement and travels with the number:
    /// *"kv_lora_rank <= 512 with 32 lanes"*.
    pub const NAIVE_MAX_PER: i32 = 16;

    /// `attention_mla_naive.cuh:48` — `qk_rope_head_dim / 32` must not exceed
    /// this.
    ///
    /// *"qk_rope_head_dim <= 128 with 32 lanes"*.
    pub const NAIVE_MAX_PE_PER: i32 = 4;

    /// `attention_mla_naive.cuh:239` — `const int kMlaWaveTarget = 296;`.
    ///
    /// The measurement it encodes is at `:235-237` and must survive the port
    /// verbatim, because nothing about the number can be re-derived from the
    /// code:
    ///
    /// > Pick the largest head group that still fills the machine. Every head
    /// > in a block walks the same keys, so a bigger group means the latent KV
    /// > is read from L1 instead of L2/HBM — but the grid is (tokens x
    /// > head-groups), so shrinking it too far starves the SMs. Two waves is
    /// > the target.
    ///
    /// 296 is two waves of 148 SMs, which is a B200. It is a target and not a
    /// bound: [`head_group`] stops halving when the grid first reaches it, so a
    /// shape that cannot reach 296 at `G == 1` simply runs at `G == 1`.
    pub const WAVE_TARGET: i64 = 296;

    /// `attention_mla_naive.cuh:238` — `constexpr int kForcedGroup = 0;`.
    ///
    /// An override left at its off value, kept because it documents that `G` is
    /// a tuning knob someone reached for. `0` means "use the wave-target
    /// search"; any positive value pins `G` and then halves it until it divides
    /// both `num_heads` and `kMlaNaiveWarps`. The pinned arm is transcribed in
    /// [`head_group_forced`] rather than dropped, so that turning it back on is
    /// a call-site change and not a re-derivation.
    pub const FORCED_GROUP: i32 = 0;

    /// `attention_mla_naive.cuh:324` — `constexpr int kBM = 16;`.
    ///
    /// *"query rows per block == heads"*. It is the mma `m16n8k16` tile's M and
    /// the divisor of the tensor-core grid's x axis, so a change here moves the
    /// grid.
    pub const MMA_BM: i32 = 16;

    /// `attention_mla_naive.cuh:682-687` — `mma_detail::smem_bytes()`,
    /// evaluated.
    ///
    /// ```text
    /// (kBM*kLdD + kStages*kBK*kLdD + kBM*kLdP) * sizeof(__nv_bfloat16)
    ///     + (kBM*kBK + 3*kBM) * sizeof(float)
    /// ```
    ///
    /// with `kBM = 16` (`:324`), `kBK = PIE_MLA_MMA_BK = 64` (`:325`, `:303`),
    /// `kStages = PIE_MLA_MMA_STAGES = 1` (`:327`, `:315`), `kLdD = kD + 8 =
    /// 584` (`:333`) and `kLdP = kBK + 8 = 72` (`:334`):
    ///
    /// ```text
    /// (16*584 + 1*64*584 + 16*72) * 2 + (16*64 + 3*16) * 4
    ///   = (9 344 + 37 376 + 1 152) * 2 + (1 024 + 48) * 4
    ///   = 95 744 + 4 288
    ///   = 100 032
    /// ```
    ///
    /// **Two independent measurements in the file corroborate it**, which is
    /// why it is written as a constant rather than recomputed here from five
    /// other constants that could each drift:
    ///
    /// * `:309-310` prices one extra pipeline stage at *"a full sK copy (73
    ///   KB)"*. `kStages*kBK*kLdD*2` is `64*584*2 = 74 752` = 73.0 KiB exactly.
    /// * `:311-313` says two stages *"on B200 drops the block occupancy from
    ///   2/SM to 1/SM"*. B200 has 228 KiB of shared memory per SM; `2 * 100 032
    ///   = 195.4 KiB` fits and `2 * 174 784` does not, and
    ///   `__launch_bounds__(kThreads, PIE_MLA_MMA_MINBLK = 2)` at `:420`/`:321`
    ///   asks for exactly the 2 that fits.
    ///
    /// **`:334`'s trailing comment `// 40` is stale and is not evidence.**
    /// `kLdP` is `kBK + 8`, which is 40 only at `kBK = 32`; the default has been
    /// 64 since `PIE_MLA_MMA_BK` was introduced. The two measurements above are
    /// computed against 64 and agree with it, so the comment is what drifted.
    /// Recorded here rather than corrected in place, because the `.cuh` is
    /// device text under a probe and this is the file that carries host
    /// arithmetic.
    pub const MMA_SMEM_BYTES: u32 = 100_032;

    /// The dynamic shared memory the scalar kernel asks for, in bytes.
    ///
    /// `attention_mla_naive.cuh:251-254`:
    ///
    /// ```text
    /// smem = (kMlaNaiveWarps * CKV + 2 * kMlaNaiveWarps) * sizeof(float)
    /// ```
    ///
    /// One `float` accumulator row per warp across the latent width, plus the
    /// two per-warp partial-softmax scalars (`m` and `l`).
    #[must_use]
    pub const fn naive_smem_bytes(kv_lora_rank: i32) -> u32 {
        let per = NAIVE_WARPS as i64 * kv_lora_rank as i64 + 2 * NAIVE_WARPS as i64;
        let bytes = per * 4;
        if bytes < 0 { 0 } else { bytes as u32 }
    }

    /// Whether an MLA naive launch ran, and which kernel.
    ///
    /// `#[must_use]` for [`crate::gemm::gemv`]'s reason: *"it declined"*
    /// must not be spellable like *"it ran"*.
    #[must_use]
    pub enum MlaNaive {
        /// The scalar flash-softmax kernel was launched on the caller's stream.
        LaunchedScalar,
        /// The tensor-core kernel was launched on the caller's stream.
        LaunchedMma,
        /// Nothing was launched, and why.
        Declined(NaiveDecline),
    }

    /// The four ways a naive MLA launch declines.
    ///
    /// Three of them were a `throw` in the C++ and one was a bare `return`, and
    /// the difference is preserved rather than flattened:
    /// [`NaiveDecline::NoTokens`] is a legal empty fire and the other three are
    /// the caller having asked for a shape this kernel pair cannot serve.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum NaiveDecline {
        /// `attention_mla_naive.cuh:211` — `if (total_tokens <= 0) return;`.
        ///
        /// A bare `return` in the C++, not a throw. Both kernels open one grid
        /// lane per token, so an empty batch is an empty grid, which the driver
        /// rejects as a geometry error.
        NoTokens,
        /// `attention_mla_naive.cuh:212-217` — a null `qo_indptr`,
        /// `kv_page_indptr` or `kv_last_page_lens`.
        ///
        /// A `throw` in the C++, whose message names all three: *"naive MLA:
        /// missing device indptr/lens (qo/kv_page_indptr/kv_last_page_lens)"*.
        /// The naive path is the only consumer of these three, which is why the
        /// FA2 path documents them as *"Ignored"* and defaults them to null.
        MissingIndptr,
        /// `attention_mla_naive.cuh:228-230` — `CKV % 32 != 0 || CKV / 32 > 16`.
        ///
        /// *"naive MLA: unsupported kv_lora_rank"*. 32 is the warp, and
        /// [`NAIVE_MAX_PER`] caps the per-lane accumulator array the kernel
        /// declares, so a wider latent would index past a register array.
        UnsupportedKvLoraRank,
        /// `attention_mla_naive.cuh:231-233` — `KPE % 32 != 0 || KPE / 32 > 4`.
        ///
        /// *"naive MLA: unsupported qk_rope_head_dim"*, and the same argument
        /// at [`NAIVE_MAX_PE_PER`].
        UnsupportedRopeDim,
    }

    /// `attention_mla_naive.cuh:698-701` — whether the tensor-core kernel
    /// applies.
    ///
    /// The C++ comment above it is the measurement and is reproduced whole,
    /// because it is the only record of which models the fast path serves:
    ///
    /// > Requires kv_lora_rank == 512, qk_rope_head_dim == 64 and
    /// > num_heads % 16 == 0 (true for GLM-5.2, Kimi K2.6 and both DeepSeek-V4
    /// > variants); anything else falls back to the scalar kernel.
    ///
    /// The three constants are `mma_detail::kCkv` (`:330`), `kKpe` (`:331`) and
    /// [`MMA_BM`] (`:324`) — the tile the `mma.sync` shapes are written
    /// against, not a tuning choice.
    ///
    /// **The C++ `forced` override is not reproduced.** `:692-697` is a
    /// `static const int forced = [] { return 0; if (false) return -1; if
    /// (false) return 1; return 0; }();` — a debug switch whose two live arms
    /// are unreachable after the first `return 0`, so it evaluates to 0 and the
    /// function is exactly the predicate below. Transcribing dead statements
    /// into a second language is how a debug switch becomes a feature.
    #[must_use]
    pub const fn mma_supported(kv_lora_rank: i32, qk_rope_head_dim: i32, num_heads: i32) -> bool {
        kv_lora_rank == 512 && qk_rope_head_dim == 64 && num_heads % MMA_BM == 0
    }

    /// `attention_mla_naive.cuh:241-249` — the head group `G`, by wave-target
    /// search.
    ///
    /// ```text
    /// int G = kMlaNaiveWarps;
    /// while (G > 1 && (num_heads % G != 0 ||
    ///                  (long long)total_tokens * (num_heads / G) < kMlaWaveTarget)) {
    ///     G >>= 1;
    /// }
    /// ```
    ///
    /// Start at 8 and halve until `G` divides the head count AND the resulting
    /// grid is at least [`WAVE_TARGET`] blocks. The loop terminates at `G == 1`,
    /// which divides everything, so a shape too small for two waves runs
    /// key-parallel — the degenerate case `:56-58` argues for: *"With `G == 1`
    /// this degenerates to the pure key-parallel layout, which is what small
    /// batches want (there the grid is the only source of parallelism)."*
    ///
    /// It is a SEARCH and not a formula, which is the whole reason no
    /// [`kernels::LaunchRule`] could state this launch: `G` divides the grid's
    /// head axis AND is passed to the kernel as its last argument.
    ///
    /// The multiplication is `long long` in the C++ and `i64` here for the same
    /// reason: `total_tokens * (num_heads / G)` is the grid's block count and a
    /// 32-bit product would wrap on a long prefill before it ever compared.
    #[must_use]
    pub fn head_group(num_heads: i32, total_tokens: i32) -> i32 {
        let mut g = NAIVE_WARPS;
        while g > 1
            && (num_heads % g != 0
                || i64::from(total_tokens) * i64::from(num_heads / g) < WAVE_TARGET)
        {
            g >>= 1;
        }
        g
    }

    /// `attention_mla_naive.cuh:242-244` — the pinned arm of the same choice.
    ///
    /// ```text
    /// G = kForcedGroup;
    /// while (G > 1 && (num_heads % G != 0 || kMlaNaiveWarps % G != 0)) G >>= 1;
    /// ```
    ///
    /// Reachable only when [`FORCED_GROUP`] is positive, which it is not. Kept
    /// because the two loops differ in a way that is easy to get wrong if it is
    /// ever re-derived: the forced arm tests `kMlaNaiveWarps % G` and NOT the
    /// wave target, so it never trades the L1 hit rate for occupancy — which is
    /// the whole reason someone would pin it.
    #[must_use]
    pub fn head_group_forced(num_heads: i32, forced: i32) -> i32 {
        let mut g = forced;
        while g > 1 && (num_heads % g != 0 || NAIVE_WARPS % g != 0) {
            g >>= 1;
        }
        g
    }

    /// Everything a naive MLA fire needs that is not a pointer.
    ///
    /// Grouped because the two kernels take fourteen scalars between them and a
    /// fourteen-argument `fn` is where an argument gets transposed. The field
    /// order is the C++ launcher's parameter order.
    ///
    /// It is a HOST aggregate and not a kernel argument, which is why nothing
    /// here is in [`super::ROUTINES`] — see the module's own note.
    #[derive(Clone, Copy, Debug)]
    pub struct NaiveShape {
        /// `layer.kv_lora_rank` — the latent width, `CKV` in the C++.
        pub kv_lora_rank: i32,
        /// `layer.qk_rope_head_dim` — the rope tail, `KPE` in the C++.
        pub qk_rope_head_dim: i32,
        /// Tokens per page.
        pub page_size: i32,
        /// Query rows in this batch.
        pub total_tokens: i32,
        /// Requests in this batch, for the CSR walk.
        pub num_requests: i32,
        /// Query heads.
        pub num_heads: i32,
        /// The softmax scale.
        pub sm_scale: f32,
        /// Whether the mask is causal.
        pub causal: bool,
        /// `index_mask`'s row stride; 0 when the mask is null.
        pub index_mask_stride: i32,
    }

    /// The device pointers both kernels take.
    ///
    /// Ordered as the `__global__`s declare them, which is the order [`fire`]
    /// binds them in.
    #[derive(Clone, Copy, Debug)]
    pub struct NaivePtrs {
        /// `[tokens, heads, kv_lora_rank]` bf16.
        pub q_nope: *const bf16,
        /// `[tokens, heads, qk_rope_head_dim]` bf16.
        pub q_pe: *const bf16,
        /// `[pages, page_size, kv_lora_rank]` bf16.
        pub ckv_pages: *const bf16,
        /// `[pages, page_size, qk_rope_head_dim]` bf16.
        pub kpe_pages: *const bf16,
        /// Per-request query offsets.
        pub qo_indptr: *const u32,
        /// The page list.
        pub kv_page_indices: *const u32,
        /// Per-request page offsets.
        pub kv_page_indptr: *const u32,
        /// Tokens used in each request's last page.
        pub kv_last_page_lens: *const u32,
        /// `[tokens, heads, kv_lora_rank]` bf16, written.
        pub o: *mut bf16,
        /// The DSA top-k mask, or null for dense.
        pub index_mask: *const u8,
    }

    /// Which kernel a shape selects and the rectangle it runs at.
    ///
    /// Separated from [`fire`] so the choice can be asserted without a CUDA
    /// context — the geometry is the part a test can check and the launch is
    /// not.
    #[must_use]
    pub enum NaivePlan {
        /// Fire [`"::pie::attn::mla_naive::mla_naive_paged_kernel"`] at this rectangle, with this head group.
        Scalar {
            /// The rectangle, `attention_mla_naive.cuh:265-266`.
            launch: Launch,
            /// `G`, which the kernel takes as its last argument AND which
            /// divided the grid's y axis. [`head_group`] computed it.
            head_group: i32,
        },
        /// Fire [`"::pie::attn::mla_naive::mma_detail::mla_mma_paged_kernel"`] at this rectangle.
        Mma {
            /// The rectangle, `attention_mla_naive.cuh:725-726`.
            launch: Launch,
        },
        /// Neither, and why.
        Declined(NaiveDecline),
    }

    /// Choose the kernel and build its rectangle.
    ///
    /// `attention_mla_naive.cuh:199-276` down to but not including the launch:
    /// the three refusals, the mma predicate, the head-group search and both
    /// shared-memory figures. No CUDA call, so this is the whole of what a test
    /// can pin.
    ///
    /// No `#[must_use]` here: [`NaivePlan`] carries it, and a plan that is
    /// dropped is a kernel that never ran.
    pub fn plan(shape: NaiveShape, have_indptr: bool) -> NaivePlan {
        /// `attention_mla_naive.cuh:329` — `kWarps * 32`, with
        /// `PIE_MLA_MMA_WARPS = 8`.
        pub const MMA_THREADS: u32 = 256;

        // `:211` — a bare `return`, not a throw.
        if shape.total_tokens <= 0 {
            return NaivePlan::Declined(NaiveDecline::NoTokens);
        }
        // `:212-217`.
        if !have_indptr {
            return NaivePlan::Declined(NaiveDecline::MissingIndptr);
        }
        // `:218-225` — the mma arm is tested BEFORE the scalar arm's shape
        // refusals, and the order is load-bearing: `kv_lora_rank == 512` passes
        // `CKV / 32 > kMlaNaiveMaxPer` at exactly 16, so the two agree today,
        // but the mma arm does not depend on the scalar arm's bounds and must
        // not start doing so if either constant moves.
        if mma_supported(shape.kv_lora_rank, shape.qk_rope_head_dim, shape.num_heads) {
            #[allow(clippy::cast_sign_loss)]
            // `:725` — `dim3 grid(num_heads / kBM, total_tokens);`. Head blocks
            // on x, tokens on y: the transpose of the scalar arm.
            let launch = Launch::grid(
                [(shape.num_heads / MMA_BM).max(0) as u32, shape.total_tokens.max(0) as u32, 1],
                [MMA_THREADS, 1, 1],
            )
            .smem(MMA_SMEM_BYTES);
            return NaivePlan::Mma { launch };
        }
        // `:227-233`.
        let ckv = shape.kv_lora_rank;
        let kpe = shape.qk_rope_head_dim;
        if ckv % 32 != 0 || ckv / 32 > NAIVE_MAX_PER {
            return NaivePlan::Declined(NaiveDecline::UnsupportedKvLoraRank);
        }
        if kpe % 32 != 0 || kpe / 32 > NAIVE_MAX_PE_PER {
            return NaivePlan::Declined(NaiveDecline::UnsupportedRopeDim);
        }
        // `:238-250`.
        let g = if FORCED_GROUP > 0 {
            head_group_forced(shape.num_heads, FORCED_GROUP)
        } else {
            head_group(shape.num_heads, shape.total_tokens)
        };
        #[allow(clippy::cast_sign_loss)]
        // `:265` — `dim3 grid(total_tokens, num_heads / G);`.
        let launch = Launch::grid(
            [shape.total_tokens.max(0) as u32, (shape.num_heads / g.max(1)).max(1) as u32, 1],
            [NAIVE_BLOCK, 1, 1],
        )
        .smem(naive_smem_bytes(ckv));
        NaivePlan::Scalar { launch, head_group: g }
    }

    /// `attention_mla_naive.cuh:199` — `launch_mla_naive_paged_raw`, whole.
    ///
    /// [`plan`] chooses; this binds the operands and fires. The operand order
    /// is each `__global__`'s own.
    ///
    /// Every pointer in `ptrs` must be a device address the caller keeps live
    /// across the launch — the obligation every `<<<>>>` made.
    ///
    /// # Errors
    ///
    /// Whatever the compile, the load or the launch refuses. A SHAPE this pair
    /// cannot serve is not an error: it is [`MlaNaive::Declined`], because
    /// three of the four were a `throw` and one a bare `return` and neither is
    /// a device failure.
    pub fn fire(ctx: &Ctx, ptrs: NaivePtrs, shape: NaiveShape) -> Result<MlaNaive, Refusal> {
        let have_indptr = !ptrs.qo_indptr.is_null()
            && !ptrs.kv_page_indptr.is_null()
            && !ptrs.kv_last_page_lens.is_null();
        match plan(shape, have_indptr) {
            NaivePlan::Declined(why) => Ok(MlaNaive::Declined(why)),
            NaivePlan::Mma { launch } => {
                // `attention_mla_naive.cuh:420-431` — the `__global__`'s
                // parameters. It takes NEITHER `kv_lora_rank` NOR
                // `qk_rope_head_dim`: both are `mma_detail` constants the
                // kernel is compiled against, which is why [`mma_supported`]
                // compares them rather than forwarding them.
                //
                // SAFETY: the caller's obligation on every pointer, above.
                unsafe {
                    ctx.launch("attn/attention_mla_naive.cuh", "::pie::attn::mla_naive::mma_detail::mla_mma_paged_kernel", launch, &[
                        ptrs.q_nope.arg(),
                        ptrs.q_pe.arg(),
                        ptrs.ckv_pages.arg(),
                        ptrs.kpe_pages.arg(),
                        ptrs.qo_indptr.arg(),
                        ptrs.kv_page_indices.arg(),
                        ptrs.kv_page_indptr.arg(),
                        ptrs.kv_last_page_lens.arg(),
                        ptrs.o.arg(),
                        ptrs.index_mask.arg(),
                        shape.index_mask_stride.arg(),
                        shape.num_requests.arg(),
                        shape.num_heads.arg(),
                        shape.page_size.arg(),
                        shape.sm_scale.arg(),
                        shape.causal.arg(),
                    ])?;
                }
                Ok(MlaNaive::LaunchedMma)
            }
            NaivePlan::Scalar { launch, head_group } => {
                // `attention_mla_naive.cuh:66-78` — the `__global__`'s
                // parameters, and note the tail: `R, H, CKV, KPE, page_size,
                // sm_scale, causal, G`. `G` is last and is the value the grid's
                // y axis was divided by.
                //
                // SAFETY: the caller's obligation on every pointer, above.
                unsafe {
                    ctx.launch("attn/attention_mla_naive.cuh", "::pie::attn::mla_naive::mla_naive_paged_kernel", launch, &[
                        ptrs.q_nope.arg(),
                        ptrs.q_pe.arg(),
                        ptrs.ckv_pages.arg(),
                        ptrs.kpe_pages.arg(),
                        ptrs.qo_indptr.arg(),
                        ptrs.kv_page_indices.arg(),
                        ptrs.kv_page_indptr.arg(),
                        ptrs.kv_last_page_lens.arg(),
                        ptrs.o.arg(),
                        ptrs.index_mask.arg(),
                        shape.index_mask_stride.arg(),
                        shape.num_requests.arg(),
                        shape.num_heads.arg(),
                        shape.kv_lora_rank.arg(),
                        shape.qk_rope_head_dim.arg(),
                        shape.page_size.arg(),
                        shape.sm_scale.arg(),
                        shape.causal.arg(),
                        head_group.arg(),
                    ])?;
                }
                Ok(MlaNaive::LaunchedScalar)
            }
        }
    }

    /// The 200 KiB opt-in, and why this module does not carry it.
    ///
    /// `attention_mla_naive.cuh:259-264` asked for **200 * 1024 = 204 800
    /// bytes** of dynamic shared memory for the scalar kernel, once per
    /// process, with this justification at `:255-258`:
    ///
    /// > Wide blocks are what make this kernel fast at decode: the grid is only
    /// > (tokens x head-groups), so with a narrow block the SMs sit at
    /// > single-digit occupancy and every key's load latency is exposed. The
    /// > partial-softmax scratch that buys the extra warps can exceed the 48 KB
    /// > default.
    ///
    /// **The first two sentences are the design and stand. The third is false
    /// against the file's own constants, and it is false by a factor of
    /// three.** [`naive_smem_bytes`] is `(8 * CKV + 16) * 4 = 32 * CKV + 64`,
    /// and the refusal at `:228` caps `CKV` at `32 * kMlaNaiveMaxPer = 512`. So
    /// the largest allocation this kernel can ever request is
    ///
    /// ```text
    /// 32 * 512 + 64 = 16 448 bytes
    /// ```
    ///
    /// — 16.1 KiB, against a 48 KiB default it would have to exceed by 3x
    /// before any opt-in were needed. To reach 49 152 the latent would have to
    /// be 1 535 wide, which `:228` rejects. **The `cudaFuncSetAttribute` was
    /// unreachable dead weight**, not a live requirement, and the Rust does not
    /// reproduce it: `raise_dynamic_smem_cap` is threshold-driven, sees 16 448,
    /// and correctly does nothing.
    ///
    /// It is recorded rather than dropped for two reasons. The 200 KiB is a
    /// MEASUREMENT — someone chose it — and the rule is that a measurement
    /// survives a port even when the port stops acting on it. And the comment's
    /// first two sentences explain why the block is 256 wide, which IS live and
    /// would have been lost with the number.
    ///
    /// The tensor-core arm is the opposite case and needs no note beyond
    /// [`MMA_SMEM_BYTES`]: 100 032 bytes genuinely exceeds 48 KiB, its C++
    /// opt-in asked for exactly `smem_bytes()` rather than a round number, and
    /// `raise_dynamic_smem_cap` raises it to exactly that.
    pub const NAIVE_OPT_IN_BYTES_UNREACHED: u32 = 200 * 1024;
}

/// The FlashInfer FA2 MLA host program — **compilable, and not yet fireable.**
pub mod mla_fa2 {
    use super::bf16;
    use super::mla_params::{MlaParams, UintFastdiv};
    use super::{Ctx, Refusal};
    use crate::jit::{Launch, Root};
    use crate::attn::plan::MlaPlanInfo;
    use crate::jit::Abi;

    /// `attn/attention_mla_fa2.cuh` — the root this arm's symbols come out of.
    ///
    /// Its two `#include`s are `attn/flashinfer/attention/mla{,_params}.cuh`,
    /// which the library header set does not answer, and `grid.sync()` needs
    /// two NVRTC options. **Both facts live in `jit::root`'s `CONFIGURED`**,
    /// keyed by this file's name, because they are properties of the C++ and
    /// not of this declaration — which is what lets a launch name the file and
    /// nothing else. They were written out here as literals until then, and
    /// the reason was the fixture: `every_instantiation_compiles` reconstructs
    /// a compile by reading the source, and a `.options(NAMED_CONST)` recovered
    /// as an EMPTY list would have compiled this root without
    /// `--device-as-default-execution-space` and reported upstream's
    /// unannotated helpers as errors of ours. The table answers that better
    /// than a literal did: there is now one statement of the options, and the
    /// fixture reads it rather than re-deriving it.
    ///
    /// The text holds no `__global__` of its own — it exists to instantiate
    /// FlashInfer's under a traits pack this file names — so what a compile of
    /// it produces is [`inst::MLA`]' six entries, [`SYMBOLS`]' six names for
    /// them and [`SMEM_ECHO`]'s three numbers, and nothing else.
    pub static ROOT: Root = Root::new("attn/attention_mla_fa2.cuh");

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// **One kernel, six points**: upstream's
    /// `::flashinfer::mla::BatchMLAPagedAttentionKernel<KTraits, Params>`
    /// (`mla.cuh:875`), instantiated once per [`ARMS`] entry per mask. The
    /// `KTraits` is the root's own `Traits` alias — eleven parameters with
    /// seven filled in, so the four that vary are written here and nothing
    /// else is — and `Params` is its `MLAParams<bf16, bf16, bf16, int32_t>`,
    /// which [`super::mla_params::MlaParams`] mirrors byte for byte.
    ///
    /// **`Traits`' parameter order is NOT [`Arm`]'s field order and the two
    /// must not be read against each other.** `KernelTraits` takes
    /// `CAUSAL_, NUM_STAGES_, QK_SHARD_, ..., CTA_TILE_KV_`, so the tile
    /// width is LAST here while [`Arm`] carries it second; the arm's
    /// `qk_shard` is the third argument and its `stages` the second.
    /// [`SMEM_ECHO`] spells the same four for three of these six and is the
    /// cross-check: its `Traits<true, 2u, true, 64u>` is `MLA[0][1]`'s.
    ///
    /// Indexed `[arm][causal]`, parallel to [`ARMS`] and to [`SYMBOLS`] —
    /// one index answers all three, which is what [`arm_index`] returns.
    /// `false` is the full mask and `true` the causal one, in that order,
    /// because `SYMBOLS` names them that way.
    ///
    /// The `u` suffixes are load-bearing in the same way `fa2`'s are: the
    /// template parameters are `std::uint32_t`, and a name expression is
    /// matched as WRITTEN before it is matched as parsed.
    pub mod inst {
        /// `mla.cuh:875` — the persistent MLA kernel, per arm and mask.
        pub const MLA: [[&str; 2]; 3] = [
            [
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<false, 2u, true, 64u>, \
                 ::pie::attn::mla_fa2::Params>",
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<true, 2u, true, 64u>, \
                 ::pie::attn::mla_fa2::Params>",
            ],
            [
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<false, 2u, true, 32u>, \
                 ::pie::attn::mla_fa2::Params>",
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<true, 2u, true, 32u>, \
                 ::pie::attn::mla_fa2::Params>",
            ],
            [
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<false, 1u, false, 16u>, \
                 ::pie::attn::mla_fa2::Params>",
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<true, 1u, false, 16u>, \
                 ::pie::attn::mla_fa2::Params>",
            ],
        ];
    }

    /// One `DISPATCH_SMEM_CONFIG` arm of `mla.cuh`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Arm {
        /// `KTraits::NUM_STAGES`.
        pub stages: u32,
        /// `KTraits::CTA_TILE_KV`.
        pub cta_tile_kv: u32,
        /// `KTraits::QK_SHARD`.
        pub qk_shard: bool,
        /// `sizeof(KTraits::SharedStorage)`, in bytes, and the smallest
        pub smem: u32,
    }

    /// The three arms, widest first, which is `DISPATCH_SMEM_CONFIG`'s order.
    pub const ARMS: [Arm; 3] = [
        Arm { stages: 2, cta_tile_kv: 64, qk_shard: true, smem: 221_696 },
        Arm { stages: 2, cta_tile_kv: 32, qk_shard: true, smem: 147_968 },
        Arm { stages: 1, cta_tile_kv: 16, qk_shard: false, smem: 92_672 },
    ];

    /// Which of [`ARMS`] this device's shared-memory budget admits, as the
    /// index that also selects in [`SYMBOLS`], [`SMEM_ECHO`] and
    /// [`inst::MLA`].
    ///
    /// The four arrays are parallel by construction and there is no second
    /// key: an arm is a position, not a `cta_tile_kv` to search for. That is
    /// what keeps a fire from having to re-derive which instantiation goes
    /// with a size it was handed.
    ///
    /// `None` when no arm fits, which is `DISPATCH_SMEM_CONFIG`'s own final
    /// `else` — upstream raises `cudaErrorNotSupported` there.
    #[must_use]
    pub const fn arm_index(smem_limit_per_sm: u32) -> Option<usize> {
        let mut i = 0;
        while i < ARMS.len() {
            if smem_limit_per_sm >= ARMS[i].smem {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// The widest arm this device's shared-memory budget admits.
    #[must_use]
    pub const fn arm_for(smem_limit_per_sm: u32) -> Option<Arm> {
        match arm_index(smem_limit_per_sm) {
            Some(i) => Some(ARMS[i]),
            None => None,
        }
    }

    /// The NVRTC options this root needs, as [`ROOT`] carries them.
    ///
    /// A read of the root rather than a second statement of the same fact —
    /// see [`ROOT`] for why the literals live in the declaration.
    #[must_use]
    pub const fn options() -> &'static [&'static str] {
        ROOT.options
    }

    /// The six row symbols, indexed by `[arm][causal]`, parallel to [`ARMS`].
    pub const SYMBOLS: [[&str; 2]; 3] = [
        ["attn::mla_fa2_kv64_full", "attn::mla_fa2_kv64_causal"],
        ["attn::mla_fa2_kv32_full", "attn::mla_fa2_kv32_causal"],
        ["attn::mla_fa2_kv16_full", "attn::mla_fa2_kv16_causal"],
    ];

    /// The compiler's own `sizeof(KTraits::SharedStorage)` per arm, as name
    pub const SMEM_ECHO: [&str; 3] = [
        "&::pie::attn::mla_fa2::smem_bytes_mla<::pie::attn::mla_fa2::Traits<true, 2u, true, 64u>>",
        "&::pie::attn::mla_fa2::smem_bytes_mla<::pie::attn::mla_fa2::Traits<true, 2u, true, 32u>>",
        "&::pie::attn::mla_fa2::smem_bytes_mla<::pie::attn::mla_fa2::Traits<true, 1u, false, 16u>>",
    ];

    /// The MLA cache's shape, as `dispatch_mla_512_64` reads it off
    #[derive(Clone, Copy, Debug)]
    pub struct Shape {
        /// Tokens per page.
        pub page_size: u32,
        /// Query heads.
        pub num_heads: u32,
        /// The compressed KV width — `HEAD_DIM_CKV`, 512.
        pub kv_lora_rank: u32,
        /// The positional width — `HEAD_DIM_KPE`, 64.
        pub qk_rope_head_dim: u32,
        /// `1 / sqrt(head_dim)`, or whatever the deployment states.
        pub sm_scale: f32,
    }

    /// Where the two workspace arenas start, and the addresses the fire
    #[derive(Clone, Copy, Debug)]
    pub struct Buffers {
        /// `AttentionWorkspaceView::int_buffer`.
        pub int_buffer: *mut u8,
        /// `AttentionWorkspaceView::float_buffer`.
        pub float_buffer: *mut u8,
        /// `[tokens, num_heads, kv_lora_rank]`.
        pub q_nope: *mut bf16,
        /// `[tokens, num_heads, qk_rope_head_dim]`.
        pub q_pe: *mut bf16,
        /// The layer's compressed KV pages.
        pub ckv_pages: *mut bf16,
        /// The layer's positional pages.
        pub kpe_pages: *mut bf16,
        /// The result, in the latent space.
        pub out: *mut bf16,
        /// The uploaded page-index array. NOT workspace-relative — it is a
        pub kv_page_indices: *mut i32,
        /// The LSE, or null when the statement does not ask for one.
        pub lse: *mut f32,
    }

    /// Fill an [`MlaParams`] the way `attention_mla.cu:264-320` does.
    ///
    /// # Safety
    ///
    /// Every pointer in `buffers` must be a device address valid for the
    /// fire, and `plan` must be the plan those arenas were uploaded from.
    /// Nothing is dereferenced here; the requirement is the kernel's.
    #[must_use]
    pub unsafe fn pack(
        plan: &MlaPlanInfo,
        shape: Shape,
        buffers: Buffers,
        want_lse: bool,
    ) -> MlaParams {
        /// `int_buf + offset`, in ELEMENTS of `T`.
        unsafe fn offset_ptr<T>(base: *mut u8, offset: i64) -> *mut T {
        unsafe { base.cast::<T>().offset(offset as isize) }
        }

        let int_buf = buffers.int_buffer;
        let float_buf = buffers.float_buffer;
        MlaParams {
            q_nope: buffers.q_nope,
            q_pe: buffers.q_pe,
            ckv: buffers.ckv_pages,
            kpe: buffers.kpe_pages,
            partial_o: unsafe { offset_ptr(float_buf, plan.partial_o_offset) },
            partial_lse: unsafe { offset_ptr(float_buf, plan.partial_lse_offset) },
            final_o: buffers.out,
            final_lse: if want_lse { buffers.lse } else { ::core::ptr::null_mut() },
            q_indptr: unsafe { offset_ptr(int_buf, plan.q_indptr_offset) },
            kv_indptr: unsafe { offset_ptr(int_buf, plan.kv_indptr_offset) },
            partial_indptr: unsafe { offset_ptr(int_buf, plan.partial_indptr_offset) },
            merge_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_start_offset)
            },
            merge_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_end_offset)
            },
            merge_partial_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_start_offset)
            },
            merge_partial_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_end_offset)
            },
            merge_partial_stride: unsafe { offset_ptr(int_buf, plan.merge_partial_stride_offset) },
            kv_indices: buffers.kv_page_indices,
            q_len: unsafe { offset_ptr(int_buf, plan.q_len_offset) },
            kv_len: unsafe { offset_ptr(int_buf, plan.kv_len_offset) },
            q_start: unsafe { offset_ptr(int_buf, plan.q_start_offset) },
            kv_start: unsafe { offset_ptr(int_buf, plan.kv_start_offset) },
            kv_end: unsafe { offset_ptr(int_buf, plan.kv_end_offset) },
            work_indptr: unsafe { offset_ptr(int_buf, plan.work_indptr_offset) },
            block_size: UintFastdiv::new(shape.page_size),
            num_heads: UintFastdiv::new(shape.num_heads),
            q_nope_stride_n: shape.num_heads * shape.kv_lora_rank,
            q_nope_stride_h: shape.kv_lora_rank,
            q_pe_stride_n: shape.num_heads * shape.qk_rope_head_dim,
            q_pe_stride_h: shape.qk_rope_head_dim,
            ckv_stride_page: shape.page_size * shape.kv_lora_rank,
            ckv_stride_n: shape.kv_lora_rank,
            kpe_stride_page: shape.page_size * shape.qk_rope_head_dim,
            kpe_stride_n: shape.qk_rope_head_dim,
            o_stride_n: shape.num_heads * shape.kv_lora_rank,
            o_stride_h: shape.kv_lora_rank,
            sm_scale: shape.sm_scale,
            ckv_scale: 1.0,
            kpe_scale: 1.0,
            return_lse_base_on_e: true,
        }
    }

    /// The grid, from the plan and nothing else.
    #[must_use]
    pub const fn grid(plan: &MlaPlanInfo, arm: Arm) -> Launch {
        Launch::grid([plan.num_blks_x as u32, plan.num_blks_y as u32, 1], [256, 1, 1])
            .smem(arm.smem)
            .cooperative()
    }

    /// Fire the arm `arm` names, at the mask `causal` names.
    ///
    /// The whole of what `attention_mla.cu`'s FA2 half did after
    /// `DISPATCH_SMEM_CONFIG` picked an arm: ONE kernel, ONE argument, and
    /// that argument is [`MlaParams`] by value — `__grid_constant__` on the
    /// device side, [`crate::jit::ArgValue::Bytes`] on this one, which is
    /// what `by_value!`'s untagged arm exists for.
    ///
    /// **`launch` is [`grid`]'s and must not be rebuilt from a rectangle.**
    /// Three residency facts ride on it and none is checkable at the launch
    /// path (`attention_mla_fa2.cuh`'s header states all three): the grid is
    /// `num_sm` blocks because `plan::mla`'s `Schedule` was built that way,
    /// the shared allocation is the arm's own `sizeof(SharedStorage)`, and
    /// the launch is COOPERATIVE because `mla.cuh:1061` calls `grid.sync()`
    /// between the two stages. A non-cooperative launch of this kernel is a
    /// deadlock rather than an error.
    ///
    /// `arm` indexes [`ARMS`]; [`arm_index`] is what produces it. Out of
    /// range is [`Refusal::Absent`] rather than a panic, because the index
    /// comes from a device query and a device that answers nothing should
    /// decline in a sentence.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] for an `arm` past [`ARMS`]; otherwise whatever the
    /// compile, the load or the launch refuses.
    pub fn fire(
        ctx: &Ctx,
        arm: usize,
        causal: bool,
        params: &MlaParams,
        launch: Launch,
    ) -> Result<(), Refusal> {
        let Some(row) = inst::MLA.get(arm) else {
            return Err(Refusal::Absent { what: "a `DISPATCH_SMEM_CONFIG` arm for this device" });
        };
        // SAFETY: every pointer in `params` is a device address the caller
        // keeps live across the launch -- the obligation every `<<<>>>` made,
        // and `pack`'s own.
        unsafe {
            ctx.launch(
                "attn/attention_mla_fa2.cuh",
                row[usize::from(causal)],
                launch,
                &[params.arg()],
            )
        }
    }
}

/// Which of MLA's two arms ran, and what it did.
///
/// `#[must_use]` for [`mla_naive::MlaNaive`]'s reason, one level up: *"it
/// declined"* must not be spellable like *"it ran"*. On a decline the output
/// buffer is untouched, and the `latent_to_v`/`o_proj` that follow an MLA
/// attention read it either way.
#[must_use]
pub enum MlaDispatch {
    /// FlashInfer's FA2 kernel was launched on the caller's stream, at this
    /// index into [`mla_fa2::ARMS`].
    ///
    /// The arm is REPORTED rather than discarded because it is the whole of
    /// what `DISPATCH_SMEM_CONFIG` decided, and it is the index that also
    /// selects in [`mla_fa2::SYMBOLS`] — so a caller logging or asserting on
    /// which configuration ran does not have to re-run the device query to
    /// find out.
    Fa2 {
        /// The widest arm this device's shared-memory budget admitted.
        arm: usize,
    },
    /// The Blackwell pair was chosen, and this is [`mla_naive::fire`]'s own
    /// answer — INCLUDING [`mla_naive::MlaNaive::Declined`], carried through
    /// rather than translated. See [`dispatch_attention_mla_bf16`]'s doc for
    /// the argument.
    Naive(mla_naive::MlaNaive),
}

/// `attn::dispatch_attention_mla_bf16` — the entry point, which chooses.
///
/// ONE trace symbol with TWO bodies. [`mla_fa2::fire`] is FlashInfer's
/// persistent FA2 kernel and [`mla_naive::fire`] the Blackwell pair, and the
/// choice between them is this device's compute-capability major: **`>= 10`
/// picks naive.** `attention_mla.cu:334-340`'s note is the only account
/// anywhere of why, so it is carried whole rather than summarised:
///
/// > FlashInfer's FA2 BatchMLAPagedAttention (a cooperative kernel) produces
/// > zero output on sm_100; the ecosystem (sglang/vllm) routes Blackwell MLA
/// > to trtllm/cutlass/ragged kernels instead. This is a correctness-first,
/// > arch-agnostic latent-space MLA: one block per (token, head), flash-style
/// > online softmax over the paged ckv/kpe cache. Output is in the kv_lora
/// > latent space (same as the FA2 path), so the rest of the MLA forward
/// > (latent_to_v, o_proj) is unchanged.
///
/// **A zero output is a WRONG ANSWER and not a fault.** Nothing raises, the
/// `o_proj` that follows reads zeros, and the model emits text. That is what
/// makes this a dispatch rather than a fallback, and it is the premise every
/// other decision below rests on.
///
/// This is [`kv_paged::write_kv_to_pages`]'s shape, further down this same
/// file: an entry point whose whole body is a choice between two complete
/// arms, with the arms' own docs carrying their kernels' arithmetic. That one
/// picks on a field of the layer; this one picks on the device.
///
/// # Two device queries, and only one of them may fall back
///
/// The capability is [`Ctx::compute_capability_major`], and a `None` REFUSES.
/// The C++ wrote `int major = 0; cudaDeviceGetAttribute(&major, ...)` and
/// tested `major >= 10`, so a device that would not answer silently got FA2 —
/// **that default is not carried.** It is incidental rather than chosen (it
/// is what the uninitialise-then-query shape leaves behind), and the thing it
/// defaults to is the one arm that returns zeros on the one architecture the
/// query exists to detect. [`xqa::attention_xqa_decode_bf16_prepared`] makes
/// the same refusal in the same words when its own capability query comes
/// back empty.
///
/// The FA2 arm needs a second number: `smem_limit_per_sm`, which is
/// `cudaDevAttrMaxSharedMemoryPerMultiprocessor`. That is exactly what
/// upstream's `BatchMLAPagedAttention` read before handing it to
/// `DISPATCH_SMEM_CONFIG` — that launcher is the `// PIE: REMOVED` marker
/// below the macro in `attn/flashinfer/attention/mla.cuh` (this crate plans
/// and fires with `cuLaunchKernel`), and the macro's three thresholds survive
/// beside it and as [`mla_fa2::ARMS`], so reading the same attribute here is
/// what keeps the port's arm equal to upstream's. It comes through
/// [`fa2::plan::fa_device`], the one place in this crate that asks: memoised,
/// so two fires in one process cannot disagree, and with `Device::L40S` as a
/// NAMED fallback rather than a guess.
///
/// **That fallback is admissible where the capability's would not be, and the
/// difference is the failure mode.** A wrong shared-memory budget picks an arm
/// whose `SharedStorage` the device cannot allocate, and the launch FAILS —
/// loudly, with the kernel named. A wrong capability picks an arm that
/// launches, succeeds, and writes zeros.
///
/// # The two C++ checks that are gone, and neither is a relaxation
///
/// `dispatch_attention_mla_bf16` opened with two `throw`s and this has
/// neither, because both were artefacts of the cache struct rather than facts
/// about the fire:
///
/// * `if (!cache.valid) throw "cache is empty; call plan first"`.
///   `MlaPlanCache` was a mutable object a separate `plan` call filled, so it
///   had a *"nobody called plan"* state to be in. [`MlaPlan`] is a value; a
///   `&MlaPlan` is a plan that was built, and there is no state to test.
/// * `if (layer.kv_lora_rank != cache.kv_lora_rank || ...) throw "layer/cache
///   shape mismatch"`. `MlaPlanCache` carried its OWN copy of the latent
///   rank, the rope width and the page size, and the check was that the two
///   copies agreed. There is one copy here: both arms' shapes are built from
///   `layer` below and from nothing else, so the disagreement is not
///   detectable because it is not constructible.
///
/// What that second check was really guarding — that `plan` was scheduled for
/// THIS layer's geometry — neither copy could ever have shown, since
/// [`plan::MlaPlanInfo`] carries offsets and a grid and no shape at all. It is
/// an obligation, and it is in `# Safety` where obligations go.
///
/// # `MlaNaive::Declined` is not an error and is not translated
///
/// It is returned as [`MlaDispatch::Naive`] holding exactly what
/// [`mla_naive::fire`] returned. Three reasons, and the first is the whole
/// argument:
///
/// * **The arm has already classified it.** Three of the four declines were a
///   `throw` in the C++ and one a bare `return`, and that arm's doc is explicit
///   that neither is a device failure. Re-classifying here would make one fact
///   answer differently depending on which door the caller came in.
/// * `Refusal` is a `Copy` value with no payload, so translating would flatten
///   four distinguishable answers — an empty batch, a null indptr trio, a
///   latent too wide for the register array, a rope tail too wide — into one
///   string or into four strings nothing could match on.
/// * The return type has to be able to say *"nothing was launched"* anyway,
///   because [`mla_naive::NaiveDecline::NoTokens`] is a legal empty fire.
///   Erasing that into `Ok(())` is how a caller comes to believe `o` was
///   written when it was not.
///
/// The contrast is what the two refusals below are for: a device that will not
/// state its capability and a device no [`mla_fa2::ARMS`] entry fits are the
/// DEVICE failing to answer, not the SHAPE going unserved, and those are
/// `Refusal`s. Keeping the two apart is the reason this does not return
/// `Result<(), Refusal>`.
///
/// # It is NOT `whole`, and the contrast is the argument
///
/// [`mla_prepare_bf16`] and [`write_mla_to_pages`] are `whole` because they
/// WALK `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`, which are
/// R-shaped, so a row window leaves that arithmetic pointing at another
/// request's rows. This reads a plan built over the whole fire and still
/// covers a row range, like the FlashInfer dispatches do — so a row window is
/// legal here and the column stays `false`. It was `false` at `9e3936fb9^`
/// and `tests/stated_columns.rs` pins it there.
///
/// # Two things the arms do not agree on, recorded rather than unified
///
/// Both are upstream's, both are load-bearing for a caller, and neither is
/// repaired here — making the two arms agree would be inventing a behaviour
/// rather than porting one, and nothing on this box could tell which
/// invention was right.
///
/// * **An empty fire.** `total_tokens <= 0` is
///   [`mla_naive::NaiveDecline::NoTokens`] on the naive arm —
///   `attention_mla_naive.cuh:211`'s bare `return` — and the FA2 arm has no
///   such test: its grid is the plan's `num_blks_x`/`num_blks_y`, so an empty
///   batch is the scheduler's business rather than the launcher's.
/// * **`lse`.** Only the FA2 arm writes one. The naive pair has no LSE output
///   at all — neither `__global__` takes the pointer — so on a `>= 10` device
///   a caller that passed a non-null `lse` gets an untouched buffer and no
///   word about it. The C++ dropped `lse_out` on that path in exactly the
///   same silence. It is not turned into a refusal here because a refusal
///   would make a Blackwell box reject a fire an sm_90 box serves, which is a
///   larger claim than *"this arm has no LSE"* and is not one this `fn` is in
///   a position to make.
///
/// # What is verified, and what is not
///
/// **Nothing calls this.** What is verified is that it compiles, and that
/// every template-id both arms name is one NVRTC lowers —
/// `tests/every_instantiation_compiles`, which reads both roots and both
/// `mod inst` blocks out of this source. That is the whole of the evidence,
/// and note what it is not: that fixture needs `libnvrtc` and not a device,
/// so it proves the strings compile and nothing about what they compute.
/// **No fire reaches this function, so no numerical result backs a single
/// line of it**, including the arm choice it exists to make.
///
/// Three things block MLA end to end and this is one of them:
///
/// * there was no dispatcher — the two arms sat beside each other with
///   nothing choosing, which is what this `fn` is;
/// * `driver-cuda`'s `fire/launch.rs`'s `kv_pools_for` refuses
///   `KvStyle::Mla`, so no latent cache is ever allocated for a fire;
/// * `serve/load.rs` refuses an MLA checkpoint at model load, so no fire with
///   one is ever built.
///
/// The other two must move for this to run, and neither moves because this
/// exists. `bind/arms/attn.rs` declines the symbol over `Cx::mla_layer` and
/// `Cx::mla_plan` for exactly that reason; its sentence there — *"what is
/// missing is a caller"* — is still true with this written.
///
/// # Errors
///
/// [`Refusal::Device`] if the device will not state its compute capability,
/// and [`Refusal::Absent`] if no [`mla_fa2::ARMS`] entry fits its shared
/// memory — which is `DISPATCH_SMEM_CONFIG`'s own final `else`, where
/// upstream raised `cudaErrorNotSupported`. Otherwise whatever the chosen
/// arm's compile, load or launch refuses. **A shape the naive pair cannot
/// serve is not among these**; see above.
///
/// # Safety
///
/// Every pointer must be a device address live across the launch — `layer`'s
/// two page pointers and `plan`'s two arenas included — which is the
/// obligation every `<<<>>>` made. Beyond it, [`mla_fa2::pack`]'s own, which
/// this inherits and does not widen: `plan` must be the plan those arenas
/// were uploaded from, because the FA2 arm reaches fourteen index arrays in
/// the int arena and two partial buffers in the float one by adding that
/// plan's offsets to those two base pointers. `plan` must also
/// have been scheduled for `layer`'s geometry and for `num_heads`; the C++
/// checked a copy of that and this has no copy to check.
pub unsafe fn dispatch_attention_mla_bf16(
    ctx: &Ctx,
    plan: &MlaPlan,
    q_nope: *const bf16,
    q_pe: *const bf16,
    layer: MlaLayer,
    o: *mut bf16,
    kv_page_indices: *const u32,
    lse: *mut f32,
    qo_indptr: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    index_mask: *const u8,
    index_mask_stride: i32,
    total_tokens: i32,
    num_requests: i32,
    num_heads: i32,
    sm_scale: f32,
    causal: bool,
) -> Result<MlaDispatch, Refusal> {
    let Some(major) = ctx.compute_capability_major() else {
        return Err(Refusal::Device {
            why: "the device would not say its compute capability, which is the whole \
                  of this dispatch: FA2 MLA writes zeros on sm_100",
        });
    };

    if major >= 10 {
        // `attention_mla.cu:359-364` — `launch_mla_naive_paged`, which took
        // the layer view apart into the five values the raw launcher wanted.
        // The arm reads NO workspace and NO plan: it walks the paged cache
        // directly, which is what makes it arch-agnostic.
        let ptrs = mla_naive::NaivePtrs {
            q_nope,
            q_pe,
            ckv_pages: layer.ckv_pages.cast::<bf16>().cast_const(),
            kpe_pages: layer.kpe_pages.cast::<bf16>().cast_const(),
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            o,
            index_mask,
        };
        let shape = mla_naive::NaiveShape {
            kv_lora_rank: layer.kv_lora_rank,
            qk_rope_head_dim: layer.qk_rope_head_dim,
            page_size: layer.page_size,
            total_tokens,
            num_requests,
            num_heads,
            sm_scale,
            causal,
            index_mask_stride,
        };
        return mla_naive::fire(ctx, ptrs, shape).map(MlaDispatch::Naive);
    }

    // `cudaDevAttrMaxSharedMemoryPerMultiprocessor`, which is what the
    // deleted upstream launcher read, through the memo that already holds it.
    let Some(arm) = mla_fa2::arm_index(fa2::plan::fa_device().max_smem_per_sm) else {
        return Err(Refusal::Absent {
            what: "a `DISPATCH_SMEM_CONFIG` arm for this device's shared memory per SM",
        });
    };
    let shape = mla_fa2::Shape {
        page_size: layer.page_size.unsigned_abs(),
        num_heads: num_heads.unsigned_abs(),
        kv_lora_rank: layer.kv_lora_rank.unsigned_abs(),
        qk_rope_head_dim: layer.qk_rope_head_dim.unsigned_abs(),
        sm_scale,
    };
    let buffers = mla_fa2::Buffers {
        int_buffer: plan.int_arena.cast::<u8>(),
        float_buffer: plan.float_arena.cast::<u8>(),
        // `attention_mla.cu:270-271` and `:292-293` — three `const_cast`s and
        // a `reinterpret_cast`, transcribed rather than argued away.
        // `MLAParams` declares the two query halves and the page-index array
        // non-const; the C++ took all three as `const` from its caller and
        // cast, which is defined only if nothing writes through the result.
        // So the struct's spelling is the wider one and this signature's is
        // the one that says what the kernel does.
        q_nope: q_nope.cast_mut(),
        q_pe: q_pe.cast_mut(),
        ckv_pages: layer.ckv_pages.cast::<bf16>(),
        kpe_pages: layer.kpe_pages.cast::<bf16>(),
        out: o,
        kv_page_indices: kv_page_indices.cast::<i32>().cast_mut(),
        lse,
    };
    // `want_lse` is DERIVED and is not a parameter: the C++ assigned
    // `params.final_lse = lse_out` unconditionally, so the null pointer WAS
    // the request, and a caller given both a pointer and a flag has two ways
    // to say one thing and one way to disagree with itself. `mla_naive::fire`
    // recovers `have_indptr` from its three pointers for the same reason.
    //
    // SAFETY: the caller's obligation, forwarded -- every pointer here is a
    // device address live across the launch, and `plan` is the plan its two
    // arenas were uploaded from. Nothing is dereferenced on this side; the
    // offsets `pack` adds are the ones the upload used.
    let params = unsafe { mla_fa2::pack(&plan.info, shape, buffers, !lse.is_null()) };
    // `attention_mla.cu:406-414` — the mask is a template parameter upstream
    // and an index here, which is why one call replaces the C++'s two.
    mla_fa2::fire(ctx, arm, causal, &params, mla_fa2::grid(&plan.info, mla_fa2::ARMS[arm]))?;
    Ok(MlaDispatch::Fa2 { arm })
}

/// The units `attn` compiles in fn-world.
pub mod qkv_fused {
    use super::bf16;
    use super::{Ctx, Launch, Refusal};
    
    use crate::jit::Abi;

    /// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` — the fused
    ///
    /// What the caller must guarantee, as `call()` states it: every pointer
    /// must be a device address valid for the fire, `packed` must hold
    /// `num_rows` rows of `(num_q_heads + 2·num_kv_heads)·head_dim` elements,
    /// and the page arrays must describe the layer the cache pointers came
    /// from.
    pub fn qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
        ctx: &Ctx,
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        row_valid: *const u8,
        num_rows: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
    ) -> Result<(), Refusal> {
        /// `attn/qkv_fused.cuh` — the root these routines compile a symbol out of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
        ///
        /// The decode forms carry TWO template arguments: the block width, and
        /// whether the kernel reads a precomputed cos/sin table. The warp form
        /// exists at three widths and nowhere else, which is what
        /// [`warp_instantiation`] answers `None` for.
        /// `BLOCK` for the packed form, and it IS the block width.
        pub const PACKED_BLOCK: u32 = 256;

        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/qkv_fused.cuh",
                "::pie::attn::qkv_packed_qk_norm_rope_vnorm_write_kv<::pie::i32(256)>",
                Launch::grid([num_rows.unsigned_abs(), heads, 1], [PACKED_BLOCK, 1, 1]),
                &[
                    packed.arg(),
                    q_out.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    q_weight.arg(),
                    k_weight.arg(),
                    positions.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    row_valid.arg(),
                    num_q_heads.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                    page_size.arg(),
                    hnd_layout.arg(),
                    theta.arg(),
                    eps.arg(),
                ],
            )
        }
    }

    /// The six warp instantiations, by `(head_dim, rope_table)`.
    ///
    /// `None` is the whole of the warp form's applicability test: it exists
    /// at three head widths and the block form covers every other.
    fn warp_instantiation(head_dim: i32, rope_table: bool) -> Option<&'static str> {
        Some(match (head_dim, rope_table) {
            (64, true) => "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(64), true>",
            (64, false) => "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(64), false>",
            (128, true) => "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(128), true>",
            (128, false) => "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(128), false>",
            (256, true) => "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(256), true>",
            (256, false) => "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(256), false>",
            _ => return None,
        })
    }

    /// `attn/qkv_fused.cu:31` — `qkv_decode_fused_dispatch`, the `static` one.
    ///
    /// What the caller must guarantee, as `call()` states it: every pointer
    /// is a device address live across the launch; the five named above may
    /// be null.
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn qkv_decode_fused_dispatch(
        ctx: &Ctx,
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        rope_table: *const f32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        w_page: *const u32,
        w_off: *const u32,
        row_valid: *const u8,
        win: *const u32,
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
    ) -> Result<(), Refusal> {
        /// `qkv_fused.cu:51` — `constexpr int WARP_BLOCK = 256;`, and it is NOT
        pub const WARP_BLOCK: u32 = 256;

        /// The block form's two arms.
        const fn block_instantiation(rope_table: bool) -> &'static str {
        if rope_table { "::pie::attn::qkv_decode_qk_norm_rope_write_kv<::pie::i32(128), true>" } else { "::pie::attn::qkv_decode_qk_norm_rope_write_kv<::pie::i32(128), false>" }
        }

        /// Warps per block: `WARP_BLOCK / 32`.
        const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

        /// `qkv_fused.cu:105` — `constexpr int BLOCK = 128;`, the DECODE block.
        pub const DECODE_BLOCK: u32 = 128;

        if q_out.is_null() {
            return Err(Refusal::Absent { what: "q_out" });
        }

        let use_rope_table = !rope_table.is_null();
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();

        // The warp form takes `num_requests` where the block form takes
        // `head_dim`: one warp walks a whole request's heads, so the kernel
        // needs the request count and reads the width from its template.
        if let Some(instantiation) = warp_instantiation(head_dim, use_rope_table) {
            let units = num_requests.unsigned_abs().saturating_mul(heads);
            // SAFETY: `call()`'s contract -- every pointer bound here
            // addresses live device memory of the extent the kernel reads it
            // as.
            return unsafe {
                ctx.launch(
                    "attn/qkv_fused.cuh",
                    instantiation,
                    Launch::grid([units.div_ceil(WARPS_PER_BLOCK), 1, 1], [WARP_BLOCK, 1, 1]),
                    &[
                        packed.arg(),
                        q_out.arg(),
                        k_pages.arg(),
                        v_pages.arg(),
                        q_weight.arg(),
                        k_weight.arg(),
                        positions.arg(),
                        rope_table.arg(),
                        kv_page_indices.arg(),
                        kv_page_indptr.arg(),
                        kv_last_page_lens.arg(),
                        w_page.arg(),
                        w_off.arg(),
                        row_valid.arg(),
                        win.arg(),
                        num_requests.arg(),
                        num_q_heads.arg(),
                        num_kv_heads.arg(),
                        page_size.arg(),
                        hnd_layout.arg(),
                        theta.arg(),
                        eps.arg(),
                    ],
                )
            };
        }

        // SAFETY: as above.
        unsafe {
            ctx.launch(
                "attn/qkv_fused.cuh",
                block_instantiation(use_rope_table),
                Launch::grid([num_requests.unsigned_abs(), heads, 1], [DECODE_BLOCK, 1, 1]),
                &[
                    packed.arg(),
                    q_out.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    q_weight.arg(),
                    k_weight.arg(),
                    positions.arg(),
                    rope_table.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    w_page.arg(),
                    w_off.arg(),
                    row_valid.arg(),
                    win.arg(),
                    num_q_heads.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                    page_size.arg(),
                    hnd_layout.arg(),
                    theta.arg(),
                    eps.arg(),
                ],
            )
        }
    }

    /// `attn/qkv_fused.cu:160` — `qkv_decode_qk_norm_rope_write_kv_bf16`.
    ///
    /// [`qkv_decode_fused_dispatch`]'s obligation.
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn qkv_decode_qk_norm_rope_write_kv_bf16(
        ctx: &Ctx,
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        rope_table: *const f32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        w_page: *const u32,
        w_off: *const u32,
        row_valid: *const u8,
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
    ) -> Result<(), Refusal> {
        // `win` is null here: this entry point is the host-window form, and
        // the kernel reads a null window as "no window".
        qkv_decode_fused_dispatch(
            ctx,
            packed,
            q_out,
            k_pages,
            v_pages,
            q_weight,
            k_weight,
            positions,
            rope_table,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            w_page,
            w_off,
            row_valid,
            core::ptr::null(),
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            hnd_layout,
            theta,
            eps,
        )
    }
}

/// `attn/dsv4_compress.cuh` — deepseek_v4's SECOND KV cache, and the eleven
pub mod dsv4_compress {
    use super::bf16;
    use super::{Ctx, Launch, Refusal};
    
    use crate::jit::Abi;

    /// `attn/dsv4_compress.cuh` — the root these routines compile a symbol
    /// out of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Four of the header's eleven kernels have no host program in this file
    /// — the two pools, the APE add and the unpaged gather — so no constant
    /// here names them.
        /// `route_rows`' warp rounding, and the clamp that makes it legal at any
    #[expect(clippy::cast_sign_loss, reason = "both are guarded positive by every caller")]
    fn route_rows(rows: i32, width: i32) -> Launch {
        let (rows, width) = (rows as u32, width as u32);
        Launch::per_row(rows, width.div_ceil(32).max(1).saturating_mul(32).min(1024))
    }

    /// Build one compressed entry per boundary token.
    ///
    /// What the caller must guarantee, as `call()` states it: every pointer
    /// addresses a live allocation of the extent the kernel reads, and `ape`
    /// and nothing else may be null.
    pub fn dsv4_compress_gather_paged_bf16(
        ctx: &Ctx,
        state_kv: *const bf16,
        state_score: *const bf16,
        ape: *const f32,
        boundary_pos: *const i32,
        boundary_req: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        out: *mut bf16,
        num_entries: i32,
        head_dim: i32,
        ratio: i32,
        coff: i32,
        page_size: i32,
    ) -> Result<(), Refusal> {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/dsv4_compress.cuh",
                "::pie::attn::dsv4_compress_gather_paged<::pie::bf16>",
                route_rows(num_entries, head_dim),
                &[
                    state_kv.arg(),
                    state_score.arg(),
                    ape.arg(),
                    boundary_pos.arg(),
                    boundary_req.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    out.arg(),
                    head_dim.arg(),
                    ratio.arg(),
                    coff.arg(),
                    page_size.arg(),
                ],
            )
        }
    }

    /// Commit those entries to the compressed cache.
    ///
    /// As above; no operand of this one is nullable.
    pub fn dsv4_store_comp_entries_bf16(
        ctx: &Ctx,
        entries: *const bf16,
        comp_kv_pages: *mut bf16,
        boundary_pos: *const i32,
        boundary_req: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        num_entries: i32,
        head_dim: i32,
        page_size: i32,
    ) -> Result<(), Refusal> {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/dsv4_compress.cuh",
                "::pie::attn::dsv4_store_comp_entries<::pie::bf16>",
                route_rows(num_entries, head_dim),
                &[
                    entries.arg(),
                    comp_kv_pages.arg(),
                    boundary_pos.arg(),
                    boundary_req.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    head_dim.arg(),
                    page_size.arg(),
                ],
            )
        }
    }
}

/// `attn/kv_paged.cuh` — the paged KV cache's appenders, its quantised
pub mod kv_paged {
    use super::bf16;
    use crate::jit::abi::MaybeConst;
    use crate::jit::fp8_kind;

    use super::{Ctx, Launch};
    
    use crate::jit::Abi;
    use crate::attn::{KvDType, KvLayer, KvScheme};
    use kernels::Refusal;

    /// `attn/kv_paged.cuh` — the root these routines compile a symbol out of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The `#hnd`/`#nhd` pairs are one template each over a page layout:
    /// `true_type` is `[head, page, dim]` and `false_type` is `[page, head,
    /// dim]`. The position-addressed append has no host program in this file,
    /// so no constant here names it.
        /// `kv_paged.cu`'s `constexpr int BLOCK = 256`, which every launch in
    const BLOCK: u32 = 256;
    /// The interpretation an fp8 page is written and read under.
    fn fp8_kind_of(storage_dtype: KvDType) -> fp8_kind {

        const NV_E5M2: u32 = 1;

        /// `::__nv_fp8_interpretation_t`'s two values, by the names the header
        const NV_E4M3: u32 = 0;

        fp8_kind(if matches!(storage_dtype, KvDType::Fp8E5M2) { NV_E5M2 } else { NV_E4M3 })
    }

    /// NVFP4's block, when the layer states none.
    fn fp4_block_size(layer: &KvLayer) -> i32 {
        if layer.block_size > 0 { layer.block_size } else { 16 }
    }

    /// An upper bound on the pages an append can touch.
    #[must_use]
    pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
        if page_size <= 0 {
            return 0;
        }
        (total_tokens + page_size - 1) / page_size + num_requests
    }

    /// `attn::write_kv_explicit_bf16` — write B rows to B explicit slots.
    ///
    /// What the caller must guarantee, as `call()` states it: every pointer
    /// must be a device allocation of the stated extent.
    pub fn write_kv_explicit_bf16(
        ctx: &Ctx,
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        b: i32,
        row_valid: *const u8,
    ) -> Result<(), Refusal> {
        assert!(layer.is_native_bf16, "attn::write_kv_explicit_bf16 requires native bf16 KV cache");

        let instantiation =
            if layer.hnd { "::pie::attn::write_kv_explicit<\
                                ::pie::true_type::value>" } else { "::pie::attn::write_kv_explicit<::pie::false_type::value>" };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/kv_paged.cuh",
                instantiation,
                Launch::per_row(b.unsigned_abs(), BLOCK),
                &[
                    k_curr.arg(),
                    v_curr.arg(),
                    layer.k_pages.cast::<bf16>().arg(),
                    layer.v_pages.cast::<bf16>().arg(),
                    w_page.arg(),
                    w_off.arg(),
                    MaybeConst::new(row_valid).arg(),
                    b.arg(),
                    layer.page_size.arg(),
                    layer.num_kv_heads.arg(),
                    layer.head_dim.arg(),
                ],
            )?;
        }

        if layer.has_envelopes && !layer.hnd {
            let _ = crate::layout::envelope_merge_written(
                ctx,
                k_curr,
                w_page,
                w_off,
                MaybeConst::new(row_valid),
                layer.k_env_min.cast(),
                layer.k_env_max.cast(),
                b,
                layer.num_kv_heads,
                layer.head_dim,
            );
        }
        Ok(())
    }

    /// `attn::write_kv_explicit_bf16_devwin` — the same write with a
    ///
    /// As [`write_kv_explicit_bf16`].
    pub fn write_kv_explicit_bf16_devwin(
        ctx: &Ctx,
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        win_d: *const u32,
        n_max: i32,
        row_valid: *const u8,
    ) -> Result<(), Refusal> {
        assert!(
            layer.is_native_bf16,
            "attn::write_kv_explicit_bf16_devwin requires native bf16 KV cache"
        );
        assert!(
            !layer.has_envelopes,
            "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
             windowed — use the host-window form"
        );

        let instantiation = if layer.hnd {
            "::pie::attn::write_kv_explicit_devwin<::pie::true_type::value>"
        } else {
            "::pie::attn::write_kv_explicit_devwin<::pie::false_type::value>"
        };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/kv_paged.cuh",
                instantiation,
                Launch::per_row(n_max.unsigned_abs(), BLOCK),
                &[
                    k_curr.arg(),
                    v_curr.arg(),
                    layer.k_pages.cast::<bf16>().arg(),
                    layer.v_pages.cast::<bf16>().arg(),
                    w_page.arg(),
                    w_off.arg(),
                    MaybeConst::new(row_valid).arg(),
                    win_d.arg(),
                    n_max.arg(),
                    layer.page_size.arg(),
                    layer.num_kv_heads.arg(),
                    layer.head_dim.arg(),
                ],
            )
        }
    }

    /// The native-bf16 append, `kv_paged.cu:60-120`.
    ///
    /// As [`write_kv_explicit_bf16`]; the four CSR arrays must describe
    /// `num_requests` requests over `total_tokens` tokens.
    pub fn write_kv_to_pages_bf16(
        ctx: &Ctx,
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        row_valid: *const u8,
        first_token: i32,
    ) -> Result<(), Refusal> {
        let launch_tokens = total_tokens - first_token;

        let instantiation = if layer.hnd { "::pie::attn::write_kv<\
                                                ::pie::true_type::value>" } else { "::pie::attn::write_kv<::pie::false_type::value>" };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/kv_paged.cuh",
                instantiation,
                Launch::per_row(launch_tokens.unsigned_abs(), BLOCK),
                &[
                    k_curr.arg(),
                    v_curr.arg(),
                    layer.k_pages.cast::<bf16>().arg(),
                    layer.v_pages.cast::<bf16>().arg(),
                    qo_indptr.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    MaybeConst::new(row_valid).arg(),
                    MaybeConst::<u32>::none().arg(),
                    num_requests.arg(),
                    layer.page_size.arg(),
                    layer.num_kv_heads.arg(),
                    layer.head_dim.arg(),
                    first_token.arg(),
                ],
            )?;
        }

        if layer.has_envelopes && !layer.hnd && total_tokens > 0 {
            let _ = crate::layout::envelope_update_appended(
                ctx,
                layer.k_pages.cast(),
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                layer.k_env_min.cast(),
                layer.k_env_max.cast(),
                num_requests,
                max_touched_pages(total_tokens, num_requests, layer.page_size),
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
            );
        }
        Ok(())
    }

    /// The quantised append, `kv_paged.cu:130-190` — four schemes, three
    ///
    /// As [`write_kv_to_pages_bf16`]; the layer's scale planes must be
    /// sized for its scheme.
    pub fn write_kv_to_pages_quantised(
        ctx: &Ctx,
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
    ) -> Result<(), Refusal> {
        let page_size = layer.page_size;
        let h_kv = layer.num_kv_heads;
        let d = layer.head_dim;
        let tokens = total_tokens.unsigned_abs();
        let heads = h_kv.unsigned_abs();

        match layer.scheme {
            // SAFETY, in all three arms: `call()`'s contract -- every pointer
            // bound here addresses live device memory of the extent the
            // kernel reads it as.
            KvScheme::Fp8PerTensor => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::write_kv_fp8_per_tensor",
                    Launch::per_row(tokens, BLOCK),
                    &[
                        k_curr.arg(),
                        v_curr.arg(),
                        layer.k_pages.cast::<u8>().arg(),
                        layer.v_pages.cast::<u8>().arg(),
                        qo_indptr.arg(),
                        kv_page_indices.arg(),
                        kv_page_indptr.arg(),
                        kv_last_page_lens.arg(),
                        num_requests.arg(),
                        page_size.arg(),
                        h_kv.arg(),
                        d.arg(),
                        fp8_kind_of(layer.storage_dtype).arg(),
                    ],
                )
            },

            KvScheme::Int8PerTokenHead | KvScheme::Fp8PerTokenHead => {
                let instantiation = if matches!(layer.scheme, KvScheme::Fp8PerTokenHead) {
                    "::pie::attn::write_kv_per_token_head<::pie::true_type::value>"
                } else {
                    "::pie::attn::write_kv_per_token_head<::pie::false_type::value>"
                };
                // One `float` per warp, twice: the per-token-per-head scale
                // is a max over the head, reduced warp-wise for k and for v.
                let smem = 2 * (BLOCK / 32) * (core::mem::size_of::<f32>() as u32);
                unsafe {
                    ctx.launch(
                        "attn/kv_paged.cuh",
                        instantiation,
                        Launch::grid([tokens, heads, 1], [BLOCK, 1, 1]).smem(smem),
                        &[
                            k_curr.arg(),
                            v_curr.arg(),
                            layer.k_pages.arg(),
                            layer.v_pages.arg(),
                            layer.k_scales.cast::<f32>().arg(),
                            layer.v_scales.cast::<f32>().arg(),
                            qo_indptr.arg(),
                            kv_page_indices.arg(),
                            kv_page_indptr.arg(),
                            kv_last_page_lens.arg(),
                            num_requests.arg(),
                            page_size.arg(),
                            h_kv.arg(),
                            d.arg(),
                        ],
                    )
                }
            }

            KvScheme::Fp4Block => {
                let block_size = fp4_block_size(layer);
                let blocks = d.div_euclid(block_size) + i32::from(d.rem_euclid(block_size) != 0);
                unsafe {
                    ctx.launch(
                        "attn/kv_paged.cuh",
                        "::pie::attn::write_kv_fp4_block",
                        Launch::grid([tokens, heads, blocks.unsigned_abs()], [32, 1, 1]),
                        &[
                            k_curr.arg(),
                            v_curr.arg(),
                            layer.k_pages.cast::<u8>().arg(),
                            layer.v_pages.cast::<u8>().arg(),
                            layer.k_scales.cast::<f32>().arg(),
                            layer.v_scales.cast::<f32>().arg(),
                            qo_indptr.arg(),
                            kv_page_indices.arg(),
                            kv_page_indptr.arg(),
                            kv_last_page_lens.arg(),
                            num_requests.arg(),
                            page_size.arg(),
                            h_kv.arg(),
                            d.arg(),
                            block_size.arg(),
                        ],
                    )
                }
            }

            KvScheme::Native => {
                Err(Refusal::Absent { what: "a quantised writer for Native storage" })
            }
        }
    }

    /// `attn::write_kv_to_pages` — the entry point, which chooses.
    ///
    /// As [`write_kv_to_pages_bf16`].
    pub fn write_kv_to_pages(
        ctx: &Ctx,
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        row_valid: *const u8,
        first_token: i32,
    ) -> Result<(), Refusal> {
        // A partial write addresses its destination by counting from
        // `first_token`, and only the native appender takes that operand: the
        // three quantised kernels have no parameter for it.
        if first_token != 0 && !layer.is_native_bf16 {
            return Err(Refusal::Absent {
                what: "a quantised appender that skips the first tokens",
            });
        }
        if layer.is_native_bf16 {
            return write_kv_to_pages_bf16(
                ctx,
                layer,
                k_curr,
                v_curr,
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                total_tokens,
                num_requests,
                row_valid,
                first_token,
            );
        }
        write_kv_to_pages_quantised(
            ctx,
            layer,
            k_curr,
            v_curr,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            total_tokens,
            num_requests,
        )
    }

    /// The fp8-per-tensor arm, called by name from
    ///
    /// What the caller must guarantee, as `call()` states it:
    /// `kv_page_indices` must list `num_pages_in_batch` valid page indices,
    /// and the layer's bf16 mirror planes must be sized for them.
    pub fn dequant_fp8_per_tensor_pages_active(
        ctx: &Ctx,
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
    ) -> Result<(), Refusal> {
        if layer.is_native_bf16 {
            return Err(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if !matches!(layer.scheme, KvScheme::Fp8PerTensor) {
            return Err(Refusal::Absent { what: "an fp8-per-tensor layer" });
        }

        let (logical_n, page_elems, launch) = active_geometry(layer, num_pages_in_batch);
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/kv_paged.cuh",
                "::pie::attn::dequant_fp8_pages_active",
                launch,
                &[
                    layer.k_pages.cast::<u8>().cast_const().arg(),
                    layer.v_pages.cast::<u8>().cast_const().arg(),
                    layer.k_bf16_pages.cast::<bf16>().arg(),
                    layer.v_bf16_pages.cast::<bf16>().arg(),
                    kv_page_indices.arg(),
                    logical_n.arg(),
                    page_elems.arg(),
                    fp8_kind_of(layer.storage_dtype).arg(),
                ],
            )
        }
    }

    /// The element count an active-page pass covers, and the grid that
    fn active_geometry(layer: &KvLayer, num_pages_in_batch: i32) -> (i64, i32, Launch) {
        let page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
        let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
        let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
        (logical_n, page_elems, Launch::grid([blocks as u32, 1, 1], [BLOCK, 1, 1]))
    }

    /// `attn::dequant_kv_cache_layer_to_bf16_active` — dequantise the pages
    ///
    /// As [`dequant_fp8_per_tensor_pages_active`].
    pub fn dequant_kv_cache_layer_to_bf16_active(
        ctx: &Ctx,
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
    ) -> Result<(), Refusal> {
        if layer.is_native_bf16 {
            return Err(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        let (logical_n, _page_elems, launch) = active_geometry(layer, num_pages_in_batch);

        match layer.scheme {
            KvScheme::Fp8PerTensor => {
                dequant_fp8_per_tensor_pages_active(ctx, layer, kv_page_indices, num_pages_in_batch)
            }

            // SAFETY, in all three arms: `call()`'s contract -- every pointer
            // bound here addresses live device memory of the extent the
            // kernel reads it as.
            KvScheme::Fp8PerTokenHead => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_fp8_per_token_head_pages_active<::pie::bf16>",
                    launch,
                    &[
                        layer.k_pages.cast::<u8>().cast_const().arg(),
                        layer.v_pages.cast::<u8>().cast_const().arg(),
                        layer.k_scales.cast::<f32>().cast_const().arg(),
                        layer.v_scales.cast::<f32>().cast_const().arg(),
                        layer.k_bf16_pages.cast::<bf16>().arg(),
                        layer.v_bf16_pages.cast::<bf16>().arg(),
                        kv_page_indices.arg(),
                        logical_n.arg(),
                        layer.page_size.arg(),
                        layer.num_kv_heads.arg(),
                        layer.head_dim.arg(),
                    ],
                )
            },

            KvScheme::Int8PerTokenHead => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_int8_per_token_head_pages_active<::pie::bf16>",
                    launch,
                    &[
                        layer.k_pages.cast::<i8>().cast_const().arg(),
                        layer.v_pages.cast::<i8>().cast_const().arg(),
                        layer.k_scales.cast::<f32>().cast_const().arg(),
                        layer.v_scales.cast::<f32>().cast_const().arg(),
                        layer.k_bf16_pages.cast::<bf16>().arg(),
                        layer.v_bf16_pages.cast::<bf16>().arg(),
                        kv_page_indices.arg(),
                        logical_n.arg(),
                        layer.page_size.arg(),
                        layer.num_kv_heads.arg(),
                        layer.head_dim.arg(),
                    ],
                )
            },

            KvScheme::Fp4Block => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_fp4_pages_active<::pie::bf16>",
                    launch,
                    &[
                        layer.k_pages.cast::<u8>().cast_const().arg(),
                        layer.v_pages.cast::<u8>().cast_const().arg(),
                        layer.k_scales.cast::<f32>().cast_const().arg(),
                        layer.v_scales.cast::<f32>().cast_const().arg(),
                        layer.k_bf16_pages.cast::<bf16>().arg(),
                        layer.v_bf16_pages.cast::<bf16>().arg(),
                        kv_page_indices.arg(),
                        logical_n.arg(),
                        layer.page_size.arg(),
                        layer.num_kv_heads.arg(),
                        layer.head_dim.arg(),
                        fp4_block_size(layer).arg(),
                    ],
                )
            },

            KvScheme::Native => {
                Err(Refusal::Absent { what: "a quantised dequant for Native storage" })
            }
        }
    }
}

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
const BLOCK: u32 = 256;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
#[must_use]
const fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    /// `u32::clamp` is not `const`, and the rule's expression is transcribed
    #[must_use]
    const fn head_dim_block(head_dim: u32) -> u32 {
        /// `runtime/launch.rs:610` — `const SINK_BLOCK_MAX: u32 = 128;`.
        const SINK_BLOCK_MAX: u32 = 128;

        /// `runtime/launch.rs:608` — `const SINK_BLOCK_MIN: u32 = WARP;`.
        const SINK_BLOCK_MIN: u32 = 32;

    if head_dim < SINK_BLOCK_MIN {
    SINK_BLOCK_MIN
    } else if head_dim > SINK_BLOCK_MAX {
    SINK_BLOCK_MAX
    } else {
    head_dim
    }
    }

    Launch::grid([rows, heads, 1], [head_dim_block(head_dim), 1, 1])
}
/// `LaunchRule::PerHead`, as the expression it evaluates to.
#[must_use]
const fn per_head(rows: u32, heads: u32) -> Launch {
    /// `runtime/launch.rs:599` — `const PAD_BLOCK: u32 = 128;`.
    const PAD_BLOCK: u32 = 128;

    Launch::grid([heads, rows, 1], [PAD_BLOCK, 1, 1])
}
/// `attn::lse_log2_to_ln` — rebase flashinfer's LSE from log2 to ln, in place.
///
/// What the caller must guarantee, as `call()` states it: `lse` must address
/// `n` live, writable `f32`s.
pub fn lse_log2_to_ln(ctx: &Ctx, lse: *mut f32, n: usize) -> Result<(), Refusal> {
    let Ok(elems) = u32::try_from(n) else {
        return Err(Refusal::Wide {
            what: "lse elements",
            at: i64::from(i32::MAX),
            max: i64::from(i32::MAX),
        });
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attn_sink.cuh",
            "::pie::attn::lse_log2_to_ln<::pie::attn::f32>",
            elementwise(elems),
            &[lse.arg(), n.arg()],
        )
    }
}

/// `attn::attention_sink_rescale_bf16` — gpt-oss's per-head sink correction,
///
/// What the caller must guarantee, as `call()` states it: `o` addresses `n *
/// num_q_heads * head_dim` live, writable bf16 elements; `lse` addresses `n *
/// num_q_heads` live `f32`s; `sinks` addresses `num_q_heads` live bf16
/// elements.
pub fn attention_sink_rescale<T>(
    ctx: &Ctx,
    o: *mut T,
    lse: *const f32,
    sinks: *const T,
    n: i32,
    num_q_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attn_sink.cuh",
            &format!("::pie::attn::attn_sink_rescale<{}>", T::CPP),
            per_head_elementwise(
                n.unsigned_abs(),
                num_q_heads.unsigned_abs(),
                head_dim.unsigned_abs(),
            ),
            &[o.arg(), lse.arg(), sinks.arg(), n.arg(), num_q_heads.arg(), head_dim.arg()],
        )
    }
}

/// `attn::split_qkv_bf16_devwin` — the packed activation cut into Q, K and V,
///
/// What the caller must guarantee, as `call()` states it: `packed`, the three
/// outputs and `win` are device addresses live across the launch. The four
/// buffer pointers must be BASE pointers — the kernel windows them itself
/// from `win`, so a pre-windowed pointer is windowed twice. The binder
/// guarantees it by the `_devwin` suffix; a hand caller must not.
pub fn split_qkv_bf16_devwin(
    ctx: &Ctx,
    packed: *const bf16,
    q_out: *mut bf16,
    k_out: *mut bf16,
    v_out: *mut bf16,
    win: *const u32,
    n_max: i32,
    q_dim: i32,
    kv_dim: i32,
) -> Result<(), Refusal> {
    /// `split_packed.cu:30` — `constexpr int BLOCK = 256;`.
    pub const SPLIT_BLOCK: u32 = 256;

    let max_dim = if q_dim > kv_dim { q_dim } else { kv_dim };
    let xblocks = max_dim.unsigned_abs().div_ceil(SPLIT_BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/split_packed.cuh",
            "::pie::attn::split_qkv_devwin<::pie::bf16>",
            Launch::grid([xblocks.max(1), n_max.unsigned_abs(), 1], [SPLIT_BLOCK, 1, 1]),
            &[
                packed.arg(),
                q_out.arg(),
                k_out.arg(),
                v_out.arg(),
                win.arg(),
                q_dim.arg(),
                kv_dim.arg(),
            ],
        )
    }
}

/// `attn::attention_naive_paged` — the reference paged attention.
///
/// What the caller must guarantee, as `call()` states it: every pointer must
/// address live device memory of the extent the kernel reads or writes.
pub fn attention_naive_paged(
    ctx: &Ctx,
    layer: &crate::attn::KvLayer,
    q: *const bf16,
    o: *mut bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    q_width: i32,
    window_left: i32,
    sm_scale: f32,
    logits_soft_cap: f32,
    lse_out: *mut f32,
) -> Result<(), Refusal> {
    /// `attention_naive_paged.cuh:223` — `constexpr int kMaxHeadDim = 1024`.
    pub const PAGED_MAX_HEAD_DIM: i32 = 1024;

    /// `attention_naive_paged.cuh:33` — `constexpr int BLOCK = 128`.
    pub const PAGED_BLOCK: u32 = 128;

    if layer.head_dim > PAGED_MAX_HEAD_DIM {
        return Err(Refusal::Wide {
            what: "head_dim",
            at: i64::from(layer.head_dim),
            max: i64::from(PAGED_MAX_HEAD_DIM),
        });
    }
    let num_q_heads = q_width / layer.head_dim;
    let smem = (layer.head_dim.unsigned_abs() + PAGED_BLOCK) * 4;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attention_naive_paged.cuh",
            "::pie::attn::naive_paged_attn<::pie::i32(128)>",
            Launch::grid(
                [
                    num_requests.unsigned_abs(),
                    total_tokens.unsigned_abs(),
                    num_q_heads.unsigned_abs(),
                ],
                [PAGED_BLOCK, 1, 1],
            )
            .smem(smem),
            &[
                q.arg(),
                layer.k_pages.cast_const().arg(),
                layer.v_pages.cast_const().arg(),
                layer.k_scales.cast::<f32>().cast_const().arg(),
                layer.v_scales.cast::<f32>().cast_const().arg(),
                o.arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                core::ptr::null::<u8>().arg(),
                core::ptr::null::<i32>().arg(),
                num_q_heads.arg(),
                layer.num_kv_heads.arg(),
                layer.head_dim.arg(),
                layer.page_size.arg(),
                kv_scheme::of(layer.scheme).arg(),
                kv_dtype::of(layer.storage_dtype).arg(),
                layer.block_size.arg(),
                window_left.arg(),
                sm_scale.arg(),
                logits_soft_cap.arg(),
                lse_out.arg(),
            ],
        )
    }
}

/// `attn::attn_res_blend_bf16` — K3's residual-block blend.
///
/// What the caller must guarantee, as `call()` states it: `prefix` and `out`
/// address `t * h` live bf16 elements, `blocks` addresses `t * b * h`, and
/// `norm_weight` and `proj_weight` address `h` each, `out` writable.
pub fn attn_res_blend<T>(
    ctx: &Ctx,
    prefix: *const T,
    blocks: *const T,
    norm_weight: *const T,
    proj_weight: *const T,
    out: *mut T,
    t: i32,
    b: i32,
    h: i32,
    block_rows: i32,
    eps: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attn_res.cuh",
            &format!("::pie::attn::attn_res_blend<{}>", T::CPP),
            Launch::per_row(t.unsigned_abs(), BLOCK),
            &[
                prefix.arg(),
                blocks.arg(),
                norm_weight.arg(),
                proj_weight.arg(),
                out.arg(),
                b.arg(),
                h.arg(),
                block_rows.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `attn::pad_head_dim_bf16` — pad every head out to a width flashinfer
///
/// What the caller must guarantee, as `call()` states it: `packed` addresses
/// `num_tokens * num_heads * head_dim` live bf16 elements and `padded`
/// addresses `num_tokens * num_heads * head_dim_padded` writable ones.
pub fn pad_head_dim<T>(
    ctx: &Ctx,
    packed: *const T,
    padded: *mut T,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Err(why);
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/head_dim_pad.cuh",
            &format!("::pie::attn::pad_head_dim<{}>", T::CPP),
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            &[packed.arg(), padded.arg(), num_heads.arg(), head_dim.arg(), head_dim_padded.arg()],
        )
    }
}

/// `attn::strip_head_dim_bf16` — the inverse of [`pad_head_dim`].
///
/// What the caller must guarantee, as `call()` states it: `padded` addresses
/// `num_tokens * num_heads * head_dim_padded` live bf16 elements and `packed`
/// addresses `num_tokens * num_heads * head_dim` writable ones.
pub fn strip_head_dim<T>(
    ctx: &Ctx,
    padded: *const T,
    packed: *mut T,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Err(why);
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/head_dim_pad.cuh",
            &format!("::pie::attn::strip_head_dim<{}>", T::CPP),
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            &[padded.arg(), packed.arg(), num_heads.arg(), head_dim.arg(), head_dim_padded.arg()],
        )
    }
}

/// The four preconditions both head-dim launchers share, resolved BEFORE
#[must_use]
fn head_dim_refusal(
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Option<Refusal> {
    if num_tokens <= 0 {
        return Some(Refusal::Empty { what: "rows" });
    }
    if num_heads <= 0 {
        return Some(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Some(Refusal::Empty { what: "head_dim" });
    }
    if head_dim_padded < head_dim {
        return Some(Refusal::Narrow { what: "head_dim_padded", at: i64::from(head_dim_padded) });
    }
    None
}

/// The guard `attn_softcap.cu`'s launcher opened with, as a refusal.
fn softcap_launch(cap: f32, n: usize) -> Result<Launch, Refusal> {
    if cap.is_nan() || cap <= 0.0 {
        return Err(Refusal::Unstated { what: "a logit soft cap" });
    }
    let Ok(elems) = u32::try_from(n) else {
        return Err(Refusal::Wide {
            what: "logit elements",
            at: i64::from(i32::MAX),
            max: i64::from(i32::MAX),
        });
    };
    Ok(elementwise(elems))
}

/// `attn::logit_softcap_bf16` — gemma's final logit cap, in place.
///
/// What the caller must guarantee, as `call()` states it: `x` must address
/// `n` live, writable `bf16`s.
pub fn logit_softcap<T>(ctx: &Ctx, x: *mut T, cap: f32, n: usize) -> Result<(), Refusal>
where
    T: Elem,
    *mut T: Abi,
{
    let launch = softcap_launch(cap, n)?;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/softcap.cuh",
            &format!("::pie::attn::logit_softcap<{}>", T::CPP),
            launch,
            &[x.arg(), cap.arg(), n.arg()],
        )
    }
}

/// `attn::logit_softcap_f16` — the same cap over an fp16 buffer.
///
/// [`logit_softcap`]'s obligation, with `f16` for `bf16`.
pub fn logit_softcap_f16(ctx: &Ctx, x: *mut f16, cap: f32, n: usize) -> Result<(), Refusal> {
    let launch = softcap_launch(cap, n)?;
    // SAFETY: as [`logit_softcap`]'s.
    unsafe {
        ctx.launch(
            "attn/softcap.cuh",
            "::pie::attn::logit_softcap<::pie::f16>",
            launch,
            &[x.arg(), cap.arg(), n.arg()],
        )
    }
}

/// `attn::kimi_split_q_b_bf16` — split a fused query projection into its
pub fn kimi_split_q_b<T>(
    ctx: &Ctx,
    q_b: *const T,
    q_nope: *mut T,
    q_pe: *mut T,
    tokens: i32,
    heads: i32,
    nope: i32,
    rope: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
    let total = i64::from(tokens) * width;
    if total > i64::from(i32::MAX) {
        return Err(Refusal::Wide {
            what: "rows",
            at: i64::from(tokens),
            max: i64::from(i32::try_from(i64::from(i32::MAX) / width).unwrap_or(i32::MAX)),
        });
    }
    let total = total as i32;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/kimi_mla.cuh",
            &format!("::pie::attn::split_q_b<{}>", T::CPP),
            elementwise(total.unsigned_abs()),
            &[
                q_b.arg(),
                q_nope.arg(),
                q_pe.arg(),
                total.arg(),
                heads.arg(),
                nope.arg(),
                rope.arg(),
            ],
        )
    }
}

/// `attn::kimi_split_kv_a_norm_bf16` — split `kv_a`, RMS-normalise the latent
pub fn kimi_split_kv_a_norm<T>(
    ctx: &Ctx,
    kv_a: *const T,
    norm_weight: *const T,
    kv_c: *mut T,
    k_pe: *mut T,
    tokens: i32,
    kv_lora: i32,
    rope: i32,
    src_row_stride: i32,
    eps: f32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    /// `LaunchRule::Rms`, as the expression it evaluates to.
    #[must_use]
    const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
    }

    if src_row_stride < kv_lora + rope {
        return Err(Refusal::Narrow { what: "src_row_stride", at: i64::from(src_row_stride) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/kimi_mla.cuh",
            &format!("::pie::attn::split_kv_a_norm<{}, 256>", T::CPP),
            rms(tokens.unsigned_abs()),
            &[
                kv_a.arg(),
                norm_weight.arg(),
                kv_c.arg(),
                k_pe.arg(),
                kv_lora.arg(),
                rope.arg(),
                src_row_stride.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `attn::combine_attn_outputs_bf16` — merge two attention halves and their
///
/// What the caller must guarantee, as `call()` states it: every pointer must
/// address the extents these three numbers describe.
pub fn combine_attn_outputs<T>(
    ctx: &Ctx,
    o1: *const T,
    lse1: *const f32,
    o2: *const T,
    lse2: *const f32,
    o_out: *mut T,
    lse_out: *mut f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    /// The merge's geometry — the grid of [`per_head_elementwise`] and a
    #[must_use]
    const fn combine_attn(rows: u32, heads: u32, head_dim: u32) -> Launch {
        /// `[32, 256]`, transcribed rather than rearranged — `u32::clamp` is not
        #[must_use]
        const fn combine_block(head_dim: u32) -> u32 {
            /// `dsv4_compress.cu:87`'s `(head_dim > 256) ? 256`. **Not
            const COMBINE_BLOCK_MAX: u32 = 256;

            /// A warp. `dsv4_compress.cu:87`'s `(head_dim < 32) ? 32`.
            const COMBINE_BLOCK_MIN: u32 = 32;

        if head_dim < COMBINE_BLOCK_MIN {
        COMBINE_BLOCK_MIN
        } else if head_dim > COMBINE_BLOCK_MAX {
        COMBINE_BLOCK_MAX
        } else {
        head_dim
        }
        }

    Launch::grid([rows, heads, 1], [combine_block(head_dim), 1, 1])
    }

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsv4_compress.cuh",
            &format!("::pie::attn::combine_attn_outputs<{}>", T::CPP),
            combine_attn(n.unsigned_abs(), num_heads.unsigned_abs(), head_dim.unsigned_abs()),
            &[
                o1.arg(),
                lse1.arg(),
                o2.arg(),
                lse2.arg(),
                o_out.arg(),
                lse_out.arg(),
                num_heads.arg(),
                head_dim.arg(),
            ],
        )
    }
}

// MLA'S ABSORB PAIR IS `x::gemm::absorb` — `mla_absorb_q_to_latent_bf16` and
// `mla_absorb_latent_to_v_bf16`, with the `cublasGemmStridedBatchedEx` helper
// and its two pinned constants. They sat here because their CALLER is MLA's
// attention lane; they are `gemm`'s because a routine's symbol is its
// family's namespace plus its name, a trace states them as `gemm::`, and no
// `Family` resolved them while an `attn` module owned the host program. That
// file's header is the whole argument.

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated beside one is what no signature carries:
/// whether a statement consumes its whole operand, and which operands must be
/// given the same address.
///
/// **The layer-view host programs are DECLARED and are not `routine!`s**, and
/// ONE reason covers every one of them: `write_kv_to_pages`, the two
/// explicit-slot appends, `dequant_kv_cache_layer_to_bf16_active` and
/// `attention_naive_paged` take a `&KvLayer`, and `mla_prepare_bf16` and
/// `write_mla_to_pages` take an `MlaLayer`. Those are host aggregates rather
/// than kernel arguments — a layer view is five to eighteen operands the fire
/// resolves together — and `Arg` is implemented for no such type, so a
/// `routine!` naming one does not compile. `driver_bound!` declares them off
/// the same `fn`s with `args` left empty, which is the honest column: how a
/// trace states a layer view is still an open question, and a declaration is
/// not an answer to it.
///
/// **Three more of the paged-KV host programs are absent, and all three are
/// inner arms rather than symbols.** `write_kv_to_pages_bf16` and
/// `write_kv_to_pages_quantised` are what `write_kv_to_pages` chooses between
/// on `KvLayer::is_native_bf16`, and `dequant_fp8_per_tensor_pages_active` is
/// one scheme of the dequantiser's `match`. No trace names any of the three,
/// because the choice is a property of the layer and not of the statement.
///
/// **Three of the four score-capture launches are absent for a different
/// reason, and it is not a gap.** [`attention_score_post`]'s three are fired
/// by `driver-cuda`'s `fire::attn_score`, at the point on the fire's stream
/// where the C++ capture dispatch used to issue them, and were never
/// statements at all — a `routine!` for one would put a row in
/// `crate::sigs()` that nothing could lower to. The fold is the exception and
/// is declared below: `dsl::cuda::attn_score_fold_heads` states the CONTRACT
/// symbol `attn::attn_score_fold_heads`, which is a different symbol from the
/// device one this file's `_dev` suffix marks, so that row IS lowerable and
/// had been carried by hand in `not_yet_crossed.rs` until this table could
/// derive it.
///
/// **MLA's two DISPATCH arms are absent, and the symbol they answer is not.**
/// [`mla_fa2::fire`] and [`mla_naive::fire`] are ARMS rather than symbols, in
/// the sense `write_kv_to_pages_bf16` and `write_kv_to_pages_quantised` are:
/// no trace names either, because which one runs is a property of the device
/// and not of the statement. What a trace names is
/// `attn::dispatch_attention_mla_bf16`, and that is
/// [`dispatch_attention_mla_bf16`] — the `fn` that chooses — declared below.
pub static ROUTINES: &[Routine] = &[
    routine!(lse_log2_to_ln, in_place = &[(0, 0)]),
    routine!(attention_sink_rescale_bf16 = attention_sink_rescale::<bf16>, in_place = &[(0, 0)]),
    routine!(attn_res_blend_bf16 = attn_res_blend::<bf16>),
    routine!(pad_head_dim_bf16 = pad_head_dim::<bf16>),
    routine!(strip_head_dim_bf16 = strip_head_dim::<bf16>),
    routine!(logit_softcap_bf16 = logit_softcap::<bf16>, in_place = &[(0, 0)]),
    routine!(logit_softcap_f16, in_place = &[(0, 0)]),
    routine!(kimi_split_q_b_bf16 = kimi_split_q_b::<bf16>),
    routine!(kimi_split_kv_a_norm_bf16 = kimi_split_kv_a_norm::<bf16>),
    routine!(combine_attn_outputs_bf16 = combine_attn_outputs::<bf16>),
    routine!(split_qkv_bf16_devwin),
    routine!(compact_page_csr, whole),
    routine!(mtp_shift_hidden_bf16 = mtp_shift_hidden::<bf16>, whole),
    routine!(mtp_update_pending_hidden_bf16 = mtp_update_pending_hidden::<bf16>, whole),
    // THESE THREE LOST A `whole` THEY NEVER EARNED, and the loss is a
    // restoration. All three were `whole: false` in `table/attn.rs` at
    // `9e3936fb9^`; each acquired `whole` when its family crossed, no comment
    // anywhere said why, and the check that would have caught it — §12c's
    // contract-agreement twin — had died with `unit!`.
    //
    // The evidence is each one's own launch shape. `dsa_index_{knorm,q}_rope`
    // are `Launch::per_row(tokens)` and `dsv4_boundary_meta_decode` is
    // `Launch::flat(n)`; none of the three reads an R-shaped index, so a row
    // window slices them exactly where the lowering says it may. The old
    // table knew this and said so CONTRASTIVELY, in the note beside
    // `dsa_index_topk_mask` two lines down: *"`whole`, and here the reason is
    // the ALGEBRA rather than the addressing."* A topk over a mask is one
    // answer computed from every row; a rope over the indexer's query rows is
    // per-row by construction. `topk_mask` needed its own sentence and these
    // did not, which is why it keeps its `whole` here.
    //
    // The direction matters. `false → true` can only ever REFUSE more —
    // `model-compiler`'s `kernels.rs` turns a `whole` kernel inside a Peel
    // region into a load-time refusal, and `lower.rs` raises
    // `Uncovered::WholeKernelSplit` for one emitted over a row window — so
    // the drift could not corrupt an answer and could refuse a model text
    // that used to load. It is being corrected as a wrong claim, not as a
    // bug: `glm_5` and `deepseek_v4` state all four, and neither builds a
    // Peel region, so nothing observable changes today either way.
    routine!(dsa_index_knorm_rope_bf16 = dsa_index_knorm_rope::<bf16>),
    routine!(dsa_index_q_rope_bf16 = dsa_index_q_rope::<bf16>),
    routine!(dsa_index_topk_mask, whole),
    routine!(dsv4_boundary_meta_decode),
    // AND THIS ONE KEEPS IT, which is why the four were not one decision.
    // Its argument list is not its neighbours': it takes `qo_indptr` and
    // `num_requests` and walks them, which is exactly the R-shaped addressing
    // the old table used to JUSTIFY `whole` on the paged MLA statements —
    // a row window leaves that arithmetic pointing at another request's
    // rows. So the `false → true` here reads as a deliberate correction made
    // during the crossing rather than a transcription slip, and reverting it
    // would reintroduce the fault the other three never had.
    routine!(dsv4_boundary_meta_paged, whole),
    routine!(attention_compressed_paged_bf16, whole),
    routine!(dsv4_compress_gather_paged_bf16),
    routine!(dsv4_store_comp_entries_bf16, whole),
    routine!(qkv_packed_qk_norm_rope_vnorm_write_kv_bf16),
    routine!(qkv_decode_qk_norm_rope_write_kv_bf16),
    routine!(qkv_decode_fused_dispatch),
    // ── what the DRIVER fires, by path ──────────────────────────────────
    //
    // Below this line, every symbol is `driver_bound!` rather than
    // `routine!`, and for all but the last of them the difference is one
    // fact: **no statement supplies its arguments.** A paged-KV write takes
    // the layer's page geometry, an MLA prepare takes the latent cache's —
    // each assembled by the driver from the fire, mentioned by no trace, and
    // therefore not describable by the extractor that reads a `fn`'s
    // parameters. The reason is written per symbol rather than once, because
    // the last one's is a different reason.
    //
    // They were rows in `not_yet_crossed.rs`, hand-transcribing columns off
    // a `fn` sitting three thousand lines up this same file. The body was
    // never what was missing.
    //
    // THE FIVE PAGED-KV SYMBOLS each take a `&KvLayer` -- one layer's page
    // geometry, five to eighteen operands the fire resolves together and no
    // trace states as one. Four of the five are armed over exactly these
    // `fn`s in `bind/arms/attn.rs`; the device-window twin has the host
    // program and no arm on either side of the seam.
    driver_bound!(write_kv_to_pages),
    driver_bound!(write_kv_explicit_bf16),
    driver_bound!(write_kv_explicit_bf16_devwin, whole),
    driver_bound!(dequant_kv_cache_layer_to_bf16_active),
    // The head dims FlashInfer's prefill template rejects (gemma-4's 512)
    // take this naive paged kernel instead: no plan at all, fire-shaped, and
    // `attention_naive_paged_arm` is what fires it. It is all that is left of
    // the dispatch lattice's row block -- the six FlashInfer dispatches are
    // `attn::fa2::ROUTINES`' now, and this shares their arm file and none of
    // their shape.
    driver_bound!(attention_naive_paged, whole),
    // The fold is DECLINED in `bind/arms/attn.rs` for one operand -- the
    // score-capture CSR, which has a producer and no `Cx` query -- and fires
    // out of band from `fire::attn_score`, at the point on the stream where
    // the capture dispatch used to issue it. Its `&KvLayer` is absent from
    // the argument list and the reason is not: `page_size` reaches it off
    // `Cx::kv_layer`, so what a statement cannot supply here is the CSR
    // rather than the view. **Declaring it is not arming it** -- the row is
    // what a lowering resolves against, and the `unbound:` sentence beside
    // its arm is what a fire still meets.
    driver_bound!(attn_score_fold_heads, whole),
    // The MLA pair take an `MlaLayer`, which is the same reason in the latent
    // cache's shape. Both are `whole` for the ADDRESSING rather than the
    // algebra: they walk `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`,
    // which are R-shaped, so a row window would leave that arithmetic
    // pointing at another request's rows. `bind/arms/attn.rs` declines both
    // over `Cx::mla_layer`, whose producer -- `pools::mla_cache::
    // MlaCachePool::layer_view` -- no `Fire` reaches; `fire/launch.rs`'s
    // `kv_pools_for` refuses `KvStyle::Mla` and `serve/load.rs` refuses an
    // MLA checkpoint at load. Three things must move together, and a
    // declaration is none of them.
    driver_bound!(mla_prepare_bf16, whole),
    driver_bound!(write_mla_to_pages, whole),
    // AND THE MLA DISPATCH, WHICH WAS THE LAST `attn` ROW IN
    // `not_yet_crossed.rs` AND IS THE REASON THAT FILE HELD ONE. Its row's
    // comment is preserved here because it is an argument and not a label,
    // and because deleting it with the row would delete the only record of
    // why the row outlived the eight above it.
    //
    // The row said: *"it is ONE trace symbol with TWO bodies ... A `Routine`
    // is one body. `driver_bound!` names one `fn`, so it cannot declare this
    // any more than `routine!` could, and the decision it is waiting on is
    // which of the two the declaration should name -- or whether the pick
    // belongs inside a third `fn` that takes the `Ctx` and forwards."*
    //
    // **It is the third `fn`**, and the row had already named it. What the
    // row was really recording is that `mla_fa2::fire` and `mla_naive::fire`
    // are ARMS -- the choice between them is `Ctx::compute_capability_major`
    // (`>= 10` picks naive, because FA2 MLA writes ZERO OUTPUT on sm_100, a
    // wrong answer rather than a fault) -- and an arm is not a symbol.
    // `write_kv_to_pages` had settled the same question for the paged-KV
    // writes in this same file, and the paragraph above says so in its own
    // words: the entry point that chooses is the symbol, the two arms it
    // chooses between are not, and neither arm is a weaker declaration of
    // the other.
    //
    // NO `whole`, and the contrast with the two rows above is the argument,
    // in the deleted row's own words: the prepare and the page write are
    // `whole` because they walk `qo_indptr` / `kv_page_indptr` /
    // `kv_last_page_lens`, which are R-shaped; *"the dispatch reads a plan
    // built over the whole fire and still covers a row range, like the
    // FlashInfer ones."* `tests/stated_columns.rs` pins it against
    // `9e3936fb9^`, which stated it the same way.
    //
    // **Declaring it is not arming it, and here that is more than the usual
    // caveat.** The refusal runs the other way round from the prepare's:
    // `bind/arms/attn.rs` declines this over `Cx::mla_layer` and
    // `Cx::mla_plan`, which have nothing to answer WITH, because
    // `fire/launch.rs`'s `kv_pools_for` refuses `KvStyle::Mla` and
    // `serve/load.rs` refuses an MLA checkpoint at model load. THREE things
    // must move together; the one that moved here is the first, which is
    // that there was no `fn` to arm at all. *"What is missing is a caller"*
    // is still the sentence beside the arm, and it is still true.
    driver_bound!(dispatch_attention_mla_bf16),
    // ── and one whose body is `driver_internal`'s ───────────────────────
    //
    // gemma-4 and the llama-like anchor lower `OpKind::SplitQkv` outside a
    // peel tail to this symbol (`model-compiler`'s `lower.rs`), so a live
    // model text names it and nothing declared it. The host program is
    // `driver_internal::split_qkv_bf16` and STAYS there; the declaration
    // cannot follow it, because `Family::symbol` is the module path's first
    // segment plus the routine's name and a `Family` in `driver_internal`
    // would offer `driver_internal::split_qkv_bf16` -- a string no lowering
    // emits. `driver_internal`'s header carries the whole argument, for this
    // and for the three sibling symbols in `layout`, `mlp` and `ssm`.
    //
    // **This declares the symbol and does not arm it.** A fire naming it
    // still refuses with `NoArm`: `bind/arms/attn.rs` binds the device-window
    // twin `attn::split_qkv_bf16_devwin` and nothing else, and the two are
    // different kernels rather than one with a fallback.
    driver_bound!(split_qkv_bf16),
];

/// `attn`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

// ── what a statement cannot supply, for this family ──────────────────
//
// From `x::cx`, which held eleven types whose only shared property was that
// a statement cannot supply them. "Context" is not a classification: it
// groups by how a value ARRIVES rather than by what it means, so the module
// had to be named for the arrival and `driver-cuda` wrote `x::cx::KvLayer`
// for a thing that is attention's and nothing else's. `Yarn` went to
// `rope` and `Gdn`/`Slab` to `ssm`; these eight are here.
//
// **[`Plan`] below is not [`plan::Plan`](crate::attn::plan::Plan)**, and the
// two now sit one path segment apart, so the difference is worth the line.
// This one is the per-request INDEX ARRAYS a paged launch walks -- four
// device pointers the fire already holds. `plan::Plan` is what the
// scheduler PRODUCES: an offset table plus the upload that fills it.

/// WHICH ROWS this fire is launching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows {
    /// The first row of this region.
    pub start: i32,
    /// How many rows this region serves.
    pub count: i32,
    /// How many rows the whole fire has.
    pub total: i32,
}
/// How a KV cache stores its elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvScheme {
    /// Pages hold the model's own element type; no scale tensors apply.
    Native = 0,
    /// One fp8 scale for the whole tensor.
    Fp8PerTensor = 1,
    /// One int8 scale per (token, head).
    Int8PerTokenHead = 2,
    /// One fp8 scale per (token, head).
    Fp8PerTokenHead = 3,
    /// fp4, blocked — `block_size` is the block.
    Fp4Block = 4,
}
/// The element type a KV page actually holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvDType {
    /// The model's own bf16.
    Bf16 = 0,
    /// fp16.
    Fp16 = 1,
    /// int8, under a per-tensor or per-token-head scheme.
    Int8 = 3,
    /// fp8 e4m3.
    Fp8E4M3 = 7,
    /// fp8 e5m2.
    Fp8E5M2 = 8,
}
/// One layer's paged KV cache, as the launchers take it.
#[derive(Clone, Copy, Debug)]
pub struct KvLayer {
    /// The layer's key pages.
    pub k_pages: *mut c_void,
    /// The layer's value pages.
    pub v_pages: *mut c_void,
    /// Rows per page.
    pub page_size: i32,
    /// Elements per head.
    pub head_dim: i32,
    /// How many KV heads the cache holds.
    pub num_kv_heads: i32,
    /// The pages are `[head, page, dim]` rather than `[page, head, dim]`.
    pub hnd: bool,
    /// How the pages are quantised, and whether the scale tensors apply.
    pub scheme: KvScheme,
    /// What a page element actually is — the model's dtype only under
    pub storage_dtype: KvDType,
    /// The quantisation block, meaningful under [`KvScheme::Fp4Block`].
    pub block_size: i32,
    /// How many pages the layer's arena holds.
    pub num_pages: i32,
    /// Key scales, null under [`KvScheme::Native`].
    pub k_scales: *mut c_void,
    /// Value scales, likewise.
    pub v_scales: *mut c_void,
    /// The bf16 shadow of the key pages, when a dequantised copy exists.
    pub k_bf16_pages: *mut c_void,
    /// The value shadow.
    pub v_bf16_pages: *mut c_void,
    /// Per-page key minimum, for the envelope path.
    pub k_env_min: *mut u16,
    /// Per-page key maximum.
    pub k_env_max: *mut u16,
    /// **A `bool`, not the fields it derives from.**
    pub has_envelopes: bool,
    /// Storage is the model's own bf16, so no dequantisation step applies.
    pub is_native_bf16: bool,
}
/// One layer's LATENT cache, as MLA's launchers take it.
#[derive(Clone, Copy, Debug)]
pub struct MlaLayer {
    /// The layer's compressed-latent pages.
    pub ckv_pages: *mut c_void,
    /// The layer's RoPE'd key pages.
    pub kpe_pages: *mut c_void,
    /// Rows per page.
    pub page_size: i32,
    /// The latent rank — `ckv`'s width.
    pub kv_lora_rank: i32,
    /// The RoPE'd half's head dimension — `kpe`'s width.
    pub qk_rope_head_dim: i32,
}
/// The attention workspace a fire was given.
#[derive(Clone, Copy, Debug)]
pub struct AttnWorkspace {
    /// The `float` half.
    pub float_buffer: *mut c_void,
    /// How many bytes it holds.
    pub float_bytes: usize,
    /// The `int` half.
    pub int_buffer: *mut c_void,
    /// How many bytes it holds.
    pub int_bytes: usize,
}
/// The plan `Prepare::MlaPlan` built for this fire.
#[derive(Clone, Copy, Debug)]
pub struct MlaPlan {
    /// The offsets and extents the scheduler computed.
    pub info: crate::attn::plan::info::MlaPlanInfo,
    /// The `int` arena those offsets index.
    pub int_arena: *mut c_void,
    /// The `float` arena.
    pub float_arena: *mut c_void,
}
/// The per-request index arrays a paged write needs.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// Where each request's query rows begin.
    pub qo_indptr: *const u32,
    /// Which pages each request holds.
    pub kv_page_indices: *const u32,
    /// Where each request's page list begins.
    pub kv_page_indptr: *const u32,
    /// How many rows of the last page each request uses.
    pub kv_last_page_lens: *const u32,
    /// Which rows are live, or null when every row is.
    pub row_valid: *const u8,
    /// How many requests.
    pub requests: i32,
}

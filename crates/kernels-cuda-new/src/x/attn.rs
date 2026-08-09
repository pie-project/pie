#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::{bf16, f16};
#[cfg(not(feature = "_cuda"))]
use crate::x::cx::{MlaLayer, Yarn};
use kernels::Refusal;

// `routine!` names a `fn` by IDENTIFIER, so the five host programs that live
// beside the `unit!` whose kernels they launch are brought into scope here to
// be named in `ROUTINES`. The routine's name is the identifier and not the
// path, which is what keeps it equal to its contract symbol's tail.
use dsv4_compress::{dsv4_compress_gather_paged_bf16, dsv4_store_comp_entries_bf16};
use qkv_fused::{
    qkv_decode_fused_dispatch, qkv_decode_qk_norm_rope_write_kv_bf16,
    qkv_packed_qk_norm_rope_vnorm_write_kv_bf16,
};

#[cfg(feature = "_cuda")]
use crate::x::cx::{MlaLayer, Yarn};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;
#[cfg(feature = "_cuda")]
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmStridedBatchedEx,
    cublasOperation_t, cublasStatus_t, cudaDataType,
};

/// `attn::device::KvScheme` — how a paged KV bank is quantised, as the device
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_scheme(pub u8);

impl kv_scheme {
    /// The device spelling of a [`Cx`](crate::x::Cx)-stated scheme.
    #[must_use]
    pub const fn of(scheme: crate::x::cx::KvScheme) -> Self {
        Self(scheme as i32 as u8)
    }
}

impl crate::x::Abi for kv_scheme {
    const CPP: &'static str = "::pie_cuda_driver::kernels::attn::device::KvScheme";
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

/// `attn::device::KvDType` — what a page element actually is.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_dtype(pub u8);

impl kv_dtype {
    /// The device spelling of a [`Cx`](crate::x::Cx)-stated storage dtype.
    #[must_use]
    pub const fn of(dtype: crate::x::cx::KvDType) -> Self {
        Self(dtype as i32 as u8)
    }
}

impl crate::x::Abi for kv_dtype {
    const CPP: &'static str = "::pie_cuda_driver::kernels::attn::device::KvDType";
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
        "::pie_cuda_driver::kernels::attn::device::StructuredMaskParams";

    /// The array of descriptors, one per lane, as `pack_structured_mask`
    impl crate::x::Abi for *const StructuredMaskParams {
        const CPP: &'static str =
            "const ::pie_cuda_driver::kernels::attn::device::StructuredMaskParams*";
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
         attn::device::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::align_of::<StructuredMaskParams>() == 4,
        "StructuredMaskParams: alignof disagrees with the measured \
         attn::device::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, kind) == 0,
        "StructuredMaskParams.kind: offset disagrees with the measured \
         attn::device::StructuredMaskParams::kind",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, window) == 4,
        "StructuredMaskParams.window: offset disagrees with the measured \
         attn::device::StructuredMaskParams::window",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, sink) == 8,
        "StructuredMaskParams.sink: offset disagrees with the measured \
         attn::device::StructuredMaskParams::sink",
    );

    /// The measured layout, as C++ `static_assert`s.
    pub static LAYOUTS: &[crate::x::Layout] = &[crate::x::Layout {
        cpp: STRUCTURED_MASK_PARAMS,
        size: 12,
        align: 4,
        fields: &[("kind", 0), ("window", 4), ("sink", 8)],
        probe: "nvrtc-probes/attn_structured_mask.py",
    }];
}

/// `attn/attn_sink.cuh` — gpt-oss's sink correction and the LSE rebase it
pub mod attn_sink {

    use crate::jit::Root;

    /// `attn/attn_sink.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "attn/attn_sink",
        include_str!("../../csrc/src/attn/attn_sink.cuh"),
        "attn/attn_sink.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Absolute, because a routine body names the instantiation itself rather
    /// than a label some other table maps to one. The `<...>` argument is what
    /// used to be a row's `elem`.
    pub(super) mod inst {
        /// `attn_sink.cuh:74` — the log2→ln rebase, over `f32`.
        pub const LSE_LOG2_TO_LN: &str = "::pie_cuda_driver::kernels::attn::device::lse_log2_to_ln\
             <::pie_cuda_driver::kernels::attn::device::f32>";
        /// `attn_sink.cuh:93` — the per-head sink correction.
        pub const ATTN_SINK_RESCALE: &str = "::pie_cuda_driver::kernels::attn::device::attn_sink_rescale\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `attn/attn_res.cuh` — K3's residual-block blend.
pub mod attn_res {

    use crate::jit::Root;

    /// `attn/attn_res.cuh` — the root this routine compiles a symbol out of.
    pub static ROOT: Root = Root::new(
        "attn/attn_res",
        include_str!("../../csrc/src/attn/attn_res.cuh"),
        "attn/attn_res.cuh",
    );

    /// The template-id NVRTC is handed, spelled as it is handed it.
    pub(super) mod inst {
        /// `attn_res.cuh:99` — K3's residual-block blend, at bf16.
        pub const ATTN_RES_BLEND: &str = "::pie_cuda_driver::kernels::attn::device::attn_res_blend\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `attn/head_dim_pad.cuh` — flashinfer's supported head widths, reached by
pub mod head_dim_pad {

    use crate::jit::Root;

    /// `attn/head_dim_pad.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "attn/head_dim_pad",
        include_str!("../../csrc/src/attn/head_dim_pad.cuh"),
        "attn/head_dim_pad.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub(super) mod inst {
        /// `head_dim_pad.cuh:73` — the pad.
        pub const PAD_HEAD_DIM: &str = "::pie_cuda_driver::kernels::attn::device::pad_head_dim\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `head_dim_pad.cuh:92` — the inverse.
        pub const STRIP_HEAD_DIM: &str = "::pie_cuda_driver::kernels::attn::device::strip_head_dim\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `attn/softcap.cuh` — the logit cap, at both numeric formats.
pub mod softcap {

    use crate::jit::Root;

    /// `attn/softcap.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "attn/softcap",
        include_str!("../../csrc/src/attn/softcap.cuh"),
        "attn/softcap.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// One template, two element types, and which one fires is the caller's
    /// choice of entry point rather than a host predicate.
    pub(super) mod inst {
        /// `softcap.cuh:67` — the cap over bf16.
        pub const LOGIT_SOFTCAP_BF16: &str = "::pie_cuda_driver::kernels::attn::device::logit_softcap\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same over f16.
        pub const LOGIT_SOFTCAP_F16: &str = "::pie_cuda_driver::kernels::attn::device::logit_softcap\
             <::pie_cuda_driver::kernels::device::f16>";
    }
}

/// `attn/split_packed.cuh` — the fused QKV product cut into three operands.
pub mod split_packed {

    use crate::jit::Root;

    /// `attn/split_packed.cuh` — the root this routine compiles a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "attn/split_packed",
        include_str!("../../csrc/src/attn/split_packed.cuh"),
        "attn/split_packed.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// `pub(in crate::x)` and not `pub(super)`: the host-window `split_qkv`
    /// is a driver op with no routine in this family, so the body that names
    /// its instantiation is `x::driver_internal`'s.
    pub(in crate::x) mod inst {
        /// `split_packed.cuh:74` — the host-window form, over a `[n_tokens,
        /// q_dim + 2 * kv_dim]` packed row.
        pub const SPLIT_QKV: &str = "::pie_cuda_driver::kernels::attn::device::split_qkv\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `split_packed.cuh:111` — the device-window form, over BASE
        /// pointers.
        pub const SPLIT_QKV_DEVWIN: &str = "::pie_cuda_driver::kernels::attn::device::split_qkv_devwin\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `attn/attention_flashinfer.cuh` — the per-head → per-request score fold.
pub mod attention_flashinfer {
    use crate::jit::{Ctx, Launch, Root};
    use crate::x::Abi;
    use kernels::Refusal;

    /// `attn/attention_flashinfer.cuh` — the root the fold compiles out of.
    pub static ROOT: Root = Root::new(
        "attn/attention_flashinfer",
        include_str!("../../csrc/src/attn/attention_flashinfer.cuh"),
        "attn/attention_flashinfer.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `attention_flashinfer.cuh:190` — the decode capture's head fold.
        pub const ATTN_SCORE_FOLD_HEADS: &str =
            "::pie_cuda_driver::kernels::attn::device::attn_score_fold_heads";
    }

    /// The fold's grid fanout: `attention_flashinfer.cu:828`'s literal `64u`.
    ///
    /// **Not an extent.** The kernel's inner loop is
    /// `for (int i = blockIdx.y; i < n; i += gridDim.y)` over the request's KV
    /// positions, so `gridDim.y` is an OCCUPANCY FANOUT: every value of it
    /// computes the same floats, and `1` computes them correctly in a
    /// sixty-fourth of the blocks. That is why it is a citation here and not a
    /// rule anywhere — a rule is a function of the fire's rectangle and `64`
    /// is not in the rectangle. The only other grid-stride literal in
    /// `csrc/src` is a *different* number
    /// ([`attention_score_post::PREFILL_FOLD_GRID_Y`]'s `32`), which is the
    /// clearest evidence available that neither is a rule.
    ///
    /// [`attention_score_post::PREFILL_FOLD_GRID_Y`]:
    ///     super::attention_score_post::PREFILL_FOLD_GRID_Y
    const FOLD_GRID_Y: u32 = 64;

    /// The fold's block width: `attention_flashinfer.cu:829`'s `256`.
    ///
    /// Load-bearing and not tuning: the kernel folds warp partials through
    /// `__shared__ float red[256 / 32]`, so a launch at another width would
    /// read reduction slots nothing wrote — a plausible score row rather than
    /// a fault.
    const FOLD_BLOCK: u32 = 256;

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
        // `attention_flashinfer.cu:817` — `if (num_requests <= 0) return;`
        if num_requests <= 0 {
            return Err(Refusal::Empty { what: "num_requests" });
        }
        // `attention_flashinfer.cu:828-829`, transcribed. `num_requests` is
        // `grid.x` because the kernel indexes the request by `blockIdx.x`; the
        // second axis is `FOLD_GRID_Y` and is not an extent.
        //
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::ATTN_SCORE_FOLD_HEADS,
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
    use crate::jit::{Ctx, Launch, Root};
    use crate::x::Abi;
    use kernels::Refusal;

    /// `attn/attention_score_post.cuh` — the root all three compile out of.
    ///
    /// ONE root for the three, and that is a cost, not a tidiness: a compile
    /// is per instantiation, so nothing is shared between them at run time —
    /// but the three are one file because they are one program, and a split
    /// would be three files that must agree about a score row's layout.
    pub static ROOT: Root = Root::new(
        "attn/attention_score_post",
        include_str!("../../csrc/src/attn/attention_score_post.cuh"),
        "attn/attention_score_post.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `attention_score_post.cuh:168` — the DECODE capture's
        /// divide-by-total.
        pub const ATTN_SCORE_NORMALIZE: &str =
            "::pie_cuda_driver::kernels::attn::device::attn_score_normalize";
        /// `:244` — the PREFILL capture's, which additionally strides the
        /// observation window.
        pub const ATTN_PREFILL_SCORE_NORMALIZE: &str =
            "::pie_cuda_driver::kernels::attn::device::attn_prefill_score_normalize";
        /// `:343` — the prefill fold, which collapses heads AND window rows.
        pub const ATTN_PREFILL_SCORE_FOLD: &str =
            "::pie_cuda_driver::kernels::attn::device::attn_prefill_score_fold";
    }

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
        if num_requests <= 0 {
            return Err(Refusal::Empty { what: "num_requests" });
        }
        if num_q_heads <= 0 {
            return Err(Refusal::Empty { what: "num_q_heads" });
        }
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::ATTN_SCORE_NORMALIZE,
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
        if num_requests <= 0 {
            return Err(Refusal::Empty { what: "num_requests" });
        }
        if num_q_heads <= 0 {
            return Err(Refusal::Empty { what: "num_q_heads" });
        }
        if window <= 0 {
            return Err(Refusal::Empty { what: "window" });
        }
        // SAFETY: as [`attn_score_normalize`]'s.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::ATTN_PREFILL_SCORE_NORMALIZE,
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
        if num_requests <= 0 {
            return Err(Refusal::Empty { what: "num_requests" });
        }
        // SAFETY: as [`attn_score_normalize`]'s, and `folded` is written
        // rather than read.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::ATTN_PREFILL_SCORE_FOLD,
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

    #[cfg(test)]
    mod tests {
        use super::inst;

        /// The three instantiations, as `attn/attention_score_post`'s three
        /// rows spelled them before the unit was deleted.
        #[test]
        fn the_instantiations_are_the_deleted_rows() {
            assert_eq!(
                inst::ATTN_SCORE_NORMALIZE,
                "::pie_cuda_driver::kernels::attn::device::attn_score_normalize",
            );
            assert_eq!(
                inst::ATTN_PREFILL_SCORE_NORMALIZE,
                "::pie_cuda_driver::kernels::attn::device::attn_prefill_score_normalize",
            );
            assert_eq!(
                inst::ATTN_PREFILL_SCORE_FOLD,
                "::pie_cuda_driver::kernels::attn::device::attn_prefill_score_fold",
            );
        }

        /// The decode fold is a DIFFERENT root, and that is what it costs.
        ///
        /// `cache::resolve` keys on the root text, so the capture pays one
        /// compile per instantiation either way — but the four kernels of one
        /// capture come out of two files, and a reader counting compiles
        /// should be able to see which.
        #[test]
        fn the_decode_fold_is_not_one_of_the_post_kernels() {
            assert_ne!(super::ROOT.name, super::super::attention_flashinfer::ROOT.name);
            assert_ne!(
                inst::ATTN_SCORE_NORMALIZE,
                super::super::attention_flashinfer::inst::ATTN_SCORE_FOLD_HEADS,
            );
        }
    }
}

/// `attn/pack_dense_mask.cuh` — the two custom-mask packers, both plain
pub mod pack_dense_mask {

    use crate::jit::Root;

    /// `attn/pack_dense_mask.cuh` — the root the two packers compile out of.
    ///
    /// No `inst` beside it: neither packer has a host program in this file.
    pub static ROOT: Root = Root::new(
        "attn/pack_dense_mask",
        include_str!("../../csrc/src/attn/pack_dense_mask.cuh"),
        "attn/pack_dense_mask.cuh",
    );
}

/// `attn/dsa_indexer.cuh` — glm5's sparse-attention index network, three
pub mod dsa_indexer {

    use crate::jit::Root;

    /// `attn/dsa_indexer.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "attn/dsa_indexer",
        include_str!("../../csrc/src/attn/dsa_indexer.cuh"),
        "attn/dsa_indexer.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub(super) mod inst {
        /// `dsa_indexer.cuh:106` — the LayerNorm-then-RoPE over the keys.
        pub const INDEX_KNORM_ROPE: &str = "::pie_cuda_driver::kernels::attn::device::index_knorm_rope\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dsa_indexer.cuh:151` — the same rotation over the queries.
        pub const INDEX_Q_ROPE: &str = "::pie_cuda_driver::kernels::attn::device::index_q_rope\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dsa_indexer.cuh:187` — the causal top-k mask over the scores.
        pub const INDEX_TOPK_MASK: &str = "::pie_cuda_driver::kernels::attn::device::index_topk_mask\
             <::pie_cuda_driver::kernels::device::bf16>";
    }

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
    use crate::jit::Root;

    /// `attn/page_compact.cuh` — the root both halves compile out of.
    pub static ROOT: Root = Root::new(
        "attn/page_compact",
        include_str!("../../csrc/src/attn/page_compact.cuh"),
        "attn/page_compact.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub(super) mod inst {
        /// `page_compact.cuh:212` — the FIRST launch: how many pages survive.
        pub const COUNT_KEPT: &str = "::pie_cuda_driver::kernels::attn::device::count_kept\
             <::pie_cuda_driver::kernels::device::i32(256)>";
        /// `page_compact.cuh:242` — the SECOND: scan and scatter, fused.
        pub const SCAN_AND_SCATTER: &str = "::pie_cuda_driver::kernels::attn::device::scan_and_scatter\
             <::pie_cuda_driver::kernels::device::i32(256)>";
    }

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
    if num_requests <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if scratch_counts.is_null() {
        return Err(Refusal::Absent { what: "the compaction scratch buffer" });
    }
    let launch = Launch::per_row(num_requests.unsigned_abs(), page_compact::K_BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as. The two launches are
    // ordered by the stream, and the second reads what the first wrote.
    unsafe {
        ctx.launch(
            &page_compact::ROOT,
            page_compact::inst::COUNT_KEPT,
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
            &page_compact::ROOT,
            page_compact::inst::SCAN_AND_SCATTER,
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

    use crate::jit::Root;

    /// `attn/attention_naive.cuh` — the root these routines compile a symbol
    /// out of.
    pub static ROOT: Root = Root::new(
        "attn/attention_naive",
        include_str!("../../csrc/src/attn/attention_naive.cuh"),
        "attn/attention_naive.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The reference attention itself — `attn::attention_naive_bf16` — has no
    /// routine in this file, so no constant here names it.
    pub(super) mod inst {
        /// `attention_naive.cuh:305` — MTP's shift, one block per TOKEN.
        pub const MTP_SHIFT_HIDDEN: &str = "::pie_cuda_driver::kernels::attn::device::mtp_shift_hidden\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The end-of-step refresh, one block per REQUEST.
        pub const MTP_UPDATE_PENDING_HIDDEN: &str = "::pie_cuda_driver::kernels::attn::device::mtp_update_pending_hidden\
             <::pie_cuda_driver::kernels::device::bf16>";
    }

    /// `attention_naive.cu:57` — `constexpr int BLOCK = device::BLOCK;`,
    pub const BLOCK: u32 = 256;
}

/// `attn::mtp_shift_hidden_bf16` — one block per TOKEN.
///
/// What the caller must guarantee, as `call()` states it: every pointer is a
/// device address live across the launch.
pub fn mtp_shift_hidden_bf16(
    ctx: &Ctx,
    target_hidden: *const bf16,
    pending_hidden: *const bf16,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    out: *mut bf16,
    total_tokens: i32,
    num_requests: i32,
    hidden_size: i32,
) -> Result<(), Refusal> {
    if total_tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if num_requests <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if hidden_size <= 0 {
        return Err(Refusal::Empty { what: "hidden width" });
    }
    if pending_hidden.is_null() {
        return Err(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &attention_naive::ROOT,
            attention_naive::inst::MTP_SHIFT_HIDDEN,
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
/// [`mtp_shift_hidden_bf16`]'s obligation.
pub fn mtp_update_pending_hidden_bf16(
    ctx: &Ctx,
    target_hidden: *const bf16,
    pending_hidden: *mut bf16,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    num_requests: i32,
    hidden_size: i32,
) -> Result<(), Refusal> {
    if num_requests <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if hidden_size <= 0 {
        return Err(Refusal::Empty { what: "hidden width" });
    }
    if pending_hidden.is_null() {
        return Err(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &attention_naive::ROOT,
            attention_naive::inst::MTP_UPDATE_PENDING_HIDDEN,
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

/// `mla_paged.cu:52` — `constexpr int BS = 256;`, the prepare block.
pub const MLA_PREPARE_BLOCK: i32 = 256;

/// `mla_paged.cu:105` — `write_mla`'s block, one per token row.
pub const MLA_WRITE_BLOCK: u32 = 256;

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

/// `mla_paged.cu:65` — the grid's second axis, less its KV lane.
#[must_use]
pub fn mla_q_blocks(heads: i32, heads_per_block: i32) -> i32 {
    if heads_per_block <= 0 {
        return 0;
    }
    heads.saturating_add(heads_per_block - 1) / heads_per_block
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
    if total_tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    let kv_lora = layer.kv_lora_rank;
    let rope = layer.qk_rope_head_dim;
    let stride = if kv_a_row_stride > 0 { kv_a_row_stride } else { kv_lora + rope };
    let per_block = mla_heads_per_block(rope);
    let blocks = mla_q_blocks(heads, per_block);

    let (low_dim, high_dim) = match yarn {
        Some(y) => crate::x::rope::ramp_bounds(
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
            &mla_paged::ROOT,
            mla_paged::inst::MLA_PREPARE,
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
    if total_tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &mla_paged::ROOT,
            mla_paged::inst::WRITE_MLA,
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

/// `dsv4_compress.cu:37` — `constexpr int ATTN_BLOCK = 128;`.
const DSV4_ATTN_BLOCK: u32 = 128;

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
    if n <= 0 {
        return Err(Refusal::Empty { what: "elements" });
    }
    if ratio <= 0 {
        return Err(Refusal::Narrow { what: "ratio", at: i64::from(ratio) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_compress::ROOT,
            dsv4_compress::inst::BOUNDARY_META_DECODE,
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
    if n <= 0 {
        return Err(Refusal::Empty { what: "elements" });
    }
    if ratio <= 0 {
        return Err(Refusal::Narrow { what: "ratio", at: i64::from(ratio) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_compress::ROOT,
            dsv4_compress::inst::BOUNDARY_META_PAGED,
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
    if total_tokens <= 0 {
        return Err(Refusal::Empty { what: "query tokens" });
    }
    if num_q_heads <= 0 {
        return Err(Refusal::Empty { what: "q heads" });
    }
    let smem = head_dim
        .max(0)
        .unsigned_abs()
        .saturating_add(DSV4_ATTN_BLOCK)
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_compress::ROOT,
            dsv4_compress::inst::COMPRESSED_ATTN_PAGED,
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
pub fn dsa_index_knorm_rope_bf16(
    ctx: &Ctx,
    idx_k: *mut bf16,
    k_norm_weight: *const bf16,
    k_norm_bias: *const bf16,
    positions: *const i32,
    tokens: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    eps: f32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsa_indexer::ROOT,
            dsa_indexer::inst::INDEX_KNORM_ROPE,
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
/// [`dsa_index_knorm_rope_bf16`]'s obligation.
pub fn dsa_index_q_rope_bf16(
    ctx: &Ctx,
    idx_q: *mut bf16,
    positions: *const i32,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsa_indexer::ROOT,
            dsa_indexer::inst::INDEX_Q_ROPE,
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
/// [`dsa_index_knorm_rope_bf16`]'s obligation.
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
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    let smem = tokens
        .unsigned_abs()
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsa_indexer::ROOT,
            dsa_indexer::inst::INDEX_TOPK_MASK,
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
    use crate::x::{ByValue, Layout};

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
pub mod mla_naive {

    use crate::jit::Root;

    /// `attn/attention_mla_naive.cuh` — the root the fallback pair compiles
    /// out of.
    ///
    pub static ROOT: Root = Root::new(
        "attn/attention_mla_naive",
        include_str!("../../csrc/src/attn/attention_mla_naive.cuh"),
        "attn/attention_mla_naive.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Neither is a template: both are plain `__global__`s, and the second
    /// sits a namespace deeper.
    pub mod inst {
        /// The scalar fallback.
        pub const NAIVE_PAGED: &str =
            "::pie_cuda_driver::kernels::attn::mla_naive::mla_naive_paged_kernel";
        /// The MMA form.
        pub const MMA_PAGED: &str =
            "::pie_cuda_driver::kernels::attn::mla_naive::mma_detail::mla_mma_paged_kernel";
    }
}

/// `attn/kimi_mla.cuh` — kimi_k3's two latent-attention preparation kernels.
pub mod kimi_mla {

    use crate::jit::Root;

    /// `attn/kimi_mla.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "attn/kimi_mla",
        include_str!("../../csrc/src/attn/kimi_mla.cuh"),
        "attn/kimi_mla.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The second carries a NON-TYPE second argument, `256`, which is the
    /// block width its `__global__` is templated over.
    pub(super) mod inst {
        /// `kimi_mla.cuh:67` — the fused `q_b` split.
        pub const SPLIT_Q_B: &str = "::pie_cuda_driver::kernels::attn::device::split_q_b\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `kimi_mla.cuh:101` — the `kv_a` split with the RMS norm fused in.
        pub const SPLIT_KV_A_NORM: &str = "::pie_cuda_driver::kernels::attn::device::split_kv_a_norm\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
    }
}

/// `attn/attention_naive_paged.cuh` — the reference paged attention.
pub mod attention_naive_paged {

    

    use crate::jit::Root;

    /// `attn/attention_naive_paged.cuh` — the root the reference pair
    /// compiles out of.
    pub static ROOT: Root = Root::new(
        "attn/attention_naive_paged",
        include_str!("../../csrc/src/attn/attention_naive_paged.cuh"),
        "attn/attention_naive_paged.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The decode half has no host program in this file, so no constant here
    /// names it.
    pub(super) mod inst {
        /// `attention_naive_paged.cuh:346` — the prefill, at `BLOCK = 128`.
        pub const NAIVE_PAGED_ATTN: &str = "::pie_cuda_driver::kernels::attn::device::naive_paged_attn\
             <::pie_cuda_driver::kernels::device::i32(128)>";
    }
}

/// `attn/mla_paged.cuh` — the MLA cache's append and its preparation pass.
pub mod mla_paged {

    use crate::jit::Root;

    /// `attn/mla_paged.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "attn/mla_paged",
        include_str!("../../csrc/src/attn/mla_paged.cuh"),
        "attn/mla_paged.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub(super) mod inst {
        /// `mla_paged.cuh:174` — the append, whose `__global__` takes no
        /// template parameter list.
        pub const WRITE_MLA: &str = "::pie_cuda_driver::kernels::attn::device::write_mla";
        /// `mla_paged.cuh:223` — the fused prepare, at `BLOCK_DIM = 256`.
        pub const MLA_PREPARE: &str = "::pie_cuda_driver::kernels::attn::device::mla_prepare\
             <::pie_cuda_driver::kernels::device::i32(256)>";
    }
}

/// The FlashInfer FA2 MLA host program — **compilable, and not yet fireable.**
pub mod mla_fa2 {
    use super::bf16;
    use super::mla_params::{MlaParams, UintFastdiv};
    use crate::jit::{Launch, Root};
    use crate::plan::MlaPlanInfo;

    /// `attn/attention_mla_fa2.cuh` — the root this arm's symbols come out of.
    ///
    /// `.upstream()` because its two `#include`s are
    /// `attn/flashinfer/attention/mla{,_params}.cuh`, which the library header
    /// set does not answer; [`OPTIONS`] because `grid.sync()` needs both of
    /// them. The text holds no `__global__` of its own — it exists to
    /// instantiate FlashInfer's under a traits pack this file names — so what
    /// a compile of it produces is [`SYMBOLS`]' six entries and [`SMEM_ECHO`]'s
    /// three numbers, and nothing else.
    pub static ROOT: Root = Root::new(
        "attn/attention_mla_fa2",
        include_str!("../../csrc/src/attn/attention_mla_fa2.cuh"),
        "attn/attention_mla_fa2.cuh",
    )
    .options(OPTIONS)
    .upstream();

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

    /// The widest arm this device's shared-memory budget admits.
    #[must_use]
    pub const fn arm_for(smem_limit_per_sm: u32) -> Option<Arm> {
        let mut i = 0;
        while i < ARMS.len() {
            if smem_limit_per_sm >= ARMS[i].smem {
                return Some(ARMS[i]);
            }
            i += 1;
        }
        None
    }

    /// The one NVRTC option this root needs, and the sixteen errors that say
    pub const OPTIONS: &[&str] = &[
        "--device-as-default-execution-space",
        // `grid.sync()` leaves `cudaCGGetIntrinsicHandle` extern; relocatable
        // device code is what lets `ptxas` emit it unresolved for the
        // `cuLink` step against `libcudadevrt.a` to close.
        "--relocatable-device-code=true",
    ];

    /// The six row symbols, indexed by `[arm][causal]`, parallel to [`ARMS`].
    pub const SYMBOLS: [[&str; 2]; 3] = [
        ["attn::mla_fa2_kv64_full", "attn::mla_fa2_kv64_causal"],
        ["attn::mla_fa2_kv32_full", "attn::mla_fa2_kv32_causal"],
        ["attn::mla_fa2_kv16_full", "attn::mla_fa2_kv16_causal"],
    ];

    /// The compiler's own `sizeof(KTraits::SharedStorage)` per arm, as name
    pub const SMEM_ECHO: [&str; 3] = [
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 64u>>",
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 32u>>",
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 1u, false, 16u>>",
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

    /// `int_buf + offset`, in ELEMENTS of `T`.
    unsafe fn offset_ptr<T>(base: *mut u8, offset: i64) -> *mut T {
        unsafe { base.cast::<T>().offset(offset as isize) }
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
}

/// The units `attn` compiles in fn-world.
pub mod qkv_fused {
    use super::bf16;
    use super::{Ctx, Launch, Refusal};
    use crate::jit::Root;
    use crate::x::Abi;

    /// `attn/qkv_fused.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "attn/qkv_fused",
        include_str!("../../csrc/src/attn/qkv_fused.cuh"),
        "attn/qkv_fused.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The decode forms carry TWO template arguments: the block width, and
    /// whether the kernel reads a precomputed cos/sin table. The warp form
    /// exists at three widths and nowhere else, which is what
    /// [`warp_instantiation`] answers `None` for.
    pub(super) mod inst {
        /// `qkv_fused.cuh:412` — the PACKED prefill epilogue.
        pub const PACKED: &str = "::pie_cuda_driver::kernels::attn::device::qkv_packed_qk_norm_rope_vnorm_write_kv\
             <::pie_cuda_driver::kernels::device::i32(256)>";
        /// `qkv_fused.cuh:115` — the BLOCK decode form, reading the table.
        pub const BLOCK_ROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv\
             <::pie_cuda_driver::kernels::device::i32(128), true>";
        /// The same, computing the rotation itself.
        pub const BLOCK_NOROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv\
             <::pie_cuda_driver::kernels::device::i32(128), false>";
        /// `qkv_fused.cuh:252` — the WARP decode form at `head_dim = 64`.
        pub const WARP_D64_ROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv_warp\
             <::pie_cuda_driver::kernels::device::i32(64), true>";
        /// The same, computing the rotation itself.
        pub const WARP_D64_NOROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv_warp\
             <::pie_cuda_driver::kernels::device::i32(64), false>";
        /// The warp form at `head_dim = 128`.
        pub const WARP_D128_ROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv_warp\
             <::pie_cuda_driver::kernels::device::i32(128), true>";
        /// The same, computing the rotation itself.
        pub const WARP_D128_NOROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv_warp\
             <::pie_cuda_driver::kernels::device::i32(128), false>";
        /// The warp form at `head_dim = 256`.
        pub const WARP_D256_ROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv_warp\
             <::pie_cuda_driver::kernels::device::i32(256), true>";
        /// The same, computing the rotation itself.
        pub const WARP_D256_NOROPE: &str = "::pie_cuda_driver::kernels::attn::device::qkv_decode_qk_norm_rope_write_kv_warp\
             <::pie_cuda_driver::kernels::device::i32(256), false>";
    }

    /// `BLOCK` for the packed form, and it IS the block width.
    pub const PACKED_BLOCK: u32 = 256;

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
        if num_rows <= 0 {
            return Err(Refusal::Empty { what: "rows" });
        }
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        if num_q_heads <= 0 {
            return Err(Refusal::Empty { what: "q heads" });
        }
        if num_kv_heads <= 0 {
            return Err(Refusal::Empty { what: "kv heads" });
        }
        if page_size <= 0 {
            return Err(Refusal::Empty { what: "page size" });
        }
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::PACKED,
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

    /// `qkv_fused.cu:51` — `constexpr int WARP_BLOCK = 256;`, and it is NOT
    pub const WARP_BLOCK: u32 = 256;

    /// `qkv_fused.cu:105` — `constexpr int BLOCK = 128;`, the DECODE block.
    pub const DECODE_BLOCK: u32 = 128;

    /// Warps per block: `WARP_BLOCK / 32`.
    const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

    /// The six warp instantiations, by `(head_dim, rope_table)`.
    ///
    /// `None` is the whole of the warp form's applicability test: it exists
    /// at three head widths and the block form covers every other.
    fn warp_instantiation(head_dim: i32, rope_table: bool) -> Option<&'static str> {
        Some(match (head_dim, rope_table) {
            (64, true) => inst::WARP_D64_ROPE,
            (64, false) => inst::WARP_D64_NOROPE,
            (128, true) => inst::WARP_D128_ROPE,
            (128, false) => inst::WARP_D128_NOROPE,
            (256, true) => inst::WARP_D256_ROPE,
            (256, false) => inst::WARP_D256_NOROPE,
            _ => return None,
        })
    }

    /// The block form's two arms.
    const fn block_instantiation(rope_table: bool) -> &'static str {
        if rope_table { inst::BLOCK_ROPE } else { inst::BLOCK_NOROPE }
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
        if num_requests <= 0 {
            return Err(Refusal::Empty { what: "requests" });
        }
        if q_out.is_null() {
            return Err(Refusal::Absent { what: "q_out" });
        }
        if num_q_heads <= 0 {
            return Err(Refusal::Empty { what: "q heads" });
        }
        if num_kv_heads <= 0 {
            return Err(Refusal::Empty { what: "kv heads" });
        }
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        if page_size <= 0 {
            return Err(Refusal::Empty { what: "page size" });
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
                    &ROOT,
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
                &ROOT,
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
    use crate::jit::Root;
    use crate::x::Abi;

    /// `attn/dsv4_compress.cuh` — the root these routines compile a symbol
    /// out of.
    pub static ROOT: Root = Root::new(
        "attn/dsv4_compress",
        include_str!("../../csrc/src/attn/dsv4_compress.cuh"),
        "attn/dsv4_compress.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Four of the header's eleven kernels have no host program in this file
    /// — the two pools, the APE add and the unpaged gather — so no constant
    /// here names them.
    pub(super) mod inst {
        /// `dsv4_compress.cuh:578` — the paged gather.
        pub const COMPRESS_GATHER_PAGED: &str = "::pie_cuda_driver::kernels::attn::device::dsv4_compress_gather_paged\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dsv4_compress.cuh:648` — the commit into the compressed cache.
        pub const STORE_COMP_ENTRIES: &str = "::pie_cuda_driver::kernels::attn::device::dsv4_store_comp_entries\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dsv4_compress.cuh:530` — which decode rows close a window.
        pub const BOUNDARY_META_DECODE: &str = "::pie_cuda_driver::kernels::attn::device::dsv4_boundary_meta_decode\
             <::pie_cuda_driver::kernels::device::i32>";
        /// `dsv4_compress.cuh:544` — the prefill form.
        pub const BOUNDARY_META_PAGED: &str = "::pie_cuda_driver::kernels::attn::device::dsv4_boundary_meta_paged\
             <::pie_cuda_driver::kernels::device::i32>";
        /// `dsv4_compress.cuh:666` — attention over the compressed cache.
        pub const COMPRESSED_ATTN_PAGED: &str =
            "::pie_cuda_driver::kernels::attn::device::compressed_attn_paged";
        /// `dsv4_compress.cuh:216` — the merge, by log-sum-exp.
        pub const COMBINE_ATTN_OUTPUTS: &str = "::pie_cuda_driver::kernels::attn::device::combine_attn_outputs\
             <::pie_cuda_driver::kernels::device::bf16>";
    }

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
        if num_entries <= 0 {
            return Err(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        if ratio <= 0 {
            return Err(Refusal::Empty { what: "ratio" });
        }
        if coff <= 0 {
            return Err(Refusal::Empty { what: "coff" });
        }
        if page_size <= 0 {
            return Err(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::COMPRESS_GATHER_PAGED,
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
        if num_entries <= 0 {
            return Err(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        if page_size <= 0 {
            return Err(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::STORE_COMP_ENTRIES,
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
    use crate::x::abi::MaybeConst;
    use crate::x::fp8_kind;

    use super::{Ctx, Launch};
    use crate::jit::Root;
    use crate::x::Abi;
    use crate::x::{KvDType, KvLayer, KvScheme};
    use kernels::Refusal;

    /// `attn/kv_paged.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "attn/kv_paged",
        include_str!("../../csrc/src/attn/kv_paged.cuh"),
        "attn/kv_paged.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The `#hnd`/`#nhd` pairs are one template each over a page layout:
    /// `true_type` is `[head, page, dim]` and `false_type` is `[page, head,
    /// dim]`. The position-addressed append has no host program in this file,
    /// so no constant here names it.
    pub mod inst {
        /// `kv_paged.cuh:367` — the cell copy, `[head, page, dim]`.
        pub const COPY_KV_CELLS_HND: &str = "::pie_cuda_driver::kernels::attn::device::copy_kv_cells\
             <::pie_cuda_driver::kernels::device::true_type::value>";
        /// The same, `[page, head, dim]`.
        pub const COPY_KV_CELLS_NHD: &str = "::pie_cuda_driver::kernels::attn::device::copy_kv_cells\
             <::pie_cuda_driver::kernels::device::false_type::value>";
        /// The windowed page view, over a device-resident window.
        pub const BUILD_WINDOW_PAGE_VIEW: &str =
            "::pie_cuda_driver::kernels::attn::device::build_window_page_view";
        /// The full split view.
        pub const BUILD_FULL_SPLIT_VIEW: &str =
            "::pie_cuda_driver::kernels::attn::device::build_full_split_view";
        /// `kv_paged.cuh:153` — the batched append, `[head, page, dim]`.
        pub const WRITE_KV_HND: &str = "::pie_cuda_driver::kernels::attn::device::write_kv\
             <::pie_cuda_driver::kernels::device::true_type::value>";
        /// The same, `[page, head, dim]`.
        pub const WRITE_KV_NHD: &str = "::pie_cuda_driver::kernels::attn::device::write_kv\
             <::pie_cuda_driver::kernels::device::false_type::value>";
        /// `kv_paged.cuh:279` — the explicit-slot append, `[head, page, dim]`.
        pub const WRITE_KV_EXPLICIT_HND: &str = "::pie_cuda_driver::kernels::attn::device::write_kv_explicit\
             <::pie_cuda_driver::kernels::device::true_type::value>";
        /// The same, `[page, head, dim]`.
        pub const WRITE_KV_EXPLICIT_NHD: &str = "::pie_cuda_driver::kernels::attn::device::write_kv_explicit\
             <::pie_cuda_driver::kernels::device::false_type::value>";
        /// `kv_paged.cuh:781` — the same under a device-resident window.
        pub const WRITE_KV_EXPLICIT_DEVWIN_HND: &str = "::pie_cuda_driver::kernels::attn::device::write_kv_explicit_devwin\
             <::pie_cuda_driver::kernels::device::true_type::value>";
        /// The same, `[page, head, dim]`.
        pub const WRITE_KV_EXPLICIT_DEVWIN_NHD: &str = "::pie_cuda_driver::kernels::attn::device::write_kv_explicit_devwin\
             <::pie_cuda_driver::kernels::device::false_type::value>";
        /// `kv_paged.cuh:390` — the fp8 append at one scale per tensor.
        pub const WRITE_KV_FP8_PER_TENSOR: &str =
            "::pie_cuda_driver::kernels::attn::device::write_kv_fp8_per_tensor";
        /// `kv_paged.cuh:425` — the int8 per-token-per-head append.
        pub const WRITE_KV_INT8_PER_TOKEN_HEAD: &str = "::pie_cuda_driver::kernels::attn::device::write_kv_per_token_head\
             <::pie_cuda_driver::kernels::device::false_type::value>";
        /// The fp8 arm of the same template.
        pub const WRITE_KV_FP8_PER_TOKEN_HEAD: &str = "::pie_cuda_driver::kernels::attn::device::write_kv_per_token_head\
             <::pie_cuda_driver::kernels::device::true_type::value>";
        /// `kv_paged.cuh:562` — the fp4 append, two values to the byte.
        pub const WRITE_KV_FP4_BLOCK: &str =
            "::pie_cuda_driver::kernels::attn::device::write_kv_fp4_block";
        /// `kv_paged.cuh:655` — the per-tensor fp8 dequantiser.
        pub const DEQUANT_FP8_PAGES_ACTIVE: &str =
            "::pie_cuda_driver::kernels::attn::device::dequant_fp8_pages_active";
        /// `kv_paged.cuh:678` — the per-token-per-head fp8 dequantiser.
        pub const DEQUANT_FP8_PER_TOKEN_HEAD: &str = "::pie_cuda_driver::kernels::attn::device::dequant_fp8_per_token_head_pages_active\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `kv_paged.cuh:708` — the same, for int8 pages.
        pub const DEQUANT_INT8_PER_TOKEN_HEAD: &str = "::pie_cuda_driver::kernels::attn::device::dequant_int8_per_token_head_pages_active\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `kv_paged.cuh:736` — the fp4 dequantiser.
        pub const DEQUANT_FP4: &str = "::pie_cuda_driver::kernels::attn::device::dequant_fp4_pages_active\
             <::pie_cuda_driver::kernels::device::bf16>";
    }

    /// `kv_paged.cu`'s `constexpr int BLOCK = 256`, which every launch in
    const BLOCK: u32 = 256;

    /// `::__nv_fp8_interpretation_t`'s two values, by the names the header
    const NV_E4M3: u32 = 0;
    const NV_E5M2: u32 = 1;

    /// The interpretation an fp8 page is written and read under.
    fn fp8_kind_of(storage_dtype: KvDType) -> fp8_kind {
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
        if b <= 0 {
            return Err(Refusal::Empty { what: "rows" });
        }

        let instantiation =
            if layer.hnd { inst::WRITE_KV_EXPLICIT_HND } else { inst::WRITE_KV_EXPLICIT_NHD };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
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
            let _ = crate::x::layout::envelope_merge_written(
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
        if n_max <= 0 {
            return Err(Refusal::Empty { what: "lanes" });
        }
        assert!(
            !layer.has_envelopes,
            "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
             windowed — use the host-window form"
        );

        let instantiation = if layer.hnd {
            inst::WRITE_KV_EXPLICIT_DEVWIN_HND
        } else {
            inst::WRITE_KV_EXPLICIT_DEVWIN_NHD
        };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
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
        if launch_tokens <= 0 {
            return Err(Refusal::Empty { what: "tokens after first_token" });
        }

        let instantiation = if layer.hnd { inst::WRITE_KV_HND } else { inst::WRITE_KV_NHD };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
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
            let _ = crate::x::layout::envelope_update_appended(
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
        if total_tokens <= 0 {
            return Err(Refusal::Empty { what: "tokens" });
        }
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
                    &ROOT,
                    inst::WRITE_KV_FP8_PER_TENSOR,
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
                    inst::WRITE_KV_FP8_PER_TOKEN_HEAD
                } else {
                    inst::WRITE_KV_INT8_PER_TOKEN_HEAD
                };
                // One `float` per warp, twice: the per-token-per-head scale
                // is a max over the head, reduced warp-wise for k and for v.
                let smem = 2 * (BLOCK / 32) * (core::mem::size_of::<f32>() as u32);
                unsafe {
                    ctx.launch(
                        &ROOT,
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
                        &ROOT,
                        inst::WRITE_KV_FP4_BLOCK,
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
        if num_pages_in_batch <= 0 {
            return Err(Refusal::Empty { what: "active pages" });
        }
        if !matches!(layer.scheme, KvScheme::Fp8PerTensor) {
            return Err(Refusal::Absent { what: "an fp8-per-tensor layer" });
        }

        let (logical_n, page_elems, launch) = active_geometry(layer, num_pages_in_batch);
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                &ROOT,
                inst::DEQUANT_FP8_PAGES_ACTIVE,
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
        if num_pages_in_batch <= 0 {
            return Err(Refusal::Empty { what: "active pages" });
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
                    &ROOT,
                    inst::DEQUANT_FP8_PER_TOKEN_HEAD,
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
                    &ROOT,
                    inst::DEQUANT_INT8_PER_TOKEN_HEAD,
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
                    &ROOT,
                    inst::DEQUANT_FP4,
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

/// `runtime/launch.rs:599` — `const PAD_BLOCK: u32 = 128;`.
const PAD_BLOCK: u32 = 128;

/// `runtime/launch.rs:608` — `const SINK_BLOCK_MIN: u32 = WARP;`.
const SINK_BLOCK_MIN: u32 = 32;

/// `runtime/launch.rs:610` — `const SINK_BLOCK_MAX: u32 = 128;`.
const SINK_BLOCK_MAX: u32 = 128;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// `LaunchRule::Rms`, as the expression it evaluates to.
#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
}

/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
#[must_use]
const fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch::grid([rows, heads, 1], [head_dim_block(head_dim), 1, 1])
}

/// `u32::clamp` is not `const`, and the rule's expression is transcribed
#[must_use]
const fn head_dim_block(head_dim: u32) -> u32 {
    if head_dim < SINK_BLOCK_MIN {
        SINK_BLOCK_MIN
    } else if head_dim > SINK_BLOCK_MAX {
        SINK_BLOCK_MAX
    } else {
        head_dim
    }
}

/// `LaunchRule::PerHead`, as the expression it evaluates to.
#[must_use]
const fn per_head(rows: u32, heads: u32) -> Launch {
    Launch::grid([heads, rows, 1], [PAD_BLOCK, 1, 1])
}

/// The merge's geometry — the grid of [`per_head_elementwise`] and a
#[must_use]
const fn combine_attn(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch::grid([rows, heads, 1], [combine_block(head_dim), 1, 1])
}

/// `[32, 256]`, transcribed rather than rearranged — `u32::clamp` is not
#[must_use]
const fn combine_block(head_dim: u32) -> u32 {
    if head_dim < COMBINE_BLOCK_MIN {
        COMBINE_BLOCK_MIN
    } else if head_dim > COMBINE_BLOCK_MAX {
        COMBINE_BLOCK_MAX
    } else {
        head_dim
    }
}

/// A warp. `dsv4_compress.cu:87`'s `(head_dim < 32) ? 32`.
const COMBINE_BLOCK_MIN: u32 = 32;

/// `dsv4_compress.cu:87`'s `(head_dim > 256) ? 256`. **Not
const COMBINE_BLOCK_MAX: u32 = 256;

/// `attn::lse_log2_to_ln` — rebase flashinfer's LSE from log2 to ln, in place.
///
/// What the caller must guarantee, as `call()` states it: `lse` must address
/// `n` live, writable `f32`s.
pub fn lse_log2_to_ln(ctx: &Ctx, lse: *mut f32, n: usize) -> Result<(), Refusal> {
    if n == 0 {
        return Err(Refusal::Empty { what: "lse elements" });
    }
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
            &attn_sink::ROOT,
            attn_sink::inst::LSE_LOG2_TO_LN,
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
pub fn attention_sink_rescale_bf16(
    ctx: &Ctx,
    o: *mut bf16,
    lse: *const f32,
    sinks: *const bf16,
    n: i32,
    num_q_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if num_q_heads <= 0 {
        return Err(Refusal::Empty { what: "num_q_heads" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &attn_sink::ROOT,
            attn_sink::inst::ATTN_SINK_RESCALE,
            per_head_elementwise(
                n.unsigned_abs(),
                num_q_heads.unsigned_abs(),
                head_dim.unsigned_abs(),
            ),
            &[o.arg(), lse.arg(), sinks.arg(), n.arg(), num_q_heads.arg(), head_dim.arg()],
        )
    }
}

/// `split_packed.cu:30` — `constexpr int BLOCK = 256;`.
pub const SPLIT_BLOCK: u32 = 256;

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
    if n_max <= 0 {
        return Err(Refusal::Empty { what: "lanes" });
    }
    let max_dim = if q_dim > kv_dim { q_dim } else { kv_dim };
    if max_dim <= 0 {
        return Err(Refusal::Empty { what: "output width" });
    }
    let xblocks = max_dim.unsigned_abs().div_ceil(SPLIT_BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &split_packed::ROOT,
            split_packed::inst::SPLIT_QKV_DEVWIN,
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

/// `attention_naive_paged.cuh:33` — `constexpr int BLOCK = 128`.
pub const PAGED_BLOCK: u32 = 128;

/// `attention_naive_paged.cuh:223` — `constexpr int kMaxHeadDim = 1024`.
pub const PAGED_MAX_HEAD_DIM: i32 = 1024;

/// `attn::attention_naive_paged` — the reference paged attention.
///
/// What the caller must guarantee, as `call()` states it: every pointer must
/// address live device memory of the extent the kernel reads or writes.
pub fn attention_naive_paged(
    ctx: &Ctx,
    layer: &crate::x::cx::KvLayer,
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
    if num_requests <= 0 {
        return Err(Refusal::Empty { what: "requests" });
    }
    if total_tokens <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if layer.head_dim <= 0 {
        return Err(Refusal::Empty { what: "the cache's head dim" });
    }
    if layer.head_dim > PAGED_MAX_HEAD_DIM {
        return Err(Refusal::Wide {
            what: "head_dim",
            at: i64::from(layer.head_dim),
            max: i64::from(PAGED_MAX_HEAD_DIM),
        });
    }
    let num_q_heads = q_width / layer.head_dim;
    if num_q_heads <= 0 {
        return Err(Refusal::Empty { what: "q heads" });
    }
    if layer.num_kv_heads <= 0 {
        return Err(Refusal::Empty { what: "kv heads" });
    }
    let smem = (layer.head_dim.unsigned_abs() + PAGED_BLOCK) * 4;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &attention_naive_paged::ROOT,
            attention_naive_paged::inst::NAIVE_PAGED_ATTN,
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
pub fn attn_res_blend_bf16(
    ctx: &Ctx,
    prefix: *const bf16,
    blocks: *const bf16,
    norm_weight: *const bf16,
    proj_weight: *const bf16,
    out: *mut bf16,
    t: i32,
    b: i32,
    h: i32,
    block_rows: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if t <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if b <= 0 {
        return Err(Refusal::Empty { what: "blocks" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &attn_res::ROOT,
            attn_res::inst::ATTN_RES_BLEND,
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
pub fn pad_head_dim_bf16(
    ctx: &Ctx,
    packed: *const bf16,
    padded: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Result<(), Refusal> {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Err(why);
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &head_dim_pad::ROOT,
            head_dim_pad::inst::PAD_HEAD_DIM,
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            &[packed.arg(), padded.arg(), num_heads.arg(), head_dim.arg(), head_dim_padded.arg()],
        )
    }
}

/// `attn::strip_head_dim_bf16` — the inverse of [`pad_head_dim_bf16`].
///
/// What the caller must guarantee, as `call()` states it: `padded` addresses
/// `num_tokens * num_heads * head_dim_padded` live bf16 elements and `packed`
/// addresses `num_tokens * num_heads * head_dim` writable ones.
pub fn strip_head_dim_bf16(
    ctx: &Ctx,
    padded: *const bf16,
    packed: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Result<(), Refusal> {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Err(why);
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &head_dim_pad::ROOT,
            head_dim_pad::inst::STRIP_HEAD_DIM,
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
    if n == 0 {
        return Err(Refusal::Empty { what: "logit elements" });
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
pub fn logit_softcap_bf16(ctx: &Ctx, x: *mut bf16, cap: f32, n: usize) -> Result<(), Refusal> {
    let launch = softcap_launch(cap, n)?;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &softcap::ROOT,
            softcap::inst::LOGIT_SOFTCAP_BF16,
            launch,
            &[x.arg(), cap.arg(), n.arg()],
        )
    }
}

/// `attn::logit_softcap_f16` — the same cap over an fp16 buffer.
///
/// [`logit_softcap_bf16`]'s obligation, with `f16` for `bf16`.
pub fn logit_softcap_f16(ctx: &Ctx, x: *mut f16, cap: f32, n: usize) -> Result<(), Refusal> {
    let launch = softcap_launch(cap, n)?;
    // SAFETY: as [`logit_softcap_bf16`]'s.
    unsafe {
        ctx.launch(
            &softcap::ROOT,
            softcap::inst::LOGIT_SOFTCAP_F16,
            launch,
            &[x.arg(), cap.arg(), n.arg()],
        )
    }
}

/// `attn::kimi_split_q_b_bf16` — split a fused query projection into its
pub fn kimi_split_q_b_bf16(
    ctx: &Ctx,
    q_b: *const bf16,
    q_nope: *mut bf16,
    q_pe: *mut bf16,
    tokens: i32,
    heads: i32,
    nope: i32,
    rope: i32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "num_heads" });
    }
    if nope <= 0 {
        return Err(Refusal::Empty { what: "qk_nope_head_dim" });
    }
    if rope <= 0 {
        return Err(Refusal::Empty { what: "qk_rope_head_dim" });
    }
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
            &kimi_mla::ROOT,
            kimi_mla::inst::SPLIT_Q_B,
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
pub fn kimi_split_kv_a_norm_bf16(
    ctx: &Ctx,
    kv_a: *const bf16,
    norm_weight: *const bf16,
    kv_c: *mut bf16,
    k_pe: *mut bf16,
    tokens: i32,
    kv_lora: i32,
    rope: i32,
    src_row_stride: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if kv_lora <= 0 {
        return Err(Refusal::Empty { what: "kv_lora_rank" });
    }
    if rope <= 0 {
        return Err(Refusal::Empty { what: "qk_rope_head_dim" });
    }
    if src_row_stride < kv_lora + rope {
        return Err(Refusal::Narrow { what: "src_row_stride", at: i64::from(src_row_stride) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &kimi_mla::ROOT,
            kimi_mla::inst::SPLIT_KV_A_NORM,
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
pub fn combine_attn_outputs_bf16(
    ctx: &Ctx,
    o1: *const bf16,
    lse1: *const f32,
    o2: *const bf16,
    lse2: *const f32,
    o_out: *mut bf16,
    lse_out: *mut f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if num_heads <= 0 {
        return Err(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_compress::ROOT,
            dsv4_compress::inst::COMBINE_ATTN_OUTPUTS,
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

/// `CUBLAS_COMPUTE_32F` — see `x::gemm::dense`'s `COMPUTE` for the tp > 1
#[cfg(feature = "_cuda")]
const ABSORB_COMPUTE: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;

/// `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, which the archive pinned on both calls.
#[cfg(feature = "_cuda")]
const ABSORB_ALGO: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

/// The archive's `check(status, api)` — `gemm.cpp:47`.
#[cfg(feature = "_cuda")]
fn absorb_check(status: cublasStatus_t, what: &str) {
    assert!(
        status == cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "cuBLAS error ({}): {what}",
        status as i32
    );
}

/// The absorb pair's shared call — `cublasGemmStridedBatchedEx` over the head
///
/// # Safety
///
/// The caller's, per entry point below.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn absorb(
    handle: *mut c_void,
    op_a: cublasOperation_t,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    stride_a: i64,
    ldb: i32,
    stride_b: i64,
    ldc: i32,
    stride_c: i64,
    heads: i32,
    what: &str,
) {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: the caller's obligation.
    let status = unsafe {
        cublasGemmStridedBatchedEx(
            handle.cast::<cublasContext>(),
            op_a,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            core::ptr::from_ref(&alpha).cast(),
            a,
            cudaDataType::CUDA_R_16BF,
            lda,
            stride_a,
            b,
            cudaDataType::CUDA_R_16BF,
            ldb,
            stride_b,
            core::ptr::from_ref(&beta).cast(),
            c,
            cudaDataType::CUDA_R_16BF,
            ldc,
            stride_c,
            heads,
            ABSORB_COMPUTE,
            ABSORB_ALGO,
        )
    };
    absorb_check(status, what);
}

/// `gemm::mla_absorb_q_to_latent_bf16` — `gemm.cpp:2419-2442`.
///
/// # Safety
///
/// `q_nope` must address `tokens * heads * qk_nope_dim` bf16 elements,
/// `kv_b_proj` the whole `heads * (qk_nope_dim + v_head_dim) * kv_lora_rank`
/// bank, and `q_latent` `tokens * heads * kv_lora_rank` writable elements.
/// `handle` must be a live `cublasHandle_t` with this fire's stream bound.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_absorb_q_to_latent_bf16(
    handle: *mut c_void,
    q_nope: *const c_void,
    kv_b_proj: *const c_void,
    q_latent: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        absorb(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            kv_b_proj,
            q_nope,
            q_latent,
            kv_lora_rank,
            tokens,
            qk_nope_dim,
            kv_lora_rank,
            i64::from(qk_nope_dim + v_head_dim) * i64::from(kv_lora_rank),
            heads * qk_nope_dim,
            i64::from(qk_nope_dim),
            heads * kv_lora_rank,
            i64::from(kv_lora_rank),
            heads,
            "mla_absorb_q_to_latent_bf16",
        );
    }
    Ok(())
}

/// `gemm::mla_absorb_latent_to_v_bf16` — `gemm.cpp:2444-2468`.
///
/// # Safety
///
/// As [`mla_absorb_q_to_latent_bf16`], with `attn_latent` in place of
/// `q_nope` and `attn_v` (`tokens * heads * v_head_dim`) as the output.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_absorb_latent_to_v_bf16(
    handle: *mut c_void,
    attn_latent: *const c_void,
    kv_b_proj: *const c_void,
    attn_v: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Result<(), Refusal> {
    if tokens <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    // SAFETY: the offset lands inside the same bank the caller guaranteed —
    let wv = unsafe {
        kv_b_proj
            .cast::<u8>()
            .add(2 * (qk_nope_dim as usize) * (kv_lora_rank as usize))
            .cast::<c_void>()
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        absorb(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            wv,
            attn_latent,
            attn_v,
            v_head_dim,
            tokens,
            kv_lora_rank,
            kv_lora_rank,
            i64::from(qk_nope_dim + v_head_dim) * i64::from(kv_lora_rank),
            heads * kv_lora_rank,
            i64::from(kv_lora_rank),
            heads * v_head_dim,
            i64::from(v_head_dim),
            heads,
            "mla_absorb_latent_to_v_bf16",
        );
    }
    Ok(())
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated beside one is what no signature carries:
/// whether a statement consumes its whole operand, and which operands must be
/// given the same address.
///
/// **Ten of this file's host programs are absent, and ONE reason covers all
/// ten**: `write_kv_to_pages` and its two arms, the two dequantisers, the two
/// explicit-slot appends and `attention_naive_paged` take a `&KvLayer`, and
/// `mla_prepare_bf16` and `write_mla_to_pages` take an `MlaLayer`. Those are
/// host aggregates rather than kernel arguments — a layer view is five to
/// eighteen operands the fire resolves together — and `Arg` is implemented for
/// no such type, so a `routine!` naming one does not compile. All ten are
/// converted and fire from their `bind!` arms; how a trace states a layer view
/// is a separate question from what this table can hold.
///
/// **The four score-capture launches are absent for a different reason, and it
/// is not a gap.** [`attention_score_post`]'s three and
/// [`attention_flashinfer::attn_score_fold_heads`] are fired by
/// `driver-cuda`'s `fire::attn_score`, at the point on the fire's stream where
/// the C++ capture dispatch used to issue them, and no trace statement names
/// one: `dsl::cuda::attn_score_fold_heads` states the CONTRACT symbol
/// `attn::attn_score_fold_heads`, which is a different symbol from the device
/// one this file's `_dev` suffix marks, and the other three were never
/// statements at all. A `routine!` for any of them would put a row in
/// `crate::sigs()` that nothing could lower to. `x::driver_internal`'s header
/// is the general form of this.
pub static ROUTINES: &[Routine] = &[
    routine!(lse_log2_to_ln, in_place = &[(0, 0)]),
    routine!(attention_sink_rescale_bf16, in_place = &[(0, 0)]),
    routine!(attn_res_blend_bf16),
    routine!(pad_head_dim_bf16),
    routine!(strip_head_dim_bf16),
    routine!(logit_softcap_bf16, in_place = &[(0, 0)]),
    routine!(logit_softcap_f16, in_place = &[(0, 0)]),
    routine!(kimi_split_q_b_bf16),
    routine!(kimi_split_kv_a_norm_bf16),
    routine!(combine_attn_outputs_bf16),
    routine!(split_qkv_bf16_devwin),
    routine!(compact_page_csr, whole),
    routine!(mtp_shift_hidden_bf16, whole),
    routine!(mtp_update_pending_hidden_bf16, whole),
    routine!(dsa_index_knorm_rope_bf16, whole),
    routine!(dsa_index_q_rope_bf16, whole),
    routine!(dsa_index_topk_mask, whole),
    routine!(dsv4_boundary_meta_decode, whole),
    routine!(dsv4_boundary_meta_paged, whole),
    routine!(attention_compressed_paged_bf16, whole),
    routine!(dsv4_compress_gather_paged_bf16),
    routine!(dsv4_store_comp_entries_bf16, whole),
    routine!(qkv_packed_qk_norm_rope_vnorm_write_kv_bf16),
    routine!(qkv_decode_qk_norm_rope_write_kv_bf16),
    routine!(qkv_decode_fused_dispatch),
];

/// `attn`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "attn", routines: ROUTINES };

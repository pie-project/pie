//! The CUDA attention family: FlashInfer's FA2 lattice and scheduler, XQA's
//! decode kernel, MLA's prefill and naive pair, the paged-KV writers, the DSV4
//! compression path and the fused QKV projections.
//!
//! Parameters are typed by what a binder can answer for. `In<N, _>`/`Out<N, _>`
//! promise a region's `rows` and `width`, so a launcher holding one divides for
//! itself rather than taking the quotient as a parameter; `Unbound<T>` is the
//! opposite — `T` plus a name for the absence of a source, promising no extent
//! and consuming neither positional counter. The `#[source(..)]` marks record
//! which driver-side query each parameter is read from, and the block below the
//! imports says where they come from and what an unmarked parameter means.

#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;

use crate::jit::abi::{Inst, bf16, f16};
use crate::jit::{Abi, Ctx, Family, Launch, Routine};
use crate::rope::Yarn;
use crate::{driver_bound, routine};
use kernels::Refusal;
use kernels::keys;
use kernels::routine::{Bank, Env, In, Out, Param, Unbound};

use attention_flashinfer::attn_score_fold_heads;
use crate::driver_internal::split_qkv_bf16;
use dsv4_compress::{dsv4_compress_gather_paged_bf16, dsv4_store_comp_entries_bf16};
use kv_paged::{
    dequant_kv_cache_layer_to_bf16_active, write_kv_explicit_bf16, write_kv_explicit_bf16_devwin,
    write_kv_to_pages, write_kv_to_pages_bf16, write_kv_to_pages_quantised,
};
use qkv_fused::{
    qkv_decode_fused_dispatch, qkv_decode_qk_norm_rope_write_kv_bf16,
    qkv_packed_qk_norm_rope_vnorm_write_kv_bf16,
};

// ── WHERE THE `#[source(..)]` MARKS COME FROM ──────────────────────────
//
// Every mark below is read off the arm that binds that symbol in
// `driver-cuda/src/bind/arms/attn.rs`. Where a launcher has no arm, the
// evidence is its row's `unbound:` sentence, which names the query each
// operand would come off.
//
// A parameter that carries no mark is one of two cases, and the sites below
// cite them as (a) and (b):
//
//   (a) no `Source` variant names the fact at all — `cx.q_out()` and the
//       `dsv4` compression `ratio`. THE LIST WAS SIX LONGER: `peel_window`,
//       `first_token`, `final_logit_softcap` and `w_page_d`/`w_off_d` all
//       have keys now, so the case they illustrated moved out from under
//       them. What is left is a RESULT and a ratio, not facts nobody named;
//
//       THE ATTENTION WORKSPACE'S FOUR WERE ON THAT LINE AND ARE NOT ANY
//       MORE, and no launcher in THIS file ever took them — `attn/xqa.rs`
//       and `attn/fa2/` did. `keys::AttnWorkspaceFloat`, `...FloatBytes`,
//       `...Int` and `...IntBytes` name what `Cx::attn_workspace` had been
//       answering all along, which makes them the clearest case of the
//       shape this list keeps mis-sorting: an UNNAMED answer reads exactly
//       like an unreachable one from the parameter's end.
//
//       THREE NAMES LEFT THIS LIST IN KILIMANJARO III STAGE 3 AND ONE LEFT
//       EARLIER. `plan()`'s `qo_indptr` and `row_valid` are
//       `keys::QoIndptr` and `keys::RowValid` (`keys.rs` §1's last two),
//       `kv_last_page_lens` is `keys::KvLastPageLens`, and `cx.sm_scale()`
//       is `keys::SmScale` — all four answered by `operand()`, so every
//       parameter in this file that reads one now SAYS which one. The list
//       above is what is left, and it is `Cx` queries with no fact behind
//       them rather than facts with no word: §1.1's *"`k_pages` is one line
//       away from `page_size`"* has no remaining instance here;
//   (b) a variant names it and `bind/table.rs`'s `operand()` cannot answer
//       it — `KvLayer`'s aggregate fields and the KV page pointers. Statement
//       params are NOT in this bucket any more: `Param<N, T>` and
//       `ParamF32<N>` state their own slots, and `operand()` answers both.
//       (b) is the worse row: the coverage test asserts the first unresolved
//       parameter is `None | Aux | Param | ParamF32`, so a true mark
//       outside that set turns a passing test into a failing one.
//
// TRAP, AND IT IS THE ONE TO READ TWICE: `Source::Named(keys::Rows)` is
// `f.rows.count`, and a region's height is filled from the same field. It is
// NOT `cx.rows().total`. The two differ on exactly the peeled fires the
// `_devwin` launchers exist for, which is why `write_kv_explicit_bf16_devwin`
// and `split_qkv_bf16_devwin` keep their `n_max` unmarked and unwrapped.
//
// `Unbound<T>` is a REFUSAL and not a mark: it is `T` plus a name for the
// absence of a source, promises no extent, and consumes neither positional
// counter. `In<N, _>`/`Out<N, _>` do promise extents — a region carries
// `rows` and `width`, so a launcher holding one divides for itself rather
// than taking the quotient as a parameter.
//
// A width that arrives through a region is `unwrap_or(0)`, where the deleted
// `#[source(OutWidth(n))]` refused a zero. Any launcher that needs the zero
// caught builds a VIEW — `x.all("out_width(0)")?`, which refuses in the word
// the launcher passes it and hands back a `Region` whose `width` cannot be
// zero and whose `stride` states the packing the body used to assume
// (`kernels/src/routine.rs`). Two launchers deliberately build none; see
// `head_dim_refusal` and the note where `width_of` used to stand.
//
// ── WHICH ROWS CARRY A COLUMN ──────────────────────────────────────────
//
// `routine!` rows do. `driver_bound!` rows do not: their bodies answer
// `Refusal::Absent { what: "a statement-bound body" }` for every argument
// list, so a column could never be honoured by the thing it feeds. Those six
// launchers still carry their marks, because the mark is where the reading is
// written down.
//
// Two rows cannot cross however complete their column gets, because a derived
// arm is bindings and nothing else: `write_kv_to_pages_quantised`'s arm
// refuses outright on a non-zero `cx.first_token()`, and
// `dequant_kv_cache_layer_to_bf16_active`'s returns `Ok(())` without
// launching when the layer is already bf16.

/// The FlashInfer FA2 lattice, its host arithmetic and its param structs.
pub mod fa2;
/// FlashAttention-4 forward: the host launcher for `attn/fa4.cuh`.
pub mod fa4;
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

    /// The [`KvScheme`] this byte spells, or `None` for a byte no scheme mints.
    ///
    /// # Why the inverse exists, and what it is NOT for
    ///
    /// `.wiki/kilimanjaro.md` §5 D1 — *"a routine takes fields, never a
    /// struct"* — makes the quantisation scheme an ARGUMENT of
    /// [`kv_paged::write_kv_to_pages_quantised`] rather than a field the
    /// routine reads off a `&KvLayer`. A `routine!` derives its table row
    /// from the signature, so every parameter has to be a
    /// [`kernels::routine::Arg`], and [`KvScheme`] is a host enum that no
    /// `Arg` impl names — this newtype is the spelling that does.
    ///
    /// The routine then has to BRANCH on it, because the four schemes are
    /// four kernels. That is what this answers, and answering it as an
    /// `Option<KvScheme>` rather than as a chain of `==` against
    /// [`Self::of`] is deliberate: the `match` at the branch stays
    /// exhaustive, so a fifth scheme is a compile error at the routine
    /// rather than a silent fall into the `Native` refusal.
    #[must_use]
    pub const fn scheme(self) -> Option<crate::attn::KvScheme> {
        use crate::attn::KvScheme;

        match self.0 {
            0 => Some(KvScheme::Native),
            1 => Some(KvScheme::Fp8PerTensor),
            2 => Some(KvScheme::Int8PerTokenHead),
            3 => Some(KvScheme::Fp8PerTokenHead),
            4 => Some(KvScheme::Fp4Block),
            _ => None,
        }
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

/// The byte a `keys::KvSchemeByte` / `keys::KvStorageDtype` `i32` spells.
///
/// `u8::MAX` for anything out of range, so a wrong number lands on the
/// `None` refusal rather than aliasing onto a scheme `& 0xff` happens to hit.
#[must_use]
const fn scheme_byte(n: i32) -> u8 {
    if n < 0 || n > u8::MAX as i32 { u8::MAX } else { n as u8 }
}

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
    use kernels::keys;
    use kernels::routine::{Env, Unbound};
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
    #[kernels_macros::routine]
    pub fn attn_score_fold_heads(
        ctx: &Ctx,
        scores: *const f32,
        score_indptr: Unbound<*const i32>,
        kv_page_indptr: Env<keys::KvPageIndptr>,
        kv_last_page_lens: Unbound<*const u32>,
        page_size: Env<keys::KvPageSize>,
        num_requests: Env<keys::RequestCount>,
        num_q_heads: Env<keys::NumQHeads>,
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
                    score_indptr.ptr.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.ptr.arg(),
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
    use kernels::routine::Env;
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
        qo_indptr: Env<*const u32>,
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
        qo_indptr: Env<*const u32>,
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
///
/// The row that declines this symbol (`arms/attn.rs:582`) says *"six of
/// eleven ARE answered: `keep` is `arg_in(0)`, the three CSR inputs and
/// `num_requests` come off `plan()`"*. Only two of the six are marks a
/// `Source` can carry; the other four are the reason the row is declined.
#[kernels_macros::routine]
pub fn compact_page_csr(
    ctx: &Ctx,
    // WHICH IS A DESCRIPTION OF CONFUSION AND NOT A FINDING. Chased to the
    // end, nothing was stopping them, and here is the whole check:
    page_indices_in: Env<keys::KvPageIndices>,
    page_indptr_in: Env<keys::KvPageIndptr>,
    last_page_lens_in: Unbound<*const u32>,
    keep: In<0, u8>,
    scratch_counts: Unbound<*mut u32>,
    keep_stride: u32,
    num_requests: Env<keys::RequestCount>,
    page_indices_out: Unbound<*mut u32>,
    page_indptr_out: Unbound<*mut u32>,
    last_page_lens_out: Unbound<*mut u32>,
) -> Result<(), Refusal> {
    if scratch_counts.ptr.is_null() {
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
                keep.ptr.arg(),
                keep_stride.arg(),
                num_requests.arg(),
                scratch_counts.ptr.arg(),
            ],
        )?;
        ctx.launch(
            "attn/page_compact.cuh",
            "::pie::attn::scan_and_scatter<::pie::i32(256)>",
            launch,
            &[
                page_indices_in.arg(),
                page_indptr_in.arg(),
                last_page_lens_in.ptr.arg(),
                keep.ptr.arg(),
                scratch_counts.ptr.cast_const().arg(),
                keep_stride.arg(),
                num_requests.arg(),
                page_indptr_out.ptr.arg(),
                last_page_lens_out.ptr.arg(),
                page_indices_out.ptr.arg(),
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
///
/// The row (`arms/attn.rs:599`) declines over ONE operand and lists the rest:
/// *"`target_hidden` and `pending_hidden` are `arg_in(0)` and `arg_in(1)` --
/// the statement hands the pending slab over as an INPUT [...] `out` is
/// `arg_out(0)`, `qo_indptr` and `num_requests` come off `plan()`,
/// `total_tokens` is `rows()` and `hidden_size` is `out_width(0)`"*.
#[kernels_macros::routine]
pub fn mtp_shift_hidden<T>(
    ctx: &Ctx,
    target_hidden: In<0, T>,
    pending_hidden: In<1, T>,
    qo_indptr: Env<keys::QoIndptr>,
    slot_ids: Unbound<*const i32>,
    out: Out<0, T>,
    num_requests: Env<keys::RequestCount>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    if pending_hidden.ptr.is_null() {
        return Err(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // THE WIDTH AND NOT THE PITCH, though `attention_naive.cuh:314-317`
    // strides all three buffers by this one number: `i < hidden_size` at
    // `:318` is the copy's own bound, so the extent is the reading `out`
    // states and the pitch is what a packed row makes of it.
    let dst = out.all("out_width(0)")?;
    let hidden_size = dst.width;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attention_naive.cuh",
            &format!("::pie::attn::mtp_shift_hidden<{}>", T::CPP),
            Launch::per_row(dst.rows.unsigned_abs(), attention_naive::BLOCK),
            &[
                target_hidden.ptr.arg(),
                pending_hidden.ptr.arg(),
                qo_indptr.arg(),
                slot_ids.ptr.arg(),
                out.ptr.arg(),
                num_requests.arg(),
                hidden_size.arg(),
            ],
        )
    }
}

/// `attn::mtp_update_pending_hidden_bf16` — one block per REQUEST.
///
/// [`mtp_shift_hidden`]'s obligation.
#[kernels_macros::routine]
pub fn mtp_update_pending_hidden<T>(
    ctx: &Ctx,
    target_hidden: In<0, T>,
    pending_hidden: Unbound<*mut T>,
    // As its twin's, and it is a NAME now: `keys::QoIndptr` = "plan.qo_indptr",
    // which `operand()` answers off `cx.plan()`. Reason (a) had this row for
    // two stages and lost it.
    qo_indptr: Env<keys::QoIndptr>,
    slot_ids: Unbound<*const i32>,
    num_requests: Env<keys::RequestCount>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    if pending_hidden.ptr.is_null() {
        return Err(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // As its twin's, off the input this form reads instead of the output it
    // does not take.
    let src = target_hidden.all("in_width(0)")?;
    let hidden_size = src.width;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attention_naive.cuh",
            &format!("::pie::attn::mtp_update_pending_hidden<{}>", T::CPP),
            Launch::per_row(num_requests.unsigned_abs(), attention_naive::BLOCK),
            &[
                target_hidden.ptr.arg(),
                pending_hidden.ptr.arg(),
                qo_indptr.arg(),
                slot_ids.ptr.arg(),
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
#[kernels_macros::routine]
pub fn mla_prepare_bf16(
    ctx: &Ctx,
    layer: MlaLayer,
    kv_a: In<0, bf16>,
    #[source(Weight(0))] kv_a_norm_weight: *const bf16,
    q_b: In<1, bf16>,
    kv_c: Out<0, bf16>,
    k_pe: Out<1, bf16>,
    q_nope: Out<2, bf16>,
    q_pe: Out<3, bf16>,
    positions: Env<keys::Positions>,
    // `keys::QoIndptr`. `row_valid` below keeps its own reason, which never
    // depended on this one: it arrives `Unbound` because the MLA rows are
    // blocked by `MlaLayer` whole, not because the fact wants a word.
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Unbound<*const u32>,
    row_valid: Unbound<*const u8>,
    num_requests: Env<keys::RequestCount>,
    heads: i32,
    qk_nope_head_dim: i32,
    eps: Env<keys::RmsEps>,
    // `Source::Named(<keys::RopeTheta as keys::Fact>::KEY)`, NOT `Source::Named(<keys::Theta as keys::Fact>::KEY)`, and the distinction is the
    // whole reason this parameter has a comment.
    theta: Env<keys::RopeTheta>,
    interleaved: bool,
    kv_a_row_stride: i32,
    // `cx.yarn()`, the YaRN scaling block, which `operand()` refuses as part
    // of the documented boundary: reason (b).
    yarn: Unbound<Option<Yarn>>,
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

    let (low_dim, high_dim) = match yarn.ptr {
        Some(y) => crate::rope::ramp_bounds(
            rope,
            **theta,
            y.beta_fast,
            y.beta_slow,
            y.original_max_position,
        ),
        None => (0.0, 0.0),
    };
    let yarn_factor = yarn.ptr.map_or(-1.0_f32, |y| y.factor);
    let yarn_mscale = yarn.ptr.map_or(1.0_f32, |y| y.attention_factor);

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/mla_paged.cuh",
            "::pie::attn::mla_prepare<::pie::i32(256)>",
            Launch::grid(
                [kv_c.rows.unsigned_abs(), blocks.saturating_add(1).max(1).unsigned_abs(), 1],
                [MLA_PREPARE_BLOCK.unsigned_abs(), 1, 1],
            ),
            &[
                kv_a.ptr.arg(),
                kv_a_norm_weight.arg(),
                q_b.ptr.arg(),
                kv_c.ptr.arg(),
                k_pe.ptr.arg(),
                q_nope.ptr.arg(),
                q_pe.ptr.arg(),
                layer.ckv_pages.cast::<bf16>().arg(),
                layer.kpe_pages.cast::<bf16>().arg(),
                positions.arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.ptr.arg(),
                row_valid.ptr.arg(),
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
#[kernels_macros::routine]
pub fn write_mla_to_pages(
    ctx: &Ctx,
    layer: MlaLayer,
    ckv_curr: In<0, bf16>,
    kpe_curr: In<1, bf16>,
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Unbound<*const u32>,
    row_valid: Unbound<*const u8>,
    num_requests: Env<keys::RequestCount>,
) -> Result<(), Refusal> {
    /// `mla_paged.cu:105` — `write_mla`'s block, one per token row.
    pub const MLA_WRITE_BLOCK: u32 = 256;

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/mla_paged.cuh",
            "::pie::attn::write_mla",
            Launch::per_row(ckv_curr.rows.unsigned_abs(), MLA_WRITE_BLOCK),
            &[
                ckv_curr.ptr.arg(),
                kpe_curr.ptr.arg(),
                layer.ckv_pages.cast::<bf16>().arg(),
                layer.kpe_pages.cast::<bf16>().arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.ptr.arg(),
                row_valid.ptr.arg(),
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
///
/// The row (`arms/attn.rs:671`) declines over one integer and lists the rest:
/// *"`positions` is `arg_in(0)`, the three outputs are `arg_out(0..2)`,
/// `row_valid` and `requests` come off `plan()`, and `n` is `rows()`"*. Every
/// one of those is a mark below except `row_valid`, which is a `plan()` array
/// no variant names.
#[kernels_macros::routine]
pub fn dsv4_boundary_meta_decode(
    ctx: &Ctx,
    // # NOT `Env<keys::Positions>`, AND THIS IS THE ONE `positions` IN THE
    // # TREE A NAME SWEEP WOULD GET WRONG WITHOUT REFUSING
    positions: In<0, i32>,
    out_pos: Out<0, i32>,
    out_req: Out<1, i32>,
    out_rope: Out<2, i32>,
    ratio: i32,
    // `plan.row_valid` = `keys::RowValid`. The row's own sentence
    // (`arms/attn.rs`) called this *"a `plan()` array no variant names"*;
    // `keys.rs` §1 names it now, so the sentence is wrong and the type is
    // right.
    row_valid: Env<keys::RowValid>,
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
            Launch::flat(out_pos.rows.unsigned_abs(), DSV4_META_BLOCK),
            &[
                positions.ptr.arg(),
                out_pos.ptr.arg(),
                out_req.ptr.arg(),
                out_rope.ptr.arg(),
                out_pos.rows.arg(),
                ratio.arg(),
                row_valid.arg(),
            ],
        )
    }
}

/// `attn::dsv4_boundary_meta_paged` — the prefill form of
///
/// [`dsv4_boundary_meta_decode`]'s obligation.
#[kernels_macros::routine]
pub fn dsv4_boundary_meta_paged(
    ctx: &Ctx,
    positions: In<0, i32>,
    qo_indptr: Env<keys::QoIndptr>,
    out_pos: Out<0, i32>,
    out_req: Out<1, i32>,
    out_rope: Out<2, i32>,
    num_requests: Env<keys::RequestCount>,
    // Its twin's: a number nothing in the driver holds. Reason (a).
    ratio: i32,
    // `plan.row_valid`, as on the decode twin above, and keyed with it.
    row_valid: Env<keys::RowValid>,
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
            Launch::flat(out_pos.rows.unsigned_abs(), DSV4_META_BLOCK),
            &[
                positions.ptr.arg(),
                qo_indptr.arg(),
                out_pos.ptr.arg(),
                out_req.ptr.arg(),
                out_rope.ptr.arg(),
                out_pos.rows.arg(),
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
///
/// The row (`arms/attn.rs:698`) says *"`q`, `o`, `lse_out`, `positions`, the
/// two page arrays, `total_tokens`, `num_q_heads`, `head_dim` and `page_size`
/// are all answered today"*, and for the last two that claim wanted checking,
/// because this kernel attends over a cache no pool allocates. It holds, and
/// the device source is why: `compressed_attn_paged` addresses
/// `comp_kv_pages` through `paged_slot(kv_page_indices, kv_page_indptr, req,
/// pos, page_size)` (`dsv4_compress.cuh:700`, `:465`) -- the ORDINARY plan's
/// page tables -- so `page_size` is the paged layer's granularity and not a
/// compressed cache's own, which is exactly `Source::Named(<keys::KvPageSize as keys::Fact>::KEY)`. And
/// `head_dim` strides the QUERY rows, `q + (qi * num_q_heads + q_head) *
/// head_dim` (`:678-679`), so it is the fire's head dim and not the KV view's,
/// which is exactly `Source::Named(<keys::HeadDim as keys::Fact>::KEY)`. Both are marked on that evidence
/// rather than on the row's word.
#[kernels_macros::routine]
pub fn attention_compressed_paged_bf16(
    ctx: &Ctx,
    // `In(0)`: the statement's `inputs: [q]`
    // (`model-dsl/src/cuda/deepseek_v4.rs:301`) has one entry.
    q: In<0, bf16>,
    comp_kv_pages: Env<*const bf16>,
    // `Out(0)`, the first of the statement's two `outs`.
    o: Out<0, bf16>,
    // The log-sum-exp side channel, written only when a later merge wants it.
    //
    // D2 APPLIED AND DECLINED (`.wiki/kilimanjaro3.md` §3.8). The absence is
    // read TWICE INSIDE THE KERNEL and never on this side:
    // `dsv4_compress.cuh:692` writes `neg_inf()` on the empty-window return
    // and `:750` writes the real log-sum-exp on the normal one, both guarded
    // `if (lse_out != nullptr && tid == 0)`. `lse_out.ptr` is threaded into
    // the argument list below beside `o.ptr` with no test between them, the
    // instantiation `::pie::attn::compressed_attn_paged` is not templated on
    // it, and `Launch::grid` reads `o.rows`, not this pointer. A split would
    // produce two bodies that differ in nothing, which is one function under
    // two names -- the rule asks for two FUNCTIONS.
    //
    // # AND THE `Or` STILL WENT, WHICH IS A DIFFERENT QUESTION FROM D2's
    //
    // Kilimanjaro III Stage 6 deletes `Or<T>` from operand position, and the
    // question it asks is not *"are these two functions"* but *"does anything
    // READ the `Provenance::Either` this plants"*. `Or` has exactly two
    // readers on a `Binds::Writes` parameter and neither reaches this one:
    //
    // * `arity_problem` (`model-ir/src/kernels.rs:303`) counted it into
    //   `opt_writes`, giving a band of `[1, 2]`. This symbol has ONE text --
    //   `dsl::cuda::attention_compressed_paged`
    //   (`model-dsl/src/cuda/deepseek_v4.rs:298`) -- whose `outs:` list is
    //   unconditional and two long, and `deepseek_v4/forward/mod.rs:155` is
    //   its only caller. `declared` is 2, always. The band is `[2, 2]` now
    //   and 2 is in it.
    // * `accepts_an_unstated_result` (`:227`) is consulted by
    //   `model-compiler/src/lower/walk.rs:472` only when `op.outputs` is
    //   EMPTY. It never is here, and deepseek-v4's forward opens no
    //   `dsl::guarded_value` or `dsl::regions` at all -- so unlike fa2's `o`
    //   and ssm's six prefill recurrences, no guard's value depends on this
    //   flag.
    //
    // The third reader, `bind/table.rs:721`'s `d.nullable => Ptr(null)`, is
    // not reachable either: this row is `arm: None, unbound: Some(..)`
    // (`arms/attn.rs`), so nothing binds it at all.
    //
    // **THE `nullptr` GUARD IN THE KERNEL IS NOT EVIDENCE THAT THE HOST MAY
    // PASS ONE.** It is device-side defensiveness on a `.cuh` shared with
    // callers this file does not own, and it is what makes the D2 verdict
    // above right. It says nothing about which arities a Rust signature must
    // admit, and the two were being conflated by one spelling.
    lse_out: Out<1, f32>,
    // `Env<keys::Positions>` and not `Env<*const i32>`: the same
    // `Source::Named(<keys::Positions as keys::Fact>::KEY)`, reached from the TYPE instead of from the name.
    positions: Env<keys::Positions>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    // *"a per-token request map that nothing in `driver-cuda` builds"*.
    req_of_token: Env<keys::RequestOfToken>,
    num_q_heads: Env<keys::NumQHeads>,
    head_dim: Env<keys::HeadDim>,
    // The `dsv4` three's ratio, as on the boundary pair: a number no context
    // holds. Reason (a).
    ratio: i32,
    page_size: Env<keys::KvPageSize>,
    // `cx.sm_scale()` = `keys::SmScale`, answered by `operand()`. The doc
    // above cites this parameter as a producer *"with no query and no
    // variant"*; the variant exists and the query is `Fire::sm_scale`, so the
    // citation is retired here and at `dispatch_attention_mla_bf16`.
    sm_scale: Env<keys::SmScale>,
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
                [o.rows.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
                [DSV4_ATTN_BLOCK, 1, 1],
            )
            .smem(smem),
            &[
                q.ptr.arg(),
                comp_kv_pages.arg(),
                o.ptr.arg(),
                lse_out.ptr.arg(),
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
///
/// The row (`arms/attn.rs:555`) declines over two operands and one number:
/// *"`dsl::cuda::dsa_index_knorm_rope` names NO weight bank, and the kernel
/// reads a LayerNorm weight AND a bias -- two operands with nothing to come
/// from, on top of the `rope_dim` its sibling also lacks. `head_dim` alone is
/// statable, as `out_width(0)`"*.
#[kernels_macros::routine]
pub fn dsa_index_knorm_rope<T>(
    ctx: &Ctx,
    idx_k: Out<0, T>,
    k_norm_weight: Unbound<*const T>,
    // A LayerNorm bias, absent on a text that norms without one.
    k_norm_bias: Unbound<*const T>,
    // The fire's position array, as on every rope in the tree.
    //
    // `fact_of` maps `positions` to `Source::Named(<keys::Positions as keys::Fact>::KEY)` and `operand()`
    // answers it, so this one resolves without a `#[source]`.
    positions: Env<keys::Positions>,
    // *"`rope_dim` appears in no statement, no shape and no context at all"*
    // (`arms/attn.rs:542`, the sibling's row). Reason (a).
    rope_dim: i32,
    theta: Env<keys::Theta>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // A HEAD DIM AND NOT A PITCH: `dsa_indexer.cuh:111` strides `idx_k` by
    // it and `:115` loops to it, so this operand's row IS one head and the
    // width the row calls `out_width(0)` is `head_dim` itself.
    let dst = idx_k.all("out_width(0)")?;
    let head_dim = dst.width;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsa_indexer.cuh",
            &format!("::pie::attn::index_knorm_rope<{}>", T::CPP),
            Launch::per_row(dst.rows.unsigned_abs(), dsa_indexer::K_BLOCK),
            &[
                idx_k.ptr.arg(),
                k_norm_weight.ptr.arg(),
                k_norm_bias.ptr.arg(),
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
///
/// **Read the `in_place` note in `ROUTINES` before crossing this one.** The
/// kernel reads AND writes `idx_q` and the registration does not say so.
#[kernels_macros::routine]
pub fn dsa_index_q_rope<T>(
    ctx: &Ctx,
    idx_q: Out<0, T>,
    positions: Env<keys::Positions>,
    n_heads: i32,
    head_dim: i32,
    // *"`rope_dim` appears in no statement, no shape and no context at all"*.
    // Reason (a).
    rope_dim: i32,
    // As its sibling's, and on the same indirect evidence.
    theta: Env<keys::Theta>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsa_indexer.cuh",
            &format!("::pie::attn::index_q_rope<{}>", T::CPP),
            Launch::per_row(idx_q.rows.unsigned_abs(), dsa_indexer::q_rope_block(n_heads)),
            &[
                idx_q.ptr.arg(),
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
///
/// Unlike its two siblings this symbol's scalar sizes are statement params:
/// the final three parameters name `Kind::Param(0..=2)` directly, matching the
/// old arm's `cx.param` reads while leaving the input/output counters at the
/// three score buffers and one mask.
#[kernels_macros::routine]
pub fn dsa_index_topk_mask(
    ctx: &Ctx,
    idx_q: In<0, bf16>,
    idx_k: In<1, bf16>,
    idx_w: In<2, bf16>,
    mask: Out<0, u8>,
    n_heads: Param<0, i32>,
    head_dim: Param<1, i32>,
    topk: Param<2, i32>,
) -> Result<(), Refusal> {
    let smem = mask
        .rows
        .unsigned_abs()
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/dsa_indexer.cuh",
            "::pie::attn::index_topk_mask<::pie::bf16>",
            Launch::per_row(mask.rows.unsigned_abs(), dsa_indexer::K_BLOCK).smem(smem),
            &[
                idx_q.ptr.arg(),
                idx_k.ptr.arg(),
                idx_w.ptr.arg(),
                mask.ptr.arg(),
                mask.rows.arg(),
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
/// picks naive.** `attention_mla.cu:334-340` is the only account anywhere of
/// why, and it is a correctness claim rather than a performance one:
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
/// `o_proj` that follows reads zeros, and the model emits text. That is the
/// premise every decision below rests on.
///
/// # Two device queries, and only one of them may fall back
///
/// The capability is [`Ctx::compute_capability_major`], and a `None` REFUSES
/// rather than defaulting: the arm a default would pick is the one that
/// returns zeros on the one architecture the query exists to detect.
///
/// The FA2 arm needs a second number, `cudaDevAttrMaxSharedMemoryPerMultiprocessor`,
/// which upstream's `BatchMLAPagedAttention` read before handing it to
/// `DISPATCH_SMEM_CONFIG`; its three thresholds survive as [`mla_fa2::ARMS`],
/// so reading the same attribute keeps this port's arm equal to upstream's.
/// It comes through [`fa2::plan::fa_device`], memoised so two fires in one
/// process cannot disagree, and with `Device::L40S` as a NAMED fallback.
/// **That fallback is admissible where the capability's would not be**, and
/// the difference is the failure mode: a wrong shared-memory budget picks an
/// arm the device cannot allocate and the launch FAILS loudly; a wrong
/// capability picks an arm that launches, succeeds, and writes zeros.
///
/// # `MlaNaive::Declined` is not an error and is not translated
///
/// It is returned as [`MlaDispatch::Naive`] holding exactly what
/// [`mla_naive::fire`] returned. The arm has already classified it, and
/// `Refusal` is a `Copy` value with no payload, so translating would flatten
/// four distinguishable answers — an empty batch, a null indptr trio, a
/// latent too wide for the register array, a rope tail too wide — into one.
/// The return type has to say *"nothing was launched"* regardless, because
/// [`mla_naive::NaiveDecline::NoTokens`] is a legal empty fire, and erasing
/// that into `Ok(())` is how a caller comes to believe `o` was written.
///
/// The two `Refusal`s below are the contrast: a device that will not state
/// its capability, and a device no [`mla_fa2::ARMS`] entry fits, are the
/// DEVICE failing to answer rather than the SHAPE going unserved.
///
/// # It is NOT `whole`
///
/// [`mla_prepare_bf16`] and [`write_mla_to_pages`] are `whole` because they
/// WALK `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`, which are
/// R-shaped, so a row window leaves that arithmetic pointing at another
/// request's rows. This reads a plan built over the whole fire and still
/// covers a row range, so a row window is legal and the column stays `false`.
/// `tests/stated_columns.rs` pins it there.
///
/// # Two things the arms do not agree on, recorded rather than unified
///
/// Both are upstream's and both are load-bearing for a caller. Making the
/// arms agree would be inventing a behaviour rather than porting one.
///
/// * **An empty fire.** `total_tokens <= 0` is
///   [`mla_naive::NaiveDecline::NoTokens`] on the naive arm; the FA2 arm has
///   no such test, since its grid is the plan's `num_blks_x`/`num_blks_y` and
///   an empty batch is the scheduler's business.
/// * **`lse`.** Only the FA2 arm writes one — neither naive `__global__`
///   takes the pointer — so on a `>= 10` device a caller that passed a
///   non-null `lse` gets an untouched buffer and no word about it. Not a
///   refusal here, because refusing would make a Blackwell box reject a fire
///   an sm_90 box serves.
///
/// # What is verified, and what is not
///
/// **No fire reaches this function, so no numerical result backs a single
/// line of it**, including the arm choice it exists to make. What is verified
/// is that it compiles and that every template-id both arms name is one NVRTC
/// lowers (`tests/every_instantiation_compiles`, which needs `libnvrtc` and
/// not a device). Two things still block MLA end to end and neither moves
/// because this exists: `fire/launch.rs`'s `kv_pools_for` refuses
/// `KvStyle::Mla`, so no latent cache is allocated; and `serve/load.rs`
/// refuses an MLA checkpoint at model load.
///
/// # Errors
///
/// [`Refusal::Device`] if the device will not state its compute capability,
/// and [`Refusal::Absent`] if no [`mla_fa2::ARMS`] entry fits its shared
/// memory — which is `DISPATCH_SMEM_CONFIG`'s own final `else`. Otherwise
/// whatever the chosen arm's compile, load or launch refuses. **A shape the
/// naive pair cannot serve is not among these**; see above.
///
/// # Safety
///
/// Every pointer must be a device address live across the launch — `layer`'s
/// two page pointers and `plan`'s two arenas included. Beyond that,
/// [`mla_fa2::pack`]'s own obligation, inherited and not widened: `plan` must
/// be the plan those arenas were uploaded from, because the FA2 arm reaches
/// fourteen index arrays in the int arena and two partial buffers in the
/// float one by adding that plan's offsets to those two base pointers.
/// `plan` must also have been scheduled for `layer`'s geometry and for
/// `num_heads`; nothing here can check it.
///
/// Marked but NOT given a column, and it could not carry a useful one: a
/// `driver_bound!` row, an `unsafe fn`, returning `MlaDispatch` rather than
/// `()`.
#[kernels_macros::routine]
pub unsafe fn dispatch_attention_mla_bf16(
    ctx: &Ctx,
    plan: &MlaPlan,
    q_nope: In<0, bf16>,
    q_pe: In<1, bf16>,
    // `Cx::mla_layer`, the same structural refusal `mla_prepare_bf16` blocks
    // on. Reason (a).
    layer: MlaLayer,
    o: Out<0, bf16>,
    kv_page_indices: Env<keys::KvPageIndices>,
    lse: Unbound<*mut f32>,
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Unbound<*const u32>,
    index_mask: Unbound<*const u8>,
    index_mask_stride: i32,
    num_requests: Env<keys::RequestCount>,
    num_heads: i32,
    // `AttnCtx::sm_scale` = `keys::SmScale`, which `operand()` answers off
    // `f.sm_scale()`. It was *"a producer with no query and no variant"*
    // when this comment was written and it is neither now.
    sm_scale: Env<keys::SmScale>,
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
            q_nope: q_nope.ptr,
            q_pe: q_pe.ptr,
            ckv_pages: layer.ckv_pages.cast::<bf16>().cast_const(),
            kpe_pages: layer.kpe_pages.cast::<bf16>().cast_const(),
            qo_indptr: **qo_indptr,
            kv_page_indices: **kv_page_indices,
            kv_page_indptr: **kv_page_indptr,
            kv_last_page_lens: kv_last_page_lens.ptr,
            o: o.ptr,
            index_mask: index_mask.ptr,
        };
        let shape = mla_naive::NaiveShape {
            kv_lora_rank: layer.kv_lora_rank,
            qk_rope_head_dim: layer.qk_rope_head_dim,
            page_size: layer.page_size,
            total_tokens: o.rows,
            num_requests: **num_requests,
            num_heads,
            sm_scale: **sm_scale,
            causal,
            index_mask_stride,
        };
        return mla_naive::fire(ctx, ptrs, shape).map(MlaDispatch::Naive);
    }

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
        sm_scale: **sm_scale,
    };
    let buffers = mla_fa2::Buffers {
        int_buffer: plan.int_arena.cast::<u8>(),
        float_buffer: plan.float_arena.cast::<u8>(),
        q_nope: q_nope.ptr.cast_mut(),
        q_pe: q_pe.ptr.cast_mut(),
        ckv_pages: layer.ckv_pages.cast::<bf16>(),
        kpe_pages: layer.kpe_pages.cast::<bf16>(),
        out: o.ptr,
        kv_page_indices: (**kv_page_indices).cast::<i32>().cast_mut(),
        lse: lse.ptr,
    };
    // SAFETY: the caller's obligation, forwarded -- every pointer here is a
    // device address live across the launch, and `plan` is the plan its two
    // arenas were uploaded from. Nothing is dereferenced on this side; the
    // offsets `pack` adds are the ones the upload used.
    let params = unsafe { mla_fa2::pack(&plan.info, shape, buffers, !lse.ptr.is_null()) };
    // `attention_mla.cu:406-414` — the mask is a template parameter upstream
    // and an index here, which is why one call replaces the C++'s two.
    mla_fa2::fire(ctx, arm, causal, &params, mla_fa2::grid(&plan.info, mla_fa2::ARMS[arm]))?;
    Ok(MlaDispatch::Fa2 { arm })
}

/// The units `attn` compiles in fn-world.
pub mod qkv_fused {
    use kernels::keys;
    use super::bf16;
    use super::{Ctx, Launch, Refusal};
    use kernels::routine::{Bank, Env, In, Out};
    
    use crate::jit::Abi;

    /// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` — the fused
    ///
    /// What the caller must guarantee, as `call()` states it: every pointer
    /// must be a device address valid for the fire, `packed` must hold
    /// `num_rows` rows of `(num_q_heads + 2·num_kv_heads)·head_dim` elements,
    /// and the page arrays must describe the layer the cache pointers came
    /// from.
    ///
    /// The column is read off `qkv_packed_post_arm`, argument for argument —
    /// and there is no longer an arm to read it off. **`arms/attn.rs`'s row
    /// says `Bound::derived` as of Kilimanjaro III Stage 3**: the two things
    /// the arm did that a column could not were `plan.row_valid`, which had
    /// no key until `keys::RowValid` landed, and `out_width(0) /
    /// layer.head_dim`, which is arithmetic and belongs to the launcher that
    /// divides (F6). Both are below, and the pin at the end of this file
    /// asserts the seventeen entries the column now derives.
    #[kernels_macros::routine]
    pub fn qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
        ctx: &Ctx,
        // `In(0)` -- the statement's one input, which is the whole of what
        // position was guessing correctly.
        packed: In<0, bf16>,
        q_out: Out<0, bf16>,
        k_pages: Env<keys::KvKeys>,
        v_pages: Env<keys::KvValues>,
        // `cx.weight(0)` and `cx.weight(1)` (`arms/attn.rs:413-414` -- the
        // citation said `:227-228` and the arm has moved twice since), the
        // POSITIONAL banks, so `Weight(n)` and not `WeightNamed`. Neither
        // mark consumes the `In` counter, which is what leaves the positional
        // pointers above them where the arm puts them.
        q_weight: Bank<0, bf16>,
        k_weight: Bank<1, bf16>,
        positions: Env<keys::Positions>,
        kv_page_indices: Env<keys::KvPageIndices>,
        kv_page_indptr: Env<keys::KvPageIndptr>,
        kv_last_page_lens: Env<keys::KvLastPageLens>,
        // `plan.row_valid`, and it is `keys::RowValid` rather than a bare
        // `Env<*const u8>` as of Kilimanjaro III Stage 3: the fact was read
        // by hand at `arms/attn.rs` for as long as this signature existed and
        // had no NAME, which is the whole of §1.5's *"reads an aggregate
        // field with no key"*. `operand()` answers it off `cx.plan()`.
        row_valid: Env<keys::RowValid>,
        // `num_q_heads` STOOD HERE AND WAS THE ARM'S ARITHMETIC — F6.
        // `qkv_packed_post_arm` read `cx.out_width(0)? / layer.head_dim`,
        // which is `q_out`'s width over a fact this signature already takes,
        // so the quotient is computed below and the parameter is gone. It was
        // the LAST thing that arm did that this body could not, and the row
        // reads `Bound::derived` now.
        num_kv_heads: Env<keys::KvNumHeads>,
        head_dim: Env<keys::KvHeadDim>,
        page_size: Env<keys::KvPageSize>,
        hnd_layout: Env<keys::KvHndLayout>,
        theta: Env<keys::Theta>,
        eps: Env<keys::RmsEps>,
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

        // THE ARM'S TWO LINES, IN THE LAUNCHER THAT DIVIDES. The guard is
        // `qkv_packed_post_arm`'s, word for word -- `Refusal::Empty { what:
        // "head_dim" }` -- and it comes first for the reason it did there: a
        // zero head width is a division by zero, which panics rather than
        // refusing. The numerator is the region's, so the operand that proves
        // the launch has a result is the operand its width is read from.
        //
        // ONE REFUSAL WORD MOVED AND NO REFUSAL DID. The arm read
        // `cx.out_width(0)?`, which is `Refusal::Absent { what: "an output's
        // width" }` (`bind/table.rs:461`); `all()` refuses `Absent { what:
        // "out_width(0)" }` on the same `width <= 0`
        // (`kernels/src/routine.rs:1146`). Same kind, same shapes refused,
        // and the new word names the operand a reader would go and look at.
        if **head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let num_q_heads = q_out.all("out_width(0)")?.width / **head_dim;
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/qkv_fused.cuh",
                "::pie::attn::qkv_packed_qk_norm_rope_vnorm_write_kv<::pie::i32(256)>",
                Launch::grid([packed.rows.unsigned_abs(), heads, 1], [PACKED_BLOCK, 1, 1]),
                &[
                    packed.ptr.arg(),
                    q_out.ptr.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    q_weight.ptr.arg(),
                    k_weight.ptr.arg(),
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
    ///
    /// The column is read off `qkv_decode_fused_arm` and turns on one
    /// property of the macro: `#[source(..)]` short-circuits ahead of both
    /// counters (`kernels-macros/src/lib.rs:205`), so marking the two weights
    /// leaves `rope_table` -- the only other bare `*const` -- at `In(1)`,
    /// which is exactly the arm's `cx.arg_in(1)`. Had the weights been
    /// wrapped `Weight<..>` instead, `stated_source` would have run and the
    /// same thing would have happened; had they been left bare, the column
    /// would have claimed `In(1)` and `In(2)` for two banks and pushed
    /// `rope_table` to `In(3)`.
    ///
    /// **THAT ARM STAYS AND ITS PACKED TWIN'S DOES NOT**, over one parameter:
    /// `q_out` is `Out<0, _>`: the region form states no result, so
    /// `TraceBuilder::push` stamps the enclosing region's buffer onto
    /// `Op::dest` and the lowerer pushes it into the operand run.
    #[allow(clippy::fn_params_excessive_bools)]
    #[kernels_macros::routine]
    pub fn qkv_decode_qk_norm_rope_write_kv_bf16(
        ctx: &Ctx,
        // `In(0)` -- the statement's packed projection.
        packed: In<0, bf16>,
        q_out: Out<0, bf16>,
        // `layer.k_pages` / `layer.v_pages`. `KvKeys` / `KvValues`, named and
        // answered as of Stage 2: reason (b) no longer applies to these two.
        k_pages: Env<keys::KvKeys>,
        v_pages: Env<keys::KvValues>,
        // `cx.weight(0)` / `cx.weight(1)`, the POSITIONAL banks.
        q_weight: Bank<0, bf16>,
        k_weight: Bank<1, bf16>,
        positions: Env<keys::Positions>,
        rope_table: In<1, f32>,
        kv_page_indices: Env<keys::KvPageIndices>,
        kv_page_indptr: Env<keys::KvPageIndptr>,
        kv_last_page_lens: Env<keys::KvLastPageLens>,
        // The `_or_null` pair: a fire that appends no KV carries a null and the
        // kernel branches on it, so the refusing spelling would decline it.
        w_page: Env<keys::KvWritePageOrNull>,
        w_off: Env<keys::KvWriteOffsetOrNull>,
        // `w_page`/`w_off` STAY BARE ON THIS ROW ALONE. `keys::KvWritePage`
        // and `keys::KvWriteOffset` bind them on `write_kv_explicit_bf16`,
        // where the arm reads `cx.w_page_d()?` and the key's null-check is the
        // same refusal. Here the arm reads `unwrap_or(null)` and the kernel
        // branches on the null itself, so the key would refuse a valid fire.
        row_valid: Env<keys::RowValid>,
        // `num_q_heads` STOOD HERE. `qkv_decode_fused_arm` computed
        // `(cx.in_width(0)? - 2 * num_kv_heads * head_dim) / head_dim` and
        // divided by a `head_dim` it never checked -- a panic rather than a
        // refusal on a layer view stating none. F6 puts the arithmetic in the
        // launcher that has all three numbers, and the guard with it.
        num_kv_heads: Env<keys::KvNumHeads>,
        head_dim: Env<keys::KvHeadDim>,
        page_size: Env<keys::KvPageSize>,
        hnd_layout: Env<keys::KvHndLayout>,
        theta: Env<keys::Theta>,
        eps: Env<keys::RmsEps>,
    ) -> Result<(), Refusal> {
        // The packed twin's guard, for the packed twin's reason, over the
        // subtraction this form does first: the packed row's width is q's
        // heads and both KV halves, so the query heads are what is left after
        // the two KV runs are taken off it. `cx.in_width(0)?` became
        // `packed.all("in_width(0)")?` -- `Refusal::Absent` either way, and
        // the word now names the slot rather than "an input's width".
        if **head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let packed_width = packed.all("in_width(0)")?.width;
        let num_q_heads = (packed_width - 2 * **num_kv_heads * **head_dim) / **head_dim;
        qkv_decode_fused_dispatch(
            ctx,
            packed.ptr,
            q_out.ptr,
            k_pages.cast::<bf16>(),
            v_pages.cast::<bf16>(),
            q_weight.ptr,
            k_weight.ptr,
            **positions,
            rope_table.ptr,
            **kv_page_indices,
            **kv_page_indptr,
            **kv_last_page_lens,
            **w_page,
            **w_off,
            **row_valid,
            core::ptr::null(),
            // The misnamed parameter's value, from the region rather than
            // from a `#[source(Rows)]` scalar. Same number, same slot.
            packed.rows,
            num_q_heads,
            **num_kv_heads,
            **head_dim,
            **page_size,
            **hnd_layout,
            **theta,
            **eps,
        )
    }
}

/// `attn/dsv4_compress.cuh` — deepseek_v4's SECOND KV cache, and the eleven
pub mod dsv4_compress {
    use super::bf16;
    use kernels::keys;
    use kernels::routine::{Env, In, Out};
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
    ///
    /// The row (`arms/attn.rs:716`) declines the whole symbol: *"`state_kv`,
    /// `state_score`, `ape`, `boundary_req`, `ratio` and `coff` have no
    /// operand to come from"*. Six of thirteen, so the marks below are the
    /// other seven and the reasons are the six.
    #[kernels_macros::routine]
    pub fn dsv4_compress_gather_paged_bf16(
        ctx: &Ctx,
        state_kv: Env<*const bf16>,
        state_score: Env<*const bf16>,
        // The absolute-position table, a load-time constant.
        ape: Env<*const f32>,
        boundary_pos: In<0, i32>,
        boundary_req: In<1, i32>,
        kv_page_indices: Env<keys::KvPageIndices>,
        kv_page_indptr: Env<keys::KvPageIndptr>,
        // `Out(0)`, the statement's single `out`, declared
        // `[Dim::Tokens, Dim::Const(head_dim)] as BF16`.
        out: Out<0, bf16>,
        num_entries: i32,
        ratio: i32,
        coff: i32,
        page_size: i32,
    ) -> Result<(), Refusal> {
        let head_dim = out.all("out_width(0)")?.width;
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
                    boundary_pos.ptr.arg(),
                    boundary_req.ptr.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    out.ptr.arg(),
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
    ///
    /// The row (`arms/attn.rs:730`): the statement *"names `entries` and
    /// `boundary_pos`, and the kernel also reads `boundary_req` [...] and
    /// needs `head_dim` and `page_size` besides"*. So `In(0)` and `In(1)` are
    /// the two it names and everything else refuses.
    #[kernels_macros::routine]
    pub fn dsv4_store_comp_entries_bf16(
        ctx: &Ctx,
        entries: In<0, bf16>,
        comp_kv_pages: Env<*mut bf16>,
        // `In(1)`, stated, and the statement does name it.
        boundary_pos: In<1, i32>,
        boundary_req: In<2, i32>,
        kv_page_indices: Env<keys::KvPageIndices>,
        kv_page_indptr: Env<keys::KvPageIndptr>,
        num_entries: i32,
        page_size: i32,
    ) -> Result<(), Refusal> {
        // The gather's guard, for the gather's reason -- `route_rows`'s
        // `.max(1)` hides a zero width from the empty-grid check. A view is
        // where that guard lives now, and it refuses in the same word.
        let head_dim = entries.all("in_width(0)")?.width;
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/dsv4_compress.cuh",
                "::pie::attn::dsv4_store_comp_entries<::pie::bf16>",
                route_rows(num_entries, head_dim),
                &[
                    entries.ptr.arg(),
                    comp_kv_pages.arg(),
                    boundary_pos.ptr.arg(),
                    boundary_req.ptr.arg(),
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
    use kernels::keys;

    use kernels::routine::{Env, In, Unbound};
    use super::bf16;
    use crate::jit::abi::MaybeConst;
    use crate::jit::fp8_kind;

    use super::{Ctx, Launch, scheme_byte};
    
    use core::ffi::c_void;

    use crate::jit::Abi;
    use crate::attn::{KvDType, KvScheme, kv_dtype, kv_scheme};
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
    ///
    /// Takes the DEVICE spelling, not [`KvDType`]: `.wiki/kilimanjaro.md` §5
    /// D1 made the storage type an argument of
    /// [`write_kv_to_pages_quantised`], and a `routine!` argument has to be
    /// an `Arg`, which only [`kv_dtype`] is. The two dequant programs still
    /// hold a `&KvLayer` and wrap at the call — see their headers for why
    /// they were left behind.
    fn fp8_kind_of(storage_dtype: kv_dtype) -> fp8_kind {

        const NV_E5M2: u32 = 1;

        /// `::__nv_fp8_interpretation_t`'s two values, by the names the header
        const NV_E4M3: u32 = 0;

        fp8_kind(if storage_dtype == kv_dtype::of(KvDType::Fp8E5M2) { NV_E5M2 } else { NV_E4M3 })
    }

    /// NVFP4's block, when the layer states none.
    fn fp4_block_size(block_size: i32) -> i32 {
        if block_size > 0 { block_size } else { 16 }
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
    /// # The layer arrives as FIELDS
    ///
    /// A routine takes fields, never a struct: the only `ArgValue` a struct
    /// could ride in is `Bytes`, whose layout agreement *"is not checked and
    /// cannot be here"* (`kernels-cuda/src/jit/value.rs:32`). Sixteen
    /// arguments buy that check back, which is why the signature is long.
    ///
    /// `is_native_bf16` is still taken, and still only asserted on: the
    /// kernel this launches writes bf16 unconditionally, so the field is the
    /// caller's claim that the pages can hold what it writes, not an operand.
    ///
    /// What the caller must guarantee, as `call()` states it: every pointer
    /// must be a device allocation of the stated extent. The `Env` marks say
    /// which of them the FIRE supplies rather than the trace statement —
    /// §6.2's arity rule counts the unmarked ones against the two inputs
    /// `write_kv_explicit` places.
    ///
    /// The column is read off `write_kv_explicit_arm` (`arms/attn.rs:324-348`)
    /// and is nine tenths refusal: two `In`s, `Rows`, `KvPageSize`, and
    /// thirteen facts the fire holds and the table has no word for.
    #[kernels_macros::routine]
    pub fn write_kv_explicit_bf16(
        ctx: &Ctx,
        k_curr: In<0, bf16>,
        v_curr: In<1, bf16>,
        k_pages: Env<keys::KvKeys>,
        v_pages: Env<keys::KvValues>,
        w_page: Env<keys::KvWritePage>,
        w_off: Env<keys::KvWriteOffset>,
        // `plan.row_valid` = `keys::RowValid` (`keys.rs` §1). This was
        // reason (a) -- a fact with no word -- until Kilimanjaro III Stage 3
        // minted one; `operand()` answers it off `cx.plan()`, so the arm
        // stops reading it by hand.
        row_valid: Env<keys::RowValid>,
        // `layer.k_env_min` / `layer.k_env_max` -- the per-page bf16 envelope
        // pair, two more `KvLayer` fields. Reason (b), `KvLayerField`, AND
        // THE FIRST REFUSAL ON THIS ROW now that `row_valid` above it is
        // answered: this is the pair `write_kv_to_pages_bf16_arm`'s doc names.
        k_env_min: Env<keys::KvEnvMin>,
        k_env_max: Env<keys::KvEnvMax>,
        page_size: Env<keys::KvPageSize>,
        num_kv_heads: Env<keys::KvNumHeads>,
        head_dim: Env<keys::KvHeadDim>,
        hnd: Env<keys::KvHndLayout>,
        // `layer.has_envelopes` = `keys::KvHasEnvelopes`, `bool` because
        // `operand()` mints an `ArgValue::Bool` for it (`table.rs:928`) and
        // an `i32` key would refuse the day an arm is written.
        has_envelopes: Env<keys::KvHasEnvelopes>,
        is_native_bf16: Env<keys::KvNativeBf16>,
    ) -> Result<(), Refusal> {
        assert!(**is_native_bf16, "attn::write_kv_explicit_bf16 requires native bf16 KV cache");

        let instantiation =
            if **hnd { "::pie::attn::write_kv_explicit<\
                                ::pie::true_type::value>" } else { "::pie::attn::write_kv_explicit<::pie::false_type::value>" };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/kv_paged.cuh",
                instantiation,
                Launch::per_row(k_curr.rows.unsigned_abs(), BLOCK),
                &[
                    k_curr.ptr.arg(),
                    v_curr.ptr.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    w_page.arg(),
                    w_off.arg(),
                    MaybeConst::new(**row_valid).arg(),
                    k_curr.rows.arg(),
                    page_size.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                ],
            )?;
        }

        if **has_envelopes && !**hnd {
            let _ = crate::layout::envelope_merge_written(
                ctx,
                Unbound { ptr: k_curr.ptr },
                Unbound { ptr: **w_page },
                Unbound { ptr: **w_off },
                Unbound { ptr: MaybeConst::new(**row_valid) },
                Unbound { ptr: (**k_env_min).cast::<bf16>().cast_mut() },
                Unbound { ptr: (**k_env_max).cast::<bf16>().cast_mut() },
                k_curr.rows,
                // `crate::layout` is not this family and its parameters are flat
                // `i32`s, so the two facts are opened at the call.
                **num_kv_heads,
                **head_dim,
            );
        }
        Ok(())
    }

    /// `attn::write_kv_explicit_bf16_devwin` — the same write with a
    ///
    /// As [`write_kv_explicit_bf16`], fields and all: §5 D1 unpacked this
    /// signature at the same time and for the same reason.
    ///
    /// **No `Bound` row names this symbol.** The marks are read off
    /// [`write_kv_explicit_bf16`]'s arm, which is the same write with a host
    /// window, plus the one thing the `_devwin` form does differently: it
    /// takes `cx.rows().total` where the host-window form takes `.count`. The
    /// `split_qkv_bf16_devwin` arm (`arms/attn.rs:146`) is the evidence for
    /// that, since it is the other `_devwin` in the family and does exactly
    /// this.
    #[kernels_macros::routine]
    pub fn write_kv_explicit_bf16_devwin(
        ctx: &Ctx,
        k_curr: In<0, bf16>,
        v_curr: In<1, bf16>,
        k_pages: Env<keys::KvKeys>,
        v_pages: Env<keys::KvValues>,
        w_page: Env<keys::KvWritePage>,
        w_off: Env<keys::KvWriteOffset>,
        win_d: Env<keys::PeelWindow>,
        row_valid: Env<keys::RowValid>,
        n_max: Env<keys::RowsTotal>,
        page_size: Env<keys::KvPageSize>,
        num_kv_heads: Env<keys::KvNumHeads>,
        head_dim: Env<keys::KvHeadDim>,
        hnd: Env<keys::KvHndLayout>,
        has_envelopes: Env<keys::KvHasEnvelopes>,
        is_native_bf16: Env<keys::KvNativeBf16>,
    ) -> Result<(), Refusal> {
        assert!(
            **is_native_bf16,
            "attn::write_kv_explicit_bf16_devwin requires native bf16 KV cache"
        );
        assert!(
            !**has_envelopes,
            "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
             windowed — use the host-window form"
        );

        let instantiation = if **hnd {
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
                Launch::per_row((**n_max).unsigned_abs(), BLOCK),
                &[
                    k_curr.ptr.arg(),
                    v_curr.ptr.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    w_page.arg(),
                    w_off.arg(),
                    MaybeConst::new(**row_valid).arg(),
                    win_d.arg(),
                    n_max.arg(),
                    page_size.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                ],
            )
        }
    }

    /// `attn::write_kv_to_pages_bf16` — the native-bf16 append,
    /// `kv_paged.cu:60-120`.
    ///
    /// # A SYMBOL, not an arm
    ///
    /// `is_native_bf16` is a BOOT FACT — `[driver] kv_cache_dtype`, fixed
    /// before the first token — so [`write_kv_to_pages`] is a MAP resolved
    /// once at model load, and this is a symbol the driver binds directly.
    ///
    /// # And it takes FIELDS
    ///
    /// §5 D1 — *"a routine takes fields, never a struct"*. Nine of
    /// `KvLayer`'s eighteen fields are read here; the other nine were
    /// carried in for nothing by a `&KvLayer` that no `ArgValue` can hold
    /// without falling back on `Bytes`, whose layout agreement
    /// *"is not checked and cannot be here"* (`kernels-cuda/src/jit/value.rs:32`).
    /// Nineteen arguments, exactly what the spec's table predicted.
    ///
    /// As [`write_kv_explicit_bf16`]; the four CSR arrays must describe
    /// `num_requests` requests over `total_tokens` tokens. `k_curr`/`v_curr`
    /// are the two operands the trace statement places and so carry no mark;
    /// everything else the FIRE supplies, which is what `Env` says.
    ///
    /// The column is read off `write_kv_to_pages_bf16_arm`
    /// (`arms/attn.rs:246-273`).
    #[kernels_macros::routine]
    pub fn write_kv_to_pages_bf16(
        ctx: &Ctx,
        k_curr: In<0, bf16>,
        v_curr: In<1, bf16>,
        k_pages: Env<keys::KvKeys>,
        v_pages: Env<keys::KvValues>,
        qo_indptr: Env<keys::QoIndptr>,
        kv_page_indices: Env<keys::KvPageIndices>,
        kv_page_indptr: Env<keys::KvPageIndptr>,
        kv_last_page_lens: Env<keys::KvLastPageLens>,
        row_valid: Env<keys::RowValid>,
        // Two `KvLayer` fields: reason (b), and the FIRST refusal on this row
        // now that the CSR above it is four keys deep.
        k_env_min: Env<keys::KvEnvMin>,
        k_env_max: Env<keys::KvEnvMax>,
        num_requests: Env<keys::RequestCount>,
        first_token: Env<keys::FirstToken>,
        page_size: Env<keys::KvPageSize>,
        num_kv_heads: Env<keys::KvNumHeads>,
        head_dim: Env<keys::KvHeadDim>,
        hnd: Env<keys::KvHndLayout>,
        has_envelopes: Env<keys::KvHasEnvelopes>,
    ) -> Result<(), Refusal> {
        let launch_tokens = k_curr.rows - **first_token;

        let instantiation = if **hnd { "::pie::attn::write_kv<\
                                                ::pie::true_type::value>" } else { "::pie::attn::write_kv<::pie::false_type::value>" };
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/kv_paged.cuh",
                instantiation,
                Launch::per_row(launch_tokens.unsigned_abs(), BLOCK),
                &[
                    k_curr.ptr.arg(),
                    v_curr.ptr.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    qo_indptr.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    MaybeConst::new(**row_valid).arg(),
                    MaybeConst::<u32>::none().arg(),
                    num_requests.arg(),
                    page_size.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                    first_token.arg(),
                ],
            )?;
        }

        if **has_envelopes && !**hnd && k_curr.rows > 0 {
            let _ = crate::layout::envelope_update_appended(
                ctx,
                Unbound { ptr: k_pages.cast::<bf16>().cast_const() },
                Unbound { ptr: **qo_indptr },
                Unbound { ptr: **kv_page_indices },
                Unbound { ptr: **kv_page_indptr },
                Unbound { ptr: **kv_last_page_lens },
                Unbound { ptr: (**k_env_min).cast::<bf16>().cast_mut() },
                Unbound { ptr: (**k_env_max).cast::<bf16>().cast_mut() },
                **num_requests,
                max_touched_pages(k_curr.rows, **num_requests, **page_size),
                **page_size,
                **num_kv_heads,
                **head_dim,
            );
        }
        Ok(())
    }

    /// `attn::write_kv_to_pages_quantised` — the quantised append,
    /// `kv_paged.cu:130-190` — four schemes, three kernels.
    ///
    /// # The other half of the boot fact
    ///
    /// The sibling of [`write_kv_to_pages_bf16`], and a symbol for the same
    /// reason: §5 D5 turned `is_native_bf16` from a per-fire branch into a
    /// load-time map, and this is what the map answers when the boot chose a
    /// quantised `[driver] kv_cache_dtype`. Read that header first — it
    /// carries the argument, and [`write_kv_to_pages`] carries the epitaph.
    ///
    /// The scheme, in contrast, is NOT a boot fact this routine could shed:
    /// four schemes over three kernels is a dispatch on an operand the layer
    /// carries, so it stays inside, and §5 D1 makes it arrive as
    /// [`kv_scheme`] — the device spelling, because a `routine!` derives its
    /// row from the signature and only that spelling is an `Arg`.
    /// [`kv_scheme::scheme`] decodes it back so the `match` stays exhaustive.
    ///
    /// TEN of `KvLayer`'s eighteen fields, eighteen arguments. The spec's D1
    /// table predicted the argument count exactly and said *"uses 9 of 18"*
    /// of the fields; the tenth is `storage_dtype`, which only the
    /// fp8-per-tensor arm reads and only to pick between `E4M3` and `E5M2`.
    /// The count that governs is the argument count — `impl_kernel_fn!` is
    /// stamped through 36 (`kernels/src/routine.rs`) — and eighteen is what
    /// this is.
    ///
    /// As [`write_kv_to_pages_bf16`]; the layer's scale planes must be
    /// sized for its scheme.
    ///
    /// `first_token` IS TAKEN AND NEVER LAUNCHED. None of the three kernels
    /// below has a parameter for it, so all three write from row zero, and a
    /// fire meaning to skip the first tokens would land its rows in the wrong
    /// slots. That is a kernel limitation and the routine's own refusal, not
    /// a caller's policy -- `write_kv_to_pages`'s doc had already located it
    /// as *"`first_token != 0 && !is_native_bf16` must refuse"*, and it sat
    /// in the driver arm only because this signature did not take the fact.
    /// It does now. `is_native_bf16` on [`write_kv_explicit_bf16`] is the
    /// same shape: taken, asserted on, never passed to a kernel.
    #[kernels_macros::routine]
    pub fn write_kv_to_pages_quantised(
        ctx: &Ctx,
        k_curr: In<0, bf16>,
        v_curr: In<1, bf16>,
        first_token: Env<keys::FirstToken>,
        // `KvKeys` / `KvValues`, reason (b), witnessed at
        // `arms/attn.rs:556-557`.
        k_pages: Env<keys::KvKeys>,
        v_pages: Env<keys::KvValues>,
        // Same TYPE as a workspace buffer, a different fact, and that is why
        // the sweep that named `attn/fa2/`'s stopped here until these two got
        // their own keys.
        k_scales: Env<keys::KvKeyScales>,
        v_scales: Env<keys::KvValueScales>,
        // Four `plan()` CSRs, ALL FOUR spelled from their keys now
        // (`arms/attn.rs`'s bindings): `qo_indptr` was the odd one out under
        // reason (a) and `keys::QoIndptr` ended that.
        qo_indptr: Env<keys::QoIndptr>,
        kv_page_indices: Env<keys::KvPageIndices>,
        kv_page_indptr: Env<keys::KvPageIndptr>,
        kv_last_page_lens: Env<keys::KvLastPageLens>,
        num_requests: Env<keys::RequestCount>,
        page_size: Env<keys::KvPageSize>,
        num_kv_heads: Env<keys::KvNumHeads>,
        head_dim: Env<keys::KvHeadDim>,
        block_size: Env<keys::KvBlockSize>,
        scheme: Env<keys::KvSchemeByte>,
        storage_dtype: Env<keys::KvStorageDtype>,
    ) -> Result<(), Refusal> {
        if **first_token != 0 {
            return Err(Refusal::Absent {
                what: "a quantised appender that skips the first tokens",
            });
        }
        let scheme = kv_scheme(scheme_byte(**scheme));
        let storage_dtype = kv_dtype(scheme_byte(**storage_dtype));
        let h_kv = num_kv_heads;
        let d = head_dim;
        let tokens = k_curr.rows.unsigned_abs();
        let heads = h_kv.unsigned_abs();

        match scheme.scheme() {
            // SAFETY, in all three arms: `call()`'s contract -- every pointer
            // bound here addresses live device memory of the extent the
            // kernel reads it as.
            Some(KvScheme::Fp8PerTensor) => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::write_kv_fp8_per_tensor",
                    Launch::per_row(tokens, BLOCK),
                    &[
                        k_curr.ptr.arg(),
                        v_curr.ptr.arg(),
                        k_pages.arg(),
                        v_pages.arg(),
                        qo_indptr.arg(),
                        kv_page_indices.arg(),
                        kv_page_indptr.arg(),
                        kv_last_page_lens.arg(),
                        num_requests.arg(),
                        page_size.arg(),
                        h_kv.arg(),
                        d.arg(),
                        fp8_kind_of(storage_dtype).arg(),
                    ],
                )
            },

            Some(KvScheme::Int8PerTokenHead | KvScheme::Fp8PerTokenHead) => {
                let instantiation = if scheme == kv_scheme::of(KvScheme::Fp8PerTokenHead) {
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
                            k_curr.ptr.arg(),
                            v_curr.ptr.arg(),
                            k_pages.arg(),
                            v_pages.arg(),
                            k_scales.cast::<f32>().arg(),
                            v_scales.cast::<f32>().arg(),
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

            Some(KvScheme::Fp4Block) => {
                let block_size = fp4_block_size(**block_size);
                let blocks = d.div_euclid(block_size) + i32::from(d.rem_euclid(block_size) != 0);
                unsafe {
                    ctx.launch(
                        "attn/kv_paged.cuh",
                        "::pie::attn::write_kv_fp4_block",
                        Launch::grid([tokens, heads, blocks.unsigned_abs()], [32, 1, 1]),
                        &[
                            k_curr.ptr.arg(),
                            v_curr.ptr.arg(),
                            k_pages.arg(),
                            v_pages.arg(),
                            k_scales.cast::<f32>().arg(),
                            v_scales.cast::<f32>().arg(),
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

            Some(KvScheme::Native) => {
                Err(Refusal::Absent { what: "a quantised writer for Native storage" })
            }

            None => Err(Refusal::Absent { what: "a KV scheme this byte names" }),
        }
    }

    /// `attn::write_kv_to_pages` — the boot fact's MAP.
    ///
    /// Not a launcher: it answers which SYMBOL serves a deployment, and the
    /// driver asks once, at model load. `is_native_bf16` follows `[driver]
    /// kv_cache_dtype`, fixed before the first token, so the choice is a map
    /// rather than a per-fire branch — and a branch would hide two routines
    /// behind one symbol, leaving the load-time arity check nothing to check
    /// them against.
    ///
    /// The partial-write guard that belongs with this choice is NOT here:
    /// `first_token != 0 && !is_native_bf16` must refuse, and only the native
    /// appender takes `first_token`, so the guard lives in the driver arm that
    /// has it in hand (`driver-cuda/src/bind/arms/attn.rs`,
    /// `write_kv_to_pages_quantised_arm`).
    ///
    /// The stated symbol keeps its `driver_bound!` row: `kernels_cuda::sigs()`
    /// is what `check_plan` measures a model text against, and every live text
    /// states this name through `model-dsl/src/cuda/base.rs:696`.
    #[must_use]
    pub const fn write_kv_to_pages(is_native_bf16: bool) -> &'static str {
        if is_native_bf16 {
            concat!("attn::", stringify!(write_kv_to_pages_bf16))
        } else {
            concat!("attn::", stringify!(write_kv_to_pages_quantised))
        }
    }

    #[allow(dead_code)]
    fn the_map_names_two_real_fns() {
        let _ = (write_kv_to_pages_bf16, write_kv_to_pages_quantised);
    }

    /// The fp8-per-tensor arm, called by name from
    ///
    /// What the caller must guarantee, as `call()` states it:
    /// `kv_page_indices` must list `num_pages_in_batch` valid page indices,
    /// and the layer's bf16 mirror planes must be sized for them.
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_fp8_per_tensor_pages_active(
        ctx: &Ctx,
        k_pages: *mut u8,
        v_pages: *mut u8,
        k_bf16_pages: *mut c_void,
        v_bf16_pages: *mut c_void,
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        scheme: kv_scheme,
        storage_dtype: kv_dtype,
        is_native_bf16: bool,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
    ) -> Result<(), Refusal> {
        if is_native_bf16 {
            return Err(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if scheme != kv_scheme::of(KvScheme::Fp8PerTensor) {
            return Err(Refusal::Absent { what: "an fp8-per-tensor layer" });
        }

        let (logical_n, page_elems, launch) =
            active_geometry(page_size, num_kv_heads, head_dim, num_pages_in_batch);
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        unsafe {
            ctx.launch(
                "attn/kv_paged.cuh",
                "::pie::attn::dequant_fp8_pages_active",
                launch,
                &[
                    k_pages.cast::<u8>().cast_const().arg(),
                    v_pages.cast::<u8>().cast_const().arg(),
                    k_bf16_pages.cast::<bf16>().arg(),
                    v_bf16_pages.cast::<bf16>().arg(),
                    kv_page_indices.arg(),
                    logical_n.arg(),
                    page_elems.arg(),
                    fp8_kind_of(storage_dtype).arg(),
                ],
            )
        }
    }

    /// The element count an active-page pass covers, and the grid that
    fn active_geometry(
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        num_pages_in_batch: i32,
    ) -> (i64, i32, Launch) {
        let page_elems = page_size * num_kv_heads * head_dim;
        let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
        let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
        (logical_n, page_elems, Launch::grid([blocks as u32, 1, 1], [BLOCK, 1, 1]))
    }

    /// `attn::dequant_kv_cache_layer_to_bf16_active` — dequantise the pages
    ///
    /// As [`dequant_fp8_per_tensor_pages_active`].
    ///
    /// A native-bf16 layer is `Ok(())` HERE and `Refusal::Absent` in the
    /// per-scheme launchers below, and the difference is which question is
    /// being asked. This is the dispatcher: "stage the cache" over a cache
    /// that needs no staging is done, not declined. Both callers had already
    /// written that answer around the old `Err` -- `arms/fa2.rs`'s
    /// `dequant_prelude` with `let _ =`, `dequant_kv_active_arm` with an
    /// `is_native_bf16` branch -- so no caller ever wanted the decline.
    #[kernels_macros::routine]
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_kv_cache_layer_to_bf16_active(
        ctx: &Ctx,
        k_pages: Env<keys::KvKeys>,
        v_pages: Env<keys::KvValues>,
        k_scales: Env<keys::KvKeyScales>,
        v_scales: Env<keys::KvValueScales>,
        // The mirrors this pass WRITES. Bound as they stand, null included:
        // `keys::KvNativeBf16` on the same row announces the layer that has
        // none, so a refusal here would contradict a fact the row carries.
        k_bf16_pages: Env<keys::KvBf16Keys>,
        v_bf16_pages: Env<keys::KvBf16Values>,
        page_size: Env<keys::KvPageSize>,
        num_kv_heads: Env<keys::KvNumHeads>,
        head_dim: Env<keys::KvHeadDim>,
        block_size: Env<keys::KvBlockSize>,
        scheme: Env<keys::KvSchemeByte>,
        storage_dtype: Env<keys::KvStorageDtype>,
        is_native_bf16: Env<keys::KvNativeBf16>,
        kv_page_indices: Env<keys::KvPageIndices>,
        // The fire-wide page BOUND, which is what this pass covers. Not
        // `xqa`'s `max_pages_per_seq`, a per-request maximum the same table
        // happens to be able to approximate.
        num_pages_in_batch: Env<keys::KvPagesInBatch>,
    ) -> Result<(), Refusal> {
        if **is_native_bf16 {
            return Ok(());
        }
        let scheme = kv_scheme(scheme_byte(**scheme));
        let storage_dtype = kv_dtype(scheme_byte(**storage_dtype));
        let (logical_n, _page_elems, launch) = active_geometry(
            **page_size,
            **num_kv_heads,
            **head_dim,
            **num_pages_in_batch,
        );

        match scheme.scheme() {
            Some(KvScheme::Fp8PerTensor) => dequant_fp8_per_tensor_pages_active(
                ctx,
                **k_pages,
                **v_pages,
                **k_bf16_pages,
                **v_bf16_pages,
                **page_size,
                **num_kv_heads,
                **head_dim,
                scheme,
                storage_dtype,
                **is_native_bf16,
                **kv_page_indices,
                **num_pages_in_batch,
            ),

            // SAFETY, in all three arms: `call()`'s contract -- every pointer
            // bound here addresses live device memory of the extent the
            // kernel reads it as.
            Some(KvScheme::Fp8PerTokenHead) => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_fp8_per_token_head_pages_active<::pie::bf16>",
                    launch,
                    &[
                        (**k_pages).cast::<u8>().cast_const().arg(),
                        (**v_pages).cast::<u8>().cast_const().arg(),
                        (**k_scales).cast::<f32>().cast_const().arg(),
                        (**v_scales).cast::<f32>().cast_const().arg(),
                        (**k_bf16_pages).cast::<bf16>().arg(),
                        (**v_bf16_pages).cast::<bf16>().arg(),
                        (**kv_page_indices).arg(),
                        logical_n.arg(),
                        page_size.arg(),
                        num_kv_heads.arg(),
                        head_dim.arg(),
                    ],
                )
            },

            Some(KvScheme::Int8PerTokenHead) => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_int8_per_token_head_pages_active<::pie::bf16>",
                    launch,
                    &[
                        (**k_pages).cast::<i8>().cast_const().arg(),
                        (**v_pages).cast::<i8>().cast_const().arg(),
                        (**k_scales).cast::<f32>().cast_const().arg(),
                        (**v_scales).cast::<f32>().cast_const().arg(),
                        (**k_bf16_pages).cast::<bf16>().arg(),
                        (**v_bf16_pages).cast::<bf16>().arg(),
                        (**kv_page_indices).arg(),
                        logical_n.arg(),
                        page_size.arg(),
                        num_kv_heads.arg(),
                        head_dim.arg(),
                    ],
                )
            },

            Some(KvScheme::Fp4Block) => unsafe {
                ctx.launch(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_fp4_pages_active<::pie::bf16>",
                    launch,
                    &[
                        (**k_pages).cast::<u8>().cast_const().arg(),
                        (**v_pages).cast::<u8>().cast_const().arg(),
                        (**k_scales).cast::<f32>().cast_const().arg(),
                        (**v_scales).cast::<f32>().cast_const().arg(),
                        (**k_bf16_pages).cast::<bf16>().arg(),
                        (**v_bf16_pages).cast::<bf16>().arg(),
                        (**kv_page_indices).arg(),
                        logical_n.arg(),
                        page_size.arg(),
                        num_kv_heads.arg(),
                        head_dim.arg(),
                        fp4_block_size(**block_size).arg(),
                    ],
                )
            },

            Some(KvScheme::Native) => {
                Err(Refusal::Absent { what: "a quantised dequant for Native storage" })
            }

            None => Err(Refusal::Absent { what: "a KV scheme this byte names" }),
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
///
/// The derived column is `[Out(0), OutElements(0)]`, and **this row is
/// crossed**: `arms/attn.rs`'s row names `table::derived_arm`.
///
/// `n: usize` against an `operand()` that mints `ArgValue::I32` is not a
/// mismatch a signature has to fix: a `Source` says WHICH FACT, not how wide.
/// `operands` runs `as_declared` over each value with this launcher's own
/// `Routine::args` in hand (`bind/table.rs:1062`), converting the `I32` to a
/// `Usize` and refusing rather than wrapping where that would lose
/// information.
#[kernels_macros::routine]
pub fn lse_log2_to_ln(
    ctx: &Ctx,
    lse: Out<0, f32>,
) -> Result<(), Refusal> {
    // `Region::elements()` IS this product, and it is safe there for the
    // reason it was guarded here: a view cannot have a zero width, so the
    // multiply cannot collapse an unstated pitch into a stated zero.
    let elems = lse.all("out_width(0)")?.elements();
    let Ok(elems) = u32::try_from(elems) else {
        return Err(Refusal::Empty { what: "lse elements" });
    };
    let n = elems as usize;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attn_sink.cuh",
            "::pie::attn::lse_log2_to_ln<::pie::attn::f32>",
            elementwise(elems),
            &[lse.ptr.arg(), n.arg()],
        )
    }
}

/// `attn::attention_sink_rescale_bf16` — gpt-oss's per-head sink correction,
///
/// What the caller must guarantee, as `call()` states it: `o` addresses `n *
/// num_q_heads * head_dim` live, writable bf16 elements; `lse` addresses `n *
/// num_q_heads` live `f32`s; `sinks` addresses `num_q_heads` live bf16
/// elements.
///
/// The derived column is `[Out(0), In(0), Weight(0), Rows, NumQHeads,
/// HeadDim]`, `alias()` shifts the `In(0)` to `In(1)` past the in-place pair,
/// and every entry resolves. **This row is crossed**: the column reads, in
/// order, `arg_out(0)`, `arg_in(1)`, `weight(0)`, `rows().count`,
/// `num_q_heads()`, `head_dim()` -- which is what the deleted
/// `attention_sink_rescale_arm` did, argument for argument.
///
/// The shift is the part worth stating, because it is the only thing here
/// that is not a transcription. The row declares `in_place = &[(0, 0)]`, the
/// column names ONE `In(_)` and the statement has two inputs, so `operands`
/// finds `named < o.ins.len()` and applies `alias()` -- which is exactly the
/// condition that guard was added for. Were the column ever to gain a second
/// `In(_)`, the shift would stop and this binding would silently change; the
/// `lse` parameter below says so at the parameter.
#[kernels_macros::routine]
pub fn attention_sink_rescale<T>(
    ctx: &Ctx,
    // `Out(0)`.
    o: Out<0, T>,
    lse: In<1, f32>,
    sinks: Bank<0, T>,
    num_q_heads: Env<keys::NumQHeads>,
    head_dim: Env<keys::HeadDim>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
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
                o.rows.unsigned_abs(),
                num_q_heads.unsigned_abs(),
                head_dim.unsigned_abs(),
            ),
            &[
                o.ptr.arg(),
                lse.ptr.arg(),
                sinks.ptr.arg(),
                o.rows.arg(),
                num_q_heads.arg(),
                head_dim.arg(),
            ],
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
#[kernels_macros::routine]
pub fn split_qkv_bf16_devwin(
    ctx: &Ctx,
    packed: In<0, bf16>,
    q_out: Out<0, bf16>,
    k_out: Out<1, bf16>,
    v_out: Out<2, bf16>,
    win: Env<keys::PeelWindow>,
    n_max: Env<keys::RowsTotal>,
) -> Result<(), Refusal> {
    /// `split_packed.cu:30` — `constexpr int BLOCK = 256;`.
    pub const SPLIT_BLOCK: u32 = 256;

    // Both are EXTENTS the kernel loops to (`split_packed.cuh:122`, `:126`)
    // before it strides by them, so both are widths; the packed row's pitch
    // -- `q_dim + 2 * kv_dim` at `:120` -- is the SOURCE's and belongs to no
    // one operand, which is why it is still computed on the device.
    let (q_dim, kv_dim) =
        (q_out.all("out_width(0)")?.width, k_out.all("out_width(1)")?.width);
    let max_dim = if q_dim > kv_dim { q_dim } else { kv_dim };
    let xblocks = max_dim.unsigned_abs().div_ceil(SPLIT_BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/split_packed.cuh",
            "::pie::attn::split_qkv_devwin<::pie::bf16>",
            Launch::grid([xblocks.max(1), (**n_max).unsigned_abs(), 1], [SPLIT_BLOCK, 1, 1]),
            &[
                packed.ptr.arg(),
                q_out.ptr.arg(),
                k_out.ptr.arg(),
                v_out.ptr.arg(),
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
///
/// A `routine!` row since the `&KvLayer` unfolded into its ten leaves. The
/// aggregate was the only thing keeping it `driver_bound!`, since `Arg` is
/// implemented for no host struct.
#[kernels_macros::routine]
pub fn attention_naive_paged(
    ctx: &Ctx,
    k_pages: Env<keys::KvKeys>,
    v_pages: Env<keys::KvValues>,
    k_scales: Env<keys::KvKeyScales>,
    v_scales: Env<keys::KvValueScales>,
    page_size: Env<keys::KvPageSize>,
    head_dim: Env<keys::KvHeadDim>,
    num_kv_heads: Env<keys::KvNumHeads>,
    scheme: Env<keys::KvSchemeByte>,
    storage_dtype: Env<keys::KvStorageDtype>,
    block_size: Env<keys::KvBlockSize>,
    q: In<0, bf16>,
    o: Out<0, bf16>,
    // `plan.qo_indptr` = `keys::QoIndptr`. The note that stood here said
    // *"`fact_of` has no key for `qo_indptr`, so it derives `None` unaided"*
    // -- both halves are dead: `fact_of` is gone (the name table with it) and
    // the key exists, so this derives `Some(Named("plan.qo_indptr"))`.
    qo_indptr: Env<keys::QoIndptr>,
    // WHAT OWNS THE THIRD: `Cx::plan`, the fire's page plan, same as the two
    // above it.
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    num_requests: Env<keys::RequestCount>,
    window_left: Env<keys::WindowLeft>,
    // `cx.sm_scale()`, and `keys::SmScale` is the word for it.
    sm_scale: Env<keys::SmScale>,
    logits_soft_cap: Env<keys::AttnLogitsSoftCap>,
    lse_out: Env<keys::AttnLseOut>,
) -> Result<(), Refusal> {
    /// `attention_naive_paged.cuh:223` — `constexpr int kMaxHeadDim = 1024`.
    pub const PAGED_MAX_HEAD_DIM: i32 = 1024;

    /// `attention_naive_paged.cuh:33` — `constexpr int BLOCK = 128`.
    pub const PAGED_BLOCK: u32 = 128;

    if **head_dim > PAGED_MAX_HEAD_DIM {
        return Err(Refusal::Wide {
            what: "head_dim",
            at: i64::from(**head_dim),
            max: i64::from(PAGED_MAX_HEAD_DIM),
        });
    }
    let src = q.all("in_width(0)")?;
    let num_q_heads = src.width.checked_div(**head_dim).unwrap_or(0);
    let smem = ((**head_dim).unsigned_abs() + PAGED_BLOCK) * 4;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attention_naive_paged.cuh",
            "::pie::attn::naive_paged_attn<::pie::i32(128)>",
            Launch::grid(
                [
                    num_requests.unsigned_abs(),
                    src.rows.unsigned_abs(),
                    num_q_heads.unsigned_abs(),
                ],
                [PAGED_BLOCK, 1, 1],
            )
            .smem(smem),
            &[
                q.ptr.arg(),
                (**k_pages).cast_const().arg(),
                (**v_pages).cast_const().arg(),
                (*k_scales).cast::<f32>().cast_const().arg(),
                (*v_scales).cast::<f32>().cast_const().arg(),
                o.ptr.arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                core::ptr::null::<u8>().arg(),
                core::ptr::null::<i32>().arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                page_size.arg(),
                kv_scheme(scheme_byte(**scheme)).arg(),
                kv_dtype(scheme_byte(**storage_dtype)).arg(),
                block_size.arg(),
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
///
/// The derived column is `[In(0), In(1), In(2), In(3), Out(0), RmsEps]`. The
/// two weights are `arg_in(2)` and `arg_in(3)` here and NOT `weight(n)`, which
/// is why they take a positional `In` and not a `Weight` mark.
#[kernels_macros::routine]
pub fn attn_res_blend<T>(
    ctx: &Ctx,
    prefix: In<0, T>,
    blocks: In<1, T>,
    norm_weight: In<2, T>,
    proj_weight: In<3, T>,
    out: Out<0, T>,
    // # AND NOW THE DIVISION IS HERE, WHICH IS THE THIRD PLACE IT HAS LIVED
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let dst = out.all("out_width(0)")?;
    let h = dst.width;
    // AND THE REGION'S OWN STRIDE IS NOT `blocks`' PITCH. `attn_res.cuh:114`
    // addresses it `(j * block_rows + t) * H` -- BLOCK-major -- so the
    // view's `b * h` row is a flattening of a plane the kernel walks in two
    // steps, and only the quotient `b` is a fact about it. `h > 0` here
    // because `dst` was built, which is what makes the divide legal.
    let b = blocks.all("in_width(1)")?.width / h;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/attn_res.cuh",
            &format!("::pie::attn::attn_res_blend<{}>", T::CPP),
            Launch::per_row(dst.rows.unsigned_abs(), BLOCK),
            &[
                prefix.ptr.arg(),
                blocks.ptr.arg(),
                norm_weight.ptr.arg(),
                proj_weight.ptr.arg(),
                out.ptr.arg(),
                b.arg(),
                h.arg(),
                dst.rows.arg(),
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
#[kernels_macros::routine]
pub fn pad_head_dim<T>(
    ctx: &Ctx,
    packed: In<0, T>,
    padded: Out<0, T>,
    head_dim: Env<keys::HeadDim>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let num_heads = packed.width.checked_div(**head_dim).unwrap_or(0);
    let head_dim_padded = padded.width.checked_div(num_heads).unwrap_or(0);
    if let Some(why) = head_dim_refusal(packed.rows, num_heads, **head_dim, head_dim_padded) {
        return Err(why);
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/head_dim_pad.cuh",
            &format!("::pie::attn::pad_head_dim<{}>", T::CPP),
            per_head(packed.rows.unsigned_abs(), num_heads.unsigned_abs()),
            &[
                packed.ptr.arg(),
                padded.ptr.arg(),
                num_heads.arg(),
                head_dim.arg(),
                head_dim_padded.arg(),
            ],
        )
    }
}

/// `attn::strip_head_dim_bf16` — the inverse of [`pad_head_dim`].
///
/// What the caller must guarantee, as `call()` states it: `padded` addresses
/// `num_tokens * num_heads * head_dim_padded` live bf16 elements and `packed`
/// addresses `num_tokens * num_heads * head_dim` writable ones.
#[kernels_macros::routine]
pub fn strip_head_dim<T>(
    ctx: &Ctx,
    padded: In<0, T>,
    packed: Out<0, T>,
    // Now `packed.width.checked_div(head_dim)`, in the body, for the reasons
    // written out at `pad_head_dim`'s.
    head_dim: Env<keys::HeadDim>,
    // Now `padded.width.checked_div(num_heads)`.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let num_heads = packed.width.checked_div(**head_dim).unwrap_or(0);
    let head_dim_padded = padded.width.checked_div(num_heads).unwrap_or(0);
    if let Some(why) = head_dim_refusal(padded.rows, num_heads, **head_dim, head_dim_padded) {
        return Err(why);
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/head_dim_pad.cuh",
            &format!("::pie::attn::strip_head_dim<{}>", T::CPP),
            per_head(padded.rows.unsigned_abs(), num_heads.unsigned_abs()),
            &[
                padded.ptr.arg(),
                packed.ptr.arg(),
                num_heads.arg(),
                head_dim.arg(),
                head_dim_padded.arg(),
            ],
        )
    }
}

// ── WHERE `width_of` WENT ──────────────────────────────────────────────
//
// A private `fn width_of(width, what) -> Result<i32, Refusal>` stood here and
// fifteen calls at twelve sites in this file went through it. It is now
// [`kernels::routine::In::all`] / [`Out::all`], which makes the same `width
// <= 0` test at the point a VIEW is built rather than at each reader
// (`kernels/src/routine.rs` -- "THE VIEW API").
//
// The reason it existed is unchanged and worth keeping where the sites are: a
// region's width is minted `out_width(n).unwrap_or(0)`
// (`bind/table.rs:1233`) and deliberately does not refuse -- the region is an
// operand, and an operand's shape is reported, not vetted. So a missing width
// arrives as a ZERO, and a zero head dim launches a kernel that strides
// nothing rather than declining to launch.
//
// `ctx.launch` catches only the subset that reaches a grid extent
// (`jit/ctx.rs`'s `Refusal::Empty { what: "the grid" }`), which is the
// wrong word in the wrong place: it names the grid for a fault in an operand,
// and it does not fire at all where the width is only an ARGUMENT or where a
// `.max(1)` stands between it and the grid (`dsv4_compress::route_rows` is
// exactly that). `all()` takes the caller's word instead of inventing one, so
// every site below still refuses in `width()`'s own -- `"out_width(0)"`,
// `"in_width(1)"` -- and the messages did not merge.
//
// TWO LAUNCHERS STILL BUILD NO VIEW, and that is the paragraph on
// [`head_dim_refusal`]: `pad_head_dim` and `strip_head_dim` never called
// `width_of` either, because their width assertions survive transitively
// through that guard in ITS words. Routing them through `all()` would be a
// different refusal for the same fire.

/// The four preconditions both head-dim launchers share, resolved BEFORE
///
/// # ITS PREMISE WAS RE-CHECKED UNDER `unwrap_or(0)` INPUTS AND IT HOLDS
///
/// This is the only guard in the family kept on the strength of text written
/// against the OLD signature, so it is the only one that had to be re-derived
/// rather than trusted. An existing guard is evidence about what the previous
/// signature owed, not about what this one does.
///
/// The two launchers deleted `Div(&InWidth(0), &HeadDim)` and
/// `Div(&OutWidth(0), &Div(..))` (`pad_head_dim`; `strip_head_dim` has the
/// two indices the other way round, `padded` being its `In<0>` and `packed`
/// its `Out<0>`). `operand()` walks a `Div`'s children through itself with
/// `?`, so BOTH marks asserted `in_width(0) > 0` AND `out_width(0) > 0`.
/// Those two assertions are what a region drops.
///
/// They survive here transitively, and each step matters:
///
/// * `head_dim <= 0` fires first, so `head_dim > 0` holds below it.
/// * `num_heads` is `packed.width.checked_div(head_dim)`. A zero `packed`
///   width makes it zero, and `num_heads <= 0` refuses. So the packed-side
///   width assertion survives.
/// * `head_dim_padded` is `padded.width.checked_div(num_heads)`, and
///   `num_heads > 0` by the line above. A zero `padded` width makes it zero,
///   and `0 < head_dim` -- head_dim being positive -- refuses `Narrow`. So
///   the padded-side width assertion survives too.
///
/// Nothing here is a sum, a `.max`, or an `.unwrap_or` that a zero can walk
/// through: each check is `<= 0` or `<` against a value already proved
/// positive. That is why these two launchers build no view and the other
/// eleven bodies do. Only the WORD differs from what `width()` would have
/// said, and the note on `pad_head_dim`'s first `Div` already trades that
/// away in `bind/table.rs:1370`'s terms.
///
/// SO THE FOUR `.width` READS IN THE TWO LAUNCHERS ABOVE STAY BARE, and they
/// are the only reads of a WRAPPER's width field left in this file.
/// `packed.all(..)` would refuse `Absent { what: .. }` where this refuses
/// `Empty { what: "num_heads" }` for the same fire -- a re-wording of a live
/// refusal, which is a change no `cargo check` sees.
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

/// `Source::Slot(Kind::OutElements, 0)` over a region, for the two softcap launchers.
///
/// `bind/table.rs:1296` mints that variant as
/// `f.rows.count.saturating_mul(out_width(n))`, and a region's two fields are
/// those two factors read from the same places (`bind/table.rs:1233`) --
/// which is exactly [`kernels::routine::Region::elements`], so the product is
/// no longer written here at all. The `usize` is what `softcap.cuh` declares
/// and what the launch list pushes.
///
/// Written once because BOTH softcap launchers need it and the f16 one has no
/// arm to have inherited it from -- see the note on its `x`.
///
/// The view is built first, for `lse_log2_to_ln`'s reason and with its
/// correction: `OutElements(n)` maps over `o.out_width(n)`, which is
/// `width()` (`bind/table.rs:612`, `:640-643`), so the variant ASSERTED
/// `out_width(0) > 0` and not merely that a width was present. A region's
/// `unwrap_or(0)` drops that assertion whole. The zero would still be refused
/// downstream -- as an empty grid, about the grid -- and `all()` refuses it
/// here, about the width, in `width()`'s word. That ordering is also what
/// makes `elements()` safe: it multiplies a width a view has already proved
/// non-zero, so it cannot fold an unstated pitch into a stated zero.
fn softcap_elems<T: kernels::Elem>(x: &Out<0, T>) -> Result<usize, Refusal> {
    let elems = x.all("out_width(0)")?.elements();
    usize::try_from(elems)
        .map_err(|_| Refusal::Narrow { what: "logit elements", at: i64::from(elems) })
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
#[kernels_macros::routine]
pub fn logit_softcap<T>(
    ctx: &Ctx,
    x: Out<0, T>,
    cap: Env<keys::FinalLogitSoftcap>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    let cap = **cap;
    let n = softcap_elems(&x)?;
    let launch = softcap_launch(cap, n)?;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/softcap.cuh",
            &format!("::pie::attn::logit_softcap<{}>", T::CPP),
            launch,
            &[x.ptr.arg(), cap.arg(), n.arg()],
        )
    }
}

/// `attn::logit_softcap_f16` — the same cap over an fp16 buffer.
///
/// [`logit_softcap`]'s obligation, with `f16` for `bf16`.
///
/// **No `Bound` row names this symbol.** The column below is derived from the
/// signature and from [`logit_softcap`]'s arm, which is the same launch over
/// a different element type; nothing else was assumed.
#[kernels_macros::routine]
pub fn logit_softcap_f16(
    ctx: &Ctx,
    x: Out<0, f16>,
    cap: Env<keys::FinalLogitSoftcap>,
) -> Result<(), Refusal> {
    let cap = **cap;
    let n = softcap_elems(&x)?;
    let launch = softcap_launch(cap, n)?;
    // SAFETY: as [`logit_softcap`]'s.
    unsafe {
        ctx.launch(
            "attn/softcap.cuh",
            "::pie::attn::logit_softcap<::pie::f16>",
            launch,
            &[x.ptr.arg(), cap.arg(), n.arg()],
        )
    }
}

/// `attn::kimi_split_q_b_bf16` — split a fused query projection into its
#[kernels_macros::routine]
pub fn kimi_split_q_b<T>(
    ctx: &Ctx,
    q_b: In<0, T>,
    q_nope: Out<0, T>,
    q_pe: Out<1, T>,
    // The split sizes are statement params, not model facts: the wrapper
    // states the exact `Kind::Param` slots the old arm read with
    // `cx.param(0..=2)`, and must not consume an input or output counter.
    heads: Param<0, i32>,
    nope: Param<1, i32>,
    rope: Param<2, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let width = i64::from(*heads) * (i64::from(*nope) + i64::from(*rope));
    let total = i64::from(q_b.rows) * width;
    if total > i64::from(i32::MAX) {
        return Err(Refusal::Wide {
            what: "rows",
            at: i64::from(q_b.rows),
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
                q_b.ptr.arg(),
                q_nope.ptr.arg(),
                q_pe.ptr.arg(),
                total.arg(),
                heads.arg(),
                nope.arg(),
                rope.arg(),
            ],
        )
    }
}

/// `attn::kimi_split_kv_a_norm_bf16` — split `kv_a`, RMS-normalise the latent
///
/// The derived column is `[In(0), Weight(0), Out(0), Out(1), RmsEps]`, every
/// entry resolving, and the row is crossed.
#[kernels_macros::routine]
pub fn kimi_split_kv_a_norm<T>(
    ctx: &Ctx,
    kv_a: In<0, T>,
    norm_weight: Bank<0, T>,
    kv_c: Out<0, T>,
    k_pe: Out<1, T>,
    // The guard below therefore compares the same three numbers it always
    // did, and refuses in the same word.
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    /// `LaunchRule::Rms`, as the expression it evaluates to.
    #[must_use]
    const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
    }

    // THE ONE SITE IN THIS FAMILY WHERE A WIDTH WAS ALREADY CALLED A PITCH.
    // `kimi_mla.cuh:110` addresses `kv_a + n * src_row_stride` and never
    // loops to it -- `kv_lora` and `rope` are the extents -- so this is the
    // ALLOCATION's number and `Stride` is what says so. It was `kv_a`'s own
    // width read under an assumed packing (§3.1's "packed, ASSUMED"); a
    // region carries the pitch as a field, so the assumption is now stated.
    //
    // Built in the tuple and in this order because the three refusals are
    // ordered: a fire with two zero widths must still name `out_width(0)`.
    let (kv_lora, rope, src) = (
        kv_c.all("out_width(0)")?.width,
        k_pe.all("out_width(1)")?.width,
        kv_a.all("in_width(0)")?,
    );
    let src_row_stride = src.stride;
    if *src_row_stride < kv_lora + rope {
        return Err(Refusal::Narrow { what: "src_row_stride", at: i64::from(*src_row_stride) });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "attn/kimi_mla.cuh",
            &format!("::pie::attn::split_kv_a_norm<{}, 256>", T::CPP),
            rms(src.rows.unsigned_abs()),
            &[
                kv_a.ptr.arg(),
                norm_weight.ptr.arg(),
                kv_c.ptr.arg(),
                k_pe.ptr.arg(),
                kv_lora.arg(),
                rope.arg(),
                src_row_stride.0.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `attn::combine_attn_outputs_bf16` — merge two attention halves and their
///
/// What the caller must guarantee, as `call()` states it: every pointer must
/// address the extents these regions and two params describe. `num_heads` and
/// `head_dim` are NOT `keys::NumQHeads`/`keys::HeadDim`; they are the
/// statement's param slots 0 and 1, made explicit so the derived column can
/// bind the same scalars the driver arm used to read.
#[kernels_macros::routine]
pub fn combine_attn_outputs<T>(
    ctx: &Ctx,
    o1: In<0, T>,
    lse1: In<1, f32>,
    o2: In<2, T>,
    lse2: In<3, f32>,
    o_out: Out<0, T>,
    lse_out: Out<1, f32>,
    num_heads: Param<0, i32>,
    head_dim: Param<1, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
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
            combine_attn(
                o_out.rows.unsigned_abs(),
                num_heads.unsigned_abs(),
                head_dim.unsigned_abs(),
            ),
            &[
                o1.ptr.arg(),
                lse1.ptr.arg(),
                o2.ptr.arg(),
                lse2.ptr.arg(),
                o_out.ptr.arg(),
                lse_out.ptr.arg(),
                num_heads.arg(),
                head_dim.arg(),
            ],
        )
    }
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated beside one is what no signature carries:
/// whether a statement consumes its whole operand, and which operands must be
/// given the same address.
///
/// **Two layer-view host programs are DECLARED and are not `routine!`s.**
/// `mla_prepare_bf16` and `write_mla_to_pages` take an `MlaLayer`. That is a
/// host aggregate rather than a kernel argument and `Arg` is implemented for
/// no such type, so a `routine!` naming one does not compile.
///
/// **It was four, and no `KvLayer` is left.** `attention_naive_paged` read ten
/// fields through the view and takes the ten as parameters now, at arity 21;
/// `dequant_kv_cache_layer_to_bf16_active` read thirteen and takes fifteen
/// leaves. Both left `driver_bound!` for `routine!` in the same move. That is
/// the general shape: a `driver_bound!` row is held by an aggregate, and an
/// aggregate whose leaves are each expressible unfolds. The ceiling is 36, so
/// the arity was never what stood in the way -- the missing WORDS were, and
/// `keys::KvBf16Keys`/`KvBf16Values`/`KvPagesInBatch` were the last three.
///
/// **Absent on purpose, so nobody goes looking for the gap:**
///
/// * `dequant_fp8_per_tensor_pages_active` is one arm of the dequantiser's
///   `match`, and the fp8 kind is a property of the layer, not the statement.
///   It takes leaves rather than the view for the same reason its caller does:
///   there is no aggregate left to forward.
/// * Three of [`attention_score_post`]'s four launches are fired by
///   `driver-cuda`'s `fire::attn_score` on the fire's own stream and were
///   never statements; a `routine!` for one would put a row in `crate::sigs()`
///   that nothing could lower to. The fold is the exception, and is declared
///   below because `dsl::cuda::attn_score_fold_heads` states the CONTRACT
///   symbol, which is a different symbol from the `_dev` device one.
/// * [`mla_fa2::fire`] and [`mla_naive::fire`] are ARMS, not symbols: which
///   one runs is a property of the device. A trace names the `fn` that
///   chooses, [`dispatch_attention_mla_bf16`], declared below.
///
/// # `derived =`, on every `routine!` row and no `driver_bound!` one
///
/// Each column reaches its row through `<FN as kernels::Derivation>::DERIVED`,
/// the impl `#[kernels_macros::routine]` emits beside the launcher, read off
/// that symbol's arm in
/// `driver-cuda/src/bind/arms/attn.rs`. `driver_bound!` rows carry none:
/// their bodies refuse every argument list, so `args: &[]` leaves the column
/// nothing to be checked beside.
///
/// `qkv_decode_fused_dispatch` carries no attribute at all, and this is the
/// one row where that is deliberate rather than pending. It is an inner LEG:
/// one caller, [`qkv_decode_qk_norm_rope_write_kv_bf16`], which has already
/// bound every operand through its own column. A derived column here would
/// resolve end to end and claim a binding no statement made, and fourteen of
/// its fifteen pointers would disagree with the caller's arm-witnessed
/// indices -- the leg counts `rope_table` at `In(4)` where the caller states
/// `In(1)`, because the caller's two `Bank` weights and its `positions` fact
/// are not operands down here. Marking all fifteen `Unbound<_>` is the trap:
/// `packed` and `rope_table` really ARE statement operands on the caller, so
/// `Unbound` would deny an index the value genuinely has. There is no honest
/// column for a leg; `NO_COLUMN_ON_PURPOSE` in `driver-cuda` is where the
/// exemption lives.
pub static ROUTINES: &[Routine] = &[
    routine!(lse_log2_to_ln, in_place = &[(0, 0)], ),
    routine!(
        attention_sink_rescale_bf16 = attention_sink_rescale::<bf16>,
        in_place = &[(0, 0)]
    ),
    routine!(attn_res_blend_bf16 = attn_res_blend::<bf16>, ),
    routine!(pad_head_dim_bf16 = pad_head_dim::<bf16>, ),
    routine!(strip_head_dim_bf16 = strip_head_dim::<bf16>, ),
    routine!(
        logit_softcap_bf16 = logit_softcap::<bf16>,
        in_place = &[(0, 0)]
    ),
    routine!(logit_softcap_f16, in_place = &[(0, 0)], ),
    routine!(kimi_split_q_b_bf16 = kimi_split_q_b::<bf16>, ),
    routine!(
        kimi_split_kv_a_norm_bf16 = kimi_split_kv_a_norm::<bf16>
    ),
    routine!(
        combine_attn_outputs_bf16 = combine_attn_outputs::<bf16>
    ),
    routine!(split_qkv_bf16_devwin, ),
    routine!(compact_page_csr, whole, ),
    routine!(
        mtp_shift_hidden_bf16 = mtp_shift_hidden::<bf16>,
        whole
    ),
    routine!(
        mtp_update_pending_hidden_bf16 = mtp_update_pending_hidden::<bf16>,
        whole
    ),
    routine!(
        dsa_index_knorm_rope_bf16 = dsa_index_knorm_rope::<bf16>,
        in_place = &[(0, 0)]
    ),
    // `index_q_rope` does exactly the same thing to `idx_q`:
    // `dsa_indexer.cuh:156-158` reads `row[d]` into a register buffer,
    // rotates it with `rope_interleave_inplace`, and writes `row[d]` back --
    // one buffer read and written, and the launcher takes ONE pointer for
    // both.
    routine!(
        dsa_index_q_rope_bf16 = dsa_index_q_rope::<bf16>,
        in_place = &[(0, 0)]
    ),
    routine!(dsa_index_topk_mask, whole, ),
    routine!(dsv4_boundary_meta_decode, ),
    routine!(dsv4_boundary_meta_paged, whole, ),
    routine!(
        attention_compressed_paged_bf16,
        whole
    ),
    routine!(
        dsv4_compress_gather_paged_bf16
    ),
    routine!(
        dsv4_store_comp_entries_bf16,
        whole
    ),
    routine!(
        qkv_packed_qk_norm_rope_vnorm_write_kv_bf16
    ),
    routine!(
        qkv_decode_qk_norm_rope_write_kv_bf16
    ),
    routine!(qkv_decode_fused_dispatch, uncolumned),
    // ── THE PAGED-KV APPENDERS, WHICH CROSSED THE LINE BELOW ────────────
    routine!(write_kv_to_pages_bf16),
    routine!(
        write_kv_to_pages_quantised
    ),
    routine!(write_kv_explicit_bf16),
    routine!(
        write_kv_explicit_bf16_devwin,
        whole
    ),
    // ── what the DRIVER fires, by path ──────────────────────────────────
    driver_bound!(write_kv_to_pages),
    routine!(dequant_kv_cache_layer_to_bf16_active),
    routine!(attention_naive_paged, whole, ),
    driver_bound!(attn_score_fold_heads, whole),
    driver_bound!(mla_prepare_bf16, whole),
    driver_bound!(write_mla_to_pages, whole),
    driver_bound!(dispatch_attention_mla_bf16),
    // ── and one whose body is `driver_internal`'s ───────────────────────
    routine!(split_qkv_bf16, ),
];

/// `attn`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

// ── what a statement cannot supply, for this family ──────────────────

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

// OPERAND SLOTS DID NOT MOVE: making `kimi_split_q_b`'s three split sizes
// `Param<N, i32>` must not advance the `In`/`Out` counters. This is rule F4;
// a `Weight<1, _>` index bug of exactly this shape shipped once and was
// reverted, so the check lives in `cargo check` rather than in a test binary.
const _: () = {
    let d = <kimi_split_q_b as kernels::Derivation>::DERIVED;
    assert!(d.len() == 6);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
};

// OPERAND SLOTS DID NOT MOVE: making `dsa_index_topk_mask`'s three scalar
// sizes `Param<N, i32>` must not consume an operand slot before the mask. This
// pins rule F4 at the old arm's boundary, where params followed three inputs
// and one output.
const _: () = {
    let d = <dsa_index_topk_mask as kernels::Derivation>::DERIVED;
    assert!(d.len() == 7);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 2))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
};

// OPERAND SLOTS DID NOT MOVE: making `combine_attn_outputs`'s merge shape
// `Param<N, i32>` must leave the four inputs and two outputs numbered as the
// arm wrote them. The params state their slots without shifting the region
// counters.
const _: () = {
    let d = <combine_attn_outputs as kernels::Derivation>::DERIVED;
    assert!(d.len() == 8);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 2))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::In, 3))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[7].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
};

// `Region::elements()` IS THE PRODUCT THE TWO ELEMENT-COUNT LAUNCHERS USED TO
// WRITE, SATURATION INCLUDED. `lse_log2_to_ln` and `softcap_elems` both read
// `width_of(x.width, ..)?.saturating_mul(x.rows)` and now read
// `x.all(..)?.elements()`; `softcap_elems` then hands the result to
// `usize::try_from`, so a product that WRAPPED where the old one saturated
// would turn a `Refusal::Narrow` about too many logits into a negative that
// converts to nothing and a launch over the wrong `n`. Neither factor is a
// const at a call site, so the arithmetic is pinned here on a region built by
// hand rather than at the readers.
const _: () = {
    let r = kernels::routine::Region {
        ptr: core::ptr::null_mut::<f32>(),
        rows: 7,
        width: 6,
        stride: kernels::routine::Stride(6),
    };
    assert!(r.elements() == 42);

    let wide = kernels::routine::Region {
        ptr: core::ptr::null_mut::<f32>(),
        rows: i32::MAX,
        width: 2,
        stride: kernels::routine::Stride(2),
    };
    assert!(wide.elements() == i32::MAX);
};
// ── THE ROW THAT CROSSED IN KILIMANJARO III STAGE 3 ────────────────────
//
// `qkv_packed_post_arm` is deleted and
// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` reads
// `Bound::derived`, so this column is now the ONLY description of how a fire
// reaches this kernel. Nothing checks it at run time before the launch: a
// wrong entry binds a live pointer of the right type from the wrong place.
//
// SEVENTEEN, AND THE COUNT IS THE FIRST ASSERTION FOR A REASON. The arm
// passed eighteen arguments; `num_q_heads` left because F6 puts the division
// in the launcher, and `ctx: &Ctx` is not in the derived column. A parameter
// added without a thought for the binder shows up here and nowhere else.
//
// THE TWO POSITIONAL RUNS ARE PINNED INDEPENDENTLY. `stated_source`
// (`kernels-macros/src/lib.rs:271`) SETS a counter to `N + 1` rather than
// bumping it, so a wrapper leaving the middle of a signature can renumber a
// later one silently: `Bank<1, _>` at `d[5]` and `In<0, _>` at `d[0]` are the
// two ends of that risk. The banks are the pair `qkv_fused.cu` uses as
// q-norm and k-norm weights -- swapping them normalises Q by K's gains, which
// is numerically plausible and wrong.
//
// `keys::RowValid` AT `d[10]` IS THE STAGE'S OWN ENTRY. It said `None` --
// "no source names this" -- for as long as this signature existed, and that
// `None` was the whole reason the arm could not go. `source_is_named`
// compares the key's `&'static str`, so this fires if the key is renamed
// without `operand()` being taught the new string.
const _: () = {
    let d = <qkv_packed_qk_norm_rope_vnorm_write_kv_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 17);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 1))));
    assert!(kernels::source_is_named(&d[2].source, <kernels::keys::KvKeys as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[3].source, <kernels::keys::KvValues as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::Positions as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::KvPageIndices as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::KvPageIndptr as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[9].source, <kernels::keys::KvLastPageLens as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::RowValid as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[11].source, <kernels::keys::KvNumHeads as kernels::keys::Fact>::KEY));
    // `KvHeadDim` AND NOT `HeadDim`, which is the one substitution in this
    // column that would type-check, bind, and be wrong on a two-kind family:
    // the LAYER's width is `cx.kv_layer()?.head_dim` and the FIRE's is
    // `cx.head_dim()`, and `bind/table.rs` names gemma-4 as where they part.
    assert!(kernels::source_is_named(&d[12].source, <kernels::keys::KvHeadDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[13].source, <kernels::keys::KvPageSize as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[14].source, <kernels::keys::KvHndLayout as kernels::keys::Fact>::KEY));
    // `Theta` AND NOT `RopeTheta`: the statement's layer with a fallback,
    // where `mla_prepare_bf16` two hundred lines up takes the fire's field.
    assert!(kernels::source_is_named(&d[15].source, <kernels::keys::Theta as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[16].source, <kernels::keys::RmsEps as kernels::keys::Fact>::KEY));
};

// ── THE DECODE TWIN, WHICH DID NOT CROSS, AND THE PARAMETER THAT HOLDS IT ──
//
// Same conversion, one parameter short of the same outcome: `q_out` is
// `cx.q_out()`, the decode fire's scratch destination, which no key names and
// no statement places. This pin says WHICH index is the refusal, so that a
// later stage that gives `q_out` a home finds out here whether it moved
// anything else.
//
// `d[7]` IS THE ONE TO WATCH. `rope_table` is the only bare region left in
// the middle of a run of `Env`s, and it is `In(1)` because `stated_source`
// short-circuits ahead of both counters for the two `Bank`s above it. Had
// those been left bare, the column would have claimed `In(1)` and `In(2)` for
// them and this would read `In(3)` -- binding the rope table to an operand
// the statement never placed.
const _: () = {
    let d = <qkv_decode_qk_norm_rope_write_kv_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 20);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // The region form states no result, so `Op::dest` carries the destination
    // and it lands in the operand run as `Out(0)`.
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 1))));
    assert!(matches!(d[7].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    // The `_or_null` pair, not `keys::KvWritePage`/`KvWriteOffset`: those
    // null-check, which `write_kv_explicit_bf16` wants and this row does not.
    assert!(kernels::source_is_named(
        &d[11].source,
        <kernels::keys::KvWritePageOrNull as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[12].source,
        <kernels::keys::KvWriteOffsetOrNull as kernels::keys::Fact>::KEY
    ));

    assert!(kernels::source_is_named(&d[13].source, <kernels::keys::RowValid as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[15].source, <kernels::keys::KvHeadDim as kernels::keys::Fact>::KEY));
};

// ── THE BF16 APPEND ──
//
// d[9]/d[10] BIND A NULL AND THAT IS THE POINT. `abi.rs:101` defines
// `has_envelopes()` as exactly "neither plane is null", so `operand()`
// refusing on a null would refuse precisely the layers d[17] -- on this same
// row -- reports `false` for, and the only two reads of the planes
// (`:3457` here, `:3285` in `write_kv_explicit_bf16`) are guarded on it.
const _: () = {
    let d = <write_kv_to_pages_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 18);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::RowValid as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[9].source, <kernels::keys::KvEnvMin as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::KvEnvMax as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[11].source, <kernels::keys::RequestCount as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[12].source, <kernels::keys::FirstToken as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[17].source, <kernels::keys::KvHasEnvelopes as kernels::keys::Fact>::KEY));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
};

// ── THE NAIVE PAGED FALLBACK ──
//
// THE `&KvLayer` AGGREGATE IS GONE and `driver_bound!` went with it: `Arg` is
// implemented for no host struct, so the aggregate was the only reason the
// row could not be a `routine!`. Ten leaves in its place, arity 21 against a
// 36 ceiling.
//
// Ten leaves, ten keys, and the same five that finished
// `write_kv_to_pages_quantised` finished this one -- the two rows were down
// to the identical set.
const _: () = {
    let d = <attention_naive_paged as kernels::Derivation>::DERIVED;
    assert!(d.len() == 21);
    assert!(kernels::source_is_named(&d[0].source, <kernels::keys::KvKeys as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[1].source, <kernels::keys::KvValues as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[2].source, <kernels::keys::KvKeyScales as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[3].source, <kernels::keys::KvValueScales as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::KvPageSize as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::KvHeadDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::KvNumHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::KvSchemeByte as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::KvStorageDtype as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[9].source, <kernels::keys::KvBlockSize as kernels::keys::Fact>::KEY));
    // THE TWO SLOTS THE UNFOLD MOVED. Ten leaves went in ahead of them and
    // neither slot shifted, because both are STATED -- `In<0>`/`Out<0>` --
    // and `stated_source` short-circuits ahead of the positional counters.
    assert!(matches!(d[10].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[11].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[12].source, <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[15].source, <kernels::keys::KvLastPageLens as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[17].source, <kernels::keys::WindowLeft as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[18].source, <kernels::keys::SmScale as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[19].source, <kernels::keys::AttnLogitsSoftCap as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[20].source, <kernels::keys::AttnLseOut as kernels::keys::Fact>::KEY));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
};

// ── THE EXPLICIT WRITER, THE SAME PAIR AT 7/8 ──
//
// `keys::KvWritePage`/`KvWriteOffset` at 4/5 are RIGHT HERE and WRONG on
// `qkv_decode_qk_norm_rope_write_kv_bf16`: `Fire::w_page_d` null-checks, this
// arm reads it with `?`, and that arm reads it with `unwrap_or(null)`. One
// key, two rows, opposite verdicts, decided by what a null means on each.
const _: () = {
    let d = <write_kv_explicit_bf16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 15);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::KvWritePage as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::KvWriteOffset as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[14].source, <kernels::keys::KvNativeBf16 as kernels::keys::Fact>::KEY));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::RowValid as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::KvEnvMin as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::KvEnvMax as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[13].source, <kernels::keys::KvHasEnvelopes as kernels::keys::Fact>::KEY));
};

// ── THE THREE VERDICTS THAT ARE NOT KEYS ──
//
// One pin per row that still carries an arm, asserting WHICH index stops the
// column, so a later pass reads the reason off the assertion rather than
// guessing from a `None`.

// `split_qkv_bf16_devwin` CROSSED. `keys::PeelWindow` and `keys::RowsTotal`
// answer the two the arm fetched, and `total` is the one to read twice: a
// region carries `rows.count` and the `_devwin` forms exist for the fires
// where the two differ.
const _: () = {
    let d = <split_qkv_bf16_devwin as kernels::Derivation>::DERIVED;
    assert!(d.len() == 6);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::PeelWindow as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::RowsTotal as kernels::keys::Fact>::KEY));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
};

// Both softcap forms, and `logit_softcap_f16` has no `Bound` row at all --
// its column is the only thing that binds it.
const _: () = {
    let d = <logit_softcap as kernels::Derivation>::DERIVED;
    assert!(d.len() == 2);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[1].source, <kernels::keys::FinalLogitSoftcap as kernels::keys::Fact>::KEY));
};
const _: () = {
    let d = <logit_softcap_f16 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 2);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&d[1].source, <kernels::keys::FinalLogitSoftcap as kernels::keys::Fact>::KEY));
};

// The `_devwin` explicit writer has no `Bound` row either; `PeelWindow` and
// `RowsTotal` are the pair `split_qkv_bf16_devwin` crossed on.
const _: () = {
    let d = <write_kv_explicit_bf16_devwin as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::PeelWindow as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::RowsTotal as kernels::keys::Fact>::KEY));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
};

// `first_token` at d[2] IS TAKEN AND NEVER LAUNCHED: the refusal it guards
// is the routine's, not the arm's.
//
// d[16]/d[17] ARE BOTH `i32` AND ADJACENT, so swapping `scheme` and
// `storage_dtype` type-checks. This pin is the only thing that does not.
const _: () = {
    let d = <write_kv_to_pages_quantised as kernels::Derivation>::DERIVED;
    assert!(d.len() == 18);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(kernels::source_is_named(&d[2].source, <kernels::keys::FirstToken as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::KvKeyScales as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::KvValueScales as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[15].source, <kernels::keys::KvBlockSize as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[16].source, <kernels::keys::KvSchemeByte as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[17].source, <kernels::keys::KvStorageDtype as kernels::keys::Fact>::KEY));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
};

// d[14] IS `cx.num_pages_in_batch()` AND IT IS NOT `xqa`'s REFUSAL. There the
// parameter means a per-request maximum and the arm passes a fire-wide bound,
// so a key would freeze an approximation. Here the parameter IS the bound --
// `active_geometry` multiplies it by the page's element count to get the whole
// batch's extent -- so the fact and the query are the same thing.
//
// d[10]/d[11] ARE BOTH `i32` AND ADJACENT, as on the quantised writer: only
// this pin distinguishes the scheme from the storage dtype.
const _: () = {
    let d = <dequant_kv_cache_layer_to_bf16_active as kernels::Derivation>::DERIVED;
    assert!(d.len() == 15);
    assert!(kernels::source_is_named(&d[0].source, <kernels::keys::KvKeys as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[1].source, <kernels::keys::KvValues as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[2].source, <kernels::keys::KvKeyScales as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[3].source, <kernels::keys::KvValueScales as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[4].source, <kernels::keys::KvBf16Keys as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[5].source, <kernels::keys::KvBf16Values as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[6].source, <kernels::keys::KvPageSize as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[7].source, <kernels::keys::KvNumHeads as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[8].source, <kernels::keys::KvHeadDim as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[9].source, <kernels::keys::KvBlockSize as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[10].source, <kernels::keys::KvSchemeByte as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[11].source, <kernels::keys::KvStorageDtype as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[12].source, <kernels::keys::KvNativeBf16 as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[13].source, <kernels::keys::KvPageIndices as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&d[14].source, <kernels::keys::KvPagesInBatch as kernels::keys::Fact>::KEY));
    let mut i = 0;
    while i < d.len() {
        assert!(d[i].source.is_some());
        i += 1;
    }
};

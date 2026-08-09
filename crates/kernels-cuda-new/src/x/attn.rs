#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;
use kernels::{Cap, Prepare};

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
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
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::U8(self.0)
    }
}

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
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::U8(self.0)
    }
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
        "::pie_cuda_driver::kernels::attn::device::StructuredMaskParams";

    /// The array of descriptors, one per lane, as `pack_structured_mask`
    impl crate::x::Abi for *const StructuredMaskParams {
        const CPP: &'static str =
            "const ::pie_cuda_driver::kernels::attn::device::StructuredMaskParams*";
        const TY: Ty = Ty::StructuredMasks;
        #[cfg(feature = "_cuda")]
        fn arg(&self) -> crate::runtime::ArgValue {
            crate::runtime::ArgValue::Ptr(*self as *mut c_void)
        }
    }

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
    use super::bf16;

    unit! {
        /// The attention-sink pair, both rows: the log2→ln rebase and the
        unit ATTN_SINK = "attn/attn_sink",
            text = include_str!("../../csrc/src/attn/attn_sink.cuh"),
            file = "attn/attn_sink.cuh";

        /// `attn_sink.cuh:74` — flashinfer publishes its LSE in log2 and the
        fn lse_log2_to_ln = "attn::device::lse_log2_to_ln" <T> (
            lse: *mut T,
            n: usize,
        ) where *mut T {
            "attn::lse_log2_to_ln" => where [T = f32] "attn::device::f32",
        }

        /// `attn_sink.cuh:93` — `o[t, h, :] *= sigmoid(ln_lse[t, h] -
        fn attn_sink_rescale = "attn::device::attn_sink_rescale" <T> (
            o: *mut T,
            lse: *const f32,
            sinks: *const T,
            n: i32,
            num_q_heads: i32,
            head_dim: i32,
        ) where *const T, *mut T {
            "attn::attention_sink_rescale_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/attn_res.cuh` — K3's residual-block blend.
pub mod attn_res {
    use super::bf16;

    unit! {
        /// K3's residual-block blend, at bf16.
        unit ATTN_RES = "attn/attn_res",
            text = include_str!("../../csrc/src/attn/attn_res.cuh"),
            file = "attn/attn_res.cuh";

        /// `attn_res.cuh:99` — one block per token, 256 threads.
        fn attn_res_blend = "attn::device::attn_res_blend" <T> (
            prefix: *const T,
            blocks: *const T,
            norm_weight: *const T,
            proj_weight: *const T,
            out: *mut T,
            b: i32,
            h: i32,
            block_rows: i32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::attn_res_blend_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/head_dim_pad.cuh` — flashinfer's supported head widths, reached by
pub mod head_dim_pad {
    use super::bf16;

    unit! {
        /// The pad and the strip, at bf16.
        unit HEAD_DIM_PAD = "attn/head_dim_pad",
            text = include_str!("../../csrc/src/attn/head_dim_pad.cuh"),
            file = "attn/head_dim_pad.cuh";

        /// `head_dim_pad.cuh:73` — copy `head_dim` values per (token, head)
        fn pad_head_dim = "attn::device::pad_head_dim" <T> (
            packed: *const T,
            padded: *mut T,
            num_heads: i32,
            head_dim: i32,
            head_dim_padded: i32,
        ) where *const T, *mut T {
            "attn::pad_head_dim_bf16" => where [T = bf16] "device::bf16",
        }

        /// `head_dim_pad.cuh:92` — the inverse, and the same five operands
        fn strip_head_dim = "attn::device::strip_head_dim" <T> (
            padded: *const T,
            packed: *mut T,
            num_heads: i32,
            head_dim: i32,
            head_dim_padded: i32,
        ) where *const T, *mut T {
            "attn::strip_head_dim_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/softcap.cuh` — the logit cap, at both numeric formats.
pub mod softcap {
    use super::{bf16, f16};

    unit! {
        /// One `__global__` template and nothing else — no host function, no
        unit SOFTCAP = "attn/softcap",
            text = include_str!("../../csrc/src/attn/softcap.cuh"),
            file = "attn/softcap.cuh";

        /// `softcap.cuh:67` — `x = cap * tanh(x / cap)`, elementwise and in
        fn logit_softcap = "attn::device::logit_softcap" <T> (
            x: *mut T,
            cap: f32,
            n: usize,
        ) where *mut T {
            "attn::logit_softcap_bf16" => where [T = bf16] "device::bf16",
            "attn::logit_softcap_f16" => where [T = f16] "device::f16",
        }
    }
}

/// `attn/split_packed.cuh` — the fused QKV product cut into three operands.
pub mod split_packed {
    use super::bf16;

    unit! {
        /// Two `__global__` templates, no host code.
        unit SPLIT_PACKED = "attn/split_packed",
            text = include_str!("../../csrc/src/attn/split_packed.cuh"),
            file = "attn/split_packed.cuh";

        /// `split_packed.cuh:74` — the host-window form, over
        fn split_qkv = "attn::device::split_qkv" <T> (
            src: *const T,
            q_out: *mut T,
            k_out: *mut T,
            v_out: *mut T,
            q_dim: i32,
            kv_dim: i32,
        ) where *const T, *mut T {
            "attn::split_qkv_bf16" => where [T = bf16] "device::bf16",
        }

        /// `split_packed.cuh:111` — the device-window form, over BASE
        fn split_qkv_devwin = "attn::device::split_qkv_devwin" <T> (
            src: *const T,
            q_out: *mut T,
            k_out: *mut T,
            v_out: *mut T,
            win: *const u32,
            q_dim: i32,
            kv_dim: i32,
        ) where *const T, *mut T {
            "attn::split_qkv_devwin" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/attention_flashinfer.cuh` — the per-head → per-request score fold.
pub mod attention_flashinfer {
    use core::ffi::c_void;

    unit! {
        /// One row, `DeviceKernel::PLAIN`.
        unit ATTENTION_FLASHINFER = "attn/attention_flashinfer",
            text = include_str!("../../csrc/src/attn/attention_flashinfer.cuh"),
            file = "attn/attention_flashinfer.cuh";

        /// The fold: per-head scores summed to one row per request.
        fn attn_score_fold_heads = "attn::device::attn_score_fold_heads" (
            scores: *const c_void,
            score_indptr: *const i32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            page_size: i32,
            num_q_heads: i32,
            folded: *mut c_void,
        ) {
            "attn::attn_score_fold_heads_dev" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/pack_dense_mask.cuh` — the two custom-mask packers, both plain
pub mod pack_dense_mask {
    use super::params::StructuredMaskParams;

    unit! {
        /// Two `__global__`s and no host code at all.
        unit PACK_DENSE_MASK = "attn/pack_dense_mask",
            text = include_str!("../../csrc/src/attn/pack_dense_mask.cuh"),
            file = "attn/pack_dense_mask.cuh";

        /// `pack_dense_mask.cuh:149` — a dense byte-per-cell mask packed to
        fn pack_dense_mask = "attn::device::pack_dense_mask" (
            kvm_dense: *const u8,
            klen: *const u32,
            qo_indptr: *const u32,
            mask_indptr: *const i32,
            packed: *mut u8,
            b: i32,
            p_page: i32,
        ) {
            "attn::pack_dense_mask" => crate::device::DeviceKernel::PLAIN,
        }

        /// `pack_dense_mask.cuh:189` — the same bitmap ABI materialised
        fn pack_structured_mask = "attn::device::pack_structured_mask" (
            positions: *const u32,
            klen: *const u32,
            qo_indptr: *const u32,
            mask_indptr: *const i32,
            masks: *const StructuredMaskParams,
            packed: *mut u8,
            b: i32,
        ) {
            "attn::pack_structured_mask" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/dsa_indexer.cuh` — glm5's sparse-attention index network, three
pub mod dsa_indexer {
    use super::bf16;

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

    unit! {
        /// Three `__global__` templates and the RoPE helper they share. No
        unit DSA_INDEXER = "attn/dsa_indexer",
            text = include_str!("../../csrc/src/attn/dsa_indexer.cuh"),
            file = "attn/dsa_indexer.cuh";

        /// `dsa_indexer.cuh:106` — LayerNorm over `head_dim` then interleaved
        fn index_knorm_rope = "attn::device::index_knorm_rope" <T> (
            idx_k: *mut T,
            w: *const T,
            b: *const T,
            positions: *const i32,
            head_dim: i32,
            rope_dim: i32,
            theta: f32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::dsa_index_knorm_rope_dev" => where [T = bf16] "device::bf16",
        }

        /// `dsa_indexer.cuh:151` — interleaved RoPE on the first `rope_dim`
        fn index_q_rope = "attn::device::index_q_rope" <T> (
            idx_q: *mut T,
            positions: *const i32,
            n_heads: i32,
            head_dim: i32,
            rope_dim: i32,
            theta: f32,
        ) where *mut T {
            "attn::dsa_index_q_rope_dev" => where [T = bf16] "device::bf16",
        }

        /// `dsa_indexer.cuh:187` — causal top-k mask over the index scores,
        fn index_topk_mask = "attn::device::index_topk_mask" <T> (
            idx_q: *const T,
            idx_k: *const T,
            idx_w: *const T,
            mask: *mut u8,
            n: i32,
            n_heads: i32,
            head_dim: i32,
            topk: i32,
        ) where *const T {
            "attn::dsa_index_topk_mask_dev" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/page_compact.cuh` — dropping the pages a keep-mask rejects and
pub mod page_compact {
    unit! {
        /// The page compactor, both halves.
        unit PAGE_COMPACT = "attn/page_compact",
            text = include_str!("../../csrc/src/attn/page_compact.cuh"),
            file = "attn/page_compact.cuh";

        /// `page_compact.cuh:212` — one block per request: how many of its
        fn count_kept = "attn::device::count_kept"(
            page_indptr_in: *const u32,
            keep: *const u8,
            keep_stride: u32,
            num_requests: i32,
            counts: *mut u32,
        ) {
            "attn::count_kept" => "device::i32(256)",
        }

        /// `page_compact.cuh:242` — scan and scatter, fused into one launch.
        fn scan_and_scatter = "attn::device::scan_and_scatter"(
            page_indices_in: *const u32,
            page_indptr_in: *const u32,
            last_page_lens_in: *const u32,
            keep: *const u8,
            counts: *const u32,
            keep_stride: u32,
            num_requests: i32,
            page_indptr_out: *mut u32,
            last_page_lens_out: *mut u32,
            page_indices_out: *mut u32,
        ) {
            "attn::scan_and_scatter" => "device::i32(256)",
        }
    }

    /// `page_compact.cuh:114` — `constexpr int kBlock = 256`.
    pub const K_BLOCK: u32 = 256;
}

/// `attn::compact_page_csr` — the page compactor's host program.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across BOTH
/// launches — `scratch_counts` especially, which is written by the first and
/// read by the second — and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn compact_page_csr(
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
    stream: *mut c_void,
) -> Fired {
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if scratch_counts.is_null() {
        return Fired::Declined(Refusal::Absent { what: "the compaction scratch buffer" });
    }
    let launch = Launch::per_row(num_requests.unsigned_abs(), page_compact::K_BLOCK);
    // SAFETY: the caller's obligation, above.
    unsafe {
        page_compact::raw::count_kept(
            "attn::count_kept",
            launch,
            page_indptr_in,
            keep,
            keep_stride,
            num_requests,
            scratch_counts,
            stream,
        );
        page_compact::raw::scan_and_scatter(
            "attn::scan_and_scatter",
            launch,
            page_indices_in,
            page_indptr_in,
            last_page_lens_in,
            keep,
            scratch_counts,
            keep_stride,
            num_requests,
            page_indptr_out,
            last_page_lens_out,
            page_indices_out,
            stream,
        );
    }
    Fired::Launched
}

/// `attn/attention_naive.cuh` — the MTP pair and the reference attention.
pub mod attention_naive {
    use super::bf16;

    unit! {
        /// Multi-token prediction's two hidden-state movers.
        unit ATTENTION_NAIVE = "attn/attention_naive",
            text = include_str!("../../csrc/src/attn/attention_naive.cuh"),
            file = "attn/attention_naive.cuh";

        /// `attention_naive.cuh:305` — the previous step's pending hidden
        fn mtp_shift_hidden = "attn::device::mtp_shift_hidden" <T> (
            target_hidden: *const T,
            pending_hidden: *const T,
            qo_indptr: *const u32,
            slot_ids: *const i32,
            out: *mut T,
            num_requests: i32,
            hidden_size: i32,
        ) where *const T, *mut T {
            "attn::mtp_shift_hidden_dev" => where [T = bf16] "device::bf16",
        }

        /// `attention_naive.cuh:337` — the end-of-step refresh: each
        fn attn_naive = "attn::device::attn_naive" <T> (
            q: *const T,
            k: *const T,
            v: *const T,
            o: *mut T,
            num_tokens: i32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            scale: f32,
        ) where *const T, *mut T {
            "attn::attention_naive_bf16" => where [T = bf16] "device::bf16",
        }

        fn mtp_update_pending_hidden = "attn::device::mtp_update_pending_hidden" <T> (
            target_hidden: *const T,
            pending_hidden: *mut T,
            qo_indptr: *const u32,
            slot_ids: *const i32,
            num_requests: i32,
            hidden_size: i32,
        ) where *const T, *mut T {
            "attn::mtp_update_pending_hidden_dev" => where [T = bf16] "device::bf16",
        }
    }

    /// `attention_naive.cu:57` — `constexpr int BLOCK = device::BLOCK;`,
    pub const BLOCK: u32 = 256;
}

/// `attn::mtp_shift_hidden_bf16` — one block per TOKEN.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_shift_hidden_bf16(
    target_hidden: *const bf16,
    pending_hidden: *const bf16,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    out: *mut bf16,
    total_tokens: i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if hidden_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden width" });
    }
    if pending_hidden.is_null() {
        return Fired::Declined(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        attention_naive::raw::mtp_shift_hidden(
            "attn::mtp_shift_hidden_dev",
            Launch::per_row(total_tokens.unsigned_abs(), attention_naive::BLOCK),
            target_hidden,
            pending_hidden,
            qo_indptr,
            slot_ids,
            out,
            num_requests,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::mtp_update_pending_hidden_bf16` — one block per REQUEST.
///
/// # Safety
///
/// [`mtp_shift_hidden_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_update_pending_hidden_bf16(
    target_hidden: *const bf16,
    pending_hidden: *mut bf16,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if hidden_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden width" });
    }
    if pending_hidden.is_null() {
        return Fired::Declined(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        attention_naive::raw::mtp_update_pending_hidden(
            "attn::mtp_update_pending_hidden_dev",
            Launch::per_row(num_requests.unsigned_abs(), attention_naive::BLOCK),
            target_hidden,
            pending_hidden,
            qo_indptr,
            slot_ids,
            num_requests,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
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
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// `layer`'s two page pointers included, and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub unsafe fn mla_prepare_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
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

    // SAFETY: the caller's obligation, above.
    unsafe {
        mla_paged::raw::mla_prepare(
            "attn::mla_prepare",
            Launch {
                grid: [total_tokens.unsigned_abs(), blocks.saturating_add(1).max(1).unsigned_abs(), 1],
                block: [MLA_PREPARE_BLOCK.unsigned_abs(), 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            kv_a,
            kv_a_norm_weight,
            q_b,
            kv_c,
            k_pe,
            q_nope,
            q_pe,
            layer.ckv_pages.cast(),
            layer.kpe_pages.cast(),
            positions,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            row_valid,
            num_requests,
            layer.page_size,
            heads,
            kv_lora,
            qk_nope_head_dim,
            rope,
            stride,
            eps,
            theta,
            interleaved,
            per_block,
            yarn_factor,
            low_dim,
            high_dim,
            yarn_mscale,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::write_mla_to_pages` — appends one step's compressed latent and rope
///
/// # Safety
///
/// [`mla_prepare_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_mla_to_pages(
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
    stream: *mut c_void,
) -> Fired {
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        mla_paged::raw::write_mla(
            "attn::write_mla",
            Launch::per_row(total_tokens.unsigned_abs(), MLA_WRITE_BLOCK),
            ckv_curr,
            kpe_curr,
            layer.ckv_pages.cast(),
            layer.kpe_pages.cast(),
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            row_valid,
            num_requests,
            layer.page_size,
            layer.kv_lora_rank,
            layer.qk_rope_head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `dsv4_compress.cu:37` — `constexpr int ATTN_BLOCK = 128;`.
#[cfg(feature = "_cuda")]
const DSV4_ATTN_BLOCK: u32 = 128;

/// `dsv4_compress.cu:139` and `:161` — the boundary-meta block.
#[cfg(feature = "_cuda")]
const DSV4_META_BLOCK: u32 = 128;

/// `attn::dsv4_boundary_meta_decode` — each decode row's compressed-block
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_boundary_meta_decode(
    positions: *const i32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    ratio: i32,
    row_valid: *const u8,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "elements" });
    }
    if ratio <= 0 {
        return Fired::Declined(Refusal::Narrow { what: "ratio", at: ratio });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsv4_compress::raw::dsv4_boundary_meta_decode(
            "attn::dsv4_boundary_meta_decode_dev",
            Launch::flat(n.unsigned_abs(), DSV4_META_BLOCK),
            positions,
            out_pos,
            out_req,
            out_rope,
            n,
            ratio,
            row_valid,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsv4_boundary_meta_paged` — the prefill form of
///
/// # Safety
///
/// [`dsv4_boundary_meta_decode`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_boundary_meta_paged(
    positions: *const i32,
    qo_indptr: *const u32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    num_requests: i32,
    ratio: i32,
    row_valid: *const u8,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "elements" });
    }
    if ratio <= 0 {
        return Fired::Declined(Refusal::Narrow { what: "ratio", at: ratio });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsv4_compress::raw::dsv4_boundary_meta_paged(
            "attn::dsv4_boundary_meta_paged_dev",
            Launch::flat(n.unsigned_abs(), DSV4_META_BLOCK),
            positions,
            qo_indptr,
            out_pos,
            out_req,
            out_rope,
            n,
            num_requests,
            ratio,
            row_valid,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attention_compressed_paged_bf16` — attention against the COMPRESSED
///
/// # Safety
///
/// [`dsv4_boundary_meta_decode`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_compressed_paged_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "query tokens" });
    }
    if num_q_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "q heads" });
    }
    let smem = head_dim
        .max(0)
        .unsigned_abs()
        .saturating_add(DSV4_ATTN_BLOCK)
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsv4_compress::raw::compressed_attn_paged(
            "attn::compressed_attn_paged_dev",
            Launch {
                grid: [total_tokens.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
                block: [DSV4_ATTN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(smem),
            q,
            comp_kv_pages,
            o,
            lse_out,
            positions,
            kv_page_indices,
            kv_page_indptr,
            req_of_token,
            num_q_heads,
            head_dim,
            ratio,
            page_size,
            sm_scale,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsa_index_knorm_rope_bf16` — LayerNorm then interleaved RoPE on the
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the
/// launch, and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_knorm_rope_bf16(
    idx_k: *mut bf16,
    k_norm_weight: *const bf16,
    k_norm_bias: *const bf16,
    positions: *const i32,
    tokens: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsa_indexer::raw::index_knorm_rope(
            "attn::dsa_index_knorm_rope_dev",
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::K_BLOCK),
            idx_k,
            k_norm_weight,
            k_norm_bias,
            positions,
            head_dim,
            rope_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsa_index_q_rope_bf16` — interleaved RoPE on the indexer's QUERY
///
/// # Safety
///
/// [`dsa_index_knorm_rope_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_q_rope_bf16(
    idx_q: *mut bf16,
    positions: *const i32,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsa_indexer::raw::index_q_rope(
            "attn::dsa_index_q_rope_dev",
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::q_rope_block(n_heads)),
            idx_q,
            positions,
            n_heads,
            head_dim,
            rope_dim,
            theta,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsa_index_topk_mask` — score every causal (query, key) pair and
///
/// # Safety
///
/// [`dsa_index_knorm_rope_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_topk_mask_bf16(
    idx_q: *const bf16,
    idx_k: *const bf16,
    idx_w: *const bf16,
    mask: *mut u8,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    topk: i32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    let smem = tokens
        .unsigned_abs()
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsa_indexer::raw::index_topk_mask(
            "attn::dsa_index_topk_mask_dev",
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::K_BLOCK).smem(smem),
            idx_q,
            idx_k,
            idx_w,
            mask,
            tokens,
            n_heads,
            head_dim,
            topk,
            stream,
        );
    }
    Fired::Launched
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
    use super::bf16;

    unit! {
        /// Two `__global__`s and the `mma_detail` helpers the second one
        unit MLA_NAIVE = "attn/attention_mla_naive",
            text = include_str!("../../csrc/src/attn/attention_mla_naive.cuh"),
            file = "attn/attention_mla_naive.cuh";

        /// `attention_mla_naive.cuh:92` — the scalar flash-softmax kernel,
        fn mla_naive_paged_kernel = "attn::mla_naive::mla_naive_paged_kernel" (
            q_nope: *const bf16,
            q_pe: *const bf16,
            ckv_pages: *const bf16,
            kpe_pages: *const bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            o: *mut bf16,
            index_mask: *const u8,
            index_mask_stride: i32,
            r: i32,
            h: i32,
            ckv: i32,
            kpe: i32,
            page_size: i32,
            sm_scale: f32,
            causal: bool,
            g: i32,
        ) {
            "attn::mla_naive_paged" => crate::device::DeviceKernel::PLAIN,
        }

        /// `attention_mla_naive.cuh:371` — the tensor-core kernel, sixteen
        fn mla_mma_paged_kernel = "attn::mla_naive::mma_detail::mla_mma_paged_kernel" (
            q_nope: *const bf16,
            q_pe: *const bf16,
            ckv_pages: *const bf16,
            kpe_pages: *const bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            o: *mut bf16,
            index_mask: *const u8,
            index_mask_stride: i32,
            r: i32,
            h: i32,
            page_size: i32,
            sm_scale: f32,
            causal: bool,
        ) {
            "attn::mla_mma_paged" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/kimi_mla.cuh` — kimi_k3's two latent-attention preparation kernels.
pub mod kimi_mla {
    use super::bf16;

    unit! {
        /// Two `__global__` templates and nothing else, which is what the
        unit KIMI_MLA = "attn/kimi_mla",
            text = include_str!("../../csrc/src/attn/kimi_mla.cuh"),
            file = "attn/kimi_mla.cuh";

        /// `kimi_mla.cuh:67` — split a fused `q_b` projection into its nope
        fn split_q_b = "attn::device::split_q_b" <T> (
            q_b: *const T,
            q_nope: *mut T,
            q_pe: *mut T,
            total: i32,
            heads: i32,
            nope: i32,
            rope: i32,
        ) where *const T, *mut T {
            "attn::kimi_split_q_b_bf16" => where [T = bf16] "device::bf16",
        }

        /// `kimi_mla.cuh:101` — split `kv_a` into a normalised latent and
        fn split_kv_a_norm = "attn::device::split_kv_a_norm" <T> (
            kv_a: *const T,
            norm_weight: *const T,
            kv_c: *mut T,
            k_pe: *mut T,
            kv_lora: i32,
            rope: i32,
            src_row_stride: i32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::kimi_split_kv_a_norm_bf16" => where [T = bf16] "device::bf16, 256",
        }
    }
}

/// `attn/attention_naive_paged.cuh` — the reference paged attention.
pub mod attention_naive_paged {
    use super::{bf16, kv_dtype, kv_scheme};
    use core::ffi::c_void;

    unit! {
        /// Two `__global__`s, both `template <int BLOCK>`, both at 128.
        unit ATTENTION_NAIVE_PAGED = "attn/attention_naive_paged",
            text = include_str!("../../csrc/src/attn/attention_naive_paged.cuh"),
            file = "attn/attention_naive_paged.cuh";

        /// `attention_naive_paged.cuh:346` — the prefill, over
        fn naive_paged_attn = "attn::device::naive_paged_attn" (
            q: *const bf16,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const f32,
            v_scales: *const f32,
            o: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            custom_mask: *const u8,
            custom_mask_indptr: *const i32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            scheme: kv_scheme,
            storage_dtype: kv_dtype,
            block_size: i32,
            window_left: i32,
            sm_scale: f32,
            logits_soft_cap: f32,
            lse_out: *mut f32,
        ) {
            "attn::attention_naive_paged_dev" => "device::i32(128)",
        }

        /// `attention_naive_paged.cuh:518` — the decode, over
        fn naive_paged_decode = "attn::device::naive_paged_decode" (
            q: *const bf16,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const f32,
            v_scales: *const f32,
            o: *mut bf16,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            scheme: kv_scheme,
            storage_dtype: kv_dtype,
            block_size: i32,
            window_left: i32,
            sm_scale: f32,
            logits_soft_cap: f32,
            lse_out: *mut f32,
        ) {
            "attn::naive_paged_decode" => "device::i32(128)",
        }
    }
}

/// `attn/mla_paged.cuh` — the MLA cache's append and its preparation pass.
pub mod mla_paged {
    use super::bf16;

    unit! {
        /// Two `__global__`s, one of them a template over its BLOCK WIDTH
        unit MLA_PAGED = "attn/mla_paged",
            text = include_str!("../../csrc/src/attn/mla_paged.cuh"),
            file = "attn/mla_paged.cuh";

        /// `mla_paged.cuh:174` — append one token's latent KV to its page.
        fn write_mla = "attn::device::write_mla" (
            ckv_curr: *const bf16,
            kpe_curr: *const bf16,
            ckv_pages: *mut bf16,
            kpe_pages: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            r: i32,
            page_size: i32,
            kv_lora_rank: i32,
            qk_rope_head_dim: i32,
        ) {
            "attn::write_mla" => crate::device::DeviceKernel::PLAIN,
        }

        /// `mla_paged.cuh:223` — the fused MLA prepare, at `BLOCK_DIM = 256`.
        fn mla_prepare = "attn::device::mla_prepare" (
            kv_a: *const bf16,
            kv_a_norm_w: *const bf16,
            q_b: *const bf16,
            kv_c: *mut bf16,
            k_pe: *mut bf16,
            q_nope: *mut bf16,
            q_pe: *mut bf16,
            ckv_pages: *mut bf16,
            kpe_pages: *mut bf16,
            positions: *const i32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            r: i32,
            page_size: i32,
            heads: i32,
            kv_lora: i32,
            nope: i32,
            rope: i32,
            src_row_stride: i32,
            eps: f32,
            theta: f32,
            interleaved: bool,
            heads_per_block: i32,
            yarn_factor: f32,
            yarn_low_dim: f32,
            yarn_high_dim: f32,
            yarn_mscale: f32,
        ) {
            "attn::mla_prepare" => "device::i32(256)",
        }
    }
}

/// The FlashInfer FA2 MLA host program — **compiled, and not yet fireable.**
pub mod mla_fa2 {
    use super::bf16;
    use super::mla_params::{MlaParams, UintFastdiv};
    use crate::plan::MlaPlanInfo;
    use crate::x::launch::Launch;

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

    unit! {
        /// `flashinfer::mla::BatchMLAPagedAttentionKernel`, six ways.
        unit MLA_FA2 = "attn/attention_mla_fa2",
            text = include_str!("../../csrc/src/attn/attention_mla_fa2.cuh"),
            file = "attn/attention_mla_fa2.cuh",
            options = OPTIONS;

        /// `mla.cuh` — the whole of paged MLA, in two stages separated by
        fn attention = "::flashinfer::mla::BatchMLAPagedAttentionKernel" (
            params: MlaParams,
        ), cooperative = true {
            "attn::mla_fa2_kv64_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 64u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv64_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 2u, true, 64u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv32_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 32u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv32_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 2u, true, 32u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv16_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 1u, false, 16u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv16_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 1u, false, 16u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
        }
    }

    /// The one NVRTC option this root needs, and the sixteen errors that say
    pub const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

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
            merge_partial_stride: unsafe {
                offset_ptr(int_buf, plan.merge_partial_stride_offset)
            },
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
        Launch {
            grid: [plan.num_blks_x as u32, plan.num_blks_y as u32, 1],
            block: [256, 1, 1],
            smem: arm.smem,
            smem_opt_in: true,
        }
    }
}

/// The units `attn` compiles in fn-world.
pub mod qkv_fused {
    use super::bf16;
    #[cfg(feature = "_cuda")]
    use super::{Fired, Launch, Refusal};

    unit! {
        /// Three `__global__` templates, no host code.
        unit QKV_FUSED = "attn/qkv_fused",
            text = include_str!("../../csrc/src/attn/qkv_fused.cuh"),
            file = "attn/qkv_fused.cuh";

        /// `qkv_fused.cuh:412` — the PACKED prefill epilogue.
        fn packed = "attn::device::qkv_packed_qk_norm_rope_vnorm_write_kv" (
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
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" => "device::i32(256)",
        }

        /// `qkv_fused.cuh:115` — the BLOCK decode form.
        fn block = "attn::device::qkv_decode_qk_norm_rope_write_kv" (
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
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_decode_qk_norm_rope_write_kv#rope" => "device::i32(128), true",
            "attn::qkv_decode_qk_norm_rope_write_kv#norope" => "device::i32(128), false",
        }

        /// `qkv_fused.cuh:252` — the WARP decode form.
        fn warp = "attn::device::qkv_decode_qk_norm_rope_write_kv_warp" (
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
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#rope" => "device::i32(128), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#norope" => "device::i32(128), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#rope" => "device::i32(64), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#norope" => "device::i32(64), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#rope" => "device::i32(256), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#norope" => "device::i32(256), false",
        }
    }

    /// `BLOCK` for the packed form, and it IS the block width.
    pub const PACKED_BLOCK: u32 = 256;

    /// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` — the fused
    ///
    /// # Safety
    ///
    /// Every pointer must be a device address valid for the fire, `packed`
    /// must hold `num_rows` rows of `(num_q_heads + 2·num_kv_heads)·head_dim`
    /// elements, and the page arrays must describe the layer the cache
    /// pointers came from.
    #[cfg(feature = "_cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
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
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_rows <= 0 {
            return Fired::Declined(Refusal::Empty { what: "rows" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if num_q_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "q heads" });
        }
        if num_kv_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "kv heads" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page size" });
        }
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();
        unsafe {
            raw::packed(
                "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
                Launch {
                    grid: [num_rows.unsigned_abs(), heads, 1],
                    block: [PACKED_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                packed,
                q_out,
                k_pages,
                v_pages,
                q_weight,
                k_weight,
                positions,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                row_valid,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps,
                stream,
            );
        }
        Fired::Launched
    }
    /// `qkv_fused.cu:51` — `constexpr int WARP_BLOCK = 256;`, and it is NOT
    pub const WARP_BLOCK: u32 = 256;

    /// `qkv_fused.cu:105` — `constexpr int BLOCK = 128;`, the DECODE block.
    pub const DECODE_BLOCK: u32 = 128;

    /// Warps per block: `WARP_BLOCK / 32`.
    const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

    /// The six warp instantiations, by `(head_dim, rope_table)`.
    #[cfg(feature = "_cuda")]
    fn warp_symbol(head_dim: i32, rope_table: bool) -> Option<&'static str> {
        Some(match (head_dim, rope_table) {
            (64, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#rope",
            (64, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#norope",
            (128, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#rope",
            (128, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#norope",
            (256, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#rope",
            (256, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#norope",
            _ => return None,
        })
    }

    /// The block form's two arms.
    #[cfg(feature = "_cuda")]
    fn block_symbol(rope_table: bool) -> &'static str {
        if rope_table {
            "attn::qkv_decode_qk_norm_rope_write_kv#rope"
        } else {
            "attn::qkv_decode_qk_norm_rope_write_kv#norope"
        }
    }

    /// `attn/qkv_fused.cu:31` — `qkv_decode_fused_dispatch`, the `static` one.
    ///
    /// # Safety
    ///
    /// Every pointer is a device address the caller keeps live across the
    /// launch; the five named above may be null. `stream` is the caller's.
    #[cfg(feature = "_cuda")]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub unsafe fn qkv_decode_fused_dispatch(
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
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_requests <= 0 {
            return Fired::Declined(Refusal::Empty { what: "requests" });
        }
        if q_out.is_null() {
            return Fired::Declined(Refusal::Absent { what: "q_out" });
        }
        if num_q_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "q heads" });
        }
        if num_kv_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "kv heads" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page size" });
        }

        let use_rope_table = !rope_table.is_null();
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();

        if let Some(symbol) = warp_symbol(head_dim, use_rope_table) {
            let units = num_requests.unsigned_abs().saturating_mul(heads);
            unsafe {
                raw::warp(
                    symbol,
                    Launch {
                        grid: [units.div_ceil(WARPS_PER_BLOCK), 1, 1],
                        block: [WARP_BLOCK, 1, 1],
                        smem: 0,
                        smem_opt_in: false,
                    },
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
                    win,
                    num_requests,
                    num_q_heads,
                    num_kv_heads,
                    page_size,
                    hnd_layout,
                    theta,
                    eps,
                    stream,
                );
            }
            return Fired::Launched;
        }

        unsafe {
            raw::block(
                block_symbol(use_rope_table),
                Launch {
                    grid: [num_requests.unsigned_abs(), heads, 1],
                    block: [DECODE_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
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
                win,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps,
                stream,
            );
        }
        Fired::Launched
    }

    /// `attn/qkv_fused.cu:160` — `qkv_decode_qk_norm_rope_write_kv_bf16`.
    ///
    /// # Safety
    ///
    /// [`qkv_decode_fused_dispatch`]'s.
    #[cfg(feature = "_cuda")]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub unsafe fn qkv_decode_qk_norm_rope_write_kv_bf16(
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
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        // SAFETY: the caller's contract, forwarded; `win` is null here.
        unsafe {
            qkv_decode_fused_dispatch(
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
                stream,
            )
        }
    }
}

/// `attn/dsv4_compress.cuh` — deepseek_v4's SECOND KV cache, and the eleven
pub mod dsv4_compress {
    use super::bf16;
    #[cfg(feature = "_cuda")]
    use super::{Fired, Launch, Refusal};

    unit! {
        unit DSV4_COMPRESS = "attn/dsv4_compress",
            text = include_str!("../../csrc/src/attn/dsv4_compress.cuh"),
            file = "attn/dsv4_compress.cuh";

        /// `:105` — the mean over each window of `ratio` input tokens.
        fn average_pool = "attn::device::average_pool"(
            input: *const bf16,
            output: *mut bf16,
            n: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::average_pool_bf16" => "device::bf16",
        }

        /// `:130` — the absolute position table, added in place.
        fn add_ape = "attn::device::add_ape"(
            data: *mut bf16,
            ape: *const f32,
            n_compressed: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::add_ape_f32" => "device::bf16",
        }

        /// `:154` — a per-dimension softmax over `ratio` gate scores, then
        fn gated_softmax_pool = "attn::device::gated_softmax_pool"(
            kv: *const bf16,
            score: *const bf16,
            output: *mut bf16,
            n: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::gated_softmax_pool_bf16" => "device::bf16",
        }

        /// The unpaged gather — one block per compressed entry, striding its
        fn dsv4_compress_gather = "attn::device::dsv4_compress_gather"(
            kv_proj: *const bf16,
            score_proj: *const bf16,
            ape: *const f32,
            boundary_tok: *const i32,
            boundary_pos: *const i32,
            window_lo: *const i32,
            out: *mut bf16,
            head_dim: i32,
            ratio: i32,
            coff: i32,
        ) {
            "attn::dsv4_compress_gather_bf16" => "device::bf16",
        }

        /// `:578` — the paged gather, and the first of the two the planner
        fn dsv4_compress_gather_paged = "attn::device::dsv4_compress_gather_paged"(
            state_kv: *const bf16,
            state_score: *const bf16,
            ape: *const f32,
            boundary_pos: *const i32,
            boundary_req: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            out: *mut bf16,
            head_dim: i32,
            ratio: i32,
            coff: i32,
            page_size: i32,
        ) {
            "attn::dsv4_compress_gather_paged_bf16" => "device::bf16",
        }

        /// `:648` — commit those entries to the compressed cache, each at its
        fn dsv4_store_comp_entries = "attn::device::dsv4_store_comp_entries"(
            entries: *const bf16,
            comp_kv_pages: *mut bf16,
            boundary_pos: *const i32,
            boundary_req: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            head_dim: i32,
            page_size: i32,
        ) {
            "attn::dsv4_store_comp_entries_bf16" => "device::bf16",
        }

        /// `:530` — which decode rows close a compression window.
        fn dsv4_boundary_meta_decode = "attn::device::dsv4_boundary_meta_decode"(
            positions: *const i32,
            out_pos: *mut i32,
            out_req: *mut i32,
            out_rope: *mut i32,
            n: i32,
            ratio: i32,
            row_valid: *const u8,
        ) {
            "attn::dsv4_boundary_meta_decode_dev" => "device::i32",
        }

        /// `:544` — the prefill form.
        fn dsv4_boundary_meta_paged = "attn::device::dsv4_boundary_meta_paged"(
            positions: *const i32,
            qo_indptr: *const u32,
            out_pos: *mut i32,
            out_req: *mut i32,
            out_rope: *mut i32,
            n: i32,
            num_requests: i32,
            ratio: i32,
            row_valid: *const u8,
        ) {
            "attn::dsv4_boundary_meta_paged_dev" => "device::i32",
        }

        /// `:666` — the attention itself, over the compressed cache.
        fn compressed_attn_paged = "attn::device::compressed_attn_paged"(
            q: *const bf16,
            comp_kv_pages: *const bf16,
            o: *mut bf16,
            lse_out: *mut f32,
            positions: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            req_of_token: *const i32,
            num_q_heads: i32,
            head_dim: i32,
            ratio: i32,
            page_size: i32,
            scale: f32,
        ) {
            "attn::compressed_attn_paged_dev" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:216` — the merge, by log-sum-exp.
        fn combine_attn_outputs = "attn::device::combine_attn_outputs"(
            o1: *const bf16,
            lse1: *const f32,
            o2: *const bf16,
            lse2: *const f32,
            o_out: *mut bf16,
            lse_out: *mut f32,
            num_heads: i32,
            head_dim: i32,
        ) {
            "attn::combine_attn_outputs_dev" => "device::bf16",
        }
    }

    /// `route_rows`' warp rounding, and the clamp that makes it legal at any
    #[cfg(feature = "_cuda")]
    #[expect(clippy::cast_sign_loss, reason = "both are guarded positive by every caller")]
    fn route_rows(rows: i32, width: i32) -> Launch {
        let (rows, width) = (rows as u32, width as u32);
        Launch::per_row(rows, width.div_ceil(32).max(1).saturating_mul(32).min(1024))
    }

    /// Build one compressed entry per boundary token.
    ///
    /// # Safety
    ///
    /// Every pointer addresses a live allocation of the extent the kernel
    /// reads, `ape` and nothing else may be null, and the stream outlives the
    /// launch.
    #[cfg(feature = "_cuda")]
    pub unsafe fn dsv4_compress_gather_paged_bf16(
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
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_entries <= 0 {
            return Fired::Declined(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if ratio <= 0 {
            return Fired::Declined(Refusal::Empty { what: "ratio" });
        }
        if coff <= 0 {
            return Fired::Declined(Refusal::Empty { what: "coff" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: the caller's contract, forwarded unchanged.
        unsafe {
            raw::dsv4_compress_gather_paged(
                "attn::dsv4_compress_gather_paged_bf16",
                route_rows(num_entries, head_dim),
                state_kv,
                state_score,
                ape,
                boundary_pos,
                boundary_req,
                kv_page_indices,
                kv_page_indptr,
                out,
                head_dim,
                ratio,
                coff,
                page_size,
                stream,
            );
        }
        Fired::Launched
    }

    /// Commit those entries to the compressed cache.
    ///
    /// # Safety
    ///
    /// As above; no operand of this one is nullable.
    #[cfg(feature = "_cuda")]
    pub unsafe fn dsv4_store_comp_entries_bf16(
        entries: *const bf16,
        comp_kv_pages: *mut bf16,
        boundary_pos: *const i32,
        boundary_req: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        num_entries: i32,
        head_dim: i32,
        page_size: i32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_entries <= 0 {
            return Fired::Declined(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: the caller's contract, forwarded unchanged.
        unsafe {
            raw::dsv4_store_comp_entries(
                "attn::dsv4_store_comp_entries_bf16",
                route_rows(num_entries, head_dim),
                entries,
                comp_kv_pages,
                boundary_pos,
                boundary_req,
                kv_page_indices,
                kv_page_indptr,
                head_dim,
                page_size,
                stream,
            );
        }
        Fired::Launched
    }
}

/// `attn/kv_paged.cuh` — the paged KV cache's appenders, its quantised
pub mod kv_paged {
    use super::bf16;
    use crate::x::abi::MaybeConst;
    use crate::x::fp8_kind;
    use core::ffi::c_void;

    #[cfg(feature = "_cuda")]
    use crate::x::contract::{Fired, Refusal};
    #[cfg(feature = "_cuda")]
    use crate::x::{KvDType, KvLayer, KvScheme};
    #[cfg(feature = "_cuda")]
    use super::Launch;

    unit! {
        unit KV_PAGED = "attn/kv_paged",
            text = include_str!("../../csrc/src/attn/kv_paged.cuh"),
            file = "attn/kv_paged.cuh";

        /// `:153` — the batched append, one block per token.
        fn write_kv = "attn::device::write_kv"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: MaybeConst<u8>,
            win: MaybeConst<u32>,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            first_token: i32,
        ) {
            "attn::write_kv_bf16#hnd" => "device::true_type::value",
            "attn::write_kv_bf16#nhd" => "device::false_type::value",
        }

        /// `:223` — the same append with each token's absolute KV position
        fn write_kv_at_positions = "attn::device::write_kv_at_positions"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            positions: *const i32,
            position_delta: i32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_at_positions_bf16#hnd" => "device::true_type::value",
            "attn::write_kv_at_positions_bf16#nhd" => "device::false_type::value",
        }

        /// `:279` — the append that is told each lane's physical page and
        fn write_kv_explicit = "attn::device::write_kv_explicit"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            b: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_explicit_bf16_dev#hnd" => "device::true_type::value",
            "attn::write_kv_explicit_bf16_dev#nhd" => "device::false_type::value",
        }

        /// `:781` — the explicit append under a device-resident window.
        fn write_kv_explicit_devwin = "attn::device::write_kv_explicit_devwin"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            win: *const u32,
            n_max: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_explicit_bf16_devwin_dev#hnd" => "device::true_type::value",
            "attn::write_kv_explicit_bf16_devwin_dev#nhd" => "device::false_type::value",
        }

        /// `:326` — cell-to-cell copy inside the page arena, one block per
        fn copy_kv_cells = "attn::device::copy_kv_cells"(
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            dst_page: *const u32,
            dst_off: *const u32,
            src_page: *const u32,
            src_off: *const u32,
            n: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::copy_kv_cells_bf16#hnd" => "device::true_type::value",
            "attn::copy_kv_cells_bf16#nhd" => "device::false_type::value",
        }

        /// `:390` — the fp8 append with one scale for the whole tensor.
        fn write_kv_fp8_per_tensor = "attn::device::write_kv_fp8_per_tensor"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut u8,
            v_pages: *mut u8,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            kind: fp8_kind,
        ) {
            "attn::write_kv_fp8_per_tensor" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:425` — the per-token-per-head quantised append, `template <bool
        fn write_kv_per_token_head = "attn::device::write_kv_per_token_head"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages_raw: *mut c_void,
            v_pages_raw: *mut c_void,
            k_scales: *mut f32,
            v_scales: *mut f32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_int8_per_token_head" => "device::false_type::value",
            "attn::write_kv_fp8_per_token_head" => "device::true_type::value",
        }

        /// `:562` — the fp4 append, two values to the byte, blocked scales.
        fn write_kv_fp4_block = "attn::device::write_kv_fp4_block"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut u8,
            v_pages: *mut u8,
            k_scales: *mut f32,
            v_scales: *mut f32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            block_size: i32,
        ) {
            "attn::write_kv_fp4_block" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:655` — the per-tensor fp8 dequantiser over the active pages.
        fn dequant_fp8_pages_active = "attn::device::dequant_fp8_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_elems: i32,
            kind: fp8_kind,
        ) {
            "attn::dequant_fp8_pages_active_bf16" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:678` — the per-token-per-head fp8 dequantiser.
        fn dequant_fp8_per_token_head = "attn::device::dequant_fp8_per_token_head_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::dequant_fp8_per_token_head_pages_active_bf16" => "device::bf16",
        }

        /// `:708` — the same, for int8 pages.
        fn dequant_int8_per_token_head = "attn::device::dequant_int8_per_token_head_pages_active"(
            k_pages: *const i8,
            v_pages: *const i8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::dequant_int8_per_token_head_pages_active_bf16" => "device::bf16",
        }

        /// `:736` — the fp4 dequantiser.
        fn dequant_fp4 = "attn::device::dequant_fp4_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            logical_n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
            block_size: i32,
        ) {
            "attn::dequant_fp4_pages_active_bf16" => "device::bf16",
        }

        /// `:826` — the window page view, one thread walking every request.
        fn build_window_page_view = "attn::device::build_window_page_view"(
            src_indices: *const u32,
            src_indptr: *const u32,
            keep_pages: i32,
            dst_indptr: *mut u32,
            dst_indices: *mut u32,
            r: i32,
        ) {
            "attn::build_window_page_view" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:860` — the full split view, one warp.
        fn build_full_split_view = "attn::device::build_full_split_view"(
            src_indptr: *const u32,
            src_last_page_len: *const u32,
            splits: i32,
            page_size: i32,
            dst_indptr: *mut u32,
            dst_indices: *mut u32,
            dst_last: *mut u32,
            src_indices: *const u32,
        ) {
            "attn::build_full_split_view" => crate::device::DeviceKernel::PLAIN,
        }
    }

    /// `kv_paged.cu`'s `constexpr int BLOCK = 256`, which every launch in
    #[cfg(feature = "_cuda")]
    const BLOCK: u32 = 256;

    /// `::__nv_fp8_interpretation_t`'s two values, by the names the header
    #[cfg(feature = "_cuda")]
    const NV_E4M3: u32 = 0;
    #[cfg(feature = "_cuda")]
    const NV_E5M2: u32 = 1;

    /// The interpretation an fp8 page is written and read under.
    #[cfg(feature = "_cuda")]
    fn fp8_kind_of(storage_dtype: KvDType) -> fp8_kind {
        fp8_kind(if matches!(storage_dtype, KvDType::Fp8E5M2) { NV_E5M2 } else { NV_E4M3 })
    }

    /// NVFP4's block, when the layer states none.
    #[cfg(feature = "_cuda")]
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
    /// # Safety
    ///
    /// Every pointer must be a device allocation of the stated extent, and
    /// `stream` a live stream.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_explicit_bf16(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        b: i32,
        stream: *mut c_void,
        row_valid: *const u8,
    ) -> Fired {
        assert!(
            layer.is_native_bf16,
            "attn::write_kv_explicit_bf16 requires native bf16 KV cache"
        );
        if b <= 0 {
            return Fired::Declined(Refusal::Empty { what: "rows" });
        }

        let symbol = if layer.hnd {
            "attn::write_kv_explicit_bf16_dev#hnd"
        } else {
            "attn::write_kv_explicit_bf16_dev#nhd"
        };
        unsafe {
            raw::write_kv_explicit(
                symbol,
                Launch::per_row(b.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                w_page,
                w_off,
                MaybeConst::new(row_valid),
                b,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            );
        }

        if layer.has_envelopes && !layer.hnd {
            let _ = unsafe {
                crate::x::layout::envelope_merge_written(
                    k_curr,
                    w_page,
                    w_off,
                    MaybeConst::new(row_valid),
                    layer.k_env_min.cast(),
                    layer.k_env_max.cast(),
                    b,
                    layer.num_kv_heads,
                    layer.head_dim,
                    stream,
                )
            };
        }
        Fired::Launched
    }

    /// `attn::write_kv_explicit_bf16_devwin` — the same write with a
    ///
    /// # Safety
    ///
    /// As [`write_kv_explicit_bf16`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_explicit_bf16_devwin(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        win_d: *const u32,
        n_max: i32,
        stream: *mut c_void,
        row_valid: *const u8,
    ) -> Fired {
        assert!(
            layer.is_native_bf16,
            "attn::write_kv_explicit_bf16_devwin requires native bf16 KV cache"
        );
        if n_max <= 0 {
            return Fired::Declined(Refusal::Empty { what: "lanes" });
        }
        assert!(
            !layer.has_envelopes,
            "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
             windowed — use the host-window form"
        );

        let symbol = if layer.hnd {
            "attn::write_kv_explicit_bf16_devwin_dev#hnd"
        } else {
            "attn::write_kv_explicit_bf16_devwin_dev#nhd"
        };
        unsafe {
            raw::write_kv_explicit_devwin(
                symbol,
                Launch::per_row(n_max.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                w_page,
                w_off,
                MaybeConst::new(row_valid),
                win_d,
                n_max,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            );
        }
        Fired::Launched
    }

    /// The native-bf16 append, `kv_paged.cu:60-120`.
    ///
    /// # Safety
    ///
    /// As [`write_kv_explicit_bf16`]; the four CSR arrays must describe
    /// `num_requests` requests over `total_tokens` tokens.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages_bf16(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
        row_valid: *const u8,
        first_token: i32,
    ) -> Fired {
        let launch_tokens = total_tokens - first_token;
        if launch_tokens <= 0 {
            return Fired::Declined(Refusal::Empty { what: "tokens after first_token" });
        }

        let symbol =
            if layer.hnd { "attn::write_kv_bf16#hnd" } else { "attn::write_kv_bf16#nhd" };
        unsafe {
            raw::write_kv(
                symbol,
                Launch::per_row(launch_tokens.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                MaybeConst::new(row_valid),
                MaybeConst::none(),
                num_requests,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                first_token,
                stream,
            );
        }

        if layer.has_envelopes && !layer.hnd && total_tokens > 0 {
            let _ = unsafe {
                crate::x::layout::envelope_update_appended(
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
                    stream,
                )
            };
        }
        Fired::Launched
    }

    /// The quantised append, `kv_paged.cu:130-190` — four schemes, three
    ///
    /// # Safety
    ///
    /// As [`write_kv_to_pages_bf16`]; the layer's scale planes must be
    /// sized for its scheme.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages_quantised(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
    ) -> Fired {
        if total_tokens <= 0 {
            return Fired::Declined(Refusal::Empty { what: "tokens" });
        }
        let page_size = layer.page_size;
        let h_kv = layer.num_kv_heads;
        let d = layer.head_dim;
        let tokens = total_tokens.unsigned_abs();
        let heads = h_kv.unsigned_abs();

        match layer.scheme {
            KvScheme::Fp8PerTensor => unsafe {
                raw::write_kv_fp8_per_tensor(
                    "attn::write_kv_fp8_per_tensor",
                    Launch::per_row(tokens, BLOCK),
                    k_curr,
                    v_curr,
                    layer.k_pages.cast::<u8>(),
                    layer.v_pages.cast::<u8>(),
                    qo_indptr,
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_lens,
                    num_requests,
                    page_size,
                    h_kv,
                    d,
                    fp8_kind_of(layer.storage_dtype),
                    stream,
                );
            },

            KvScheme::Int8PerTokenHead | KvScheme::Fp8PerTokenHead => {
                let symbol = if matches!(layer.scheme, KvScheme::Fp8PerTokenHead) {
                    "attn::write_kv_fp8_per_token_head"
                } else {
                    "attn::write_kv_int8_per_token_head"
                };
                let smem = 2 * (BLOCK / 32) * (core::mem::size_of::<f32>() as u32);
                let launch = Launch {
                    grid: [tokens, heads, 1],
                    block: [BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(smem);
                unsafe {
                    raw::write_kv_per_token_head(
                        symbol,
                        launch,
                        k_curr,
                        v_curr,
                        layer.k_pages,
                        layer.v_pages,
                        layer.k_scales.cast::<f32>(),
                        layer.v_scales.cast::<f32>(),
                        qo_indptr,
                        kv_page_indices,
                        kv_page_indptr,
                        kv_last_page_lens,
                        num_requests,
                        page_size,
                        h_kv,
                        d,
                        stream,
                    );
                }
            }

            KvScheme::Fp4Block => {
                let block_size = fp4_block_size(layer);
                let blocks = d.div_euclid(block_size) + i32::from(d.rem_euclid(block_size) != 0);
                let launch = Launch {
                    grid: [tokens, heads, blocks.unsigned_abs()],
                    block: [32, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                };
                unsafe {
                    raw::write_kv_fp4_block(
                        "attn::write_kv_fp4_block",
                        launch,
                        k_curr,
                        v_curr,
                        layer.k_pages.cast::<u8>(),
                        layer.v_pages.cast::<u8>(),
                        layer.k_scales.cast::<f32>(),
                        layer.v_scales.cast::<f32>(),
                        qo_indptr,
                        kv_page_indices,
                        kv_page_indptr,
                        kv_last_page_lens,
                        num_requests,
                        page_size,
                        h_kv,
                        d,
                        block_size,
                        stream,
                    );
                }
            }

            KvScheme::Native => {
                return Fired::Declined(Refusal::Absent {
                    what: "a quantised writer for Native storage",
                });
            }
        }
        Fired::Launched
    }

    /// `attn::write_kv_to_pages` — the entry point, which chooses.
    ///
    /// # Safety
    ///
    /// As [`write_kv_to_pages_bf16`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
        row_valid: *const u8,
        first_token: i32,
    ) -> Fired {
        assert!(
            first_token == 0 || layer.is_native_bf16,
            "attn::write_kv_to_pages: partial (first_token) writes require the \
             native bf16 cache"
        );
        if layer.is_native_bf16 {
            return unsafe {
                write_kv_to_pages_bf16(
                    layer,
                    k_curr,
                    v_curr,
                    qo_indptr,
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_lens,
                    total_tokens,
                    num_requests,
                    stream,
                    row_valid,
                    first_token,
                )
            };
        }
        unsafe {
            write_kv_to_pages_quantised(
                layer,
                k_curr,
                v_curr,
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                total_tokens,
                num_requests,
                stream,
            )
        }
    }

    /// The fp8-per-tensor arm, called by name from
    ///
    /// # Safety
    ///
    /// `kv_page_indices` must list `num_pages_in_batch` valid page indices,
    /// and the layer's bf16 mirror planes must be sized for them.
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub unsafe fn dequant_fp8_per_tensor_pages_active(
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
        stream: *mut c_void,
    ) -> Fired {
        if layer.is_native_bf16 {
            return Fired::Declined(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if num_pages_in_batch <= 0 {
            return Fired::Declined(Refusal::Empty { what: "active pages" });
        }
        if !matches!(layer.scheme, KvScheme::Fp8PerTensor) {
            return Fired::Declined(Refusal::Absent { what: "an fp8-per-tensor layer" });
        }

        let (logical_n, page_elems, launch) = active_geometry(layer, num_pages_in_batch);
        unsafe {
            raw::dequant_fp8_pages_active(
                "attn::dequant_fp8_pages_active_bf16",
                launch,
                layer.k_pages.cast::<u8>().cast_const(),
                layer.v_pages.cast::<u8>().cast_const(),
                layer.k_bf16_pages.cast::<bf16>(),
                layer.v_bf16_pages.cast::<bf16>(),
                kv_page_indices,
                logical_n,
                page_elems,
                fp8_kind_of(layer.storage_dtype),
                stream,
            );
        }
        Fired::Launched
    }

    /// The element count an active-page pass covers, and the grid that
    #[cfg(feature = "_cuda")]
    fn active_geometry(layer: &KvLayer, num_pages_in_batch: i32) -> (i64, i32, Launch) {
        let page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
        let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
        let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
        let launch = Launch {
            grid: [blocks as u32, 1, 1],
            block: [BLOCK, 1, 1],
            smem: 0,
            smem_opt_in: false,
        };
        (logical_n, page_elems, launch)
    }

    /// `attn::dequant_kv_cache_layer_to_bf16_active` — dequantise the pages
    ///
    /// # Safety
    ///
    /// As [`dequant_fp8_per_tensor_pages_active`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub unsafe fn dequant_kv_cache_layer_to_bf16_active(
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
        stream: *mut c_void,
    ) -> Fired {
        if layer.is_native_bf16 {
            return Fired::Declined(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if num_pages_in_batch <= 0 {
            return Fired::Declined(Refusal::Empty { what: "active pages" });
        }
        let (logical_n, _page_elems, launch) = active_geometry(layer, num_pages_in_batch);

        match layer.scheme {
            KvScheme::Fp8PerTensor => unsafe {
                dequant_fp8_per_tensor_pages_active(
                    layer,
                    kv_page_indices,
                    num_pages_in_batch,
                    stream,
                )
            },

            KvScheme::Fp8PerTokenHead => {
                unsafe {
                    raw::dequant_fp8_per_token_head(
                        "attn::dequant_fp8_per_token_head_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<u8>().cast_const(),
                        layer.v_pages.cast::<u8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        stream,
                    );
                }
                Fired::Launched
            }

            KvScheme::Int8PerTokenHead => {
                unsafe {
                    raw::dequant_int8_per_token_head(
                        "attn::dequant_int8_per_token_head_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<i8>().cast_const(),
                        layer.v_pages.cast::<i8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        stream,
                    );
                }
                Fired::Launched
            }

            KvScheme::Fp4Block => {
                unsafe {
                    raw::dequant_fp4(
                        "attn::dequant_fp4_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<u8>().cast_const(),
                        layer.v_pages.cast::<u8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        fp4_block_size(layer),
                        stream,
                    );
                }
                Fired::Launched
            }

            KvScheme::Native => {
                Fired::Declined(Refusal::Absent { what: "a quantised dequant for Native storage" })
            }
        }
    }
}

pub static UNITS: &[Unit] = &[
    attention_flashinfer::ATTENTION_FLASHINFER,
    attention_naive::ATTENTION_NAIVE,
    attention_naive_paged::ATTENTION_NAIVE_PAGED,
    attn_res::ATTN_RES,
    attn_sink::ATTN_SINK,
    dsa_indexer::DSA_INDEXER,
    dsv4_compress::DSV4_COMPRESS,
    head_dim_pad::HEAD_DIM_PAD,
    kimi_mla::KIMI_MLA,
    kv_paged::KV_PAGED,
    mla_fa2::MLA_FA2,
    mla_naive::MLA_NAIVE,
    mla_paged::MLA_PAGED,
    pack_dense_mask::PACK_DENSE_MASK,
    page_compact::PAGE_COMPACT,
    qkv_fused::QKV_FUSED,
    softcap::SOFTCAP,
    split_packed::SPLIT_PACKED,
];

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
    Launch {
        grid: [rows, heads, 1],
        block: [head_dim_block(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
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
    Launch { grid: [heads, rows, 1], block: [PAD_BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// The merge's geometry — the grid of [`per_head_elementwise`] and a
#[must_use]
const fn combine_attn(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [combine_block(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
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
/// # Safety
///
/// `lse` must address `n` live, writable `f32`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn lse_log2_to_ln(lse: *mut f32, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "lse elements" });
    }
    let Ok(elems) = u32::try_from(n) else {
        return Fired::Declined(Refusal::Wide {
            what: "lse elements",
            at: i32::MAX,
            max: i32::MAX,
        });
    };
    unsafe {
        attn_sink::raw::lse_log2_to_ln(
            "attn::lse_log2_to_ln",
            elementwise(elems),
            lse,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attention_sink_rescale_bf16` — gpt-oss's per-head sink correction,
///
/// # Safety
///
/// `o` addresses `n * num_q_heads * head_dim` live, writable bf16 elements;
/// `lse` addresses `n * num_q_heads` live `f32`s; `sinks` addresses
/// `num_q_heads` live bf16 elements. All three live on `stream`, which must
/// outlive the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attention_sink_rescale_bf16(
    o: *mut bf16,
    lse: *const f32,
    sinks: *const bf16,
    n: i32,
    num_q_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if num_q_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        attn_sink::raw::attn_sink_rescale(
            "attn::attention_sink_rescale_bf16",
            per_head_elementwise(n.unsigned_abs(), num_q_heads.unsigned_abs(), head_dim.unsigned_abs()),
            o,
            lse,
            sinks,
            n,
            num_q_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `split_packed.cu:30` — `constexpr int BLOCK = 256;`.
#[cfg(feature = "_cuda")]
pub const SPLIT_BLOCK: u32 = 256;

/// `attn::split_qkv_bf16_devwin` — the packed activation cut into Q, K and V,
///
/// # Safety
///
/// `packed`, the three outputs and `win` are device addresses live across the
/// launch, and `stream` is the caller's. The four buffer pointers must be
/// BASE pointers — the kernel windows them itself from `win`, so a
/// pre-windowed pointer is windowed twice. The binder guarantees it by the
/// `_devwin` suffix; a hand caller must not.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn split_qkv_bf16_devwin(
    packed: *const bf16,
    q_out: *mut bf16,
    k_out: *mut bf16,
    v_out: *mut bf16,
    win: *const u32,
    n_max: i32,
    q_dim: i32,
    kv_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n_max <= 0 {
        return Fired::Declined(Refusal::Empty { what: "lanes" });
    }
    let max_dim = if q_dim > kv_dim { q_dim } else { kv_dim };
    if max_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "output width" });
    }
    let xblocks = max_dim.unsigned_abs().div_ceil(SPLIT_BLOCK);
    unsafe {
        split_packed::raw::split_qkv_devwin(
            "attn::split_qkv_devwin",
            Launch {
                grid: [xblocks.max(1), n_max.unsigned_abs(), 1],
                block: [SPLIT_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            packed,
            q_out,
            k_out,
            v_out,
            win,
            q_dim,
            kv_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `attention_naive_paged.cuh:33` — `constexpr int BLOCK = 128`.
#[cfg(feature = "_cuda")]
pub const PAGED_BLOCK: u32 = 128;

/// `attention_naive_paged.cuh:223` — `constexpr int kMaxHeadDim = 1024`.
#[cfg(feature = "_cuda")]
pub const PAGED_MAX_HEAD_DIM: i32 = 1024;

/// `attn::attention_naive_paged` — the reference paged attention.
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the kernel
/// reads or writes, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attention_naive_paged(
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
    stream: *mut c_void,
) -> Fired {
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if layer.head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the cache's head dim" });
    }
    if layer.head_dim > PAGED_MAX_HEAD_DIM {
        return Fired::Declined(Refusal::Wide {
            what: "head_dim",
            at: layer.head_dim,
            max: PAGED_MAX_HEAD_DIM,
        });
    }
    let num_q_heads = q_width / layer.head_dim;
    if num_q_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "q heads" });
    }
    if layer.num_kv_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "kv heads" });
    }
    let smem = (layer.head_dim.unsigned_abs() + PAGED_BLOCK) * 4;
    unsafe {
        attention_naive_paged::raw::naive_paged_attn(
            "attn::attention_naive_paged_dev",
            Launch {
                grid: [
                    num_requests.unsigned_abs(),
                    total_tokens.unsigned_abs(),
                    num_q_heads.unsigned_abs(),
                ],
                block: [PAGED_BLOCK, 1, 1],
                smem,
                smem_opt_in: false,
            },
            q,
            layer.k_pages.cast_const(),
            layer.v_pages.cast_const(),
            layer.k_scales.cast::<f32>().cast_const(),
            layer.v_scales.cast::<f32>().cast_const(),
            o,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            core::ptr::null(),
            core::ptr::null(),
            num_q_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.page_size,
            kv_scheme::of(layer.scheme),
            kv_dtype::of(layer.storage_dtype),
            layer.block_size,
            window_left,
            sm_scale,
            logits_soft_cap,
            lse_out,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attn_res_blend_bf16` — K3's residual-block blend.
///
/// # Safety
///
/// `prefix` and `out` address `t * h` live bf16 elements, `blocks` addresses
/// `t * b * h`, and `norm_weight` and `proj_weight` address `h` each. `out`
/// is writable and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attn_res_blend_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if t <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if b <= 0 {
        return Fired::Declined(Refusal::Empty { what: "blocks" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        attn_res::raw::attn_res_blend(
            "attn::attn_res_blend_bf16",
            Launch::per_row(t.unsigned_abs(), BLOCK),
            prefix,
            blocks,
            norm_weight,
            proj_weight,
            out,
            b,
            h,
            block_rows,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::pad_head_dim_bf16` — pad every head out to a width flashinfer
///
/// # Safety
///
/// `packed` addresses `num_tokens * num_heads * head_dim` live bf16 elements
/// and `padded` addresses `num_tokens * num_heads * head_dim_padded`
/// writable ones. `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn pad_head_dim_bf16(
    packed: *const bf16,
    padded: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
    stream: *mut c_void,
) -> Fired {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Fired::Declined(why);
    }
    unsafe {
        head_dim_pad::raw::pad_head_dim(
            "attn::pad_head_dim_bf16",
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            packed,
            padded,
            num_heads,
            head_dim,
            head_dim_padded,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::strip_head_dim_bf16` — the inverse of [`pad_head_dim_bf16`].
///
/// # Safety
///
/// `padded` addresses `num_tokens * num_heads * head_dim_padded` live bf16
/// elements and `packed` addresses `num_tokens * num_heads * head_dim`
/// writable ones. `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn strip_head_dim_bf16(
    padded: *const bf16,
    packed: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
    stream: *mut c_void,
) -> Fired {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Fired::Declined(why);
    }
    unsafe {
        head_dim_pad::raw::strip_head_dim(
            "attn::strip_head_dim_bf16",
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            padded,
            packed,
            num_heads,
            head_dim,
            head_dim_padded,
            stream,
        );
    }
    Fired::Launched
}

/// The four preconditions both head-dim launchers share, resolved BEFORE
#[cfg(feature = "_cuda")]
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
        return Some(Refusal::Narrow { what: "head_dim_padded", at: head_dim_padded });
    }
    None
}

/// The guard `attn_softcap.cu`'s launcher opened with, as a refusal.
#[cfg(feature = "_cuda")]
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
            at: i32::MAX,
            max: i32::MAX,
        });
    };
    Ok(elementwise(elems))
}

/// `attn::logit_softcap_bf16` — gemma's final logit cap, in place.
///
/// # Safety
///
/// `x` must address `n` live, writable `bf16`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn logit_softcap_bf16(
    x: *mut bf16,
    cap: f32,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    let launch = match softcap_launch(cap, n) {
        Ok(launch) => launch,
        Err(refusal) => return Fired::Declined(refusal),
    };
    unsafe {
        softcap::raw::logit_softcap("attn::logit_softcap_bf16", launch, x, cap, n, stream);
    }
    Fired::Launched
}

/// `attn::logit_softcap_f16` — the same cap over an fp16 buffer.
///
/// # Safety
///
/// `x` must address `n` live, writable `f16`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn logit_softcap_f16(x: *mut f16, cap: f32, n: usize, stream: *mut c_void) -> Fired {
    let launch = match softcap_launch(cap, n) {
        Ok(launch) => launch,
        Err(refusal) => return Fired::Declined(refusal),
    };
    unsafe {
        softcap::raw::logit_softcap("attn::logit_softcap_f16", launch, x, cap, n, stream);
    }
    Fired::Launched
}

/// `attn::kimi_split_q_b_bf16` — split a fused query projection into its
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn kimi_split_q_b_bf16(
    q_b: *const bf16,
    q_nope: *mut bf16,
    q_pe: *mut bf16,
    tokens: i32,
    heads: i32,
    nope: i32,
    rope: i32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if nope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_nope_head_dim" });
    }
    if rope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_rope_head_dim" });
    }
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
    let total = i64::from(tokens) * width;
    if total > i64::from(i32::MAX) {
        return Fired::Declined(Refusal::Wide {
            what: "rows",
            at: tokens,
            max: i32::try_from(i64::from(i32::MAX) / width).unwrap_or(i32::MAX),
        });
    }
    let total = total as i32;
    unsafe {
        kimi_mla::raw::split_q_b(
            "attn::kimi_split_q_b_bf16",
            elementwise(total.unsigned_abs()),
            q_b,
            q_nope,
            q_pe,
            total,
            heads,
            nope,
            rope,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::kimi_split_kv_a_norm_bf16` — split `kv_a`, RMS-normalise the latent
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn kimi_split_kv_a_norm_bf16(
    kv_a: *const bf16,
    norm_weight: *const bf16,
    kv_c: *mut bf16,
    k_pe: *mut bf16,
    tokens: i32,
    kv_lora: i32,
    rope: i32,
    src_row_stride: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if kv_lora <= 0 {
        return Fired::Declined(Refusal::Empty { what: "kv_lora_rank" });
    }
    if rope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_rope_head_dim" });
    }
    if src_row_stride < kv_lora + rope {
        return Fired::Declined(Refusal::Narrow { what: "src_row_stride", at: src_row_stride });
    }
    unsafe {
        kimi_mla::raw::split_kv_a_norm(
            "attn::kimi_split_kv_a_norm_bf16",
            rms(tokens.unsigned_abs()),
            kv_a,
            norm_weight,
            kv_c,
            k_pe,
            kv_lora,
            rope,
            src_row_stride,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::combine_attn_outputs_bf16` — merge two attention halves and their
///
/// # Safety
///
/// Every pointer must address the extents these three numbers describe, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn combine_attn_outputs_bf16(
    o1: *const bf16,
    lse1: *const f32,
    o2: *const bf16,
    lse2: *const f32,
    o_out: *mut bf16,
    lse_out: *mut f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        dsv4_compress::raw::combine_attn_outputs(
            "attn::combine_attn_outputs_dev",
            combine_attn(n.unsigned_abs(), num_heads.unsigned_abs(), head_dim.unsigned_abs()),
            o1,
            lse1,
            o2,
            lse2,
            o_out,
            lse_out,
            num_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
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
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
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
    Fired::Launched
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
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
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
    Fired::Launched
}

contract! {
    /// FlashInfer publishes its LSE in log2 and the sink correction works in
    LSE_LOG2_TO_LN = "attn::lse_log2_to_ln" as lse_log2_to_ln {
        in_place: &[(0, 0)],
    }

    /// Rescales the attention output IN PLACE against the per-head sink
    ATTENTION_SINK_RESCALE = "attn::attention_sink_rescale_bf16" as attention_sink_rescale {
        in_place: &[(0, 0)],
    }

    /// K3's residual-block blend: a prefix and `B` candidate blocks, scored
    ATTN_RES_BLEND = "attn::attn_res_blend_bf16" as attn_res_blend

    /// The pad half of what `head_dim_padded` COSTS.
    PAD_HEAD_DIM = "attn::pad_head_dim_bf16" as pad_head_dim

    /// The strip half. See [`PAD_HEAD_DIM`].
    STRIP_HEAD_DIM = "attn::strip_head_dim_bf16" as strip_head_dim

    /// Gemma's final logit cap — `cap * tanh(x / cap)` over the logits,
    LOGIT_SOFTCAP = "attn::logit_softcap_bf16" as logit_softcap {
        in_place: &[(0, 0)],
    }

    /// kimi_k3's fused query projection, split into the halves attention
    KIMI_SPLIT_Q_B = "attn::kimi_split_q_b_bf16" as kimi_split_q_b

    /// The key/value half of the same split, with an RMS norm fused into it.
    KIMI_SPLIT_KV_A_NORM = "attn::kimi_split_kv_a_norm_bf16" as kimi_split_kv_a_norm

    /// deepseek_v4's compressed-cache gather — one entry per boundary token,
    DSV4_COMPRESS_GATHER_PAGED = "attn::dsv4_compress_gather_paged_bf16"
        as dsv4_compress_gather_paged {
        sink: Some("kv.compressed"),
    }

    /// The commit half — those entries into the compressed cache, each at its
    DSV4_STORE_COMP_ENTRIES = "attn::dsv4_store_comp_entries_bf16"
        as dsv4_store_comp_entries {
        whole: true,
    }

    /// The merge that puts deepseek_v4's two attention branches back
    COMBINE_ATTN_OUTPUTS = "attn::combine_attn_outputs_bf16" as combine_attn_outputs

    /// The fused QKV prefill epilogue — six statements in one launch, and
    QKV_PACKED_POST = "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" as qkv_packed_post {
        sink: Some("kv.pages"),
    }

    /// The decode form of the same fusion, and the last row `ROW_TABLES` had.
    QKV_DECODE_FUSED = "attn::qkv_decode_qk_norm_rope_write_kv_bf16" as qkv_decode_fused {
        sink: Some("kv.pages"),
    }

    /// Attention over the latent cache — DeepSeek/Kimi MLA, the row that
    ATTENTION_MLA = "attn::dispatch_attention_mla_bf16" as attention_mla {
        needs: Prepare::MlaPlan,
        lacks: &[Cap::Scores],
    }
    /// The paged KV append — **the most-fired symbol in this family**, once
    WRITE_KV_TO_PAGES = "attn::write_kv_to_pages" as write_kv_to_pages

    /// The explicit-slot append: the fire states each row's destination page
    WRITE_KV_EXPLICIT = "attn::write_kv_explicit_bf16" as write_kv_explicit

    /// The same write over a device-carried window.
    WRITE_KV_EXPLICIT_DEVWIN = "attn::write_kv_explicit_bf16_devwin" as write_kv_explicit_devwin {
        whole: true,
        sink: Some("kv.pages"),
    }

    /// Dequantise the pages this batch touches into the layer's bf16 mirror.
    DEQUANT_KV_ACTIVE = "attn::dequant_kv_cache_layer_to_bf16_active" as dequant

    /// The plain paged decode.
    FA2_DECODE = "attn::dispatch_attention_flashinfer_decode" as fa2_decode {
        needs: Prepare::DecodePlan,
        sink: Some("kv.pages"),
        depth_prefix_plan: true,
    }

    /// The same decode with the attention scores captured.
    FA2_DECODE_CAPTURE = "attn::dispatch_attention_flashinfer_decode_capture" as fa2_decode_capture {
        needs: Prepare::DecodePlan,
        sink: Some("kv.pages"),
    }

    /// The plain paged prefill.
    FA2_PREFILL = "attn::dispatch_attention_flashinfer_prefill_bf16" as fa2_prefill {
        needs: Prepare::PrefillPlan,
        sink: Some("kv.pages"),
    }

    /// The prefill sibling of the capturing decode — SnapKV's half.
    FA2_PREFILL_CAPTURE = "attn::dispatch_attention_flashinfer_prefill_capture_bf16" as fa2_prefill_capture {
        needs: Prepare::PrefillPlan,
        sink: Some("kv.pages"),
    }

    /// The custom-mask prefill.
    FA2_PREFILL_CUSTOM = "attn::dispatch_attention_flashinfer_prefill_custom" as fa2_prefill_custom {
        needs: Prepare::CustomPlan,
        sink: Some("kv.pages"),
    }

    /// The PLANLESS prefill: it plans and fires in one call.
    FA2_PREFILL_PLANLESS = "attn::attention_flashinfer_prefill" as fa2_prefill_planless {
        whole: true,
        needs: Prepare::FireWide,
        sink: Some("kv.pages"),
    }

    /// MLA's first absorption: `q_nope` into the latent basis.
    MLA_ABSORB_Q_TO_LATENT = "gemm::mla_absorb_q_to_latent_bf16" as mla_absorb_q_to_latent

    /// MLA's second absorption: the attention latent back out to `v`.
    MLA_ABSORB_LATENT_TO_V = "gemm::mla_absorb_latent_to_v_bf16" as mla_absorb_latent_to_v

    /// LayerNorm then interleaved RoPE on the indexer's keys.
    DSA_INDEX_KNORM_ROPE = "attn::dsa_index_knorm_rope_bf16" as dsa_index_knorm_rope {
        whole: true,
    }

    /// Interleaved RoPE on the indexer's queries.
    DSA_INDEX_Q_ROPE = "attn::dsa_index_q_rope_bf16" as dsa_index_q_rope {
        whole: true,
    }

    /// The causal top-k mask over the index scores. The one of the three
    DSA_INDEX_TOPK_MASK = "attn::dsa_index_topk_mask" as dsa_index_topk_mask {
        whole: true,
    }

    /// The decode row's compressed-block boundary metadata.
    DSV4_BOUNDARY_META_DECODE = "attn::dsv4_boundary_meta_decode" as dsv4_boundary_meta_decode {
        whole: true,
    }

    /// The prefill form, resolving the request by binary search.
    DSV4_BOUNDARY_META_PAGED = "attn::dsv4_boundary_meta_paged" as dsv4_boundary_meta_paged {
        whole: true,
    }

    /// Attention over the COMPRESSED KV pages.
    DSV4_ATTENTION_COMPRESSED_PAGED =
        "attn::attention_compressed_paged_bf16" as attention_compressed_paged {
        whole: true,
    }

    /// The whole MLA prologue in one kernel.
    MLA_PREPARE = "attn::mla_prepare_bf16" as mla_prepare {
        whole: true,
    }

    /// One step's latent and rope plane, appended to the paged MLA cache.
    WRITE_MLA_TO_PAGES = "attn::write_mla_to_pages" as write_mla_to_pages {
        whole: true,
    }

    /// The page compactor — TWO launches, one stream, in order.
    COMPACT_PAGE_CSR = "attn::compact_page_csr" as compact_page_csr {
        whole: true,
    }

    /// MTP's shift: the previous step's pending hidden becomes this step's
    MTP_SHIFT_HIDDEN = "attn::mtp_shift_hidden_bf16" as mtp_shift_hidden {
        whole: true,
    }

    /// MTP's refresh: each request's last hidden becomes its slot's pending.
    MTP_UPDATE_PENDING_HIDDEN =
        "attn::mtp_update_pending_hidden_bf16" as mtp_update_pending_hidden {
        whole: true,
    }

    /// The packed QKV split over a DEVICE-RESIDENT row window.
    SPLIT_QKV_DEVWIN = "attn::split_qkv_bf16_devwin" as split_qkv_devwin

    /// The per-head → per-request fold of captured attention scores.
    ATTN_SCORE_FOLD_HEADS = "attn::attn_score_fold_heads" as attn_score_fold_heads {
        whole: true,
    }

    /// The reference paged attention — `table::attn`'s next-to-last row.
    ATTENTION_NAIVE_PAGED = "attn::attention_naive_paged" as attention_naive_paged {
        whole: true,
        sink: Some("kv.pages"),
    }
}

#[cfg(feature = "_cuda")]
bind! {
    LSE_LOG2_TO_LN => { cx, stream => {
        let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
        let Ok(n) = usize::try_from(elems) else {
            return Err(Refusal::Empty { what: "lse elements" });
        };
        unsafe { lse_log2_to_ln(cx.arg_out(0)?.cast::<f32>(), n, stream) }.ok()
    }},

    ATTENTION_SINK_RESCALE => { cx, stream => {
        unsafe {
            attention_sink_rescale_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.num_q_heads()?,
                cx.head_dim()?,
                stream,
            )
        }
        .ok()
    }},

    ATTN_RES_BLEND => { cx, stream => {
        let h = cx.out_width(0)?;
        let b = cx.in_width(1)? / h;
        unsafe {
            attn_res_blend_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_in(3)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                b,
                h,
                cx.rows().count,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    PAD_HEAD_DIM => { cx, stream => {
        let head_dim = cx.head_dim()?;
        let packed_width = cx.in_width(0)?;
        let num_heads = packed_width / head_dim;
        if num_heads <= 0 {
            return Err(Refusal::Narrow { what: "in_width(0)", at: packed_width });
        }
        unsafe {
            pad_head_dim_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                num_heads,
                head_dim,
                cx.out_width(0)? / num_heads,
                stream,
            )
        }
        .ok()
    }},

    STRIP_HEAD_DIM => { cx, stream => {
        let head_dim = cx.head_dim()?;
        let packed_width = cx.out_width(0)?;
        let num_heads = packed_width / head_dim;
        if num_heads <= 0 {
            return Err(Refusal::Narrow { what: "out_width(0)", at: packed_width });
        }
        unsafe {
            strip_head_dim_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                num_heads,
                head_dim,
                cx.in_width(0)? / num_heads,
                stream,
            )
        }
        .ok()
    }},

    LOGIT_SOFTCAP => { cx, stream => {
        let cap = cx.final_logit_softcap()?;
        let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
        let Ok(n) = usize::try_from(elems) else {
            return Err(Refusal::Narrow { what: "logit elements", at: elems });
        };
        unsafe { logit_softcap_bf16(cx.arg_out(0)?.cast::<bf16>(), cap, n, stream) }.ok()
    }},

    KIMI_SPLIT_Q_B => { cx, stream => {
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let heads = param(0)?;
        let nope = param(1)?;
        let rope = param(2)?;
        unsafe {
            kimi_split_q_b_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                heads,
                nope,
                rope,
                stream,
            )
        }
        .ok()
    }},

    KIMI_SPLIT_KV_A_NORM => { cx, stream => {
        unsafe {
            kimi_split_kv_a_norm_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.out_width(1)?,
                cx.in_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    DSA_INDEX_TOPK_MASK => { cx, stream => {
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        unsafe {
            dsa_index_topk_mask_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<u8>(),
                cx.rows().count,
                param(0)?,
                param(1)?,
                param(2)?,
                stream,
            )
        }
        .ok()
    }},

    DSA_INDEX_Q_ROPE => { none:
        "the indexer's query rotation is not a shape this trace states: \
         `dsl::cuda::dsa_index_q_rope` records ONE input and NO parameters, \
         and puts `heads` and `head_dim` into the RESULT SHAPE only -- so \
         `out_width(0)` is their product and nothing splits it -- while \
         `rope_dim` appears in no statement, no shape and no context at all. \
         The host program is written, in `x::attn::dsa_index_q_rope_bf16`, \
         and what it is waiting for is a statement rather than a query"
    },

    DSA_INDEX_KNORM_ROPE => { none:
        "the key half is blocked by the same statement and by one more: \
         `dsl::cuda::dsa_index_knorm_rope` names NO weight bank, and the \
         kernel reads a LayerNorm weight AND a bias -- two operands with \
         nothing to come from, on top of the `rope_dim` its sibling also \
         lacks. `head_dim` alone is statable, as `out_width(0)`. The host \
         program is `x::attn::dsa_index_knorm_rope_bf16` and is complete"
    },

    ATTN_SCORE_FOLD_HEADS => { none: "`score_indptr_d` -- the score-capture CSR, \
        which says where each request's rows begin in the folded sink. Eight of \
        the nine operands are queries that exist: `scores` is `arg_in(0)`, \
        `folded` is `arg_out(0)`, three come off `plan()`, `num_q_heads` and \
        `page_size` off `num_q_heads()` and `kv_layer()`. The ninth is an \
        `AttnCtx` field with a real producer -- `attn_score::DecodeScoreCapturePlan` \
        publishes it as an arena-stable device base -- and no `Cx` query reaches \
        it. Same shape as `first_token` and `w_page_d` before they landed, and \
        NOT `Cx::mla_layer`'s shape, which refuses because nothing fills it" },

    SPLIT_QKV_DEVWIN => { cx, stream => {
        let win = cx.peel_window()?;
        unsafe {
            split_qkv_bf16_devwin(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.arg_out(2)?.cast::<bf16>(),
                win.as_ptr().cast_const(),
                cx.rows().total,
                cx.out_width(0)?,
                cx.out_width(1)?,
                stream,
            )
        }
        .ok()
    }},

    COMPACT_PAGE_CSR => { none:
        "the statement declares ONE result and the kernel writes THREE CSR \
         arrays plus a scratch: `dsl::cuda::compact_page_csr` records one \
         input, no parameters, a `StateRef` and a single `[Requests] I32` \
         result, so `arg_out(0)` answers one of `page_indptr_out`, \
         `last_page_lens_out` and `page_indices_out` and there is no way to \
         say WHICH -- while `scratch_counts`, the buffer that carries the \
         dependency BETWEEN the two launches, and `keep_stride` have nothing \
         at all. Six of eleven ARE answered: `keep` is `arg_in(0)`, the three \
         CSR inputs and `num_requests` come off `plan()`. The host program is \
         `x::attn::compact_page_csr`, both launches in order with both \
         refusals hoisted ahead of the first, and it is complete"
    },

    MTP_SHIFT_HIDDEN => { none:
        "ONE operand of nine, and it is `slot_ids`: the only query that \
         reaches a request->slot map is `Cx::gdn()`, whose `slot_ids_d` is \
         exactly this pointer, and `Facts::gdn` answers `None` unless the \
         fire has a RECURRENT shape. An MTP head on a dense transformer has \
         none, so the query refuses for the fire that needs it. Everything \
         else is answered: `target_hidden` and `pending_hidden` are \
         `arg_in(0)` and `arg_in(1)` -- the statement hands the pending slab \
         over as an INPUT, so no `Slab` variant is wanted here -- `out` is \
         `arg_out(0)`, `qo_indptr` and `num_requests` come off `plan()`, \
         `total_tokens` is `rows()` and `hidden_size` is `out_width(0)`. The \
         host program is `x::attn::mtp_shift_hidden_bf16` and is complete"
    },

    MTP_UPDATE_PENDING_HIDDEN => { none:
        "its twin's `slot_ids`, and one more of a different kind: this \
         statement records NO result and a `StateRef { store: \
         RecurrentState }`, so `pending_hidden` -- which this kernel WRITES \
         -- is a slab reference rather than an argument, and `Slab` has two \
         variants, `Conv` and `Recurrent`, neither of which is the MTP \
         pending-hidden row. `RecurrentStateCache` carries it as a third \
         half, `Buffer::MtpHidden`, addressed by SLOT and not by layer, so it \
         is a slab kind rather than a stride on an existing one -- which is \
         the change `Slab`'s own doc asks for: `the next person to add a slab \
         kind adds a stride to Gdn in the same change`. `target_hidden` is \
         `arg_in(0)`, `hidden_size` is `in_width(0)`, `qo_indptr` and \
         `num_requests` come off `plan()`. The host program is \
         `x::attn::mtp_update_pending_hidden_bf16` and is complete"
    },

    MLA_PREPARE => { none:
        "`Cx::mla_layer` refuses, and it is the whole blocker: the two page \
         arrays, `page_size`, `kv_lora_rank` and `qk_rope_head_dim` all come \
         out of one view, so five of this kernel's thirty operands go \
         together or not at all. That query's refusal is STRUCTURAL and its \
         own doc says so -- `AttnCtx` carries `layers: Vec<KvCacheLayerView>` \
         and no MLA equivalent, and the views come from \
         `pools::mla_cache::MlaCachePool::layer_view`, which no `Fire` can \
         reach. This is `ATTENTION_MLA`'s refusal, one kernel earlier in the \
         same lane, and it is a DIFFERENT SHAPE from the `dsv4` three's \
         ratio: the ratio has no producer anywhere, and this has a producer \
         no fire reaches. Everything else is answered -- the four query \
         outputs and two KV outputs are `arg_out(0..5)`, `kv_a`/`q_b` are \
         `arg_in`, the norm weight is `weight(0)`, the four CSR arrays and \
         `row_valid` come off `plan()`, `eps` is `rms_eps()`, `theta` is \
         `rope_theta()`, `interleaved` is `rope_interleaved()` and `yarn` is \
         `yarn()`. The host program is `x::attn::mla_prepare_bf16` and is \
         complete"
    },

    WRITE_MLA_TO_PAGES => { none:
        "the same view, and nothing else missing: this kernel's thirteen \
         operands are two inputs, the four CSR arrays, `row_valid`, \
         `num_requests` -- all answered -- and the five that ARE the layer \
         view. `serve/load.rs` refuses every MLA checkpoint at load today, so \
         the refusal this states is the one a model would meet anyway, one \
         layer lower and in a sentence. The host program is \
         `x::attn::write_mla_to_pages` and is complete"
    },

    DSV4_BOUNDARY_META_DECODE => { none:
        "the compression RATIO is not a value this trace carries: \
         `dsl::cuda::dsv4_boundary_meta` records its inputs with \
         `record_many` and NO parameters, so the one integer the kernel \
         DIVIDES BY has no operand — and it appears in no `AttnCtx` field, \
         no `DispatchCtx` field and no `Facts` query either, so there is \
         nothing to answer it with. Everything ELSE is statable: `positions` \
         is `arg_in(0)`, the three outputs are `arg_out(0..2)`, `row_valid` \
         and `requests` come off `plan()`, and `n` is `rows()`. The host \
         program is `x::attn::dsv4_boundary_meta_decode` and is complete"
    },

    DSV4_BOUNDARY_META_PAGED => { none:
        "its twin's ratio, and nothing else: `qo_indptr` and `num_requests` \
         BOTH come off `plan()`, so the prefill form's two extra operands \
         are the two that are already answered. One statement carries both \
         rows -- `dsl::cuda::dsv4_boundary_meta` -- so a parameter added \
         there lands on both at once. The host program is \
         `x::attn::dsv4_boundary_meta_paged` and is complete"
    },

    DSV4_ATTENTION_COMPRESSED_PAGED => { none:
        "the same ratio and two buffers with no producer anywhere: \
         `comp_kv_pages` is deepseek_v4's COMPRESSED cache, which no pool \
         allocates and no context carries, and `req_of_token` is a \
         per-token request map that nothing in `driver-cuda` builds. \
         `sm_scale` is the one blocker of a different kind -- it HAS a \
         producer, `AttnCtx::sm_scale` at `bind/mod.rs:1489`, and six \
         generated arms read it -- so it is a query that could exist and \
         does not, where the other three are values that do not exist. \
         `q`, `o`, `lse_out`, `positions`, the two page arrays, \
         `total_tokens`, `num_q_heads`, `head_dim` and `page_size` are all \
         answered today. The host program is \
         `x::attn::attention_compressed_paged_bf16` and is complete"
    },

    DSV4_COMPRESS_GATHER_PAGED => { none:
        "deepseek_v4's compression state is not a value this trace names: \
         `dsl::cuda::dsv4_compress_gather_paged` records ONE input \
         (`boundary_pos`) and NO parameters for a kernel that reads five \
         buffers and three integers, so `state_kv`, `state_score`, `ape`, \
         `boundary_req`, `ratio` and `coff` have no operand to come from — \
         the host program is written, in `x::attn::dsv4_compress::\
         dsv4_compress_gather_paged_bf16`, and what it is waiting for is a \
         statement rather than a query"
    },

    DSV4_STORE_COMP_ENTRIES => { none:
        "the commit half is blocked by the same statement as the gather: \
         `dsl::cuda::dsv4_store_comp_entries` names `entries` and \
         `boundary_pos`, and the kernel also reads `boundary_req` — \
         `dsv4_boundary_meta_*`'s second output, which the trace discards — \
         and needs `head_dim` and `page_size` besides; the host program is \
         `x::attn::dsv4_compress::dsv4_store_comp_entries_bf16`"
    },

    COMBINE_ATTN_OUTPUTS => { cx, stream => {
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let num_heads = param(0)?;
        let head_dim = param(1)?;
        unsafe {
            combine_attn_outputs_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                num_heads,
                head_dim,
                stream,
            )
        }
        .ok()
    }},

    QKV_PACKED_POST => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        if layer.head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let num_q_heads = cx.out_width(0)? / layer.head_dim;
        unsafe {
            qkv_fused::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.positions()?,
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                plan.row_valid,
                cx.rows().count,
                num_q_heads,
                layer.num_kv_heads,
                layer.head_dim,
                layer.page_size,
                layer.hnd,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    ATTENTION_MLA => { none: "attention over the latent cache cannot be bound         because this driver does not build one: `fire/launch.rs`' `kv_pools_for`         refuses `KvStyle::Mla` and `serve/load.rs` refuses the checkpoint at         model load, so `Cx::mla_layer` and `Cx::mla_plan` have nothing to         answer with — both host programs are written, in `x::attn::mla_fa2` and         `driver-cuda/src/fire/mla_naive.rs`, and choosing between them needs a         device compute capability besides" },

    WRITE_KV_TO_PAGES => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        unsafe {
            kv_paged::write_kv_to_pages(
                &layer,
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.arg_in(1)?.cast::<bf16>().cast_const(),
                plan.qo_indptr,
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                cx.rows().count,
                plan.requests,
                stream,
                plan.row_valid,
                cx.first_token()?,
            )
        }
        .ok()
    }},

    WRITE_KV_EXPLICIT => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        unsafe {
            kv_paged::write_kv_explicit_bf16(
                &layer,
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.arg_in(1)?.cast::<bf16>().cast_const(),
                cx.w_page_d()?,
                cx.w_off_d()?,
                cx.rows().count,
                stream,
                plan.row_valid,
            )
        }
        .ok()
    }},

    ATTENTION_NAIVE_PAGED => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        unsafe {
            attention_naive_paged(
                &layer,
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.arg_out(0)?.cast::<bf16>(),
                plan.qo_indptr,
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                cx.rows().count,
                plan.requests,
                cx.in_width(0)?,
                cx.window_left()?,
                cx.sm_scale()?,
                cx.logits_soft_cap()?,
                cx.lse_out()?,
                stream,
            )
        }
        .ok()
    }},

    QKV_DECODE_FUSED => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        let w_page = cx.w_page_d().unwrap_or(core::ptr::null());
        let w_off = cx.w_off_d().unwrap_or(core::ptr::null());
        let head_dim = layer.head_dim;
        let num_kv_heads = layer.num_kv_heads;
        let num_q_heads = (cx.in_width(0)? - 2 * num_kv_heads * head_dim) / head_dim;
        unsafe {
            qkv_fused::qkv_decode_qk_norm_rope_write_kv_bf16(
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.q_out()?.cast::<bf16>(),
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                cx.weight(0)?.cast::<bf16>().cast_const(),
                cx.weight(1)?.cast::<bf16>().cast_const(),
                cx.positions()?,
                cx.arg_in(1)?.cast::<f32>().cast_const(),
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                w_page,
                w_off,
                plan.row_valid,
                cx.rows().count,
                num_q_heads,
                num_kv_heads,
                head_dim,
                layer.page_size,
                layer.hnd,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

 WRITE_KV_EXPLICIT_DEVWIN => { none: "the device-carried window has no producer: `AttnCtx` states `w_page_d` and `w_off_d` but no window array, so `win_d` is missing its FILL and not merely its query — unlike `first_token`, `num_pages_in_batch`, `w_page_d` and `w_off_d`, which `AttnCtx` had carried since before fn-world existed. The host program is `x::attn::kv_paged::write_kv_explicit_bf16_devwin` and is complete" },

    DEQUANT_KV_ACTIVE => { cx, stream => {
        let layer = cx.kv_layer()?;
        unsafe {
            kv_paged::dequant_kv_cache_layer_to_bf16_active(
                &layer,
                cx.plan()?.kv_page_indices,
                cx.num_pages_in_batch()?,
                stream,
            )
        }
        .ok()
    }},
}

#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;
#[cfg(feature = "_cuda")]
use core::ptr::NonNull;

/// `quant/dtype_cast.cuh` — five cast templates and one row scaler, ten rows.
pub mod dtype_cast {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// Five templates, ten instantiations, and the two ahead-of-time
        unit DTYPE_CAST = "quant/dtype_cast",
            text = include_str!("../../csrc/src/quant/dtype_cast.cuh"),
            file = "quant/dtype_cast.cuh";

        /// `dtype_cast.cuh:104` — `dst[i] = (T)src[i]` over `n` f32.
        fn cast_f32_to = "quant::device::cast_f32_to" <T> (
            src: *const f32,
            dst: *mut T,
            n: usize,
        ) where *mut T {
            "quant::cast_fp32_to_bf16" => where [T = bf16] "device::bf16",
            "quant::cast_fp32_to_f16" => where [T = f16] "device::f16",
        }

        /// `dtype_cast.cuh:112` — the widening direction.
        fn cast_to_f32 = "quant::device::cast_to_f32" <T> (
            src: *const T,
            dst: *mut f32,
            n: usize,
        ) where *const T {
            "quant::cast_bf16_to_fp32" => where [T = bf16] "device::bf16",
            "quant::cast_f16_to_fp32" => where [T = f16] "device::f16",
        }

        /// `dtype_cast.cuh:120` — f16 in, anything out.
        fn cast_f16_to = "quant::device::cast_f16_to" <T> (
            src: *const f16,
            dst: *mut T,
            n: usize,
        ) where *mut T {
            "quant::cast_f16_to_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dtype_cast.cuh:133` — an E8M0 exponent byte widened to f32.
        fn cast_e8m0_to = "quant::device::cast_e8m0_to" <T> (
            src: *const u8,
            dst: *mut T,
            n: usize,
        ) where *mut T {
            "quant::cast_e8m0_to_fp32" => where [T = f32] "quant::device::f32",
        }

        /// `dtype_cast.cuh:149` — `dst[i] = src[i] * factor`.
        fn scale = "quant::device::scale" <T> (
            src: *const T,
            dst: *mut T,
            n: usize,
            factor: f32,
        ) where *const T, *mut T {
            "quant::scale_bf16" => where [T = bf16] "device::bf16",
            "quant::scale_f16" => where [T = f16] "device::f16",
            "quant::scale_fp32" => where [T = f32] "quant::device::f32",
        }

        /// `dtype_cast.cuh:263` — `buf[r, c] *= l[c]`, in place.
        fn scale_rows = "quant::device::scale_rows" <T> (
            buf: *mut T,
            l: *const T,
            width: i32,
        ) where *mut T, *const T {
            "quant::scale_rows_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `quant/dequant_fp8.cuh` — four scale shapes, five rows.
pub mod dequant_fp8 {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// The FP8 E4M3 dequantisers: flat, per-channel, per-tile,
        unit DEQUANT_FP8 = "quant/dequant_fp8",
            text = include_str!("../../csrc/src/quant/dequant_fp8.cuh"),
            file = "quant/dequant_fp8.cuh";

        /// `dequant_fp8.cuh:88` — one f32 scale for the whole tensor.
        fn dequant_fp8_e4m3 = "quant::device::dequant_fp8_e4m3" <T> (
            src: *const u8,
            dst: *mut T,
            scale: f32,
            n: usize,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16" => where [T = bf16] "device::bf16",
            "quant::dequant_fp8_e4m3_to_f16" => where [T = f16] "device::f16",
        }

        /// `dequant_fp8.cuh:97` — one f32 scale per output channel.
        fn dequant_fp8_e4m3_per_channel = "quant::device::dequant_fp8_e4m3_per_channel" <T> (
            src: *const u8,
            dst: *mut T,
            scale_inv: *const f32,
            cols: i32,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16_per_channel" => where [T = bf16] "device::bf16",
        }

        /// `dequant_fp8.cuh:143` — a 2-D tile of scales.
        fn dequant_fp8_e4m3_blocked = "quant::device::dequant_fp8_e4m3_blocked" <T> (
            src: *const u8,
            dst: *mut T,
            scales: *const f32,
            cols: i32,
            row_block: i32,
            col_block: i32,
            scale_cols: i32,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16_blocked" => where [T = bf16] "device::bf16",
        }

        /// `dequant_fp8.cuh:162` — one f32 scale per contiguous group along
        fn dequant_fp8_e4m3_per_group = "quant::device::dequant_fp8_e4m3_per_group" <T> (
            src: *const u8,
            dst: *mut T,
            scales: *const f32,
            cols: i32,
            group_size: i32,
        ) where *mut T {
            "quant::dequant_fp8_e4m3_to_bf16_per_group" => where [T = bf16] "device::bf16",
        }
    }
}

/// `quant/quant_bf16_to_mxfp4.cuh` — the MXFP4 encoder, one row.
pub mod quant_mxfp4 {
    use crate::x::abi::bf16;

    unit! {
        /// One block per row, 32 values per E8M0 scale, two outputs.
        unit QUANT_BF16_TO_MXFP4 = "quant/quant_bf16_to_mxfp4",
            text = include_str!("../../csrc/src/quant/quant_bf16_to_mxfp4.cuh"),
            file = "quant/quant_bf16_to_mxfp4.cuh";

        /// `quant_bf16_to_mxfp4.cuh:115` — a row to E2M1 nibbles plus its
        fn quant_bf16_to_mxfp4_row = "quant::device::quant_bf16_to_mxfp4_row" <T> (
            src: *const T,
            packed: *mut u8,
            scales: *mut u8,
            cols: i32,
        ) where *const T {
            "quant::quantize_bf16_to_mxfp4_e2m1_per_block" => where [T = bf16] "device::bf16",
        }
    }
}

/// `quant/quant_bf16_to_fp8.cuh` — the narrow-format quantisers, eleven rows.
pub mod quant_fp8 {
    use crate::x::abi::bf16;

    unit! {
        /// Nine `__global__`s where the ahead-of-time file had twelve.
        unit QUANT_BF16_TO_FP8 = "quant/quant_bf16_to_fp8",
            text = include_str!("../../csrc/src/quant/quant_bf16_to_fp8.cuh"),
            file = "quant/quant_bf16_to_fp8.cuh";

        /// `quant_bf16_to_fp8.cuh:170` — `out[i] = Fmt(W[i] * scale_inv)`,
        fn quant_flat = "quant::device::quant_flat" <S> (
            w: *const bf16,
            out: *mut S,
            scale_inv: f32,
            n: usize,
        ) where *mut S {
            "quant::quant_bf16_to_fp8_e4m3" => where [S = u8] "quant::device::fp8_e4m3",
        }

        /// `quant_bf16_to_fp8.cuh:185` — `x[i] = x[i] / Fmt::max_abs()`, in
        fn absmax_to_scale_inv = "quant::device::absmax_to_scale_inv" (
            x: *mut f32,
            n: i32,
        ) {
            "quant::absmax_to_scale_inv_fp8" => "quant::device::fp8_e4m3",
            "quant::absmax_to_scale_inv_int8" => "quant::device::int8_sym",
        }

        /// `quant_bf16_to_fp8.cuh:266` — INT8 back to `T`, flat, with the
        fn dequant_int8_per_channel = "quant::device::dequant_int8_per_channel" <T> (
            w: *const i8,
            out: *mut T,
            scale_inv: *const f32,
            cols: i32,
            n: usize,
        ) where *mut T {
            "quant::dequant_int8_to_bf16_per_channel" => where [T = bf16] "device::bf16",
        }

        /// `quant_bf16_to_fp8.cuh:198` — the per-row absmax, on its own.
        fn absmax_per_row = "quant::device::absmax_per_row" <T> (
            w: *const T,
            absmax_out: *mut f32,
            cols: i32,
        ) where *const T {
            "quant::absmax_per_row_bf16" => where [T = bf16] "device::bf16",
        }

        /// `quant_bf16_to_fp8.cuh:234` — narrow a row AND emit its scale.
        fn quant_per_channel = "quant::device::quant_per_channel" <S> (
            w: *const bf16,
            out: *mut S,
            scale_inv: *mut f32,
            cols: i32,
        ) where *mut S {
            "quant::quantize_bf16_to_fp8_e4m3_per_channel" => where [S = u8] "quant::device::fp8_e4m3",
            "quant::quantize_bf16_to_int8_per_channel" => where [S = i8] "quant::device::int8_sym",
        }

        /// `quant_bf16_to_fp8.cuh:215` — stage 2: narrow a row with a scale
        fn cast_per_channel = "quant::device::cast_per_channel" <S> (
            w: *const bf16,
            out: *mut S,
            scale_inv: *const f32,
            cols: i32,
        ) where *mut S {
            "quant::cast_bf16_to_fp8_e4m3_per_channel" => where [S = u8] "quant::device::fp8_e4m3",
            "quant::cast_bf16_to_int8_per_channel" => where [S = i8] "quant::device::int8_sym",
        }

        /// `quant_bf16_to_fp8.cuh:382` — the W8A8 epilogue.
        fn w8a8_dequant = "quant::device::w8a8_dequant" (
            acc: *const i32,
            act_scale_inv: *const f32,
            w_scale_inv: *const f32,
            out: *mut bf16,
            m: i32,
            n: i32,
        ) {
            "quant::dequant_int32_w8a8_to_bf16" => crate::device::DeviceKernel::PLAIN,
        }

        /// `quant_bf16_to_fp8.cuh:330` — blockwise activation quantisation.
        fn quant_act_fp8_per_group = "quant::device::quant_act_fp8_per_group" (
            act: *const bf16,
            out: *mut u8,
            scale_out: *mut f32,
            m: i32,
            k: i32,
            gs: i32,
            n_groups: i32,
        ) {
            "quant::quantize_bf16_to_fp8_e4m3_per_token_group" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `quant/mxfp4_marlin.cuh` — the two repackers a row selector drives.
pub mod mxfp4_marlin {
    use crate::x::abi::{bf16, f16};

    unit! {
        /// Three rows over two templates.
        unit MXFP4_MARLIN = "quant/mxfp4_marlin",
            text = include_str!("../../csrc/src/quant/mxfp4_marlin.cuh"),
            file = "quant/mxfp4_marlin.cuh";

        /// `mxfp4_marlin.cuh:145` — E8M0 block scales into Marlin's order.
        fn mxfp4_scales_to_marlin_e8m0 = "quant::device::mxfp4_scales_to_marlin_e8m0" <T> (
            raw: *const T,
            out: *mut T,
            source_rows: i32,
            source_row_offset: i32,
            selected_rows: i32,
            valid_rows: i32,
            source_stride_groups: i32,
            source_group_offset: i32,
            source_groups: i32,
            target_groups: i32,
            row_select: i32,
        ) where *const T, *mut T {
            "quant::mxfp4_scales_to_marlin_e8m0" => where [T = u8] "device::u8",
        }

        /// `mxfp4_marlin.cuh:197` — a sparse row map gathered dense.
        fn row_map_to_dense = "quant::device::row_map_to_dense" <T> (
            raw: *const T,
            out: *mut T,
            batch: i32,
            source_rows: i32,
            source_row_offset: i32,
            selected_rows: i32,
            valid_rows: i32,
            row_select: i32,
        ) where *const T, *mut T {
            "quant::bf16_row_map_to_dense" => where [T = bf16] "device::bf16",
            "quant::f16_row_map_to_dense" => where [T = f16] "device::f16",
        }
    }
}

/// `quant/dequant_fp4.cuh` — the MXFP4 decoder and the two routed MoE decode
pub mod dequant_fp4 {
    use crate::x::abi::{bf16, f16};
    use core::ffi::c_void;
    use core::ptr::NonNull;

    unit! {
        /// The MXFP4 root: the decoder, then the two routed decode GEMVs.
        unit DEQUANT_FP4 = "quant/dequant_fp4",
            text = include_str!("../../csrc/src/quant/dequant_fp4.cuh"),
            file = "quant/dequant_fp4.cuh";

        /// `dequant_fp4.cuh:98` — packed E2M1 nibbles and E8M0 block scales
        fn dequant_mxfp4 = "quant::device::dequant_mxfp4" <T> (
            packed: *const u8,
            block_scale: *const u8,
            out: *mut T,
            in_dim: i32,
        ) where *mut T {
            "quant::dequant_mxfp4_to_bf16" => where [T = bf16] "device::bf16",
            "quant::dequant_mxfp4_to_f16" => where [T = f16] "device::f16",
        }

        /// `dequant_fp4.cuh:210` — BOTH routed projections of gpt-oss's
        fn mxfp4_moe_gate_up_decode = "quant::device::mxfp4_moe_gate_up_decode" (
            act: *const f16,
            topk_idx: *const i32,
            packed_ptrs: *const *const u8,
            scale_ptrs: *const *const u8,
            gate_bias_ptrs: *const *const c_void,
            up_bias_ptrs: *const *const c_void,
            gate_out: *mut bf16,
            up_out: *mut bf16,
            act_out_fp16: Option<NonNull<f16>>,
            glu_limit: f32,
            glu_alpha: f32,
            top_k: i32,
            hidden: i32,
            intermediate: i32,
        ) {
            "quant::mxfp4_moe_gate_up_decode_bf16" => "device::i32(4)",
        }

        /// `dequant_fp4.cuh:346` — the routed down projection.
        fn mxfp4_moe_down_decode = "quant::device::mxfp4_moe_down_decode" (
            act: *const f16,
            topk_idx: *const i32,
            packed_ptrs: *const *const u8,
            scale_ptrs: *const *const u8,
            bias_ptrs: *const *const c_void,
            out: *mut bf16,
            hidden: i32,
            intermediate: i32,
        ) {
            "quant::mxfp4_moe_down_decode_bf16" => "device::i32(4)",
        }
    }
}

/// `quant/dequant_wna16.cuh` — the W4A16 decoder, the fp16 narrowing cast and
pub mod dequant_wna16 {
    use crate::x::abi::{bf16, f16};
    use core::ffi::c_void;

    unit! {
        /// The W4A16 root: the two a `fn` fired before the crossing, then
        unit DEQUANT_WNA16 = "quant/dequant_wna16",
            text = include_str!("../../csrc/src/quant/dequant_wna16.cuh"),
            file = "quant/dequant_wna16.cuh";

        /// `dequant_wna16.cuh:142` — INT4B8 words with a `T` scale per group
        fn dequant_wna16_int4b8 = "quant::device::dequant_wna16_int4b8" <T> (
            packed: *const i32,
            scale: *const T,
            out: *mut T,
            in_dim: i32,
            group_size: i32,
        ) where *const T, *mut T {
            "quant::dequant_wna16_int4b8_to_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dequant_wna16.cuh:567` — bf16 to a narrow type, vectorised eight
        fn bf16_to_narrow = "quant::device::bf16_to_narrow" <T> (
            in_bf16: *const bf16,
            out: *mut T,
            n: i64,
        ) where *mut T {
            "quant::bf16_to_fp16" => where [T = f16] "device::f16",
        }

        /// `dequant_wna16.cuh:281` — the routed gate and up projections off
        fn wna16_gate_up_decode = "quant::device::wna16_gate_up_decode" (
            act: *const f16,
            topk_idx: *const i32,
            gate_packed_ptrs: *const *const i32,
            gate_scale_ptrs: *const *const c_void,
            up_packed_ptrs: *const *const i32,
            up_scale_ptrs: *const *const c_void,
            gate_out: *mut bf16,
            up_out: *mut bf16,
            top_k: i32,
            hidden: i32,
            intermediate: i32,
            group_size: i32,
        ) {
            "quant::wna16_gate_up_decode_bf16" => "device::i32(0)",
        }

        /// `dequant_wna16.cuh:360` — the routed down projection, **and the
        fn wna16_down_decode = "quant::device::wna16_down_decode" (
            act: *const f16,
            topk_idx: *const i32,
            down_packed_ptrs: *const *const i32,
            down_scale_ptrs: *const *const c_void,
            out: *mut bf16,
            top_k: i32,
            hidden: i32,
            intermediate: i32,
            group_size: i32,
        ) {
            "quant::wna16_down_decode_bf16" => "device::i32(0)",
        }
    }
}

/// The family's seven units, one per `.cuh`.
pub static UNITS: &[Unit] = &[
    dtype_cast::DTYPE_CAST,
    dequant_fp8::DEQUANT_FP8,
    quant_mxfp4::QUANT_BF16_TO_MXFP4,
    quant_fp8::QUANT_BF16_TO_FP8,
    mxfp4_marlin::MXFP4_MARLIN,
    dequant_fp4::DEQUANT_FP4,
    dequant_wna16::DEQUANT_WNA16,
];

/// `quant_bf16_to_fp8.cu:23` and `dtype_cast.cu:20` — `constexpr int BLOCK =
const BLOCK: u32 = 256;

/// A warp, for the block widths that round up to one.
const WARP: u32 = 32;

/// The largest block CUDA will launch, which [`route_rows`] caps at.
const MAX_BLOCK: u32 = 1024;

/// `runtime/launch.rs:659` — [`kernels::LaunchRule::Slab`] divides by the
const SLAB_VEC: u32 = 8;

/// `runtime/launch.rs:668` — and then caps the grid, because a slab kernel is
const SLAB_GRID_MAX: u32 = 1024;

/// `quant_bf16_to_fp8.cu:109` — `constexpr int BX = 32, BY = 8;`, the W8A8
const W8A8_BX: u32 = 32;
/// The other half of the pair above.
const W8A8_BY: u32 = 8;

/// `quant_bf16_to_fp8.cu:131` — the blockwise FP8 quantiser's `128`.
const GROUP_QUANT_BLOCK: u32 = 128;

/// [`kernels::LaunchRule::Elementwise`] — `bind/launch.rs:128`.
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// [`kernels::LaunchRule::ElementwiseRows`] — `bind/launch.rs:143`.
const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch { grid: [rows, width.div_ceil(BLOCK), 1], block: [BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// [`kernels::LaunchRule::Rms`] — `bind/launch.rs:116`.
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

/// [`kernels::LaunchRule::RouteRows`] — `bind/launch.rs:157`.
fn route_rows(rows: u32, width: u32) -> Launch {
    Launch::per_row(rows, width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK))
}

/// [`kernels::LaunchRule::Slab`] — `runtime/launch.rs:985-1015`.
fn slab(n: u32) -> Launch {
    let units = if n >= SLAB_VEC { n / SLAB_VEC } else { n };
    Launch::per_row(units.div_ceil(BLOCK).clamp(1, SLAB_GRID_MAX), BLOCK)
}

/// A `usize` element count as a 32-bit launch extent, or a panic naming it.
#[cfg(feature = "_cuda")]
fn extent(symbol: &str, n: usize) -> u32 {
    let Ok(elems) = u32::try_from(n) else {
        panic!(
            "{symbol}: {n} elements does not fit a 32-bit launch extent; a truncating \
             cast would launch over the low 32 bits and leave the rest of the \
             destination unwritten"
        );
    };
    elems
}

/// `dst[i] = (bf16)src[i]` for `n` fp32 elements — `quant::cast_fp32_to_bf16`.
///
/// # Safety
///
/// `src_fp32` must address `n` live fp32 elements and `dst_bf16` `n` writable
/// bf16 elements, and `stream` must be live across the launch — the same
/// obligations the caller met when this was a `pie_k_*` call handing the
/// stream to a `<<<>>>`.
#[cfg(feature = "_cuda")]
pub unsafe fn cast_fp32_to_bf16(
    src_fp32: *const f32,
    dst_bf16: *mut bf16,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = elementwise(extent("quant::cast_fp32_to_bf16", n));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dtype_cast::raw::cast_f32_to::<bf16>(
            "quant::cast_fp32_to_bf16",
            launch,
            src_fp32,
            dst_bf16,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `buf[r, c] *= l[c]` over a `rows x width` bf16 buffer, in place —
///
/// # Safety
///
/// `buf_bf16` must address `rows * width` writable bf16 elements, `l_bf16`
/// `width` readable ones, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn scale_rows_bf16(
    buf_bf16: *mut bf16,
    l_bf16: *const bf16,
    rows: i32,
    width: i32,
    stream: *mut c_void,
) -> Fired {
    if rows == 0 || width == 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    assert!(rows > 0 && width > 0, "quant::scale_rows_bf16: {rows} x {width} is not an extent");
    let launch = route_rows(rows.unsigned_abs(), width.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dtype_cast::raw::scale_rows::<bf16>(
            "quant::scale_rows_bf16",
            launch,
            buf_bf16,
            l_bf16,
            width,
            stream,
        );
    }
    Fired::Launched
}

/// Narrow a bf16 activation to fp16 — `quant::bf16_to_fp16`.
///
/// # Safety
///
/// `in_bf16` must address `n` live bf16 elements, `out_fp16` `n` writable
/// fp16 elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn bf16_to_fp16(
    in_bf16: *const bf16,
    out_fp16: *mut f16,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = slab(extent("quant::bf16_to_fp16", n));
    let count = i64::try_from(n).unwrap_or(i64::MAX);
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_wna16::raw::bf16_to_narrow::<f16>(
            "quant::bf16_to_fp16",
            launch,
            in_bf16,
            out_fp16,
            count,
            stream,
        );
    }
    Fired::Launched
}

/// One f32 scale for a whole FP8 E4M3 tensor —
///
/// # Safety
///
/// `fp8_in` addresses `n` live E4M3 bytes, `bf16_out` `n` writable bf16
/// elements, and `stream` is live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_fp8_e4m3_to_bf16(
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scale: f32,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = elementwise(extent("quant::dequant_fp8_e4m3_to_bf16", n));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp8::raw::dequant_fp8_e4m3::<bf16>(
            "quant::dequant_fp8_e4m3_to_bf16",
            launch,
            fp8_in,
            bf16_out,
            scale,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// One f32 scale per output channel —
///
/// # Safety
///
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scale_inv` `rows` f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_fp8_e4m3_to_bf16_per_channel(
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp8::raw::dequant_fp8_e4m3_per_channel::<bf16>(
            "quant::dequant_fp8_e4m3_to_bf16_per_channel",
            launch,
            fp8_in,
            bf16_out,
            scale_inv,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// One f32 scale per contiguous group along K, the DeepSeek block-FP8 weight
///
/// # Safety
///
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scales` `rows * ceil(cols / group_size)` f32, and
/// `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_fp8_e4m3_to_bf16_per_group(
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scales: *const f32,
    rows: i32,
    cols: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    if group_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_size" });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp8::raw::dequant_fp8_e4m3_per_group::<bf16>(
            "quant::dequant_fp8_e4m3_to_bf16_per_group",
            launch,
            fp8_in,
            bf16_out,
            scales,
            cols,
            group_size,
            stream,
        );
    }
    Fired::Launched
}

/// Packed E2M1 nibbles and E8M0 block scales to bf16 —
///
/// # Safety
///
/// `packed` addresses `out_dim * in_dim / 2` live bytes, `block_scale`
/// `out_dim * in_dim / 32`, `out` `out_dim * in_dim` writable bf16 elements,
/// and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_mxfp4_to_bf16(
    packed: *const u8,
    block_scale: *const u8,
    out: *mut bf16,
    out_dim: i32,
    in_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if out_dim <= 0 || in_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = route_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp4::raw::dequant_mxfp4::<bf16>(
            "quant::dequant_mxfp4_to_bf16",
            launch,
            packed,
            block_scale,
            out,
            in_dim,
            stream,
        );
    }
    Fired::Launched
}

/// INT4B8 words with a bf16 scale per group along K —
///
/// # Safety
///
/// `packed` addresses `out_dim * in_dim / 8` live `int32`s, `scale`
/// `out_dim * in_dim / group_size` bf16, `out` `out_dim * in_dim` writable
/// bf16, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_wna16_int4b8_to_bf16(
    packed: *const i32,
    scale: *const bf16,
    out: *mut bf16,
    out_dim: i32,
    in_dim: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    if out_dim <= 0 || in_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    if group_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_size" });
    }
    if in_dim % 8 != 0 {
        return Fired::Declined(Refusal::Narrow {
            what: "in_dim's tail past the last whole packed int32 word of 8 int4 values",
            at: in_dim % 8,
        });
    }
    if in_dim % group_size != 0 {
        return Fired::Declined(Refusal::Narrow {
            what: "in_dim's tail past the last whole scale group",
            at: in_dim % group_size,
        });
    }
    let launch = elementwise_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_wna16::raw::dequant_wna16_int4b8::<bf16>(
            "quant::dequant_wna16_int4b8_to_bf16",
            launch,
            packed,
            scale,
            out,
            in_dim,
            group_size,
            stream,
        );
    }
    Fired::Launched
}

/// E8M0 block scales into Marlin's order —
///
/// # Safety
///
/// `raw` addresses `source_rows * source_stride_groups` live E8M0 bytes, `out`
/// `selected_rows * target_groups` writable ones, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn mxfp4_scales_to_marlin_e8m0(
    raw: *const u8,
    out: *mut u8,
    source_rows: i32,
    source_row_offset: i32,
    selected_rows: i32,
    valid_rows: i32,
    source_stride_groups: i32,
    source_group_offset: i32,
    source_groups: i32,
    target_groups: i32,
    row_select: i32,
    stream: *mut c_void,
) -> Fired {
    if selected_rows <= 0 || target_groups <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the repacked rectangle" });
    }
    let total = selected_rows.unsigned_abs().saturating_mul(target_groups.unsigned_abs());
    let launch = elementwise(total);
    // SAFETY: the caller's obligation, above.
    unsafe {
        mxfp4_marlin::raw::mxfp4_scales_to_marlin_e8m0::<u8>(
            "quant::mxfp4_scales_to_marlin_e8m0",
            launch,
            raw,
            out,
            source_rows,
            source_row_offset,
            selected_rows,
            valid_rows,
            source_stride_groups,
            source_group_offset,
            source_groups,
            target_groups,
            row_select,
            stream,
        );
    }
    Fired::Launched
}

/// A bf16 rectangle to MXFP4 nibbles plus their E8M0 block scales —
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `w_packed` `rows * cols / 2`
/// writable bytes, `w_scale_e8m0` `rows * cols / 32` writable bytes, and
/// `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_mxfp4_e2m1_per_block(
    w_bf16: *const bf16,
    w_packed: *mut u8,
    w_scale_e8m0: *mut u8,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    if cols < 32 {
        return Fired::Declined(Refusal::Narrow { what: "cols, in 32-element blocks", at: cols });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs() / 32);
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_mxfp4::raw::quant_bf16_to_mxfp4_row::<bf16>(
            "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
            launch,
            w_bf16,
            w_packed,
            w_scale_e8m0,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// Per-row FP8 E4M3 quantisation with the scale emitted beside it —
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `w_fp8` as many writable
/// bytes, `scale_inv` `rows` writable f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_channel(
    w_bf16: *const bf16,
    w_fp8: *mut u8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = rms(rows.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::quant_per_channel::<u8>(
            "quant::quantize_bf16_to_fp8_e4m3_per_channel",
            launch,
            w_bf16,
            w_fp8,
            scale_inv,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// Per-row symmetric INT8 quantisation of a `[rows, cols]` bf16 rectangle —
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `out_int8` as many writable
/// **signed** bytes, `scale_inv` `rows` writable f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_int8_per_channel(
    w_bf16: *const bf16,
    out_int8: *mut i8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let launch = rms(rows.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::quant_per_channel::<i8>(
            "quant::quantize_bf16_to_int8_per_channel",
            launch,
            w_bf16,
            out_int8,
            scale_inv,
            cols,
            stream,
        );
    }
    Fired::Launched
}

/// INT8 back to bf16 through a per-channel scale —
///
/// # Safety
///
/// `w_int8` addresses `rows * cols` live signed bytes, `out` as many writable
/// bf16, `scale_inv` `rows` f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_int8_to_bf16_per_channel(
    w_int8: *const i8,
    out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 || cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the rectangle" });
    }
    let n = rows.unsigned_abs() as usize * cols.unsigned_abs() as usize;
    let launch = elementwise(extent("quant::dequant_int8_to_bf16_per_channel", n));
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::dequant_int8_per_channel::<bf16>(
            "quant::dequant_int8_to_bf16_per_channel",
            launch,
            w_int8,
            out,
            scale_inv,
            cols,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// The W8A8 epilogue: an `[M, N]` int32 accumulator widened to bf16 through a
///
/// # Safety
///
/// `acc` addresses `m * n` live i32, `act_scale_inv` `m` f32, `w_scale_inv`
/// `n` f32, `out` `m * n` writable bf16, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn dequant_int32_w8a8_to_bf16(
    acc: *const i32,
    act_scale_inv: *const f32,
    w_scale_inv: *const f32,
    out: *mut bf16,
    m: i32,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if m <= 0 || n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the accumulator" });
    }
    let launch = Launch {
        grid: [n.unsigned_abs().div_ceil(W8A8_BX), m.unsigned_abs().div_ceil(W8A8_BY), 1],
        block: [W8A8_BX, W8A8_BY, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::w8a8_dequant(
            "quant::dequant_int32_w8a8_to_bf16",
            launch,
            acc,
            act_scale_inv,
            w_scale_inv,
            out,
            m,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// Blockwise (per-token-group) FP8 E4M3 activation quantisation — the
///
/// # Safety
///
/// `act_bf16` addresses `m * k` live bf16, `act_fp8` as many writable bytes,
/// `act_scale` `m * ceil(k / group_size)` writable f32, and `stream` is live.
#[cfg(feature = "_cuda")]
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_token_group(
    act_bf16: *const bf16,
    act_fp8: *mut u8,
    act_scale: *mut f32,
    m: i32,
    k: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    if m <= 0 || k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the activation" });
    }
    if group_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "group_size" });
    }
    let n_groups = (k + group_size - 1) / group_size;
    let launch = Launch {
        grid: [n_groups.unsigned_abs(), m.unsigned_abs(), 1],
        block: [GROUP_QUANT_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        quant_fp8::raw::quant_act_fp8_per_group(
            "quant::quantize_bf16_to_fp8_e4m3_per_token_group",
            launch,
            act_bf16,
            act_fp8,
            act_scale,
            m,
            k,
            group_size,
            n_groups,
            stream,
        );
    }
    Fired::Launched
}

/// `dequant_fp4.cu:39` — `constexpr int kMxfp4DecodeBlock = 128;`.
const MXFP4_DECODE_BLOCK: u32 = 128;

/// Output rows one WARP of the MXFP4 decode GEMVs owns — the template
const MXFP4_ROWS_PER_WARP: u32 = 4;

/// `dequant_fp4.cu:67-70` and `:152-156` — `dim3(routes, ceil(width / 16))`
const fn routed_qmv_quad(routes: u32, width: u32) -> Launch {
    let tile = (MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP;
    Launch {
        grid: [routes, width.div_ceil(tile), 1],
        block: [MXFP4_DECODE_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `dequant_wna16.cu:73-75`, before §43.9 deleted the launcher as unreached —
const fn routed_qmv(routes: u32, width: u32) -> Launch {
    Launch {
        grid: [routes, width.div_ceil(BLOCK / WARP), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `dequant_wna16.cu:101-104` — [`routed_qmv`]'s two axes SWAPPED.
const fn routed_qmv_transposed(routes: u32, width: u32) -> Launch {
    Launch {
        grid: [width.div_ceil(BLOCK / WARP), routes, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// The routed fanout, checked — `num_tokens * top_k`, which every one of the
#[cfg(feature = "_cuda")]
fn routes_of(num_tokens: i32, top_k: i32) -> Result<u32, Refusal> {
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "the routed fanout" });
    }
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "the token count" });
    }
    Ok(num_tokens.unsigned_abs().saturating_mul(top_k.unsigned_abs()))
}

/// The MXFP4 reduction axis, checked — a multiple of 32, which is one E8M0
#[cfg(feature = "_cuda")]
fn mxfp4_axis(what: &'static str, axis: i32) -> Result<(), Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what });
    }
    if axis % 32 != 0 {
        return Err(Refusal::Narrow { what, at: axis });
    }
    Ok(())
}

/// The W4A16 reduction axis and its group size, checked — **THREE guards, and
#[cfg(feature = "_cuda")]
fn wna16_axis(what: &'static str, axis: i32, group_size: i32) -> Result<(), Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what });
    }
    if group_size <= 0 {
        return Err(Refusal::Empty { what: "the quantisation group size" });
    }
    if group_size % 8 != 0 {
        return Err(Refusal::Narrow { what: "the quantisation group size", at: group_size });
    }
    if axis % 8 != 0 || axis % group_size != 0 {
        return Err(Refusal::Narrow { what, at: axis });
    }
    Ok(())
}

/// gpt-oss's routed gate and up projections, decode-shaped —
///
/// # Safety
///
/// `act` addresses `num_tokens * hidden` live fp16 elements; `topk_idx`
/// `num_tokens * top_k` live `int32`s; the four banks address one device
/// pointer per expert and each pointer its expert's table; `gate_out` and
/// `up_out` each `num_tokens * top_k * intermediate` writable bf16 elements;
/// `act_out_fp16`, when present, the same count in fp16; and `stream` is live
/// across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mxfp4_moe_gate_up_decode_bf16(
    act: *const f16,
    topk_idx: *const i32,
    packed_ptrs: *const *const u8,
    scale_ptrs: *const *const u8,
    gate_bias_ptrs: *const *const c_void,
    up_bias_ptrs: *const *const c_void,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    act_out_fp16: Option<NonNull<f16>>,
    glu_limit: f32,
    glu_alpha: f32,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut c_void,
) -> Fired {
    let routes = match routes_of(num_tokens, top_k) {
        Ok(r) => r,
        Err(e) => return Fired::Declined(e),
    };
    if let Err(e) = mxfp4_axis("hidden", hidden) {
        return Fired::Declined(e);
    }
    if intermediate <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = routed_qmv_quad(routes, intermediate.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp4::raw::mxfp4_moe_gate_up_decode(
            "quant::mxfp4_moe_gate_up_decode_bf16",
            launch,
            act,
            topk_idx,
            packed_ptrs,
            scale_ptrs,
            gate_bias_ptrs,
            up_bias_ptrs,
            gate_out,
            up_out,
            act_out_fp16,
            glu_limit,
            glu_alpha,
            top_k,
            hidden,
            intermediate,
            stream,
        );
    }
    Fired::Launched
}

/// gpt-oss's routed down projection, decode-shaped —
///
/// # Safety
///
/// As [`mxfp4_moe_gate_up_decode_bf16`], with `act` addressing
/// `num_tokens * top_k * intermediate` live fp16 elements — the routed
/// extent, because this leg consumes the activation the gate/up leg produced
/// — and `out` `num_tokens * top_k * hidden` writable bf16 elements.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mxfp4_moe_down_decode_bf16(
    act: *const f16,
    topk_idx: *const i32,
    packed_ptrs: *const *const u8,
    scale_ptrs: *const *const u8,
    bias_ptrs: *const *const c_void,
    out: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    stream: *mut c_void,
) -> Fired {
    let routes = match routes_of(num_tokens, top_k) {
        Ok(r) => r,
        Err(e) => return Fired::Declined(e),
    };
    if let Err(e) = mxfp4_axis("intermediate", intermediate) {
        return Fired::Declined(e);
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    let launch = routed_qmv_quad(routes, hidden.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_fp4::raw::mxfp4_moe_down_decode(
            "quant::mxfp4_moe_down_decode_bf16",
            launch,
            act,
            topk_idx,
            packed_ptrs,
            scale_ptrs,
            bias_ptrs,
            out,
            hidden,
            intermediate,
            stream,
        );
    }
    Fired::Launched
}

/// The routed W4A16 gate and up projections, decode-shaped —
///
/// # Safety
///
/// `act` addresses `num_tokens * hidden` live fp16 elements; `topk_idx`
/// `num_tokens * top_k` live `int32`s; the four banks one device pointer per
/// expert and each pointer its expert's table; `gate_out` and `up_out` each
/// `num_tokens * top_k * intermediate` writable bf16 elements; and `stream`
/// is live across the launch.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn wna16_gate_up_decode_bf16(
    act: *const f16,
    topk_idx: *const i32,
    gate_packed_ptrs: *const *const i32,
    gate_scale_ptrs: *const *const c_void,
    up_packed_ptrs: *const *const i32,
    up_scale_ptrs: *const *const c_void,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    let routes = match routes_of(num_tokens, top_k) {
        Ok(r) => r,
        Err(e) => return Fired::Declined(e),
    };
    if let Err(e) = wna16_axis("hidden", hidden, group_size) {
        return Fired::Declined(e);
    }
    if intermediate <= 0 {
        return Fired::Declined(Refusal::Empty { what: "intermediate" });
    }
    let launch = routed_qmv(routes, intermediate.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_wna16::raw::wna16_gate_up_decode(
            "quant::wna16_gate_up_decode_bf16",
            launch,
            act,
            topk_idx,
            gate_packed_ptrs,
            gate_scale_ptrs,
            up_packed_ptrs,
            up_scale_ptrs,
            gate_out,
            up_out,
            top_k,
            hidden,
            intermediate,
            group_size,
            stream,
        );
    }
    Fired::Launched
}

/// The routed W4A16 down projection, decode-shaped —
///
/// # Safety
///
/// As [`wna16_gate_up_decode_bf16`], with `act` addressing
/// `num_tokens * intermediate` live fp16 elements and `out`
/// `num_tokens * hidden` writable bf16 elements.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn wna16_down_decode_bf16(
    act: *const f16,
    topk_idx: *const i32,
    down_packed_ptrs: *const *const i32,
    down_scale_ptrs: *const *const c_void,
    out: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    group_size: i32,
    stream: *mut c_void,
) -> Fired {
    let routes = match routes_of(num_tokens, top_k) {
        Ok(r) => r,
        Err(e) => return Fired::Declined(e),
    };
    if let Err(e) = wna16_axis("intermediate", intermediate, group_size) {
        return Fired::Declined(e);
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    let launch = routed_qmv_transposed(routes, hidden.unsigned_abs());
    // SAFETY: the caller's obligation, above.
    unsafe {
        dequant_wna16::raw::wna16_down_decode(
            "quant::wna16_down_decode_bf16",
            launch,
            act,
            topk_idx,
            down_packed_ptrs,
            down_scale_ptrs,
            out,
            top_k,
            hidden,
            intermediate,
            group_size,
            stream,
        );
    }
    Fired::Launched
}

contract! {
    /// 4-bit weights with a bf16 scale per group along K.
    DEQUANT_WNA16_INT4B8_TO_BF16 = "quant::dequant_wna16_int4b8_to_bf16" as dequant_wna16_int4b8

    /// The loader's narrowing cast, called by name from Rust since before
    CAST_FP32_TO_BF16 = "quant::cast_fp32_to_bf16" as cast_f32_to_bf16

    /// E8M0 block scales repacked into Marlin's order.
    MXFP4_SCALES_TO_MARLIN_E8M0 = "quant::mxfp4_scales_to_marlin_e8m0" as mxfp4_scales_to_marlin

    /// One f32 scale for the whole tensor.
    DEQUANT_FP8_E4M3_TO_BF16 = "quant::dequant_fp8_e4m3_to_bf16" as dequant_fp8_e4m3

    /// One f32 scale per output channel.
    DEQUANT_FP8_E4M3_TO_BF16_PER_CHANNEL = "quant::dequant_fp8_e4m3_to_bf16_per_channel"
        as dequant_fp8_e4m3_per_channel

    /// One f32 scale per contiguous group along K.
    DEQUANT_FP8_E4M3_TO_BF16_PER_GROUP = "quant::dequant_fp8_e4m3_to_bf16_per_group"
        as dequant_fp8_e4m3_per_group

    /// MXFP4 nibbles and their E8M0 block scales, widened.
    DEQUANT_MXFP4_TO_BF16 = "quant::dequant_mxfp4_to_bf16" as dequant_mxfp4

    /// The activation cast the MXFP4 MoE decode GEMVs read through.
    BF16_TO_FP16 = "quant::bf16_to_fp16" as bf16_to_fp16

    /// Fold a per-column vector into a weight after a merge.
    SCALE_ROWS_BF16 = "quant::scale_rows_bf16" as scale_rows

    /// The loader's Encode path, MXFP4 half — two outputs.
    QUANTIZE_BF16_TO_MXFP4_E2M1_PER_BLOCK = "quant::quantize_bf16_to_mxfp4_e2m1_per_block"
        as quantize_bf16_to_mxfp4

    /// The loader's Encode path, FP8 half — two outputs.
    QUANTIZE_BF16_TO_FP8_E4M3_PER_CHANNEL = "quant::quantize_bf16_to_fp8_e4m3_per_channel"
        as quantize_bf16_to_fp8_per_channel

    /// gpt-oss's routed gate and up projections, one launch off the packed
    MXFP4_MOE_GATE_UP_DECODE_BF16 = "quant::mxfp4_moe_gate_up_decode_bf16"
        as mxfp4_moe_gate_up_decode

    /// The routed down projection, same bank convention.
    MXFP4_MOE_DOWN_DECODE_BF16 = "quant::mxfp4_moe_down_decode_bf16"
        as mxfp4_moe_down_decode

    /// The routed W4A16 gate and up projections, four positional banks.
    WNA16_GATE_UP_DECODE_BF16 = "quant::wna16_gate_up_decode_bf16"
        as wna16_gate_up_decode

    /// The routed W4A16 down projection — the TRANSPOSED grid.
    WNA16_DOWN_DECODE_BF16 = "quant::wna16_down_decode_bf16" as wna16_down_decode
}

#[cfg(feature = "_cuda")]
bind! {
    DEQUANT_WNA16_INT4B8_TO_BF16 => { cx, stream => {
        let group_size = i32::try_from(cx.param(0)?).unwrap_or(0);
        unsafe {
            dequant_wna16_int4b8_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                group_size,
                stream,
            )
        }
        .ok()
    }},

    CAST_FP32_TO_BF16 => { cx, stream => {
        let rows = cx.rows().count;
        let width = cx.out_width(0)?;
        if rows <= 0 || width <= 0 {
            return Err(Refusal::Empty { what: "the output rectangle" });
        }
        let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
        unsafe {
            cast_fp32_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    MXFP4_SCALES_TO_MARLIN_E8M0 => { cx, stream => {
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        unsafe {
            mxfp4_scales_to_marlin_e8m0(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<u8>(),
                param(0)?,
                param(1)?,
                cx.rows().count,
                param(2)?,
                param(3)?,
                param(4)?,
                param(5)?,
                cx.out_width(0)?,
                param(6)?,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_FP8_E4M3_TO_BF16 => { cx, stream => {
        let rows = cx.rows().count;
        let width = cx.out_width(0)?;
        if rows <= 0 || width <= 0 {
            return Err(Refusal::Empty { what: "the output rectangle" });
        }
        let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
        unsafe {
            dequant_fp8_e4m3_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.param_f32(0)?,
                n,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_FP8_E4M3_TO_BF16_PER_CHANNEL => { cx, stream => {
        unsafe {
            dequant_fp8_e4m3_to_bf16_per_channel(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_FP8_E4M3_TO_BF16_PER_GROUP => { cx, stream => {
        let group_size = i32::try_from(cx.param(0)?).unwrap_or(0);
        unsafe {
            dequant_fp8_e4m3_to_bf16_per_group(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                group_size,
                stream,
            )
        }
        .ok()
    }},

    DEQUANT_MXFP4_TO_BF16 => { cx, stream => {
        unsafe {
            dequant_mxfp4_to_bf16(
                cx.arg_in(0)?.cast_const().cast::<u8>(),
                cx.arg_in(1)?.cast_const().cast::<u8>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    BF16_TO_FP16 => { cx, stream => {
        let rows = cx.rows().count;
        let width = cx.out_width(0)?;
        if rows <= 0 || width <= 0 {
            return Err(Refusal::Empty { what: "the output rectangle" });
        }
        let n = rows.unsigned_abs() as usize * width.unsigned_abs() as usize;
        unsafe {
            bf16_to_fp16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    SCALE_ROWS_BF16 => { cx, stream => {
        unsafe {
            scale_rows_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    QUANTIZE_BF16_TO_MXFP4_E2M1_PER_BLOCK => { cx, stream => {
        unsafe {
            quantize_bf16_to_mxfp4_e2m1_per_block(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<u8>(),
                cx.arg_out(1)?.cast::<u8>(),
                cx.rows().count,
                cx.in_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    QUANTIZE_BF16_TO_FP8_E4M3_PER_CHANNEL => { cx, stream => {
        unsafe {
            quantize_bf16_to_fp8_e4m3_per_channel(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<u8>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    MXFP4_MOE_GATE_UP_DECODE_BF16 => { cx, stream => {
        let top_k = cx.in_width(0)?;
        let hidden = cx.in_width(1)?;
        if top_k <= 0 {
            return Err(Refusal::Empty { what: "the routed fanout" });
        }
        let intermediate = cx.out_width(0)? / top_k;
        unsafe {
            mxfp4_moe_gate_up_decode_bf16(
                cx.arg_in(1)?.cast_const().cast::<f16>(),
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.weight(0)?.cast_const().cast::<*const u8>(),
                cx.weight_suffixed("_scales")
                    .ok_or(Refusal::Absent { what: "scale_ptrs" })?
                    .cast_const()
                    .cast::<*const u8>(),
                cx.weight_suffixed("_gate_bias").unwrap_or(core::ptr::null_mut())
                    .cast_const()
                    .cast::<*const c_void>(),
                cx.weight_suffixed("_up_bias").unwrap_or(core::ptr::null_mut())
                    .cast_const()
                    .cast::<*const c_void>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                None,
                cx.glu_limit()?,
                cx.glu_alpha()?,
                cx.rows().count,
                top_k,
                hidden,
                intermediate,
                stream,
            )
        }
        .ok()
    }},

    MXFP4_MOE_DOWN_DECODE_BF16 => { cx, stream => {
        let top_k = cx.in_width(0)?;
        if top_k <= 0 {
            return Err(Refusal::Empty { what: "the routed fanout" });
        }
        let hidden = cx.out_width(0)? / top_k;
        let intermediate = cx.in_width(1)? / top_k;
        unsafe {
            mxfp4_moe_down_decode_bf16(
                cx.arg_in(1)?.cast_const().cast::<f16>(),
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.weight(0)?.cast_const().cast::<*const u8>(),
                cx.weight_suffixed("_scales")
                    .ok_or(Refusal::Absent { what: "scale_ptrs" })?
                    .cast_const()
                    .cast::<*const u8>(),
                cx.weight_bias().unwrap_or(core::ptr::null_mut())
                    .cast_const()
                    .cast::<*const c_void>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                top_k,
                hidden,
                intermediate,
                stream,
            )
        }
        .ok()
    }},

    WNA16_GATE_UP_DECODE_BF16 => { cx, stream => {
        unsafe {
            wna16_gate_up_decode_bf16(
                cx.arg_in(0)?.cast_const().cast::<f16>(),
                cx.arg_in(1)?.cast_const().cast::<i32>(),
                cx.weight(0)?.cast_const().cast::<*const i32>(),
                cx.weight(1)?.cast_const().cast::<*const c_void>(),
                cx.weight(2)?.cast_const().cast::<*const i32>(),
                cx.weight(3)?.cast_const().cast::<*const c_void>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.in_width(1)?,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.wna16_group_size()?,
                stream,
            )
        }
        .ok()
    }},

    WNA16_DOWN_DECODE_BF16 => { cx, stream => {
        unsafe {
            wna16_down_decode_bf16(
                cx.arg_in(0)?.cast_const().cast::<f16>(),
                cx.arg_in(1)?.cast_const().cast::<i32>(),
                cx.weight(0)?.cast_const().cast::<*const i32>(),
                cx.weight(1)?.cast_const().cast::<*const c_void>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.in_width(1)?,
                cx.out_width(0)?,
                cx.in_width(0)?,
                cx.wna16_group_size()?,
                stream,
            )
        }
        .ok()
    }},
}

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::{bf16, f16};
use kernels::Refusal;

use core::ffi::c_void;
use core::ptr::NonNull;

/// `quant/dtype_cast.cuh` — five cast templates and one row scaler, ten rows.
pub mod dtype_cast {
    use crate::jit::Root;

    /// `quant/dtype_cast.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "quant/dtype_cast",
        include_str!("../../csrc/src/quant/dtype_cast.cuh"),
        "quant/dtype_cast.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Absolute, because a routine body names the instantiation itself rather
    /// than a label some other table maps to one. The `<...>` argument is what
    /// used to be a row's `elem`.
    pub mod inst {
        /// `dtype_cast.cuh:104` — f32 narrowed to bf16.
        pub const CAST_F32_TO_BF16: &str = "::pie_cuda_driver::kernels::quant::device::cast_f32_to\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, narrowed to f16.
        pub const CAST_F32_TO_F16: &str = "::pie_cuda_driver::kernels::quant::device::cast_f32_to\
             <::pie_cuda_driver::kernels::device::f16>";
        /// `dtype_cast.cuh:112` — bf16 widened to f32.
        pub const CAST_BF16_TO_F32: &str = "::pie_cuda_driver::kernels::quant::device::cast_to_f32\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, from f16.
        pub const CAST_F16_TO_F32: &str = "::pie_cuda_driver::kernels::quant::device::cast_to_f32\
             <::pie_cuda_driver::kernels::device::f16>";
        /// `dtype_cast.cuh:120` — f16 to bf16.
        pub const CAST_F16_TO_BF16: &str = "::pie_cuda_driver::kernels::quant::device::cast_f16_to\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dtype_cast.cuh:133` — an E8M0 exponent byte widened to f32.
        pub const CAST_E8M0_TO_F32: &str = "::pie_cuda_driver::kernels::quant::device::cast_e8m0_to\
             <::pie_cuda_driver::kernels::quant::device::f32>";
        /// `dtype_cast.cuh:149` — `dst[i] = src[i] * factor`, over bf16.
        pub const SCALE_BF16: &str = "::pie_cuda_driver::kernels::quant::device::scale\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, over f16.
        pub const SCALE_F16: &str = "::pie_cuda_driver::kernels::quant::device::scale\
             <::pie_cuda_driver::kernels::device::f16>";
        /// The same, over f32.
        pub const SCALE_F32: &str = "::pie_cuda_driver::kernels::quant::device::scale\
             <::pie_cuda_driver::kernels::quant::device::f32>";
        /// `dtype_cast.cuh:263` — `buf[r, c] *= l[c]`, in place.
        pub const SCALE_ROWS_BF16: &str = "::pie_cuda_driver::kernels::quant::device::scale_rows\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `quant/dequant_fp8.cuh` — four scale shapes, five rows.
pub mod dequant_fp8 {
    use crate::jit::Root;

    /// `quant/dequant_fp8.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "quant/dequant_fp8",
        include_str!("../../csrc/src/quant/dequant_fp8.cuh"),
        "quant/dequant_fp8.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `dequant_fp8.cuh:88` — one f32 scale for the whole tensor.
        pub const E4M3_TO_BF16: &str = "::pie_cuda_driver::kernels::quant::device::dequant_fp8_e4m3\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, to f16.
        pub const E4M3_TO_F16: &str = "::pie_cuda_driver::kernels::quant::device::dequant_fp8_e4m3\
             <::pie_cuda_driver::kernels::device::f16>";
        /// `dequant_fp8.cuh:97` — one f32 scale per output channel.
        pub const E4M3_TO_BF16_PER_CHANNEL: &str = "::pie_cuda_driver::kernels::quant::device::dequant_fp8_e4m3_per_channel\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dequant_fp8.cuh:143` — a 2-D tile of scales.
        pub const E4M3_TO_BF16_BLOCKED: &str = "::pie_cuda_driver::kernels::quant::device::dequant_fp8_e4m3_blocked\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dequant_fp8.cuh:162` — one f32 scale per contiguous group along K.
        pub const E4M3_TO_BF16_PER_GROUP: &str = "::pie_cuda_driver::kernels::quant::device::dequant_fp8_e4m3_per_group\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `quant/quant_bf16_to_mxfp4.cuh` — the MXFP4 encoder, one row.
pub mod quant_mxfp4 {
    use crate::jit::Root;

    /// `quant/quant_bf16_to_mxfp4.cuh` — the root this routine compiles its
    /// symbol out of.
    pub static ROOT: Root = Root::new(
        "quant/quant_bf16_to_mxfp4",
        include_str!("../../csrc/src/quant/quant_bf16_to_mxfp4.cuh"),
        "quant/quant_bf16_to_mxfp4.cuh",
    );

    /// The template-id NVRTC is handed, spelled as it is handed it.
    pub mod inst {
        /// `quant_bf16_to_mxfp4.cuh:115` — a row to E2M1 nibbles plus its
        /// E8M0 block scales.
        pub const QUANT_BF16_TO_MXFP4_ROW: &str = "::pie_cuda_driver::kernels::quant::device::quant_bf16_to_mxfp4_row\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `quant/quant_bf16_to_fp8.cuh` — the narrow-format quantisers, eleven rows.
pub mod quant_fp8 {
    use crate::jit::Root;

    /// `quant/quant_bf16_to_fp8.cuh` — the root these routines compile a
    /// symbol out of.
    pub static ROOT: Root = Root::new(
        "quant/quant_bf16_to_fp8",
        include_str!("../../csrc/src/quant/quant_bf16_to_fp8.cuh"),
        "quant/quant_bf16_to_fp8.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The `<...>` argument here is the narrow FORMAT, not the wide element
    /// type: one template serves E4M3 and symmetric INT8, and which of the two
    /// a symbol is is the whole of the difference between these pairs.
    pub mod inst {
        /// `quant_bf16_to_fp8.cuh:170` — `out[i] = Fmt(W[i] * scale_inv)`.
        pub const QUANT_FLAT_FP8_E4M3: &str = "::pie_cuda_driver::kernels::quant::device::quant_flat\
             <::pie_cuda_driver::kernels::quant::device::fp8_e4m3>";
        /// `quant_bf16_to_fp8.cuh:185` — `x[i] = x[i] / Fmt::max_abs()`.
        pub const ABSMAX_TO_SCALE_INV_FP8: &str = "::pie_cuda_driver::kernels::quant::device::absmax_to_scale_inv\
             <::pie_cuda_driver::kernels::quant::device::fp8_e4m3>";
        /// The same, over INT8's `max_abs`.
        pub const ABSMAX_TO_SCALE_INV_INT8: &str = "::pie_cuda_driver::kernels::quant::device::absmax_to_scale_inv\
             <::pie_cuda_driver::kernels::quant::device::int8_sym>";
        /// `quant_bf16_to_fp8.cuh:266` — INT8 back to bf16, flat.
        pub const DEQUANT_INT8_PER_CHANNEL_BF16: &str = "::pie_cuda_driver::kernels::quant::device::dequant_int8_per_channel\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `quant_bf16_to_fp8.cuh:198` — the per-row absmax, on its own.
        pub const ABSMAX_PER_ROW_BF16: &str = "::pie_cuda_driver::kernels::quant::device::absmax_per_row\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `quant_bf16_to_fp8.cuh:234` — narrow a row to E4M3 AND emit its
        /// scale.
        pub const QUANT_PER_CHANNEL_FP8_E4M3: &str = "::pie_cuda_driver::kernels::quant::device::quant_per_channel\
             <::pie_cuda_driver::kernels::quant::device::fp8_e4m3>";
        /// The same, to symmetric INT8.
        pub const QUANT_PER_CHANNEL_INT8: &str = "::pie_cuda_driver::kernels::quant::device::quant_per_channel\
             <::pie_cuda_driver::kernels::quant::device::int8_sym>";
        /// `quant_bf16_to_fp8.cuh:215` — narrow a row to E4M3 with a scale
        /// already computed.
        pub const CAST_PER_CHANNEL_FP8_E4M3: &str = "::pie_cuda_driver::kernels::quant::device::cast_per_channel\
             <::pie_cuda_driver::kernels::quant::device::fp8_e4m3>";
        /// The same, to symmetric INT8.
        pub const CAST_PER_CHANNEL_INT8: &str = "::pie_cuda_driver::kernels::quant::device::cast_per_channel\
             <::pie_cuda_driver::kernels::quant::device::int8_sym>";
        /// `quant_bf16_to_fp8.cuh:382` — the W8A8 epilogue.
        pub const W8A8_DEQUANT: &str = "::pie_cuda_driver::kernels::quant::device::w8a8_dequant";
        /// `quant_bf16_to_fp8.cuh:330` — blockwise activation quantisation.
        pub const QUANT_ACT_FP8_PER_GROUP: &str =
            "::pie_cuda_driver::kernels::quant::device::quant_act_fp8_per_group";
    }
}

/// `quant/mxfp4_marlin.cuh` — the two repackers a row selector drives.
pub mod mxfp4_marlin {
    use crate::jit::Root;

    /// `quant/mxfp4_marlin.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "quant/mxfp4_marlin",
        include_str!("../../csrc/src/quant/mxfp4_marlin.cuh"),
        "quant/mxfp4_marlin.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `mxfp4_marlin.cuh:145` — E8M0 block scales into Marlin's order.
        pub const SCALES_TO_MARLIN_E8M0: &str = "::pie_cuda_driver::kernels::quant::device::mxfp4_scales_to_marlin_e8m0\
             <::pie_cuda_driver::kernels::device::u8>";
        /// `mxfp4_marlin.cuh:197` — a sparse row map gathered dense, bf16.
        pub const ROW_MAP_TO_DENSE_BF16: &str = "::pie_cuda_driver::kernels::quant::device::row_map_to_dense\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, over f16.
        pub const ROW_MAP_TO_DENSE_F16: &str = "::pie_cuda_driver::kernels::quant::device::row_map_to_dense\
             <::pie_cuda_driver::kernels::device::f16>";
    }
}

/// `quant/dequant_fp4.cuh` — the MXFP4 decoder and the two routed MoE decode
pub mod dequant_fp4 {
    use crate::jit::Root;

    

    /// `quant/dequant_fp4.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "quant/dequant_fp4",
        include_str!("../../csrc/src/quant/dequant_fp4.cuh"),
        "quant/dequant_fp4.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The two decode GEMVs' argument is `i32(4)`: the output rows one warp
    /// owns, a compile-time tile and not a runtime extent.
    pub mod inst {
        /// `dequant_fp4.cuh:98` — packed E2M1 nibbles and E8M0 block scales,
        /// widened to bf16.
        pub const DEQUANT_MXFP4_BF16: &str = "::pie_cuda_driver::kernels::quant::device::dequant_mxfp4\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, to f16.
        pub const DEQUANT_MXFP4_F16: &str = "::pie_cuda_driver::kernels::quant::device::dequant_mxfp4\
             <::pie_cuda_driver::kernels::device::f16>";
        /// `dequant_fp4.cuh:210` — BOTH routed projections of gpt-oss's gate
        /// and up.
        pub const MOE_GATE_UP_DECODE: &str = "::pie_cuda_driver::kernels::quant::device::mxfp4_moe_gate_up_decode\
             <::pie_cuda_driver::kernels::device::i32(4)>";
        /// `dequant_fp4.cuh:346` — the routed down projection.
        pub const MOE_DOWN_DECODE: &str = "::pie_cuda_driver::kernels::quant::device::mxfp4_moe_down_decode\
             <::pie_cuda_driver::kernels::device::i32(4)>";
    }
}

/// `quant/dequant_wna16.cuh` — the W4A16 decoder, the fp16 narrowing cast and
pub mod dequant_wna16 {
    use crate::jit::Root;

    

    /// `quant/dequant_wna16.cuh` — the root these routines compile a symbol
    /// out of.
    pub static ROOT: Root = Root::new(
        "quant/dequant_wna16",
        include_str!("../../csrc/src/quant/dequant_wna16.cuh"),
        "quant/dequant_wna16.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// The two decode GEMVs' argument is `i32(0)`, which is a compile-time
    /// tile of zero and not an extent: the group size is a runtime operand.
    pub mod inst {
        /// `dequant_wna16.cuh:142` — INT4B8 words with a bf16 scale per group.
        pub const DEQUANT_INT4B8_BF16: &str = "::pie_cuda_driver::kernels::quant::device::dequant_wna16_int4b8\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dequant_wna16.cuh:567` — bf16 narrowed to f16, eight at a time.
        pub const BF16_TO_F16: &str = "::pie_cuda_driver::kernels::quant::device::bf16_to_narrow\
             <::pie_cuda_driver::kernels::device::f16>";
        /// `dequant_wna16.cuh:281` — the routed gate and up projections.
        pub const GATE_UP_DECODE: &str = "::pie_cuda_driver::kernels::quant::device::wna16_gate_up_decode\
             <::pie_cuda_driver::kernels::device::i32(0)>";
        /// `dequant_wna16.cuh:360` — the routed down projection.
        pub const DOWN_DECODE: &str = "::pie_cuda_driver::kernels::quant::device::wna16_down_decode\
             <::pie_cuda_driver::kernels::device::i32(0)>";
    }
}

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
    Launch::grid([rows, width.div_ceil(BLOCK), 1], [BLOCK, 1, 1])
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
/// bf16 elements, and `ctx`'s stream must be live across the launch — the same
/// obligations the caller met when this was a `pie_k_*` call handing the
/// stream to a `<<<>>>`.
pub fn cast_fp32_to_bf16(
    ctx: &Ctx,
    src_fp32: *const f32,
    dst_bf16: *mut bf16,
    n: usize,
) -> Result<(), Refusal> {
    if n == 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    let launch = elementwise(extent("quant::cast_fp32_to_bf16", n));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dtype_cast::ROOT,
            dtype_cast::inst::CAST_F32_TO_BF16,
            launch,
            &[src_fp32.arg(), dst_bf16.arg(), n.arg()],
        )
    }
}

/// `buf[r, c] *= l[c]` over a `rows x width` bf16 buffer, in place —
///
/// # Safety
///
/// `buf_bf16` must address `rows * width` writable bf16 elements, `l_bf16`
/// `width` readable ones, and `ctx`'s stream must be live across the launch.
pub fn scale_rows_bf16(
    ctx: &Ctx,
    buf_bf16: *mut bf16,
    l_bf16: *const bf16,
    rows: i32,
    width: i32,
) -> Result<(), Refusal> {
    if rows == 0 || width == 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    assert!(rows > 0 && width > 0, "quant::scale_rows_bf16: {rows} x {width} is not an extent");
    let launch = route_rows(rows.unsigned_abs(), width.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dtype_cast::ROOT,
            dtype_cast::inst::SCALE_ROWS_BF16,
            launch,
            &[buf_bf16.arg(), l_bf16.arg(), width.arg()],
        )
    }
}

/// Narrow a bf16 activation to fp16 — `quant::bf16_to_fp16`.
///
/// # Safety
///
/// `in_bf16` must address `n` live bf16 elements, `out_fp16` `n` writable
/// fp16 elements, and `ctx`'s stream must be live across the launch.
pub fn bf16_to_fp16(
    ctx: &Ctx,
    in_bf16: *const bf16,
    out_fp16: *mut f16,
    n: usize,
) -> Result<(), Refusal> {
    if n == 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    let launch = slab(extent("quant::bf16_to_fp16", n));
    let count = i64::try_from(n).unwrap_or(i64::MAX);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_wna16::ROOT,
            dequant_wna16::inst::BF16_TO_F16,
            launch,
            &[in_bf16.arg(), out_fp16.arg(), count.arg()],
        )
    }
}

/// One f32 scale for a whole FP8 E4M3 tensor —
///
/// # Safety
///
/// `fp8_in` addresses `n` live E4M3 bytes, `bf16_out` `n` writable bf16
/// elements, and `ctx`'s stream is live across the launch.
pub fn dequant_fp8_e4m3_to_bf16(
    ctx: &Ctx,
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scale: f32,
    n: usize,
) -> Result<(), Refusal> {
    if n == 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    let launch = elementwise(extent("quant::dequant_fp8_e4m3_to_bf16", n));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_fp8::ROOT,
            dequant_fp8::inst::E4M3_TO_BF16,
            launch,
            &[fp8_in.arg(), bf16_out.arg(), scale.arg(), n.arg()],
        )
    }
}

/// One f32 scale per output channel —
///
/// # Safety
///
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scale_inv` `rows` f32, and `ctx`'s stream is live.
pub fn dequant_fp8_e4m3_to_bf16_per_channel(
    ctx: &Ctx,
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if rows <= 0 || cols <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_fp8::ROOT,
            dequant_fp8::inst::E4M3_TO_BF16_PER_CHANNEL,
            launch,
            &[fp8_in.arg(), bf16_out.arg(), scale_inv.arg(), cols.arg()],
        )
    }
}

/// One f32 scale per contiguous group along K, the DeepSeek block-FP8 weight
///
/// # Safety
///
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scales` `rows * ceil(cols / group_size)` f32, and
/// `ctx`'s stream is live.
pub fn dequant_fp8_e4m3_to_bf16_per_group(
    ctx: &Ctx,
    fp8_in: *const u8,
    bf16_out: *mut bf16,
    scales: *const f32,
    rows: i32,
    cols: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    if rows <= 0 || cols <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    if group_size <= 0 {
        return Err(Refusal::Empty { what: "group_size" });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_fp8::ROOT,
            dequant_fp8::inst::E4M3_TO_BF16_PER_GROUP,
            launch,
            &[fp8_in.arg(), bf16_out.arg(), scales.arg(), cols.arg(), group_size.arg()],
        )
    }
}

/// Packed E2M1 nibbles and E8M0 block scales to bf16 —
///
/// # Safety
///
/// `packed` addresses `out_dim * in_dim / 2` live bytes, `block_scale`
/// `out_dim * in_dim / 32`, `out` `out_dim * in_dim` writable bf16 elements,
/// and `ctx`'s stream is live.
pub fn dequant_mxfp4_to_bf16(
    ctx: &Ctx,
    packed: *const u8,
    block_scale: *const u8,
    out: *mut bf16,
    out_dim: i32,
    in_dim: i32,
) -> Result<(), Refusal> {
    if out_dim <= 0 || in_dim <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    let launch = route_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_fp4::ROOT,
            dequant_fp4::inst::DEQUANT_MXFP4_BF16,
            launch,
            &[packed.arg(), block_scale.arg(), out.arg(), in_dim.arg()],
        )
    }
}

/// INT4B8 words with a bf16 scale per group along K —
///
/// # Safety
///
/// `packed` addresses `out_dim * in_dim / 8` live `int32`s, `scale`
/// `out_dim * in_dim / group_size` bf16, `out` `out_dim * in_dim` writable
/// bf16, and `ctx`'s stream is live.
pub fn dequant_wna16_int4b8_to_bf16(
    ctx: &Ctx,
    packed: *const i32,
    scale: *const bf16,
    out: *mut bf16,
    out_dim: i32,
    in_dim: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    if out_dim <= 0 || in_dim <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    if group_size <= 0 {
        return Err(Refusal::Empty { what: "group_size" });
    }
    if in_dim % 8 != 0 {
        return Err(Refusal::Narrow {
            what: "in_dim's tail past the last whole packed int32 word of 8 int4 values",
            at: i64::from(in_dim % 8),
        });
    }
    if in_dim % group_size != 0 {
        return Err(Refusal::Narrow {
            what: "in_dim's tail past the last whole scale group",
            at: i64::from(in_dim % group_size),
        });
    }
    let launch = elementwise_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_wna16::ROOT,
            dequant_wna16::inst::DEQUANT_INT4B8_BF16,
            launch,
            &[packed.arg(), scale.arg(), out.arg(), in_dim.arg(), group_size.arg()],
        )
    }
}

/// E8M0 block scales into Marlin's order —
///
/// # Safety
///
/// `raw` addresses `source_rows * source_stride_groups` live E8M0 bytes, `out`
/// `selected_rows * target_groups` writable ones, and `ctx`'s stream is live.
pub fn mxfp4_scales_to_marlin_e8m0(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    if selected_rows <= 0 || target_groups <= 0 {
        return Err(Refusal::Empty { what: "the repacked rectangle" });
    }
    let total = selected_rows.unsigned_abs().saturating_mul(target_groups.unsigned_abs());
    let launch = elementwise(total);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &mxfp4_marlin::ROOT,
            mxfp4_marlin::inst::SCALES_TO_MARLIN_E8M0,
            launch,
            &[
                raw.arg(),
                out.arg(),
                source_rows.arg(),
                source_row_offset.arg(),
                selected_rows.arg(),
                valid_rows.arg(),
                source_stride_groups.arg(),
                source_group_offset.arg(),
                source_groups.arg(),
                target_groups.arg(),
                row_select.arg(),
            ],
        )
    }
}

/// A bf16 rectangle to MXFP4 nibbles plus their E8M0 block scales —
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `w_packed` `rows * cols / 2`
/// writable bytes, `w_scale_e8m0` `rows * cols / 32` writable bytes, and
/// `ctx`'s stream is live.
pub fn quantize_bf16_to_mxfp4_e2m1_per_block(
    ctx: &Ctx,
    w_bf16: *const bf16,
    w_packed: *mut u8,
    w_scale_e8m0: *mut u8,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if rows <= 0 || cols <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    if cols < 32 {
        return Err(Refusal::Narrow { what: "cols, in 32-element blocks", at: i64::from(cols) });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs() / 32);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &quant_mxfp4::ROOT,
            quant_mxfp4::inst::QUANT_BF16_TO_MXFP4_ROW,
            launch,
            &[w_bf16.arg(), w_packed.arg(), w_scale_e8m0.arg(), cols.arg()],
        )
    }
}

/// Per-row FP8 E4M3 quantisation with the scale emitted beside it —
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `w_fp8` as many writable
/// bytes, `scale_inv` `rows` writable f32, and `ctx`'s stream is live.
pub fn quantize_bf16_to_fp8_e4m3_per_channel(
    ctx: &Ctx,
    w_bf16: *const bf16,
    w_fp8: *mut u8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if rows <= 0 || cols <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    let launch = rms(rows.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &quant_fp8::ROOT,
            quant_fp8::inst::QUANT_PER_CHANNEL_FP8_E4M3,
            launch,
            &[w_bf16.arg(), w_fp8.arg(), scale_inv.arg(), cols.arg()],
        )
    }
}

/// Per-row symmetric INT8 quantisation of a `[rows, cols]` bf16 rectangle —
///
/// # Safety
///
/// `w_bf16` addresses `rows * cols` live bf16, `out_int8` as many writable
/// **signed** bytes, `scale_inv` `rows` writable f32, and `ctx`'s stream is
/// live.
pub fn quantize_bf16_to_int8_per_channel(
    ctx: &Ctx,
    w_bf16: *const bf16,
    out_int8: *mut i8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if rows <= 0 || cols <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    let launch = rms(rows.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &quant_fp8::ROOT,
            quant_fp8::inst::QUANT_PER_CHANNEL_INT8,
            launch,
            &[w_bf16.arg(), out_int8.arg(), scale_inv.arg(), cols.arg()],
        )
    }
}

/// INT8 back to bf16 through a per-channel scale —
///
/// # Safety
///
/// `w_int8` addresses `rows * cols` live signed bytes, `out` as many writable
/// bf16, `scale_inv` `rows` f32, and `ctx`'s stream is live.
pub fn dequant_int8_to_bf16_per_channel(
    ctx: &Ctx,
    w_int8: *const i8,
    out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if rows <= 0 || cols <= 0 {
        return Err(Refusal::Empty { what: "the rectangle" });
    }
    let n = rows.unsigned_abs() as usize * cols.unsigned_abs() as usize;
    let launch = elementwise(extent("quant::dequant_int8_to_bf16_per_channel", n));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &quant_fp8::ROOT,
            quant_fp8::inst::DEQUANT_INT8_PER_CHANNEL_BF16,
            launch,
            &[w_int8.arg(), out.arg(), scale_inv.arg(), cols.arg(), n.arg()],
        )
    }
}

/// The W8A8 epilogue: an `[M, N]` int32 accumulator widened to bf16 through a
///
/// # Safety
///
/// `acc` addresses `m * n` live i32, `act_scale_inv` `m` f32, `w_scale_inv`
/// `n` f32, `out` `m * n` writable bf16, and `ctx`'s stream is live.
pub fn dequant_int32_w8a8_to_bf16(
    ctx: &Ctx,
    acc: *const i32,
    act_scale_inv: *const f32,
    w_scale_inv: *const f32,
    out: *mut bf16,
    m: i32,
    n: i32,
) -> Result<(), Refusal> {
    if m <= 0 || n <= 0 {
        return Err(Refusal::Empty { what: "the accumulator" });
    }
    let launch = Launch::grid(
        [n.unsigned_abs().div_ceil(W8A8_BX), m.unsigned_abs().div_ceil(W8A8_BY), 1],
        [W8A8_BX, W8A8_BY, 1],
    );
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &quant_fp8::ROOT,
            quant_fp8::inst::W8A8_DEQUANT,
            launch,
            &[acc.arg(), act_scale_inv.arg(), w_scale_inv.arg(), out.arg(), m.arg(), n.arg()],
        )
    }
}

/// Blockwise (per-token-group) FP8 E4M3 activation quantisation — the
///
/// # Safety
///
/// `act_bf16` addresses `m * k` live bf16, `act_fp8` as many writable bytes,
/// `act_scale` `m * ceil(k / group_size)` writable f32, and `ctx`'s stream is
/// live.
pub fn quantize_bf16_to_fp8_e4m3_per_token_group(
    ctx: &Ctx,
    act_bf16: *const bf16,
    act_fp8: *mut u8,
    act_scale: *mut f32,
    m: i32,
    k: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    if m <= 0 || k <= 0 {
        return Err(Refusal::Empty { what: "the activation" });
    }
    if group_size <= 0 {
        return Err(Refusal::Empty { what: "group_size" });
    }
    let n_groups = (k + group_size - 1) / group_size;
    let launch =
        Launch::grid([n_groups.unsigned_abs(), m.unsigned_abs(), 1], [GROUP_QUANT_BLOCK, 1, 1]);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &quant_fp8::ROOT,
            quant_fp8::inst::QUANT_ACT_FP8_PER_GROUP,
            launch,
            &[
                act_bf16.arg(),
                act_fp8.arg(),
                act_scale.arg(),
                m.arg(),
                k.arg(),
                group_size.arg(),
                n_groups.arg(),
            ],
        )
    }
}

/// `dequant_fp4.cu:39` — `constexpr int kMxfp4DecodeBlock = 128;`.
const MXFP4_DECODE_BLOCK: u32 = 128;

/// Output rows one WARP of the MXFP4 decode GEMVs owns — the template
const MXFP4_ROWS_PER_WARP: u32 = 4;

/// `dequant_fp4.cu:67-70` and `:152-156` — `dim3(routes, ceil(width / 16))`
const fn routed_qmv_quad(routes: u32, width: u32) -> Launch {
    let tile = (MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP;
    Launch::grid([routes, width.div_ceil(tile), 1], [MXFP4_DECODE_BLOCK, 1, 1])
}

/// `dequant_wna16.cu:73-75`, before §43.9 deleted the launcher as unreached —
const fn routed_qmv(routes: u32, width: u32) -> Launch {
    Launch::grid([routes, width.div_ceil(BLOCK / WARP), 1], [BLOCK, 1, 1])
}

/// `dequant_wna16.cu:101-104` — [`routed_qmv`]'s two axes SWAPPED.
const fn routed_qmv_transposed(routes: u32, width: u32) -> Launch {
    Launch::grid([width.div_ceil(BLOCK / WARP), routes, 1], [BLOCK, 1, 1])
}

/// The routed fanout, checked — `num_tokens * top_k`, which every one of the
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
fn mxfp4_axis(what: &'static str, axis: i32) -> Result<(), Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what });
    }
    if axis % 32 != 0 {
        return Err(Refusal::Narrow { what, at: i64::from(axis) });
    }
    Ok(())
}

/// The W4A16 reduction axis and its group size, checked — **THREE guards, and
fn wna16_axis(what: &'static str, axis: i32, group_size: i32) -> Result<(), Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what });
    }
    if group_size <= 0 {
        return Err(Refusal::Empty { what: "the quantisation group size" });
    }
    if group_size % 8 != 0 {
        return Err(Refusal::Narrow {
            what: "the quantisation group size",
            at: i64::from(group_size),
        });
    }
    if axis % 8 != 0 || axis % group_size != 0 {
        return Err(Refusal::Narrow { what, at: i64::from(axis) });
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
/// `act_out_fp16`, when present, the same count in fp16; and `ctx`'s stream is
/// live across the launch.
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_moe_gate_up_decode_bf16(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    let routes = routes_of(num_tokens, top_k)?;
    mxfp4_axis("hidden", hidden)?;
    if intermediate <= 0 {
        return Err(Refusal::Empty { what: "intermediate" });
    }
    let launch = routed_qmv_quad(routes, intermediate.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_fp4::ROOT,
            dequant_fp4::inst::MOE_GATE_UP_DECODE,
            launch,
            &[
                act.arg(),
                topk_idx.arg(),
                packed_ptrs.arg(),
                scale_ptrs.arg(),
                gate_bias_ptrs.arg(),
                up_bias_ptrs.arg(),
                gate_out.arg(),
                up_out.arg(),
                act_out_fp16.arg(),
                glu_limit.arg(),
                glu_alpha.arg(),
                top_k.arg(),
                hidden.arg(),
                intermediate.arg(),
            ],
        )
    }
}

/// gpt-oss's routed down projection, decode-shaped —
///
/// # Safety
///
/// As [`mxfp4_moe_gate_up_decode_bf16`], with `act` addressing
/// `num_tokens * top_k * intermediate` live fp16 elements — the routed
/// extent, because this leg consumes the activation the gate/up leg produced
/// — and `out` `num_tokens * top_k * hidden` writable bf16 elements.
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_moe_down_decode_bf16(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    let routes = routes_of(num_tokens, top_k)?;
    mxfp4_axis("intermediate", intermediate)?;
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    let launch = routed_qmv_quad(routes, hidden.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_fp4::ROOT,
            dequant_fp4::inst::MOE_DOWN_DECODE,
            launch,
            &[
                act.arg(),
                topk_idx.arg(),
                packed_ptrs.arg(),
                scale_ptrs.arg(),
                bias_ptrs.arg(),
                out.arg(),
                hidden.arg(),
                intermediate.arg(),
            ],
        )
    }
}

/// The routed W4A16 gate and up projections, decode-shaped —
///
/// # Safety
///
/// `act` addresses `num_tokens * hidden` live fp16 elements; `topk_idx`
/// `num_tokens * top_k` live `int32`s; the four banks one device pointer per
/// expert and each pointer its expert's table; `gate_out` and `up_out` each
/// `num_tokens * top_k * intermediate` writable bf16 elements; and `ctx`'s
/// stream is live across the launch.
#[allow(clippy::too_many_arguments)]
pub fn wna16_gate_up_decode_bf16(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    let routes = routes_of(num_tokens, top_k)?;
    wna16_axis("hidden", hidden, group_size)?;
    if intermediate <= 0 {
        return Err(Refusal::Empty { what: "intermediate" });
    }
    let launch = routed_qmv(routes, intermediate.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_wna16::ROOT,
            dequant_wna16::inst::GATE_UP_DECODE,
            launch,
            &[
                act.arg(),
                topk_idx.arg(),
                gate_packed_ptrs.arg(),
                gate_scale_ptrs.arg(),
                up_packed_ptrs.arg(),
                up_scale_ptrs.arg(),
                gate_out.arg(),
                up_out.arg(),
                top_k.arg(),
                hidden.arg(),
                intermediate.arg(),
                group_size.arg(),
            ],
        )
    }
}

/// The routed W4A16 down projection, decode-shaped —
///
/// # Safety
///
/// As [`wna16_gate_up_decode_bf16`], with `act` addressing
/// `num_tokens * intermediate` live fp16 elements and `out`
/// `num_tokens * hidden` writable bf16 elements.
#[allow(clippy::too_many_arguments)]
pub fn wna16_down_decode_bf16(
    ctx: &Ctx,
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
) -> Result<(), Refusal> {
    let routes = routes_of(num_tokens, top_k)?;
    wna16_axis("intermediate", intermediate, group_size)?;
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    let launch = routed_qmv_transposed(routes, hidden.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dequant_wna16::ROOT,
            dequant_wna16::inst::DOWN_DECODE,
            launch,
            &[
                act.arg(),
                topk_idx.arg(),
                down_packed_ptrs.arg(),
                down_scale_ptrs.arg(),
                out.arg(),
                top_k.arg(),
                hidden.arg(),
                intermediate.arg(),
                group_size.arg(),
            ],
        )
    }
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. Nothing is stated beside a name here because no `quant`
/// contract states anything: not one of the fifteen carries `whole`,
/// `in_place` or `depth_prefix_plan`, which is what a family of pure
/// element-for-element conversions looks like from the trace's side.
pub static ROUTINES: &[Routine] = &[
    routine!(cast_fp32_to_bf16),
    routine!(scale_rows_bf16),
    routine!(bf16_to_fp16),
    routine!(dequant_fp8_e4m3_to_bf16),
    routine!(dequant_fp8_e4m3_to_bf16_per_channel),
    routine!(dequant_fp8_e4m3_to_bf16_per_group),
    routine!(dequant_mxfp4_to_bf16),
    routine!(dequant_wna16_int4b8_to_bf16),
    routine!(mxfp4_scales_to_marlin_e8m0),
    routine!(quantize_bf16_to_mxfp4_e2m1_per_block),
    routine!(quantize_bf16_to_fp8_e4m3_per_channel),
    routine!(quantize_bf16_to_int8_per_channel),
    routine!(dequant_int8_to_bf16_per_channel),
    routine!(dequant_int32_w8a8_to_bf16),
    routine!(quantize_bf16_to_fp8_e4m3_per_token_group),
    routine!(mxfp4_moe_gate_up_decode_bf16),
    routine!(mxfp4_moe_down_decode_bf16),
    routine!(wna16_gate_up_decode_bf16),
    routine!(wna16_down_decode_bf16),
];

/// `quant`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "quant", routines: ROUTINES };

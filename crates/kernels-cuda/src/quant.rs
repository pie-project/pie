//! `quant` — the dtype crossings: cast, scale, quantize, dequantize, and the
//! MXFP4/WNA16 decode GEMVs that read a stored bank without materialising it.
//!
//! `In`/`Out` are a statement's operands and results, `Env` is the fire's
//! rather than the trace's, and `Bank` is a weight's stored form. Scale
//! granularity (`_per_channel`, `_per_group`, `_per_token_group`) is part of
//! the symbol name because the kernels differ, not the call. Every
//! `# Safety` section also assumes `ctx`'s stream is live across the launch.

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::{bf16, f16};
use kernels::routine::{Bank, Env, In, Out, Param, ParamF32};
use kernels::keys;
use kernels::Refusal;

use core::ffi::c_void;
use core::ptr::NonNull;

/// `quant/transcode.cuh` — one fused quant→quant kernel per (source, target).
///
/// No `Source` derives a Decode or an Encode: they come from a loader's tile
/// plan, not a trace statement. The launcher is
/// [`driver_internal`](crate::driver_internal)'s; the loader asserts
/// `fusion_mask == 0`, so this crate has no caller for it yet.
pub mod transcode {
    
    use crate::jit::abi::DevicePtr;

    /// `transcode.cuh` — a raw BF16 source, and no source scale.
    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct DecodeBf16 {
        /// `[rows, cols]` bf16, row-major.
        pub src: DevicePtr,
        /// The row stride, in elements.
        pub cols: i32,
    }

    /// `transcode.cuh` — FP8 E4M3 under one f32 scale per `[group_size, group_size]` tile.
    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct DecodeFp8E4m3PerGroup {
        /// `[rows, cols]` E4M3 bytes, row-major.
        pub src: DevicePtr,
        /// `[ceil(rows/gs), scale_cols]` f32.
        pub scales: DevicePtr,
        /// The source's row stride, in elements.
        pub cols: i32,
        /// The scale plane's row stride, in elements.
        pub scale_cols: i32,
        /// The tile edge both scale indices divide by.
        pub group_size: i32,
    }

    /// `transcode.cuh` — 32 floats out as an E8M0 byte and 16 packed E2M1 nibble pairs.
    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct EncodeMxfp4 {
        /// `[rows, cols/2]` bytes.
        pub packed: DevicePtr,
        /// `[rows, cols/32]` E8M0 bytes.
        pub scales: DevicePtr,
        /// The source's row stride, in elements — the encoder derives both of
        /// its own strides from it.
        pub cols: i32,
    }

    // `kGroup`, `kPackedPerByte` and `kBytesPerGroup` are C++ `static
    // constexpr` members, not fields: they take no storage, so the mirror is
    // 24 bytes, not 36.
    crate::by_value! {
        DecodeBf16 as "::pie::transcode::DecodeBf16",
        untagged,
        probe = "nvrtc-probes/quant_transcode.py",
        size = 16, align = 8,
        {
            src  @ 0 as "src",
            cols @ 8 as "cols",
        }
    }

    crate::by_value! {
        DecodeFp8E4m3PerGroup as "::pie::transcode::DecodeFp8E4m3PerGroup",
        untagged,
        probe = "nvrtc-probes/quant_transcode.py",
        size = 32, align = 8,
        {
            src        @ 0  as "src",
            scales     @ 8  as "scales",
            cols       @ 16 as "cols",
            scale_cols @ 20 as "scale_cols",
            group_size @ 24 as "group_size",
        }
    }

    crate::by_value! {
        EncodeMxfp4 as "::pie::transcode::EncodeMxfp4",
        untagged,
        probe = "nvrtc-probes/quant_transcode.py",
        size = 24, align = 8,
        {
            packed @ 0  as "packed",
            scales @ 8  as "scales",
            cols   @ 16 as "cols",
        }
    }

    /// The three measured layouts, as C++ `static_assert`s: `tests/typecheck_tu.rs`
    /// checks them against the header these types mirror, which `by_value!`'s
    /// own assertions never see.
    pub static LAYOUTS: &[crate::jit::Layout] = &[
        <DecodeBf16 as crate::jit::ByValue>::LAYOUT,
        <DecodeFp8E4m3PerGroup as crate::jit::ByValue>::LAYOUT,
        <EncodeMxfp4 as crate::jit::ByValue>::LAYOUT,
    ];
}

/// `quant_bf16_to_fp8.cu` and `dtype_cast.cu` — `constexpr int BLOCK`.
const BLOCK: u32 = 256;

/// A warp, for the block widths that round up to one.
const WARP: u32 = 32;

/// [`kernels::LaunchRule::Elementwise`], as `bind/launch.rs` evaluates it.
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// [`kernels::LaunchRule::Rms`], as `bind/launch.rs` evaluates it.
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

/// [`kernels::LaunchRule::RouteRows`], as `bind/launch.rs` evaluates it.
fn route_rows(rows: u32, width: u32) -> Launch {
    /// The largest block CUDA will launch, which [`route_rows`] caps at.
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(rows, width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK))
}

/// [`kernels::LaunchRule::Slab`], as `runtime/launch.rs` evaluates it.
fn slab(n: u32) -> Launch {
    /// `runtime/launch.rs` — divides the element count by this vector width.
    const SLAB_VEC: u32 = 8;

    /// `runtime/launch.rs` — caps the grid; a slab kernel strides over what's left.
    const SLAB_GRID_MAX: u32 = 1024;

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
/// # Safety
/// `src_fp32` must address `n` live fp32 elements and `dst_bf16` `n` writable bf16 elements.
#[kernels_macros::routine]
pub fn cast_fp32_to<T>(
    ctx: &Ctx,
    src_fp32: In<0, f32>,
    dst_bf16: Out<0, T>,
    // `usize`, not `i32`: the only typed caller is a byte run that can
    // exceed `i32::MAX`, with no rows/width to split it into.
    #[source(OutElements(0))] n: usize,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    let launch = elementwise(extent("quant::cast_fp32_to_bf16", n));
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dtype_cast.cuh",
            &format!("::pie::quant::cast_f32_to<{}>", T::CPP),
            launch,
            &[src_fp32.ptr.arg(), dst_bf16.ptr.arg(), n.arg()],
        )
    }
}

/// `buf[r, c] *= l[c]` over a `rows x width` bf16 buffer, in place.
/// # Safety
/// `buf_bf16` must address `buf_bf16.rows * buf_bf16.width` writable bf16
/// elements, `l_bf16` `buf_bf16.width` readable ones.
#[kernels_macros::routine]
pub fn scale_rows<T>(
    ctx: &Ctx,
    // In place: `dsl::cuda::scale_rows` hands back the same buffer it took as
    // input 0, so this result is also that operand.
    buf_bf16: Out<0, T>,
    l_bf16: In<1, T>,
    // Load-bearing: `route_rows` clamps a zero width up to one warp, so an
    // unguarded empty rectangle would launch successfully and do nothing.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // `Refusal::Empty`, not a panic: `unsigned_abs` would turn a negative
    // rows into a small positive launch if the sign weren't rejected first.
    let rows = buf_bf16.rows;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if buf_bf16.width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    // Guard first, view second: `all()` refuses `Absent`, the guard above
    // refuses `Empty` — only the latter reaches callers. Every other view
    // here is built the same way, for the same reason.
    let buf = buf_bf16.all("width")?;
    let launch = route_rows(rows.unsigned_abs(), buf.width.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dtype_cast.cuh",
            &format!("::pie::quant::scale_rows<{}>", T::CPP),
            launch,
            // A row pitch, not a width — `dtype_cast.cuh` also indexes the
            // scale vector by it, which has no extent of its own to state.
            &[buf.ptr.arg(), l_bf16.ptr.arg(), buf.stride.0.arg()],
        )
    }
}

/// Narrow a bf16 activation to fp16 — `quant::bf16_to_fp16`.
/// # Errors
/// [`Refusal::Empty`] for a non-positive output rectangle, and
/// [`Refusal::Narrow`] when it does not fit one 32-bit launch extent.
/// # Safety
/// `in_bf16` must address `n` live bf16 elements, `out_fp16` `n` writable fp16 elements.
#[kernels_macros::routine]
pub fn bf16_to_fp16(
    ctx: &Ctx,
    in_bf16: In<0, bf16>,
    out_fp16: Out<0, f16>,
) -> Result<(), Refusal> {
    // In 64 bits, since an `i32` multiply would wrap. `Region::elements`
    // computes the same product saturated in `i32`, not refused.
    let count = i64::from(out_fp16.rows) * i64::from(out_fp16.width);
    if count <= 0 {
        return Err(Refusal::Empty { what: "the output rectangle" });
    }
    let Ok(grid) = u32::try_from(count) else {
        return Err(Refusal::Narrow { what: "the output rectangle in one 32-bit launch extent", at: count });
    };
    let launch = slab(grid);
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_wna16.cuh",
            "::pie::quant::bf16_to_narrow<::pie::f16>",
            launch,
            &[in_bf16.ptr.arg(), out_fp16.ptr.arg(), count.arg()],
        )
    }
}

/// One f32 scale for a whole FP8 E4M3 tensor —
/// # Safety
/// `fp8_in` addresses `n` live E4M3 bytes, `bf16_out` `n` writable bf16 elements.
#[kernels_macros::routine]
pub fn dequant_fp8_e4m3_to<T>(
    ctx: &Ctx,
    fp8_in: In<0, u8>,
    bf16_out: Out<0, T>,
    // A weight walker: this launcher and those through
    // `mxfp4_scales_to_marlin_e8m0` dequantise a weight, whose shape is a
    // checkpoint property, not a fire fact. Its leading extent (`rows`/
    // `out_dim`) has no `Source`: `keys::Rows` would compile but read the
    // fire's token count instead. Its trailing extent (`cols`/`in_dim`) is
    // a row pitch a region already carries; `scale` here is `params[0]`.
    scale: ParamF32<0>,
    // The weight's own shape, which the statement carries at `params[1..3]`.
    // Not `keys::Rows` — that is the fire's token count.
    rows: Param<1, i32>,
    cols: Param<2, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    let n = (*rows as usize).saturating_mul(*cols as usize);
    let launch = elementwise(extent("quant::dequant_fp8_e4m3_to_bf16", n));
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp8.cuh",
            &format!("::pie::quant::dequant_fp8_e4m3<{}>", T::CPP),
            launch,
            &[fp8_in.ptr.arg(), bf16_out.ptr.arg(), scale.arg(), n.arg()],
        )
    }
}

/// One f32 scale per output channel —
/// # Safety
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scale_inv` `rows` f32.
#[kernels_macros::routine]
pub fn dequant_fp8_e4m3_to_bf16_per_channel(
    ctx: &Ctx,
    fp8_in: In<0, u8>,
    bf16_out: Out<0, bf16>,
    scale_inv: In<1, f32>,
    // The weight's own shape, at `params[1..3]`; see `dequant_fp8_e4m3_to`.
    rows: Param<1, i32>,
    cols: Param<2, i32>,
) -> Result<(), Refusal> {
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp8.cuh",
            "::pie::quant::dequant_fp8_e4m3_per_channel<::pie::bf16>",
            launch,
            &[fp8_in.ptr.arg(), bf16_out.ptr.arg(), scale_inv.ptr.arg(), cols.arg()],
        )
    }
}

/// One f32 scale per contiguous group along K, the DeepSeek block-FP8 weight
/// # Safety
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scales` `rows * ceil(cols / group_size)` f32.
#[kernels_macros::routine]
pub fn dequant_fp8_e4m3_to_bf16_per_group(
    ctx: &Ctx,
    fp8_in: In<0, u8>,
    bf16_out: Out<0, bf16>,
    scales: In<1, f32>,
    // `rows` reads `Env<keys::Rows>` — the fire's token window, not this
    // weight's checkpoint-fixed count (wrong, but pre-existing:
    // `rectangle_rows` discards the `Dim::Const` leading extent). `cols`'s
    // `OutWidth(0)` is correct.
    rows: Env<keys::Rows>,
    #[source(OutWidth(0))]
    cols: i32,
    group_size: Param<0, i32>,
) -> Result<(), Refusal> {
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp8.cuh",
            "::pie::quant::dequant_fp8_e4m3_per_group<::pie::bf16>",
            launch,
            &[fp8_in.ptr.arg(), bf16_out.ptr.arg(), scales.ptr.arg(), cols.arg(), group_size.arg()],
        )
    }
}

/// Packed E2M1 nibbles and E8M0 block scales to bf16 —
/// # Safety
/// `packed` addresses `out_dim * in_dim / 2` live bytes, `block_scale`
/// `out_dim * in_dim / 32`, `out` `out_dim * in_dim` writable bf16 elements.
#[kernels_macros::routine]
pub fn dequant_mxfp4_to<T>(
    ctx: &Ctx,
    packed: In<0, u8>,
    block_scale: In<1, u8>,
    out: Out<0, T>,
    // The weight's own shape, at `params[0..2]`; this form spends no `params`
    // on a scale, so it starts at zero.
    out_dim: Param<0, i32>,
    in_dim: Param<1, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *mut T: Abi,
{
    let launch = route_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp4.cuh",
            &format!("::pie::quant::dequant_mxfp4<{}>", T::CPP),
            launch,
            &[packed.ptr.arg(), block_scale.ptr.arg(), out.ptr.arg(), in_dim.arg()],
        )
    }
}

/// INT4B8 words with a bf16 scale per group along K —
/// # Safety
/// `packed` addresses `out_dim * in_dim / 8` live `int32`s, `scale`
/// `out_dim * in_dim / group_size` bf16, `out` `out_dim * in_dim` writable bf16.
#[kernels_macros::routine]
pub fn dequant_wna16_int4b8_to<T>(
    ctx: &Ctx,
    packed: In<0, i32>,
    scale: In<1, T>,
    out: Out<0, T>,
    // A weight walker — see [`dequant_fp8_e4m3_to`]'s block. `out.rows`/
    // The weight's own shape, at `params[1..3]`; `[0]` is the group size.
    out_dim: Param<1, i32>,
    in_dim: Param<2, i32>,
    // `group_size` is `Param<0, i32>`, the same slot and quantisation
    // convention as `dequant_fp8_e4m3_to_bf16_per_group`'s.
    group_size: Param<0, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    /// [`kernels::LaunchRule::ElementwiseRows`], as `bind/launch.rs` evaluates it.
    const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch::grid([rows, width.div_ceil(BLOCK), 1], [BLOCK, 1, 1])
    }

    let (out_dim, in_dim) = (*out_dim, *in_dim);
    if in_dim % 8 != 0 {
        return Err(Refusal::Narrow {
            what: "in_dim's tail past the last whole packed int32 word of 8 int4 values",
            at: i64::from(in_dim % 8),
        });
    }
    if in_dim % *group_size != 0 {
        return Err(Refusal::Narrow {
            what: "in_dim's tail past the last whole scale group",
            at: i64::from(in_dim % *group_size),
        });
    }
    let launch = elementwise_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_wna16.cuh",
            &format!("::pie::quant::dequant_wna16_int4b8<{}>", T::CPP),
            launch,
            &[packed.ptr.arg(), scale.ptr.arg(), out.ptr.arg(), in_dim.arg(), group_size.arg()],
        )
    }
}

/// E8M0 block scales into Marlin's order —
/// # Safety
/// `raw` addresses `source_rows * source_stride_groups` live E8M0 bytes, `out`
/// `selected_rows * target_groups` writable ones.
#[kernels_macros::routine]
pub fn mxfp4_scales_to_marlin_e8m0(
    ctx: &Ctx,
    raw: In<0, u8>,
    out: Out<0, u8>,
    // Nine scalars: seven ride `params[]` (`Param<N, i32>`); `selected_rows`
    // and `target_groups` are a weight-shaped result's own extents (no
    // `Source` names them — see [`dequant_fp8_e4m3_to`]'s block), and the
    // two offsets are into the checkpoint's scale table, a fact no fire
    // holds. No windowed view is built (a window needs a byte offset, which
    // needs the pointee's size this signature lacks), so the kernel bounds-
    // checks and zero-fills silently past them (`mxfp4_marlin.cuh`) instead.
    source_rows: Param<0, i32>,
    source_row_offset: Param<1, i32>,
    #[source(Rows)]
    selected_rows: i32,
    valid_rows: Param<2, i32>,
    source_stride_groups: Param<3, i32>,
    source_group_offset: Param<4, i32>,
    source_groups: Param<5, i32>,
    #[source(OutWidth(0))]
    target_groups: i32,
    row_select: Param<6, i32>,
) -> Result<(), Refusal> {
    let total = selected_rows.unsigned_abs().saturating_mul(target_groups.unsigned_abs());
    let launch = elementwise(total);
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/mxfp4_marlin.cuh",
            "::pie::quant::mxfp4_scales_to_marlin_e8m0<::pie::u8>",
            launch,
            &[
                raw.ptr.arg(),
                out.ptr.arg(),
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
/// # Safety
/// `w_bf16` addresses `rows * cols` live bf16, `w_packed` `rows * cols / 2`
/// writable bytes, `w_scale_e8m0` `rows * cols / 32` writable bytes.
#[kernels_macros::routine]
pub fn quantize_bf16_to_mxfp4_e2m1_per_block(
    ctx: &Ctx,
    w_bf16: In<0, bf16>,
    w_packed: Out<0, u8>,
    w_scale_e8m0: Out<1, u8>,
    // Unbound, unlike the weight walkers above: no statement at all — this
    // loader quantiser runs once at load, from the transform plan.
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if cols < 32 {
        return Err(Refusal::Narrow { what: "cols, in 32-element blocks", at: i64::from(cols) });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs() / 32);
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_mxfp4.cuh",
            "::pie::quant::quant_bf16_to_mxfp4_row<::pie::bf16>",
            launch,
            &[w_bf16.ptr.arg(), w_packed.ptr.arg(), w_scale_e8m0.ptr.arg(), cols.arg()],
        )
    }
}

/// Per-row FP8 E4M3 quantisation with the scale emitted beside it —
/// # Safety
/// `w_bf16` addresses `rows * cols` live bf16, `w_fp8` as many writable
/// bytes, `scale_inv` `rows` writable f32.
#[kernels_macros::routine]
pub fn quantize_bf16_to_fp8_e4m3_per_channel(
    ctx: &Ctx,
    w_bf16: In<0, bf16>,
    w_fp8: Out<0, u8>,
    scale_inv: Out<1, f32>,
    // The loader's other quantiser -- see [`quantize_bf16_to_mxfp4_e2m1_per_block`].
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    let launch = rms(rows.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_per_channel<::pie::quant::fp8_e4m3>",
            launch,
            &[w_bf16.ptr.arg(), w_fp8.ptr.arg(), scale_inv.ptr.arg(), cols.arg()],
        )
    }
}

/// Per-row symmetric INT8 quantisation of a `[rows, cols]` bf16 rectangle —
/// # Safety
/// `w_bf16` addresses `rows * cols` live bf16, `out_int8` as many writable
/// signed bytes, `scale_inv` `rows` writable f32.
#[kernels_macros::routine]
pub fn quantize_bf16_to_int8_per_channel(
    ctx: &Ctx,
    w_bf16: In<0, bf16>,
    out_int8: Out<0, i8>,
    scale_inv: Out<1, f32>,
    // The quantised GEMM's staging four (with
    // [`dequant_int8_to_bf16_per_channel`], [`dequant_int32_w8a8_to_bf16`]
    // and [`quantize_bf16_to_fp8_e4m3_per_token_group`]): all unsourced,
    // fired mid-matmul by `gemm/quant.rs`, so no trace states the rectangle.
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    let launch = rms(rows.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_per_channel<::pie::quant::int8_sym>",
            launch,
            &[w_bf16.ptr.arg(), out_int8.ptr.arg(), scale_inv.ptr.arg(), cols.arg()],
        )
    }
}

/// INT8 back to bf16 through a per-channel scale —
/// # Safety
/// `w_int8` addresses `rows * cols` live signed bytes, `out` as many writable
/// bf16, `scale_inv` `rows` f32.
#[kernels_macros::routine]
pub fn dequant_int8_to_bf16_per_channel(
    ctx: &Ctx,
    w_int8: In<0, i8>,
    out: Out<0, bf16>,
    scale_inv: In<1, f32>,
    // Staging four -- see [`quantize_bf16_to_int8_per_channel`].
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    let n = rows.unsigned_abs() as usize * cols.unsigned_abs() as usize;
    let launch = elementwise(extent("quant::dequant_int8_to_bf16_per_channel", n));
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::dequant_int8_per_channel<::pie::bf16>",
            launch,
            &[w_int8.ptr.arg(), out.ptr.arg(), scale_inv.ptr.arg(), cols.arg(), n.arg()],
        )
    }
}

/// The W8A8 epilogue: an `[M, N]` int32 accumulator widened to bf16 through a
/// # Safety
/// `acc` addresses `m * n` live i32, `act_scale_inv` `m` f32, `w_scale_inv`
/// `n` f32, `out` `m * n` writable bf16.
#[kernels_macros::routine]
pub fn dequant_int32_w8a8_to_bf16(
    ctx: &Ctx,
    acc: In<0, i32>,
    act_scale_inv: In<1, f32>,
    w_scale_inv: In<2, f32>,
    out: Out<0, bf16>,
    // Staging four — see [`quantize_bf16_to_int8_per_channel`]. `n` here is
    // a width, not the flat element count other launchers spell the same way.
    m: i32,
    n: i32,
) -> Result<(), Refusal> {
    const W8A8_BY: u32 = 8;

    /// `quant_bf16_to_fp8.cu` — `constexpr int BX = 32, BY = 8;`, the W8A8 tile.
    const W8A8_BX: u32 = 32;

    let launch = Launch::grid(
        [n.unsigned_abs().div_ceil(W8A8_BX), m.unsigned_abs().div_ceil(W8A8_BY), 1],
        [W8A8_BX, W8A8_BY, 1],
    );
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::w8a8_dequant",
            launch,
            &[acc.ptr.arg(), act_scale_inv.ptr.arg(), w_scale_inv.ptr.arg(), out.ptr.arg(), m.arg(), n.arg()],
        )
    }
}

/// Blockwise (per-token-group) FP8 E4M3 activation quantisation — the
/// # Safety
/// `act_bf16` addresses `m * k` live bf16, `act_fp8` as many writable bytes,
/// `act_scale` `m * ceil(k / group_size)` writable f32.
#[kernels_macros::routine]
pub fn quantize_bf16_to_fp8_e4m3_per_token_group(
    ctx: &Ctx,
    act_bf16: In<0, bf16>,
    act_fp8: Out<0, u8>,
    act_scale: Out<1, f32>,
    // Staging four — see [`quantize_bf16_to_int8_per_channel`]. `m` is a
    // token count, still unbindable: the caller is a matmul body, not a fire.
    m: i32,
    k: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    /// `quant_bf16_to_fp8.cu` — the blockwise FP8 quantiser's `128`.
           const GROUP_QUANT_BLOCK: u32 = 128;

    let n_groups = (k + group_size - 1) / group_size;
    let launch =
        Launch::grid([n_groups.unsigned_abs(), m.unsigned_abs(), 1], [GROUP_QUANT_BLOCK, 1, 1]);
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_act_fp8_per_group",
            launch,
            &[
                act_bf16.ptr.arg(),
                act_fp8.ptr.arg(),
                act_scale.ptr.arg(),
                m.arg(),
                k.arg(),
                group_size.arg(),
                n_groups.arg(),
            ],
        )
    }
}

/// `dequant_fp4.cu` — `dim3(routes, ceil(width / 16))`.
const fn routed_qmv_quad(routes: u32, width: u32) -> Launch {
    /// `dequant_fp4.cu` — `constexpr int kMxfp4DecodeBlock = 128;`.
    const MXFP4_DECODE_BLOCK: u32 = 128;

    /// Output rows one warp of the MXFP4 decode GEMVs owns.
    const MXFP4_ROWS_PER_WARP: u32 = 4;

    let tile = (MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP;
    Launch::grid([routes, width.div_ceil(tile), 1], [MXFP4_DECODE_BLOCK, 1, 1])
}

/// `dequant_wna16.cu` — the routed W4A16 GEMV launch shape.
const fn routed_qmv(routes: u32, width: u32) -> Launch {
    Launch::grid([routes, width.div_ceil(BLOCK / WARP), 1], [BLOCK, 1, 1])
}
/// The routed fanout, checked — `num_tokens * top_k`.
fn routes_of(num_tokens: i32, top_k: i32) -> Result<u32, Refusal> {
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "the routed fanout" });
    }
    Ok(num_tokens.unsigned_abs().saturating_mul(top_k.unsigned_abs()))
}

/// The width one route holds, of a `width` that carries `top_k` of them.
fn per_route(width: i32, top_k: i32) -> Result<i32, Refusal> {
    if top_k <= 0 {
        return Err(Refusal::Empty { what: "the routed fanout" });
    }
    if width <= 0 {
        return Err(Refusal::Empty { what: "the routed row" });
    }
    if width % top_k != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of routes",
            at: i64::from(width),
        });
    }
    Ok(width / top_k)
}

/// The MXFP4 reduction axis, checked — a multiple of 32, one E8M0 block.
fn mxfp4_axis(what: &'static str, axis: i32) -> Result<(), Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what });
    }
    if axis % 32 != 0 {
        return Err(Refusal::Narrow { what, at: i64::from(axis) });
    }
    Ok(())
}

/// The W4A16 reduction axis and its group size, checked — the axis must be
/// positive, a multiple of 8, and a whole number of groups.
fn wna16_axis(what: &'static str, axis: i32, group_size: i32) -> Result<(), Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what });
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
/// # Safety
/// `act`/`topk_idx` address their `.rows * .width` extents (fp16, `int32`);
/// the four banks one device pointer per expert; `gate_out`/`up_out` each
/// `gate_out.rows * gate_out.width` writable bf16 elements.
#[allow(clippy::too_many_arguments)]
#[kernels_macros::routine]
pub fn mxfp4_moe_gate_up_decode_bf16(
    ctx: &Ctx,
    // Inverted: the statement places `vec![experts.id, x.id]`, so `act` is
    // `In(1)` and `topk_idx` `In(0)` — `In(0)` here would compile and bind
    // the index run where the activation belongs.
    act: In<1, f16>,
    topk_idx: In<0, i32>,
    // `Bank<0, _>`, not `Weight<0, _>`: both compile, but `Weight` derives
    // `WeightNamed` (`spec.weight`), a different table than this positional
    // bank at `args[n_in + n_out]`.
    packed_ptrs: Bank<0, *const u8>,
    // The bank's other three planes, not separate operands: the driver
    // resolves `_scales`/`_gate_bias`/`_up_bias` off the one named bank by
    // suffix. `Env<keys::WeightScales>` names this one.
    scale_ptrs: Env<keys::WeightScales>,
    // Always `Env`, never bound: whether either bias plane exists is a
    // property of which gpt-oss export loaded — no CUDA statement observes
    // the tensor inventory — so an absent plane binds null instead of refusing.
    gate_bias_ptrs: Env<*const *const c_void>,
    up_bias_ptrs: Env<*const *const c_void>,
    // `gate_out.width` is the fanned-out result row that `per_route` divides.
    gate_out: Out<0, bf16>,
    up_out: Out<1, bf16>,
    // gpt-oss's clamp and SwiGLU alpha, both `Source::Named` facts
    // (`keys::GluLimit`, `keys::GluAlpha`) computed the same way by
    // `mlp::gpt_oss_glu`. Adjacent same-typed f32s — a swap compiles.
    glu_limit: Env<keys::GluLimit>,
    glu_alpha: Env<keys::GluAlpha>,
) -> Result<(), Refusal> {
    // The fanout is `topk_idx`'s own rectangle: the index run is
    // `[Tokens, Const(top_k)]`, one `i32` per (token, route).
    let top_k = topk_idx.width;
    let intermediate = per_route(gate_out.width, top_k)?;
    let routes = routes_of(topk_idx.rows, top_k)?;
    mxfp4_axis("hidden", act.width)?;
    // The activation's view, per [`scale_rows`]'s reason. `top_k` and
    // `gate_out.width` are already refused before a view could fire `Absent`.
    let x = act.all("hidden")?;
    let launch = routed_qmv_quad(routes, intermediate.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp4.cuh",
            "::pie::quant::mxfp4_moe_gate_up_decode<::pie::i32(4)>",
            launch,
            &[
                x.ptr.arg(),
                topk_idx.ptr.arg(),
                packed_ptrs.ptr.arg(),
                scale_ptrs.arg(),
                gate_bias_ptrs.arg(),
                up_bias_ptrs.arg(),
                gate_out.ptr.arg(),
                up_out.ptr.arg(),
                // The optional fp16 side-output: the kernel writes it when
                // non-null, and no caller wants it.
                Option::<NonNull<f16>>::None.arg(),
                glu_limit.arg(),
                glu_alpha.arg(),
                top_k.arg(),
                // A pitch: `dequant_fp4.cuh` also reads it as the expert
                // row's word count (`words_per_row = hidden / 8`) — a
                // `Bank` has no width of its own to carry that separately.
                x.stride.0.arg(),
                intermediate.arg(),
            ],
        )
    }
}

/// gpt-oss's routed down projection, decode-shaped —
/// # Safety
/// As [`mxfp4_moe_gate_up_decode_bf16`], with `act` addressing the routed
/// extent (`act.rows * act.width`) and `out` `out.rows * out.width` writable bf16.
#[allow(clippy::too_many_arguments)]
#[kernels_macros::routine]
pub fn mxfp4_moe_down_decode_bf16(
    ctx: &Ctx,
    // [`mxfp4_moe_gate_up_decode_bf16`]'s three pointer bindings and
    // reasons: `act`/`topk_idx` inverted, `packed_ptrs` a positional `Bank`.
    act: In<1, f16>,
    topk_idx: In<0, i32>,
    packed_ptrs: Bank<0, *const u8>,
    // The `_scales` plane of the one stated bank — `keys::WeightScales`, as its twin names it.
    scale_ptrs: Env<keys::WeightScales>,
    // The bank's `_bias` plane — `keys::WeightBias`, not `keys::NamedWeight2`
    // (`spec.weight2`, a different fact this driver has). `Unstated`, not
    // null: every gpt-oss export publishes `{down}_bias`.
    bias_ptrs: Env<keys::WeightBias>,
    out: Out<0, bf16>,
) -> Result<(), Refusal> {
    // `top_k` off `topk_idx`, not `act` — this family states
    // `vec![experts.id, x.id]`, so the index run is operand zero.
    let top_k = topk_idx.width;
    let hidden = per_route(out.width, top_k)?;
    let intermediate = per_route(act.width, top_k)?;
    let routes = routes_of(topk_idx.rows, top_k)?;
    mxfp4_axis("intermediate", intermediate)?;
    // No view on this leg: the kernel strides by `per_route` quotients, not
    // a region's token pitch — using that would multiply every address by
    // `top_k`. The fanned-out row is a reshape `Region` can't express yet.
    let launch = routed_qmv_quad(routes, hidden.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp4.cuh",
            "::pie::quant::mxfp4_moe_down_decode<::pie::i32(4)>",
            launch,
            &[
                act.ptr.arg(),
                topk_idx.ptr.arg(),
                packed_ptrs.ptr.arg(),
                scale_ptrs.arg(),
                bias_ptrs.arg(),
                out.ptr.arg(),
                hidden.arg(),
                intermediate.arg(),
            ],
        )
    }
}

/// The routed W4A16 gate and up projections, decode-shaped —
/// # Safety
/// `act`/`topk_idx` address their `.rows * .width` extents (fp16, `int32`);
/// the four banks one device pointer per expert; `gate_out`/`up_out` each
/// `gate_out.rows * topk_idx.width * gate_out.width` writable bf16 elements.
#[allow(clippy::too_many_arguments)]
#[kernels_macros::routine]
pub fn wna16_gate_up_decode_bf16(
    ctx: &Ctx,
    // Reversed from the MXFP4 pair: here operand 0 is `act` (`hidden`) and
    // operand 1 is `topk_idx` (`top_k`) — not a transcription slip, the
    // statement really places them the other way round.
    act: In<0, f16>,
    topk_idx: In<1, i32>,
    // Four positional banks (`args[n_in + n_out]`), read via `Bank<N, _>`;
    // `Weight<N, _>` would compile too but derive `WeightNamed`, a
    // different table, silently.
    gate_packed_ptrs: Bank<0, *const i32>,
    gate_scale_ptrs: Bank<1, *const c_void>,
    up_packed_ptrs: Bank<2, *const i32>,
    up_scale_ptrs: Bank<3, *const c_void>,
    // `gate_out.width` is already per-route here, unlike the mxfp4 pair's
    // (`[Tokens, top_k, intermediate]`, undone by `per_route`): this states
    // `[Tokens, intermediate]` directly, so no `per_route` appears below.
    gate_out: Out<0, bf16>,
    up_out: Out<1, bf16>,
    // Unbound: a packed weight's checkpoint property, not a launch shape.
    // Also why this symbol never fires: `Fire::wna16_group_size` answers
    // `None` unconditionally, so the arm refuses every fire that reaches it.
    group_size: i32,
) -> Result<(), Refusal> {
    let top_k = topk_idx.width;
    let routes = routes_of(topk_idx.rows, top_k)?;
    wna16_axis("hidden", act.width, group_size)?;
    // Spelled separately from `wna16_axis`: that helper also demands `% 8`
    // and `% group_size`, claims about the activation row, not this one.
    if gate_out.width <= 0 {
        return Err(Refusal::Empty { what: "the routed row" });
    }
    // Both views after both guards — see [`scale_rows`]. `up_out` gets
    // none: its width is never read, and nothing says it equals the first's.
    let x = act.all("hidden")?;
    let gate = gate_out.all("the routed row")?;
    let launch = routed_qmv(routes, gate.width.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_wna16.cuh",
            "::pie::quant::wna16_gate_up_decode<::pie::i32(0)>",
            launch,
            &[
                x.ptr.arg(),
                topk_idx.ptr.arg(),
                gate_packed_ptrs.ptr.arg(),
                gate_scale_ptrs.ptr.arg(),
                up_packed_ptrs.ptr.arg(),
                up_scale_ptrs.ptr.arg(),
                gate.ptr.arg(),
                up_out.ptr.arg(),
                top_k.arg(),
                // Two pitches: `x.stride.0` advances a token row,
                // `gate.stride.0` a route — `gate_out`'s width is already
                // per-route, so there's no `per_route` division to undo.
                x.stride.0.arg(),
                gate.stride.0.arg(),
                group_size.arg(),
            ],
        )
    }
}

/// The routed W4A16 down projection, decode-shaped —
/// # Safety
/// As [`wna16_gate_up_decode_bf16`], with `act` addressing `act.rows *
/// act.width` fp16 elements and `out` `out.rows * out.width` writable bf16.
#[allow(clippy::too_many_arguments)]
#[kernels_macros::routine]
pub fn wna16_down_decode_bf16(
    ctx: &Ctx,
    // [`wna16_gate_up_decode_bf16`]'s operand order, unchanged: same
    // statement shape, so `act`/`topk_idx` invert the same way.
    act: In<0, f16>,
    topk_idx: In<1, i32>,
    // [`wna16_gate_up_decode_bf16`]'s four, halved: one bank and its
    // scales, positional `Bank`s for the same reason `Weight<0, _>` would
    // silently read a different table.
    down_packed_ptrs: Bank<0, *const i32>,
    down_scale_ptrs: Bank<1, *const c_void>,
    // The down leg swaps which side each width comes from: `hidden` is the
    // result's (`out.width`) and `intermediate` the activation's.
    out: Out<0, bf16>,
    // Unbound and fatal, exactly as the gate/up leg's: a packed weight's
    // checkpoint property that `Fire::wna16_group_size` never supplies.
    group_size: i32,
) -> Result<(), Refusal> {
    /// `dequant_wna16.cu` — [`routed_qmv`]'s two axes swapped.
    const fn routed_qmv_transposed(routes: u32, width: u32) -> Launch {
        Launch::grid([width.div_ceil(BLOCK / WARP), routes, 1], [BLOCK, 1, 1])
    }

    let top_k = topk_idx.width;
    let routes = routes_of(topk_idx.rows, top_k)?;
    wna16_axis("intermediate", act.width, group_size)?;
    // The zero clause, as in [`wna16_gate_up_decode_bf16`] and for its reason.
    if out.width <= 0 {
        return Err(Refusal::Empty { what: "the routed row" });
    }
    // Both views after both guards, as in the gate/up leg.
    let x = act.all("intermediate")?;
    let y = out.all("the routed row")?;
    let launch = routed_qmv_transposed(routes, y.width.unsigned_abs());
    // SAFETY: `call()`'s contract — every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_wna16.cuh",
            "::pie::quant::wna16_down_decode<::pie::i32(0)>",
            launch,
            &[
                x.ptr.arg(),
                topk_idx.ptr.arg(),
                down_packed_ptrs.ptr.arg(),
                down_scale_ptrs.ptr.arg(),
                y.ptr.arg(),
                top_k.arg(),
                // The gate/up leg's two pitches with the sides swapped:
                // here both regions are strided by the route, and the
                // result's width is the `hidden` the transposed grid covers.
                y.stride.0.arg(),
                x.stride.0.arg(),
                group_size.arg(),
            ],
        )
    }
}

// The derived operand column: every `pub fn` below carries
// `#[kernels_macros::routine]`, emitting a `&[kernels::Derived]` naming each
// parameter's source. These `assert!`s pin what it derives today — a
// `const` can't be a test, so a derivation change stops this compiling.
const _: () = {
    assert!(<scale_rows as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<scale_rows as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<scale_rows as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));

    assert!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 10);
    assert!(matches!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    // `stated` blocks a binder from "correcting" this index to a guess.
    assert!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(matches!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    // The scales plane: swapping in `WeightUpBias` here would compile and
    // bind a plane that is usually absent.
    assert!(kernels::source_is_named(
        &<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[3].source,
        <kernels::keys::WeightScales as kernels::keys::Fact>::KEY
    ));
    // The nulled bias planes: `Unstated`, not a gap — their existence is a
    // property of which gpt-oss export loaded, which no statement observes.
    assert!(!<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[4].nullable);
    assert!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[4].source.is_none());
    assert!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[5].source.is_none());
    // The two GLU constants, pinned apart: adjacent same-shaped `f32`s, so a
    // swap would be invisible to both the type system and this length.
    assert!(kernels::source_is_named(
        &<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[8].source,
        <kernels::keys::GluLimit as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[9].source,
        <kernels::keys::GluAlpha as kernels::keys::Fact>::KEY
    ));

    // `scale_ptrs`/`bias_ptrs`, pinned apart: two suffixes off one bank,
    // adjacent slots, same pointer type — a swap a length check would miss.
    assert!(<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(kernels::source_is_named(
        &<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED[3].source,
        <kernels::keys::WeightScales as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED[4].source,
        <kernels::keys::WeightBias as kernels::keys::Fact>::KEY
    ));
    assert!(matches!(<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED[5].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The mirror's load-bearing record: `[0]`/`[1]` are `In(0)`/`In(1)` here,
    // the opposite twenty lines up — only these four lines catch a reorder.
    assert!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 9);
    assert!(matches!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(matches!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    // The positional bank, not `Facts::weight_named`.
    assert!(matches!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[5].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 3))));
    assert!(matches!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[6].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED[7].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));

    // The down leg's `[0]`/`[1]` follow its gate/up twin, not the MXFP4
    // pair — the direction a reader is likeliest to guess backwards.
    assert!(<wna16_down_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(<wna16_down_decode_bf16 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<wna16_down_decode_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<wna16_down_decode_bf16 as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The weight's own shape rides `params[1..3]`, `group_size` `params[0]`.
    assert!(<dequant_wna16_int4b8_to as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(<dequant_wna16_int4b8_to as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<dequant_wna16_int4b8_to as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(<dequant_wna16_int4b8_to as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
    assert!(matches!(<dequant_wna16_int4b8_to as ::kernels::Derivation>::DERIVED[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));

    // The weight walkers, pinned `None` on purpose: the arm passes
    // `cx.rows().count` for a checkpoint's output-channel count, so copying
    // the arm to "fix" the column means deleting this line first.
    assert!(<dequant_fp8_e4m3_to_bf16_per_channel as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(matches!(<dequant_fp8_e4m3_to_bf16_per_channel as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(<dequant_fp8_e4m3_to_bf16_per_channel as ::kernels::Derivation>::DERIVED[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
};

/// This family's routines, and what a trace may say about each: none
/// carries `whole`, `in_place` or `depth_prefix_plan` — a family of pure
/// element-for-element conversions states nothing extra.
pub static ROUTINES: &[Routine] = &[
    routine!(cast_fp32_to_bf16 = cast_fp32_to::<bf16>, ),
    routine!(scale_rows_bf16 = scale_rows::<bf16>, ),
    routine!(bf16_to_fp16, ),
    routine!(dequant_fp8_e4m3_to_bf16 = dequant_fp8_e4m3_to::<bf16>, ),
    routine!(dequant_fp8_e4m3_to_bf16_per_channel, ),
    routine!(dequant_fp8_e4m3_to_bf16_per_group, ),
    routine!(dequant_mxfp4_to_bf16 = dequant_mxfp4_to::<bf16>, ),
    routine!(dequant_wna16_int4b8_to_bf16 = dequant_wna16_int4b8_to::<bf16>, ),
    routine!(mxfp4_scales_to_marlin_e8m0, ),
    routine!(quantize_bf16_to_mxfp4_e2m1_per_block, ),
    routine!(quantize_bf16_to_fp8_e4m3_per_channel, ),
    routine!(quantize_bf16_to_int8_per_channel, ),
    routine!(dequant_int8_to_bf16_per_channel, ),
    routine!(dequant_int32_w8a8_to_bf16, ),
    routine!(quantize_bf16_to_fp8_e4m3_per_token_group, ),
    routine!(mxfp4_moe_gate_up_decode_bf16, ),
    routine!(mxfp4_moe_down_decode_bf16, ),
    routine!(wna16_gate_up_decode_bf16, ),
    routine!(wna16_down_decode_bf16, ),
];

/// `quant`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

#[cfg(test)]
mod tests {
    use super::{Ctx, In, Out, Refusal, scale_rows};
    use crate::jit::abi::bf16;

    /// A degenerate extent is refused, and a negative one is too.
    ///
    /// The zero case is also caught downstream (the JIT refuses an empty
    /// grid); the negative case would not be, since `unsigned_abs` turns -4
    /// into a plausible four-row launch. A runtime `assert_eq!`, not a
    /// compile-time `assert!` — the value isn't fixed until the call.
    #[test]
    fn a_degenerate_or_negative_extent_is_refused() {
        // SAFETY: the stream is never used -- every case here refuses first.
        let ctx = unsafe { Ctx::on(std::ptr::null_mut()) };
        let buf = std::ptr::null_mut::<bf16>();
        let l = std::ptr::null::<bf16>();
        for rows in [0, -1, -4, i32::MIN] {
            assert_eq!(
                scale_rows(&ctx, Out { ptr: buf, rows, width: 128 }, In { ptr: l, rows: 0, width: 0 }),
                Err(Refusal::Empty { what: "rows" }),
                "rows {rows} must refuse"
            );
        }
        for width in [0, -1, -128, i32::MIN] {
            assert_eq!(
                scale_rows(&ctx, Out { ptr: buf, rows: 4, width }, In { ptr: l, rows: 0, width: 0 }),
                Err(Refusal::Empty { what: "width" }),
                "width {width} must refuse"
            );
        }
    }
}

// `dequant_fp8_e4m3_to_bf16_per_group`'s three marked parameters, pinned to
// the sources the arm used to bind by hand.
const _: () = {
    let d = <dequant_fp8_e4m3_to_bf16_per_group as kernels::Derivation>::DERIVED;
    assert!(d.len() == 6);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[3].source, Some(kernels::Source::Named(_))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::OutWidth, 0))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
};

// The weight's shape rides `params[1..3]`; `params[0]` is the per-tensor
// scale, read as f32, so the two channels do not collide.
const _: () = {
    let d = <dequant_fp8_e4m3_to as kernels::Derivation>::DERIVED;
    assert!(d.len() == 5);
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
};

// Seven of the nine scalars are `Param<N, i32>` at specific slots (an
// off-by-one silently repacks the wrong window); the other two (`d[4]`,
// `d[9]`) are `None` — a weight's own shape. `raw`/`out` stay `In(0)`/`Out(0)`.
const _: () = {
    let d = <mxfp4_scales_to_marlin_e8m0 as kernels::Derivation>::DERIVED;
    assert!(d.len() == 11);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // One assert per slot (a `const` can't loop); the slot number is the
    // only record of which `params[]` entry each scalar came from.
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[4].source, Some(kernels::Source::Named(_))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::Param, 3))));
    assert!(matches!(d[7].source, Some(kernels::Source::Slot(kernels::Kind::Param, 4))));
    assert!(matches!(d[8].source, Some(kernels::Source::Slot(kernels::Kind::Param, 5))));
    assert!(matches!(d[9].source, Some(kernels::Source::Slot(kernels::Kind::OutWidth, 0))));
    assert!(matches!(d[10].source, Some(kernels::Source::Slot(kernels::Kind::Param, 6))));
};

// The five weight walkers: every slot sourced, so each row derives.
const _: () = {
    let mut ok = true;
    let mut i = 0;
    let d = <dequant_wna16_int4b8_to as kernels::Derivation>::DERIVED;
    while i < d.len() {
        if d[i].source.is_none() {
            ok = false;
        }
        i += 1;
    }
    assert!(ok, "dequant_wna16_int4b8_to has an unsourced slot");
};

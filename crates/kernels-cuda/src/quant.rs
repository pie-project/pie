//! `quant` — the dtype crossings: cast, scale, quantize, dequantize, and the
//! MXFP4/WNA16 decode GEMVs that read a stored bank without materialising it.
//!
//! `In`/`Out` are a statement's operands and results, `Env` is the fire's
//! rather than the trace's, and `Bank` is a weight's stored form. Scale
//! granularity (`_per_channel`, `_per_group`, `_per_token_group`) is part of
//! the symbol name because the kernels differ, not the call. Every
//! `# Safety` section also assumes `ctx`'s stream is live across the launch.
//!
//! # The six that take raw pointers
//!
//! `In`/`Out` say a STATEMENT places this operand, so a launcher no trace
//! reaches must not wear them: the wrapper would promise a statement that does
//! not exist, and its `rows`/`width` would be the zeros every caller already
//! passed. Those six are plain `unsafe fn`s, out of [`ROUTINES`] and out of
//! `driver-cuda`'s arm registry, reached by path from their one caller:
//!
//! * THE LOADER'S TWO — [`quantize_bf16_to_mxfp4_e2m1_per_block`] and
//!   [`quantize_bf16_to_fp8_e4m3_per_channel`]. `model-loader`'s transform
//!   plan runs them once at load over a CHECKPOINT MATRIX. The plan still
//!   names them as strings (`plan/passes/tile.rs`), which is the loader's own
//!   vocabulary and not this crate's registry.
//! * THE QUANTISED GEMM'S STAGING FOUR — [`quantize_bf16_to_int8_per_channel`],
//!   [`dequant_int8_to_bf16_per_channel`], [`dequant_int32_w8a8_to_bf16`] and
//!   [`quantize_bf16_to_fp8_e4m3_per_token_group`], fired mid-matmul by
//!   `gemm/quant.rs`. A text states the matmul and the weight REPRESENTATION;
//!   the staging is what reading that representation costs, chosen inside the
//!   body from the dtypes.
//!
//! A `#[routine]` wrapper could be added over any of them the day a trace
//! states one — the pure `fn` is the half that would not need rewriting.

use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch};
use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use kernels::routine::{Asks, Const, In, InOut, Out};
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
#[routine(bf16)]
pub fn cast_fp32_to<T>(
    ctx: &Ctx<'_>,
    src_fp32: In<Tensor<f32>>,
    dst_bf16: Out<Tensor<T>>) -> Result<(), Refusal> {
    // THE ELEMENT COUNT COMES OFF THE MARK, not off a fact. `keys::
    // OutElements0` is *"rows times the result's row width"* -- the two
    // numbers this operand already carries -- and asking for it made a
    // hand-written caller unable to state it at all: `Ctx::on(stream)` has no
    // fire behind it, so the ask refuses. `Region::elements` is the same
    // product, read from the value the caller placed.
    let n = usize::try_from(dst_bf16.all("the cast's destination width")?.elements())
        .map_err(|_| Refusal::Empty { what: "the cast's element count" })?;
    let launch = elementwise(extent("quant::cast_fp32_to_bf16", n));
    ctx.fire(Fire::at("quant/dtype_cast.cuh", crate::jit::symbol(&format!("::pie::quant::cast_f32_to<{}>", T::CPP))).apply(launch), &[src_fp32.arg(), dst_bf16.arg(), n.arg()])
}

/// `buf[r, c] *= l[c]` over a `rows x width` bf16 buffer, in place.
/// # Safety
/// `buf_bf16` must address `buf_bf16.rows * buf_bf16.width` writable bf16
/// elements, `l_bf16` `buf_bf16.width` readable ones.
#[routine(bf16)]
pub fn scale_rows<T>(
    ctx: &Ctx<'_>,
    // In place: `dsl::cuda::scale_rows` hands back the same buffer it took as
    // input 0, so this result is also that operand -- which is `InOut`, and
    // which is also what makes `l_bf16` input 1 rather than input 0.
    buf_bf16: InOut<Tensor<T>>,
    l_bf16: In<Tensor<T>>,
    // Load-bearing: `route_rows` clamps a zero width up to one warp, so an
    // unguarded empty rectangle would launch successfully and do nothing.


) -> Result<(), Refusal> {
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
    ctx.fire(Fire::at("quant/dtype_cast.cuh", crate::jit::symbol(&format!("::pie::quant::scale_rows<{}>", T::CPP))).apply(launch), &[buf.ptr.arg(), l_bf16.arg(), buf.stride.arg()])
}

/// Narrow a bf16 activation to fp16 — `quant::bf16_to_fp16`.
/// # Errors
/// [`Refusal::Empty`] for a non-positive output rectangle, and
/// [`Refusal::Narrow`] when it does not fit one 32-bit launch extent.
/// # Safety
/// `in_bf16` must address `n` live bf16 elements, `out_fp16` `n` writable fp16 elements.
#[routine]
pub fn bf16_to_fp16(
    ctx: &Ctx<'_>,
    in_bf16: In<Tensor<bf16>>,
    out_fp16: Out<Tensor<f16>>) -> Result<(), Refusal> {
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
    ctx.fire(Fire::at("quant/dequant_wna16.cuh", "::pie::quant::bf16_to_narrow<::pie::f16>").apply(launch), &[in_bf16.arg(), out_fp16.arg(), count.arg()])
}

/// One f32 scale for a whole FP8 E4M3 tensor —
/// # Safety
/// `fp8_in` addresses `n` live E4M3 bytes, `bf16_out` `n` writable bf16 elements.
#[routine(bf16)]
pub fn dequant_fp8_e4m3_to<T>(
    ctx: &Ctx<'_>,
    fp8_in: In<Tensor<u8>>,
    bf16_out: Out<Tensor<T>>,
    // A weight walker: this launcher and those through
    // `mxfp4_scales_to_marlin_e8m0` dequantise a weight, whose shape is a
    // checkpoint property, not a fire fact. Its leading extent (`rows`/
    // `out_dim`) has no `Source`: `keys::Rows` would compile but read the
    // fire's token count instead. Its trailing extent (`cols`/`in_dim`) is
    // a row pitch a region already carries; `scale` here is `params[0]`.
    scale: Const<f32>,
    // The weight's own shape, which the statement carries at `params[1..3]`.
    // Not `keys::Rows` — that is the fire's token count.
    rows: Const<i32>,
    cols: Const<i32>) -> Result<(), Refusal> {
    let n = (*rows as usize).saturating_mul(*cols as usize);
    let launch = elementwise(extent("quant::dequant_fp8_e4m3_to_bf16", n));
    ctx.fire(Fire::at("quant/dequant_fp8.cuh", crate::jit::symbol(&format!("::pie::quant::dequant_fp8_e4m3<{}>", T::CPP))).apply(launch), &[fp8_in.arg(), bf16_out.arg(), scale.arg(), n.arg()])
}

/// One f32 scale per output channel —
/// # Safety
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scale_inv` `rows` f32.
#[routine]
pub fn dequant_fp8_e4m3_to_bf16_per_channel(
    ctx: &Ctx<'_>,
    fp8_in: In<Tensor<u8>>,
    bf16_out: Out<Tensor<bf16>>,
    scale_inv: In<Tensor<f32>>,
    // The weight's own shape, at `params[1..3]`; see `dequant_fp8_e4m3_to`.
    rows: Const<i32>,
    cols: Const<i32>) -> Result<(), Refusal> {
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    ctx.fire(Fire::at("quant/dequant_fp8.cuh", "::pie::quant::dequant_fp8_e4m3_per_channel<::pie::bf16>").apply(launch), &[fp8_in.arg(), bf16_out.arg(), scale_inv.arg(), cols.arg()])
}

/// One f32 scale per contiguous group along K, the DeepSeek block-FP8 weight
/// # Safety
/// `fp8_in` addresses `rows * cols` live E4M3 bytes, `bf16_out` as many
/// writable bf16 elements, `scales` `rows * ceil(cols / group_size)` f32.
#[routine]
pub fn dequant_fp8_e4m3_to_bf16_per_group(
    ctx: &Ctx<'_>,
    fp8_in: In<Tensor<u8>>,
    bf16_out: Out<Tensor<bf16>>,
    scales: In<Tensor<f32>>,
    group_size: Const<i32>) -> Result<(), Refusal> {
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let cols = bf16_out.width;
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    ctx.fire(Fire::at("quant/dequant_fp8.cuh", "::pie::quant::dequant_fp8_e4m3_per_group<::pie::bf16>").apply(launch), &[fp8_in.arg(), bf16_out.arg(), scales.arg(), cols.arg(), group_size.arg()])
}

/// Packed E2M1 nibbles and E8M0 block scales to bf16 —
/// # Safety
/// `packed` addresses `out_dim * in_dim / 2` live bytes, `block_scale`
/// `out_dim * in_dim / 32`, `out` `out_dim * in_dim` writable bf16 elements.
#[routine(bf16)]
pub fn dequant_mxfp4_to<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<u8>>,
    block_scale: In<Tensor<u8>>,
    out: Out<Tensor<T>>,
    // The weight's own shape, at `params[0..2]`; this form spends no `params`
    // on a scale, so it starts at zero.
    out_dim: Const<i32>,
    in_dim: Const<i32>) -> Result<(), Refusal> {
    let launch = route_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    ctx.fire(Fire::at("quant/dequant_fp4.cuh", crate::jit::symbol(&format!("::pie::quant::dequant_mxfp4<{}>", T::CPP))).apply(launch), &[packed.arg(), block_scale.arg(), out.arg(), in_dim.arg()])
}

/// INT4B8 words with a bf16 scale per group along K —
/// # Safety
/// `packed` addresses `out_dim * in_dim / 8` live `int32`s, `scale`
/// `out_dim * in_dim / group_size` bf16, `out` `out_dim * in_dim` writable bf16.
#[routine(bf16)]
pub fn dequant_wna16_int4b8_to<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<i32>>,
    scale: In<Tensor<T>>,
    out: Out<Tensor<T>>,
    // FIRST, BECAUSE IT IS `params[0]` -- the quantisation convention
    // `dequant_fp8_e4m3_to_bf16_per_group` shares. The slot is the mark's
    // position among the params marks now, so the order in this list IS the
    // order in the statement's run.
    group_size: Const<i32>,
    // A weight walker — see [`dequant_fp8_e4m3_to`]'s block.
    // The weight's own shape, at `params[1..3]`.
    out_dim: Const<i32>,
    in_dim: Const<i32>) -> Result<(), Refusal> {
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
    ctx.fire(Fire::at("quant/dequant_wna16.cuh", crate::jit::symbol(&format!("::pie::quant::dequant_wna16_int4b8<{}>", T::CPP))).apply(launch), &[packed.arg(), scale.arg(), out.arg(), in_dim.arg(), group_size.arg()])
}

/// E8M0 block scales into Marlin's order —
/// # Safety
/// `raw` addresses `source_rows * source_stride_groups` live E8M0 bytes, `out`
/// `selected_rows * target_groups` writable ones.
#[routine]
pub fn mxfp4_scales_to_marlin_e8m0(
    ctx: &Ctx<'_>,
    raw: In<Tensor<u8>>,
    out: Out<Tensor<u8>>,
    // Nine scalars: seven ride `params[]` (`Param<N, i32>`); `selected_rows`
    // and `target_groups` are a weight-shaped result's own extents (no
    // `Source` names them — see [`dequant_fp8_e4m3_to`]'s block), and the
    // two offsets are into the checkpoint's scale table, a fact no fire
    // holds. No windowed view is built (a window needs a byte offset, which
    // needs the pointee's size this signature lacks), so the kernel bounds-
    // checks and zero-fills silently past them (`mxfp4_marlin.cuh`) instead.
    source_rows: Const<i32>,
    source_row_offset: Const<i32>,
    valid_rows: Const<i32>,
    source_stride_groups: Const<i32>,
    source_group_offset: Const<i32>,
    source_groups: Const<i32>,
    row_select: Const<i32>) -> Result<(), Refusal> {
    let selected_rows = ctx.ask::<i32, keys::Rows>()?;
    let target_groups = out.width;
    let total = selected_rows.unsigned_abs().saturating_mul(target_groups.unsigned_abs());
    let launch = elementwise(total);
    ctx.fire(Fire::at("quant/mxfp4_marlin.cuh", "::pie::quant::mxfp4_scales_to_marlin_e8m0<::pie::u8>").apply(launch), &[
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
            ])
}

/// A bf16 rectangle to MXFP4 nibbles plus their E8M0 block scales —
///
/// One of THE LOADER'S TWO, and not a `#[routine]`: `model-loader`'s transform
/// plan fires it once at load against a checkpoint matrix, so there is no
/// statement to derive an operand from and no fire whose rectangle `rows` and
/// `cols` could be. See the module header.
///
/// # Safety
/// `w_bf16` addresses `rows * cols` live bf16, `w_packed` `rows * cols / 2`
/// writable bytes, `w_scale_e8m0` `rows * cols / 32` writable bytes.
pub unsafe fn quantize_bf16_to_mxfp4_e2m1_per_block(
    ctx: &Ctx<'_>,
    w_bf16: *const bf16,
    w_packed: *mut u8,
    w_scale_e8m0: *mut u8,
    rows: i32,
    cols: i32) -> Result<(), Refusal> {
    if cols < 32 {
        return Err(Refusal::Narrow { what: "cols, in 32-element blocks", at: i64::from(cols) });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs() / 32);
    ctx.fire(Fire::at("quant/quant_bf16_to_mxfp4.cuh", "::pie::quant::quant_bf16_to_mxfp4_row<::pie::bf16>").apply(launch), &[w_bf16.arg(), w_packed.arg(), w_scale_e8m0.arg(), cols.arg()])
}

/// Per-row FP8 E4M3 quantisation with the scale emitted beside it —
///
/// The loader's other quantiser — see [`quantize_bf16_to_mxfp4_e2m1_per_block`].
///
/// # Safety
/// `w_bf16` addresses `rows * cols` live bf16, `w_fp8` as many writable
/// bytes, `scale_inv` `rows` writable f32.
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_channel(
    ctx: &Ctx<'_>,
    w_bf16: *const bf16,
    w_fp8: *mut u8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32) -> Result<(), Refusal> {
    let launch = rms(rows.unsigned_abs());
    ctx.fire(Fire::at("quant/quant_bf16_to_fp8.cuh", "::pie::quant::quant_per_channel<::pie::quant::fp8_e4m3>").apply(launch), &[w_bf16.arg(), w_fp8.arg(), scale_inv.arg(), cols.arg()])
}

/// Per-row symmetric INT8 quantisation of a `[rows, cols]` bf16 rectangle —
///
/// THE QUANTISED GEMM'S STAGING FOUR (with
/// [`dequant_int8_to_bf16_per_channel`], [`dequant_int32_w8a8_to_bf16`] and
/// [`quantize_bf16_to_fp8_e4m3_per_token_group`]): none is a `#[routine]`,
/// because `gemm/quant.rs` fires them from inside a matmul body. The caller is
/// a matmul, not a fire, so no trace states the rectangle. See the module
/// header.
///
/// # Safety
/// `w_bf16` addresses `rows * cols` live bf16, `out_int8` as many writable
/// signed bytes, `scale_inv` `rows` writable f32.
pub unsafe fn quantize_bf16_to_int8_per_channel(
    ctx: &Ctx<'_>,
    w_bf16: *const bf16,
    out_int8: *mut i8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32) -> Result<(), Refusal> {
    let launch = rms(rows.unsigned_abs());
    ctx.fire(Fire::at("quant/quant_bf16_to_fp8.cuh", "::pie::quant::quant_per_channel<::pie::quant::int8_sym>").apply(launch), &[w_bf16.arg(), out_int8.arg(), scale_inv.arg(), cols.arg()])
}

/// INT8 back to bf16 through a per-channel scale —
///
/// Staging four — see [`quantize_bf16_to_int8_per_channel`].
///
/// # Safety
/// `w_int8` addresses `rows * cols` live signed bytes, `out` as many writable
/// bf16, `scale_inv` `rows` f32.
pub unsafe fn dequant_int8_to_bf16_per_channel(
    ctx: &Ctx<'_>,
    w_int8: *const i8,
    out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32) -> Result<(), Refusal> {
    let n = rows.unsigned_abs() as usize * cols.unsigned_abs() as usize;
    let launch = elementwise(extent("quant::dequant_int8_to_bf16_per_channel", n));
    ctx.fire(Fire::at("quant/quant_bf16_to_fp8.cuh", "::pie::quant::dequant_int8_per_channel<::pie::bf16>").apply(launch), &[w_int8.arg(), out.arg(), scale_inv.arg(), cols.arg(), n.arg()])
}

/// The W8A8 epilogue: an `[M, N]` int32 accumulator widened to bf16 through a
///
/// Staging four — see [`quantize_bf16_to_int8_per_channel`]. `n` here is a
/// width, not the flat element count other launchers spell the same way.
///
/// # Safety
/// `acc` addresses `m * n` live i32, `act_scale_inv` `m` f32, `w_scale_inv`
/// `n` f32, `out` `m * n` writable bf16.
pub unsafe fn dequant_int32_w8a8_to_bf16(
    ctx: &Ctx<'_>,
    acc: *const i32,
    act_scale_inv: *const f32,
    w_scale_inv: *const f32,
    out: *mut bf16,
    m: i32,
    n: i32) -> Result<(), Refusal> {
    const W8A8_BY: u32 = 8;

    /// `quant_bf16_to_fp8.cu` — `constexpr int BX = 32, BY = 8;`, the W8A8 tile.
    const W8A8_BX: u32 = 32;

    let launch = Launch::grid(
        [n.unsigned_abs().div_ceil(W8A8_BX), m.unsigned_abs().div_ceil(W8A8_BY), 1],
        [W8A8_BX, W8A8_BY, 1],
    );
    ctx.fire(Fire::at("quant/quant_bf16_to_fp8.cuh", "::pie::quant::w8a8_dequant").apply(launch), &[acc.arg(), act_scale_inv.arg(), w_scale_inv.arg(), out.arg(), m.arg(), n.arg()])
}

/// Blockwise (per-token-group) FP8 E4M3 activation quantisation — the
///
/// Staging four — see [`quantize_bf16_to_int8_per_channel`]. `m` is a token
/// count, still unbindable: the caller is a matmul body, not a fire.
///
/// # Safety
/// `act_bf16` addresses `m * k` live bf16, `act_fp8` as many writable bytes,
/// `act_scale` `m * ceil(k / group_size)` writable f32.
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_token_group(
    ctx: &Ctx<'_>,
    act_bf16: *const bf16,
    act_fp8: *mut u8,
    act_scale: *mut f32,
    m: i32,
    k: i32,
    group_size: i32) -> Result<(), Refusal> {
    /// `quant_bf16_to_fp8.cu` — the blockwise FP8 quantiser's `128`.
    const GROUP_QUANT_BLOCK: u32 = 128;

    let n_groups = (k + group_size - 1) / group_size;
    let launch =
        Launch::grid([n_groups.unsigned_abs(), m.unsigned_abs(), 1], [GROUP_QUANT_BLOCK, 1, 1]);
    ctx.fire(Fire::at("quant/quant_bf16_to_fp8.cuh", "::pie::quant::quant_act_fp8_per_group").apply(launch), &[
                act_bf16.arg(),
                act_fp8.arg(),
                act_scale.arg(),
                m.arg(),
                k.arg(),
                group_size.arg(),
                n_groups.arg(),
            ])
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
#[routine]
pub fn mxfp4_moe_gate_up_decode_bf16(
    ctx: &Ctx<'_>,
    // THE ORDER IS THE SLOT. The statement places `vec![experts.id, x.id]`,
    // so the index run is operand 0 and the activation operand 1 — and the
    // mark's POSITION is what says so now, where `In<1, _>`/`In<0, _>` used
    // to say it against the declaration order. Swapping these two lines binds
    // the index run where the activation belongs, and compiles.
    topk_idx: In<Tensor<i32>>,
    act: In<Tensor<f16>>,
    // The one bank the statement names. `Const<Tensor<_>>` derives the chain
    // `Or(Named("weight"), Slot(Weight, 0))` — the named bank first and the
    // positional one after.
    packed_ptrs: Const<Tensor<u8>>,
    // `gate_out.width` is the fanned-out result row that `per_route` divides.
    gate_out: Out<Tensor<bf16>>,
    up_out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let glu_limit = ctx.ask::<f32, keys::GluLimit>()?;
    let glu_alpha = ctx.ask::<f32, keys::GluAlpha>()?;

    // The fanout is `topk_idx`'s own rectangle: the index run is
    // `[Tokens, Const(top_k)]`, one `i32` per (token, route).
    // THE BANK'S OTHER THREE PLANES, ASKED FOR RATHER THAN PLACED. The driver
    // resolves `_scales`/`_gate_bias`/`_up_bias` off the ONE named bank by
    // suffix, so no statement places them and none ever did (they were
    // `Env<keys::WeightScales>` and two nulls). A `Const<Tensor<_>>` here
    // would derive `Or(Named("weight2"), Slot(Weight, 1))` — `spec.weight2`,
    // a different fact — and would also make the statement one weight short.
    //
    // Whether either bias plane exists is a property of which gpt-oss export
    // loaded; no CUDA statement observes the tensor inventory, so an absent
    // plane binds null instead of refusing.
    let scale_ptrs = ctx.ask::<*const u8, keys::WeightScales>()?;
    let gate_bias_ptrs = ctx.absent()?;
    let up_bias_ptrs = ctx.absent()?;
    let top_k = topk_idx.width;
    let intermediate = per_route(gate_out.width, top_k)?;
    let routes = routes_of(topk_idx.rows, top_k)?;
    mxfp4_axis("hidden", act.width)?;
    // The activation's view, per [`scale_rows`]'s reason. `top_k` and
    // `gate_out.width` are already refused before a view could fire `Absent`.
    let x = act.all("hidden")?;
    let launch = routed_qmv_quad(routes, intermediate.unsigned_abs());
    ctx.fire(Fire::at("quant/dequant_fp4.cuh", "::pie::quant::mxfp4_moe_gate_up_decode<::pie::i32(4)>").apply(launch), &[
                x.ptr.arg(),
                topk_idx.arg(),
                packed_ptrs.arg(),
                scale_ptrs.arg(),
                gate_bias_ptrs,
                up_bias_ptrs,
                gate_out.arg(),
                up_out.arg(),
                // The optional fp16 side-output: the kernel writes it when
                // non-null, and no caller wants it.
                Option::<NonNull<f16>>::None.arg(),
                glu_limit.arg(),
                glu_alpha.arg(),
                top_k.arg(),
                // A pitch: `dequant_fp4.cuh` also reads it as the expert
                // row's word count (`words_per_row = hidden / 8`) — a
                // `Bank` has no width of its own to carry that separately.
                x.stride.arg(),
                intermediate.arg(),
            ])
}

/// gpt-oss's routed down projection, decode-shaped —
/// # Safety
/// As [`mxfp4_moe_gate_up_decode_bf16`], with `act` addressing the routed
/// extent (`act.rows * act.width`) and `out` `out.rows * out.width` writable bf16.
#[routine]
pub fn mxfp4_moe_down_decode_bf16(
    ctx: &Ctx<'_>,
    // [`mxfp4_moe_gate_up_decode_bf16`]'s bindings and reasons: the index run
    // is operand 0 and the activation operand 1, and `packed_ptrs` is the one
    // bank the statement names.
    topk_idx: In<Tensor<i32>>,
    act: In<Tensor<f16>>,
    packed_ptrs: Const<Tensor<u8>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // The `_scales` and `_bias` planes of the one stated bank, asked for as
    // its twin asks — see [`mxfp4_moe_gate_up_decode_bf16`]. `keys::WeightBias`,
    // not `keys::NamedWeight2` (`spec.weight2`, a different fact this driver
    // has); every gpt-oss export publishes `{down}_bias`.
    let scale_ptrs = ctx.ask::<*const u8, keys::WeightScales>()?;
    let bias_ptrs = ctx.ask::<*const u8, keys::WeightBias>()?;
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
    ctx.fire(Fire::at("quant/dequant_fp4.cuh", "::pie::quant::mxfp4_moe_down_decode<::pie::i32(4)>").apply(launch), &[
                act.arg(),
                topk_idx.arg(),
                packed_ptrs.arg(),
                scale_ptrs.arg(),
                bias_ptrs.arg(),
                out.arg(),
                hidden.arg(),
                intermediate.arg(),
            ])
}

/// The routed W4A16 gate and up projections, decode-shaped —
/// # Safety
/// `act`/`topk_idx` address their `.rows * .width` extents (fp16, `int32`);
/// the four banks one device pointer per expert; `gate_out`/`up_out` each
/// `gate_out.rows * topk_idx.width * gate_out.width` writable bf16 elements.
#[routine]
pub fn wna16_gate_up_decode_bf16(
    ctx: &Ctx<'_>,
    // Reversed from the MXFP4 pair: here operand 0 is `act` (`hidden`) and
    // operand 1 is `topk_idx` (`top_k`) — not a transcription slip, the
    // statement really places them the other way round.
    act: In<Tensor<f16>>,
    topk_idx: In<Tensor<i32>>,
    // Four positional banks (`args[n_in + n_out]`), read via `Bank<N, _>`;
    // `Weight<N, _>` would compile too but derive `WeightNamed`, a
    // different table, silently.
    gate_packed_ptrs: Const<Tensor<i32>>,
    gate_scale_ptrs: Const<Tensor<c_void>>,
    up_packed_ptrs: Const<Tensor<i32>>,
    up_scale_ptrs: Const<Tensor<c_void>>,
    // `gate_out.width` is already per-route here, unlike the mxfp4 pair's
    // (`[Tokens, top_k, intermediate]`, undone by `per_route`): this states
    // `[Tokens, intermediate]` directly, so no `per_route` appears below.
    gate_out: Out<Tensor<bf16>>,
    up_out: Out<Tensor<bf16>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    group_size: i32) -> Result<(), Refusal> {
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
    ctx.fire(Fire::at("quant/dequant_wna16.cuh", "::pie::quant::wna16_gate_up_decode<::pie::i32(0)>").apply(launch), &[
                x.ptr.arg(),
                topk_idx.arg(),
                gate_packed_ptrs.arg(),
                gate_scale_ptrs.arg(),
                up_packed_ptrs.arg(),
                up_scale_ptrs.arg(),
                gate.ptr.arg(),
                up_out.arg(),
                top_k.arg(),
                // Two pitches: `x.stride.0` advances a token row,
                // `gate.stride.0` a route — `gate_out`'s width is already
                // per-route, so there's no `per_route` division to undo.
                x.stride.arg(),
                gate.stride.arg(),
                group_size.arg(),
            ])
}

/// The routed W4A16 down projection, decode-shaped —
/// # Safety
/// As [`wna16_gate_up_decode_bf16`], with `act` addressing `act.rows *
/// act.width` fp16 elements and `out` `out.rows * out.width` writable bf16.
#[routine]
pub fn wna16_down_decode_bf16(
    ctx: &Ctx<'_>,
    // [`wna16_gate_up_decode_bf16`]'s operand order, unchanged: Env<same, keys::Unstated>
    // statement shape, so `act`/`topk_idx` invert the same way.
    act: In<Tensor<f16>>,
    topk_idx: In<Tensor<i32>>,
    // [`wna16_gate_up_decode_bf16`]'s four, halved: Env<one bank and its, keys::Unstated>
    // scales, positional `Bank`s for the same reason `Weight<0, _>` would
    // silently read a different table.
    down_packed_ptrs: Const<Tensor<i32>>,
    down_scale_ptrs: Const<Tensor<c_void>>,
    // The down leg swaps which side each width comes from: `hidden` is the
    // result's (`out.width`) and `intermediate` the activation's.
    out: Out<Tensor<bf16>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    group_size: i32) -> Result<(), Refusal> {
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
    ctx.fire(Fire::at("quant/dequant_wna16.cuh", "::pie::quant::wna16_down_decode<::pie::i32(0)>").apply(launch), &[
                x.ptr.arg(),
                topk_idx.arg(),
                down_packed_ptrs.arg(),
                down_scale_ptrs.arg(),
                y.ptr.arg(),
                top_k.arg(),
                // The gate/up leg's two pitches with the sides swapped:
                // here both regions are strided by the route, and the
                // result's width is the `hidden` the transposed grid covers.
                y.stride.arg(),
                x.stride.arg(),
                group_size.arg(),
            ])
}

// The derived operand column: every `pub fn` below carries
// `#[routine]`, emitting a `&[kernels::Derived]` naming each
// parameter's source. These `assert!`s pin what it derives today — a
// `const` can't be a test, so a derivation change stops this compiling.
const _: () = {
    assert!(<scale_rows as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(scale_rows::<bf16>)[0], Some(kernels::Source::Alias(0, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(scale_rows::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));

    // SIX, NOT TEN: the bank's `_scales`/`_gate_bias`/`_up_bias` planes are
    // asked for in the body, not placed by the statement. `[0]` is the INDEX
    // run and `[1]` the activation — this family states `vec![experts.id,
    // x.id]`, so the order of these two lines is the binding.
    assert!(<mxfp4_moe_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_gate_up_decode_bf16)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_gate_up_decode_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_gate_up_decode_bf16)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_gate_up_decode_bf16)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // The two GLU constants are ASKED for now, so nothing follows the second
    // result in this column.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_gate_up_decode_bf16)[4], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));

    // The down leg's mirror, four marks: index, activation, the one bank and
    // the result. Its `_scales` and `_bias` planes are asked for too.
    assert!(<mxfp4_moe_down_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_down_decode_bf16)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_down_decode_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_down_decode_bf16)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_moe_down_decode_bf16)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The mirror's load-bearing record: `[0]`/`[1]` are `In(0)`/`In(1)` here,
    // the opposite twenty lines up — only these four lines catch a reorder.
    assert!(<wna16_gate_up_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 9);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_gate_up_decode_bf16)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_gate_up_decode_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    // The positional bank, not `Facts::weight_named`.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_gate_up_decode_bf16)[5], Some(kernels::Source::Slot(kernels::Kind::Weight, 3))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_gate_up_decode_bf16)[6], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_gate_up_decode_bf16)[7], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));

    // The down leg's `[0]`/`[1]` follow its gate/up twin, not the MXFP4
    // pair — the direction a reader is likeliest to guess backwards.
    assert!(<wna16_down_decode_bf16 as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_down_decode_bf16)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_down_decode_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(wna16_down_decode_bf16)[4], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The weight's own shape rides `params[1..3]`, `group_size` `params[0]`.
    assert!(<dequant_wna16_int4b8_to as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_wna16_int4b8_to::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_wna16_int4b8_to::<bf16>)[3], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_wna16_int4b8_to::<bf16>)[4], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_wna16_int4b8_to::<bf16>)[5], Some(kernels::Source::Slot(kernels::Kind::Param, 2))));

    // The weight walkers, pinned `None` on purpose: the arm passes
    // `cx.rows().count` for a checkpoint's output-channel count, so copying
    // the arm to "fix" the column means deleting this line first.
    assert!(<dequant_fp8_e4m3_to_bf16_per_channel as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_fp8_e4m3_to_bf16_per_channel)[3], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_fp8_e4m3_to_bf16_per_channel)[4], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
};


#[cfg(test)]
mod tests {
    use super::{Ctx, In, InOut, Refusal, scale_rows};
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
                scale_rows::<bf16>(&ctx, InOut { ptr: buf, rows, width: 128 }, In { ptr: l, rows: 0, width: 0 }),
                Err(Refusal::Empty { what: "rows" }),
                "rows {rows} must refuse"
            );
        }
        for width in [0, -1, -128, i32::MIN] {
            assert_eq!(
                scale_rows::<bf16>(&ctx, InOut { ptr: buf, rows: 4, width }, In { ptr: l, rows: 0, width: 0 }),
                Err(Refusal::Empty { what: "width" }),
                "width {width} must refuse"
            );
        }
    }
}

// `dequant_fp8_e4m3_to_bf16_per_group`'s three marked parameters, pinned to
// the sources the arm used to bind by hand.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_fp8_e4m3_to_bf16_per_group);
    assert!(d.len() == 4);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
};

// The weight's shape rides `params[1..3]`; `params[0]` is the per-tensor
// scale, read as f32, so the two channels do not collide.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_fp8_e4m3_to::<bf16>);
    assert!(d.len() == 5);
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
};

// Seven of the nine scalars are `Param<N, i32>` at specific slots (an
// off-by-one silently repacks the wrong window); the other two (`d[4]`,
// `d[9]`) are `None` — a weight's own shape. `raw`/`out` stay `In(0)`/`Out(0)`.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(mxfp4_scales_to_marlin_e8m0);
    assert!(d.len() == 9);
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // One assert per slot (a `const` can't loop); the slot number is the
    // only record of which `params[]` entry each scalar came from.
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Param, 3))));
    assert!(matches!(d[6], Some(kernels::Source::Slot(kernels::Kind::Param, 4))));
    assert!(matches!(d[7], Some(kernels::Source::Slot(kernels::Kind::Param, 5))));
    assert!(matches!(d[8], Some(kernels::Source::Slot(kernels::Kind::Param, 6))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
};

// The five weight walkers: every slot sourced, so each row derives.
const _: () = {
    let mut ok = true;
    let mut i = 0;
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(dequant_wna16_int4b8_to::<bf16>);
    while i < d.len() {
        if d[i].is_none() {
            ok = false;
        }
        i += 1;
    }
    assert!(ok, "dequant_wna16_int4b8_to has an unsourced slot");
};

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::jit::Abi;
use crate::jit::abi::Elem;
use crate::jit::abi::{bf16, f16};
use kernels::Refusal;

use core::ffi::c_void;
use core::ptr::NonNull;

/// `quant/transcode.cuh` — one fused quant→quant kernel per (source, target).
///
/// # Why this root is not a routine, and why it exists anyway
///
/// The file's own header gives two reasons no ROW could name its kernel, and
/// the second of them has expired. The first stands: three template
/// parameters, one of them an `int`, where `DeviceKernel::instantiation`
/// spelled `path<elem>` — but a `mod inst` writes the whole template-id out,
/// so that argument was against the row world's spelling and not against this.
///
/// The second was that *"`kernels::Ty` has no kind for a by-value aggregate,
/// and `runtime::args` marshals pointers, `I32`, `U32`, `F32` and `Usize` and
/// refuses everything else"*. [`ArgValue::Bytes`](crate::jit::ArgValue::Bytes)
/// and [`by_value!`](crate::by_value) are the answer to exactly that sentence,
/// grown for XQA's `KVCacheList` and FA2's `__grid_constant__` params, and the
/// three mirrors below are that machinery used the way it was written to be
/// used. **The header now says so.**
///
/// What is still true is that no `Source` produces a Decode or an Encode: they
/// are assembled from a loader plan's tile facts, not from a trace statement.
/// So the launcher is [`driver_internal`](crate::driver_internal)'s, and it
/// has no caller — `driver-cuda/src/weights/plan.rs:212` asserts
/// `fusion_mask == 0`, which is the loader saying this driver has no fused
/// transcode kernel. It has one again; turning the bit on is the weights
/// executor's decision and not this crate's.
pub mod transcode {
    
    use crate::jit::abi::DevicePtr;

    /// `quant/transcode.cuh` — the root every (source, target) pair compiles
    /// its own symbol out of.
        /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// One per (source, target) pair, which is the `a x B` the kernel template
    /// exists to avoid hand-writing: two sources and one target is two
    /// symbols, and a third source adds one line here rather than a kernel.
    ///
    /// The group width is spelled `EncodeMxfp4::kGroup` rather than `32`.
    /// It is the same constant the functor's `encode_group` sizes its array
    /// with, and the kernel's `float vals[GROUP]` has to agree with that array
    /// or the compile is a silent mis-read of a register block — so naming the
    /// member is the spelling that cannot drift, and NVRTC lowers it because a
    /// non-type argument wants a constant expression and not a literal.
        /// `transcode.cuh:131` — a raw BF16 source, and no source scale.
    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct DecodeBf16 {
        /// `[rows, cols]` bf16, row-major.
        pub src: DevicePtr,
        /// The row stride, in elements.
        pub cols: i32,
    }

    /// `transcode.cuh:143` — FP8 E4M3 under one f32 scale per
    /// `[group_size, group_size]` tile.
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

    /// `transcode.cuh:163` — 32 floats out as an E8M0 byte and 16 packed E2M1
    /// nibble pairs.
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

    // The `static constexpr` members are not fields and take no storage, which
    // is why `EncodeMxfp4` measures 24 bytes and not 36: `kGroup`,
    // `kPackedPerByte` and `kBytesPerGroup` are compile-time constants of the
    // C++ type, and the mirror is a mirror of the OBJECT.
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

    /// The three measured layouts, as C++ `static_assert`s.
    ///
    /// `tests/typecheck_tu.rs` compiles them against this root's own text,
    /// which is the half `by_value!`'s `const` assertions cannot cover: those
    /// watch the Rust mirror, these watch the header it mirrors.
    pub static LAYOUTS: &[crate::jit::Layout] = &[
        <DecodeBf16 as crate::jit::ByValue>::LAYOUT,
        <DecodeFp8E4m3PerGroup as crate::jit::ByValue>::LAYOUT,
        <EncodeMxfp4 as crate::jit::ByValue>::LAYOUT,
    ];
}

/// `quant_bf16_to_fp8.cu:23` and `dtype_cast.cu:20` — `constexpr int BLOCK =
const BLOCK: u32 = 256;

/// A warp, for the block widths that round up to one.
const WARP: u32 = 32;

/// [`kernels::LaunchRule::Elementwise`] — `bind/launch.rs:128`.
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// [`kernels::LaunchRule::Rms`] — `bind/launch.rs:116`.
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

/// [`kernels::LaunchRule::RouteRows`] — `bind/launch.rs:157`.
fn route_rows(rows: u32, width: u32) -> Launch {
    /// The largest block CUDA will launch, which [`route_rows`] caps at.
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(rows, width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK))
}

/// [`kernels::LaunchRule::Slab`] — `runtime/launch.rs:985-1015`.
fn slab(n: u32) -> Launch {
    /// `runtime/launch.rs:659` — [`kernels::LaunchRule::Slab`] divides by the
    const SLAB_VEC: u32 = 8;

    /// `runtime/launch.rs:668` — and then caps the grid, because a slab kernel is
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
///
/// # Safety
///
/// `src_fp32` must address `n` live fp32 elements and `dst_bf16` `n` writable
/// bf16 elements, and `ctx`'s stream must be live across the launch — the same
/// obligations the caller met when this was a `pie_k_*` call handing the
/// stream to a `<<<>>>`.
pub fn cast_fp32_to<T>(
    ctx: &Ctx,
    src_fp32: *const f32,
    dst_bf16: *mut T,
    n: usize,
) -> Result<(), Refusal>
where
    T: Elem,
    *mut T: Abi,
{
    let launch = elementwise(extent("quant::cast_fp32_to_bf16", n));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dtype_cast.cuh",
            &format!("::pie::quant::cast_f32_to<{}>", T::CPP),
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
pub fn scale_rows<T>(
    ctx: &Ctx,
    buf_bf16: *mut T,
    l_bf16: *const T,
    rows: i32,
    width: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    assert!(rows > 0 && width > 0, "quant::scale_rows_bf16: {rows} x {width} is not an extent");
    let launch = route_rows(rows.unsigned_abs(), width.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dtype_cast.cuh",
            &format!("::pie::quant::scale_rows<{}>", T::CPP),
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
    let launch = slab(extent("quant::bf16_to_fp16", n));
    let count = i64::try_from(n).unwrap_or(i64::MAX);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_wna16.cuh",
            "::pie::quant::bf16_to_narrow<::pie::f16>",
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
pub fn dequant_fp8_e4m3_to<T>(
    ctx: &Ctx,
    fp8_in: *const u8,
    bf16_out: *mut T,
    scale: f32,
    n: usize,
) -> Result<(), Refusal>
where
    T: Elem,
    *mut T: Abi,
{
    let launch = elementwise(extent("quant::dequant_fp8_e4m3_to_bf16", n));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp8.cuh",
            &format!("::pie::quant::dequant_fp8_e4m3<{}>", T::CPP),
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
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp8.cuh",
            "::pie::quant::dequant_fp8_e4m3_per_channel<::pie::bf16>",
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
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp8.cuh",
            "::pie::quant::dequant_fp8_e4m3_per_group<::pie::bf16>",
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
pub fn dequant_mxfp4_to<T>(
    ctx: &Ctx,
    packed: *const u8,
    block_scale: *const u8,
    out: *mut T,
    out_dim: i32,
    in_dim: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *mut T: Abi,
{
    let launch = route_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp4.cuh",
            &format!("::pie::quant::dequant_mxfp4<{}>", T::CPP),
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
pub fn dequant_wna16_int4b8_to<T>(
    ctx: &Ctx,
    packed: *const i32,
    scale: *const T,
    out: *mut T,
    out_dim: i32,
    in_dim: i32,
    group_size: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    /// [`kernels::LaunchRule::ElementwiseRows`] — `bind/launch.rs:143`.
    const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch::grid([rows, width.div_ceil(BLOCK), 1], [BLOCK, 1, 1])
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
            "quant/dequant_wna16.cuh",
            &format!("::pie::quant::dequant_wna16_int4b8<{}>", T::CPP),
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
    let total = selected_rows.unsigned_abs().saturating_mul(target_groups.unsigned_abs());
    let launch = elementwise(total);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/mxfp4_marlin.cuh",
            "::pie::quant::mxfp4_scales_to_marlin_e8m0<::pie::u8>",
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
    if cols < 32 {
        return Err(Refusal::Narrow { what: "cols, in 32-element blocks", at: i64::from(cols) });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs() / 32);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_mxfp4.cuh",
            "::pie::quant::quant_bf16_to_mxfp4_row<::pie::bf16>",
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
    let launch = rms(rows.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_per_channel<::pie::quant::fp8_e4m3>",
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
    let launch = rms(rows.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_per_channel<::pie::quant::int8_sym>",
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
    let n = rows.unsigned_abs() as usize * cols.unsigned_abs() as usize;
    let launch = elementwise(extent("quant::dequant_int8_to_bf16_per_channel", n));
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::dequant_int8_per_channel<::pie::bf16>",
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
    /// The other half of the pair above.
    const W8A8_BY: u32 = 8;

    /// `quant_bf16_to_fp8.cu:109` — `constexpr int BX = 32, BY = 8;`, the W8A8
    const W8A8_BX: u32 = 32;

    let launch = Launch::grid(
        [n.unsigned_abs().div_ceil(W8A8_BX), m.unsigned_abs().div_ceil(W8A8_BY), 1],
        [W8A8_BX, W8A8_BY, 1],
    );
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::w8a8_dequant",
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
           /// `quant_bf16_to_fp8.cu:131` — the blockwise FP8 quantiser's `128`.
           const GROUP_QUANT_BLOCK: u32 = 128;

    let n_groups = (k + group_size - 1) / group_size;
    let launch =
        Launch::grid([n_groups.unsigned_abs(), m.unsigned_abs(), 1], [GROUP_QUANT_BLOCK, 1, 1]);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_act_fp8_per_group",
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

/// `dequant_fp4.cu:67-70` and `:152-156` — `dim3(routes, ceil(width / 16))`
const fn routed_qmv_quad(routes: u32, width: u32) -> Launch {
    /// `dequant_fp4.cu:39` — `constexpr int kMxfp4DecodeBlock = 128;`.
    const MXFP4_DECODE_BLOCK: u32 = 128;

    /// Output rows one WARP of the MXFP4 decode GEMVs owns — the template
    const MXFP4_ROWS_PER_WARP: u32 = 4;

    let tile = (MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP;
    Launch::grid([routes, width.div_ceil(tile), 1], [MXFP4_DECODE_BLOCK, 1, 1])
}

/// `dequant_wna16.cu:73-75`, before §43.9 deleted the launcher as unreached —
const fn routed_qmv(routes: u32, width: u32) -> Launch {
    Launch::grid([routes, width.div_ceil(BLOCK / WARP), 1], [BLOCK, 1, 1])
}
/// The routed fanout, checked — `num_tokens * top_k`, which every one of the
fn routes_of(num_tokens: i32, top_k: i32) -> Result<u32, Refusal> {
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
    let launch = routed_qmv_quad(routes, intermediate.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp4.cuh",
            "::pie::quant::mxfp4_moe_gate_up_decode<::pie::i32(4)>",
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
    let launch = routed_qmv_quad(routes, hidden.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_fp4.cuh",
            "::pie::quant::mxfp4_moe_down_decode<::pie::i32(4)>",
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
    let launch = routed_qmv(routes, intermediate.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_wna16.cuh",
            "::pie::quant::wna16_gate_up_decode<::pie::i32(0)>",
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
    /// `dequant_wna16.cu:101-104` — [`routed_qmv`]'s two axes SWAPPED.
    const fn routed_qmv_transposed(routes: u32, width: u32) -> Launch {
    Launch::grid([width.div_ceil(BLOCK / WARP), routes, 1], [BLOCK, 1, 1])
    }

    let routes = routes_of(num_tokens, top_k)?;
    wna16_axis("intermediate", intermediate, group_size)?;
    let launch = routed_qmv_transposed(routes, hidden.unsigned_abs());
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "quant/dequant_wna16.cuh",
            "::pie::quant::wna16_down_decode<::pie::i32(0)>",
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
    routine!(cast_fp32_to_bf16 = cast_fp32_to::<bf16>),
    routine!(scale_rows_bf16 = scale_rows::<bf16>),
    routine!(bf16_to_fp16),
    routine!(dequant_fp8_e4m3_to_bf16 = dequant_fp8_e4m3_to::<bf16>),
    routine!(dequant_fp8_e4m3_to_bf16_per_channel),
    routine!(dequant_fp8_e4m3_to_bf16_per_group),
    routine!(dequant_mxfp4_to_bf16 = dequant_mxfp4_to::<bf16>),
    routine!(dequant_wna16_int4b8_to_bf16 = dequant_wna16_int4b8_to::<bf16>),
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
pub static FAMILY: Family = crate::family!(ROUTINES);

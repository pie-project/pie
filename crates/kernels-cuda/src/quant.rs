use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use crate::jit::{Ctx, Launch};
use crate::views::ExpertWeights;
use kernels::Refusal;
use kernels::raises::Struct;
use kernels::routine::{Asks, Const, In, InOut, Out};
use kernels::{Bind, Fire};
use kernels_macros::routine;

use core::ffi::c_void;
use core::ptr::NonNull;

pub mod transcode {

    use crate::jit::abi::DevicePtr;

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct DecodeBf16 {
        pub src: DevicePtr,
        pub cols: i32,
    }

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct DecodeFp8E4m3PerGroup {
        pub src: DevicePtr,
        pub scales: DevicePtr,
        pub cols: i32,
        pub scale_cols: i32,
        pub group_size: i32,
    }

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct EncodeMxfp4 {
        pub packed: DevicePtr,
        pub scales: DevicePtr,
        pub cols: i32,
    }

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

    pub static LAYOUTS: &[crate::jit::Layout] = &[
        <DecodeBf16 as crate::jit::ByValue>::LAYOUT,
        <DecodeFp8E4m3PerGroup as crate::jit::ByValue>::LAYOUT,
        <EncodeMxfp4 as crate::jit::ByValue>::LAYOUT,
    ];
}

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

fn slab(n: u32) -> Launch {
    const SLAB_VEC: u32 = 8;

    const SLAB_GRID_MAX: u32 = 1024;

    let units = if n >= SLAB_VEC { n / SLAB_VEC } else { n };
    Launch::per_row(units.div_ceil(BLOCK).clamp(1, SLAB_GRID_MAX), BLOCK)
}

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

#[routine(bf16)]
pub fn cast_fp32_to<T>(
    ctx: &Ctx<'_>,
    src_fp32: In<Tensor<f32>>,
    dst_bf16: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let n = usize::try_from(dst_bf16.all("the cast's destination width")?.elements()).map_err(
        |_| Refusal::Empty {
            what: "the cast's element count",
        },
    )?;
    let launch = elementwise(extent("quant::cast_fp32_to_bf16", n));
    ctx.fire(
        Fire::at(
            "quant/dtype_cast.cuh",
            crate::jit::symbol(&format!("::pie::quant::cast_f32_to<{}>", T::CPP)),
        )
        .apply(launch),
        &[src_fp32.arg(), dst_bf16.arg(), n.arg()],
    )
}

#[routine(bf16, out(buf_bf16 = like(buf_bf16)))]
pub fn scale_rows<T>(
    ctx: &Ctx<'_>,
    buf_bf16: InOut<Tensor<T>>,
    l_bf16: In<Tensor<T>>,
) -> Result<(), Refusal> {
    let rows = buf_bf16.rows;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if buf_bf16.width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }

    let buf = buf_bf16.all("width")?;
    let launch = route_rows(rows.unsigned_abs(), buf.width.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dtype_cast.cuh",
            crate::jit::symbol(&format!("::pie::quant::scale_rows<{}>", T::CPP)),
        )
        .apply(launch),
        &[buf.ptr.arg(), l_bf16.arg(), buf.stride.arg()],
    )
}

#[routine]
pub fn bf16_to_fp16(
    ctx: &Ctx<'_>,
    in_bf16: In<Tensor<bf16>>,
    out_fp16: Out<Tensor<f16>>,
) -> Result<(), Refusal> {
    let count = i64::from(out_fp16.rows) * i64::from(out_fp16.width);
    if count <= 0 {
        return Err(Refusal::Empty {
            what: "the output rectangle",
        });
    }
    let Ok(grid) = u32::try_from(count) else {
        return Err(Refusal::Narrow {
            what: "the output rectangle in one 32-bit launch extent",
            at: count,
        });
    };
    let launch = slab(grid);
    ctx.fire(
        Fire::at(
            "quant/dequant_wna16.cuh",
            "::pie::quant::bf16_to_narrow<::pie::f16>",
        )
        .apply(launch),
        &[in_bf16.arg(), out_fp16.arg(), count.arg()],
    )
}

#[routine(bf16)]
pub fn dequant_fp8_e4m3_to<T>(
    ctx: &Ctx<'_>,
    fp8_in: In<Tensor<u8>>,
    bf16_out: Out<Tensor<T>>,
    scale: Const<f32>,
    rows: Const<i32>,
    cols: Const<i32>,
) -> Result<(), Refusal> {
    let n = (*rows as usize).saturating_mul(*cols as usize);
    let launch = elementwise(extent("quant::dequant_fp8_e4m3_to_bf16", n));
    ctx.fire(
        Fire::at(
            "quant/dequant_fp8.cuh",
            crate::jit::symbol(&format!("::pie::quant::dequant_fp8_e4m3<{}>", T::CPP)),
        )
        .apply(launch),
        &[fp8_in.arg(), bf16_out.arg(), scale.arg(), n.arg()],
    )
}

#[routine]
pub fn dequant_fp8_e4m3_to_bf16_per_channel(
    ctx: &Ctx<'_>,
    fp8_in: In<Tensor<u8>>,
    bf16_out: Out<Tensor<bf16>>,
    scale_inv: In<Tensor<f32>>,
    rows: Const<i32>,
    cols: Const<i32>,
) -> Result<(), Refusal> {
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dequant_fp8.cuh",
            "::pie::quant::dequant_fp8_e4m3_per_channel<::pie::bf16>",
        )
        .apply(launch),
        &[fp8_in.arg(), bf16_out.arg(), scale_inv.arg(), cols.arg()],
    )
}

#[routine]
pub fn dequant_fp8_e4m3_to_bf16_per_group(
    ctx: &Ctx<'_>,
    fp8_in: In<Tensor<u8>>,
    bf16_out: Out<Tensor<bf16>>,
    scales: In<Tensor<f32>>,
    group_size: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let rows = *rows;
    let cols = bf16_out.width;
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dequant_fp8.cuh",
            "::pie::quant::dequant_fp8_e4m3_per_group<::pie::bf16>",
        )
        .apply(launch),
        &[
            fp8_in.arg(),
            bf16_out.arg(),
            scales.arg(),
            cols.arg(),
            group_size.arg(),
        ],
    )
}

#[routine(bf16)]
pub fn dequant_mxfp4_to<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<u8>>,
    block_scale: In<Tensor<u8>>,
    out: Out<Tensor<T>>,
    out_dim: Const<i32>,
    in_dim: Const<i32>,
) -> Result<(), Refusal> {
    let launch = route_rows(out_dim.unsigned_abs(), in_dim.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dequant_fp4.cuh",
            crate::jit::symbol(&format!("::pie::quant::dequant_mxfp4<{}>", T::CPP)),
        )
        .apply(launch),
        &[packed.arg(), block_scale.arg(), out.arg(), in_dim.arg()],
    )
}

#[routine(bf16)]
pub fn dequant_wna16_int4b8_to<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<i32>>,
    scale: In<Tensor<T>>,
    out: Out<Tensor<T>>,
    group_size: Const<i32>,
    out_dim: Const<i32>,
    in_dim: Const<i32>,
) -> Result<(), Refusal> {
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
    ctx.fire(
        Fire::at(
            "quant/dequant_wna16.cuh",
            crate::jit::symbol(&format!("::pie::quant::dequant_wna16_int4b8<{}>", T::CPP)),
        )
        .apply(launch),
        &[
            packed.arg(),
            scale.arg(),
            out.arg(),
            in_dim.arg(),
            group_size.arg(),
        ],
    )
}

#[routine]
pub fn mxfp4_scales_to_marlin_e8m0(
    ctx: &Ctx<'_>,
    raw: In<Tensor<u8>>,
    out: Out<Tensor<u8>>,
    source_rows: Const<i32>,
    source_row_offset: Const<i32>,
    valid_rows: Const<i32>,
    source_stride_groups: Const<i32>,
    source_group_offset: Const<i32>,
    source_groups: Const<i32>,
    row_select: Const<i32>,
    selected_rows: Const<i32>,
) -> Result<(), Refusal> {
    let selected_rows = *selected_rows;
    let target_groups = out.width;
    let total = selected_rows
        .unsigned_abs()
        .saturating_mul(target_groups.unsigned_abs());
    let launch = elementwise(total);
    ctx.fire(
        Fire::at(
            "quant/mxfp4_marlin.cuh",
            "::pie::quant::mxfp4_scales_to_marlin_e8m0<::pie::u8>",
        )
        .apply(launch),
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

/// # Safety
///
/// `w_bf16` must address device memory of the extent `rows` and `cols`
/// state, `w_packed` and `w_scale_e8m0` must address the quantised extent this writes, and
/// `ctx`'s stream must be live. Nothing here reads a length: the shape
/// arrives as two integers and is believed.
pub unsafe fn quantize_bf16_to_mxfp4_e2m1_per_block(
    ctx: &Ctx<'_>,
    w_bf16: *const bf16,
    w_packed: *mut u8,
    w_scale_e8m0: *mut u8,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if cols < 32 {
        return Err(Refusal::Narrow {
            what: "cols, in 32-element blocks",
            at: i64::from(cols),
        });
    }
    let launch = route_rows(rows.unsigned_abs(), cols.unsigned_abs() / 32);
    ctx.fire(
        Fire::at(
            "quant/quant_bf16_to_mxfp4.cuh",
            "::pie::quant::quant_bf16_to_mxfp4_row<::pie::bf16>",
        )
        .apply(launch),
        &[w_bf16.arg(), w_packed.arg(), w_scale_e8m0.arg(), cols.arg()],
    )
}

/// # Safety
///
/// `w_bf16` must address device memory of the extent `rows` and `cols`
/// state, `w_fp8` and `scale_inv` must address the quantised extent this writes, and
/// `ctx`'s stream must be live. Nothing here reads a length: the shape
/// arrives as two integers and is believed.
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_channel(
    ctx: &Ctx<'_>,
    w_bf16: *const bf16,
    w_fp8: *mut u8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    let launch = rms(rows.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_per_channel<::pie::quant::fp8_e4m3>",
        )
        .apply(launch),
        &[w_bf16.arg(), w_fp8.arg(), scale_inv.arg(), cols.arg()],
    )
}

/// # Safety
///
/// `w_bf16` must address device memory of the extent `rows` and `cols`
/// state, `out_int8` and `scale_inv` must address the quantised extent this writes, and
/// `ctx`'s stream must be live. Nothing here reads a length: the shape
/// arrives as two integers and is believed.
pub unsafe fn quantize_bf16_to_int8_per_channel(
    ctx: &Ctx<'_>,
    w_bf16: *const bf16,
    out_int8: *mut i8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    let launch = rms(rows.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_per_channel<::pie::quant::int8_sym>",
        )
        .apply(launch),
        &[w_bf16.arg(), out_int8.arg(), scale_inv.arg(), cols.arg()],
    )
}

/// # Safety
///
/// `w_int8` and `scale_inv` must address device memory of the extent `rows` and `cols`
/// state, `out` must address the quantised extent this writes, and
/// `ctx`'s stream must be live. Nothing here reads a length: the shape
/// arrives as two integers and is believed.
pub unsafe fn dequant_int8_to_bf16_per_channel(
    ctx: &Ctx<'_>,
    w_int8: *const i8,
    out: *mut bf16,
    scale_inv: *const f32,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    let n = rows.unsigned_abs() as usize * cols.unsigned_abs() as usize;
    let launch = elementwise(extent("quant::dequant_int8_to_bf16_per_channel", n));
    ctx.fire(
        Fire::at(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::dequant_int8_per_channel<::pie::bf16>",
        )
        .apply(launch),
        &[
            w_int8.arg(),
            out.arg(),
            scale_inv.arg(),
            cols.arg(),
            n.arg(),
        ],
    )
}

/// # Safety
///
/// `acc` must address `m` by `n` accumulators, the two scale pointers the
/// per-row and per-column inverses that go with them, and `out` the same
/// rectangle in bf16. `ctx`'s stream must be live.
pub unsafe fn dequant_int32_w8a8_to_bf16(
    ctx: &Ctx<'_>,
    acc: *const i32,
    act_scale_inv: *const f32,
    w_scale_inv: *const f32,
    out: *mut bf16,
    m: i32,
    n: i32,
) -> Result<(), Refusal> {
    const W8A8_BY: u32 = 8;

    const W8A8_BX: u32 = 32;

    let launch = Launch::grid(
        [
            n.unsigned_abs().div_ceil(W8A8_BX),
            m.unsigned_abs().div_ceil(W8A8_BY),
            1,
        ],
        [W8A8_BX, W8A8_BY, 1],
    );
    ctx.fire(
        Fire::at("quant/quant_bf16_to_fp8.cuh", "::pie::quant::w8a8_dequant").apply(launch),
        &[
            acc.arg(),
            act_scale_inv.arg(),
            w_scale_inv.arg(),
            out.arg(),
            m.arg(),
            n.arg(),
        ],
    )
}

/// # Safety
///
/// `act_bf16` must address `m` by `k` activations, `act_fp8` the same
/// rectangle in fp8, and `act_scale` one float per group of `group_size`
/// along `k`. `ctx`'s stream must be live.
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_token_group(
    ctx: &Ctx<'_>,
    act_bf16: *const bf16,
    act_fp8: *mut u8,
    act_scale: *mut f32,
    m: i32,
    k: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    const GROUP_QUANT_BLOCK: u32 = 128;

    let n_groups = (k + group_size - 1) / group_size;
    let launch = Launch::grid(
        [n_groups.unsigned_abs(), m.unsigned_abs(), 1],
        [GROUP_QUANT_BLOCK, 1, 1],
    );
    ctx.fire(
        Fire::at(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_act_fp8_per_group",
        )
        .apply(launch),
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

const fn routed_qmv_quad(routes: u32, width: u32) -> Launch {
    const MXFP4_DECODE_BLOCK: u32 = 128;

    const MXFP4_ROWS_PER_WARP: u32 = 4;

    let tile = (MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP;
    Launch::grid(
        [routes, width.div_ceil(tile), 1],
        [MXFP4_DECODE_BLOCK, 1, 1],
    )
}

const fn routed_qmv(routes: u32, width: u32) -> Launch {
    Launch::grid([routes, width.div_ceil(BLOCK / WARP), 1], [BLOCK, 1, 1])
}

fn routes_of(num_tokens: i32, top_k: i32) -> Result<u32, Refusal> {
    if top_k <= 0 {
        return Err(Refusal::Empty {
            what: "the routed fanout",
        });
    }
    Ok(num_tokens
        .unsigned_abs()
        .saturating_mul(top_k.unsigned_abs()))
}

fn per_route(width: i32, top_k: i32) -> Result<i32, Refusal> {
    if top_k <= 0 {
        return Err(Refusal::Empty {
            what: "the routed fanout",
        });
    }
    if width <= 0 {
        return Err(Refusal::Empty {
            what: "the routed row",
        });
    }
    if width % top_k != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of routes",
            at: i64::from(width),
        });
    }
    Ok(width / top_k)
}

fn mxfp4_axis(what: &'static str, axis: i32) -> Result<(), Refusal> {
    if axis <= 0 {
        return Err(Refusal::Empty { what });
    }
    if axis % 32 != 0 {
        return Err(Refusal::Narrow {
            what,
            at: i64::from(axis),
        });
    }
    Ok(())
}

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
        return Err(Refusal::Narrow {
            what,
            at: i64::from(axis),
        });
    }
    Ok(())
}

#[routine]
pub fn mxfp4_moe_gate_up_decode_bf16(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    act: In<Tensor<f16>>,
    _packed_bank: Const<Tensor<u8>>,
    gate_out: Out<Tensor<bf16>>,
    up_out: Out<Tensor<bf16>>,
    glu_limit: Const<f32>,
    glu_alpha: Const<f32>,
    ew: In<Struct<ExpertWeights>>,
) -> Result<(), Refusal> {
    if ew.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the expert-weight view this statement names",
        });
    }
    let ew = unsafe { &*ew.ptr };

    let packed_ptrs = ew.ptrs;
    let glu_limit = *glu_limit;
    let glu_alpha = *glu_alpha;

    let scale_ptrs = ew.scale_ptrs;
    let gate_bias_ptrs = ctx.absent()?;
    let up_bias_ptrs = ctx.absent()?;
    let top_k = topk_idx.width;
    let intermediate = per_route(gate_out.width, top_k)?;
    let routes = routes_of(topk_idx.rows, top_k)?;
    mxfp4_axis("hidden", act.width)?;

    let x = act.all("hidden")?;
    let launch = routed_qmv_quad(routes, intermediate.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dequant_fp4.cuh",
            "::pie::quant::mxfp4_moe_gate_up_decode<::pie::i32(4)>",
        )
        .apply(launch),
        &[
            x.ptr.arg(),
            topk_idx.arg(),
            packed_ptrs.arg(),
            scale_ptrs.arg(),
            gate_bias_ptrs,
            up_bias_ptrs,
            gate_out.arg(),
            up_out.arg(),
            Option::<NonNull<f16>>::None.arg(),
            glu_limit.arg(),
            glu_alpha.arg(),
            top_k.arg(),
            x.stride.arg(),
            intermediate.arg(),
        ],
    )
}

#[routine]
pub fn mxfp4_moe_down_decode_bf16(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    act: In<Tensor<f16>>,
    _packed_bank: Const<Tensor<u8>>,
    out: Out<Tensor<bf16>>,
    ew: In<Struct<ExpertWeights>>,
) -> Result<(), Refusal> {
    if ew.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the expert-weight view this statement names",
        });
    }
    let ew = unsafe { &*ew.ptr };

    let packed_ptrs = ew.ptrs;

    let scale_ptrs = ew.scale_ptrs;
    // The ARRAY, not the plane: the kernel's first act on it is
    // `bias_ptrs[expert]`. Reading the `_bias` plane's own base here read
    // eight bytes of bf16 bias data as an address — CUDA 700 at gpt-oss's
    // first routed layer, sticky. The view's field is the `_bias_ptrs`
    // carve, which is the fix carried on the operand channel.
    let bias_ptrs = ew.bias_ptrs;

    let top_k = topk_idx.width;
    let hidden = per_route(out.width, top_k)?;
    let intermediate = per_route(act.width, top_k)?;
    let routes = routes_of(topk_idx.rows, top_k)?;
    mxfp4_axis("intermediate", intermediate)?;

    let launch = routed_qmv_quad(routes, hidden.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dequant_fp4.cuh",
            "::pie::quant::mxfp4_moe_down_decode<::pie::i32(4)>",
        )
        .apply(launch),
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

#[routine]
pub fn wna16_gate_up_decode_bf16(
    ctx: &Ctx<'_>,
    act: In<Tensor<f16>>,
    topk_idx: In<Tensor<i32>>,
    gate_packed_ptrs: Const<Tensor<i32>>,
    gate_scale_ptrs: Const<Tensor<c_void>>,
    up_packed_ptrs: Const<Tensor<i32>>,
    up_scale_ptrs: Const<Tensor<c_void>>,
    gate_out: Out<Tensor<bf16>>,
    up_out: Out<Tensor<bf16>>,
    group_size: Const<i32>,
) -> Result<(), Refusal> {
    let group_size = *group_size;
    let top_k = topk_idx.width;
    let routes = routes_of(topk_idx.rows, top_k)?;
    wna16_axis("hidden", act.width, group_size)?;

    if gate_out.width <= 0 {
        return Err(Refusal::Empty {
            what: "the routed row",
        });
    }

    let x = act.all("hidden")?;
    let gate = gate_out.all("the routed row")?;
    let launch = routed_qmv(routes, gate.width.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dequant_wna16.cuh",
            "::pie::quant::wna16_gate_up_decode<::pie::i32(0)>",
        )
        .apply(launch),
        &[
            x.ptr.arg(),
            topk_idx.arg(),
            gate_packed_ptrs.arg(),
            gate_scale_ptrs.arg(),
            up_packed_ptrs.arg(),
            up_scale_ptrs.arg(),
            gate.ptr.arg(),
            up_out.arg(),
            top_k.arg(),
            x.stride.arg(),
            gate.stride.arg(),
            group_size.arg(),
        ],
    )
}

#[routine]
pub fn wna16_down_decode_bf16(
    ctx: &Ctx<'_>,
    act: In<Tensor<f16>>,
    topk_idx: In<Tensor<i32>>,
    down_packed_ptrs: Const<Tensor<i32>>,
    down_scale_ptrs: Const<Tensor<c_void>>,
    out: Out<Tensor<bf16>>,
    group_size: Const<i32>,
) -> Result<(), Refusal> {
    let group_size = *group_size;
    const fn routed_qmv_transposed(routes: u32, width: u32) -> Launch {
        Launch::grid([width.div_ceil(BLOCK / WARP), routes, 1], [BLOCK, 1, 1])
    }

    let top_k = topk_idx.width;
    let routes = routes_of(topk_idx.rows, top_k)?;
    wna16_axis("intermediate", act.width, group_size)?;

    if out.width <= 0 {
        return Err(Refusal::Empty {
            what: "the routed row",
        });
    }

    let x = act.all("intermediate")?;
    let y = out.all("the routed row")?;
    let launch = routed_qmv_transposed(routes, y.width.unsigned_abs());
    ctx.fire(
        Fire::at(
            "quant/dequant_wna16.cuh",
            "::pie::quant::wna16_down_decode<::pie::i32(0)>",
        )
        .apply(launch),
        &[
            x.ptr.arg(),
            topk_idx.arg(),
            down_packed_ptrs.arg(),
            down_scale_ptrs.arg(),
            y.ptr.arg(),
            top_k.arg(),
            y.stride.arg(),
            x.stride.arg(),
            group_size.arg(),
        ],
    )
}

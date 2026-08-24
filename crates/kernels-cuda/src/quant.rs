use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::routine::{Const, In, InOut, Out};
use kernels::{Bind, Fire};


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

pub fn cast_fp32_to<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    src_fp32: In<Tensor<f32>>,
    dst_bf16: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    let n = usize::try_from(dst_bf16.all("the cast's destination width")?.elements()).map_err(
        |_| Refusal::Empty {
            what: "the cast's element count",
        },
    )?;
    let launch = elementwise(extent("quant::cast_fp32_to", n));
    ctx.fire(
        Fire::at(
            "quant/dtype_cast.cuh",
            crate::jit::symbol(&format!("::pie::quant::cast_f32_to<{}>", T::CPP)),
        )
        .apply(launch),
        &[src_fp32.arg(), dst_bf16.arg(), n.arg()],
    )
}

pub fn scale_rows<T: crate::RoutineElem>(
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

pub fn dequant_fp8_e4m3_to<T: crate::RoutineElem>(
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

pub fn dequant_mxfp4_to<T: crate::RoutineElem>(
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

use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::plane::{In, InOut, Out};
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
        probe = "nvrtc-probes/quant_transcode.py",
        size = 16, align = 8,
        {
            src  @ 0 as "src",
            cols @ 8 as "cols",
        }
    }

    crate::by_value! {
        DecodeFp8E4m3PerGroup as "::pie::transcode::DecodeFp8E4m3PerGroup",
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

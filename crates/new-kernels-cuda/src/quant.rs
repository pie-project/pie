//! `quant`: dtype casts and the weight quantizers the loader drives. No IR
//! variant lands here directly; the driver and loader call these entries
//! while staging checkpoints and recurrent state.
//!
//! The jit-stamped transcode descriptors (`DecodeBf16`,
//! `DecodeFp8E4m3PerGroup`, `EncodeMxfp4`) are by-value aggregates; they
//! arrive with the attn wave, together with the `by_value!` abi machinery
//! that measures their layouts. The moe wave settled its half of that
//! deferral without them: an mxfp4 bank travels as the explicit
//! `(codes, scales)` pair and marshals as two plain pointers
//! (`moe::matmul_select_bias`), so no descriptor is needed on the routed
//! path.

use new_kernels::KernelError;
use new_model_ir::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const CAST_FILE: &str = "quant/dtype_cast.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

/// One block per row, warp-shuffle reduction scratch beside it.
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * 4)
}

/// One block per row, sized to the row in whole warps.
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

/// A 32-bit launch extent, refused rather than truncated — a clamped grid
/// would launch over the low 32 bits and leave the rest of the destination
/// unwritten.
fn extent(op: &'static str, n: u64) -> Result<u32, KernelError> {
    u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })
}

pub fn cast_fp32_to(ctx: &Ctx, src: Tensor, dst: &mut Tensor) -> Result<(), KernelError> {
    const OP: &str = "quant.cast_fp32_to";
    debug_assert_eq!(src.dtype, Dtype::F32, "`{OP}` casts an f32 source");
    let t = dtype_dispatch!(OP, dst.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let n = dst.elements();
    let lanes = nonzero(OP, "the cast's element count", extent(OP, n)?)?;
    ctx.fire(
        OP,
        Fire::at(
            CAST_FILE,
            symbol(&format!("::pie::quant::cast_f32_to<{t}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[src.arg(), dst.arg(), n.arg()],
    )
}

/// Scales each row of `buf` by the matching row of `l`, in place.
pub fn scale_rows(ctx: &Ctx, l: Tensor, buf: &mut Tensor) -> Result<(), KernelError> {
    const OP: &str = "quant.scale_rows";
    let t = dtype_dispatch!(OP, buf.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", buf.rows)?;
    let width = stated(OP, nonzero(OP, "width", buf.width)?)?;
    ctx.fire(
        OP,
        Fire::at(CAST_FILE, symbol(&format!("::pie::quant::scale_rows<{t}>")))
            .apply(route_rows(buf.rows, buf.width)),
        &[buf.arg(), l.arg(), width.arg()],
    )
}

/// Packs a bf16 weight into mxfp4: e2m1 codes in 32-element blocks with one
/// shared e8m0 scale each.
pub fn quantize_bf16_to_mxfp4_e2m1_per_block(
    ctx: &Ctx,
    w: Tensor,
    packed: &mut Tensor,
    scales: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "quant.bf16_to_mxfp4";
    dtype_dispatch!(OP, w.dtype, { Bf16 => () });
    if w.width < 32 {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide row does not hold one 32-element block",
                w.width
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(
            "quant/quant_bf16_to_mxfp4.cuh",
            "::pie::quant::quant_bf16_to_mxfp4_row<::pie::bf16>",
        )
        .apply(route_rows(w.rows, w.width / 32)),
        &[
            w.arg(),
            packed.arg(),
            scales.arg(),
            stated(OP, w.width)?.arg(),
        ],
    )
}

/// Quantizes a bf16 weight to fp8 e4m3 with one inverse scale per channel.
pub fn quantize_bf16_to_fp8_e4m3_per_channel(
    ctx: &Ctx,
    w: Tensor,
    fp8: &mut Tensor,
    scale_inv: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "quant.bf16_to_fp8";
    dtype_dispatch!(OP, w.dtype, { Bf16 => () });
    debug_assert_eq!(
        scale_inv.dtype,
        Dtype::F32,
        "`{OP}` writes f32 inverse scales"
    );
    nonzero(OP, "rows", w.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            "quant/quant_bf16_to_fp8.cuh",
            "::pie::quant::quant_per_channel<::pie::quant::fp8_e4m3>",
        )
        .apply(rms(w.rows)),
        &[
            w.arg(),
            fp8.arg(),
            scale_inv.arg(),
            stated(OP, w.width)?.arg(),
        ],
    )
}

//! `quant`: dtype casts and the weight quantizers the loader drives. No IR
//! variant lands here directly; the engine and loader call these entries
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

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/quant.cuh";

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
fn extent(op: &'static str, n: u64) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })
}

/// `linear.matmul` over a weight the store seats as an MLX affine triplet —
/// the dense twin of `moe_matmul_select_quant`, and this shell's spelling of
/// `kernels_metal::linear::quant::matmul`. The bit width is the codes
/// plane's own: a `[n, k]` weight stores `k` bytes a row at eight bits and
/// `k / 2` at four, so the rectangle already says which.
pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(ctx, "linear.matmul", act, codes, scales, biases, y, seat)
}

/// [`matmul`] under the head's own op name, `linear::gemm`'s pairing kept.
pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(ctx, "linear.lm_head", act, codes, scales, biases, y, seat)
}

/// The one launch behind both dense entries. `Σ (c·s + b)·x` folds as
/// `s·Σ c·x + b·Σ x` (the select kernel's identity), sixty-four codes to a
/// factor pair; a fire with no rows is the same silent no-op the dense gemm
/// keeps, and for the same capture reason.
#[allow(clippy::too_many_arguments)]
fn dense_affine(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    const MLX_GROUP: u32 = 64;
    const ROWS_PER_WARP: u32 = 4;
    const BLOCK_LANES: u32 = 128;

    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    let Some(biases) = biases else {
        return Err(refuse(
            op,
            "an mxfp4 dense projection has no gemm point on this plane; the \
             affine triplet is the packed landing this entry reads",
        ));
    };
    debug_assert_eq!(biases.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    if !k.is_multiple_of(MLX_GROUP) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {MLX_GROUP}-code affine groups"),
        ));
    }
    // The factor plane is `[n, k / group]` bf16 bound as its byte rectangle,
    // and the kernel walks groups of sixty-four; a plane grouped otherwise
    // is refused rather than mis-scaled.
    if scales.width != (k / MLX_GROUP) * 2 {
        return Err(refuse(
            op,
            format!(
                "a {}-byte factor row does not group a {k}-wide row by {MLX_GROUP}",
                scales.width
            ),
        ));
    }
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else {
        return Err(refuse(
            op,
            format!("a {}-byte code row stores a {k}-wide row at neither four nor eight bits", codes.width),
        ));
    };
    if y.rows == 0 {
        return Ok(());
    }
    let tile = (BLOCK_LANES / WARP) * ROWS_PER_WARP;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_mlx_affine<{t}, ::pie::i32({bits}), \
                 ::pie::i32({ROWS_PER_WARP})>"
            )),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [BLOCK_LANES, 1, 1],
        )),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ArgValue::Ptr(seat.cell),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

pub fn cast_fp32_to(ctx: &Ctx, src: Tensor, dst: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_cast_fp32_to";
    debug_assert_eq!(src.dtype, Dtype::F32, "`{OP}` casts an f32 source");
    let t = dtype_dispatch!(OP, dst.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let n = dst.elements();
    let lanes = nonzero(OP, "the cast's element count", extent(OP, n)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::cast_f32_to<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[src.arg(), dst.arg(), n.arg()],
    )
}

/// Scales each row of `buf` by the matching row of `l`, in place.
pub fn scale_rows(ctx: &Ctx, l: Tensor, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_scale_rows";
    let t = dtype_dispatch!(OP, buf.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", buf.rows)?;
    let width = stated(OP, nonzero(OP, "width", buf.width)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::scale_rows<{t}>")))
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
) -> Result<(), Error> {
    const OP: &str = "linear.quant_bf16_to_mxfp4";
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
        Fire::at(FILE, "::pie::linear::quant_bf16_to_mxfp4_row<::pie::bf16>")
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
) -> Result<(), Error> {
    const OP: &str = "linear.quant_bf16_to_fp8";
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
            FILE,
            "::pie::linear::quant_per_channel<::pie::linear::fp8_e4m3>",
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

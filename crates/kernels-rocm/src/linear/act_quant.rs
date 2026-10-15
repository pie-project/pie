//! Activations quantized for a low-precision product, and the FP8
//! projections (`fp8b128`, DeepSeek's W8A8 dense weights) they multiply:
//! `kernels/linear/act_quant.hip`. The quantized activations land as their
//! exact bf16 values, so the product is a bf16 one over the same values the
//! W8A8 GEMM multiplies, summed in f32.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::linear::gemm;
use crate::linear::quant::DECODED_WEIGHT;
use crate::tensor::Tensor;

const FILE: &str = "linear/act_quant.hip";

const BLOCK: u32 = 256;

const WAVE: u32 = 64;

/// The fp8b128 block side.
pub const FP8_BLOCK: u32 = 128;

/// Rows a fused-dequant GEMV pass takes; at most this many rows take the
/// GEMV rather than the decoded tile and the GEMM.
pub const GEMV_ROWS: u32 = 16;

/// The K granules the quantizer is built for.
const GRANS: [u32; 4] = [32, 64, 128, 256];

/// `x`'s rows quantized as e4m3 under a ue8m0 scale a `gran`, landed in `y`
/// as their bf16 values.
pub fn e4m3_values(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    gran: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    if !GRANS.contains(&gran) {
        return Err(refuse(
            op,
            format!("e4m3 activations in {gran}-wide blocks are not one of {GRANS:?}"),
        ));
    }
    if x.width == 0 || !x.width.is_multiple_of(gran) {
        return Err(refuse(
            op,
            format!("a {}-wide row is no whole number of {gran}-wide blocks", x.width),
        ));
    }
    if y.dtype != Dtype::Bf16 || y.rows != x.rows || y.width != x.width {
        return Err(refuse(op, "the quantized rows are not the bf16 rectangle of x"));
    }
    if x.rows == 0 {
        return Ok(());
    }
    let blocks = u64::from(x.rows) * u64::from(x.width / gran);
    let per_wave = u64::from((WAVE / gran).max(1));
    let waves = blocks.div_ceil(per_wave);
    let threads = u32::try_from(waves * u64::from(WAVE))
        .map_err(|_| refuse(op, "the quantizer's threads do not fit a 32-bit extent"))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::act_quant_e4m3_values<{gran}>")),
        )
        .apply(Launch::flat(threads, BLOCK)),
        &[
            x.arg(),
            y.arg(),
            stated(op, x.rows)?.arg(),
            stated(op, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

/// `x` quantized as `e4m3_values` does, into this fire's scratch.
pub fn quantized(ctx: &Ctx, op: &'static str, x: Tensor, gran: u32) -> Result<Tensor, Error> {
    let bytes = (x.rows as usize)
        .saturating_mul(x.width as usize)
        .saturating_mul(2);
    let at = ctx.scratch(op, "linear.act_quant_values", bytes.max(16))? as usize as u64;
    let mut y = Tensor::new(at, x.rows, x.width, Dtype::Bf16);
    e4m3_values(ctx, op, x, gran, &mut y)?;
    Ok(y)
}

/// That `codes` (e4m3, `[n, k]`) and `scales` (e8m0, one a 128x128 block)
/// are one fp8b128 projection.
fn fp8b128_planes(op: &'static str, codes: Tensor, scales: Tensor) -> Result<(u32, u32), Error> {
    let (n, k) = (codes.rows, codes.width);
    if !matches!(codes.dtype, Dtype::E4m3 | Dtype::U8) || !matches!(scales.dtype, Dtype::E8m0 | Dtype::U8) {
        return Err(refuse(
            op,
            format!(
                "an fp8b128 projection is e4m3 codes beside e8m0 exponents, not {:?} beside {:?}",
                codes.dtype, scales.dtype
            ),
        ));
    }
    if scales.rows != n.div_ceil(FP8_BLOCK) || scales.width != k.div_ceil(FP8_BLOCK) {
        return Err(refuse(
            op,
            format!(
                "a [{n}, {k}] fp8b128 projection's exponents are [{}, {}], not [{}, {}]",
                scales.rows,
                scales.width,
                n.div_ceil(FP8_BLOCK),
                k.div_ceil(FP8_BLOCK)
            ),
        ));
    }
    if n == 0 || k == 0 || !k.is_multiple_of(16) || codes.ptr % 16 != 0 {
        return Err(refuse(
            op,
            format!("a [{n}, {k}] e4m3 plane is not whole 16-byte runs a row"),
        ));
    }
    Ok((n, k))
}

/// An fp8b128 projection's values in bf16 at `dst`, `[n, k]`.
pub fn fp8b128_decode_into(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    dst: u64,
) -> Result<Tensor, Error> {
    let (n, k) = fp8b128_planes(op, codes, scales)?;
    let threads = u32::try_from((u64::from(n) * u64::from(k)).div_ceil(8))
        .map_err(|_| refuse(op, "the decode's threads do not fit a 32-bit extent"))?;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::linear::fp8b128_decode").apply(Launch::flat(threads, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            ArgValue::Ptr(dst),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(dst, n, k, Dtype::Bf16))
}

/// `y = act · wᵀ` for an fp8b128 `w` against e4m3 activations in `gran`-wide
/// blocks (DeepSeek's W8A8). Few rows, or a plane past the stream's decoded
/// tile (`tile_holds` false), take the fused-dequant GEMV; the rest decode
/// the plane into the tile and take the bf16 GEMM.
#[allow(clippy::too_many_arguments)]
pub fn matmul_fp8b128(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    gran: u32,
    tile_holds: bool,
    y: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let (n, k) = fp8b128_planes(op, codes, scales)?;
    if act.width != k || y.width != n || y.rows != act.rows || y.dtype != Dtype::Bf16 {
        return Err(refuse(
            op,
            format!(
                "a [{}, {}] act against a [{n}, {k}] projection does not land [{}, {}]",
                act.rows, act.width, y.rows, y.width
            ),
        ));
    }
    if act.rows == 0 {
        return Ok(());
    }
    let q = quantized(ctx, op, act, gran)?;
    if act.rows <= GEMV_ROWS || !tile_holds {
        let threads = n
            .checked_mul(WAVE)
            .ok_or_else(|| refuse(op, "the GEMV's threads do not fit a 32-bit extent"))?;
        nonzero(op, "the projection's rows", n)?;
        return ctx.fire(
            op,
            Fire::at(FILE, symbol(&format!("::pie::linear::fp8b128_gemv<{}>", 4)))
                .apply(Launch::flat(threads, BLOCK)),
            &[
                q.arg(),
                codes.arg(),
                scales.arg(),
                y.arg(),
                stated(op, act.rows)?.arg(),
                stated(op, n)?.arg(),
                stated(op, k)?.arg(),
                ctx.stage(),
            ],
        );
    }
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch_stream(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let w = fp8b128_decode_into(ctx, op, codes, scales, tile)?;
    gemm::act_x_wt(ctx, op, q, w, None, y)
}

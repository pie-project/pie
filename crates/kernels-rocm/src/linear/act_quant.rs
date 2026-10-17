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

/// `x` quantized as `e4m3_values` does, into this fire's scratch: as many
/// rows as a product reading it launches over (`Ctx::opaque_rows`, the
/// bucket a forward's rows pad to), so a GEMM over the padded rows reads
/// memory this slab holds.
pub fn quantized(ctx: &Ctx, op: &'static str, x: Tensor, gran: u32) -> Result<Tensor, Error> {
    let rows = usize::try_from(ctx.opaque_rows(i32::try_from(x.rows).unwrap_or(i32::MAX))).unwrap_or(x.rows as usize);
    let bytes = rows.max(x.rows as usize)
        .saturating_mul(x.width as usize)
        .saturating_mul(2);
    let at = ctx.scratch(op, "linear.act_quant_values", bytes.max(16))? as usize as u64;
    let mut y = Tensor::new(at, x.rows, x.width, Dtype::Bf16);
    e4m3_values(ctx, op, x, gran, &mut y)?;
    Ok(y)
}

/// That `codes` (e4m3, `[n, k]`) and `scales` (e8m0, one a 128x128 block,
/// or one a 32x32 block: `fp8b32`) are one block-FP8 projection; with its
/// block side.
fn fp8b128_planes(op: &'static str, codes: Tensor, scales: Tensor) -> Result<(u32, u32, u32), Error> {
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
    let Some(side) = [FP8_BLOCK, 32]
        .into_iter()
        .find(|&b| scales.rows == n.div_ceil(b) && scales.width == k.div_ceil(b))
    else {
        return Err(refuse(
            op,
            format!(
                "a [{n}, {k}] block-FP8 projection's exponents are [{}, {}], one a 128x128 or a 32x32 block",
                scales.rows, scales.width,
            ),
        ));
    };
    if n == 0 || k == 0 || !k.is_multiple_of(16) || codes.ptr % 16 != 0 {
        return Err(refuse(
            op,
            format!("a [{n}, {k}] e4m3 plane is not whole 16-byte runs a row"),
        ));
    }
    Ok((n, k, side))
}

/// An fp8b128 projection's values in bf16 at `dst`, `[n, k]`.
pub fn fp8b128_decode_into(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    dst: u64,
) -> Result<Tensor, Error> {
    let (n, k, side) = fp8b128_planes(op, codes, scales)?;
    let threads = u32::try_from((u64::from(n) * u64::from(k)).div_ceil(8))
        .map_err(|_| refuse(op, "the decode's threads do not fit a 32-bit extent"))?;
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::linear::fp8b128_decode<{side}>"))).apply(Launch::flat(threads, BLOCK)),
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
    let (n, k, side) = fp8b128_planes(op, codes, scales)?;
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
            Fire::at(FILE, symbol(&format!("::pie::linear::fp8b128_gemv<{}, {side}>", 4)))
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

/// That `codes` (e4m3, `[n, k]`) and `scales` (f32, one a 128x128 block) are
/// one `e4m3tile128` projection.
fn e4m3tile128_planes(op: &'static str, codes: Tensor, scales: Tensor) -> Result<(u32, u32), Error> {
    let (n, k) = (codes.rows, codes.width);
    if !matches!(codes.dtype, Dtype::E4m3 | Dtype::U8) || scales.dtype != Dtype::F32 {
        return Err(refuse(
            op,
            format!(
                "an e4m3tile128 projection is e4m3 codes beside f32 factors, not {:?} beside {:?}",
                codes.dtype, scales.dtype
            ),
        ));
    }
    if scales.rows != n.div_ceil(FP8_BLOCK) || scales.width != k.div_ceil(FP8_BLOCK) {
        return Err(refuse(
            op,
            format!(
                "a [{n}, {k}] e4m3tile128 projection's factors are [{}, {}], not [{}, {}]",
                scales.rows,
                scales.width,
                n.div_ceil(FP8_BLOCK),
                k.div_ceil(FP8_BLOCK)
            ),
        ));
    }
    if n == 0 || k == 0 || !k.is_multiple_of(FP8_BLOCK) || !n.is_multiple_of(16) || !codes.ptr.is_multiple_of(16) {
        return Err(refuse(
            op,
            format!(
                "a [{n}, {k}] e4m3tile128 projection contracts whole 128-code blocks over whole 16-row strips \
                 (the matrix core's order)"
            ),
        ));
    }
    Ok((n, k))
}

/// `x`'s rows quantized as e4m3 codes under an f32 scale a `gran` (`amax /
/// 448`): the OCP bytes (`[rows, k]`, negative zero as zero) and each
/// block's scale (`[rows, k / gran]`), in this fire's scratch, as the FP8
/// matrix-core products read them (`linear/fp8.hip`).
pub fn quantized_codes(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    gran: u32,
) -> Result<(Tensor, Tensor), Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    if gran != FP8_BLOCK || x.width == 0 || !x.width.is_multiple_of(gran) {
        return Err(refuse(
            op,
            format!("the FP8 matrix core reads e4m3 in 128-wide blocks of whole rows, not {gran} over {}", x.width),
        ));
    }
    let blocks = x.width / gran;
    // One slab a stream: the product right after consumes them.
    let at = ctx.scratch_stream(op, "linear.act_quant_codes", (x.rows as usize * x.width as usize).max(16))? as usize as u64;
    let sat =
        ctx.scratch_stream(op, "linear.act_quant_scales", (x.rows as usize * blocks as usize * 4).max(16))? as usize as u64;
    let codes = Tensor::new(at, x.rows, x.width, Dtype::U8);
    let scales = Tensor::new(sat, x.rows, blocks, Dtype::F32);
    quantize_codes_into(ctx, op, x, gran, codes, scales)?;
    Ok((codes, scales))
}

fn quantize_codes_into(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    gran: u32,
    codes: Tensor,
    scales: Tensor,
) -> Result<(), Error> {
    let blocks = x.width / gran;
    if x.rows == 0 {
        return Ok(());
    }
    let threads = u32::try_from(u64::from(x.rows) * u64::from(blocks) * u64::from(WAVE))
        .map_err(|_| refuse(op, "the quantizer's threads do not fit a 32-bit extent"))?;
    ctx.fire(
        op,
        Fire::at("linear/fp8.hip", symbol(&format!("::pie::linear::act_quant_e4m3_codes<{gran}>")))
            .apply(Launch::flat(threads, BLOCK)),
        &[
            x.arg(),
            codes.arg(),
            scales.arg(),
            stated(op, x.rows)?.arg(),
            stated(op, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

/// The K a stage of the FP8 matrix-core tile takes: one 128-code block
/// (two measured no faster at GLM-5.3's shapes, their LDS holding a CU to
/// one block).
#[must_use]
pub const fn fp8_stage(_k: u32, _mt: u32) -> u32 {
    128
}

/// The item rows the FP8 matrix-core tile takes for `rows`.
#[must_use]
pub const fn fp8_item_rows(rows: u32) -> u32 {
    match rows {
        0..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// `y = act · wᵀ` for an `e4m3tile128` `w` (`codes` e4m3 `[n, k]` made
/// canonical at load, `factors` f32 a 128x128 block) against `act` (bf16)
/// quantized as e4m3 under f32 scales a 1x128 block (`amax / 448`), on the
/// FP8 matrix core (the rows quantized by `quantized_codes` first): each
/// block's products summed in f32, then folded under the weight's factor
/// and the row's scale. Up to 64 rows the decode tile, past them the
/// prefill tile, 128 rows against 128 of the weight. K is sliced while the
/// tiles are few, each slice's partial summed after.
#[allow(clippy::too_many_arguments)]
pub fn matmul_fp8(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    factors: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    if act.rows == 0 {
        return Ok(());
    }
    let (q, act_scales) = quantized_codes(ctx, op, act, FP8_BLOCK)?;
    matmul_fp8_quantized(ctx, op, q, act_scales, codes, factors, y)
}

/// The holds a region keeps for quantized rows (`quantized_codes_held`).
pub const FP8_HOLDS: usize = 8;

const HELD_CODES: [&str; FP8_HOLDS] = [
    "linear.act_quant_codes_held.0",
    "linear.act_quant_codes_held.1",
    "linear.act_quant_codes_held.2",
    "linear.act_quant_codes_held.3",
    "linear.act_quant_codes_held.4",
    "linear.act_quant_codes_held.5",
    "linear.act_quant_codes_held.6",
    "linear.act_quant_codes_held.7",
];
const HELD_SCALES: [&str; FP8_HOLDS] = [
    "linear.act_quant_scales_held.0",
    "linear.act_quant_scales_held.1",
    "linear.act_quant_scales_held.2",
    "linear.act_quant_scales_held.3",
    "linear.act_quant_scales_held.4",
    "linear.act_quant_scales_held.5",
    "linear.act_quant_scales_held.6",
    "linear.act_quant_scales_held.7",
];

/// The rows quantized as the FP8 products read them (`quantized_codes`),
/// held for later products of the same rows in this walk: a decode batch's
/// (64 rows or fewer) in the region's hold `hold` (of `FP8_HOLDS`), which
/// only the region's next quantizing into that hold overwrites; past them
/// in the stream's slab, as `quantized_codes`.
pub fn quantized_codes_held(ctx: &Ctx, op: &'static str, x: Tensor, hold: usize) -> Result<(Tensor, Tensor), Error> {
    if x.rows > 64 {
        return quantized_codes(ctx, op, x, FP8_BLOCK);
    }
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    if x.width == 0 || !x.width.is_multiple_of(FP8_BLOCK) {
        return Err(refuse(op, format!("the FP8 matrix core reads e4m3 in 128-wide blocks of whole rows, not {}", x.width)));
    }
    let blocks = x.width / FP8_BLOCK;
    let hold = hold % FP8_HOLDS;
    let at = ctx.scratch(op, HELD_CODES[hold], (x.rows as usize * x.width as usize).max(16))? as usize as u64;
    let sat = ctx.scratch(op, HELD_SCALES[hold], (x.rows as usize * blocks as usize * 4).max(16))? as usize as u64;
    let codes = Tensor::new(at, x.rows, x.width, Dtype::U8);
    let scales = Tensor::new(sat, x.rows, blocks, Dtype::F32);
    quantize_codes_into(ctx, op, x, FP8_BLOCK, codes, scales)?;
    Ok((codes, scales))
}

/// `y = act · wᵀ` as `matmul_fp8`, the rows already quantized (`act` their
/// codes, `act_scales` their scales).
#[allow(clippy::too_many_arguments)]
pub fn matmul_fp8_quantized(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    act_scales: Tensor,
    codes: Tensor,
    factors: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let act = q;
    let (n, k) = e4m3tile128_planes(op, codes, factors)?;
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
    if act.dtype != Dtype::U8 || act_scales.dtype != Dtype::F32 || act_scales.rows != act.rows || act_scales.width != k / FP8_BLOCK {
        return Err(refuse(op, "the FP8 product reads e4m3 row codes beside f32 scales a 128-wide block"));
    }
    let stages = k / FP8_BLOCK;
    let (splits, part) = if act.rows > 64 {
        // The prefill tile; K sliced while the tiles are fewer than 128
        // (measured at GLM-5.3's shapes: past that, slicing costs more than
        // the blocks win).
        let (items, tiles) = (act.rows.div_ceil(128), n.div_ceil(128));
        let mut splits = 1u32;
        while splits < 8 && tiles * items * splits < 128 && stages / (2 * splits) >= 4 {
            splits *= 2;
        }
        let part = fp8_partials(ctx, op, splits, act.rows, n)?;
        ctx.fire(
            op,
            Fire::at("linear/fp8.hip", "::pie::linear::dense_fp8_gemm8")
                .apply(Launch::grid([items, tiles, splits], [2 * BLOCK, 1, 1])),
            &[
                q.arg(),
                act_scales.arg(),
                codes.arg(),
                factors.arg(),
                y.arg(),
                part,
                stated(op, act.rows)?.arg(),
                stated(op, n)?.arg(),
                stated(op, k)?.arg(),
                stated(op, stages.div_ceil(splits))?.arg(),
                ctx.stage(),
            ],
        )?;
        (splits, part)
    } else {
        // The decode tile; K sliced while the tiles and items are short of
        // two blocks a CU.
        let mt = fp8_item_rows(act.rows);
        let nb = fp8_strips();
        let (items, tiles) = (act.rows.div_ceil(mt), n.div_ceil(64 * nb));
        let mut splits = 1u32;
        while splits < 8 && tiles * items * splits < 608 && stages / (2 * splits) >= 4 {
            splits *= 2;
        }
        let part = fp8_partials(ctx, op, splits, act.rows, n)?;
        ctx.fire(
            op,
            Fire::at("linear/fp8.hip", symbol(&format!("::pie::linear::dense_fp8_mfma<::pie::i32({mt}), ::pie::i32({nb})>")))
                .apply(Launch::grid([tiles.div_ceil(8) * 8 * items * splits, 1, 1], [BLOCK, 1, 1])),
            &[
                q.arg(),
                act_scales.arg(),
                codes.arg(),
                factors.arg(),
                y.arg(),
                part,
                stated(op, act.rows)?.arg(),
                stated(op, n)?.arg(),
                stated(op, k)?.arg(),
                stated(op, items)?.arg(),
                stated(op, tiles)?.arg(),
                stated(op, splits)?.arg(),
                stated(op, stages.div_ceil(splits))?.arg(),
                ctx.stage(),
            ],
        )?;
        (splits, part)
    };
    if splits == 1 {
        return Ok(());
    }
    let grid = u32::try_from((u64::from(act.rows) * u64::from(n)).div_ceil(256).min(4096)).unwrap_or(4096);
    ctx.fire(
        op,
        Fire::at("linear/fp8.hip", "::pie::linear::fp8_mfma_finalize").apply(Launch::grid([grid, 1, 1], [BLOCK, 1, 1])),
        &[
            part,
            y.arg(),
            stated(op, act.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, splits)?.arg(),
            ctx.stage(),
        ],
    )
}

/// The f32 partials of `splits` slices of K over `[rows, n]` (one slab a
/// stream: the sum right after consumes them), none for one slice.
pub(crate) fn fp8_partials(ctx: &Ctx, op: &'static str, splits: u32, rows: u32, n: u32) -> Result<ArgValue, Error> {
    if splits <= 1 {
        return Ok(ArgValue::ABSENT);
    }
    let bytes = (splits as usize) * (rows as usize) * (n as usize) * 4;
    Ok(ArgValue::Ptr(ctx.scratch_stream(op, "linear.fp8_partials", bytes)? as u64))
}

/// An e4m3fn plane of `rows` x `k` codes (`rows` a multiple of 16, `k` of
/// 64), row-major at `src`, into `dst` in the FP8 matrix core's order (each
/// strip of 16 rows as K / 64 units of 1 KB, lane l's 16 bytes the A operand
/// of two k-steps: `fp8_unit_at` in `linear/fp8.hip`), made canonical as
/// the matrix core reads it: negative zero as zero (fnuz's NaN; the value it
/// is), and the NaN codes (0x7f, 0xff: numbers to fnuz) counted into `nans`
/// for the caller to refuse the plane.
pub fn e4m3fn_shuffle(
    ctx: &Ctx,
    op: &'static str,
    src: u64,
    dst: u64,
    rows: u64,
    k: u64,
    nans: u64,
) -> Result<(), Error> {
    if rows == 0 || k == 0 {
        return Ok(());
    }
    if !rows.is_multiple_of(16) || !k.is_multiple_of(64) || !src.is_multiple_of(16) || !dst.is_multiple_of(16) {
        return Err(refuse(
            op,
            format!("an e4m3 plane of {rows} x {k} is not whole 16-row strips of 64-code units"),
        ));
    }
    let (r, kk) = (
        i32::try_from(rows).map_err(|_| refuse(op, "the plane's rows pass an i32"))?,
        i32::try_from(k).map_err(|_| refuse(op, "the plane's width passes an i32"))?,
    );
    let chunks = rows * k / 16;
    let grid = u32::try_from(chunks.div_ceil(u64::from(BLOCK)).min(16384)).unwrap_or(16384);
    ctx.fire(
        op,
        Fire::at("linear/fp8.hip", "::pie::linear::e4m3fn_shuffle<true>")
            .apply(Launch::grid([grid, 1, 1], [BLOCK, 1, 1])),
        &[ArgValue::Ptr(src), ArgValue::Ptr(dst), ArgValue::I32(r), ArgValue::I32(kk), ArgValue::Ptr(nans)],
    )
}

/// The 16-row strips a wave of the FP8 decode tile takes: one (two measured
/// no faster at GLM-5.3's shapes).
#[must_use]
pub const fn fp8_strips() -> u32 {
    1
}

/// A bf16 plane of `rows` x `k` (`rows` a multiple of 16, `k` of 32),
/// row-major at `src`, into `dst` in the matrix core's order: a row's 2k
/// bytes laid as `e4m3fn_shuffle` lays k codes, so lane l's 8 bytes of a
/// k-step are the four elements its bf16 A operand takes.
pub fn bf16_shuffle(
    ctx: &Ctx,
    op: &'static str,
    src: u64,
    dst: u64,
    rows: u64,
    k: u64,
) -> Result<(), Error> {
    if rows == 0 || k == 0 {
        return Ok(());
    }
    if !rows.is_multiple_of(16)
        || !k.is_multiple_of(32)
        || !src.is_multiple_of(16)
        || !dst.is_multiple_of(16)
    {
        return Err(refuse(
            op,
            format!("a bf16 plane of {rows} x {k} is not whole 16-row strips of 32-element units"),
        ));
    }
    let (r, kb) = (
        i32::try_from(rows).map_err(|_| refuse(op, "the plane's rows pass an i32"))?,
        i32::try_from(2 * k).map_err(|_| refuse(op, "the plane's row bytes pass an i32"))?,
    );
    let grid =
        u32::try_from((rows * 2 * k / 16).div_ceil(u64::from(BLOCK)).min(16384)).unwrap_or(16384);
    ctx.fire(
        op,
        Fire::at("linear/fp8.hip", "::pie::linear::e4m3fn_shuffle<false>")
            .apply(Launch::grid([grid, 1, 1], [BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(src),
            ArgValue::Ptr(dst),
            ArgValue::I32(r),
            ArgValue::I32(kb),
            ArgValue::ABSENT,
        ],
    )
}
